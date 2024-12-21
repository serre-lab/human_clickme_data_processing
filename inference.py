import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from train_subject_classifier import RNN, ClickDataset, compute_clicks  # Import from your training file

def load_metadata():
    """Load the model metadata saved during training"""
    metadata = np.load('participant_model_metadata.npz')
    return (
        metadata['max_x'].item(),
        metadata['max_y'].item(),
        metadata['click_div'].item(),
        metadata['n_hidden'].item(),
        metadata['input_dim'].item(),
        metadata['n_classes'].item()
    )

def prepare_dataset(data_path, max_x, max_y, click_div):
    """Prepare dataset for inference without filtering"""
    # Load data
    data = np.load(data_path, allow_pickle=True)
    image_path = data["file_pointer"]
    clickmap_x = data["clickmap_x"]
    clickmap_y = data["clickmap_y"]
    user_id = data["user_id"]
    print(f"Total samples: {len(image_path)}")
    print(f"Total users: {len(np.unique(user_id))}")
    
    # Remove empties
    empty_x = [i for i, x in enumerate(clickmap_x) if len(x) > 0]
    empty_y = [i for i, y in enumerate(clickmap_y) if len(y) > 0]
    not_empty = np.unique(np.concatenate([empty_x, empty_y]))
    
    image_path = image_path[not_empty]
    clickmap_x = clickmap_x[not_empty]
    clickmap_y = clickmap_y[not_empty]
    user_id = user_id[not_empty]
    
    # Process clicks
    clicks = compute_clicks(clickmap_x, clickmap_y)
    
    # Create dataframe for the dataset
    
    df = pd.DataFrame({
        "image_path": image_path,
        "clicks": clicks,
        "user_id": user_id
    })
    
    return df, data

def get_user_classifications(predictions, user_ids, threshold=0.0):
    """
    Determine if a user is good or bad based on their predictions.
    A user is considered bad if the ratio of their samples larger than the threshold.
    """
    user_predictions = {}
    for pred, user_id in zip(predictions, user_ids):
        if user_id not in user_predictions:
            user_predictions[user_id] = []
        user_predictions[user_id].append(pred)
    
    user_classifications = {}
    for user_id, preds in user_predictions.items():
        # Calculate the proportion of bad samples (label 0)
        bad_ratio = sum(p == 0 for p in preds) / len(preds)
        # Classify user as bad (True) if majority of samples are bad
        user_classifications[user_id] = bad_ratio > threshold
    
    return user_classifications

def main():
    parser = argparse.ArgumentParser(description='Inference script for click me dataset')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data file for inference')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save filtered data')
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model (rnn/lstm/gru)')
    parser.add_argument('--use-attention', action='store_true', default=False, help='Whether to use attention in the model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.0, help='Threshold for bad ratio (default: 0.0)')

    args = parser.parse_args()

    # Load model metadata
    max_x, max_y, click_div, n_hidden, input_dim, n_classes = load_metadata()
    
    # Initialize model
    model = RNN(
        input_dim, 
        n_hidden, 
        n_classes,
        model_name=args.model_name,
        use_attention=args.use_attention
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Prepare dataset
    df, original_data = prepare_dataset(args.data_path, max_x, max_y, click_div)
    
    # Create dataloader
    dataset = ClickDataset(df, max_x, max_y, click_div=click_div, inference=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    
    # Run inference
    predictions = []
    with torch.no_grad():
        for click_enc in dataloader:
            outputs = model(click_enc)
            pred = torch.argmax(outputs, dim=1)
            predictions.extend(pred.cpu().numpy())
    
    # Convert predictions to numpy array
    predictions = np.array(predictions)

    # for test only
    # load predictions
    # predictions = np.load('predictions.npy')

    # Classify users
    print("Bad threshold: ", args.threshold)
    user_classifications = get_user_classifications(predictions,
                                                    df['user_id'],
                                                    threshold=args.threshold)
    
    # Get indices of good players (label 1)
    # user_classifications[user_id] == False means the user is good
    good_indices = [i for i, user_id in enumerate(original_data['user_id']) 
                if not user_classifications[user_id]]
    

    # Filter original data
    filtered_data = {
        'file_pointer': original_data['file_pointer'][good_indices],
        'clickmap_x': original_data['clickmap_x'][good_indices],
        'clickmap_y': original_data['clickmap_y'][good_indices],
        'user_id': original_data['user_id'][good_indices]
    }
    
    if 'user_catch_trial' in original_data:
        filtered_data['user_catch_trial'] = original_data['user_catch_trial'][good_indices]
    
    # Save filtered data
    np.savez(args.output_path, **filtered_data)
    
    # Print statistics
    n_bad_users = sum(1 for is_bad in user_classifications.values() if is_bad)
    n_good_users = sum(1 for is_bad in user_classifications.values() if not is_bad)
    n_total_users = len(user_classifications)
    
    print(f"\nClassification Summary:")
    print(f"Threshold for bad_ratio: {args.threshold}")
    print(f"Total users: {n_total_users}")
    print(f"Bad users (bad_ratio > threshold): {n_bad_users} ({n_bad_users/n_total_users*100:.2f}%)")
    print(f"Good users (bad_ratio ≤ threshold): {n_good_users} ({n_good_users/n_total_users*100:.2f}%)")
    print(f"\nData Summary:")
    print(f"Original samples: {len(original_data['user_id'])}")
    print(f"Samples after filtering: {len(good_indices)}")
    print(f"Filtered data saved to: {args.output_path}")

if __name__ == "__main__":
    main()
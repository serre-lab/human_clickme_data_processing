import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from train_subject_classifier_sequence_v2 import MemoryEfficientClickModel, SequentialClickDataset, compute_clicks, remove_empty_clickmaps


def load_metadata():
    """Load the model metadata saved during training"""
    metadata = np.load('participant_model_metadata.npz')
    return (
        metadata['max_x'].item(),
        metadata['max_y'].item(),
        metadata['click_div'].item(),
        metadata['n_hidden'].item(),
        metadata['input_dim'].item(),
        metadata['n_classes'].item(),
        metadata['n_layers'].item() if 'n_layers' in metadata else 2,  # Default to 2 if not found
        metadata['use_attention'].item() if 'use_attention' in metadata else True  # Default to True if not found
    )


def prepare_dataset_for_sequence(data_path, click_div):
    """Prepare sequential dataset for inference without filtering"""
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
    
    # Create dataframe for individual clickmaps
    df = pd.DataFrame({
        "image_path": image_path,
        "clicks": clicks,
        "user_id": user_id
    })
    
    # Sort by user_id to ensure temporal ordering of clickmaps
    df = df.sort_values(['user_id', 'image_path'])
    
    # Create user-level dataframe for sequences
    user_df = pd.DataFrame()
    user_df['user_id'] = df['user_id'].unique()
    
    # Aggregate clicks and image paths for each user
    user_df['all_clicks'] = [df[df['user_id'] == uid]['clicks'].tolist() for uid in user_df['user_id']]
    user_df['all_image_paths'] = [df[df['user_id'] == uid]['image_path'].tolist() for uid in user_df['user_id']]
    
    # Remove users with empty clickmaps
    user_df = remove_empty_clickmaps(user_df)
    
    return user_df, data


def get_user_classifications(predictions, user_ids):
    """
    Determine if a user is good or bad based on their predictions.
    A user is considered bad if the ratio of their samples is larger than the threshold.
    """
    # Since we're now predicting at the user level directly, each prediction corresponds to one user
    user_classifications = {user_id: pred == 0 for user_id, pred in zip(user_ids, predictions)}
    
    return user_classifications


def main():
    parser = argparse.ArgumentParser(description='Inference script for click me dataset')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data file for inference')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save filtered data')
    parser.add_argument('--use-attention', action='store_true', default=True, help='Whether to use attention in the model')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--seq-length', type=int, default=50, help='Sequence length for processing')


    args = parser.parse_args()

    # Load model metadata
    max_x, max_y, click_div, n_hidden, input_dim, n_classes, n_layers, use_attention = load_metadata()
    
    # Initialize model
    model = MemoryEfficientClickModel(
        input_dim=input_dim, 
        n_hidden=n_hidden, 
        n_classes=n_classes,
        dropout_rate=0.3,  # Default value during inference
        use_attention=args.use_attention or use_attention,
        n_layers=n_layers
    )
    
    # Load model weights
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Prepare dataset
    user_df, original_data = prepare_dataset_for_sequence(args.data_path, click_div)
    
    # Create dataloader
    dataset = SequentialClickDataset(
        user_df, 
        max_x, 
        max_y, 
        seq_length=args.seq_length,
        click_div=click_div, 
        inference=True
    )
    
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
    
    # Map user-level predictions back to original dataset
    # Classify users
    user_classifications = get_user_classifications(
        predictions,
        user_df['user_id']
    )
    
    # Get indices of good players (label 1)
    # user_classifications[user_id] == False means the user is good
    good_indices = [i for i, user_id in enumerate(original_data['user_id']) 
                   if user_id in user_classifications and not user_classifications[user_id]]
    
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
    print(f"Total users processed: {n_total_users}")
    print(f"Users classified as bad: {n_bad_users} ({n_bad_users/n_total_users*100:.2f}%)")
    print(f"Users classified as good: {n_good_users} ({n_good_users/n_total_users*100:.2f}%)")
    print(f"\nData Summary:")
    print(f"Original samples: {len(original_data['user_id'])}")
    print(f"Samples after filtering: {len(good_indices)}")
    print(f"Filtered data saved to: {args.output_path}")
    
    # Save the user classifications for reference
    classification_df = pd.DataFrame([
        {"user_id": user_id, "is_bad": is_bad}
        for user_id, is_bad in user_classifications.items()
    ])
    classification_path = args.output_path.replace('.npz', '_classifications.csv')
    classification_df.to_csv(classification_path, index=False)
    print(f"User classifications saved to: {classification_path}")


if __name__ == "__main__":
    main()
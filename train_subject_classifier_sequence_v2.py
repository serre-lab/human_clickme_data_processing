import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from torch import nn
import schedulefree
from tqdm import tqdm as std_tqdm
from accelerate import InitProcessGroupKwargs
from accelerate import Accelerator
from datetime import timedelta
from functools import partial
from sklearn.model_selection import train_test_split
import argparse
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SequentialClickDataset(Dataset):
    def __init__(self, df, max_x, max_y, seq_length=100, click_div=4, inference=False):
        self.df = df
        self.max_x = max_x
        self.max_y = max_y
        self.click_div = click_div
        self.max_clicks = 100
        self.inference = inference
        self.seq_length = seq_length

    def __len__(self):
        return len(self.df)

    def _process_clickmap(self, clicks):
        """Process a single clickmap."""
        # Convert clicks to array and divide by click_div
        clicks = np.asarray(clicks) // self.click_div
        
        # Create one-hot encodings
        x_enc = np.zeros((len(clicks), self.max_x))
        y_enc = np.zeros((len(clicks), self.max_y))
        
        for i, (x, y) in enumerate(clicks):
            # Ensure x and y are within bounds
            x = min(x, self.max_x - 1)
            y = min(y, self.max_y - 1)
            x_enc[i, x] = 1
            y_enc[i, y] = 1
            
        click_enc = np.concatenate((x_enc, y_enc), 1)
        
        # Handle varying lengths
        if len(click_enc) > self.max_clicks:
            click_enc = click_enc[:self.max_clicks]
        elif len(click_enc) < self.max_clicks:
            click_enc = np.pad(click_enc, ((0, self.max_clicks - len(click_enc)), (0, 0)))

        if not self.inference and random.random() > 0.7:
            # Randomly mask 10-30% of clicks
            mask_ratio = random.uniform(0.1, 0.3)
            mask_indices = random.sample(range(len(click_enc)), int(len(click_enc) * mask_ratio))
            for idx in mask_indices:
                click_enc[idx] = 0
            
        return torch.from_numpy(click_enc).float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        clickmap_sequence = []
        
        # Process sequence of clickmaps
        all_clicks = row['all_clicks']
        
        # Take seq_length clickmaps or pad if fewer
        for clicks in all_clicks[:self.seq_length]:
            click_enc = self._process_clickmap(clicks)
            clickmap_sequence.append(click_enc)
        
        # Pad sequence if needed
        while len(clickmap_sequence) < self.seq_length:
            # Create empty clickmap encoding
            empty_enc = torch.zeros((self.max_clicks, self.max_x + self.max_y))
            clickmap_sequence.append(empty_enc)
        
        # Stack all clickmaps into a single tensor
        clickmap_sequence = torch.stack(clickmap_sequence)
        
        if self.inference:
            return clickmap_sequence
        return row['label'], clickmap_sequence


def compute_clicks(clickmap_x, clickmap_y, n_jobs=-1):
    """
    Parallelizes the processing of clickmaps by zipping x and y coordinates.

    Args:
        clickmap_x (list or array): List of x coordinates.
        clickmap_y (list or array): List of y coordinates.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (use all cores).

    Returns:
        list: List of zipped (x, y) tuples for each clickmap.
    """
    def zip_clicks(x, y):
        return list(zip(x, y))
    
    clicks = Parallel(n_jobs=n_jobs)(
        delayed(zip_clicks)(x, y) 
        for x, y in tqdm(zip(clickmap_x, clickmap_y), desc="Reformatting clicks", total=len(clickmap_x))
    )
    return clicks


def seed_everything(s: int):
    """
    This function allows us to set the seed for all of our random functions
    so that we can get reproducible results.

    Parameters
    ----------
    s : int
        seed to seed all random functions with
    """
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True


def preprocess_dataset_segment(data_path, good_players, cheaters_and_bad_players, catch_thresh, click_div, split_factor=5):
    # Load data
    data = np.load(data_path, allow_pickle=True)
    image_path = data["file_pointer"]
    clickmap_x = data["clickmap_x"]
    clickmap_y = data["clickmap_y"]
    user_id = data["user_id"]
    user_catch_trial = data["user_catch_trial"]

    # Remove empties
    empty_x = [i for i, x in enumerate(clickmap_x) if len(x) > 0]
    empty_y = [i for i, y in enumerate(clickmap_y) if len(y) > 0]
    not_empty = np.unique(np.concatenate([empty_x, empty_y]))
    image_path = image_path[not_empty]
    clickmap_x = clickmap_x[not_empty]
    clickmap_y = clickmap_y[not_empty]
    user_id = user_id[not_empty]
    user_catch_trial = user_catch_trial[not_empty]

    # Get max x and y
    max_x = max([max(x) for x in clickmap_x if len(x)])
    max_y = max([max(x) for x in clickmap_y if len(x)])
    max_size = max(max_x, max_y) // click_div
    max_x, max_y = max_size, max_size

    # Filter subjects by catch trials
    catch_trials = user_catch_trial >= catch_thresh
    image_path = image_path[catch_trials]
    clickmap_x = clickmap_x[catch_trials]
    clickmap_y = clickmap_y[catch_trials]
    user_id = user_id[catch_trials]
    print("Catch trial filter from {} to {}".format(len(user_catch_trial), catch_trials.sum()))

    # Filter down to good/bad players
    all_players = np.concatenate([good_players, cheaters_and_bad_players])
    flt = np.in1d(user_id, all_players)
    image_path = image_path[flt]
    clickmap_x = clickmap_x[flt]
    clickmap_y = clickmap_y[flt]
    user_id = user_id[flt]
    label = np.in1d(user_id, good_players).astype(int)

    # Process clicks
    clicks = compute_clicks(clickmap_x, clickmap_y)

    df = pd.DataFrame({
        "image_path": image_path,
        "label": label,
        "clicks": clicks,
        "user_id": user_id
    })
    
    # Sort by user_id to ensure temporal ordering of clickmaps
    df = df.sort_values(['user_id', 'image_path'])

    user_df = pd.DataFrame()
    user_df['user_id'] = df['user_id'].unique()

    print("Total users: {}".format(len(user_df)))
    
    # Add labels (they're the same for each user)
    user_df['label'] = [df[df['user_id'] == uid]['label'].iloc[0] for uid in user_df['user_id']]
    
    # Aggregate clicks and image paths for each user
    user_df['all_clicks'] = [df[df['user_id'] == uid]['clicks'].tolist() for uid in user_df['user_id']]
    user_df['all_image_paths'] = [df[df['user_id'] == uid]['image_path'].tolist() for uid in user_df['user_id']]

    del data.f
    data.close()  # avoid the "too many files are open" error

    # Print label distribution
    num_good = (user_df['label'] == 1).sum()
    num_bad = (user_df['label'] == 0).sum()
    print(f"\nLabel Distribution:")
    print(f"Good players (label 1): {num_good}")
    print(f"Bad players (label 0): {num_bad}")

    if split_factor == None:
        # we don't want to split the data
        print("Not splitting data")
        return df, None, max_x, max_y
    
    # Create new DataFrame for segmented data
    segmented_rows = []
    
    # Process each user
    for _, row in user_df.iterrows():
        if row['label'] == 1:
            # Keep good players as is
            segmented_rows.append(row.to_dict())
        else:
            # Split bad players into segments
            total_length = len(row['all_image_paths'])
            segment_size = total_length // split_factor
            
            # Create synthetic users for each segment
            for seg_idx in range(split_factor):
                start_idx = seg_idx * segment_size
                # For the last segment, include any remaining items
                end_idx = start_idx + segment_size if seg_idx < split_factor - 1 else total_length
                
                synthetic_user = {
                    'user_id': f"{row['user_id']}_seg{seg_idx}",  # Create unique ID for each segment
                    'label': 0,  # Maintain the bad player label
                    'all_clicks': row['all_clicks'][start_idx:end_idx],
                    'all_image_paths': row['all_image_paths'][start_idx:end_idx]
                }
                segmented_rows.append(synthetic_user)

    # Create new DataFrame with segmented data
    segmented_user_df = pd.DataFrame(segmented_rows)

    # Print new label distribution
    num_good = (segmented_user_df['label'] == 1).sum()
    num_bad = (segmented_user_df['label'] == 0).sum()
    print(f"\nSegmented Label Distribution:")
    print(f"Good players (label 1): {num_good}")
    print(f"Bad players (label 0): {num_bad}")
    print(f"Total users: {len(segmented_user_df)}")

    return df, segmented_user_df, max_x, max_y


def remove_empty_clickmaps(df):
    df = df[df['all_clicks'].map(len) > 0]
    return df


class MemoryEfficientClickModel(nn.Module):
    def __init__(self, input_dim, n_hidden, n_classes, dropout_rate=0.3, use_attention=True, n_layers=2):
        super(MemoryEfficientClickModel, self).__init__()
        self.use_attention = use_attention
        self.n_layers = n_layers
        
        # Memory-efficient feature extraction
        # Use smaller number of filters and more aggressive pooling
        self.features = nn.Sequential(
            # First block with smaller filter size
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # More aggressive pooling
            nn.BatchNorm2d(32),
            nn.Dropout2d(dropout_rate),
            
            # Second block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),  # More aggressive pooling
            nn.BatchNorm2d(64),
            nn.Dropout2d(dropout_rate)
        )
        
        # Calculate the size after the aggressive pooling
        # With two 4x4 pooling layers, dimensions are reduced by factor of 16
        feature_size = 64 * (100 // 16) * (input_dim // 16)
        
        # Dimensionality reduction (smaller intermediate size)
        self.dim_reduction = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Bidirectional GRU with fewer parameters
        self.bidirectional = True
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_rate if n_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Simpler attention mechanism
        if use_attention:
            gru_output_dim = n_hidden * 2 if self.bidirectional else n_hidden
            self.attention = nn.Sequential(
                nn.Linear(gru_output_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # Classification head with fewer parameters
        gru_output_dim = n_hidden * 2 if self.bidirectional else n_hidden
        self.classifier = nn.Sequential(
            nn.Linear(gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, sequence_length, max_clicks, input_dim]
        batch_size, seq_len, max_clicks, input_dim = x.shape
        
        # Process each sequence element
        processed_sequence = []
        
        # Memory-efficient approach: process each sequence item individually
        for i in range(seq_len):
            # Get current sequence item
            x_i = x[:, i, :, :]  # [batch_size, max_clicks, input_dim]
            
            # Reshape for CNN
            x_i = x_i.view(batch_size, 1, max_clicks, input_dim)
            
            # Apply CNN feature extraction
            x_i = self.features(x_i)
            
            # Flatten and reduce dimension
            x_i = x_i.view(batch_size, -1)
            x_i = self.dim_reduction(x_i)
            
            processed_sequence.append(x_i)
        
        # Stack processed sequence
        x_seq = torch.stack(processed_sequence, dim=1)  # [batch_size, seq_len, reduced_dim]
        
        # Apply GRU
        self.gru.flatten_parameters()  # For efficiency
        gru_out, _ = self.gru(x_seq)  # [batch_size, seq_len, n_hidden*2]
        
        # Apply attention if enabled
        if self.use_attention:
            # Simple attention mechanism
            attn_weights = self.attention(gru_out)  # [batch_size, seq_len, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Apply attention weights
            context = torch.sum(gru_out * attn_weights, dim=1)  # [batch_size, n_hidden*2]
        else:
            # Just take the last hidden state
            context = gru_out[:, -1, :]  # [batch_size, n_hidden*2]
        
        # Apply classifier
        output = self.classifier(context)
        
        return output


def create_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            min_lr_ratio,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description='Train a subject classifier model')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train the model')
    parser.add_argument('--output', type=str, required=True, help='Output file name')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='Checkpoint to trained model.')
    parser.add_argument('--train-data-path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--val-data-path', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--evaluate-only', action='store_true', default=False, help='Whether to only evaluate the model')
    parser.add_argument('--seq-length', type=int, default=50, help='Sequence length for processing clickmaps')
    # model side
    parser.add_argument('--use-attention', action='store_true', default=True, help='Whether to use attention in the model')
    parser.add_argument('--hidden-size', type=int, default=64, help='Hidden size for GRU')
    parser.add_argument('--n-layers', type=int, default=2, help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use mixed precision training')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-3, help='Weight decay')

    args = parser.parse_args()

    SEED = 42
    seed_everything(SEED)

    # Inputs
    cheaters = [780,1045,1551,1548,1549,1550,1173,2059,2056,2061,2065,2055,2064,1661,2057,2062,2054,2063,2058]
    bad_players = [1164,664,933,219,961,596,1378,501]
    good_players = [1131,1176,350,279,758,969,431, 339,1420,331,1346,878,540,607,1221,686,849, 984,355,931,790,575,1425,1099,347, 743,522,293,264, 976, 988, 619, 869, 1417, 1294, 707, 329, 930, 952, 270, 1382, 1441, 1391, 1486, 404, 1430, 317, 855, 703, 945, 708, 1354, 525, 1124, 182, 783, 222, 870, 326, 382, 434, 701, 1339, 367, 611, 1063, 1042, 385, 694, 625, 1006, 370, 463, 1258, 852, 1278,1002, 671,1076, 1016,729,337,420,1061,281, 368,811,1485,566]
    catch_thresh = 0.95
    train_batch_size = args.batch_size
    train_num_workers = 0
    click_div = 8 # original 6 but we use bigger for downsampling
    lr = args.lr
    weight_decay = args.weight_decay
    ckpts = "checkpoints"
    os.makedirs(ckpts, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Prepare indices
    cheaters_and_bad_players = np.concatenate([cheaters, bad_players])
    good_players = np.asarray(good_players)
    n = len(cheaters_and_bad_players)
    np.random.seed(42)
    good_players = good_players[np.random.permutation(len(good_players))[:n]]

    try:
        # Try to load preprocessed data first
        print("Attempting to load preprocessed data from data.npz")
        data = np.load("data.npz", allow_pickle=True)
        
        # Function to safely evaluate string representations of lists
        def parse_clicks(click_str):
            if isinstance(click_str, str):
                return eval(click_str)
            return click_str

        # Reconstruct train_df
        train_df = pd.DataFrame({
            'user_id': data['train_user_ids'],
            'label': data['train_labels'],
            'all_clicks': [parse_clicks(clicks) for clicks in data['train_clicks']]
        })

        # Reconstruct val_df
        val_df = pd.DataFrame({
            'user_id': data['val_user_ids'],
            'label': data['val_labels'],
            'all_clicks': [parse_clicks(clicks) for clicks in data['val_clicks']]
        })

        train_max_x, train_max_y = data['train_max_x'], data['train_max_y']
        val_max_x, val_max_y = data['val_max_x'], data['val_max_y']
        
        print("Successfully loaded preprocessed data")
    except:
        print("Could not load preprocessed data, preprocessing from raw files")
        # Process raw data
        _, train_df, train_max_x, train_max_y = preprocess_dataset_segment(
            args.train_data_path, 
            good_players, 
            cheaters_and_bad_players, 
            catch_thresh, 
            click_div
        )

        _, val_df, val_max_x, val_max_y = preprocess_dataset_segment(
            args.val_data_path,
            good_players,
            cheaters_and_bad_players,
            catch_thresh,
            click_div
        )

        train_df = remove_empty_clickmaps(train_df)
        val_df = remove_empty_clickmaps(val_df)
        
        # Save preprocessed data


    # Use maximum dimensions across both datasets but add downsampling
    max_x = max(train_max_x, val_max_y) // (click_div // 4)  # Additional downsampling
    max_y = max(train_max_y, val_max_y) // (click_div // 4)  # Additional downsampling
    
    print(f"Data loaded: Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    print(f"Using dimensions: max_x={max_x}, max_y={max_y}")
    
    # Create data loaders with appropriate sequence length
    train_label_index = train_df.label.values
    unique_classes, class_sample_count = np.unique(train_label_index, return_counts=True)
    # weighted sampling by class
    class_weights = compute_class_weight("balanced", classes=unique_classes, y=train_label_index)
    class_weights = torch.from_numpy(class_weights).float()
    samples_weight_train = np.asarray([class_weights[t] for t in train_label_index])
    samples_weight_train = torch.from_numpy(samples_weight_train).double()
    sampler = WeightedRandomSampler(samples_weight_train, len(samples_weight_train))
    print("Building dataloaders")

    train_dataset = SequentialClickDataset(
        train_df, max_x, max_y, seq_length=args.seq_length, click_div=click_div
    )
    
    val_dataset = SequentialClickDataset(
        val_df, max_x, max_y, seq_length=args.seq_length, click_div=click_div
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=train_num_workers
    )

    # Create validation data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=train_num_workers
    )

    # Initialize model
    print("Preparing model")
    n_hidden = args.hidden_size
    input_dim = max_x + max_y  # One-hot encoding of x concatenated with one-hot encoding of y
    print("Input dimensions:", input_dim)
    print("Model config: hidden_size={}, n_layers={}, dropout={}".format(
        n_hidden, args.n_layers, args.dropout))

    # Create memory-efficient model
    model = MemoryEfficientClickModel(
        input_dim=input_dim, 
        n_hidden=n_hidden, 
        n_classes=len(unique_classes),
        dropout_rate=args.dropout,
        use_attention=args.use_attention,
        n_layers=args.n_layers
    )
    
    if args.checkpoint_path is not None:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    # Save meta data needed to run the model
    np.savez(
        "participant_model_metadata.npz",
        max_x=max_x,
        max_y=max_y,
        click_div=click_div,
        n_hidden=n_hidden,
        input_dim=input_dim,
        n_classes=len(unique_classes),
        n_layers=args.n_layers,
        use_attention=args.use_attention
    )

    # Configure accelerator with mixed precision
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(
        mixed_precision='fp16' if args.fp16 else 'no', 
        kwargs_handlers=[process_group_kwargs]
    )
    device = accelerator.device
    tqdm = partial(std_tqdm, dynamic_ncols=True)
    
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    # optimizer = schedulefree.AdamWScheduleFree(
    #     model.parameters(), 
    #     lr=lr,
    #     weight_decay=weight_decay
    # )
    
    # Create learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.05 * total_steps)  # 10% of total steps for warmup
    scheduler = None
    # scheduler = create_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.1,  # Reduce LR by half when plateau detected
        patience=2,  # Wait 2 epochs before reducing
        min_lr=1e-6,
        verbose=True
    )

    # Prepare accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Initialize tracking variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    patience = 5  # For early stopping
    patience_counter = 0
    
    # Train model
    for epoch in range(args.epochs):
        if not args.evaluate_only:
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            # Training loop with progress bar
            train_progress = tqdm(
                total=len(train_loader),
                desc=f"Training Epoch {epoch+1}/{args.epochs}"
            ) if accelerator.is_main_process else None
            
            for label, click_enc in train_loader:
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    pred = model(click_enc)
                    loss = F.cross_entropy(pred, label)
                
                # Skip if NaN
                if torch.isnan(loss):
                    print("Skipping batch with NaN loss")
                    continue
                
                # Backward pass
                accelerator.backward(loss)
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                
                optimizer.step()
                
                # Calculate accuracy
                predicted = torch.argmax(pred, dim=1)
                batch_correct = (predicted == label).sum().item()
                batch_total = label.size(0)
                
                # Accumulate stats
                epoch_loss += loss.item()
                correct += batch_correct
                total += batch_total
                
                # Update progress bar
                if accelerator.is_main_process and train_progress is not None:
                    train_progress.set_postfix({
                        "loss": f"{loss.item():.4f}", 
                        "acc": f"{batch_correct/batch_total:.4f}"
                    })
                    train_progress.update()
            
            # Calculate epoch stats
            avg_train_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total if total > 0 else 0
            train_losses.append(avg_train_loss)
            train_accs.append(train_accuracy)
            
            if accelerator.is_main_process:
                with open(args.output, "a") as f:
                    msg = f"Epoch {epoch+1}/{args.epochs} - Train loss: {avg_train_loss:.4f}, Train acc: {train_accuracy:.4f}\n"
                    print(msg, end="")
                    f.write(msg)
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for label, click_enc in val_loader:
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    pred = model(click_enc)
                    loss = F.cross_entropy(pred, label)
                
                # Calculate accuracy
                predicted = torch.argmax(pred, dim=1)
                batch_correct = (predicted == label).sum().item()
                batch_total = label.size(0)
                
                # Accumulate stats
                val_loss += loss.item()
                val_correct += batch_correct
                val_total += batch_total
        
        # Calculate validation stats
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_losses.append(avg_val_loss)
        val_accs.append(val_accuracy)

        if scheduler is not None:
                scheduler.step(avg_val_loss) # for ReduceLROnPlateau
        
        if accelerator.is_main_process:
            with open(args.output, "a") as f:
                msg = f"Epoch {epoch+1}/{args.epochs} - Val loss: {avg_val_loss:.4f}, Val acc: {val_accuracy:.4f}\n"
                print(msg, end="")
                f.write(msg)
        
        # Check for improvement
        improved = False
        
        # First priority: better validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_accuracy
            improved = True
            
        # Save model if improved
        if improved and accelerator.is_main_process:
            patience_counter = 0
            checkpoint_filename = os.path.join(ckpts, f'epoch_{epoch+1}_val_acc_{val_accuracy:.2f}.pth')
            torch.save(accelerator.unwrap_model(model).state_dict(), checkpoint_filename)
            
            # Also save as best model
            best_model_path = os.path.join(ckpts, f'best.pth')
            torch.save(accelerator.unwrap_model(model).state_dict(), best_model_path)
            
            with open(args.output, "a") as f:
                msg = f"Model improved! Saved checkpoint to {checkpoint_filename}\n"
                print(msg, end="")
                f.write(msg)
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break
            
        accelerator.wait_for_everyone()

    # Final evaluation
    if accelerator.is_main_process:
        with open(args.output, "a") as f:
            msg = "\nTraining Complete!\n"
            msg += f"Best validation accuracy: {best_val_acc:.4f}\n"
            msg += f"Best validation loss: {best_val_loss:.4f}\n"
            print(msg, end="")
            f.write(msg)

if __name__ == "__main__":
    main()
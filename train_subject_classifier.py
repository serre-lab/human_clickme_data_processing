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


# Define a custom dataset class
class ClickDataset(Dataset):
    def __init__(self, df, max_x, max_y, click_div=4):
        self.df = df
        self.max_x = max_x
        self.max_y = max_y
        self.click_div = click_div  # Resample factor
        self.max_clicks = 100

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label, clicks = row["label"], row["clicks"]

        # Turn clicks into a one-hot encoding of x and y
        clicks = np.asarray(clicks) // self.click_div
        x_enc = np.zeros((len(clicks), self.max_x))
        y_enc = np.zeros((len(clicks), self.max_y))
        x_enc[:, clicks[:, 0]] = 1
        y_enc[:, clicks[:, 1]] = 1
        click_enc = np.concatenate((x_enc, y_enc), 1)
        if len(click_enc) > self.max_clicks:
            click_enc = click_enc[:self.max_clicks]
        elif len(click_enc) < self.max_clicks:
            click_enc = np.pad(click_enc, ((0, self.max_clicks - len(click_enc)), (0, 0)))
        click_enc = torch.from_numpy(click_enc).float()
        return label, click_enc

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.xy_proj = nn.Linear(input_size, hidden_size)
        self.rnn = nn.GRU(
            hidden_size * 2, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.readout = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        x_proj = self.xy_proj(input[:, ..., :input.shape[-1]//2])
        y_proj = self.xy_proj(input[:, ..., input.shape[-1]//2:])
        proj = torch.concat((x_proj, y_proj), dim=-1)
        output, _ = self.rnn(proj)  # Default hidden at t0 to 0 init
        output = self.readout(output[:, -1])
        return output


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


def main():
    # Inputs
    cheaters = [780,1045,1551,1548,1549,1550,1173]
    bad_players = [1164,664,933,219,961,596,1378,501]
    good_players = [1131,1176,350,279,758,969,431, 339,1420,331,1346,878,540,607,1221,686,849, 984,355,931,790,575,1425,1099,347, 743,522,293,264, 976, 988, 619, 869, 1417, 1294, 707, 329, 930, 952, 270, 1382, 1441, 1391, 1486, 404, 1430, 317, 855, 703, 945, 708, 1354, 525, 1124, 182, 783, 222, 870, 326, 382, 434, 701, 1339, 367, 611, 1063, 1042, 385, 694, 625, 1006, 370, 463, 1258, 852, 1278,1002, 671,1076, 1016,729,337,420,1061,281, 368,811,1485,566]
    catch_thresh = 0.95
    data_file = "clickme_datasets/prj_clickmev2_train_imagenet_10_10_2024.npz"
    train_batch_size = 32
    train_num_workers = 0
    # max_x, max_y = 1000, 1000
    click_div = 6
    lr = 1e-3
    ckpts = "checkpoints"
    os.makedirs(ckpts, exist_ok=True)

    # Prepare indices
    cheaters_and_bad_players = np.concatenate([cheaters, bad_players])
    good_players = np.asarray(good_players)
    n = len(cheaters_and_bad_players)
    np.random.seed(42)
    good_players = good_players[np.random.permutation(len(good_players))[:n]]

    # Get data
    data = np.load(data_file, allow_pickle=True)
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

    # Usage
    clicks = compute_clicks(clickmap_x, clickmap_y)
    # clicks = [list(zip(x, y)) for x, y in tqdm(zip(clickmap_x, clickmap_y), desc="Reformtting clicks", total=len(clickmap_x))]

    # Create dataframe
    df = pd.DataFrame({"image_path": image_path, "label": label, "clicks": clicks, "user_id": user_id})

    # Close npz
    del data.f
    data.close()  # avoid the "too many files are open" error

    # Create data loaders
    train_label_index = df.label.values
    unique_classes, class_sample_count = np.unique(train_label_index, return_counts=True)
    class_weights = compute_class_weight("balanced", classes=unique_classes, y=train_label_index)
    class_weights = torch.from_numpy(class_weights).float()
    samples_weight_train = np.asarray([class_weights[t] for t in train_label_index])
    samples_weight_train = torch.from_numpy(samples_weight_train).double()
    sampler = WeightedRandomSampler(samples_weight_train, len(samples_weight_train))
    print("Building dataloaders")
    train_loader = DataLoader(
        ClickDataset(df, max_x, max_y, click_div=click_div),
        batch_size=train_batch_size,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=train_num_workers
    )

    # Initialize model
    print("Preparing models")
    n_hidden = 32
    input_dim = max_x  # Do a one-hot encoding of x concatenated with one-hot encoding of y
    model = RNN(input_dim, n_hidden, len(unique_classes))

    # Save meta data needed to run the model
    np.savez(
        "participant_model_metadata.npz",
        max_x=max_x,
        max_y=max_y,
        click_div=click_div,
        n_hidden=n_hidden,
        input_dim=input_dim,
        n_classes=len(unique_classes)
    )

    # Prepare everything else
    process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[process_group_kwargs])
    device = accelerator.device
    tqdm = partial(std_tqdm, dynamic_ncols=True)
    optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=lr)

    # Prepare accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    # Train model
    epochs = 10
    best_loss = np.inf
    losses = []
    steps_per_epoch = None
    for epoch in range(epochs):
        model.train()
        if steps_per_epoch is None:
            steps_per_epoch = len(train_loader)
        if accelerator.is_main_process:
            train_progress = tqdm(
                total=steps_per_epoch, 
                desc=f"Training Epoch {epoch+1}/{epochs}"
            )

        for label, click_enc in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(click_enc)
            loss = F.cross_entropy(pred, label)
            if torch.isnan(loss):
                print("Skipping nan loss")
            else:
                accelerator.backward(loss)
                optimizer.step()

                loss = loss.item()
                losses.append(loss)

            if accelerator.is_main_process:
                train_progress.set_postfix({"Train loss": f"{loss:.4f}"})
                train_progress.update()

            if accelerator.is_main_process:
                if loss < best_loss:
                    checkpoint_filename = os.path.join(ckpts, f'model_epoch_{epoch+1}.pth')
                    torch.save(accelerator.unwrap_model(model).state_dict(), checkpoint_filename)
                    best_loss = loss
        accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()

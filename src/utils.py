import re
import os
import sys
import torch
import yaml
import numpy as np
import pandas as pd
from torch.nn import functional as F
from tqdm import tqdm
from torchvision.transforms import functional as tvF
from glob import glob
from train_subject_classifier import RNN
from accelerate import Accelerator
from joblib import Parallel, delayed
import joblib
import psutil
import h5py
from PIL import Image
from scipy.ndimage import maximum_filter
import time

# Near the top of the file (around line 10), add torch.cuda memory management functions
try:
    import torch.cuda
    import gc
except ImportError:
    pass

def load_masks(mask_dir, wc="*.pth"):
    files = glob(os.path.join(mask_dir, wc))
    assert len(files), "No masks found in {}".format(mask_dir)
    masks = {}
    for f in files:
        loaded = torch.load(f, weights_only=False)  # Akash added image/mask/category
        # image = loaded[0]
        mask = loaded[1]
        cat = loaded[2]
        try:
            if isinstance(cat, list):
                cat = cat[0]
            masks[os.path.join(cat, os.path.basename(f).split(".")[0])] = mask
        except:
            import pdb; pdb.set_trace()
    return masks


def filter_classes(
        clickmaps,
        class_filter_file):

    # Import category map from the specified file
    category_map = np.load(
        class_filter_file,
        allow_pickle=True).item()
    category_map_keys = np.asarray([k for k in category_map.keys()])

    # Filter clickmaps based on the category map
    filtered_clickmaps = {}
    for image_path, maps in clickmaps.items():
        category = image_path.split('/')[0]  # Assuming category is the first part of the path
        if category in category_map_keys:  # If the category is in our filter list
            filtered_clickmaps[image_path] = maps
    return filtered_clickmaps


def filter_participants(clickmaps, metadata_file="participant_model_metadata.npz", debug=False):
    metadata = np.load(metadata_file)
    max_x = metadata["max_x"]
    max_y = metadata["max_y"]
    click_div = int(metadata["click_div"])
    n_hidden = int(metadata["n_hidden"])
    input_dim = int(metadata["input_dim"])
    n_classes = int(metadata["n_classes"])
    accelerator = Accelerator()
    device = accelerator.device

    # Load the model
    ckpts = glob(os.path.join("checkpoints", "*.pth"))
    sorted_ckpts = sorted(ckpts, key=os.path.getmtime)[-1]
    model = RNN(input_dim, n_hidden, n_classes)
    model.load_state_dict(torch.load(sorted_ckpts))
    model.eval()
    model.to(device)

    # Get the predictions
    processed_clickmaps = {}
    remove_count = 0
    with torch.no_grad():
        for image_path, maps in tqdm(clickmaps.items(), desc="Filtering participants", total=len(clickmaps)):
            # Prepare the data
            new_maps = []
            for map in maps:
                clicks = np.asarray(map) // click_div
                x_enc = np.zeros((len(clicks), max_x))
                y_enc = np.zeros((len(clicks), max_y))
                x_enc[:, clicks[:, 0]] = 1
                y_enc[:, clicks[:, 1]] = 1
                click_enc = np.concatenate((x_enc, y_enc), 1)
                click_enc = torch.from_numpy(click_enc).float().to(device)
                pred = model(click_enc[None])
                if debug:
                    # Take the cheaters
                    pred = pred.argmin(1).item()
                else:
                    pred = pred.argmax(1).item()
                if pred:
                    new_maps.append(map)
                else:
                    remove_count += 1
            processed_clickmaps[image_path] = new_maps
    print(f"Removed {remove_count} participant maps")

    # Remove empty images
    processed_clickmaps = {k: v for k, v in processed_clickmaps.items() if len(v)}
    return processed_clickmaps


def filter_for_foreground_masks(
        final_clickmaps,
        all_clickmaps,
        categories,
        masks,
        mask_threshold,
        quantize_threshold=0.5):
    """
    quantize_threshold: If <= 0, use mean. If > 0, use this probability threshold for binarization.
    """
    proc_final_clickmaps = {}
    proc_all_clickmaps = []
    proc_categories = []
    proc_final_keep_index = []
    # missing = []
    for idx, k in enumerate(final_clickmaps.keys()):
        mask_key = k.split(".")[0]  # Remove image extension
        if mask_key in masks.keys():
            mask = masks[mask_key].squeeze()
            click_map = all_clickmaps[idx]
            mean_click_map = click_map.mean(0)
            if quantize_threshold <= 0:
                clickmap_threshold = np.mean(click_map)
            else:
                mean_click_map = mean_click_map / mean_click_map.max()
                clickmap_threshold = quantize_threshold
            thresh_click_map = (mean_click_map > clickmap_threshold).astype(np.float32)
            try:
                iou = fast_ious(thresh_click_map, mask)
                if iou < mask_threshold:
                    proc_final_clickmaps[k] = final_clickmaps[k]
                    # proc_all_clickmaps[k] = click_map
                    proc_all_clickmaps.append(click_map)
                    proc_categories.append(categories[idx])
                    proc_final_keep_index.append(k)
                else:
                    pass
                    # import pdb; pdb.set_trace()
                    # from matplotlib import pyplot as plt;plt.subplot(121);plt.imshow(mean_click_map);plt.subplot(122);plt.imshow(mask[0]);plt.show()
            except:
                pass
        else:
            print(f"No mask found for {mask_key}")
            # missing.append(mask_key)
    return proc_final_clickmaps, proc_all_clickmaps, proc_categories, proc_final_keep_index


def process_clickme_data(data_file, filter_mobile, catch_thresh=0.95):
    if "csv" in data_file:
        df = pd.read_csv(data_file)
        return pd.read_csv(data_file)
    elif "npz" in data_file:
        print("Load npz")
        data = np.load(data_file, allow_pickle=True)
        print("Loaded npz")
        image_path = data["file_pointer"]
        clickmap_x = data["clickmap_x"]
        clickmap_y = data["clickmap_y"]
        user_id = data["user_id"]
        user_catch_trial = data["user_catch_trial"]
        is_mobile = []
        is_mobile_lens = []
        # TODO: Figure out why there's empty entries in this
        for x in data["is_mobile"]:
            if len(x):
                is_mobile.append(x[0])
            else:
                is_mobile.append(False)
            is_mobile_lens.append(len(x))
        is_mobile = np.asarray(is_mobile)

        # Filter subjects by catch trials
        catch_trials = user_catch_trial >= catch_thresh
        if filter_mobile:
            catch_trials = catch_trials & ~is_mobile
        image_path = image_path[catch_trials]
        clickmap_x = clickmap_x[catch_trials]
        clickmap_y = clickmap_y[catch_trials]
        user_id = user_id[catch_trials]
        print("Trials filtered from {} to {}".format(len(user_catch_trial), catch_trials.sum()))

        # Combine clickmap_x/y into tuples to match Jay's format
        clicks = [list(zip(x, y)) for x, y in zip(clickmap_x, clickmap_y)]
        print("Create dataframe")
        # Create dataframe
        df = pd.DataFrame({"image_path": image_path, "clicks": clicks, "user_id": user_id})
        print("Created dataframe")
        # Close npz
        del data.f
        data.close()  # avoid the "too many files are open" error
        return df
    else:
        raise NotImplementedError("Cannot process {}".format(data_file))


def get_config(argv):
    if len(argv) == 2:
        if "configs" + os.path.sep in argv[1]:
            config_file = argv[1]
        else:
            config_file = os.path.join("configs", argv[1])
    else:
        raise ValueError("Usage: python clickme_prepare_maps_for_modeling.py <config_file>")
    assert os.path.exists(config_file), "Cannot find config file: {}".format(config_file)
    return config_file


def process_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


def process_clickmap_files(
        clickme_data,
        min_clicks,
        max_clicks,
        image_path,
        file_inclusion_filter=None,
        file_exclusion_filter=None,
        process_max="trim"):
    clickmaps = {}
    if file_inclusion_filter == "CO3D_ClickmeV2" or file_inclusion_filter == "CO3D_ClickMe2":
        # Patch for val co3d
        image_files = glob(os.path.join(image_path, "**", "*.png"))
    for clicks, image_path, user_id in zip(
            clickme_data["clicks"].values,
            clickme_data["image_path"].values,
            clickme_data["user_id"].values
        ):
        # image_file_name = os.path.sep.join(image_path.split(os.path.sep)[-2:])
        image_file_name = image_path
        if file_inclusion_filter == "CO3D_ClickmeV2" or file_inclusion_filter == "CO3D_ClickMe2":
            # Patch for val co3d
            if not np.any([image_file_name in x for x in image_files]):
                continue
        elif file_inclusion_filter and file_inclusion_filter not in image_file_name:
            continue

        if isinstance(file_exclusion_filter, list):
            if any(f in image_file_name for f in file_exclusion_filter):
                continue
        elif file_exclusion_filter and file_exclusion_filter in image_file_name:
            continue

        if image_file_name not in clickmaps.keys():
            clickmaps[image_file_name] = [clicks]
        else:
            clickmaps[image_file_name].append(clicks)

    number_of_maps = []
    proc_clickmaps = {}
    n_empty_clickmap = 0
    for image in clickmaps:
        n_clickmaps = 0
        for clickmap in clickmaps[image]:
            if not len(clickmap):
                n_empty_clickmap += 1
                continue

            if isinstance(clickmap, str):
                clean_string = re.sub(r'[{}"]', '', clickmap)
                tuple_strings = clean_string.split(', ')
                data_list = tuple_strings[0].strip("()").split("),(")
                if len(data_list) == 1:  # Remove empty clickmaps
                    n_empty_clickmap += 1
                    continue
                tuples_list = [tuple(map(int, pair.split(','))) for pair in data_list]
            else:
                tuples_list = clickmap
                if len(tuples_list) == 1:  # Remove empty clickmaps
                    n_empty_clickmap += 1
                    continue

            if process_max == "exclude":
                if len(tuples_list) <= min_clicks | len(tuples_list) > max_clicks:
                    n_empty_clickmap += 1
                    continue
            elif process_max == "trim":
                if len(tuples_list) <= min_clicks:
                    n_empty_clickmap += 1
                    continue

                # Trim to the first max_clicks clicks
                tuples_list = tuples_list[:max_clicks]
            else:
                raise NotImplementedError(process_max)

            if image not in proc_clickmaps.keys():
                proc_clickmaps[image] = []

            n_clickmaps += 1
            proc_clickmaps[image].append(tuples_list)
        number_of_maps.append(n_clickmaps)
    return proc_clickmaps, number_of_maps


def process_clickmap_files_parallel(
        clickme_data,
        min_clicks,
        max_clicks,
        image_path,
        file_inclusion_filter=None,
        file_exclusion_filter=None,
        process_max="trim",
        n_jobs=-1):
    """Parallelized version of process_clickmap_files using joblib"""
    image_file_names = []
    def process_single_row(row):
        """Helper function to process a single row"""
        image_file_name = os.path.sep.join(row['image_path'].split(os.path.sep)[-2:])
        # Handle CO3D_ClickmeV2 special case
        if file_inclusion_filter == "CO3D_ClickmeV2" or file_inclusion_filter == "CO3D_ClickMe2" :
            image_files = glob(os.path.join(image_path, "**", "*.png"))
            if not np.any([image_file_name in x for x in image_files]):
                return None
        elif file_inclusion_filter == 'CO3D_Constancy':
            image_files = glob(os.path.join(image_path, "*.png"))
            class_name = image_file_name.split('/')[0]
            folder_image_file_name = image_file_name.split('/')[1]
            if class_name not in folder_image_file_name:
                folder_image_file_name = f'{class_name}_{folder_image_file_name}'
            if not np.any([folder_image_file_name in x for x in image_files]):
                return None
            image_file_name = folder_image_file_name
            image_file_names.append(folder_image_file_name)  
        elif file_inclusion_filter and file_inclusion_filter not in image_file_name:
            return None

        if isinstance(file_exclusion_filter, list):
            if any(f in image_file_name for f in file_exclusion_filter):
                return None
        elif file_exclusion_filter and file_exclusion_filter in image_file_name:
            return None

        clickmap = row["clicks"]
        if isinstance(clickmap, str):
            clean_string = re.sub(r'[{}"\[\]]', '', clickmap)
            # tuple_strings = clean_string.split(', ')
            # data_list = tuple_strings.strip("()").split("),(")
            data_list = clean_string.strip("()").split("), (")
            if len(data_list) == 1:  # Remove empty clickmaps
                return None
            tuples_list = [tuple(map(int, pair.split(','))) for pair in data_list]
        else:
            tuples_list = clickmap
            if len(tuples_list) == 1:  # Remove empty clickmaps
                return None

        if process_max == "exclude":
            if len(tuples_list) <= min_clicks or len(tuples_list) > max_clicks:
                return None
        elif process_max == "trim":
            if len(tuples_list) <= min_clicks:
                return None
            # Trim to the first max_clicks clicks
            tuples_list = tuples_list[:max_clicks]
        else:
            raise NotImplementedError(process_max)

        return (image_file_name, tuples_list)

    # Process rows in parallel
    # results = Parallel(n_jobs=1)(
    #     delayed(process_single_row)(row) 
    #     for _, row in tqdm(clickme_data.iterrows(), total=len(clickme_data), desc="Processing clickmaps")
    # )

    results = []
    for _, row in tqdm(clickme_data.iterrows(), total=len(clickme_data), desc="Processing clickmaps"):
        single_row_result = process_single_row(row)
        results.append(single_row_result)

    # Combine results
    proc_clickmaps = {}
    number_of_maps = []
    
    for result in results:
        if result is not None:
            image_file_name, tuples_list = result
            if image_file_name not in proc_clickmaps:
                proc_clickmaps[image_file_name] = []
            proc_clickmaps[image_file_name].append(tuples_list)
    # Count number of maps per image
    for image in proc_clickmaps:
        number_of_maps.append(len(proc_clickmaps[image]))
    return proc_clickmaps, number_of_maps


def circle_kernel(size, sigma=None, device='cpu'):
    """
    Create a flat circular kernel where the values are the average of the total number of on pixels in the filter.

    Args:
        size (int): The diameter of the circle and the size of the kernel (size x size).
        sigma (float, optional): Not used for flat kernel. Included for compatibility. Default is None.
        device (str, optional): Device to place the kernel on. Default is 'cpu'.

    Returns:
        torch.Tensor: A 2D circular kernel normalized so that the sum of its elements is 1.
    """
    # Create a grid of (x,y) coordinates
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    center = (size - 1) / 2
    radius = (size - 1) / 2

    # Create a mask for the circle
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2

    # Initialize kernel with zeros and set ones inside the circle
    kernel = torch.zeros((size, size), dtype=torch.float32)
    kernel[mask] = 1.0

    # Normalize the kernel so that the sum of all elements is 1
    num_on_pixels = mask.sum()
    if num_on_pixels > 0:
        kernel = kernel / num_on_pixels

    # Add batch and channel dimensions
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Move to the specified device
    return kernel.to(device)


def convolve(heatmap, kernel, double_conv=False, device='cpu'):
    """
    Apply Gaussian blur to a heatmap.

    Args:
        heatmap (torch.Tensor): The input heatmap (3D or 4D tensor).
        kernel (torch.Tensor): The Gaussian kernel.

    Returns:
        torch.Tensor: The blurred heatmap (3D tensor).
    """
    # heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = F.conv2d(heatmap, kernel, padding='same')
    if double_conv:
        blurred_heatmap = F.conv2d(blurred_heatmap, kernel, padding='same')
    return blurred_heatmap.to(device)  # [0]



def save_single_clickmap(all_clickmaps, idx, img_name, image_path, file_inclusion_filter=None, save_dir=None):
    """Helper function to save a single clickmap"""
    # Check multiple possible paths for the image
    image_exists = False
    possible_paths = [
        os.path.join(image_path, img_name)  # Standard path
    ]
    
    # Add paths with filter variations if a filter is specified
    if file_inclusion_filter:
        possible_paths.extend([
            os.path.join(image_path.replace(file_inclusion_filter + os.path.sep, ""), img_name),  # Filter removed from middle
            os.path.join(image_path.replace(file_inclusion_filter, ""), img_name)  # Filter removed from beginning
        ])
    
    for path in possible_paths:
        if os.path.exists(path):
            image_exists = True
            break
    
    if not image_exists:
        return 0
    hmp = all_clickmaps[idx]
    # Save to disk
    np.save(
        os.path.join(save_dir, f"{img_name.replace('/', '_')}.npy"), 
        hmp
    )
    return 1


def combine_clickmaps(examples):
    """Combine all clickmaps into single images"""
    for idx, example in enumerate(examples):
        examples[idx]['clickmaps'] = examples[idx]['clickmaps'].sum(0)
    return examples


def save_clickmaps_parallel(all_clickmaps, final_keep_index, output_dir, experiment_name, image_path, n_jobs=-1, file_inclusion_filter=None):
    """
    Save clickmaps to disk in parallel.
    
    Parameters:
    -----------
    all_clickmaps : list
        List of clickmaps to save
    final_keep_index : list
        List of image names corresponding to the clickmaps
    output_dir : str
        Directory to save the clickmaps
    experiment_name : str
        Name of the experiment (used for creating subdirectory)
    image_path : str
        Path to the original images (used for checking existence)
    n_jobs : int
        Number of parallel jobs to run
    file_inclusion_filter : str, optional
        Filter to identify included files, used for path checking
        
    Returns:
    --------
    int
        Number of successfully saved files
    """
    
    # Create output directory if it doesn't exist
    save_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Use tqdm to show progress
    final_keep_index = np.asarray(final_keep_index)  # .reshape(-1, all_clickmaps[0].shape[0])[:, 0]
    with tqdm(total=len(final_keep_index), desc="Saving files in parallel", 
             colour="cyan") as save_pbar:
        
        # Process in smaller batches to update progress bar more frequently
        batch_size = max(1, min(100, len(final_keep_index) // 10))
        saved_count = 0
        for i in range(0, len(final_keep_index), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(final_keep_index))))
            batch_img_names = [final_keep_index[j] for j in batch_indices]
            
            # Save batch in parallel
            results = Parallel(n_jobs=1)(  # Force sequential
                delayed(save_single_clickmap)(all_clickmaps, j, img_name, image_path, file_inclusion_filter, save_dir) 
                for j, img_name in zip(batch_indices, batch_img_names)
            )
            
            # Update progress bar
            batch_saved = sum(results)
            saved_count += batch_saved
            save_pbar.update(len(batch_indices))
    
    return saved_count

def save_clickmaps_to_hdf5(all_clickmaps, final_keep_index, hdf5_path, clickmap_bins, n_jobs=1, compression="gzip", compression_level=0):
    """
    Save clickmaps to HDF5 file safely without running into "Too many open files" error.
    
    Parameters:
    -----------
    all_clickmaps : list
        List of clickmaps to save
    final_keep_index : list
        List of image names corresponding to the clickmaps
    hdf5_path : str
        Path to the HDF5 file
    n_jobs : int
        Number of parallel jobs to run (not used for HDF5 file access)
    compression : str or None
        Compression algorithm to use ("gzip", "lzf", etc.). If None, no compression is used.
    compression_level : int
        Compression level (1-9, higher = more compression but slower)
    clickmap_bins : dict
        Dictionary of clickmap bins
    Returns:
    --------
    int
        Number of successfully saved files
    """

    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    
    # Validate compression parameters -- disable for now
    compression = None
    compression_kwargs = {}
    valid_compression = False
    
    # Process all clickmaps in batches
    batch_size = 100
    total_items = len(final_keep_index)
    total_batches = (total_items + batch_size - 1) // batch_size
    saved_count = 0
    
    # Process in batches using a single file handle
    for batch_idx in tqdm(range(total_batches), desc="Saving to HDF5", unit="batch"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_items)
        batch_size_actual = end_idx - start_idx
        
        # Open HDF5 file once per batch
        with h5py.File(hdf5_path, 'a') as f:
            # Ensure the clickmaps group exists
            if "clickmaps" not in f:
                f.create_group("clickmaps")
                
            # Process each clickmap in the batch
            for i in range(start_idx, end_idx):
                # Get image name and clean it for HDF5
                img_name = final_keep_index[i]
                dataset_name = img_name.replace('/', '_')
                
                # Get the clickmap
                hmp = all_clickmaps[img_name]
                bin_clickmaps = clickmap_bins[img_name]
                # Check if dataset already exists and delete it if it does
                if dataset_name in f["clickmaps"]:
                    del f["clickmaps"][dataset_name]
                
                # Create a group for this image's data
                img_group = f["clickmaps"].create_group(dataset_name)
                
                # Create the main clickmap dataset
                if valid_compression:
                    img_group.create_dataset("clickmap", data=hmp, **compression_kwargs)
                    img_group.create_dataset("bin_clickmaps", data=np.array(bin_clickmaps), **compression_kwargs)
                else:
                    img_group.create_dataset("clickmap", data=hmp)
                    img_group.create_dataset("bin_clickmaps", data=np.array(bin_clickmaps)) 

                # Add metadata about the dataset
                ds = f["clickmaps"][dataset_name]
                ds.attrs["shape"] = hmp.shape
                ds.attrs["original_path"] = img_name
                saved_count += 1
            
            # Update metadata after each batch
            if "metadata" not in f:
                f.create_group("metadata")
            
            f["metadata"].attrs["total_clickmaps_so_far"] = saved_count
            f["metadata"].attrs["last_updated"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            if valid_compression:
                f["metadata"].attrs["compression"] = np.bytes_(compression)
                f["metadata"].attrs["compression_level"] = compression_level
            else:
                f["metadata"].attrs["compression"] = np.bytes_("none")
            
            # Explicit flush to ensure data is written
            f.flush()
        
        # Add a small delay to ensure file is properly closed
        time.sleep(0.1)
    
    # Final update to metadata
    try:
        with h5py.File(hdf5_path, 'a') as f:
            if "metadata" not in f:
                f.create_group("metadata")
            
            f["metadata"].attrs["total_clickmaps"] = saved_count
            f["metadata"].attrs["completed"] = True
            f["metadata"].attrs["last_updated"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print(f"Warning: Could not update final HDF5 metadata: {e}")
    
    return saved_count


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_medians(clickmaps, mode, thresh=95):
    """Compute median number of clicks in each category"""
    medians = {}
    if mode == 'all':
        csize = [len(x) for x in clickmaps.values()]
        if not csize:
            return {}
        medians['median'] = int(np.median(csize))
        medians['mean'] = float(np.mean(csize))
        medians['threshold'] = int(np.percentile(csize, thresh))
    elif mode == 'image':
        medians['image'] = {}
        for img_name, clicks in clickmaps.items():
            medians['image'][img_name] = int(np.median([len(x) for x in clicks]))
    elif mode == 'category':
        medians['category'] = {}
        cats = {}
        for img_name, clicks in clickmaps.items():
            img_category = img_name.split('/')[0]
            if img_category not in cats:
                cats[img_category] = []
            cats[img_category].append(len(clicks))
        for category, counts in cats.items():
            medians['category'][category] = int(np.median(counts))
    return medians


def filter_duplicate_participants(clickme_data):
    """
    Filter out duplicate submissions from the same participant for each image
    keeping only the first submission.
    
    Parameters:
    -----------
    clickme_data : pandas.DataFrame
        DataFrame containing clickme data with at least 'image_path' and 'participant' columns
        
    Returns:
    --------
    filtered_data : pandas.DataFrame
        DataFrame with duplicate participant submissions removed
    """
    print("Filtering duplicate participant submissions...")
    # Make a copy to avoid modifying the original
    data = clickme_data.copy()
    
    # Check if 'user_id' column exists
    if 'user_id' not in data.columns:
        print("Warning: Cannot filter duplicate participants - 'user_id' column missing")
        return data
    
    # Count before filtering
    total_before = len(data)
    
    # Sort by timestamp if available to ensure we keep earliest submission
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp')
    
    # Keep only the first occurrence of each participant-image combination
    filtered_data = data.drop_duplicates(subset=['user_id', 'image_path'], keep='first')
    
    # Count after filtering
    total_after = len(filtered_data)
    removed = total_before - total_after
    
    print(f"Removed {removed} duplicate participant submissions ({removed/total_before:.2%} of total)")
    
    return filtered_data


def process_all_maps_gpu(
        clickmaps, 
        config, 
        metadata=None, 
        create_clickmap_func=None, 
        fast_duplicate_detection=None,
        average_maps=True,
        blur_sigma_function=None,
        max_kernel_size=51
        ):
    """
    Simplified function to blur clickmaps on GPU in batches with adaptive kernel sizing
    """
    import torch
    from tqdm import tqdm
    import numpy as np
    
    # Extract basic parameters
    blur_size = config["blur_size"]
    blur_sigma = config.get("blur_sigma", blur_size)
    image_shape = config["image_shape"]
    min_subjects = config["min_subjects"]
    min_clicks = config["min_clicks"]
    
    # Get GPU batch size for processing
    gpu_batch_size = config.get("gpu_batch_size", 4096)
    
    # Set default blur_sigma_function if not provided
    if blur_sigma_function is None:
        blur_sigma_function = lambda x: x
    
    print(f"Processing {len(clickmaps)} unique images with GPU (batch size: {gpu_batch_size})...")
    
    # Step 1: Prepare binary maps and average them
    print("Pre-processing clickmaps on CPU...")
    
    # Prepare data structures
    all_clickmaps = []
    keep_index = []
    categories = []
    final_clickmaps = {}
    click_counts = {}  # Track click counts for each image
    image_metadata = {}  # Track metadata for each processed image
    
    # Preprocess all clickmaps first to binary maps
    for key, trials in tqdm(clickmaps.items(), desc="Creating binary maps"):
        if len(trials) < min_subjects:
            continue
            
        # Create binary clickmaps
        if metadata and key in metadata:
            native_size = metadata[key]
            binary_maps = np.asarray([create_clickmap_func([trial], native_size[::-1]) for trial in trials])
            image_metadata[key] = native_size
        else:
            binary_maps = np.asarray([create_clickmap_func([trial], tuple(image_shape)) for trial in trials])
            image_metadata[key] = None
        
        # Count total clicks in each trial
        total_clicks_per_trial = len(binary_maps)
        
        # Only keep maps with enough valid pixels using mask
        mask = binary_maps.sum((-2, -1)) >= min_clicks
        binary_maps = binary_maps[mask]
        
        # If we have enough valid maps, average them and keep this image
        if len(binary_maps) >= min_subjects:
            if average_maps:
                all_clickmaps.append(np.array(binary_maps).mean(0, keepdims=True))
            else:
                all_clickmaps.append(np.array(binary_maps))
            # Note that if we are measuring ceiling we need to keep all maps ^^ change above.
            categories.append(key.split("/")[0])
            keep_index.append(key)
            final_clickmaps[key] = trials
            click_counts[key] = total_clicks_per_trial  # Store total clicks for this image
    
    if not all_clickmaps:
        print("No valid clickmaps to process")
        return {}, [], [], {}, {}
    
    # Step 2: Prepare for batch blurring on GPU with adaptive kernel sizing
    total_maps = len(all_clickmaps)
    print(f"Preparing to blur {total_maps} image clickmaps using GPU with adaptive kernel sizing...")
    
    # Convert all maps to tensors
    all_tensors = [torch.from_numpy(maps).float() for maps in all_clickmaps]
    
    # Group images by their required kernel size to batch efficiently
    kernel_groups = {}
    for idx, key in enumerate(keep_index):
        native_size = image_metadata[key]
        
        if native_size is not None:
            # Calculate adaptive kernel size
            short_side = min(native_size)
            scale = short_side / min(image_shape)
            adj_blur_size = int(np.round(blur_size * scale))
            if not adj_blur_size % 2:
                adj_blur_size += 1
            adj_blur_size = min(adj_blur_size, max_kernel_size)
            adj_blur_sigma = blur_sigma_function(adj_blur_size)
        else:
            # Use default kernel size
            adj_blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            adj_blur_sigma = blur_sigma
        
        kernel_key = (adj_blur_size, adj_blur_sigma)
        if kernel_key not in kernel_groups:
            kernel_groups[kernel_key] = []
        kernel_groups[kernel_key].append(idx)
    
    print(f"Processing {len(kernel_groups)} different kernel sizes...")
    
    # Process each kernel group separately
    for (kernel_size, kernel_sigma), image_indices in tqdm(kernel_groups.items(), desc="Processing kernel groups"):
        print(f"Processing {len(image_indices)} images with kernel size {kernel_size}, sigma {kernel_sigma}")
        
        # Create kernel for this group
        # kernel = circle_kernel(kernel_size, kernel_sigma, 'cuda')
        kernel = circle_kernel(kernel_size, kernel_size, 'cuda')  # Matching kernel size and kernel sigma
        
        # Process images in this group in batches
        group_batch_size = min(gpu_batch_size, len(image_indices))
        num_batches = (len(image_indices) + group_batch_size - 1) // group_batch_size
        
        for batch_idx in range(num_batches):
            # Get batch indices for this kernel group
            batch_start = batch_idx * group_batch_size
            batch_end = min(batch_start + group_batch_size, len(image_indices))
            batch_image_indices = image_indices[batch_start:batch_end]
            
            # Get tensors for this batch
            batch_tensors = [all_tensors[idx] for idx in batch_image_indices]
            
            try:
                # Try to concatenate tensors (works if all have same shape)
                batch_tensor = torch.cat(batch_tensors, dim=0).unsqueeze(1).to('cuda')
                
                # Apply blurring to this batch
                blurred_tensor = convolve(batch_tensor, kernel, double_conv=True)
                
                # Convert back to numpy and update results
                blurred_maps = blurred_tensor.squeeze(1).cpu().numpy()
                
                for i, img_idx in enumerate(batch_image_indices):
                    all_clickmaps[img_idx] = blurred_maps[i:i+1]  # Keep the same shape with extra dimension
                
                # Clean up GPU memory for this batch
                del batch_tensor, blurred_tensor
                torch.cuda.empty_cache()
                
            except Exception as e:
                # If concatenation fails (different shapes), process individually
                print(f"Batch processing failed, processing {len(batch_tensors)} images individually: {e}")
                for i, img_idx in enumerate(batch_image_indices):
                    tensor = batch_tensors[i].unsqueeze(1).to('cuda')
                    blurred_tensor = convolve(tensor, kernel, double_conv=True)
                    all_clickmaps[img_idx] = blurred_tensor.squeeze(1).cpu().numpy()
                    
                    # Clean up GPU memory
                    del tensor, blurred_tensor
                    torch.cuda.empty_cache()
        
        # Clean up kernel for this group
        del kernel
        torch.cuda.empty_cache()
    
    print(f"Finished blurring {total_maps} clickmaps with adaptive kernel sizing. Kept {len(keep_index)} images.")
    return final_clickmaps, all_clickmaps, categories, keep_index, click_counts


def process_all_maps_multi_thresh_gpu(
        clickmaps, 
        config, 
        metadata=None, 
        create_clickmap_func=None, 
        fast_duplicate_detection=None,
        average_maps=True,
        thresholds=10,
        return_before_blur=False,
        time_based_bins=False,
        save_to_disk=False,
        maximum_length=5000,
        ):
    """
    Simplified function to blur clickmaps on GPU in batches with adaptive kernel sizing
    """
    if save_to_disk:
        assert return_before_blur
        temp_file = h5py.File(config['temp_dir'], 'w')
        temp_group = temp_file.create_group("clickmaps")
    # Extract basic parameters
    blur_size = config["blur_size"]
    blur_sigma = config.get("blur_sigma", blur_size)
    image_shape = config["image_shape"]
    min_subjects = config["min_subjects"]
    max_subjects = config["max_subjects"]
    min_clicks = config["min_clicks"]
    max_kernel_size = config.get("max_kernel_size", 51)
    blur_sigma_function = config.get("blur_sigma_function", lambda x: x)
    
    # Get GPU batch size for processing
    gpu_batch_size = config.get("gpu_batch_size", 4096)
    
    # Get max GPU memory usage for shape batches (in GB)
    max_gpu_memory_gb = config.get("max_gpu_memory_gb", 8)
    
    print(f"Processing {len(clickmaps)} unique images with GPU (batch size: {gpu_batch_size})...")
    
    # Step 1: Prepare binary maps and average them
    print("Pre-processing clickmaps on CPU...")
    
    # Prepare data structures
    all_clickmaps = {}
    keep_index = []
    categories = []
    final_clickmaps = {}
    clickmap_bins = {}
    click_counts = {}  # Track click counts for each image
    total_maps = 0
    if save_to_disk:
        save_count = 0
    # Preprocess all clickmaps first to binary maps
    for clickmap_idx, (key, trials) in tqdm(enumerate(clickmaps.items()), "Pre-processing on CPU"):
        if len(trials) < min_subjects:
            # print("Not enough subjects", key, len(trials))
            continue
        if time_based_bins:
            lens = [len(x) for x in trials]
            bins = []
            for trial in trials:
                max_count = len(trial)
                half_count = int(max_count/2)
                trial_bin = np.linspace(max(half_count * .1, min_clicks), max_count, thresholds).astype(int)
                bins.append(trial_bin)
            bin_clickmaps = []
            bin_counts = []
            for i in range(thresholds):
                thresholded_trials = []
                for j, trial in enumerate(trials):
                    thresholded_trials.append(trial[:bins[j][i]])
                bin_counts.append(sum([len(x) for x in thresholded_trials]))
                # Create binary clickmaps
                if metadata and key in metadata:
                    native_size = metadata[key]
                    binary_maps = np.asarray([create_clickmap_func([trial], native_size[::-1]) for trial in thresholded_trials])
                else:
                    binary_maps = np.asarray([create_clickmap_func([trial], tuple(image_shape)) for trial in thresholded_trials])
                # Only keep maps with enough valid pixels using mask
                mask = binary_maps.sum((-2, -1)) >= min_clicks
                # org_number = binary_maps.sum((-2, -1))
                binary_maps = binary_maps[mask]
                # If we have enough valid maps, average them and keep this image
                if average_maps:
                    bin_clickmaps.append(np.array(binary_maps).mean(0, keepdims=True))
                else:
                    bin_clickmaps.append(np.array(binary_maps))
                # else:
                #     print("Not enough subjects", key, len(binary_maps))                    

        else:
            # Get max count then do thresholds from that
            lens = [len(x) for x in trials]
            max_count = max(lens)
            # min_count = max(int(min(lens) * .1), 1)
            # bins = np.linspace(min_count, max_count + 1, thresholds).astype(int)
            mean_lens = int(np.mean(lens))
            below_mean = np.linspace(mean_lens * .1, mean_lens + 1, thresholds // 2).astype(int)
            above_mean = np.linspace(mean_lens + 1, max_count + 1, thresholds // 2).astype(int)
            bins = np.concatenate([below_mean, above_mean])
            # bins = np.concatenate([below_mean, above_mean])
            bin_clickmaps = []
            bin_counts = []
            
            for bin in bins:
                # Threshold trials
                thresholded_trials = [x[:bin] for x in trials]
                bin_counts.append(sum([len(x) for x in thresholded_trials]))

                # Create binary clickmaps
                if metadata and key in metadata:
                    native_size = metadata[key]
                    binary_maps = np.asarray([create_clickmap_func([trial], native_size[::-1]) for trial in thresholded_trials])
                else:
                    binary_maps = np.asarray([create_clickmap_func([trial], tuple(image_shape)) for trial in thresholded_trials])
                
                # Only keep maps with enough valid pixels using mask
                mask = binary_maps.sum((-2, -1)) >= min_clicks
                binary_maps = binary_maps[mask]
                # If we have enough valid maps, average them and keep this image
                if average_maps:
                    bin_clickmaps.append(np.array(binary_maps).mean(0, keepdims=True))
                else:
                    bin_clickmaps.append(np.array(binary_maps))
                # else:
                #     print("Not enough subjects", key, len(binary_maps))
        
        # Skip if we don't have any valid bin_clickmaps
        if not bin_clickmaps:
            continue
            
        # For stacking, check if all maps have same shape
        if return_before_blur:
            shapes = [m.shape for m in bin_clickmaps]
            if len(set(str(s) for s in shapes)) > 1:
                # print(f"Warning: Inconsistent shapes for {key}: {shapes}, fixing.")
                min_shape = min([s[0] for s in shapes])
                if min_shape <= 1:
                    print(f"Minimum shape is 0 for {key}")
                    sys.exit()
                bin_clickmaps = [m[:min_shape] for m in bin_clickmaps]
                # continue

        # Only add to tracking structures if we can successfully process this image
        categories.append(key.split("/")[0])
        keep_index.append(key)
        final_clickmaps[key] = trials
        click_counts[key] = len(trials)  # Store total clicks for this image
        clickmap_bins[key] = np.asarray(bin_counts)
        # Add to all_clickmaps with the appropriate method
        if save_to_disk:
            key = key.replace('/', '_')
            temp_group.create_dataset(key, data=np.stack(bin_clickmaps, axis=0))

        elif return_before_blur:
            bin_clickmaps = np.stack(bin_clickmaps, axis=0)
            if max_subjects > 0:
                max_subjects = min(max_subjects, bin_clickmaps.shape[1])
                bin_clickmaps = bin_clickmaps[:, :max_subjects, :, :]
            all_clickmaps[key] = (np.stack(bin_clickmaps, axis=0))
        else:
            all_clickmaps[key] = (np.concatenate(bin_clickmaps, axis=0))
    if save_to_disk:
        temp_file.close()

    if not save_to_disk and not all_clickmaps:
        print("No valid clickmaps to process")
        return {}, {}, [], [], {}, {}
    
    if return_before_blur:
        return final_clickmaps, all_clickmaps, categories, keep_index, click_counts, clickmap_bins
    
    # Step 2: Prepare for batch blurring on GPU with adaptive kernel sizing
    total_maps = len(all_clickmaps)
    print(f"Preparing to blur {total_maps} image clickmaps using GPU with adaptive kernel sizing...")
    
    # Convert all maps to tensors
    # all_tensors = [torch.from_numpy(maps).float() for maps in all_clickmaps]
    
    # Group images by their required kernel size to batch efficiently
    kernel_groups = {}
    for idx, key in enumerate(keep_index):
        if metadata and key in metadata:
            # Calculate adaptive kernel size for this specific image
            native_size = metadata[key]
            short_side = min(native_size)
            scale = short_side / min(image_shape)
            adj_blur_size = int(np.round(blur_size * scale))
            if not adj_blur_size % 2:
                adj_blur_size += 1
            adj_blur_size = min(adj_blur_size, max_kernel_size)
            adj_blur_sigma = blur_sigma_function(adj_blur_size)
        else:
            # Use default kernel size for images without metadata
            adj_blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
            adj_blur_sigma = blur_sigma_function(adj_blur_size)
        
        kernel_key = (adj_blur_size, adj_blur_sigma)
        if kernel_key not in kernel_groups:
            kernel_groups[kernel_key] = []
        kernel_groups[kernel_key].append(key)
    print(f"Processing {len(kernel_groups)} different kernel sizes...")

    # Process each kernel group separately
    for (kernel_size, kernel_sigma), image_keys in tqdm(kernel_groups.items(), desc="Processing kernel groups"):
        print(f"Processing {len(image_keys)} images with kernel size {kernel_size}, sigma {kernel_sigma}")
        # print(f"Processing {len(image_indices)} images with kernel size {kernel_size}, sigma {kernel_sigma}")
        
        # Create kernel for this group
        # kernel = circle_kernel(kernel_size, kernel_sigma, 'cuda')
        kernel = circle_kernel(kernel_size, kernel_sigma, 'cuda')
        
        # Process images in this group in batches
        group_batch_size = min(gpu_batch_size, len(image_keys))
        num_batches = (len(image_keys) + group_batch_size - 1) // group_batch_size
        for batch_idx in range(num_batches):
            # Get batch indices for this kernel group
            batch_start = batch_idx * group_batch_size
            batch_end = min(batch_start + group_batch_size, len(image_keys))
            batch_image_keys = image_keys[batch_start:batch_end]
            # Get tensors for this batch
            # batch_tensors = [all_tensors[idx] for idx in batch_image_indices]
            batch_tensors = {}
            for key in batch_image_keys:
                single_tensor = torch.from_numpy(all_clickmaps[key]).float()
                batch_tensors[key] = single_tensor
            # Group batch tensors by shape to handle different dimensions within the same kernel group
            shape_groups = {}
            for i, (key, tensor) in enumerate(batch_tensors.items()):
                shape_key = tuple(tensor.shape)
                if shape_key not in shape_groups:
                    shape_groups[shape_key] = []
                shape_groups[shape_key].append((i, tensor, key))
            # Process each shape group separately
            for shape, tensor_data in shape_groups.items():
                indices, tensors, img_keys = zip(*tensor_data)
                # Calculate memory-safe batch size for this shape group
                # Estimate memory usage: shape[0] * shape[1] * shape[2] * 4 bytes per float32
                memory_per_tensor = shape[0] * shape[1] * shape[2] * 4  # bytes
                # Target max configurable GB per batch to be safe (leave room for intermediate tensors)
                max_memory_bytes = max_gpu_memory_gb * 1024 * 1024 * 1024  # Convert GB to bytes
                safe_batch_size = min(len(tensors), max(1, max_memory_bytes // memory_per_tensor))
                
                # Process in smaller sub-batches if needed
                num_shape_batches = (len(tensors) + safe_batch_size - 1) // safe_batch_size
                
                if num_shape_batches > 1:
                    print(f"  Shape {shape}: splitting {len(tensors)} tensors into {num_shape_batches} batches of max {safe_batch_size} (estimated {memory_per_tensor/(1024**3):.2f} GB per tensor)")
                
                for shape_batch_idx in range(num_shape_batches):
                    start_idx = shape_batch_idx * safe_batch_size
                    end_idx = min(start_idx + safe_batch_size, len(tensors))
                    
                    batch_tensors_subset = tensors[start_idx:end_idx]
                    batch_img_keys_subset = img_keys[start_idx:end_idx]
                    try:
                        # Try to concatenate tensors of the same shape
                        shape_batch_tensor = torch.cat(batch_tensors_subset, dim=0).unsqueeze(1).to('cuda')
                        
                        # Apply blurring to this shape batch
                        blurred_tensor = convolve(shape_batch_tensor, kernel, double_conv=True)
                        # Convert back to numpy and update results
                        blurred_maps = blurred_tensor.squeeze(1).cpu().numpy()
                        for i, img_key in enumerate(batch_img_keys_subset):
                        
                            all_clickmaps[img_key] = blurred_maps[i*thresholds:(i+1)*thresholds]  # Keep the same shape with extra dimension
                        # Clean up GPU memory for this shape batch
                        del shape_batch_tensor, blurred_tensor
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        # If concatenation still fails, process individually
                        print(f"Shape batch processing failed for shape {shape} (batch {shape_batch_idx+1}/{num_shape_batches}), processing {len(batch_tensors_subset)} images individually: {e}")
                        for i, (tensor, img_key) in enumerate(zip(batch_tensors_subset, batch_img_keys_subset)):
                            gpu_tensor = tensor.unsqueeze(1).to('cuda')
                            blurred_tensor = convolve(gpu_tensor, kernel, double_conv=True)
                            all_clickmaps[img_key] = blurred_tensor.squeeze(1).cpu().numpy()
                            
                            # Clean up GPU memory
                            del gpu_tensor, blurred_tensor
                            torch.cuda.empty_cache()
        
        # Clean up kernel for this group
        del kernel
        torch.cuda.empty_cache()
    return final_clickmaps, all_clickmaps, categories, keep_index, click_counts, clickmap_bins

def blur_maps_for_cf(all_clickmaps, blur_size, blur_sigma, gpu_batch_size, native_size=None):
    # Step 2: Prepare for batch blurring on GPU
    total_maps = len(all_clickmaps)
    
    # Convert all maps to tensors
    all_tensors = [torch.from_numpy(maps).float() for maps in all_clickmaps]
    # Create circular kernel
    if blur_size % 2 == 0:
        adjusted_blur_size = blur_size + 1  # Ensure odd kernel size
    else:
        adjusted_blur_size = blur_size
        
    kernel = circle_kernel(adjusted_blur_size, blur_sigma, 'cuda')
    
    # Process in batches based on the GPU batch size
    try:
        # Just check if we can access the first tensor
        _ = all_tensors[0].shape
        num_batches = (total_maps + gpu_batch_size - 1) // gpu_batch_size
    except Exception as e:
        print(f"Error accessing tensors: {e}")
        num_batches = total_maps
        gpu_batch_size = 1
    
    for batch_idx in range(num_batches):
        # Get batch indices
        start_idx = batch_idx * gpu_batch_size
        end_idx = min(start_idx + gpu_batch_size, total_maps)
        current_batch_size = end_idx - start_idx
        
        # Get tensors for this batch
        batch_tensors = all_tensors[start_idx:end_idx]
        
        # Track original shapes and flatten all maps for processing
        original_shapes = []
        flattened_maps = []
        map_indices = []  # Keep track of which image each map belongs to
        
        # Prepare flattened list of maps with their indices
        for i, tensor in enumerate(batch_tensors):
            batch_img_idx = start_idx + i
            original_shapes.append(tensor.shape)
            # For each map in this tensor
            for j in range(tensor.shape[0]):
                flattened_maps.append(tensor[j:j+1])  # Keep as [1, H, W]
                map_indices.append(batch_img_idx)
        
        # Process maps in smaller sub-batches to avoid memory issues
        if flattened_maps:
            sub_batch_size = 256  # Smaller sub-batch size for stability
            num_maps = len(flattened_maps)
            num_sub_batches = (num_maps + sub_batch_size - 1) // sub_batch_size
            
            processed_maps = []
            for sub_idx in range(num_sub_batches):
                sub_start = sub_idx * sub_batch_size
                sub_end = min(sub_start + sub_batch_size, num_maps)
                
                # Get maps for this sub-batch
                sub_batch_maps = flattened_maps[sub_start:sub_end]
                
                # Concatenate and process
                sub_batch_tensor = torch.cat(sub_batch_maps, dim=0).unsqueeze(1).to('cuda')
                blurred_tensor = convolve(sub_batch_tensor, kernel, double_conv=True)
                
                # Store processed maps
                processed_maps.extend(blurred_tensor.squeeze(1).cpu())
                
                # Clean up GPU memory
                del sub_batch_tensor, blurred_tensor
                torch.cuda.empty_cache()
            
            # Reconstruct the original tensor structures
            for i, tensor_shape in enumerate(original_shapes):
                batch_img_idx = start_idx + i
                
                # Find all maps for this image
                img_map_indices = [j for j, idx in enumerate(map_indices) if idx == batch_img_idx]
                img_maps = [processed_maps[j] for j in img_map_indices]
                
                # Stack them back to original shape
                stacked_maps = torch.stack(img_maps, dim=0)
                
                # Verify shape matches the original
                if stacked_maps.shape != tensor_shape:
                    print(f"Warning: Reconstructed shape {stacked_maps.shape} doesn't match original {tensor_shape}")
                
                # Update the original tensor with processed maps
                all_clickmaps[batch_img_idx] = stacked_maps.numpy()
            
            # Clean up
            del processed_maps, flattened_maps, map_indices
            torch.cuda.empty_cache()
    
    # Clean up remaining GPU memory
    del kernel
    torch.cuda.empty_cache()    
    return all_clickmaps

def sparse_scale(img, scale, device='cpu', pad=True):
    if isinstance(img, np.ndarray):
        img = torch.tensor(img).to(device)
    input_shape = img.shape

    if len(img.shape) < 3:
        img = img[None, :, :]    
    
    B, org_h, org_w = img.shape
    new_h = int(org_h/scale)
    new_w = int(org_w/scale)
    scaled_img = torch.zeros((B, new_h, new_w), dtype=img.dtype, device=device)
    count_img = torch.zeros((B, new_h, new_w), dtype=img.dtype, device=device)

    i = torch.arange(org_h, device=device)
    j = torch.arange(org_w, device=device)
    ii, jj = torch.meshgrid(i, j, indexing='ij')
    i_scaled = (ii.float()/scale).long().clamp(0, new_h-1)
    j_scaled = (jj.float()/scale).long().clamp(0, new_w-1)

    for b in range(B):
        scaled_img[b].index_put_(
            (i_scaled, j_scaled),
            img[b], accumulate=True
        )
        count_img[b].index_put_(
            (i_scaled, j_scaled),
            torch.ones_like(img[b]),
            accumulate=True
        )
    
    count_img[count_img==0] = 1
    scaled_img = scaled_img/count_img
    if pad:
        pad_h = (org_h - new_h) // 2
        pad_w = (org_w - new_w) // 2
        diff_h = org_h - new_h - pad_h*2
        diff_w = org_w - new_w - pad_w*2
        padded_img = F.pad(scaled_img, (pad_h, pad_h+diff_h, pad_w, pad_w+diff_w))
        padded_img = padded_img.reshape(input_shape)
        return padded_img
    else:
        return scaled_img
        
def scale_img(img, scale, device='cpu'):
    if isinstance(img, np.ndarray):
        img = torch.tensor(img)
        
    if isinstance(img, torch.Tensor):
        if len(img.shape) < 3:
            img = img[None, None, :, :]
        if len(img.shape) < 4:
            img = img[:, None, :, :]
        org_x = img.shape[-2]
        org_y = img.shape[-1]
        new_x = int(org_x/scale)
        new_y = int(org_y/scale)
        scaled_size = [new_x, new_y]
        img = F.interpolate(img, size=scaled_size, mode="nearest-exact")
        pad_x = (org_x - new_x) // 2
        pad_y = (org_y - new_y) // 2
        diff_x = org_x - new_x - pad_x*2
        diff_y = org_y - new_y - pad_y*2
        new_img = F.pad(img, (pad_x, pad_x+diff_x, pad_y, pad_y+diff_y))
        new_img = new_img.squeeze()
        # if new_img.shape[1] == 1:
        #     new_img = new_img.squeeze(dim=1)
    elif isinstance(img, Image):
        img = img.convert("RGB")
        org_size = img.size
        new_size = (int(org_size[0]/scale), int(org_size[1]/scale))
        new_img = Image.new("RGB", org_size, (0, 0, 0))
        paste_x = (org_size[0] - img.width) // 2
        paste_y = (org_size[1] - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
    return new_img

def to_torch(x, device, dtype):
    if isinstance(x, torch.Tensor):
        return x.to(device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)

def project_img_gpu(img, depth, target_depth, w2c_s, w2c_t, K_s, K_t, device):

    org_dtype = img.dtype if hasattr(img, 'dtype') else 'float32'
    input_shape = img.shape
    was_tensor = isinstance(img, torch.Tensor)
    
    # Move everything to torch
    if isinstance(img, torch.Tensor):
        img = img
    else:
        img = torch.tensor(img).float()
    dtype = img.dtype
    
    img = img.to(device)
    depth = to_torch(depth, device, dtype)
    # Convert to numpy to avoid lazy tensor operation in parallel
    if isinstance(K_s, torch.Tensor):
        K_s = K_s.cpu().numpy()
    K_s_inv = np.linalg.inv(K_s)
    K_s_inv = to_torch(K_s_inv, device, dtype)
    K_t = to_torch(K_t, device, dtype)
    w2c_s = to_torch(w2c_s, device, dtype)
    w2c_t = to_torch(w2c_t, device, dtype)

    if img.ndim < 3:
        img = img.unsqueeze(0)  # (1,H,W)
    elif img.ndim > 3:
        img = img.reshape(-1, img.shape[-2], img.shape[-1])
    C, H, W = img.shape
    assert depth.shape[-2:] == (H, W), "depth must match HxW of img"
    R_s, T_s = w2c_s[:3, :3], w2c_s[:3, 3]
    R_t, T_t = w2c_t[:3, :3], w2c_t[:3, 3]

    # ---- Pixel grid ----
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )

    pixels_h = torch.stack([xs, ys, torch.ones_like(xs)], dim=0).reshape(3, -1)  # (3, N)
    depth_flat = depth.reshape(-1)  # (N,)

    # ---- Back-project to 3D in source camera ----
    # K_s_inv   = torch.inverse(K_s)
    cam_pts_s = (K_s_inv @ pixels_h) * depth_flat  # (3, N)

    # ---- Transform to world then to target camera ----
    X_world = R_s.transpose(0, 1) @ (cam_pts_s - T_s.view(3, 1))
    X_cam_t = R_t @ X_world + T_t.view(3, 1)  # (3,N)

    # ---- Project with K_t ----
    proj = K_t @ X_cam_t  # (3,N)
    z_proj = proj[2]
    x_proj = proj[:2] / z_proj.clamp(min=1e-8)

    # ---- Integer pixel positions (nearest) ----
    x_t = torch.round(x_proj[0]).long()
    y_t = torch.round(x_proj[1]).long()

    # ---- Valid mask ----
    valid = (
        (x_t >= 0) & (x_t < W) &
        (y_t >= 0) & (y_t < H) # &
        # (z_proj > 0)
    )


    if not valid.any():
        out = torch.zeros_like(img)
        if not was_tensor:
            out = out.cpu().numpy()
        return out.reshape(input_shape)

    x_t = x_t[valid]
    y_t = y_t[valid]
    z_t = z_proj[valid]

    img_flat = img.view(img.shape[0], -1)[:, valid]  # (C, N_valid)

    # ---- Z-buffer via scatter_reduce (amin) ----
    flat_indices = y_t * W + x_t  # (N_valid,)
    num_pixels = H * W

    # Per-target-pixel min depth
    # z_min = torch.full((num_pixels,), float('inf'), device=device, dtype=z_t.dtype)
    # Use this to mask any point that's behind the actual object
    z_min = torch.tensor(target_depth*1.1).reshape(-1).to(device).to(z_t.dtype)
    # PyTorch >= 1.12: Tensor.scatter_reduce_
    z_min = z_min.scatter_reduce(0, flat_indices, z_t, reduce='amin', include_self=True)

    # Keep only points that match that minimum depth
    keep = (z_t == z_min[flat_indices])

    flat_indices = flat_indices[keep]
    src_vals = img_flat[:, keep]  # (C, K)

    target = torch.zeros((C, num_pixels), device=device, dtype=img.dtype)
    target.index_copy_(1, flat_indices, src_vals)   # (C, HW)

    target = target.view(C, H, W)

    # Restore original shape/dtype
    if not was_tensor:
        target = target.detach().cpu().numpy().astype(org_dtype)
    target = target.reshape(input_shape)
    return target

def get_scale_target(img_idx, zoom):
    target_img_ids = []
    target_img_diffs = []
    if zoom == 0:
        target_img_ids = [img_idx, img_idx + 1, img_idx + 2]
        target_img_diffs = [0, 1, 2]
    elif zoom == 1:
        target_img_ids = [img_idx-1, img_idx, img_idx+1]
        target_img_diffs = [-1, 0, 1]
    elif zoom == 2:
        target_img_ids = [img_idx-2, img_idx-1, img_idx]
        target_img_diffs = [-2, -1, 0]
    return target_img_ids, target_img_diffs

def get_rot_target(img_idx):
    target_img_ids = []
    target_img_diffs = []
    for i in range(-3, 5):
        target_img_idx = (img_idx + i*3 + 24) % 24
        target_img_ids.append(target_img_idx)
        target_img_diffs.append(i)

    return target_img_ids, target_img_diffs

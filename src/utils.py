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
import psutil

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
        return pd.read_csv(data_file)
    elif "npz" in data_file:
        data = np.load(data_file, allow_pickle=True)
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

        # Create dataframe
        df = pd.DataFrame({"image_path": image_path, "clicks": clicks, "user_id": user_id})

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
    
    def process_single_row(row):
        """Helper function to process a single row"""
        image_file_name = os.path.sep.join(row['image_path'].split(os.path.sep)[-2:])
        
        # Handle CO3D_ClickmeV2 special case
        # if file_inclusion_filter == "CO3D_ClickmeV2" or file_inclusion_filter == "CO3D_ClickMe2":
        #     import pdb;pdb.set_trace()
        #     image_files = glob(os.path.join(image_path, "**", "*.png"))
        #     if not np.any([image_file_name in x for x in image_files]):
        #         return None
        # elif file_inclusion_filter and file_inclusion_filter not in image_file_name:
        #     return None
        if file_inclusion_filter and file_inclusion_filter not in image_file_name:
            return None

        if isinstance(file_exclusion_filter, list):
            if any(f in image_file_name for f in file_exclusion_filter):
                return None
        elif file_exclusion_filter and file_exclusion_filter in image_file_name:
            return None

        clickmap = row["clicks"]
        if isinstance(clickmap, str):
            clean_string = re.sub(r'[{}"]', '', clickmap)
            tuple_strings = clean_string.split(', ')
            data_list = tuple_strings[0].strip("()").split("),(")
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
    results = Parallel(n_jobs=1)(
        delayed(process_single_row)(row) 
        for _, row in tqdm(clickme_data.iterrows(), total=len(clickme_data), desc="Processing clickmaps")
    )

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


def process_single_image(image_key, image_trials, image_shape, blur_size, blur_sigma, 
                        min_pixels, min_subjects, center_crop, metadata, blur_sigma_function,
                        kernel_type, duplicate_thresh, max_kernel_size, blur_kernel, 
                        create_clickmap_func, fast_duplicate_detection, device='cpu'):
    """Helper function to process a single image for parallel processing"""

    # Process metadata and create clickmaps
    if metadata is not None:
        if image_key not in metadata:
            # Use provided create_clickmap_func
            clickmaps = np.asarray([create_clickmap_func([trials], image_shape) for trials in image_trials])
            clickmaps = torch.from_numpy(clickmaps).float().unsqueeze(1).to(device)
            if kernel_type == "gaussian":
                clickmaps = convolve(clickmaps, blur_kernel)
            elif kernel_type == "circle":
                clickmaps = convolve(clickmaps, blur_kernel, double_conv=True)
        else:
            native_size = metadata[image_key]
            short_side = min(native_size)
            scale = short_side / min(image_shape)
            adj_blur_size = int(np.round(blur_size * scale))
            if not adj_blur_size % 2:
                adj_blur_size += 1
            adj_blur_size = min(adj_blur_size, max_kernel_size)
            adj_blur_sigma = blur_sigma_function(adj_blur_size)
            # Use provided create_clickmap_func
            clickmaps = np.asarray([create_clickmap_func([trials], native_size[::-1]) for trials in image_trials])
            clickmaps = torch.from_numpy(clickmaps).float().unsqueeze(1).to(device)
            if kernel_type == "gaussian":
                adj_blur_kernel = gaussian_kernel(adj_blur_size, adj_blur_sigma, device)
                clickmaps = convolve(clickmaps, adj_blur_kernel)
            elif kernel_type == "circle":
                adj_blur_kernel = circle_kernel(adj_blur_size, adj_blur_sigma, device)
                clickmaps = convolve(clickmaps, adj_blur_kernel, double_conv=True)
    else:
        # Use provided create_clickmap_func
        clickmaps = np.asarray([create_clickmap_func([trials], image_shape) for trials in image_trials])
        clickmaps = torch.from_numpy(clickmaps).float().unsqueeze(1).to(device)
        if kernel_type == "gaussian":
            clickmaps = convolve(clickmaps, blur_kernel)
        elif kernel_type == "circle":
            clickmaps = convolve(clickmaps, blur_kernel, double_conv=True)

    if center_crop:
        clickmaps = tvF.resize(clickmaps, min(image_shape))
        clickmaps = tvF.center_crop(clickmaps, center_crop)
    clickmaps = clickmaps.squeeze().numpy()

    # Filter processing
    if len(clickmaps.shape) == 2:  # Single map
        return None

    # Filter 1: Remove empties
    empty_check = (clickmaps > 0).sum((1, 2)) > min_pixels
    clickmaps = clickmaps[empty_check]
    if len(clickmaps) < min_subjects:
        return None

    # Filter 2: Remove duplicates using provided fast_duplicate_detection
    clickmaps_vec = clickmaps.reshape(len(clickmaps), -1)
    
    # Use the function passed as argument
    non_duplicate_indices = fast_duplicate_detection(clickmaps_vec, duplicate_thresh)
    clickmaps = clickmaps[non_duplicate_indices]
    
    # dm = cdist(clickmaps_vec, clickmaps_vec)
    # idx = np.tril_indices(len(dm), k=-1)
    # lt_dm = dm[idx]
    # if np.any(lt_dm < duplicate_thresh):
    #     remove = np.unique(np.where((dm + np.eye(len(dm)) == 0)))[0]
    #     rng = np.arange(len(dm))
    #     dup_idx = rng[~np.in1d(rng, remove)]
    #     clickmaps = clickmaps[dup_idx]

    if len(clickmaps) >= min_subjects:
        return (image_key, clickmaps)
    return None

def prepare_maps_batched_gpu(
        final_clickmaps,
        blur_size,
        blur_sigma,
        image_shape,
        min_pixels,
        min_subjects,
        center_crop,
        metadata=None,
        blur_sigma_function=None,
        kernel_type="circle",
        duplicate_thresh=0.01,
        max_kernel_size=51,
        device='cuda',
        batch_size=512,  # Reduced from 1024 to 512
        n_jobs=-1,
        timeout=600,  # Add timeout parameter (10 minutes default)
        verbose=True,  # Add verbose parameter to control detailed logging
        create_clickmap_func=None,
        fast_duplicate_detection=None):
    """
    Optimized version of prepare_maps that separates CPU and GPU work:
    1. Pre-processes clickmaps in parallel on CPU
    2. Processes batches of blurring on GPU
    3. Post-processes results in parallel on CPU
    
    Args:
        final_clickmaps (list): List of dictionaries mapping image keys to clickmap trials
        blur_size (int): Size of the blur kernel
        blur_sigma (float): Sigma value for the blur kernel
        image_shape (list/tuple): Shape of the images [height, width]
        min_pixels (int): Minimum number of pixels for a valid map
        min_subjects (int): Minimum number of subjects for a valid map
        center_crop (list/tuple): Size for center cropping
        metadata (dict, optional): Metadata dictionary. Defaults to None.
        blur_sigma_function (function, optional): Function to calculate blur sigma. Required.
        kernel_type (str, optional): Type of kernel to use. Defaults to "circle".
        duplicate_thresh (float, optional): Threshold for duplicate detection. Defaults to 0.01.
        max_kernel_size (int, optional): Maximum kernel size. Defaults to 51.
        device (str, optional): Device to use for GPU operations. Defaults to 'cuda'.
        batch_size (int, optional): Batch size for GPU processing. Defaults to 512.
        n_jobs (int, optional): Number of parallel jobs for CPU operations. Defaults to -1.
        timeout (int, optional): Timeout in seconds for parallel jobs. Defaults to 600.
        verbose (bool): Whether to show detailed progress logging
        create_clickmap_func (function): Function to create the initial clickmap
        fast_duplicate_detection (function): Function for duplicate detection
        
    Returns:
        tuple: (new_final_clickmaps, all_clickmaps, categories, keep_index)
    """
    import torch
    import torch.nn.functional as F
    from joblib import Parallel, delayed
    from scipy.spatial.distance import cdist
    import numpy as np
    from tqdm import tqdm
    import torchvision.transforms.functional as tvF
    
    assert blur_sigma_function is not None, "Blur sigma function not passed."
    # Check if functions were passed
    assert create_clickmap_func is not None, "create_clickmap function must be provided."
    assert fast_duplicate_detection is not None, "fast_duplicate_detection function must be provided."
    
    # Step 1: Create kernels on GPU
    if kernel_type == "gaussian":
        blur_kernel = gaussian_kernel(blur_size, blur_sigma, device)
    elif kernel_type == "circle":
        blur_kernel = circle_kernel(blur_size, blur_sigma, device)
    else:
        raise NotImplementedError(kernel_type)
    
    # We'll store all results here
    all_final_results = {
        'all_clickmaps': [],
        'categories': [],
        'keep_index': [],
        'new_final_clickmaps': {}
    }
    
    # FIX: More carefully merge dictionaries to avoid mixing maps from different images
    # We use a dict to track the source of each clickmap to ensure we're not mixing maps
    merged_clickmaps = {}
    image_sources = {}  # Track which dict each image came from
    map_counts_before = {}  # Track number of maps before merging
    map_counts_after = {}   # Track number of maps after merging

    for dict_idx, clickmap_dict in enumerate(final_clickmaps):
        # Count maps in this dictionary
        for image_key, maps in clickmap_dict.items():
            if image_key not in map_counts_before:
                map_counts_before[image_key] = 0
            map_counts_before[image_key] += len(maps)
        
        for image_key, maps in clickmap_dict.items():
            if image_key in merged_clickmaps:
                # If this image already exists, we need to make sure we're not 
                # accidentally mixing maps from different images
                print(f"Warning: Image {image_key} found in multiple dictionaries. Combining maps.")
                # Append maps while preserving the source tracking
                merged_clickmaps[image_key].extend(maps)
                if isinstance(image_sources[image_key], list):
                    image_sources[image_key].append(dict_idx)
                else:
                    image_sources[image_key] = [image_sources[image_key], dict_idx]
            else:
                # First occurrence of this image
                merged_clickmaps[image_key] = maps
                image_sources[image_key] = dict_idx

    # Count maps after merging
    for image_key, maps in merged_clickmaps.items():
        map_counts_after[image_key] = len(maps)

    # Log if we found any duplicate keys across dictionaries
    duplicate_keys = [k for k, v in image_sources.items() if isinstance(v, list)]
    if duplicate_keys:
        print(f"Found {len(duplicate_keys)} images with maps in multiple dictionaries. These have been properly combined.")
        
        # Extra verification in verbose mode
        if verbose:
            print("\nVerification of map combining:")
            for key in duplicate_keys:
                if map_counts_before[key] != map_counts_after[key]:
                    print(f"  ERROR: Map count mismatch for {key}: Before={map_counts_before[key]}, After={map_counts_after[key]}")
                else:
                    print(f"  OK: Successfully combined {map_counts_after[key]} maps for {key}")
            print("")
    
    # Step 2: Get all keys and prepare for batch processing
    all_keys = list(merged_clickmaps.keys())
    total_images = len(all_keys)
    
    # Calculate number of batches based on total unique images
    # Set more conservative batch sizes for stability
    # cpu_batch_size = min(batch_size, 5000)  # Cap at 5000 for stability 
    cpu_batch_size = batch_size
    num_cpu_batches = (total_images + cpu_batch_size - 1) // cpu_batch_size
    effective_n_jobs = min(n_jobs if n_jobs > 0 else os.cpu_count(), os.cpu_count(), 16)  # Cap at 16 workers

    print(f"Processing {total_images} unique images in {num_cpu_batches} CPU batches (GPU batch size: {batch_size})...")
    print(f"Using {effective_n_jobs} parallel jobs for CPU pre/post-processing.")
    
    # Process each batch of images
    processed_count = 0
    with tqdm(total=total_images, desc="Processing Image Batches") as pbar:
        for cpu_batch_idx in range(num_cpu_batches):
            batch_start = cpu_batch_idx * cpu_batch_size
            batch_end = min(batch_start + cpu_batch_size, total_images)
            
            # print(f"\n│  ├─ Processing CPU batch {cpu_batch_idx+1}/{num_cpu_batches} (images {batch_start}-{batch_end})...")
            
            # Get keys for this batch
            batch_keys = all_keys[batch_start:batch_end]
            
            # Step 3: Pre-process only this batch of clickmaps in parallel on CPU
            # print(f"│  │  ├─ Pre-processing clickmaps on CPU (parallel, n_jobs={effective_n_jobs})...")
            
            def preprocess_clickmap(image_key, image_trials, image_shape, metadata=None):
                """Helper function to pre-process a clickmap before GPU processing"""
                # Process metadata and create clickmaps (creates binary maps, no blurring)
                # Ensure image_shape is a tuple as required by create_clickmap_func
                image_shape_tuple = tuple(image_shape) if isinstance(image_shape, list) else image_shape
                
                if metadata is not None and image_key in metadata:
                    native_size = metadata[image_key]
                    # Use provided create_clickmap_func
                    clickmaps = np.asarray([create_clickmap_func([trials], native_size[::-1]) for trials in image_trials])
                    return {
                        'key': image_key,
                        'clickmaps': clickmaps,
                        'native_size': native_size if image_key in metadata else None
                    }
                else:
                    # Use provided create_clickmap_func
                    clickmaps = np.asarray([create_clickmap_func([trials], image_shape_tuple) for trials in image_trials])
                    return {
                        'key': image_key,
                        'clickmaps': clickmaps,
                        'native_size': None
                    }
            
            # Use parallel processing for pre-processing only this batch
            # Set a timeout for workers to avoid indefinite hanging
            preprocessed = Parallel(n_jobs=effective_n_jobs, timeout=timeout)(
                delayed(preprocess_clickmap)(
                    key, 
                    merged_clickmaps[key], 
                    image_shape, 
                    metadata
                ) for key in tqdm(batch_keys, desc="Pre-processing", total=len(batch_keys), leave=False)
            )
            
            # Only keep non-empty preprocessed data
            preprocessed = [p for p in preprocessed if p is not None and len(p['clickmaps']) > 0]

            # Step 4: Process GPU blurring
            # print(f"│  │  ├─ Processing blurring on GPU (batch_size={batch_size})...")
            
            # Process in smaller GPU sub-batches to prevent OOM errors
            gpu_batch_size = batch_size  # min(batch_size, 256)  # Cap at 256 to prevent OOM errors
            batch_results = []
            
            # Flatten the list of clickmaps for efficient GPU batching
            gpu_processing_list = []
            for item in preprocessed:
                # Each item in preprocessed has a list of clickmaps for one image
                # We need to process each clickmap individually on the GPU eventually
                gpu_processing_list.append(item)

            # Process GPU batches with a progress bar 
            total_gpu_batches = (len(gpu_processing_list) + gpu_batch_size - 1) // gpu_batch_size
            if verbose:
                print(f"Processing {len(gpu_processing_list)} images in {total_gpu_batches} GPU batches (size: {gpu_batch_size})...")
                
            with tqdm(total=total_gpu_batches, desc="GPU batches", leave=False) as gpu_batch_pbar:
                for gpu_batch_idx in range(0, len(gpu_processing_list), gpu_batch_size):

                    # Log current batch information 
                    batch_start = gpu_batch_idx
                    batch_end = min(gpu_batch_idx + gpu_batch_size, len(gpu_processing_list))
                    current_batch_size = batch_end - batch_start
                                                                
                    # Get smaller sub-batch to process
                    gpu_batch_items = gpu_processing_list[gpu_batch_idx : gpu_batch_idx + gpu_batch_size]
                    
                    # Skip empty batches
                    if not gpu_batch_items:
                        gpu_batch_pbar.update(1)
                        continue
                        
                    # Log tensor preparation step
                    if verbose:
                        print(f"  │  ├─ Preparing tensors for {len(gpu_batch_items)} images...")
                    
                    # Prepare tensors for this GPU batch
                    tensors_to_blur = []
                    metadata_for_batch = []
                    keys_for_batch = []
                    map_counts = [] # Track how many maps belong to each original image key

                    for item in gpu_batch_items:
                        key = item['key']
                        clickmaps_np = item['clickmaps']
                        native_size = item['native_size']
                        
                        # Convert numpy arrays to PyTorch tensors
                        # Important: Keep track of how many maps belong to this key
                        num_maps_for_key = len(clickmaps_np)
                        if num_maps_for_key > 0:
                            clickmaps_tensor = torch.from_numpy(clickmaps_np).float().unsqueeze(1).to(device)
                            tensors_to_blur.append(clickmaps_tensor)
                            # Repeat metadata for each map belonging to this key
                            metadata_for_batch.extend([(key, native_size)] * num_maps_for_key)
                            keys_for_batch.extend([key] * num_maps_for_key)
                            map_counts.append(num_maps_for_key)
                    
                    if not tensors_to_blur:
                        if verbose:
                            print(f"  │  ├─ No valid tensors to process, skipping batch")
                        gpu_batch_pbar.update(1)
                        continue

                    # Log batch tensor creation
                    if verbose:
                        print(f"  │  ├─ Concatenating {len(tensors_to_blur)} tensors with {sum(map_counts)} total maps...")
                        
                    # Concatenate tensors for efficient batch processing
                    batch_tensor = torch.cat(tensors_to_blur, dim=0)
                    
                    # Log tensor shape for debugging
                    if verbose:
                        print(f"  │  ├─ Batch tensor shape: {batch_tensor.shape}, processing blurring...")
                    
                    # Clear up memory
                    del tensors_to_blur
                    torch.cuda.empty_cache()
                    
                    # Apply blurring (needs to handle potential metadata variations within batch)
                    blurred_batch = torch.zeros_like(batch_tensor)
                    current_idx = 0
                    
                    # Apply blurring with a more memory-efficient approach
                    sub_batch_size = 100  # Process in small sub-batches for stability
                    if verbose and len(gpu_batch_items) > 1:
                        print(f"  │  ├─ Processing {len(gpu_batch_items)} image items in batches of {sub_batch_size}...")
                        
                    for item_idx, item in enumerate(tqdm(gpu_batch_items, desc="Blurring items", leave=False, disable=not verbose)):
                        # Apply blurring based on the specific item's metadata
                        key = item['key']
                        num_maps = len(item['clickmaps'])
                        native_size = item['native_size']
                        
                        if num_maps == 0:
                            continue
                            
                        item_tensor = batch_tensor[current_idx : current_idx + num_maps]
                        import pdb; pdb.set_trace()
                        try:
                            # Process with proper error handling
                            if native_size is not None:
                                short_side = min(native_size)
                                scale = short_side / min(image_shape)
                                adj_blur_size = int(np.round(blur_size * scale))
                                if not adj_blur_size % 2:
                                    adj_blur_size += 1
                                adj_blur_size = min(adj_blur_size, max_kernel_size)
                                adj_blur_sigma = blur_sigma_function(adj_blur_size)
                                
                                if kernel_type == "gaussian":
                                    adj_blur_kernel = gaussian_kernel(adj_blur_size, adj_blur_sigma, device)
                                    blurred_item = convolve(item_tensor, adj_blur_kernel)
                                elif kernel_type == "circle":
                                    adj_blur_kernel = circle_kernel(adj_blur_size, adj_blur_sigma, device)
                                    blurred_item = convolve(item_tensor, adj_blur_kernel, double_conv=True)
                                    
                                # Free memory for next iteration
                                if 'adj_blur_kernel' in locals(): 
                                    del adj_blur_kernel
                            else:
                                # Use the standard kernel
                                if kernel_type == "gaussian":
                                    blurred_item = convolve(item_tensor, blur_kernel)
                                elif kernel_type == "circle":
                                    blurred_item = convolve(item_tensor, blur_kernel, double_conv=True)
                            
                            blurred_batch[current_idx : current_idx + num_maps] = blurred_item
                            
                            # Free memory 
                            del blurred_item
                        except Exception as e:
                            if verbose:
                                print(f"  │  ├─ ERROR processing item {item_idx} (key: {key}): {e}")
                            # In case of error, just keep original
                            blurred_batch[current_idx : current_idx + num_maps] = item_tensor
                            
                        current_idx += num_maps
                        
                        # Periodically clear cache
                        if item_idx % 50 == 0:
                            torch.cuda.empty_cache()
                    
                    # Log center crop info if applicable
                    if center_crop and verbose:
                        print(f"  │  ├─ Applying center crop from {blurred_batch.shape[-2:]} to {center_crop}...")
                        
                    # Apply center crop if needed (applied to the whole batch)
                    if center_crop:
                        # Resize first if dimensions are different
                        if blurred_batch.shape[-2:] != image_shape:
                            blurred_batch = tvF.resize(blurred_batch, list(image_shape), antialias=True)
                        blurred_batch = tvF.center_crop(blurred_batch, list(center_crop))
                    
                    # Log conversion to numpy
                    if verbose:
                        print(f"  │  ├─ Converting to numpy and organizing results...")
                        
                    # Convert back to numpy and store results indexed by key
                    processed_maps_np = blurred_batch.squeeze(1).cpu().numpy()
                    
                    # Reconstruct the results grouped by image key
                    start_idx = 0
                    item_idx = 0
                    while start_idx < len(processed_maps_np):
                        key = keys_for_batch[start_idx]
                        num_maps = map_counts[item_idx]
                        end_idx = start_idx + num_maps
                        batch_results.append({
                            'key': key,
                            'clickmaps': processed_maps_np[start_idx:end_idx]
                        })
                        start_idx = end_idx
                        item_idx += 1
                    
                    # Log memory cleanup
                    if verbose:
                        print(f"  │  └─ Cleaning up memory...")
                        
                    # Free GPU memory
                    del batch_tensor, blurred_batch, item_tensor
                    if 'adj_blur_kernel' in locals(): del adj_blur_kernel
                    # check_gpu_memory_usage(threshold=0.5, force_cleanup=True)
                    
                    # Add a small delay to allow system to stabilize
                    import time
                    time.sleep(0.1)
                    
                    # Update GPU batch progress bar
                    gpu_batch_pbar.update(1)
                    

            # Add post-processing progress logging
            if verbose:
                print(f"Post-processing {len(batch_results)} results...")
                
            # Use parallel processing for post-processing with timeout
            post_results = Parallel(n_jobs=effective_n_jobs, timeout=timeout)(
                delayed(postprocess_clickmap)(
                    batch_results[i],
                    min_pixels,
                    min_subjects,
                    duplicate_thresh
                ) for i in tqdm(range(len(batch_results)), desc="Post-processing", disable=not verbose) 
            )
            
            # Step 6: Compile final results for this batch
            for result in post_results:
                if result is not None:
                    image_key, clickmaps = result
                    category = image_key.split("/")[0]
                    
                    all_final_results['all_clickmaps'].append(clickmaps)
                    all_final_results['categories'].append(category)
                    all_final_results['keep_index'].append(image_key)
                    all_final_results['new_final_clickmaps'][image_key] = merged_clickmaps[image_key]
            
            # Update progress bar
            pbar.update(len(batch_keys))
            processed_count += len(batch_keys)

            # Free memory
            del preprocessed, batch_results, post_results
            if 'gpu_processing_list' in locals(): del gpu_processing_list
            torch.cuda.empty_cache()
    
    # # Final cleanup before returning
    # check_gpu_memory_usage(threshold=0.0, force_cleanup=True)
    
    print(f"\nFinished processing {processed_count} images.")
    # Return combined results
    return (
        all_final_results['new_final_clickmaps'], 
        all_final_results['all_clickmaps'], 
        all_final_results['categories'], 
        all_final_results['keep_index']
    )

# Custom wrapper for prepare_maps_batched_gpu with progress display
def prepare_maps_with_gpu_batching(final_clickmaps, **kwargs):
    """
    Wrapper for prepare_maps_batched_gpu that displays progress and follows
    the same signature as prepare_maps_with_progress for easy swapping.
    
    This version optimizes processing by:
    1. Pre-processing clickmaps in parallel on CPU
    2. Processing batches of blurring operations on GPU
    3. Post-processing results in parallel on CPU

    Args:
        final_clickmaps (list): List of dictionaries mapping image keys to clickmap trials
        **kwargs: Additional arguments to pass to prepare_maps_batched_gpu
    
    Returns:
        tuple: (new_final_clickmaps, all_clickmaps, categories, keep_index)
    """
    batch_size = kwargs.pop('batch_size', 512)  # Default batch size of 512
    verbose = kwargs.pop('verbose', True)  # Add verbose parameter, default to True
    
    # Display more information if verbose
    if verbose:
        print(f"│  ├─ Processing with GPU-optimized batching (batch_size={batch_size})...")
    
    # Pass the required functions from kwargs
    create_clickmap_func = kwargs.get('create_clickmap_func')
    fast_duplicate_detection = kwargs.get('fast_duplicate_detection')
            
    return prepare_maps_batched_gpu(
        final_clickmaps=final_clickmaps, 
        batch_size=batch_size,
        verbose=verbose,  # Pass verbose parameter
        create_clickmap_func=create_clickmap_func,
        fast_duplicate_detection=fast_duplicate_detection,
        **{k: v for k, v in kwargs.items() if k not in ('create_clickmap_func', 'fast_duplicate_detection', 'batch_size', 'verbose')}
    )

# GPU-accelerated correlation metrics
def compute_AUC_gpu(pred, target, device='cuda'):
    """
    GPU-accelerated implementation of AUC score computation.
    
    Args:
        pred (np.ndarray): Predicted heatmap
        target (np.ndarray): Target heatmap
        device (str): Device to run computation on ('cuda' or 'cpu')
        
    Returns:
        float: AUC score
    """
    import torch
    from sklearn import metrics
    
    # Flatten arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Convert to PyTorch tensors
    pred_tensor = torch.tensor(pred_flat, device=device)
    target_tensor = torch.tensor(target_flat, device=device)
    
    # Create a binary mask of non-zero target pixels
    mask = target_tensor > 0
    
    # If no positive pixels, return 0.5 (random chance)
    if not torch.any(mask):
        return 0.5
    
    # Get masked predictions and binary ground truth
    masked_pred = pred_tensor[mask].cpu().numpy()
    masked_target = torch.ones_like(target_tensor[mask]).cpu().numpy()
    
    # Get an equal number of negative samples
    neg_mask = ~mask
    if torch.sum(neg_mask) > 0:
        # Select same number of negative samples as positive ones
        num_pos = torch.sum(mask).item()
        num_neg = min(num_pos, torch.sum(neg_mask).item())
        
        # Get indices of negative samples
        neg_indices = torch.nonzero(neg_mask).squeeze()
        if neg_indices.numel() > 0:
            if neg_indices.numel() > num_neg:
                # Random sample negative indices if we have more than we need
                perm = torch.randperm(neg_indices.numel(), device=device)
                neg_indices = neg_indices[perm[:num_neg]]
            
            # Get predictions for negative samples and set target to 0
            neg_pred = pred_tensor[neg_indices].cpu().numpy()
            neg_target = torch.zeros(neg_indices.numel()).numpy()
            
            # Combine positive and negative samples
            masked_pred = np.concatenate([masked_pred, neg_pred])
            masked_target = np.concatenate([masked_target, neg_target])
    
    # Compute AUC score
    try:
        return metrics.roc_auc_score(masked_target, masked_pred)
    except ValueError:
        # In case of errors, fallback to 0.5
        return 0.5

def compute_spearman_correlation_gpu(pred, target, device='cuda'):
    """
    GPU-accelerated implementation of Spearman correlation computation.
    
    Args:
        pred (np.ndarray): Predicted heatmap
        target (np.ndarray): Target heatmap
        device (str): Device to run computation on ('cuda' or 'cpu')
        
    Returns:
        float: Spearman correlation coefficient
    """
    import torch
    
    # Flatten arrays
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Convert to PyTorch tensors
    pred_tensor = torch.tensor(pred_flat, device=device)
    target_tensor = torch.tensor(target_flat, device=device)
    
    # Compute ranks
    pred_rank = torch.argsort(torch.argsort(pred_tensor)).float()
    target_rank = torch.argsort(torch.argsort(target_tensor)).float()
    
    # Compute mean ranks
    pred_mean = torch.mean(pred_rank)
    target_mean = torch.mean(target_rank)
    
    # Compute numerator and denominator
    numerator = torch.sum((pred_rank - pred_mean) * (target_rank - target_mean))
    denominator = torch.sqrt(torch.sum((pred_rank - pred_mean)**2) * torch.sum((target_rank - target_mean)**2))
    
    # Compute correlation
    if denominator > 0:
        correlation = numerator / denominator
        return correlation.cpu().item()
    else:
        return 0.0

def compute_crossentropy_gpu(pred, target, eps=1e-10, device='cuda'):
    """
    GPU-accelerated implementation of cross-entropy computation.
    
    Args:
        pred (np.ndarray): Predicted heatmap
        target (np.ndarray): Target heatmap
        eps (float): Small value to avoid numerical issues
        device (str): Device to run computation on ('cuda' or 'cpu')
        
    Returns:
        float: Cross-entropy loss
    """
    import torch
    import torch.nn.functional as F
    
    # Convert to PyTorch tensors
    pred_tensor = torch.tensor(pred, device=device).float()
    target_tensor = torch.tensor(target, device=device).float()
    
    # Normalize target to sum to 1
    target_sum = torch.sum(target_tensor)
    if target_sum > 0:
        target_tensor = target_tensor / target_sum
    
    # Normalize prediction to sum to 1
    pred_sum = torch.sum(pred_tensor)
    if pred_sum > 0:
        pred_tensor = pred_tensor / pred_sum
    
    # Add small epsilon to avoid log(0)
    pred_tensor = torch.clamp(pred_tensor, min=eps)
    
    # Compute cross-entropy loss
    loss = -torch.sum(target_tensor * torch.log(pred_tensor))
    
    return loss.cpu().item()

# Function to process a batch of correlation computations on GPU
def batch_compute_correlations_gpu(test_maps, reference_maps, metric='auc', device='cuda'):
    """
    Process a batch of correlation computations on GPU for improved performance.
    For Spearman correlation, uses scipy's implementation instead of the GPU version
    
    Args:
        test_maps (list): List of test maps
        reference_maps (list): List of reference maps
        metric (str): Metric to use ('auc', 'spearman', 'crossentropy')
        device (str): Device to run computation on ('cuda' or 'cpu')
        
    Returns:
        list: Correlation scores for each pair of maps
    """
    assert len(test_maps) == len(reference_maps), "Number of test and reference maps must match"
    from scipy.stats import spearmanr
    
    results = []
    for test_map, reference_map in zip(test_maps, reference_maps):
        # Normalize maps
        test_map = (test_map - test_map.min()) / (test_map.max() - test_map.min() + 1e-10)
        reference_map = (reference_map - reference_map.min()) / (reference_map.max() - reference_map.min() + 1e-10)
        
        # Compute correlation using appropriate function
        if metric.lower() == 'auc':
            score = compute_AUC_gpu(test_map, reference_map, device)
        elif metric.lower() == 'spearman':
            # Use scipy's spearman implementation instead of GPU version
            score, _ = spearmanr(test_map.flatten(), reference_map.flatten())
        elif metric.lower() == 'crossentropy':
            score = compute_crossentropy_gpu(test_map, reference_map, device)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        results.append(score)
    
    return results

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
    from joblib import Parallel, delayed
    import os
    import numpy as np
    from tqdm import tqdm
    
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
    import os
    import numpy as np
    from tqdm import tqdm
    import h5py
    import time
    
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
                hmp = all_clickmaps[i]
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
        kernel = circle_kernel(kernel_size, kernel_sigma, 'cuda')
        
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
        return_before_blur=False
        ):
    """
    Simplified function to blur clickmaps on GPU in batches
    """
    
    # Extract basic parameters
    blur_size = config["blur_size"]
    blur_sigma = config.get("blur_sigma", blur_size)
    image_shape = config["image_shape"]
    min_subjects = config["min_subjects"]
    min_clicks = config["min_clicks"]
    max_kernel_size = config.get("max_kernel_size", 51)
    blur_sigma_function = config.get("blur_sigma_function", lambda x: blur_sigma)
    
    # Get GPU batch size for processing
    gpu_batch_size = config.get("gpu_batch_size", 4096)
    
    print(f"Processing {len(clickmaps)} unique images with GPU (batch size: {gpu_batch_size})...")
    
    # Step 1: Prepare binary maps and average them
    print("Pre-processing clickmaps on CPU...")
    
    # Prepare data structures
    all_clickmaps = []
    keep_index = []
    categories = []
    final_clickmaps = {}
    clickmap_bins = {}
    click_counts = {}  # Track click counts for each image
    
    # Preprocess all clickmaps first to binary maps
    for key, trials in clickmaps.items():
        if len(trials) < min_subjects:
            continue
        
        # Get max count then do thresholds from that
        lens = [len(x) for x in trials]
        max_count = max(lens)
        # min_count = max(int(min(lens) * .1), 1)
        # bins = np.linspace(min_count, max_count + 1, thresholds).astype(int)
        mean_lens = int(np.mean(lens))
        below_mean = np.linspace(mean_lens * .1, mean_lens + 1, thresholds // 2).astype(int)
        above_mean = np.linspace(mean_lens, max_count + 1, thresholds // 2).astype(int)
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
            if len(binary_maps) >= min_subjects:
                if average_maps:
                    bin_clickmaps.append(np.array(binary_maps).mean(0, keepdims=True))
                else:
                    bin_clickmaps.append(np.array(binary_maps))
        
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
        if return_before_blur:
            all_clickmaps.append(np.stack(bin_clickmaps, axis=0))
        else:
            all_clickmaps.append(np.concatenate(bin_clickmaps, axis=0))
    
    if not all_clickmaps:
        print("No valid clickmaps to process")
        return {}, [], [], [], {}
    
    if return_before_blur:
        return final_clickmaps, all_clickmaps, categories, keep_index, click_counts, clickmap_bins
    
    # Step 2: Prepare for batch blurring on GPU with adaptive kernel sizing
    total_maps = len(all_clickmaps)
    print(f"Preparing to blur {total_maps} image clickmaps using GPU with adaptive kernel sizing...")
    
    # Convert all maps to tensors
    all_tensors = [torch.from_numpy(maps).float() for maps in all_clickmaps]
    
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
        kernel_groups[kernel_key].append(idx)
    
    print(f"Processing {len(kernel_groups)} different kernel sizes...")
    
    # Process in batches based on the GPU batch size
    try:
        # Just check if we can access the first tensor
        _ = all_tensors[0].shape
        num_batches = (total_maps + gpu_batch_size - 1) // gpu_batch_size
    except Exception as e:
        print(f"Error accessing tensors: {e}")
        num_batches = total_maps
        gpu_batch_size = 1
    
    # Process each kernel group separately
    for (kernel_size, kernel_sigma), image_indices in tqdm(kernel_groups.items(), desc="Processing kernel groups"):
        print(f"Processing {len(image_indices)} images with kernel size {kernel_size}, sigma {kernel_sigma}")
        
        # Create kernel for this group
        kernel = circle_kernel(kernel_size, kernel_sigma, 'cuda')
        
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
            # Group maps by size to handle different dimensions
            size_groups = {}
            for i, map_tensor in enumerate(flattened_maps):
                size_key = tuple(map_tensor.shape)
                if size_key not in size_groups:
                    size_groups[size_key] = []
                size_groups[size_key].append((i, map_tensor))
            
            # Process each size group separately
            processed_maps = [None] * len(flattened_maps)  # Pre-allocate to maintain order
            for size, tensors in size_groups.items():
                indices = [t[0] for t in tensors]
                size_tensors = [t[1] for t in tensors]
                
                # Process this size group in sub-batches
                for sub_idx in range(0, len(size_tensors), sub_batch_size):
                    sub_end = min(sub_idx + sub_batch_size, len(size_tensors))
                    sub_batch = size_tensors[sub_idx:sub_end]
                    sub_indices = indices[sub_idx:sub_end]
                    
                    # Concatenate and process tensors of the same size
                    sub_batch_tensor = torch.cat(sub_batch, dim=0).unsqueeze(1).to('cuda')
                    blurred_tensor = convolve(sub_batch_tensor, kernel, double_conv=True)
                    
                    # Store processed maps at their original positions
                    for j, idx in enumerate(sub_indices):
                        processed_maps[idx] = blurred_tensor[j].squeeze(0).cpu()
                    
                    # Clean up GPU memory
                    del sub_batch_tensor, blurred_tensor
                    torch.cuda.empty_cache()
            
            # Reconstruct the original tensor structures
            for i, tensor_shape in enumerate(original_shapes):
                batch_img_idx = start_idx + i
                
                # Find all maps for this image
                img_map_indices = [j for j, idx in enumerate(map_indices) if idx == batch_img_idx]
                img_maps = [processed_maps[j] for j in img_map_indices]
                
                # For variable-sized maps, we can't use stack - store as a list
                if len(set(m.shape for m in img_maps)) > 1:
                    all_clickmaps[batch_img_idx] = img_maps
                else:
                    # Stack them back to original shape if all same size
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
    
    return final_clickmaps, all_clickmaps, categories, keep_index, click_counts, clickmap_bins


def blur_maps_for_cf(all_clickmaps, blur_size, blur_sigma, gpu_batch_size):
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

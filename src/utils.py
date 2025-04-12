import re
import os
import sys
import torch
import yaml
import numpy as np
import pandas as pd
from torch.nn import functional as F
from scipy.stats import spearmanr
from tqdm import tqdm
from torchvision.transforms import functional as tvF
from scipy.spatial.distance import cdist
from glob import glob
from train_subject_classifier import RNN
from accelerate import Accelerator
from joblib import Parallel, delayed


import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def normalize_heatmap(heatmap):
    """Normalize heatmap to sum to 1 while preserving zeros"""
    total = heatmap.sum()
    return heatmap / total if total != 0 else heatmap


def get_cost_matrix(shape, device='cpu'):
    """
    Compute cost matrix for grid coordinates.
    Vectorized implementation for speed.
    """
    h, w = shape
    xp = cp if (device == 'gpu' and HAS_CUPY) else np
    
    y, x = xp.meshgrid(xp.arange(h), xp.arange(w), indexing='ij')
    points = xp.stack([y.flatten(), x.flatten()], axis=1)
    
    # Compute pairwise distances using broadcasting
    diff = points[:, None, :] - points[None, :, :]
    costs = xp.sqrt(xp.sum(diff ** 2, axis=2))
    
    return costs


def sinkhorn_knopp(cost_matrix, a, b, epsilon=1e-1, max_iters=100, tol=1e-6):
    """
    Fast Sinkhorn-Knopp algorithm for approximate optimal transport.
    
    Parameters:
    -----------
    cost_matrix : array
        Matrix of transport costs
    a, b : array
        Source and target distributions
    epsilon : float
        Regularization parameter (smaller = more accurate but slower)
    """
    xp = cp if isinstance(cost_matrix, cp.ndarray) else np
    
    # Initialization
    K = xp.exp(-cost_matrix / epsilon)
    
    # Initialize scaling vectors
    u = xp.ones_like(a)
    v = xp.ones_like(b)
    
    # Sinkhorn iterations
    for _ in range(max_iters):
        u_old = u.copy()
        
        # u update
        u = a / (K @ v)
        # v update
        v = b / (K.T @ u)
        
        # Check convergence
        err = xp.max(xp.abs(u - u_old))
        if err < tol:
            break
    
    # Compute transport plan
    P = xp.diag(u) @ K @ xp.diag(v)
    
    # Compute total cost
    cost = xp.sum(P * cost_matrix)
    
    return cost


def fast_normalized_emd(heatmap1, heatmap2, epsilon=1e-1):
    """
    Compute normalized Earth Mover's Distance between two heatmaps.
    Uses Sinkhorn algorithm for speed and GPU if available.
    
    Parameters:
    -----------
    heatmap1, heatmap2 : np.ndarray
        Input heatmaps
    epsilon : float
        Regularization parameter for Sinkhorn
        
    Returns:
    --------
    float
        Normalized EMD value
    """
    if heatmap1.shape != heatmap2.shape:
        raise ValueError("Heatmaps must have same shape")
    
    # Use GPU if available
    if HAS_CUPY:
        heatmap1 = cp.array(heatmap1)
        heatmap2 = cp.array(heatmap2)
        device = 'gpu'
    else:
        device = 'cpu'
    
    # Normalize heatmaps
    a = normalize_heatmap(heatmap1).flatten()
    b = normalize_heatmap(heatmap2).flatten()
    
    # Get or compute cost matrix
    cost_matrix = get_cost_matrix(heatmap1.shape, device)
    
    # Compute EMD using Sinkhorn
    distance = sinkhorn_knopp(cost_matrix, a, b, epsilon=epsilon)
    
    # Move back to CPU if needed
    if HAS_CUPY:
        distance = cp.asnumpy(distance)
    
    return distance


def benchmark_speed(shape=(64, 64), n_iterations=100):
    """Benchmark the speed of the EMD computation"""
    import time
    
    # Generate random heatmaps
    heatmap1 = np.random.rand(*shape)
    heatmap2 = np.random.rand(*shape)
    
    start = time.time()
    for _ in range(n_iterations):
        _ = fast_normalized_emd(heatmap1, heatmap2)
    end = time.time()
    
    avg_time = (end - start) / n_iterations
    print(f"Average time per computation: {avg_time*1000:.2f}ms")
    return avg_time


def compute_similarity(heatmap1, heatmap2, method='spatial_correlation', **kwargs):
    """
    Wrapper function to compute heatmap similarity using different methods.
    
    Parameters:
    -----------
    heatmap1, heatmap2 : np.ndarray
        Input heatmaps
    method : str
        Similarity method to use
    **kwargs : dict
        Additional parameters for specific methods
    
    Returns:
    --------
    float
        Similarity score
    """
    methods = {
        'spatial_correlation': spatial_correlation_similarity,
        'spearman': lambda x, y: stats.spearmanr(x.flatten(), y.flatten())[0],
        'pearson': lambda x, y: stats.pearsonr(x.flatten(), y.flatten())[0]
    }
    
    if method not in methods:
        raise ValueError(f"Method {method} not supported")
    
    return methods[method](heatmap1, heatmap2, **kwargs)


def load_masks(mask_dir, wc="*.pth"):
    files = glob(os.path.join(mask_dir, wc))
    assert len(files), "No masks found in {}".format(mask_dir)
    masks = {}
    for f in files:
        loaded = torch.load(f)  # Akash added image/mask/category
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
        image_file_name = os.path.sep.join(image_path.split(os.path.sep)[-2:])
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

    import pdb; pdb.set_trace()

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
        if file_inclusion_filter == "CO3D_ClickmeV2" or file_inclusion_filter == "CO3D_ClickMe2":
            image_files = glob(os.path.join(image_path, "**", "*.png"))
            if not np.any([image_file_name in x for x in image_files]):
                return None
        elif file_inclusion_filter and file_inclusion_filter not in image_file_name:
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
    results = Parallel(n_jobs=n_jobs)(
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


def process_single_image(image_key, image_trials, image_shape, blur_size, blur_sigma, 
                        min_pixels, min_subjects, center_crop, metadata, blur_sigma_function,
                        kernel_type, duplicate_thresh, max_kernel_size, blur_kernel, device='cpu'):
    """Helper function to process a single image for parallel processing"""

    # Process metadata and create clickmaps
    if metadata is not None:
        if image_key not in metadata:
            clickmaps = np.asarray([create_clickmap([trials], image_shape) for trials in image_trials])
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
            clickmaps = np.asarray([create_clickmap([trials], native_size[::-1]) for trials in image_trials])
            clickmaps = torch.from_numpy(clickmaps).float().unsqueeze(1).to(device)
            if kernel_type == "gaussian":
                adj_blur_kernel = gaussian_kernel(adj_blur_size, adj_blur_sigma, device)
                clickmaps = convolve(clickmaps, adj_blur_kernel)
            elif kernel_type == "circle":
                adj_blur_kernel = circle_kernel(adj_blur_size, adj_blur_sigma, device)
                clickmaps = convolve(clickmaps, adj_blur_kernel, double_conv=True)
    else:
        clickmaps = np.asarray([create_clickmap([trials], image_shape) for trials in image_trials])
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

    # Filter 2: Remove duplicates
    clickmaps_vec = clickmaps.reshape(len(clickmaps), -1)
    dm = cdist(clickmaps_vec, clickmaps_vec)
    idx = np.tril_indices(len(dm), k=-1)
    lt_dm = dm[idx]
    if np.any(lt_dm < duplicate_thresh):
        remove = np.unique(np.where((dm + np.eye(len(dm)) == 0)))
        rng = np.arange(len(dm))
        dup_idx = rng[~np.in1d(rng, remove)]
        clickmaps = clickmaps[dup_idx]

    if len(clickmaps) >= min_subjects:
        return (image_key, clickmaps)
    return None

def prepare_maps_parallel(
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
        n_jobs=-1):
    """Parallelized version of prepare_maps using joblib"""
    
    assert blur_sigma_function is not None, "Blur sigma function not passed."

    if kernel_type == "gaussian":
        blur_kernel = gaussian_kernel(blur_size, blur_sigma, device)
    elif kernel_type == "circle":
        blur_kernel = circle_kernel(blur_size, blur_sigma, device)
    else:
        raise NotImplementedError(kernel_type)

    # Process images in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_single_image)(
            ikeys, 
            imaps,
            image_shape,
            blur_size,
            blur_sigma,
            min_pixels,
            min_subjects,
            center_crop,
            metadata,
            blur_sigma_function,
            kernel_type,
            duplicate_thresh,
            max_kernel_size,
            blur_kernel,
            device
        ) for ikeys, imaps in tqdm(final_clickmaps.items(), total=len(final_clickmaps), desc="Processing images")
    )

    # Process results
    all_clickmaps = []
    categories = []
    keep_index = []
    new_final_clickmaps = {}

    for result in results:
        if result is not None:
            image_key, clickmaps = result
            category = image_key.split("/")[0]
            
            all_clickmaps.append(clickmaps)
            categories.append(category)
            keep_index.append(image_key)
            new_final_clickmaps[image_key] = final_clickmaps[image_key]

    return new_final_clickmaps, all_clickmaps, categories, keep_index


# Custom wrapper for prepare_maps_parallel to add progress bar
def prepare_maps_with_progress(final_clickmaps, **kwargs):
    """
    Create a wrapper that shows progress for prepare_maps_parallel.
    
    Since joblib's parallel processing is difficult to track directly,
    we'll use a simple spinner animation to show that processing is happening.
    """
    total_images = len(final_clickmaps[0])
    
    # Display information about the processing
    print(f"│  ├─ Processing {total_images} images...")
    
    # Simple spinner animation characters
    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    
    # Use a context manager to handle terminal output and cleanup
    import sys
    import time
    import threading
    from contextlib import contextmanager
    
    @contextmanager
    def spinner_animation():
        # Create a flag to control the spinner animation
        stop_spinner = threading.Event()
        
        # Function to animate the spinner
        def spin():
            i = 0
            while not stop_spinner.is_set():
                # Print the spinner character and processing message
                sys.stdout.write(f"\r│  ├─ {spinner[i]} Processing images... ")
                sys.stdout.flush()
                time.sleep(0.1)
                i = (i + 1) % len(spinner)
            
            # Clear the line when done
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()
        
        # Start the spinner in a separate thread
        spinner_thread = threading.Thread(target=spin)
        spinner_thread.daemon = True
        spinner_thread.start()
        
        try:
            # Return control to the caller
            yield
        finally:
            # Stop the spinner when the context exits
            stop_spinner.set()
            spinner_thread.join()
    
    # Use the spinner animation while processing
    with spinner_animation():
        # Call the original function
        for chunk in final_clickmaps:
            result = prepare_maps_parallel(final_clickmaps=chunk, **kwargs)
    
    # Print completion message
    print(f"│  ├─ ✓ Processed {total_images} images")
    
    return result

def compute_average_map(trial_indices, clickmaps, resample=False):
    """
    Compute the average map from selected trials.

    Args:
        trial_indices (list of int): Indices of the trials to be averaged.
        clickmaps (np.ndarray): 3D array of clickmaps.
        resample (bool): If True, resample trials with replacement. Default is False.

    Returns:
        np.ndarray: The average clickmap.
    """
    if resample:
        trial_indices = np.random.choice(trial_indices, size=len(trial_indices), replace=True)
    return clickmaps[trial_indices].mean(0)


def compute_spearman_correlation(map1, map2):
    """
    Compute the Spearman correlation between two maps.

    Args:
        map1 (np.ndarray): The first map.
        map2 (np.ndarray): The second map.

    Returns:
        float: The Spearman correlation coefficient, or NaN if computation is not possible.
    """
    filtered_map1 = map1.flatten()
    filtered_map2 = map2.flatten()

    if filtered_map1.size > 1 and filtered_map2.size > 1:
        correlation, _ = spearmanr(filtered_map1, filtered_map2)
        return correlation
    else:
        return float('nan')


def fast_ious(v1, v2):
    """
    Compute the IoU between two images.

    Args:
        image_1 (np.ndarray): The first image.
        image_2 (np.ndarray): The second image.

    Returns:
        float: The IoU between the two images.
    """
    # Compute intersection and union
    intersection = np.logical_and(v1, v2).sum()
    union = np.logical_or(v1, v2).sum()

    # Compute IoU
    iou = intersection / union if union != 0 else 0.0

    return iou
    

def gaussian_kernel(size, sigma, device='cpu'):
    """
    Create a Gaussian kernel.

    Args:
        size (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A 2D Gaussian kernel with added batch and channel dimensions.
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = torch.meshgrid(x_range, y_range, indexing='ij')
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
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


def integrate_surface(iou_scores, x, z, average_areas=True, normalize=False):
    # Integrate along x axis (classifier thresholds)
    if len(z) == 1:
        return iou_scores.mean()

    int_x = np.trapz(iou_scores, x, axis=1)
    
    if average_areas:
        return int_x.mean()

    # Integrate along z axis (label thresholds)
    int_xz = np.trapz(int_x, z)

    if normalize:
        x_range = x[-1] - x[0]
        z_range = z[-1] - z[0]
        total_area = x_range * z_range
        int_xz /= total_area

    return int_xz


def compute_RSA(map1, map2):
    """
    Compute the RSA between two maps.

    Returns:    
        float: The RSA between the two maps.
    """
    import pdb; pdb.set_trace()
    return np.corrcoef(map1, map2)[0, 1]


def compute_AUC(
        pred_map,
        target_map,
        prediction_threshs=21,
        target_threshold_min=0.25,
        target_threshold_max=0.75,
        target_threshs=9):
    """
    We will compute IOU between pred and target over multiple threshodls of the target map.

    Args:
        map1 (np.ndarray): The first map.
        map2 (np.ndarray): The second map.  

    Returns:
        float: The AUC between the two maps.
    """
    # Make sure both maps are probability distributions
    # if normalize:
    #     map1 = map1 / map1.sum()
    #     map2 = map2 / map2.sum()
    inner_thresholds = np.linspace(0, 1, prediction_threshs)
    target_thresholds = np.linspace(target_threshold_min, target_threshold_max, target_threshs)
    # thresholds = [0.25, 0.5, 0.75, 1]
    thresh_ious = []
    for outer_t in target_thresholds:
        thresh_target_map = (target_map >= outer_t).astype(int).ravel()
        ious = []
        for t in inner_thresholds:
            thresh_pred_map = (pred_map >= t).astype(int).ravel()
            iou = fast_ious(thresh_target_map, thresh_pred_map)
            ious.append(iou)
        thresh_ious.append(np.asarray(ious))
    thresh_ious = np.stack(thresh_ious, 0)
    return integrate_surface(thresh_ious, inner_thresholds, target_thresholds, normalize=True)


def compute_crossentropy(map1, map2):
    """
    Compute the cross-entropy between two maps.

    Args:
        map1 (np.ndarray): The first map.
        map2 (np.ndarray): The second map.  

    Returns:
        float: The cross-entropy between the two maps.
    """
    map1 = torch.from_numpy(map1).float().ravel()
    map2 = torch.from_numpy(map2).float().ravel()
    return F.cross_entropy(map1, map2).numpy()


def create_clickmap(point_lists, image_shape, exponential_decay=False, tau=0.5):
    """
    Create a clickmap from click points.

    Args:
        click_points (list of tuples): List of (x, y) coordinates where clicks occurred.
        image_shape (tuple): Shape of the image (height, width).
        blur_kernel (torch.Tensor, optional): Gaussian kernel for blurring. Default is None.
        tau (float, optional): Decay rate for exponential decay. Default is 0.5 but this needs to be tuned.

    Returns:
        np.ndarray: A 2D array representing the clickmap, blurred if kernel provided.
    """
    heatmap = np.zeros(image_shape, dtype=np.uint8)
    for click_points in point_lists:
        if exponential_decay:
            for idx, point in enumerate(click_points):

                if 0 <= point[1] < image_shape[0] and 0 <= point[0] < image_shape[1]:
                    heatmap[point[1], point[0]] += np.exp(-idx / tau)
        else:
            for point in click_points:
                if 0 <= point[1] < image_shape[0] and 0 <= point[0] < image_shape[1]:
                    heatmap[point[1], point[0]] += 1
    return heatmap


def alt_gaussian_kernel(size=10, sigma=10):
    """
    Generates a 2D Gaussian kernel.

    Parameters
    ----------
    size : int, optional
        Kernel size, by default 10
    sigma : int, optional
        Kernel sigma, by default 10

    Returns
    -------
    kernel : torch.Tensor
        A Gaussian kernel.
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = torch.meshgrid(x_range, y_range, indexing='ij')
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return kernel

def alt_gaussian_blur(heatmap, kernel):
    """
    Blurs a heatmap with a Gaussian kernel.

    Parameters
    ----------
    heatmap : torch.Tensor
        The heatmap to blur.
    kernel : torch.Tensor
        The Gaussian kernel.

    Returns
    -------
    blurred_heatmap : torch.Tensor
        The blurred heatmap.
    """
    # Ensure heatmap and kernel have the correct dimensions
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = torch.nn.functional.conv2d(heatmap, kernel, padding='same')

    return blurred_heatmap[0]

def save_clickmaps_parallel(all_clickmaps, final_keep_index, output_dir, experiment_name, image_path, n_jobs=-1):
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
    
    def save_single_clickmap(idx, img_name):
        """Helper function to save a single clickmap"""
        if not os.path.exists(os.path.join(image_path, img_name)):
            return 0
            
        hmp = all_clickmaps[idx]
        # Save to disk
        np.save(
            os.path.join(save_dir, f"{img_name.replace('/', '_')}.npy"), 
            hmp
        )
        return 1
    
    # Use tqdm to show progress
    with tqdm(total=len(final_keep_index), desc="Saving files in parallel", 
             colour="cyan") as save_pbar:
        
        # Process in smaller batches to update progress bar more frequently
        batch_size = max(1, min(100, len(final_keep_index) // 10))
        saved_count = 0
        
        for i in range(0, len(final_keep_index), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(final_keep_index))))
            batch_img_names = [final_keep_index[j] for j in batch_indices]
            
            # Save batch in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(save_single_clickmap)(j, img_name) 
                for j, img_name in zip(batch_indices, batch_img_names)
            )
            
            # Update progress bar
            batch_saved = sum(results)
            saved_count += batch_saved
            save_pbar.update(len(batch_indices))
    
    return saved_count

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
        batch_size=1024,
        n_jobs=-1):
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
        batch_size (int, optional): Batch size for GPU processing. Defaults to 1024.
        n_jobs (int, optional): Number of parallel jobs for CPU operations. Defaults to -1.
        
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
    
    # Step 1: Create kernels on GPU
    if kernel_type == "gaussian":
        blur_kernel = gaussian_kernel(blur_size, blur_sigma, device)
    elif kernel_type == "circle":
        blur_kernel = circle_kernel(blur_size, blur_sigma, device)
    else:
        raise NotImplementedError(kernel_type)
    
    # Process each chunk in the final_clickmaps list
    results_for_all_chunks = []
    
    # Process each chunk in the list
    for chunk_idx, clickmap_chunk in enumerate(final_clickmaps):
        print(f"│  ├─ Processing chunk {chunk_idx+1}/{len(final_clickmaps)}...")
        
        # Step 2: Prepare data for pre-processing
        all_keys = list(clickmap_chunk.keys())
        all_maps = [clickmap_chunk[k] for k in all_keys]
        
        # Step 3: Pre-process clickmaps in parallel on CPU (prepare points, no blurring)
        print(f"│  │  ├─ Pre-processing clickmaps on CPU (parallel, n_jobs={n_jobs})...")
        
        def preprocess_clickmap(image_key, image_trials, image_shape, metadata=None):
            """Helper function to pre-process a clickmap before GPU processing"""
            # Process metadata and create clickmaps (creates binary maps, no blurring)
            if metadata is not None and image_key in metadata:
                native_size = metadata[image_key]
                clickmaps = np.asarray([create_clickmap([trials], native_size[::-1]) for trials in image_trials])
                return {
                    'key': image_key,
                    'clickmaps': clickmaps,
                    'native_size': native_size if image_key in metadata else None
                }
            else:
                clickmaps = np.asarray([create_clickmap([trials], image_shape) for trials in image_trials])
                return {
                    'key': image_key,
                    'clickmaps': clickmaps,
                    'native_size': None
                }
        
        # Use parallel processing for pre-processing
        preprocessed = Parallel(n_jobs=n_jobs)(
            delayed(preprocess_clickmap)(
                all_keys[i], 
                all_maps[i], 
                image_shape, 
                metadata
            ) for i in tqdm(range(len(all_keys)), desc="Pre-processing clickmaps")
        )
        
        # Only keep non-empty preprocessed data
        preprocessed = [p for p in preprocessed if len(p['clickmaps']) > 0]
        
        # Step 4: Process blurring in batches on GPU
        print(f"│  │  ├─ Processing blurring in batches on GPU (batch_size={batch_size})...")
        
        results = []
        
        # Process in batches
        for batch_idx in tqdm(range(0, len(preprocessed), batch_size), desc="Processing GPU batches"):
            batch = preprocessed[batch_idx:batch_idx + batch_size]
            batch_results = []
            
            for item in batch:
                image_key = item['key']
                clickmaps = item['clickmaps']
                native_size = item['native_size']
                
                # Convert numpy arrays to PyTorch tensors
                clickmaps_tensor = torch.from_numpy(clickmaps).float().unsqueeze(1).to(device)
                
                # Apply the appropriate blurring based on metadata
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
                        clickmaps_tensor = convolve(clickmaps_tensor, adj_blur_kernel)
                    elif kernel_type == "circle":
                        adj_blur_kernel = circle_kernel(adj_blur_size, adj_blur_sigma, device)
                        clickmaps_tensor = convolve(clickmaps_tensor, adj_blur_kernel, double_conv=True)
                else:
                    # Use the standard kernel
                    if kernel_type == "gaussian":
                        clickmaps_tensor = convolve(clickmaps_tensor, blur_kernel)
                    elif kernel_type == "circle":
                        clickmaps_tensor = convolve(clickmaps_tensor, blur_kernel, double_conv=True)
                
                # Apply center crop if needed
                if center_crop:
                    clickmaps_tensor = tvF.resize(clickmaps_tensor, min(image_shape))
                    clickmaps_tensor = tvF.center_crop(clickmaps_tensor, center_crop)
                
                # Convert back to numpy
                processed_clickmaps = clickmaps_tensor.squeeze().cpu().numpy()
                
                # Add to batch results
                batch_results.append({
                    'key': image_key,
                    'clickmaps': processed_clickmaps
                })
            
            # Add batch results to overall results
            results.extend(batch_results)
        
        # Step 5: Post-process results in parallel on CPU
        print(f"│  │  ├─ Post-processing results on CPU (parallel, n_jobs={n_jobs})...")
        
        def postprocess_clickmap(result, min_pixels, min_subjects, duplicate_thresh):
            """Helper function to post-process a clickmap after GPU processing"""
            image_key = result['key']
            clickmaps = result['clickmaps']
            
            # Filter processing
            if len(clickmaps.shape) == 2:  # Single map
                return None
            
            # Filter 1: Remove empties
            empty_check = (clickmaps > 0).sum((1, 2)) > min_pixels
            clickmaps = clickmaps[empty_check]
            if len(clickmaps) < min_subjects:
                return None
            
            # Filter 2: Remove duplicates
            clickmaps_vec = clickmaps.reshape(len(clickmaps), -1)
            dm = cdist(clickmaps_vec, clickmaps_vec)
            idx = np.tril_indices(len(dm), k=-1)
            lt_dm = dm[idx]
            if np.any(lt_dm < duplicate_thresh):
                remove = np.unique(np.where((dm + np.eye(len(dm)) == 0)))
                rng = np.arange(len(dm))
                dup_idx = rng[~np.in1d(rng, remove)]
                clickmaps = clickmaps[dup_idx]
            
            if len(clickmaps) >= min_subjects:
                return (image_key, clickmaps)
            return None
        
        # Use parallel processing for post-processing
        post_results = Parallel(n_jobs=n_jobs)(
            delayed(postprocess_clickmap)(
                results[i],
                min_pixels,
                min_subjects,
                duplicate_thresh
            ) for i in tqdm(range(len(results)), desc="Post-processing results")
        )
        
        # Step 6: Compile final results for this chunk
        chunk_result = {
            'all_clickmaps': [],
            'categories': [],
            'keep_index': [],
            'new_final_clickmaps': {}
        }
        
        for result in post_results:
            if result is not None:
                image_key, clickmaps = result
                category = image_key.split("/")[0]
                
                chunk_result['all_clickmaps'].append(clickmaps)
                chunk_result['categories'].append(category)
                chunk_result['keep_index'].append(image_key)
                chunk_result['new_final_clickmaps'][image_key] = clickmap_chunk[image_key]
        
        results_for_all_chunks.append(chunk_result)
    
    # If there's only one chunk, return its results directly
    if len(results_for_all_chunks) == 1:
        r = results_for_all_chunks[0]
        return r['new_final_clickmaps'], r['all_clickmaps'], r['categories'], r['keep_index']
    
    # Combine results from all chunks
    all_clickmaps = []
    categories = []
    keep_index = []
    new_final_clickmaps = {}
    
    for r in results_for_all_chunks:
        all_clickmaps.extend(r['all_clickmaps'])
        categories.extend(r['categories'])
        keep_index.extend(r['keep_index'])
        new_final_clickmaps.update(r['new_final_clickmaps'])
    
    return new_final_clickmaps, all_clickmaps, categories, keep_index

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
    batch_size = kwargs.pop('batch_size', 1024)  # Default batch size of 1024
    
    # Simply call the batched GPU function
    print(f"│  ├─ Processing with GPU-optimized batching (batch_size={batch_size})...")
    return prepare_maps_batched_gpu(final_clickmaps=final_clickmaps, batch_size=batch_size, **kwargs)

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
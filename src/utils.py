import re
import os
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
        final_keep_index,
        masks,
        mask_threshold,
        quantize_threshold=0.5):
    """
    quantize_threshold: If <= 0, use mean. If > 0, use this probability threshold for binarization.
    """
    proc_final_clickmaps, proc_all_clickmaps = {}, {}
    proc_categories, proc_final_keep_index = [], []
    # missing = []
    for idx, k in enumerate(final_keep_index):
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
                    proc_all_clickmaps[k] = click_map
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
        file_inclusion_filter=None,
        file_exclusion_filter=None,
        process_max="trim"):
    clickmaps = {}
    for _, row in clickme_data.iterrows():
        image_file_name = os.path.sep.join(row['image_path'].split(os.path.sep)[-2:])
        if file_inclusion_filter is not None and file_inclusion_filter not in image_file_name:
            continue
        if file_exclusion_filter is not None and file_exclusion_filter in image_file_name:
            continue
        if image_file_name not in clickmaps.keys():
            clickmaps[image_file_name] = [row["clicks"]]
        else:
            clickmaps[image_file_name].append(row["clicks"])

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


def circle_kernel(size, sigma=None):
    """
    Create a flat circular kernel where the values are the average of the total number of on pixels in the filter.

    Args:
        size (int): The diameter of the circle and the size of the kernel (size x size).
        sigma (float, optional): Not used for flat kernel. Included for compatibility. Default is None.

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

    return kernel


def prepare_maps(
        final_clickmaps,
        blur_size,
        blur_sigma,
        image_shape,
        min_pixels,
        min_subjects,
        center_crop,
        metadata=None,
        blur_sigma_function=None,
        kernel_type="circle",  # circle or gaussian
        duplicate_thresh=0.01,
        max_kernel_size=51):
    
    assert blur_sigma_function is not None, "Blur sigma function not passed."
    if kernel_type == "gaussian":
        blur_kernel = gaussian_kernel(blur_size, blur_sigma)
    elif kernel_type == "circle":
        blur_kernel = circle_kernel(blur_size, blur_sigma)
    else:
        raise NotImplementedError(kernel_type)

    category_correlations = {}
    all_clickmaps = []
    keep_index = []
    categories = []
    count = 0
    import pdb; pdb.set_trace()
    for image_key in tqdm(final_clickmaps, desc="Preparing maps", total=len(final_clickmaps)):
        count += 1
        category = image_key.split("/")[0]
        if category not in category_correlations.keys():
            category_correlations[category] = []
        image_trials = final_clickmaps[image_key]
        # clickmaps = np.asarray([create_clickmap([trials], image_shape) for trials in image_trials])
        # clickmaps = torch.from_numpy(clickmaps).float().unsqueeze(1)
        if metadata is not None:
            if image_key not in metadata:
                print(f"Image key {image_key} not in metadata")
                clickmaps = np.asarray([create_clickmap([trials], image_shape) for trials in image_trials])
                clickmaps = torch.from_numpy(clickmaps).float().unsqueeze(1)
                if kernel_type == "gaussian":
                    clickmaps = convolve(clickmaps, blur_kernel)
                elif kernel_type == "circle":
                    clickmaps = convolve(clickmaps, blur_kernel, double_conv=True)
                else:
                    raise NotImplementedError(kernel_type)
            else:
                native_size = metadata[image_key]
                short_side = min(native_size)
                scale = short_side / min(image_shape)
                adj_blur_size = int(np.round(blur_size * scale))
                if not adj_blur_size % 2:
                    adj_blur_size += 1  # Ensure odd kernel size
                adj_blur_size = min(adj_blur_size, max_kernel_size)
                adj_blur_sigma = blur_sigma_function(adj_blur_size)
                clickmaps = np.asarray([create_clickmap([trials], native_size[::-1]) for trials in image_trials])
                clickmaps = torch.from_numpy(clickmaps).float().unsqueeze(1)
                if kernel_type == "gaussian":
                    adj_blur_kernel = gaussian_kernel(adj_blur_size, adj_blur_sigma)
                    clickmaps = convolve(clickmaps, adj_blur_kernel)
                elif kernel_type == "circle":
                    adj_blur_kernel = circle_kernel(adj_blur_size, adj_blur_sigma)
                    clickmaps = convolve(clickmaps, adj_blur_kernel, double_conv=True)
                else:
                    raise NotImplementedError(kernel_type)
                del adj_blur_kernel
        else:
            clickmaps = np.asarray([create_clickmap([trials], image_shape) for trials in image_trials])
            clickmaps = torch.from_numpy(clickmaps).float().unsqueeze(1)
            if kernel_type == "gaussian":
                clickmaps = convolve(clickmaps, blur_kernel)
            elif kernel_type == "circle":
                clickmaps = convolve(clickmaps, blur_kernel, double_conv=True)
            else:
                raise NotImplementedError(kernel_type)
        if center_crop:
            clickmaps = tvF.center_crop(clickmaps, center_crop)
        clickmaps = clickmaps.squeeze()
        clickmaps = clickmaps.numpy()

        # Filter 1: Remove empties
        if len(clickmaps.shape) == 2:
            # Single map
            continue

        empty_check = (clickmaps > 0).sum((1, 2)) > min_pixels
        clickmaps = clickmaps[empty_check]
        if len(clickmaps) < min_subjects:
            continue

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
            # Store
            all_clickmaps.append(clickmaps)
            categories.append(category)
            keep_index.append(image_key)
    final_clickmaps = {k: v for k, v in final_clickmaps.items() if k in keep_index}
    return final_clickmaps, all_clickmaps, categories, keep_index


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
    

def gaussian_kernel(size, sigma):
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
    
    return kernel


def convolve(heatmap, kernel, double_conv=False):
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
    return blurred_heatmap  # [0]


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
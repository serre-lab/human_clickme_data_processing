import os
import numpy as np
import torch
import torch.nn.functional as F
import random
from scipy.stats import spearmanr
from PIL import Image
import re
import pandas as pd
from utils import gaussian_kernel, gaussian_blur, create_clickmap
from torchvision.transforms import functional as tvF
from matplotlib import pyplot as plt
from tqdm import tqdm


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

def gaussian_blur(heatmap, kernel):
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

    return blurred_heatmap  # [0]

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

def split_half_correlation(
        image_trials,
        image_shape,
        resample_means=False,
        blur_kernel=None,
        n_splits=1000,
        bootstrap=False,
        center_crop=None,
        version="single_subject",
        metric="crossentropy"
):
    """
    Compute the split-half correlation for a set of image trials.

    Args:
        image_trials (list of list of tuples): List of trials, each containing click points.
        image_shape (tuple): Shape of the image (height, width).
        resample_means (bool): If True, resample trials when computing means. Default is False.
        blur_kernel (torch.Tensor, optional): Gaussian kernel for blurring. Default is None.
        n_splits (int): Number of splits to perform. Default is 1000.
        bootstrap (bool): If True, use bootstrap resampling. Default is False.

    Returns:
        float: The average split-half correlation across all splits.
    """
    correlations = []

    # clickmaps = create_clickmap(image_trials, image_shape)
    clickmaps = np.asarray([create_clickmap([trials], image_shape) for trials in image_trials])
    clickmaps = torch.from_numpy(clickmaps).float().unsqueeze(1)
    clickmaps = gaussian_blur(clickmaps, blur_kernel)
    if center_crop is not None:
        clickmaps = tvF.center_crop(clickmaps, center_crop)
    clickmaps = clickmaps.squeeze()
    clickmaps = clickmaps.numpy()

    # Check if any maps are all zeros and remove them
    zero_maps = clickmaps.sum((1, 2)) == 0
    clickmaps = clickmaps[~zero_maps]
    num_trials = len(clickmaps)
    half = num_trials // 2

    if version == "single_subject":
        for i in range(len(clickmaps)):
            test_map = clickmaps[i]
            test_map = (test_map - test_map.min()) / (test_map.max() - test_map.min())
            remaining_maps = clickmaps[~np.in1d(np.arange(len(clickmaps)), i)].mean(0)
            remaining_maps = (remaining_maps - remaining_maps.min()) / (remaining_maps.max() - remaining_maps.min())
            if metric == "crossentropy":
                correlation = compute_crossentropy(test_map, remaining_maps)
            else:
                correlation = compute_spearman_correlation(test_map, remaining_maps)
            correlations.append(correlation)
    elif version == "multiple_subjects":
        for _ in range(n_splits):
            indices = np.arange(num_trials)
            indices = np.random.choice(indices, size=num_trials, replace=bootstrap)
            first_half_indices = indices[:half]
            second_half_indices = indices[half:]

            avg_map1 = clickmaps[first_half_indices].mean(0)
            avg_map2 = clickmaps[second_half_indices].mean(0)
            avg_map1 = (avg_map1 - avg_map1.min()) / (avg_map1.max() - avg_map1.min())
            avg_map2 = (avg_map2 - avg_map2.min()) / (avg_map2.max() - avg_map2.min())
            if metric == "crossentropy":
                correlation = compute_crossentropy(avg_map1, avg_map2)
            else:
                correlation = compute_spearman_correlation(avg_map1, avg_map2)
            correlations.append(correlation)
    else:
        raise ValueError(f"Invalid version: {version}")
    return np.nanmean(correlations), clickmaps

def main(
    final_clickmaps,
    co3d_clickme_folder,
    n_splits=1000,
    debug=False,
    blur_size=33,
    blur_sigma=11,
    null_iterations=1000,
    image_shape=[256, 256],
    center_crop=[224, 224],
    metric="spearman"
):
    """
    Calculate split-half correlations for clickmaps across different image categories.

    Args:
        final_clickmaps (dict): A dictionary where keys are image identifiers and values
                                are lists of click trials for each image.
        co3d_clickme_folder (str): Path to the folder containing the images.
        n_splits (int): Number of splits to use in split-half correlation calculation.
        debug (bool): If True, print debug information.
        blur_size (int): Size of the Gaussian blur kernel.
        blur_sigma (float): Sigma value for the Gaussian blur kernel.
        image_shape (list): Shape of the image [height, width].

    Returns:
        tuple: A tuple containing two elements:
            - dict: Category-wise mean correlations.
            - list: All individual image correlations.
    """
    category_correlations = {}
    all_correlations = []
    all_clickmaps = []
    blur_kernel = gaussian_kernel(blur_size, blur_sigma)

    for image_key in tqdm(final_clickmaps, desc="Processing ceiling score"):
        category = image_key.split("/")[0]
        if category not in category_correlations.keys():
            category_correlations[category] = []
        image_path = os.path.join(co3d_clickme_folder, image_key)
        image_data = Image.open(image_path)
        image_trials = final_clickmaps[image_key]
        
        mean_correlation, clickmap = split_half_correlation(
            image_trials,
            image_shape,
            resample_means=True,
            blur_kernel=blur_kernel,
            center_crop=center_crop,
            n_splits=n_splits,
            metric=metric)
        
        if debug:
            print(f"Mean split-half correlation for {image_key} : {mean_correlation}")
        category_correlations[category].append(mean_correlation)
        all_correlations.append(mean_correlation)
        all_clickmaps.append(clickmap)
    import pdb; pdb.set_trace()
    print(f"Mean Human Correlation: {np.nanmean(all_correlations)}")

    null_correlations = []
    for i in range(null_iterations):
        fhs = []
        shs = []
        for cm in all_clickmaps:
            num_subs = len(cm)
            rand_subs = np.random.choice(num_subs, size=num_subs, replace=False)
            fh_subs = rand_subs[:num_subs//2]
            sh_subs = rand_subs[num_subs//2:]
            fhs.append(cm[fh_subs].mean(0))
            shs.append(cm[sh_subs].mean(0))
        if metric == "crossentropy":
            null_correlations.append(compute_crossentropy(fhs, shs))
        else:
            null_correlations.append(compute_spearman_correlation(fhs, shs))
    null_correlations = np.array(null_correlations)
    print(f"Null Correlations: {np.nanmean(null_correlations)}")

    for cat in category_correlations:
        category_correlations[cat] = np.nanmean(np.array(category_correlations[cat]))

    print(f"Category-wise Correlations: {category_correlations}")

    return category_correlations, all_correlations, null_correlations

if __name__ == "__main__":
    co3d_clickme = pd.read_csv("clickme_vCO3D.csv")
    co3d_clickme_folder = "CO3D_ClickMe2"

    clickmaps = {}

    for index, row in co3d_clickme.iterrows():
        image_file_name = row['image_path'].replace("CO3D_ClickMe2/", "")
        if image_file_name not in clickmaps.keys():
            clickmaps[image_file_name] = [row["clicks"]]
        else:
            clickmaps[image_file_name].append(row["clicks"])

    number_of_maps = []
    final_clickmaps = {}
    counters = 0
    n_empty_clickmap = 0

    for image in clickmaps:
        n_clickmaps = 0
        for clickmap in clickmaps[image]:
            if len(clickmap) == 2:
                n_empty_clickmap += 1
                continue

            n_clickmaps += 1

            clean_string = re.sub(r'[{}"]', '', clickmap)
            tuple_strings = clean_string.split(', ')
            data_list = tuple_strings[0].strip("()").split("),(")
            tuples_list = [tuple(map(int, pair.split(','))) for pair in data_list]

            if image not in final_clickmaps.keys():
                final_clickmaps[image] = []
            final_clickmaps[image].append(tuples_list)

        number_of_maps.append(n_clickmaps)

    category_correlations, all_correlations, null_correlations = main(final_clickmaps=final_clickmaps, co3d_clickme_folder=co3d_clickme_folder)
    import pdb; pdb.set_trace()
    np.savez(
        "human_ceiling_results.npz",
        category_correlations=category_correlations,
        ceiling_correlations=all_correlations,
        null_correlations=null_correlations
    )

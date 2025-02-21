import os, sys
import numpy as np
from PIL import Image
from src import JAY_utils as utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.stats as stats
import random


def minmax_normalize(x):
    """Simple min-max normalization to avoid division by zero."""
    return (x - x.min()) / (x.max() - x.min() + 1e-7)

def single_image_noise_ceiling(image_clickmaps, metric="spearman", iterations=100):
    """
    Split-half reliability (noise ceiling) for a single image.
    image_clickmaps: np.array of shape [X, H, W].
    """
    scores = []
    n_subs = len(image_clickmaps)
    for _ in range(iterations):
        idx = np.random.permutation(n_subs)
        fh = image_clickmaps[idx[: n_subs // 2]].mean(0)
        sh = image_clickmaps[idx[n_subs // 2 :]].mean(0)
        fh, sh = minmax_normalize(fh), minmax_normalize(sh)
        if metric.lower() == "spearman":
            scores.append(utils.compute_spearman_correlation(fh, sh))
        elif metric.lower() == "auc":
            scores.append(utils.compute_AUC(fh, sh))
        elif metric.lower() == "crossentropy":
            scores.append(utils.compute_crossentropy(fh, sh))
        else:
            raise ValueError("Invalid metric.")
    return np.mean(scores)

def null_distribution_single_pair(i_maps, j_maps, metric="spearman", iterations=100):
    """
    Combine clickmaps from image i and j, then split-half for null distribution.
    i_maps, j_maps: each [X, H, W].
    """
    combined = np.concatenate([i_maps, j_maps], axis=0)
    scores = []
    n_subs = len(combined)
    for _ in range(iterations):
        idx = np.random.permutation(n_subs)
        fh = combined[idx[: n_subs // 2]].mean(0)
        sh = combined[idx[n_subs // 2 :]].mean(0)
        fh, sh = minmax_normalize(fh), minmax_normalize(sh)
        if metric.lower() == "spearman":
            scores.append(utils.compute_spearman_correlation(fh, sh))
        elif metric.lower() == "auc":
            scores.append(utils.compute_AUC(fh, sh))
        elif metric.lower() == "crossentropy":
            scores.append(utils.compute_crossentropy(fh, sh))
        else:
            raise ValueError("Invalid metric.")
    return np.mean(scores)

def sample_clickmaps(clickmaps, num_subs):
    """Randomly sample a subset of subjects from each image's clickmaps.
    
    Args:
        clickmaps (np.ndarray): Array of shape (n_images, n_subjects, height, width)
        num_subs (int): Number of subjects to sample
    
    Returns:
        np.ndarray: Array of sampled clickmaps with shape (n_images, num_subs, height, width)
    """
    sel_clickmaps = []
    for clickmap in clickmaps:
        idx = np.random.permutation(len(clickmap))[:num_subs]
        sel_clickmaps.append(clickmap[idx])
    return np.asarray(sel_clickmaps)



def main(
        clickme_data,
        clickme_image_folder,
        debug=False,
        blur_size=11 * 2,
        blur_sigma=np.sqrt(11 * 2),
        null_iterations=10,
        image_shape=[256, 256],
        center_crop=[224, 224],
        min_pixels=30,
        min_subjects=10,
        max_subjects=20,
        min_clicks=10,
        max_clicks=50,
        randomization_iters=10,
        metadata=None,
        metric="auc",  # AUC, crossentropy, spearman, RSA
        blur_sigma_function=None,
        mask_dir=None,
        mask_threshold=0.5,
        class_filter_file=False,
        participant_filter=False,
        file_inclusion_filter=False,
        file_exclusion_filter=False,
    ):
    """
    Calculate split-half correlations for clickmaps across different image categories.

    Args:
        final_clickmaps (dict): A dictionary where keys are image identifiers and values
                                are lists of click trials for each image.
        clickme_image_folder (str): Path to the folder containing the images.
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

    assert blur_sigma_function is not None, "Blur sigma function needs to be provided."

    # Process files in serial
    if config["parallel_prepare_maps"]:
        process_clickmap_files = utils.process_clickmap_files_parallel
    else:
        process_clickmap_files = utils.process_clickmap_files
    clickmaps, _ = process_clickmap_files(
        clickme_data=clickme_data,
        image_path=clickme_image_folder,
        file_inclusion_filter=file_inclusion_filter,
        file_exclusion_filter=file_exclusion_filter,
        min_clicks=min_clicks,
        max_clicks=max_clicks)
    # Filter classes if requested
    if class_filter_file:
        clickmaps = utils.filter_classes(
            clickmaps=clickmaps,
            class_filter_file=class_filter_file)
    # Filter participants if requested
    if participant_filter:
        clickmaps = utils.filter_participants(clickmaps)

    # Prepare maps
    if config["parallel_prepare_maps"]:
        prepare_maps = utils.prepare_maps_parallel
    else:
        prepare_maps = utils.prepare_maps
    
    # Simplify subject filtering: do partial filtering here
    filtered_clickmaps = {}
    for k, v in clickmaps.items():
        if len(v) >= min_subjects:
            filtered_clickmaps[k] = v
    clickmaps = filtered_clickmaps
    
    final_clickmaps, all_clickmaps, categories, _ = prepare_maps(
        final_clickmaps=clickmaps,
        blur_size=blur_size,
        blur_sigma=blur_sigma,
        image_shape=image_shape,
        min_pixels=min_pixels,
        min_subjects=min_subjects,
        max_subjects=max_subjects,
        metadata=metadata,
        blur_sigma_function=blur_sigma_function,
        center_crop=center_crop)
        
    # Filter for foreground mask overlap if requested  
    if mask_dir:
        masks = utils.load_masks(mask_dir)
        final_clickmaps, all_clickmaps, categories, final_keep_index = utils.filter_for_foreground_masks(
            final_clickmaps=final_clickmaps,
            all_clickmaps=all_clickmaps,
            categories=categories,
            masks=masks,
            mask_threshold=mask_threshold)
            
    if debug:
        for imn in range(len(final_clickmaps)):
            f = [x for x in final_clickmaps.keys()][imn]
            image_path = os.path.join(clickme_image_folder, f)
            image_data = Image.open(image_path)
            for idx in range(min(len(all_clickmaps[imn]), 18)):
                plt.subplot(4, 5, idx + 1)
                plt.imshow(all_clickmaps[imn][np.argsort(all_clickmaps[imn].sum((1, 2)))[idx]])
                plt.axis("off")
            plt.subplot(4, 5, 20)
            plt.subplot(4,5,19);plt.imshow(all_clickmaps[imn].mean(0))
            plt.axis('off');plt.title("mean")
            plt.subplot(4,5,20);plt.imshow(np.asarray(image_data)[16:-16, 16:-16]);plt.axis('off')
            plt.show()

    # Compute correlations using bootstrapping
    min_subjects, max_subjects = 3, 10
    mean_results, stdev_results = [], []
    null_mean_results, null_stdev_results = [], []
    
    for num_subs in tqdm(range(min_subjects, max_subjects+1), desc="Number of subjects", total=max_subjects-min_subjects):
        sel_clickmaps = sample_clickmaps(all_clickmaps, num_subs)
        # Correlation
        image_scores = []
        for enum, clickmap in enumerate(sel_clickmaps):
            # Loop through images and compute scores
            image_scores.append(single_image_noise_ceiling(clickmap))
        mean_results.append(np.mean(image_scores))  # Store average score accross images
        print(f'avg corr at {num_subs} subjects is {np.mean(image_scores)}')
        stdev_results.append(np.std(image_scores))  # Changed from np.mean to np.std
        print(f'std at {num_subs} subjects is {np.std(image_scores)}')
        
        # Null scores
        null_scores = []
        for i, clickmap in enumerate(sel_clickmaps):
            # Select a random different image
            j = np.random.choice([idx for idx in range(len(sel_clickmaps)) if idx != i])
            test_maps = sel_clickmaps[i]
            ref_maps = sel_clickmaps[j]
            null_scores.append(null_distribution_single_pair(test_maps, ref_maps))
        null_mean_results.append(np.mean(null_scores))  # Store average score accross images
        print(f'avg null corr at {num_subs} subjects is {np.mean(null_scores)}')
        null_stdev_results.append(np.std(null_scores))  # Changed from np.mean to np.std
        print(f'std null at {num_subs} subjects is {np.std(null_scores)}')
    
    return final_clickmaps, mean_results, null_mean_results, all_clickmaps


if __name__ == "__main__":

    # Get config file
    config_file = utils.get_config(sys.argv)

    # Other Args
    # blur_sigma_function = lambda x: np.sqrt(x)
    # blur_sigma_function = lambda x: x / 2
    blur_sigma_function = lambda x: x

    # Load config
    config = utils.process_config(config_file)
    output_dir = config["assets"]
    blur_size = config["blur_size"]
    blur_sigma = np.sqrt(blur_size)
    min_pixels = (2 * blur_size) ** 2  # Minimum number of pixels for a map to be included following filtering

    # Load metadata
    if config["metadata_file"]:
        metadata = np.load(config["metadata_file"], allow_pickle=True).item()
    else:
        metadata = None

    # Load data
    clickme_data = utils.process_clickme_data(
        config["clickme_data"],
        config["filter_mobile"])

    # Process data
    final_clickmaps, all_correlations, null_correlations, all_clickmaps = main(
        clickme_data=clickme_data,
        blur_sigma=blur_sigma,
        min_pixels=min_pixels,
        debug=config["debug"],
        blur_size=blur_size,
        clickme_image_folder=config["image_path"],
        null_iterations=config["null_iterations"],
        image_shape=config["image_shape"],
        center_crop=config["center_crop"],
        min_subjects=config["min_subjects"],
        max_subjects=config["max_subjects"],
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"],
        metadata=metadata,
        metric=config["metric"],
        blur_sigma_function=blur_sigma_function,
        mask_dir=config["mask_dir"],
        mask_threshold=config["mask_threshold"],
        class_filter_file=config["class_filter_file"],
        participant_filter=config["participant_filter"],
        file_inclusion_filter=config["file_inclusion_filter"],
        file_exclusion_filter=config["file_exclusion_filter"])
    
    # Compute mean and 95% confidence interval
    mean_human_correlation = np.nanmean(all_correlations)
    ci_lower, ci_upper = np.percentile(all_correlations, [2.5, 97.5])

    # Compute mean and CI for null
    mean_null_correlation = np.nanmean(null_correlations)
    ci_null_lower, ci_null_upper = np.percentile(null_correlations, [2.5, 97.5])

    print(f"Mean human correlation full set: {mean_human_correlation}")
    print(f"Number of iterations contributing to mean correlation: {len(all_correlations) - np.isnan(all_correlations).sum()}")
    print(f"Human correlation, 95% CI: [{ci_lower}, {ci_upper}]")
    
    print(f"Mean null correlation: {mean_null_correlation}")
    print(f"Null correlation, 95% CI: [{ci_null_lower}, {ci_null_upper}]")

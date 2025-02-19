import random
import os, sys
import numpy as np
from PIL import Image
from src import JAY_utils as utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.stats as stats

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

def main(clickme_data, clickme_image_folder, min_clicks=10, max_clicks=50, metric="spearman", iterations=100, seed=0):
    """
    1) Filter to all images with exactly 10 subjects.
    2) Randomly select 10 of those images.
    3) For each X in [3..10], sample X maps (no replacement) from each image => compute noise ceiling.
    4) For null distribution, pick X maps from image i, X from another image j => compute split-half.
    5) Print results for 3..10 (no plotting).
    """
    np.random.seed(seed)
    random.seed(seed)

    # Load your data (or do any needed filtering/preprocessing here).
    # Example:
    clickmaps_dict, _ = utils.process_clickmap_files_parallel(
        clickme_data=clickme_data,
        image_path=clickme_image_folder,
        min_clicks=min_clicks,
        max_clicks=max_clicks
    )

    # Filter to images with exactly 10 subjects
    images_with_10 = [img_key for img_key, maps in clickmaps_dict.items() if len(maps) == 10]
    if len(images_with_10) < 10:
        raise ValueError("Not enough images with exactly 10 subjects.")

    # Randomly choose 10 of those images
    chosen_images = random.sample(images_with_10, 10)

    # Collect clickmaps in a list for convenience: each entry -> (10, H, W)
    all_clickmaps_10 = []
    for img_key in chosen_images:
        # Suppose each maps entry is already in 2D form. If needed, do your normal preprocessing here.
        all_clickmaps_10.append(np.stack(clickmaps_dict[img_key], axis=0))  # shape: [10, H, W]

    # We'll store noise & null results for each X
    noise_results = {}
    null_results = {}

    # Loop from X=3 to 10
    for X in range(3, 11):
        # Noise Ceiling
        per_image_nc = []
        for i_maps in all_clickmaps_10:
            chosen_idx = np.random.choice(10, X, replace=False)
            selected = i_maps[chosen_idx]  # shape [X, H, W]
            per_image_nc.append(single_image_noise_ceiling(selected, metric=metric, iterations=iterations))
        noise_results[X] = float(np.mean(per_image_nc))

        # Null Distribution
        per_image_null = []
        n_images = len(all_clickmaps_10)
        for i in range(n_images):
            # pick a random j != i
            valid_j = [jj for jj in range(n_images) if jj != i]
            if len(valid_j) == 0:
                continue
            j = np.random.choice(valid_j)
            i_idx = np.random.choice(10, X, replace=False)
            j_idx = np.random.choice(10, X, replace=False)
            i_subset = all_clickmaps_10[i][i_idx]
            j_subset = all_clickmaps_10[j][j_idx]
            per_image_null.append(
                null_distribution_single_pair(i_subset, j_subset, metric=metric, iterations=iterations)
            )
        null_results[X] = float(np.mean(per_image_null))

    # Print final results
    print("==== Noise Ceiling (Within-Image) ====")
    for X in range(3, 11):
        print(f"X={X}: {noise_results[X]:.4f}")

    print("\n==== Null Distribution (Across-Image) ====")
    for X in range(3, 11):
        print(f"X={X}: {null_results[X]:.4f}")

if __name__ == "__main__":
    # Get config file
    config_file = utils.get_config(sys.argv)

    config = utils.process_config(config_file)

    main(
        clickme_data=config["clickme_data"],
        clickme_image_folder=config["image_path"],
        metric=config["metric"],
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"],
        iterations=100,
        seed=42
    )

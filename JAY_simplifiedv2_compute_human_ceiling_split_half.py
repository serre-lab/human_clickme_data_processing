import os, sys
import numpy as np
from PIL import Image
from src import JAY_utils as utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import scipy.stats as stats
import random
import pickle


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


def main():
    
    all_clickmaps = pickle.load(open("assets/all_clickmaps.pkl", "rb"))
    all_clickmaps = all_clickmaps[:20]
    
    
    min_subjects, max_subjects = 3, 10
    mean_results, stdev_results = [], []
    null_mean_results, null_stdev_results = [], []
    
    for num_subs in tqdm(range(min_subjects, max_subjects), desc="Number of subjects", total=max_subjects-min_subjects):
        sel_clickmaps = sample_clickmaps(all_clickmaps, num_subs)
        # Correlation
        image_scores = []
        for enum, clickmap in enumerate(sel_clickmaps):
            print(enum)
            # Loop through images and compute scores
            image_scores.append(single_image_noise_ceiling(clickmap))
        mean_results.append(np.mean(image_scores))  # Store average score accross images
        print(f'avg corr at {num_subs} subjects is {np.mean(image_scores)}')
        stdev_results.append(np.std(image_scores))  # Changed from np.mean to np.std
        print(f'std at {num_subs} subjects is {np.std(image_scores)}')
        
        # Null scores
        null_scores = []
        for i, clickmap in enumerate(sel_clickmaps):
            print(i)
            # Select a random different image
            j = np.random.choice([idx for idx in range(len(sel_clickmaps)) if idx != i])
            test_maps = sel_clickmaps[i]
            ref_maps = sel_clickmaps[j]
            null_scores.append(null_distribution_single_pair(test_maps, ref_maps))
        null_mean_results.append(np.mean(null_scores))  # Store average score accross images
        print(f'avg null corr at {num_subs} subjects is {np.mean(null_scores)}')
        null_stdev_results.append(np.std(null_scores))  # Changed from np.mean to np.std
        print(f'std null at {num_subs} subjects is {np.std(null_scores)}')
    
    # Add plotting
    plt.figure(figsize=(10, 6))
    x = range(min_subjects, max_subjects)
    plt.errorbar(x, mean_results, yerr=stdev_results, fmt='o-', capsize=5)
    plt.xlabel('Number of Subjects')
    plt.ylabel('Noise Ceiling Correlation')
    plt.title('Human Noise Ceiling vs Number of Subjects')
    plt.grid(True)
    plt.show()
    
    return 1, all_correlations, null_correlations, all_clickmaps


if __name__ == "__main__":

    main()

    # Process data
    _, all_correlations, null_correlations, all_clickmaps = main()
    
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

    np.savez(
        os.path.join(output_dir, "human_ceiling_split_half_{}.npz".format(config["experiment_name"])),
        final_clickmaps=final_clickmaps,
        ceiling_correlations=all_correlations,
        null_correlations=null_correlations,
    )

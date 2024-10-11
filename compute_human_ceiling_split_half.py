import os, sys
import numpy as np
from PIL import Image
from src import utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed


def compute_inner_correlations(i, all_clickmaps, category_indices, metric):
    category_index = category_indices[i]
    inner_correlations = []
    instance_correlations = {}

    if i not in instance_correlations.keys():
        instance_correlations[i] = []

    # Reference map is the ith map
    reference_map = all_clickmaps[i].mean(0)
    reference_map = (reference_map - reference_map.min()) / (reference_map.max() - reference_map.min())

    # Test map is a random subject from a different image
    sub_vec = np.where(category_indices != category_index)[0]
    rand_map = np.random.choice(sub_vec)
    test_map = all_clickmaps[rand_map]
    num_subs = len(test_map)
    rand_sub = np.random.choice(num_subs)
    test_map = test_map[rand_sub]
    test_map = (test_map - test_map.min()) / (test_map.max() - test_map.min())

    if metric.lower() == "crossentropy":
        correlation = utils.compute_crossentropy(test_map, reference_map)
    elif metric.lower() == "auc":
        correlation = utils.compute_AUC(test_map, reference_map)
    elif metric.lower() == "rsa":
        correlation = utils.compute_RSA(test_map, reference_map)
    elif metric.lower() == "spearman":
        correlation = utils.compute_spearman_correlation(test_map, reference_map)
    else:
        raise ValueError(f"Invalid metric: {metric}")

    inner_correlations.append(correlation)
    instance_correlations[i].append(correlation)

    return inner_correlations, instance_correlations


def main(
        clickme_data,
        clickme_folder,
        debug=False,
        blur_size=11 * 2,
        blur_sigma=np.sqrt(11 * 2),
        null_iterations=10,
        image_shape=[256, 256],
        center_crop=[224, 224],
        min_pixels=30,
        min_subjects=10,
        min_clicks=10,
        max_clicks=50,
        randomization_iters=10,
        metadata=None,
        metric="auc",  # AUC, crossentropy, spearman, RSA
        blur_sigma_function=None
    ):
    """
    Calculate split-half correlations for clickmaps across different image categories.

    Args:
        final_clickmaps (dict): A dictionary where keys are image identifiers and values
                                are lists of click trials for each image.
        clickme_folder (str): Path to the folder containing the images.
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
    clickmaps, _ = utils.process_clickmap_files(
        clickme_data=clickme_data,
        min_clicks=min_clicks,
        max_clicks=max_clicks)

    # Prepare maps
    final_clickmaps, all_clickmaps, categories, _ = utils.prepare_maps(
        final_clickmaps=clickmaps,
        blur_size=blur_size,
        blur_sigma=blur_sigma,
        image_shape=image_shape,
        min_pixels=min_pixels,
        min_subjects=min_subjects,
        metadata=metadata,
        blur_sigma_function=blur_sigma_function,
        center_crop=center_crop)

    if debug:
        for imn in range(len(final_clickmaps)):
            f = [x for x in final_clickmaps.keys()][imn]
            image_path = os.path.join(clickme_folder, f)
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

    # Compute scores through split-halfs
    all_correlations = []
    for clickmaps in tqdm(all_clickmaps, desc="Processing ceiling", total=len(all_clickmaps)):
        n = len(clickmaps)
        rand_corrs = []
        for _ in range(randomization_iters):
            rand_perm = np.random.permutation(n)
            fh = rand_perm[:(n // 2)]
            sh = rand_perm[(n // 2):]
            test_maps = clickmaps[fh].mean(0)
            remaining_maps = clickmaps[sh].mean(0)
            test_maps = (test_maps - test_maps.min()) / (test_maps.max() - test_maps.min())
            remaining_maps = (remaining_maps - remaining_maps.min()) / (remaining_maps.max() - remaining_maps.min())
            if metric.lower() == "crossentropy":
                correlation = utils.compute_crossentropy(test_maps, remaining_maps)
            elif metric.lower() == "auc":
                correlation = utils.compute_AUC(test_maps, remaining_maps)
            elif metric.lower() == "spearman":
                correlation = utils.compute_spearman_correlation(test_maps, remaining_maps)
            else:
                raise ValueError(f"Invalid metric: {metric}")
            rand_corrs.append(correlation)
        all_correlations.append(np.mean(rand_corrs))
    all_correlations = np.asarray(all_correlations)

    # Compute null scores
    null_correlations = []
    click_len = len(all_clickmaps)
    for _ in tqdm(range(null_iterations), total=null_iterations, desc="Computing null scores"):
        inner_correlations = []
        for i in range(click_len):
            selected_clickmaps = all_clickmaps[i]
            tmp_rng = np.arange(click_len)
            j = tmp_rng[~np.in1d(tmp_rng, i)]
            j = j[np.random.permutation(len(j))][0]  # Select a random other image
            other_clickmaps = all_clickmaps[j]
            rand_perm_sel = np.random.permutation(len(selected_clickmaps))
            rand_perm_other = np.random.permutation(len(other_clickmaps))
            fh = rand_perm_sel[:(n // 2)]
            sh = rand_perm_other[(n // 2):]
            test_maps = selected_clickmaps[fh].mean(0)
            remaining_maps = other_clickmaps[sh].mean(0)
            test_maps = (test_maps - test_maps.min()) / (test_maps.max() - test_maps.min())
            remaining_maps = (remaining_maps - remaining_maps.min()) / (remaining_maps.max() - remaining_maps.min())
            if metric.lower() == "crossentropy":
                correlation = utils.compute_crossentropy(test_maps, remaining_maps)
            elif metric.lower() == "auc":
                correlation = utils.compute_AUC(test_maps, remaining_maps)
            elif metric.lower() == "spearman":
                correlation = utils.compute_spearman_correlation(test_maps, remaining_maps)
            else:
                raise ValueError(f"Invalid metric: {metric}")
            inner_correlations.append(correlation)
        null_correlations.append(np.nanmean(inner_correlations))
    null_correlations = np.asarray(null_correlations)
    return final_clickmaps, all_correlations, null_correlations, all_clickmaps


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
        clickme_folder=config["image_dir"],
        null_iterations=config["null_iterations"],
        image_shape=config["image_shape"],
        center_crop=config["center_crop"],
        min_subjects=config["min_subjects"],
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"],
        metadata=metadata,
        metric=config["metric"],
        blur_sigma_function=blur_sigma_function)
    print(f"Mean human correlation full set: {np.nanmean(all_correlations)}")
    print(f"Null correlations full set: {np.nanmean(null_correlations)}")
    np.savez(
        os.path.join(output_dir, "human_ceiling_split_half_{}.npz".format(config["experiment_name"])),
        final_clickmaps=final_clickmaps,
        ceiling_correlations=all_correlations,
        null_correlations=null_correlations,
    )

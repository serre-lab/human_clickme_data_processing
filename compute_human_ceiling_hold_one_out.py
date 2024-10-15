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
        clickme_image_folder,
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
        metric="auc",  # AUC, crossentropy, spearman, RSA
        blur_sigma_function=None,
        mask_dir=None,
        mask_threshold=0.5,
        class_filter_file=None,
        file_inclusion_filter=None,
        file_exclusion_filter=None,
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
        max_clicks=max_clicks,
        file_inclusion_filter=file_inclusion_filter,
        file_exclusion_filter=file_exclusion_filter)

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

    # Filter classes if requested
    if config["class_filter_file"]:
        clickmaps = utils.filter_classes(
            clickmaps=clickmaps,
            class_filter_file=config["class_filter_file"])

    # Filter participants if requested
    if config["participant_filter"]:
        clickmaps = utils.filter_participants(clickmaps)

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

    # Compute scores
    all_correlations = []
    for clickmaps in tqdm(all_clickmaps, desc="Processing ceiling", total=len(all_clickmaps)):
        for i in range(len(clickmaps)):
            test_map = clickmaps[i]
            test_map = (test_map - test_map.min()) / (test_map.max() - test_map.min())
            remaining_maps = clickmaps[~np.in1d(np.arange(len(clickmaps)), i)].mean(0)
            remaining_maps = (remaining_maps - remaining_maps.min()) / (remaining_maps.max() - remaining_maps.min())
            if metric.lower() == "crossentropy":
                correlation = utils.compute_crossentropy(test_map, remaining_maps)
            elif metric.lower() == "auc":
                correlation = utils.compute_AUC(test_map, remaining_maps)
            elif metric.lower() == "spearman":
                correlation = utils.compute_spearman_correlation(test_map, remaining_maps)
            else:
                raise ValueError(f"Invalid metric: {metric}")
            all_correlations.append(correlation)
    all_correlations = np.asarray(all_correlations)

    # Filter for foreground mask overlap if requested
    if mask_dir:
        masks = utils.load_masks(mask_dir)
        final_clickmaps, all_clickmaps, categories, _ = utils.filter_for_foreground_masks(
            final_clickmaps=final_clickmaps,
            all_clickmaps=all_clickmaps,
            categories=categories,
            masks=masks,
            mask_threshold=mask_threshold)

    # Compute null scores
    _, category_indices = np.unique(categories, return_inverse=True)
    null_correlations = []
    instance_correlations = {}
    for _ in tqdm(range(null_iterations), total=null_iterations, desc="Computing null scores"):
        results = Parallel(n_jobs=-1)(delayed(compute_inner_correlations)(i, all_clickmaps, category_indices, metric) for i in range(len(all_clickmaps)))
        inner_correlations = [result[0] for result in results]
        instance_correlations = {k: v for result in results for k, v in result[1].items()}
        null_correlations.append(np.nanmean(inner_correlations))
    null_correlations = np.asarray(null_correlations)
    return final_clickmaps, instance_correlations, all_correlations, null_correlations, all_clickmaps


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
    final_clickmaps, instance_correlations, all_correlations, null_correlations, all_clickmaps = main(
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
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"],
        metric=config["metric"],
        blur_sigma_function=blur_sigma_function,
        mask_dir=config["mask_dir"],
        mask_threshold=config["mask_threshold"],
        class_filter_file=config["class_filter_file"],
        participant_filter=config["participant_filter"],
        file_inclusion_filter=config["file_inclusion_filter"],
        file_exclusion_filter=config["file_exclusion_filter"])
    print(f"Mean human correlation full set: {np.nanmean(all_correlations)}")
    print(f"Null correlations full set: {np.nanmean(null_correlations)}")
    np.savez(
        os.path.join(output_dir, "human_ceiling_hold_one_out_{}.npz".format(config["experiment_name"])),
        final_clickmaps=final_clickmaps,
        ceiling_correlations=all_correlations,
        null_correlations=null_correlations,
        instance_correlations=instance_correlations,
    )

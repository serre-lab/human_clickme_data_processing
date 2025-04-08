import os, sys
import numpy as np
from PIL import Image
from src import utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
from scipy.stats import spearmanr


def compute_correlation_batch(batch_indices, all_clickmaps, metric, n_iterations=10, device='cuda'):
    """Compute split-half correlations for a batch of clickmaps in parallel"""
    batch_results = []
    
    # Handle Spearman correlation separately if that's the metric
    if metric.lower() == "spearman":
        for i in batch_indices:
            clickmaps = all_clickmaps[i]
            n = len(clickmaps)
            rand_corrs = []
            
            for _ in range(n_iterations):
                rand_perm = np.random.permutation(n)
                fh = rand_perm[:(n // 2)]
                sh = rand_perm[(n // 2):]
                
                # Create the test and reference maps
                test_map = clickmaps[fh].mean(0)
                reference_map = clickmaps[sh].mean(0)
                
                # Normalize maps
                test_map = (test_map - test_map.min()) / (test_map.max() - test_map.min() + 1e-10)
                reference_map = (reference_map - reference_map.min()) / (reference_map.max() - reference_map.min() + 1e-10)
                
                # Use scipy's spearman correlation
                correlation, _ = spearmanr(test_map.flatten(), reference_map.flatten())
                rand_corrs.append(correlation)
            
            batch_results.append(np.mean(rand_corrs))
        return batch_results
    
    # For other metrics, use GPU acceleration
    # Lists to store test and reference maps for batch GPU processing
    test_maps_batch = []
    reference_maps_batch = []
    
    for i in batch_indices:
        clickmaps = all_clickmaps[i]
        n = len(clickmaps)
        
        for _ in range(n_iterations):
            rand_perm = np.random.permutation(n)
            fh = rand_perm[:(n // 2)]
            sh = rand_perm[(n // 2):]
            
            # Create the test and reference maps
            test_map = clickmaps[fh].mean(0)
            reference_map = clickmaps[sh].mean(0)
            
            # Normalize maps
            test_map = (test_map - test_map.min()) / (test_map.max() - test_map.min() + 1e-10)
            reference_map = (reference_map - reference_map.min()) / (reference_map.max() - reference_map.min() + 1e-10)
            
            # Add to batch for GPU processing
            test_maps_batch.append(test_map)
            reference_maps_batch.append(reference_map)
    
    # Process all correlations in one GPU batch for maximum efficiency
    if test_maps_batch:
        scores = utils.batch_compute_correlations_gpu(
            test_maps=test_maps_batch,
            reference_maps=reference_maps_batch,
            metric=metric,
            device=device
        )
        
        # Reshape scores back to match the original batch structure
        idx = 0
        for i in batch_indices:
            batch_results.append(np.mean(scores[idx:idx+n_iterations]))
            idx += n_iterations
    
    return batch_results


def compute_null_correlation_batch(batch_index, num_samples, all_clickmaps, click_len, metric, device='cuda'):
    """Compute a batch of null correlations"""
    # For Spearman, use scipy directly
    if metric.lower() == "spearman":
        inner_correlations = []
        for _ in range(num_samples):
            for i in range(click_len):
                selected_clickmaps = all_clickmaps[i]
                tmp_rng = np.arange(click_len)
                j = tmp_rng[~np.in1d(tmp_rng, i)]
                j = j[np.random.permutation(len(j))][0]  # Select a random other image
                other_clickmaps = all_clickmaps[j]
                
                rand_perm_sel = np.random.permutation(len(selected_clickmaps))
                rand_perm_other = np.random.permutation(len(other_clickmaps))
                fh = rand_perm_sel[:(len(selected_clickmaps) // 2)]
                sh = rand_perm_other[(len(other_clickmaps) // 2):]
                
                # Create test and reference maps
                test_map = selected_clickmaps[fh].mean(0)
                reference_map = other_clickmaps[sh].mean(0)
                
                # Normalize maps
                test_map = (test_map - test_map.min()) / (test_map.max() - test_map.min() + 1e-10)
                reference_map = (reference_map - reference_map.min()) / (reference_map.max() - reference_map.min() + 1e-10)
                
                # Use scipy's spearman correlation
                correlation, _ = spearmanr(test_map.flatten(), reference_map.flatten())
                inner_correlations.append(correlation)
        return inner_correlations
    
    # For other metrics, use GPU acceleration
    # Lists to store test and reference maps for batch GPU processing
    test_maps_batch = []
    reference_maps_batch = []
    
    for _ in range(num_samples):
        for i in range(click_len):
            selected_clickmaps = all_clickmaps[i]
            tmp_rng = np.arange(click_len)
            j = tmp_rng[~np.in1d(tmp_rng, i)]
            j = j[np.random.permutation(len(j))][0]  # Select a random other image
            other_clickmaps = all_clickmaps[j]
            
            rand_perm_sel = np.random.permutation(len(selected_clickmaps))
            rand_perm_other = np.random.permutation(len(other_clickmaps))
            fh = rand_perm_sel[:(len(selected_clickmaps) // 2)]
            sh = rand_perm_other[(len(other_clickmaps) // 2):]
            
            # Create test and reference maps
            test_map = selected_clickmaps[fh].mean(0)
            reference_map = other_clickmaps[sh].mean(0)
            
            # Normalize maps
            test_map = (test_map - test_map.min()) / (test_map.max() - test_map.min() + 1e-10)
            reference_map = (reference_map - reference_map.min()) / (reference_map.max() - reference_map.min() + 1e-10)
            
            # Add to batch for GPU processing
            test_maps_batch.append(test_map)
            reference_maps_batch.append(reference_map)
    
    # Process all correlations in one GPU batch for maximum efficiency
    if test_maps_batch:
        return utils.batch_compute_correlations_gpu(
            test_maps=test_maps_batch,
            reference_maps=reference_maps_batch,
            metric=metric,
            device=device
        )
    
    return []


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
        n_jobs=-1,
        gpu_batch_size=64,
        correlation_batch_size=16,
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
        n_jobs (int): Number of parallel jobs for CPU operations.
        gpu_batch_size (int): Batch size for GPU operations.
        correlation_batch_size (int): Number of correlation computations per batch.

    Returns:
        tuple: A tuple containing two elements:
            - dict: Category-wise mean correlations.
            - list: All individual image correlations.
    """

    assert blur_sigma_function is not None, "Blur sigma function needs to be provided."

    # Check if GPU is available
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Setting batch size to {gpu_batch_size} for GPU operations")
    else:
        device = 'cpu'
        print("No GPU detected, using CPU for processing")
        gpu_batch_size = 16  # Smaller batch size for CPU

    # Process files
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

    # Prepare maps using GPU acceleration by default
    print(f"Using GPU-accelerated blurring (batch_size={gpu_batch_size}, n_jobs={n_jobs})...")
    
    # Wrap clickmaps in a list as expected by prepare_maps_with_gpu_batching
    final_clickmaps, all_clickmaps, categories, _ = utils.prepare_maps_with_gpu_batching(
        final_clickmaps=[clickmaps],  # Wrap in list to match expected format
        blur_size=blur_size,
        blur_sigma=blur_sigma,
        image_shape=image_shape,
        min_pixels=min_pixels,
        min_subjects=min_subjects,
        metadata=metadata,
        blur_sigma_function=blur_sigma_function,
        center_crop=center_crop,
        batch_size=gpu_batch_size,
        device=device,
        n_jobs=n_jobs)
        
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

    # Compute scores through split-halfs
    # Optimize by processing in batches for better parallelization
    print(f"Computing split-half correlations in parallel (n_jobs={n_jobs}, batch_size={correlation_batch_size})...")
    num_clickmaps = len(all_clickmaps)
    
    # Prepare batches for correlation computation
    indices = list(range(num_clickmaps))
    batches = [indices[i:i+correlation_batch_size] for i in range(0, len(indices), correlation_batch_size)]
    
    # Process correlation batches in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_correlation_batch)(
            batch_indices=batch,
            all_clickmaps=all_clickmaps,
            metric=metric,
            n_iterations=randomization_iters,
            device=device
        ) for batch in tqdm(batches, desc="Computing split-half correlations")
    )
    
    # Flatten the results
    all_correlations = []
    for batch_result in results:
        all_correlations.extend(batch_result)
    all_correlations = np.asarray(all_correlations)

    # Compute null scores in batches for better parallelization
    print(f"Computing null correlations in parallel (n_jobs={n_jobs})...")
    
    # Calculate samples per batch to distribute work evenly
    click_len = len(all_clickmaps)
    samples_per_batch = max(1, null_iterations // n_jobs)
    num_batches = null_iterations // samples_per_batch + (1 if null_iterations % samples_per_batch else 0)
    
    # Create a batch for each parallel job
    null_batches = [(i, samples_per_batch if i < num_batches-1 else null_iterations - i*samples_per_batch) 
                    for i in range(num_batches)]
    
    # Process null correlation batches in parallel
    null_results = Parallel(n_jobs=min(n_jobs, len(null_batches)))(
        delayed(compute_null_correlation_batch)(
            batch_index=batch_idx,
            num_samples=num_samples,
            all_clickmaps=all_clickmaps,
            click_len=click_len,
            metric=metric,
            device=device
        ) for batch_idx, num_samples in tqdm(null_batches, desc="Computing null correlations")
    )
    
    # Combine null correlation results
    null_correlations = []
    for batch_result in null_results:
        null_correlations.extend(batch_result)
    
    # Compute means for final results
    null_correlations = np.array([np.nanmean(null_correlations)])
    
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

    # Get parallelization settings
    n_jobs = config.get("n_jobs", -1)  # Default to all cores
    gpu_batch_size = config.get("gpu_batch_size", 64)  # Larger default batch size
    correlation_batch_size = config.get("correlation_batch_size", 16)  # Batch size for correlation computations

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
        file_exclusion_filter=config["file_exclusion_filter"],
        n_jobs=n_jobs,
        gpu_batch_size=gpu_batch_size,
        correlation_batch_size=correlation_batch_size
    )
    print(f"Mean human correlation full set: {np.nanmean(all_correlations)}")
    print(f"Null correlations full set: {np.nanmean(null_correlations)}")
    np.savez(
        os.path.join(output_dir, "human_ceiling_split_half_{}.npz".format(config["experiment_name"])),
        final_clickmaps=final_clickmaps,
        ceiling_correlations=all_correlations,
        null_correlations=null_correlations,
    )

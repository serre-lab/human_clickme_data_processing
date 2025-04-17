import os, sys
import numpy as np
from PIL import Image
from src import utils
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
from scipy.stats import spearmanr
import argparse
import gc


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
        try:
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
        except Exception as e:
            print(f"GPU processing error: {e}. Falling back to CPU...")
            # Fallback to CPU processing in case of GPU errors
            device = 'cpu'
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
        try:
            return utils.batch_compute_correlations_gpu(
                test_maps=test_maps_batch,
                reference_maps=reference_maps_batch,
                metric=metric,
                device=device
            )
        except Exception as e:
            print(f"GPU processing error in null correlations: {e}. Falling back to CPU...")
            # Fallback to CPU processing in case of GPU errors
            device = 'cpu'
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
        gpu_batch_size=1024,
        correlation_batch_size=1024,
        use_gpu=True,
        max_retries=2,
        chunk_size=None,
        verbose=False,
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
        use_gpu (bool): Whether to use GPU acceleration.
        max_retries (int): Maximum number of retries for GPU operations before falling back to CPU.
        chunk_size (int): Size of chunks to process at once to reduce memory usage.
        verbose (bool): Whether to show detailed progress information.

    Returns:
        tuple: A tuple containing two elements:
            - dict: Category-wise mean correlations.
            - list: All individual image correlations.
    """

    assert blur_sigma_function is not None, "Blur sigma function needs to be provided."

    # Import required functions
    try:
        from src import cython_utils
        create_clickmap_func = cython_utils.create_clickmap_fast
        fast_duplicate_detection = cython_utils.fast_duplicate_detection
        print("Using Cython-optimized functions")
    except (ImportError, ModuleNotFoundError) as e:
        from src import python_utils
        create_clickmap_func = python_utils.create_clickmap_fast
        fast_duplicate_detection = python_utils.fast_duplicate_detection
        print(f"Cython modules not available: {e}")
        print("Falling back to Python implementation. For best performance, run 'python compile_cython.py build_ext --inplace' first.")

    # Check if GPU is available and requested
    if torch.cuda.is_available() and use_gpu:
        device = 'cuda'
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Setting batch size to {gpu_batch_size} for GPU operations")
    else:
        device = 'cpu'
        if not torch.cuda.is_available() and use_gpu:
            print("No GPU detected, using CPU for processing")
        elif not use_gpu:
            print("GPU usage disabled, using CPU for processing")
        gpu_batch_size = min(128, gpu_batch_size)  # Smaller batch size for CPU

    # Process files
    print("Processing clickmap files...")
    if hasattr(utils, "process_clickmap_files_parallel") and n_jobs > 1:
        process_clickmap_files = utils.process_clickmap_files_parallel
        print(f"Using parallel processing with {n_jobs} workers")
    else:
        process_clickmap_files = utils.process_clickmap_files
        print("Using sequential processing")
    
    clickmaps, ccounts = process_clickmap_files(
        clickme_data=clickme_data,
        image_path=clickme_image_folder,
        file_inclusion_filter=file_inclusion_filter,
        file_exclusion_filter=file_exclusion_filter,
        min_clicks=min_clicks,
        max_clicks=max_clicks,
        n_jobs=n_jobs)
    
    # Check dataset size and adjust parameters
    total_maps = len(clickmaps)
    print(f"Processing {total_maps} unique images")
    
    # Auto-adjust chunk size if not provided
    if chunk_size is None:
        if total_maps > 10000:
            chunk_size = 5000
            print(f"Large dataset detected. Using chunk size of {chunk_size}")
        else:
            chunk_size = total_maps  # Process all at once for smaller datasets
    
    # Filter classes if requested
    if class_filter_file:
        print("Filtering classes...")
        clickmaps = utils.filter_classes(
            clickmaps=clickmaps,
            class_filter_file=class_filter_file)
    
    # Filter participants if requested
    if participant_filter:
        print("Filtering participants...")
        clickmaps = utils.filter_participants(clickmaps)

    # Process in chunks if needed
    all_final_clickmaps = {}
    all_clickmaps_list = []
    
    # Determine number of chunks
    if chunk_size < total_maps:
        img_keys = list(clickmaps.keys())
        num_chunks = (total_maps + chunk_size - 1) // chunk_size
        print(f"Processing data in {num_chunks} chunks")
        
        with tqdm(total=num_chunks, desc="Processing chunks", position=0) as chunk_pbar:
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, total_maps)
                
                if verbose:
                    print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} ({chunk_start}-{chunk_end})")
                
                # Get chunk of clickmaps
                chunk_keys = img_keys[chunk_start:chunk_end]
                chunk_clickmaps = {k: clickmaps[k] for k in chunk_keys}
                
                # Process chunk
                retry_count = 0
                success = False
                
                while retry_count <= max_retries and not success:
                    try:
                        # Use GPU-optimized batched processing
                        chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.prepare_maps_with_gpu_batching(
                            final_clickmaps=[chunk_clickmaps],  # Wrap in list to match expected format
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
                            n_jobs=n_jobs,
                            create_clickmap_func=create_clickmap_func,
                            fast_duplicate_detection=fast_duplicate_detection,
                            verbose=verbose)
                        
                        success = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            print(f"Error processing chunk after {max_retries} retries: {e}")
                            print("Falling back to CPU processing...")
                            try:
                                # Fall back to CPU processing
                                chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.prepare_maps_with_progress(
                                    final_clickmaps=[chunk_clickmaps],
                                    blur_size=blur_size,
                                    blur_sigma=blur_sigma,
                                    image_shape=image_shape,
                                    min_pixels=min_pixels,
                                    min_subjects=min_subjects,
                                    metadata=metadata,
                                    blur_sigma_function=blur_sigma_function,
                                    center_crop=center_crop,
                                    n_jobs=n_jobs,
                                    create_clickmap_func=create_clickmap_func,
                                    fast_duplicate_detection=fast_duplicate_detection)
                                success = True
                            except Exception as e2:
                                print(f"CPU processing also failed: {e2}")
                                print("Skipping this chunk...")
                                chunk_all_clickmaps = []
                                chunk_final_clickmaps = {}
                                break
                        else:
                            print(f"Error processing chunk: {e}")
                            print(f"Retry {retry_count}/{max_retries} with smaller batch size...")
                            # Reduce batch size for retry
                            gpu_batch_size = max(256, gpu_batch_size // 2)
                
                # Filter for foreground mask overlap if requested
                if success and mask_dir and chunk_all_clickmaps:
                    masks = utils.load_masks(mask_dir)
                    chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.filter_for_foreground_masks(
                        final_clickmaps=chunk_final_clickmaps,
                        all_clickmaps=chunk_all_clickmaps,
                        categories=chunk_categories,
                        masks=masks,
                        mask_threshold=mask_threshold)
                
                # Add to overall results
                all_final_clickmaps.update(chunk_final_clickmaps)
                all_clickmaps_list.extend(chunk_all_clickmaps)
                
                # Free memory
                del chunk_clickmaps, chunk_final_clickmaps, chunk_categories, chunk_final_keep_index
                gc.collect()
                
                # Update progress
                chunk_pbar.update(1)
        
        # Set all_clickmaps to the combined list
        all_clickmaps = all_clickmaps_list
    else:
        # Process all data at once
        print("Processing all data at once...")
        try:
            if device == 'cuda':
                print(f"Using GPU-accelerated blurring (batch_size={gpu_batch_size}, n_jobs={n_jobs})...")
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
                    n_jobs=n_jobs,
                    create_clickmap_func=create_clickmap_func,
                    fast_duplicate_detection=fast_duplicate_detection,
                    verbose=verbose)
            else:
                print(f"Using CPU processing (n_jobs={n_jobs})...")
                final_clickmaps, all_clickmaps, categories, _ = utils.prepare_maps_with_progress(
                    final_clickmaps=[clickmaps],
                    blur_size=blur_size,
                    blur_sigma=blur_sigma,
                    image_shape=image_shape,
                    min_pixels=min_pixels,
                    min_subjects=min_subjects,
                    metadata=metadata,
                    blur_sigma_function=blur_sigma_function,
                    center_crop=center_crop,
                    n_jobs=n_jobs,
                    create_clickmap_func=create_clickmap_func,
                    fast_duplicate_detection=fast_duplicate_detection)
        except Exception as e:
            print(f"Error during processing: {e}")
            print("Trying CPU fallback...")
            final_clickmaps, all_clickmaps, categories, _ = utils.prepare_maps_with_progress(
                final_clickmaps=[clickmaps],
                blur_size=blur_size,
                blur_sigma=blur_sigma,
                image_shape=image_shape,
                min_pixels=min_pixels,
                min_subjects=min_subjects,
                metadata=metadata,
                blur_sigma_function=blur_sigma_function,
                center_crop=center_crop,
                n_jobs=n_jobs,
                create_clickmap_func=create_clickmap_func,
                fast_duplicate_detection=fast_duplicate_detection)
        
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
        for imn in range(min(len(final_clickmaps), 5)):  # Limit to 5 images for debugging
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
    # Add command line arguments
    parser = argparse.ArgumentParser(description="Compute human ceiling via split-half correlations")
    parser.add_argument('config', nargs='?', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualizations')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU processing')
    parser.add_argument('--gpu-batch-size', type=int, help='Override GPU batch size')
    parser.add_argument('--correlation-batch-size', type=int, help='Override correlation batch size')
    parser.add_argument('--n-jobs', type=int, help='Number of parallel jobs')
    parser.add_argument('--chunk-size', type=int, help='Size of chunks to process')
    args = parser.parse_args()

    # Get config file
    if args.config:
        config_file = args.config
    else:
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

    # Override config with command line arguments if provided
    if args.debug:
        config["debug"] = True
    
    if args.n_jobs:
        config["n_jobs"] = args.n_jobs
    elif "n_jobs" not in config:
        config["n_jobs"] = -1  # Default to all cores
    
    if args.gpu_batch_size:
        config["gpu_batch_size"] = args.gpu_batch_size
    elif "gpu_batch_size" not in config:
        config["gpu_batch_size"] = 1024  # Default batch size
    
    if args.correlation_batch_size:
        config["correlation_batch_size"] = args.correlation_batch_size
    elif "correlation_batch_size" not in config:
        config["correlation_batch_size"] = 1024  # Default correlation batch size
    
    # Set CPU/GPU mode
    config["use_gpu"] = not args.cpu_only
    
    # Set chunk size if provided
    if args.chunk_size:
        config["chunk_size"] = args.chunk_size
    elif "chunk_size" not in config:
        config["chunk_size"] = None  # Will be auto-determined based on dataset size
    
    # Set verbose mode
    config["verbose"] = args.verbose
    
    # Set retry count if not in config
    if "max_retries" not in config:
        config["max_retries"] = 2

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
        n_jobs=config["n_jobs"],
        gpu_batch_size=config["gpu_batch_size"],
        correlation_batch_size=config["correlation_batch_size"],
        use_gpu=config["use_gpu"],
        max_retries=config["max_retries"],
        chunk_size=config.get("chunk_size"),
        verbose=config["verbose"],
    )
    print(f"Mean human correlation full set: {np.nanmean(all_correlations)}")
    print(f"Null correlations full set: {np.nanmean(null_correlations)}")
    np.savez(
        os.path.join(output_dir, "human_ceiling_split_half_{}.npz".format(config["experiment_name"])),
        final_clickmaps=final_clickmaps,
        ceiling_correlations=all_correlations,
        null_correlations=null_correlations,
    )

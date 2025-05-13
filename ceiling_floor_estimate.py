import os, sys
import numpy as np
from PIL import Image
import json
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from src import utils
from tqdm import tqdm
import h5py
import gc
import torch
from joblib import Parallel, delayed


def compute_correlation_batch(batch_indices, all_clickmaps, metric, n_iterations=10, device='cuda'):
    """Compute split-half correlations for a batch of clickmaps in parallel"""
    batch_results = []
    
    # Handle Spearman correlation separately if that's the metric
    if metric.lower() == "spearman":
        for i in batch_indices:
            clickmaps = all_clickmaps[i]
            import pdb; pdb.set_trace()
            levels = len(clickmaps)  # What is K for the thresholds
            level_corrs = []
            for clickmap_at_k in clickmaps:
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
                j = tmp_rng[~np.isin(tmp_rng, [i])]  # Select indices not equal to i
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
            j = tmp_rng[~np.isin(tmp_rng, [i])]  # Select indices not equal to i
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


if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(description="Process clickme data for modeling")
    parser.add_argument('config', nargs='?', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable additional debug output')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress for GPU processing')
    parser.add_argument('--gpu-batch-size', type=int, default=None, help='Override GPU batch size')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of CPU workers')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
    parser.add_argument('--filter-duplicates', action='store_false', help='Filter duplicate participant submissions, keeping only the first submission per image')
    args = parser.parse_args()
    
    # Start profiling if requested
    if args.profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    
    # Load config file
    if args.config:
        config_file = args.config if "configs" + os.path.sep in args.config else os.path.join("configs", args.config)
        assert os.path.exists(config_file), f"Cannot find config file: {config_file}"
        config = utils.process_config(config_file)
    else:
        config_file = utils.get_config(sys.argv)
        config = utils.process_config(config_file)
    
    # Add filter_duplicates to config if not present
    if "filter_duplicates" not in config:
        config["filter_duplicates"] = args.filter_duplicates
    
    # Load clickme data
    print(f"Loading clickme data...")
    clickme_data = utils.process_clickme_data(
        config["clickme_data"],
        config["filter_mobile"])
    total_maps = len(clickme_data)

    # Apply duplicate filtering if requested
    if config["filter_duplicates"] or args.filter_duplicates:
        clickme_data = utils.filter_duplicate_participants(clickme_data)
        total_maps = len(clickme_data)
    
    # Validate clickme data structure
    print(f"Validating clickme data structure for {total_maps} maps...")
    image_paths = clickme_data['image_path'].unique()
    total_unique_images = len(image_paths)
    print(f"Found {total_unique_images} unique images")
    
    # Set up GPU configuration
    if args.gpu_batch_size:
        config["gpu_batch_size"] = args.gpu_batch_size
    else:
        config["gpu_batch_size"] = 4096
    
    # Optimize number of workers based on CPU count
    cpu_count = os.cpu_count()
    if args.max_workers:
        config["n_jobs"] = min(args.max_workers, cpu_count)
    else:
        # Leave some cores free for system operations
        config["n_jobs"] = max(1, min(cpu_count - 1, 8))
    
    # Verify GPU is available
    config["use_gpu_blurring"] = torch.cuda.is_available()
    if config["use_gpu_blurring"]:
        # Print GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
    else:
        print("GPU not available, exiting.")
        sys.exit(1)
    
    # Set up output format
    if "output_format" not in config or config["output_format"] == "auto":
        config["output_format"] = "hdf5" if total_maps > 100000 else "numpy"
    output_format = config["output_format"]
    
    # Ensure all directories exist
    output_dir = config["assets"]
    image_output_dir = config["example_image_output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, config["experiment_name"]), exist_ok=True)
    
    # Create dedicated directory for click counts
    click_counts_dir = os.path.join(output_dir, f"{config['experiment_name']}_click_counts")
    os.makedirs(click_counts_dir, exist_ok=True)
    
    # Original code for non-HDF5 format
    hdf5_path = os.path.join(output_dir, f"{config['experiment_name']}.h5")
    print(f"Saving results to file: {hdf5_path}")
    with h5py.File(hdf5_path, 'w') as f:
        f.create_group("clickmaps")
        f.create_group("click_counts")  # Add group for click counts
        meta_grp = f.create_group("metadata")
        meta_grp.attrs["total_unique_images"] = total_unique_images
        meta_grp.attrs["total_maps"] = total_maps
        meta_grp.attrs["filter_duplicates"] = np.bytes_("True" if config["filter_duplicates"] else "False")
        meta_grp.attrs["creation_date"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Print optimization settings
    print("\nProcessing settings:")
    print(f"- Dataset size: {total_maps} maps, {total_unique_images} images")
    print(f"- GPU batch size: {config['gpu_batch_size']}")
    print(f"- CPU workers: {config['n_jobs']}")
    print(f"- Output format: {config['output_format']}")
    print(f"- Filter duplicates: {config['filter_duplicates']}")
    print(f"- Memory usage at start: {utils.get_memory_usage():.2f} MB\n")
    
    # Choose processing method (compiled Cython vs. Python)
    use_cython = config.get("use_cython", True)
    if use_cython:
        try:
            from src import cython_utils
            create_clickmap_func = cython_utils.create_clickmap_fast
            fast_duplicate_detection = cython_utils.fast_duplicate_detection
            fast_ious_binary = cython_utils.fast_ious_binary
            print("Using Cython-optimized functions")
        except (ImportError, ModuleNotFoundError) as e:
            use_cython = False
            from src import python_utils
            create_clickmap_func = python_utils.create_clickmap_fast
            fast_duplicate_detection = python_utils.fast_duplicate_detection
            fast_ious_binary = python_utils.fast_ious_binary
            print(f"Cython modules not available: {e}")
            print("Falling back to Python implementation. For best performance, run 'python compile_cython.py build_ext --inplace' first.")
    else:
        from src import python_utils
        create_clickmap_func = python_utils.create_clickmap_fast
        fast_duplicate_detection = python_utils.fast_duplicate_detection
        fast_ious_binary = python_utils.fast_ious_binary

    # Load metadata
    if config["metadata_file"]:
        metadata = np.load(config["metadata_file"], allow_pickle=True).item()
    else:
        metadata = None

    print("Processing clickme data...")
    # Always use parallel processing for large datasets
    clickmaps, ccounts = utils.process_clickmap_files_parallel(
        clickme_data=clickme_data,
        image_path=config["image_path"],
        file_inclusion_filter=config["file_inclusion_filter"],
        file_exclusion_filter=config["file_exclusion_filter"],
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"],
        n_jobs=config["n_jobs"])
    
    # Apply filters if necessary
    if config["class_filter_file"]:
        print("Filtering classes...")
        clickmaps = utils.filter_classes(
            clickmaps=clickmaps,
            class_filter_file=config["class_filter_file"])
    
    if config["participant_filter"]:
        print("Filtering participants...")
        clickmaps = utils.filter_participants(clickmaps)
    
    # Process all maps with our new single-batch GPU function
    print(f"Processing with GPU (batch size: {config['gpu_batch_size']})...")
    if 1:  # config.get("multi_thresh_gpu", False):
        process_fun = utils.process_all_maps_multi_thresh_gpu
    # else:
    #     process_fun = utils.process_all_maps_gpu
    final_clickmaps, all_clickmaps, categories, final_keep_index, click_counts = process_fun(
        clickmaps=clickmaps,
        config=config,
        metadata=metadata,
        create_clickmap_func=create_clickmap_func,
        fast_duplicate_detection=fast_duplicate_detection,
        return_before_blur=True,
        average_maps=False,
    )
    # Apply mask filtering if needed
    if final_keep_index and config["mask_dir"]:
        print("Applying mask filtering...")
        masks = utils.load_masks(config["mask_dir"])
        final_clickmaps, all_clickmaps, categories, final_keep_index = utils.filter_for_foreground_masks(
            final_clickmaps=final_clickmaps,
            all_clickmaps=all_clickmaps,
            categories=categories,
            masks=masks,
            mask_threshold=config["mask_threshold"])
        # Update click counts to match filtered images
        click_counts = {k: click_counts[k] for k in final_keep_index if k in click_counts}
        
    # Convert all_clickmaps to the format expected by the correlation code
    image_shape = config["image_shape"]
    correlation_batch_size = config["correlation_batch_size"]
    null_iterations = config["null_iterations"]
    metric = config["metric"]
    n_jobs = config["n_jobs"]
    gpu_batch_size = config["gpu_batch_size"]

    # Check if GPU is available
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Setting batch size to {gpu_batch_size} for GPU operations")
    else:
        device = 'cpu'
        print("No GPU detected, using CPU for processing")
        gpu_batch_size = 16  # Smaller batch size for CPU
    print(f"Converting clickmaps for correlation analysis...")
    
    # Compute scores through split-halfs
    # Optimize by processing in batches for better parallelization
    print(f"Computing split-half correlations in parallel (n_jobs={n_jobs}, batch_size={correlation_batch_size})...")
    num_clickmaps = len(all_clickmaps)
    
    # Prepare batches for correlation computation
    indices = list(range(num_clickmaps))
    batches = [indices[i:i+correlation_batch_size] for i in range(0, len(indices), correlation_batch_size)]
    
    # Process correlation batches in parallel
    n_jobs=1
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_correlation_batch)(
            batch_indices=batch,
            all_clickmaps=all_clickmaps,
            metric=metric,
            n_iterations=null_iterations,
            device=device
        ) for batch in tqdm(batches, desc="Computing split-half correlations")
    )
    
    # Flatten the results
    all_correlations = []
    for batch_result in results:
        all_correlations.extend(batch_result)
    all_correlations = np.asarray(all_correlations)



    # End profiling if it was enabled
    if args.profile:
        profiler.disable()
        import pstats
        from io import StringIO
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(30)  # Print top 30 functions by time
        print(s.getvalue())
        
        # Save profile results to file
        ps.dump_stats(os.path.join(output_dir, "profile_results.prof"))
        print(f"Profile results saved to {os.path.join(output_dir, 'profile_results.prof')}")
    
    print(f"\nProcessing complete! Final memory usage: {utils.get_memory_usage():.2f} MB")

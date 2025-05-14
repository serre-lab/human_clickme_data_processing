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
from scipy.stats import spearmanr


def auc(test_map, reference_map, thresholds=100):
    """Compute the area under the IOU curve for a test map and a reference map"""
    ious = []
    # Create evenly spaced thresholds from 0 to max value
    test_thresholds = np.linspace(0, test_map.max(), thresholds)
    reference_thresholds = np.linspace(0, reference_map.max(), thresholds)
    
    # Calculate IOU at each threshold pair
    for test_threshold, reference_threshold in zip(test_thresholds, reference_thresholds):
        test_binary = test_map > test_threshold
        ref_binary = reference_map > reference_threshold
        intersection = np.sum(np.logical_and(test_binary, ref_binary))
        union = np.sum(np.logical_or(test_binary, ref_binary))
        iou = intersection / union if union > 0 else 0.0
        ious.append(iou)
    
    # Return the area under the curve (trapezoidal integration)
    # We're integrating over normalized threshold range [0,1]
    return np.trapz(ious, dx=1.0/thresholds) if thresholds > 1 else np.mean(ious)


def compute_correlation_batch(batch_indices, all_clickmaps, metric="auc", n_iterations=10, device='cuda', blur_size=11, blur_sigma=1.5, floor=False):
    """Compute split-half correlations for a batch of clickmaps in parallel"""
    batch_results = []
    for i in batch_indices:
        clickmaps = all_clickmaps[i]
        level_corrs = []
        if floor:
            rand_i = np.random.choice([j for j in range(len(all_clickmaps)) if j != i])
        for k, clickmap_at_k in enumerate(clickmaps):
            rand_corrs = []
            n = len(clickmap_at_k)
            for _ in range(n_iterations):
                rand_perm = np.random.permutation(n)
                fh = rand_perm[:(n // 2)]
                sh = rand_perm[(n // 2):]
                
                # Create the test and reference maps
                test_map = clickmap_at_k[fh].mean(0)
                if floor:
                    rand_perm = np.random.permutation(len(all_clickmaps[rand_i][k]))
                    sh = rand_perm[(n // 2):]
                    reference_map = all_clickmaps[rand_i][k][sh].mean(0)  # Take maps from the same level in a random other image
                    reference_map = utils.blur_maps_for_cf(
                        reference_map[None, None],
                        blur_size,
                        blur_sigma,
                        gpu_batch_size=1).squeeze()
                    test_map = utils.blur_maps_for_cf(
                        test_map[None, None],
                        blur_size,
                        blur_sigma,
                        gpu_batch_size=1).squeeze()
                else:
                    reference_map = clickmap_at_k[sh].mean(0)

                    # Make maps for each
                    blur_clickmaps = utils.blur_maps_for_cf(
                        np.stack((test_map, reference_map), axis=0)[None],
                        blur_size,
                        blur_sigma,
                        gpu_batch_size=2).squeeze()
                    test_map = blur_clickmaps[0]
                    reference_map = blur_clickmaps[1]
                
                # Use scipy's spearman correlation
                if metric == "auc":
                    score = auc(test_map.flatten(), reference_map.flatten())
                else:
                    score, _ = spearmanr(test_map.flatten(), reference_map.flatten())
                rand_corrs.append(score)
            rand_corrs = np.asarray(rand_corrs).mean()  # Take the mean of the random correlations
            level_corrs.append(rand_corrs)
        batch_results.append(np.asarray(level_corrs).mean())  # Integrate over the levels
    return batch_results


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
    ceiling_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_correlation_batch)(
            batch_indices=batch,
            all_clickmaps=all_clickmaps,
            metric=metric,
            n_iterations=null_iterations,
            device=device,
            blur_size=config["blur_size"],
            blur_sigma=config.get("blur_sigma", config["blur_size"])
        ) for batch in tqdm(batches, desc="Computing ceilings")
    )
    floor_results = Parallel(n_jobs=n_jobs)(
        delayed(compute_correlation_batch)(
            batch_indices=batch,
            all_clickmaps=all_clickmaps,
            metric=metric,
            n_iterations=null_iterations,
            device=device,
            floor=True,
            blur_size=config["blur_size"],
            blur_sigma=config.get("blur_sigma", config["blur_size"])
        ) for batch in tqdm(batches, desc="Computing floors")
    )
    
    # Flatten the results
    all_ceilings = np.concatenate(ceiling_results)
    all_floors = np.concatenate(floor_results)

    # Compute the mean of the ceilings and floors
    mean_ceiling = all_ceilings.mean()
    mean_floor = all_floors.mean()

    # Compute the ratio of the mean of the ceilings to the mean of the floors
    ratio = mean_ceiling / mean_floor
    print(f"Mean ceiling: {mean_ceiling}, Mean floor: {mean_floor}, Ratio: {ratio}")

    # Save the results
    np.savez(
        os.path.join(output_dir, f"{config['experiment_name']}_ceiling_floor_results.npz"),
        mean_ceiling=mean_ceiling,
        mean_floor=mean_floor,
        all_ceilings=all_ceilings,
        all_floors=all_floors,
        ratio=ratio)

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

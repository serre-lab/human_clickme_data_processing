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
import psutil
import torch
from scipy.stats import spearmanr


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_medians(clickmaps, mode, thresh=95):
    """Compute median number of clicks in each category"""
    medians = {}
    if mode == 'all':
        csize = [len(x) for x in clickmaps.values()]
        if not csize:
            return {}
        medians['median'] = int(np.median(csize))
        medians['mean'] = float(np.mean(csize))
        medians['threshold'] = int(np.percentile(csize, thresh))
    elif mode == 'image':
        medians['image'] = {}
        for img_name, clicks in clickmaps.items():
            medians['image'][img_name] = int(np.median([len(x) for x in clicks]))
    elif mode == 'category':
        medians['category'] = {}
        cats = {}
        for img_name, clicks in clickmaps.items():
            img_category = img_name.split('/')[0]
            if img_category not in cats:
                cats[img_category] = []
            cats[img_category].append(len(clicks))
        for category, counts in cats.items():
            medians['category'][category] = int(np.median(counts))
    return medians


def split_clickmaps(clickmaps):
    """Split clickmaps for each image into two random halves"""
    first_half = {}
    second_half = {}
    
    for img_name, trials in clickmaps.items():
        if len(trials) < 2:  # Need at least 2 trials to split
            continue
            
        # Randomly shuffle the trials
        shuffled_trials = trials.copy()
        np.random.shuffle(shuffled_trials)
        
        # Split into two equal halves (or as equal as possible)
        mid_point = len(shuffled_trials) // 2
        first_half[img_name] = shuffled_trials[:mid_point]
        second_half[img_name] = shuffled_trials[mid_point:]
    
    return first_half, second_half


def calculate_spearman_correlation(map1, map2):
    """Calculate Spearman correlation between two 2D maps"""
    # Flatten maps
    flat1 = map1.flatten()
    flat2 = map2.flatten()
    
    # Calculate Spearman correlation
    corr, _ = spearmanr(flat1, flat2)
    
    return corr


def process_split_maps_gpu(clickmaps, config, metadata=None, create_clickmap_func=None, fast_duplicate_detection=None):
    """
    Process a split of clickmaps on GPU in batches
    """
    # Extract basic parameters
    blur_size = config["blur_size"]
    blur_sigma = config.get("blur_sigma", blur_size)
    image_shape = config["image_shape"]
    min_subjects = max(2, config["min_subjects"] // 2)  # Adjust minimum subjects for half the data
    min_clicks = config["min_clicks"]
    
    # Get GPU batch size for processing
    gpu_batch_size = config.get("gpu_batch_size", 4096)
    
    print(f"Processing {len(clickmaps)} unique images with GPU (batch size: {gpu_batch_size})...")
    
    # Prepare data structures
    all_clickmaps = []
    keep_index = []
    categories = []
    final_clickmaps = {}
    
    # Preprocess all clickmaps first to binary maps
    for key, trials in tqdm(clickmaps.items(), desc="Creating binary maps"):
        if len(trials) < min_subjects:
            continue
            
        # Create binary clickmaps
        if metadata and key in metadata:
            native_size = metadata[key]
            binary_maps = np.asarray([create_clickmap_func([trial], native_size[::-1]) for trial in trials])
        else:
            binary_maps = np.asarray([create_clickmap_func([trial], tuple(image_shape)) for trial in trials])
        
        # Only keep maps with enough valid pixels using mask
        mask = binary_maps.sum((-2, -1)) >= min_clicks
        binary_maps = binary_maps[mask]
        
        # If we have enough valid maps, average them and keep this image
        if len(binary_maps) >= min_subjects:
            all_clickmaps.append(np.array(binary_maps).mean(0, keepdims=True))
            categories.append(key.split("/")[0])
            keep_index.append(key)
            final_clickmaps[key] = trials
    
    if not all_clickmaps:
        print("No valid clickmaps to process")
        return {}, [], [], []
    
    # Prepare for batch blurring on GPU
    total_maps = len(all_clickmaps)
    print(f"Preparing to blur {total_maps} image clickmaps using GPU...")
    
    # Convert all maps to tensors
    all_tensors = [torch.from_numpy(maps).float() for maps in all_clickmaps]
    
    # Create circular kernel
    if blur_size % 2 == 0:
        adjusted_blur_size = blur_size + 1  # Ensure odd kernel size
    else:
        adjusted_blur_size = blur_size
        
    kernel = utils.circle_kernel(adjusted_blur_size, blur_sigma, 'cuda')
    
    # Process in batches based on the GPU batch size
    try:
        torch.cat(all_tensors[:1000])
        num_batches = (total_maps + gpu_batch_size - 1) // gpu_batch_size
    except Exception:
        # Pad all clickmaps to the same size, blur, then crop after.
        num_batches = total_maps
        gpu_batch_size = 1
    
    print(f"Processing in {num_batches} batches of up to {gpu_batch_size} maps each...")
    for batch_idx in tqdm(range(num_batches), desc="Processing GPU batches"):
        # Get batch indices
        start_idx = batch_idx * gpu_batch_size
        end_idx = min(start_idx + gpu_batch_size, total_maps)
        current_batch_size = end_idx - start_idx
        
        # Create batch tensor
        batch_tensors = all_tensors[start_idx:end_idx]
        batch_tensor = torch.cat(batch_tensors, dim=0).unsqueeze(1).to('cuda')
        
        # Apply blurring to this batch
        blurred_tensor = utils.convolve(batch_tensor, kernel, double_conv=True)
        
        # Convert back to numpy
        blurred_maps = blurred_tensor.squeeze(1).cpu().numpy()
        
        # Update results
        for i in range(current_batch_size):
            map_idx = start_idx + i
            all_clickmaps[map_idx] = blurred_maps[i:i+1]  # Keep the same shape with extra dimension
        
        # Clean up GPU memory for this batch
        del batch_tensor, blurred_tensor
        torch.cuda.empty_cache()
    
    # Clean up remaining GPU memory
    del kernel
    torch.cuda.empty_cache()
    
    print(f"Finished blurring {total_maps} clickmaps. Kept {len(keep_index)} images.")
    return final_clickmaps, all_clickmaps, categories, keep_index


def compute_ceiling_floor_estimates(clickmaps, config, K=20, metadata=None, create_clickmap_func=None, fast_duplicate_detection=None):
    """
    Compute ceiling and floor estimates by splitting clickmaps and calculating correlations
    """
    ceiling_corrs = []
    floor_corrs = []
    image_results = {}
    
    for k in tqdm(range(K), desc=f"Running {K} iterations of ceiling/floor estimation"):
        # Split clickmaps into two random halves
        first_half, second_half = split_clickmaps(clickmaps)
        
        # Process both halves
        import pdb; pdb.set_trace()
        _, first_maps, _, first_indices = process_split_maps_gpu(
            first_half, config, metadata, create_clickmap_func, fast_duplicate_detection)
        
        _, second_maps, _, second_indices = process_split_maps_gpu(
            second_half, config, metadata, create_clickmap_func, fast_duplicate_detection)
        
        # Find common images between the two halves
        common_indices = set(first_indices).intersection(set(second_indices))
        print(f"Iteration {k+1}/{K}: Found {len(common_indices)} common images between splits")
        
        if not common_indices:
            print(f"Warning: No common images in iteration {k+1}, skipping")
            continue
        
        # Calculate correlations for this iteration
        iteration_ceiling_corrs = []
        iteration_floor_corrs = []
        
        # Create copy of second half indices for shuffling
        shuffled_indices = second_indices.copy()
        np.random.shuffle(shuffled_indices)
        
        # Process each common image
        for i, img_name in enumerate(common_indices):
            # Find indices in each list
            first_idx = first_indices.index(img_name)
            second_idx = second_indices.index(img_name)
            
            # Get the processed maps
            map1 = first_maps[first_idx][0]  # First element of batch
            map2 = second_maps[second_idx][0]  # First element of batch
            
            # Calculate ceiling correlation (actual correlation between halves)
            ceiling_corr = calculate_spearman_correlation(map1, map2)
            iteration_ceiling_corrs.append(ceiling_corr)
            
            # For floor, use a random different image from second half
            random_idx = shuffled_indices.index(list(common_indices)[i])
            random_map_idx = second_indices.index(shuffled_indices[random_idx])
            map2_random = second_maps[random_map_idx][0]
            
            # Calculate floor correlation (correlation with random different image)
            floor_corr = calculate_spearman_correlation(map1, map2_random)
            iteration_floor_corrs.append(floor_corr)
            
            # Store per-image results
            if img_name not in image_results:
                image_results[img_name] = {"ceiling": [], "floor": []}
            
            image_results[img_name]["ceiling"].append(ceiling_corr)
            image_results[img_name]["floor"].append(floor_corr)
        
        # Add iteration averages to overall results
        if iteration_ceiling_corrs:
            ceiling_corrs.append(np.nanmean(iteration_ceiling_corrs))
        if iteration_floor_corrs:
            floor_corrs.append(np.nanmean(iteration_floor_corrs))
    
    # Calculate overall averages
    avg_ceiling = np.nanmean(ceiling_corrs)
    avg_floor = np.nanmean(floor_corrs)
    
    print(f"\nResults after {K} iterations:")
    print(f"Average ceiling correlation: {avg_ceiling:.4f}")
    print(f"Average floor correlation: {avg_floor:.4f}")
    
    # Prepare results for CSV
    results_df = pd.DataFrame({
        "image": list(image_results.keys()),
        "avg_ceiling": [np.nanmean(v["ceiling"]) for v in image_results.values()],
        "avg_floor": [np.nanmean(v["floor"]) for v in image_results.values()],
        "std_ceiling": [np.nanstd(v["ceiling"]) for v in image_results.values()],
        "std_floor": [np.nanstd(v["floor"]) for v in image_results.values()],
        "num_iterations": [len(v["ceiling"]) for v in image_results.values()]
    })
    
    # Add overall averages to the dataframe
    overall_df = pd.DataFrame({
        "image": ["OVERALL"],
        "avg_ceiling": [avg_ceiling],
        "avg_floor": [avg_floor],
        "std_ceiling": [np.nanstd(ceiling_corrs)],
        "std_floor": [np.nanstd(floor_corrs)],
        "num_iterations": [K]
    })
    
    results_df = pd.concat([results_df, overall_df], ignore_index=True)
    
    return results_df, ceiling_corrs, floor_corrs


if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(description="Calculate ceiling and floor estimates for clickme data")
    parser.add_argument('config', nargs='?', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable additional debug output')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress for GPU processing')
    parser.add_argument('--gpu-batch-size', type=int, default=None, help='Override GPU batch size')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of CPU workers')
    parser.add_argument('--iterations', '-K', type=int, default=20, help='Number of iterations for ceiling/floor calculation')
    args = parser.parse_args()
    
    # Load config file
    if args.config:
        config_file = args.config if "configs" + os.path.sep in args.config else os.path.join("configs", args.config)
        assert os.path.exists(config_file), f"Cannot find config file: {config_file}"
        config = utils.process_config(config_file)
    else:
        config_file = utils.get_config(sys.argv)
        config = utils.process_config(config_file)
    
    # Load clickme data
    print(f"Loading clickme data...")
    clickme_data = utils.process_clickme_data(
        config["clickme_data"],
        config["filter_mobile"])
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
    
    # Ensure all directories exist
    output_dir = config["assets"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, config["experiment_name"]), exist_ok=True)
    
    # Print optimization settings
    print("\nProcessing settings:")
    print(f"- Dataset size: {total_maps} maps, {total_unique_images} images")
    print(f"- GPU batch size: {config['gpu_batch_size']}")
    print(f"- CPU workers: {config['n_jobs']}")
    print(f"- Ceiling/floor iterations: {args.iterations}")
    print(f"- Memory usage at start: {get_memory_usage():.2f} MB\n")
    
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
    
    # Compute ceiling and floor estimates
    print(f"\nComputing ceiling and floor estimates with {args.iterations} iterations...")
    results_df, ceiling_corrs, floor_corrs = compute_ceiling_floor_estimates(
        clickmaps=clickmaps,
        config=config,
        K=args.iterations,
        metadata=metadata,
        create_clickmap_func=create_clickmap_func,
        fast_duplicate_detection=fast_duplicate_detection
    )
    
    # Save results to CSV
    csv_path = os.path.join(output_dir, f"{config['experiment_name']}_ceiling_floor_estimates.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([ceiling_corrs, floor_corrs], labels=['Ceiling', 'Floor'])
    plt.ylabel('Spearman Correlation')
    plt.title('Ceiling and Floor Estimates')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f"{config['experiment_name']}_ceiling_floor_plot.png")
    plt.savefig(plot_path)
    print(f"Summary plot saved to {plot_path}")
    
    print(f"\nProcessing complete! Final memory usage: {get_memory_usage():.2f} MB") 
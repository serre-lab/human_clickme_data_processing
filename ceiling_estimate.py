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


def split_clickmaps(clickmaps, min_subjects_per_half=2):
    """Split clickmaps for each image into two random halves, ensuring both halves have enough subjects"""
    first_half = {}
    second_half = {}
    
    # First pass: filter to keep only images with enough trials for both halves
    filtered_clickmaps = {}
    for img_name, trials in clickmaps.items():
        # Need at least 2*min_subjects_per_half trials to have enough for both halves
        if len(trials) >= 2*min_subjects_per_half:
            filtered_clickmaps[img_name] = trials
    
    print(f"Splitting {len(filtered_clickmaps)} images with sufficient trials (≥{2*min_subjects_per_half})")
    
    # Second pass: split the filtered clickmaps
    for img_name, trials in filtered_clickmaps.items():
        # Randomly shuffle the trials
        shuffled_trials = trials.copy()
        np.random.shuffle(shuffled_trials)
        
        # Split into two equal halves (or as equal as possible)
        mid_point = len(shuffled_trials) // 2
        first_half[img_name] = shuffled_trials[:mid_point]
        second_half[img_name] = shuffled_trials[mid_point:]
    
    return first_half, second_half, filtered_clickmaps.keys()


def apply_center_crop(image_map, crop_size):
    """
    Apply center cropping to an image map
    
    Parameters:
    -----------
    image_map : numpy.ndarray
        The image map to crop
    crop_size : list or tuple
        The dimensions of the crop [height, width]
        
    Returns:
    --------
    numpy.ndarray
        The center-cropped image map
    """
    # Get current dimensions
    if len(image_map.shape) == 2:  # 2D map
        h, w = image_map.shape
    elif len(image_map.shape) == 3:  # 3D map (with batch dimension)
        h, w = image_map.shape[1:]
    else:
        return image_map  # Return as-is if dimensions don't match expected
    
    # Calculate crop coordinates
    crop_h, crop_w = crop_size
    
    # If image is smaller than crop size, pad it
    if h < crop_h or w < crop_w:
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        
        if len(image_map.shape) == 2:
            padded = np.pad(image_map, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), 
                            mode='constant', constant_values=0)
        else:
            padded = np.pad(image_map, ((0, 0), (pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), 
                            mode='constant', constant_values=0)
        image_map = padded
        
        # Update dimensions
        if len(image_map.shape) == 2:
            h, w = image_map.shape
        else:
            h, w = image_map.shape[1:]
    
    # Calculate crop coordinates
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    
    # Apply crop
    if len(image_map.shape) == 2:  # 2D map
        return image_map[start_h:start_h+crop_h, start_w:start_w+crop_w]
    elif len(image_map.shape) == 3:  # 3D map (with batch dimension)
        return image_map[:, start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    return image_map  # Return as-is if dimensions don't match expected


def calculate_spearman_correlation(map1, map2, crop_size=None):
    """Calculate Spearman correlation between two 2D maps"""
    # Apply center crop if specified
    if crop_size is not None:
        map1 = apply_center_crop(map1, crop_size)
        map2 = apply_center_crop(map2, crop_size)
    
    # Make sure maps have consistent dimensions
    if map1.shape != map2.shape:
        print(f"Warning: Map shapes don't match: {map1.shape} vs {map2.shape}")
        # Resize the second map to match the first
        if len(map1.shape) == 2:
            target_shape = map1.shape
            resized_map2 = np.zeros(target_shape, dtype=map2.dtype)
            min_h = min(target_shape[0], map2.shape[0])
            min_w = min(target_shape[1], map2.shape[1])
            resized_map2[:min_h, :min_w] = map2[:min_h, :min_w]
            map2 = resized_map2
        elif len(map1.shape) == 3:
            target_shape = map1.shape
            resized_map2 = np.zeros(target_shape, dtype=map2.dtype)
            min_h = min(target_shape[1], map2.shape[1])
            min_w = min(target_shape[2], map2.shape[2])
            resized_map2[:, :min_h, :min_w] = map2[:, :min_h, :min_w]
            map2 = resized_map2
    
    # Flatten maps
    flat1 = map1.flatten()
    flat2 = map2.flatten()
    
    # Check for constant arrays (which cause problems with Spearman correlation)
    if np.all(flat1 == flat1[0]) or np.all(flat2 == flat2[0]):
        return np.nan
    
    # Calculate Spearman correlation
    corr, _ = spearmanr(flat1, flat2)
    
    # Check if correlation is NaN or infinite
    if np.isnan(corr) or np.isinf(corr):
        return np.nan
        
    return corr


def process_split_maps_gpu(clickmaps, config, image_keys=None, metadata=None, create_clickmap_func=None, fast_duplicate_detection=None):
    """
    Process a split of clickmaps on GPU in batches
    If image_keys is provided, only process these images and preserve them even if they have few trials
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
    
    # If image_keys is provided, only process those keys
    keys_to_process = image_keys if image_keys is not None else clickmaps.keys()
    
    # Preprocess all clickmaps first to binary maps
    for key in tqdm(keys_to_process, desc="Creating binary maps"):
        if key not in clickmaps:
            continue
            
        trials = clickmaps[key]
        
        # Create binary clickmaps
        if metadata and key in metadata:
            native_size = metadata[key]
            binary_maps = np.asarray([create_clickmap_func([trial], native_size[::-1]) for trial in trials])
        else:
            binary_maps = np.asarray([create_clickmap_func([trial], tuple(image_shape)) for trial in trials])
        
        # Only keep maps with enough valid pixels using mask
        mask = binary_maps.sum((-2, -1)) >= min_clicks
        binary_maps = binary_maps[mask]
        
        # If we have any valid maps, average them and keep this image
        if len(binary_maps) > 0:
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
    
    # Check if center crop should be applied
    crop_size = config.get("center_crop", None)
    if crop_size is not None:
        print(f"Will apply center cropping with size {crop_size}")
    
    for k in tqdm(range(K), desc=f"Running {K} iterations of ceiling/floor estimation"):
        # Split clickmaps into two random halves, making sure both halves contain the same images
        min_subjects_per_half = max(1, config["min_subjects"] // 2)
        first_half, second_half, common_image_keys = split_clickmaps(clickmaps, min_subjects_per_half)
        
        print(f"Iteration {k+1}/{K}: Split data for {len(common_image_keys)} common images")
        
        if not common_image_keys:
            print(f"Warning: No common images in iteration {k+1}, skipping")
            continue
        
        # Process both halves, ensuring they contain the same images
        _, first_maps, _, first_indices = process_split_maps_gpu(
            first_half, config, common_image_keys, metadata, create_clickmap_func, fast_duplicate_detection)
        
        _, second_maps, _, second_indices = process_split_maps_gpu(
            second_half, config, common_image_keys, metadata, create_clickmap_func, fast_duplicate_detection)
        
        # Verify both halves contain the same images
        if set(first_indices) != set(second_indices):
            # If there's a mismatch, only use images present in both
            common_indices = set(first_indices).intersection(set(second_indices))
            print(f"Warning: Image mismatch after processing. Using {len(common_indices)} common images.")
        else:
            common_indices = set(first_indices)  # Convert to set for consistency
            print(f"Successfully processed {len(common_indices)} common images in both halves")
        
        if not common_indices:
            print(f"Warning: No common images after processing in iteration {k+1}, skipping")
            continue
        
        # Calculate correlations for this iteration
        iteration_ceiling_corrs = []
        iteration_floor_corrs = []
        
        # Create copy of second half indices for shuffling (for floor estimate)
        shuffled_indices = second_indices.copy()
        np.random.shuffle(shuffled_indices)
        
        # Process each common image
        common_indices_list = list(common_indices)  # Convert to list for iteration
        for i, img_name in enumerate(common_indices_list):
            # Find indices in each list
            first_idx = first_indices.index(img_name)
            second_idx = second_indices.index(img_name)
            
            # Get the processed maps
            map1 = first_maps[first_idx][0]  # First element of batch
            map2 = second_maps[second_idx][0]  # First element of batch
            
            # Calculate ceiling correlation (actual correlation between halves)
            ceiling_corr = calculate_spearman_correlation(map1, map2, crop_size)
            iteration_ceiling_corrs.append(ceiling_corr)
            
            # For floor, use a random different image from second half
            if len(common_indices) > 1:  # Need at least 2 images for floor estimate
                # Find a different image to use for floor estimate
                other_images = [img for img in common_indices_list if img != img_name]
                random_img = np.random.choice(other_images)
                random_idx = second_indices.index(random_img)
                map2_random = second_maps[random_idx][0]
                
                # Calculate floor correlation (correlation with random different image)
                floor_corr = calculate_spearman_correlation(map1, map2_random, crop_size)
                iteration_floor_corrs.append(floor_corr)
            
            # Store per-image results
            if img_name not in image_results:
                image_results[img_name] = {"ceiling": [], "floor": []}
            
            image_results[img_name]["ceiling"].append(ceiling_corr)
            if len(common_indices) > 1 and 'floor_corr' in locals():
                image_results[img_name]["floor"].append(floor_corr)
        
        # Add iteration averages to overall results
        if iteration_ceiling_corrs:
            ceiling_corrs.append(np.nanmean(iteration_ceiling_corrs))
        if iteration_floor_corrs:
            floor_corrs.append(np.nanmean(iteration_floor_corrs))
    
    # Calculate overall averages
    avg_ceiling = np.nanmean(ceiling_corrs) if ceiling_corrs else np.nan
    avg_floor = np.nanmean(floor_corrs) if floor_corrs else np.nan
    
    print(f"\nResults after {K} iterations:")
    print(f"Average ceiling correlation: {avg_ceiling:.4f}")
    print(f"Average floor correlation: {avg_floor:.4f}")
    
    # Prepare results for CSV
    results_df = pd.DataFrame({
        "image": list(image_results.keys()),
        "avg_ceiling": [np.nanmean(v["ceiling"]) for v in image_results.values()],
        "avg_floor": [np.nanmean(v.get("floor", [np.nan])) for v in image_results.values()],
        "std_ceiling": [np.nanstd(v["ceiling"]) if len(v["ceiling"]) > 1 else np.nan for v in image_results.values()],
        "std_floor": [np.nanstd(v.get("floor", [np.nan])) if len(v.get("floor", [])) > 1 else np.nan for v in image_results.values()],
        "num_iterations": [len(v["ceiling"]) for v in image_results.values()]
    })
    
    # Add overall averages to the dataframe
    overall_df = pd.DataFrame({
        "image": ["OVERALL"],
        "avg_ceiling": [avg_ceiling],
        "avg_floor": [avg_floor],
        "std_ceiling": [np.nanstd(ceiling_corrs) if len(ceiling_corrs) > 1 else np.nan],
        "std_floor": [np.nanstd(floor_corrs) if len(floor_corrs) > 1 else np.nan],
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
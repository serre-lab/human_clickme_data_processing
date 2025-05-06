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


def filter_duplicate_participants(clickme_data):
    """
    Filter out duplicate submissions from the same participant for each image
    keeping only the first submission.
    
    Parameters:
    -----------
    clickme_data : pandas.DataFrame
        DataFrame containing clickme data with at least 'image_path' and 'participant' columns
        
    Returns:
    --------
    filtered_data : pandas.DataFrame
        DataFrame with duplicate participant submissions removed
    """
    print("Filtering duplicate participant submissions...")
    # Make a copy to avoid modifying the original
    data = clickme_data.copy()
    
    # Check if 'user_id' column exists
    if 'user_id' not in data.columns:
        print("Warning: Cannot filter duplicate participants - 'user_id' column missing")
        return data
    
    # Count before filtering
    total_before = len(data)
    
    # Sort by timestamp if available to ensure we keep earliest submission
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp')
    
    # Keep only the first occurrence of each participant-image combination
    filtered_data = data.drop_duplicates(subset=['user_id', 'user_id'], keep='first')
    
    # Count after filtering
    total_after = len(filtered_data)
    removed = total_before - total_after
    
    print(f"Removed {removed} duplicate participant submissions ({removed/total_before:.2%} of total)")
    
    return filtered_data


def process_all_maps_gpu(clickmaps, config, metadata=None, create_clickmap_func=None, fast_duplicate_detection=None):
    """
    Simplified function to blur clickmaps on GPU in batches
    """
    import torch
    from tqdm import tqdm
    import numpy as np
    
    # Extract basic parameters
    blur_size = config["blur_size"]
    blur_sigma = config.get("blur_sigma", blur_size)
    image_shape = config["image_shape"]
    min_subjects = config["min_subjects"]
    min_clicks = config["min_clicks"]
    
    # Get GPU batch size for processing
    gpu_batch_size = config.get("gpu_batch_size", 4096)
    
    print(f"Processing {len(clickmaps)} unique images with GPU (batch size: {gpu_batch_size})...")
    
    # Step 1: Prepare binary maps and average them
    print("Pre-processing clickmaps on CPU...")
    
    # Prepare data structures
    all_clickmaps = []
    keep_index = []
    categories = []
    final_clickmaps = {}
    click_counts = {}  # Track click counts for each image
    
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
        
        # Count total clicks in each trial
        total_clicks_per_trial = [len(trial) for trial in trials]
        
        # Only keep maps with enough valid pixels using mask
        mask = binary_maps.sum((-2, -1)) >= min_clicks
        binary_maps = binary_maps[mask]
        
        # If we have enough valid maps, average them and keep this image
        if len(binary_maps) >= min_subjects:
            all_clickmaps.append(np.array(binary_maps).mean(0, keepdims=True))
            # Note that if we are measuring ceiling we need to keep all maps ^^ change above.
            categories.append(key.split("/")[0])
            keep_index.append(key)
            final_clickmaps[key] = trials
            click_counts[key] = sum(total_clicks_per_trial)  # Store total clicks for this image
    
    if not all_clickmaps:
        print("No valid clickmaps to process")
        return {}, [], [], [], {}
    
    # Step 2: Prepare for batch blurring on GPU
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
    except Exception as e:
        # TODO: Pad all clickmaps to the same size, blur, then crop after.
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
    return final_clickmaps, all_clickmaps, categories, keep_index, click_counts


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
        clickme_data = filter_duplicate_participants(clickme_data)
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
    
    # Setup HDF5 file if needed
    hdf5_path = None
    if output_format == "hdf5":
        # Print optimization settings
        print("\nProcessing settings:")
        print(f"- Dataset size: {total_maps} maps, {total_unique_images} images")
        print(f"- GPU batch size: {config['gpu_batch_size']}")
        print(f"- CPU workers: {config['n_jobs']}")
        print(f"- Output format: {config['output_format']}")
        print(f"- Filter duplicates: {config['filter_duplicates']}")
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
        
        # Process in batches to avoid memory issues
        max_batch_size = config.get("batch_size", 50000)  # Default to 50k images per batch
        
        # Get list of all image keys
        all_keys = list(clickmaps.keys())
        total_unique_images = len(all_keys)
        num_batches = (total_unique_images + max_batch_size - 1) // max_batch_size
        
        print(f"Processing dataset in {num_batches} batches of up to {max_batch_size} images each")
        
        # Store final results
        all_medians = {'image': {}, 'category': {}, 'all': {}}
        processed_images_count = 0
        all_click_counts = {}  # To store click counts across all batches
        
        for batch_num in range(num_batches):
            print(f"\n--- Processing batch {batch_num+1}/{num_batches} ---")
            
            # Calculate batch indices
            start_idx = batch_num * max_batch_size
            end_idx = min(start_idx + max_batch_size, total_unique_images)
            batch_keys = all_keys[start_idx:end_idx]
            
            # Create batch-specific clickmaps dictionary
            batch_clickmaps = {k: clickmaps[k] for k in batch_keys}
            batch_size = len(batch_clickmaps)
            
            # Create batch-specific HDF5 file with suffix
            batch_suffix = f"_batch{batch_num+1:03d}" if num_batches > 1 else ""
            hdf5_path = os.path.join(output_dir, f"{config['experiment_name']}{batch_suffix}.h5")
            print(f"Saving batch results to HDF5 file: {hdf5_path}")
            
            with h5py.File(hdf5_path, 'w') as f:
                f.create_group("clickmaps")
                f.create_group("click_counts")  # Add group for click counts
                meta_grp = f.create_group("metadata")
                meta_grp.attrs["batch_number"] = batch_num + 1
                meta_grp.attrs["total_batches"] = num_batches
                meta_grp.attrs["batch_size"] = batch_size
                meta_grp.attrs["total_unique_images"] = total_unique_images
                meta_grp.attrs["total_maps"] = total_maps
                meta_grp.attrs["filter_duplicates"] = np.bytes_("True" if config["filter_duplicates"] else "False")
                meta_grp.attrs["creation_date"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Process this batch of clickmaps
            print(f"Processing batch with GPU (batch size: {config['gpu_batch_size']})...")
            
            batch_final_clickmaps, batch_all_clickmaps, batch_categories, batch_final_keep_index, batch_click_counts = process_all_maps_gpu(
                clickmaps=batch_clickmaps,
                config=config,
                metadata=metadata,
                create_clickmap_func=create_clickmap_func,
                fast_duplicate_detection=fast_duplicate_detection
            )

            # Apply mask filtering if needed
            if batch_final_keep_index and config["mask_dir"]:
                print("Applying mask filtering...")
                masks = utils.load_masks(config["mask_dir"])
                batch_final_clickmaps, batch_all_clickmaps, batch_categories, batch_final_keep_index = utils.filter_for_foreground_masks(
                    final_clickmaps=batch_final_clickmaps,
                    all_clickmaps=batch_all_clickmaps,
                    categories=batch_categories,
                    masks=masks,
                    mask_threshold=config["mask_threshold"])
                # Update click counts to match filtered images
                batch_click_counts = {k: batch_click_counts[k] for k in batch_final_keep_index if k in batch_click_counts}
            
            # Save results for this batch
            if batch_final_keep_index:
                print(f"Saving {len(batch_final_keep_index)} processed maps for batch {batch_num+1}...")
                processed_images_count += len(batch_final_keep_index)
                
                # Store click counts
                all_click_counts.update(batch_click_counts)
                
                # Save click counts to HDF5
                with h5py.File(hdf5_path, 'a') as f:
                    for img_name, count in batch_click_counts.items():
                        dataset_name = img_name.replace('/', '_')
                        f["click_counts"].create_dataset(dataset_name, data=count)
                
                # Use optimized HDF5 saving with compression
                saved_count = utils.save_clickmaps_to_hdf5(
                    all_clickmaps=batch_all_clickmaps,
                    final_keep_index=batch_final_keep_index,
                    hdf5_path=hdf5_path,
                    n_jobs=config["n_jobs"],
                    compression=config.get("hdf5_compression"),
                    compression_level=config.get("hdf5_compression_level", 0)
                )
                print(f"Saved {saved_count} files in batch {batch_num+1}")
                
                # Calculate batch medians and update global medians
                batch_medians = get_medians(batch_final_clickmaps, 'image', thresh=config["percentile_thresh"])
                all_medians['image'].update(batch_medians.get('image', {}))
                
                batch_cat_medians = get_medians(batch_final_clickmaps, 'category', thresh=config["percentile_thresh"])
                for cat, val in batch_cat_medians.get('category', {}).items():
                    if cat not in all_medians['category']:
                        all_medians['category'][cat] = []
                    all_medians['category'][cat].append(val)
                
                batch_all_medians = get_medians(batch_final_clickmaps, 'all', thresh=config["percentile_thresh"])
                for k, v in batch_all_medians.items():
                    if k not in all_medians['all']:
                        all_medians['all'][k] = []
                    all_medians['all'][k].append(v)
            
            # Free memory after each batch
            del batch_clickmaps, batch_final_clickmaps, batch_all_clickmaps, batch_categories, batch_final_keep_index, batch_click_counts
            gc.collect()
            if config["use_gpu_blurring"]:
                torch.cuda.empty_cache()
            
            print(f"Memory usage after batch {batch_num+1}: {get_memory_usage():.2f} MB")
        
        # Finalize global medians
        print("Calculating global medians...")
        final_medians = {'image': all_medians['image']}
        
        # Aggregate category medians
        final_medians['category'] = {}
        for cat, values in all_medians['category'].items():
            final_medians['category'][cat] = int(np.median(values))
        
        # Aggregate overall medians
        final_medians['all'] = {}
        for k, values in all_medians['all'].items():
            if k in ['median', 'threshold']:
                final_medians['all'][k] = int(np.median(values))
            elif k == 'mean':
                final_medians['all'][k] = float(np.mean(values))
        
        # Save final medians
        medians_json = json.dumps(final_medians, indent=4)
        with open(os.path.join(output_dir, config["processed_medians"]), 'w') as f:
            f.write(medians_json)
        
        # Save click counts to a separate JSON file
        click_counts_json = json.dumps(all_click_counts, indent=4)
        with open(os.path.join(output_dir, f"{config['experiment_name']}_click_counts.json"), 'w') as f:
            f.write(click_counts_json)
        
        print(f"Processed and saved a total of {processed_images_count} images across {num_batches} batches")
        
        # Set finals for visualization
        final_clickmaps = batch_final_clickmaps if 'batch_final_clickmaps' in locals() else {}
        
    else:
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
        
        # Process all maps with our new single-batch GPU function
        print(f"Processing with GPU (batch size: {config['gpu_batch_size']})...")
        
        final_clickmaps, all_clickmaps, categories, final_keep_index, click_counts = process_all_maps_gpu(
            clickmaps=clickmaps,
            config=config,
            metadata=metadata,
            create_clickmap_func=create_clickmap_func,
            fast_duplicate_detection=fast_duplicate_detection
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
        
        # Save results
        if final_keep_index:
            print(f"Saving {len(final_keep_index)} processed maps...")
            
            # Save click counts to HDF5
            with h5py.File(hdf5_path, 'a') as f:
                for img_name, count in click_counts.items():
                    dataset_name = img_name.replace('/', '_')
                    f["click_counts"].create_dataset(dataset_name, data=count)
            
            # Save click counts to a separate JSON file
            click_counts_json = json.dumps(click_counts, indent=4)
            with open(os.path.join(output_dir, f"{config['experiment_name']}_click_counts.json"), 'w') as f:
                f.write(click_counts_json)
            
            # Use parallel saving
            saved_count = utils.save_clickmaps_parallel(
                all_clickmaps=all_clickmaps,
                final_keep_index=final_keep_index,
                output_dir=output_dir,
                experiment_name=config["experiment_name"],
                image_path=config["image_path"],
                n_jobs=config["n_jobs"],
                file_inclusion_filter=config.get("file_inclusion_filter")
            )
            
            # Save click counts alongside individual numpy files
            for i, img_name in enumerate(final_keep_index):
                count_file = os.path.join(output_dir, config["experiment_name"], f"{img_name.replace('/', '_')}_count.npy")
                np.save(count_file, click_counts[img_name])
            
            print(f"Saved {saved_count} files and their click counts")
        
        # Get median number of clicks from the results
        print("Calculating medians...")
        percentile_thresh = config["percentile_thresh"]
        medians = get_medians(final_clickmaps, 'image', thresh=percentile_thresh)
        medians.update(get_medians(final_clickmaps, 'category', thresh=percentile_thresh))
        medians.update(get_medians(final_clickmaps, 'all', thresh=percentile_thresh))
        medians_json = json.dumps(medians, indent=4)

        # Save medians
        with open(os.path.join(output_dir, config["processed_medians"]), 'w') as f:
            f.write(medians_json)
    
    # Process visualization for display images if needed
    if config["display_image_keys"]:
        if config["display_image_keys"] == "auto":
            sz_dict = {k: len(v) for k, v in final_clickmaps.items()}
            arg = np.argsort(list(sz_dict.values()))
            config["display_image_keys"] = np.asarray(list(sz_dict.keys()))[arg[-10:]]
            
        print("Generating visualizations for display images...")
        for img_name in config["display_image_keys"]:
            # Find the corresponding heatmap
            try:
                if output_format == "hdf5":
                    # Read from HDF5 file
                    with h5py.File(hdf5_path, 'r') as f:
                        dataset_name = img_name.replace('/', '_')
                        if dataset_name in f["clickmaps"]:
                            hmp = f["clickmaps"][dataset_name][:]
                            # Also read click count if available
                            click_count = f["click_counts"][dataset_name][()] if dataset_name in f["click_counts"] else None
                        else:
                            print(f"Heatmap not found for {img_name}")
                            continue
                else:
                    # Read from numpy file
                    heatmap_path = os.path.join(output_dir, config["experiment_name"], f"{img_name.replace('/', '_')}.npy")
                    if not os.path.exists(heatmap_path):
                        print(f"Heatmap not found for {img_name}")
                        continue
                        
                    hmp = np.load(heatmap_path)
                    # Try to load click count
                    count_path = os.path.join(output_dir, config["experiment_name"], f"{img_name.replace('/', '_')}_count.npy")
                    click_count = np.load(count_path) if os.path.exists(count_path) else None
                    
                # Load image
                if os.path.exists(os.path.join(config["image_path"], img_name)):
                    img = Image.open(os.path.join(config["image_path"], img_name))
                elif os.path.exists(os.path.join(config["image_path"].replace(config["file_inclusion_filter"] + os.path.sep, ""), img_name)):
                    img = Image.open(os.path.join(config["image_path"].replace(config["file_inclusion_filter"] + os.path.sep, ""), img_name))
                elif os.path.exists(os.path.join(config["image_path"].replace(config["file_inclusion_filter"], ""), img_name)):
                    img = Image.open(os.path.join(config["image_path"].replace(config["file_inclusion_filter"], ""), img_name))
                else:
                    print(f"Image not found for {img_name}")
                    continue
                    
                if metadata:
                    click_match = [k_ for k_ in final_clickmaps.keys() if img_name in k_]
                    if click_match:
                        metadata_size = metadata[click_match[0]]
                        img = img.resize(metadata_size)
                
                # Save visualization
                f = plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(np.asarray(img))
                title = f"{img_name}"
                if click_count is not None:
                    title += f"\nTotal clicks: {click_count}"
                plt.title(title)
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(hmp.mean(0))
                plt.axis("off")
                plt.savefig(os.path.join(image_output_dir, img_name.replace('/', '_')))
                plt.close()
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                continue
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
    
    print(f"\nProcessing complete! Final memory usage: {get_memory_usage():.2f} MB")

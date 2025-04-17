import os, sys
import numpy as np
from PIL import Image
import json
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from src import utils
from tqdm import tqdm
import h5py  # Add h5py for HDF5 support


def get_medians(point_lists, mode='image', thresh=50):
    medians = {}
    if mode == 'image':
        for image in point_lists:
            clickmaps = point_lists[image]
            num_clicks = []
            for clickmap in clickmaps:
                num_clicks.append(len(clickmap))
            if num_clicks:  # Check if the list is not empty
                medians[image] = np.percentile(num_clicks, thresh)
            else:
                medians[image] = 0  # Default value when no clicks
    elif mode == 'category':
        for image in point_lists:
            category = image.split('/')[0]
            if category not in medians.keys():
                medians[category] = []
            clickmaps = point_lists[image]
            for clickmap in clickmaps:
                medians[category].append(len(clickmap))
        for category in medians:
            if medians[category]:  # Check if the list is not empty
                medians[category] = np.percentile(medians[category], thresh)
            else:
                medians[category] = 0  # Default value when no clicks
    elif mode == 'all':
        num_clicks = []
        for image in point_lists:
            clickmaps = point_lists[image]
            for clickmap in clickmaps:
                num_clicks.append(len(clickmap))
        if num_clicks:  # Check if the list is not empty
            medians['all'] = np.percentile(num_clicks, thresh)
        else:
            medians['all'] = 0  # Default value when no clicks
            print("Warning: No clicks found when calculating 'all' median")
    else:
        raise NotImplementedError(mode)
    return medians


if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(description="Process clickme data for modeling")
    parser.add_argument('config', nargs='?', help='Path to config file')
    parser.add_argument('--start-chunk', type=int, default=1, help='Chunk number to start processing from')
    parser.add_argument('--end-chunk', type=int, default=None, help='Chunk number to end processing at')
    parser.add_argument('--debug', action='store_true', help='Enable additional debug output')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress for GPU processing')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU-only processing regardless of config')
    parser.add_argument('--gpu-batch-size', type=int, default=None, help='Override GPU batch size')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of CPU workers')
    args = parser.parse_args()
    
    # Load config file
    if args.config:
        config_file = args.config if "configs" + os.path.sep in args.config else os.path.join("configs", args.config)
        assert os.path.exists(config_file), f"Cannot find config file: {config_file}"
        config = utils.process_config(config_file)
    else:
        config_file = utils.get_config(sys.argv)
        config = utils.process_config(config_file)
    
    # Load clickme data first to assess dataset size
    clickme_data = utils.process_clickme_data(
        config["clickme_data"],
        config["filter_mobile"])
    total_maps = len(clickme_data)

    # Add data validation to ensure all maps are properly grouped by image_path
    print(f"Validating clickme data structure for {total_maps} maps...")
    image_paths = clickme_data['image_path'].unique()
    print(f"Found {len(image_paths)} unique images")


    # Group clickme_data by image_path to ensure all maps for an image are processed together
    unique_images = clickme_data['image_path'].unique()
    total_unique_images = len(unique_images)

    # Verify that images are properly grouped before chunking
    print(f"Verifying that all maps for each image will be grouped correctly...")

    # Performance tuning based on dataset size
    if total_maps > 500000:  # Large dataset
        # Smaller batch sizes for large datasets
        if args.gpu_batch_size:
            config["gpu_batch_size"] = args.gpu_batch_size
        else:
            config["gpu_batch_size"] = 1024  # Significantly smaller batch size for better throughput
        
        # Limit CPU workers
        if args.max_workers:
            config["n_jobs"] = min(args.max_workers, os.cpu_count())
        else:
            config["n_jobs"] = min(8, os.cpu_count())  # Limit CPU workers
        
        config["parallel_clickmap_processing"] = True  # Enable parallel processing
        
        # Force CPU-only if requested
        if args.cpu_only:
            config["use_gpu_blurring"] = False
        else:
            config["use_gpu_blurring"] = True  # Keep GPU acceleration
        
        # Add more aggressive chunking
        config["chunk_size"] = min(50000, config.get("chunk_size", 100000))  # Smaller chunks for large datasets
        
        print(f"Large dataset detected ({total_maps} maps). Using optimized settings:")
        print(f"- GPU batch size: {config['gpu_batch_size']}")
        print(f"- CPU workers: {config['n_jobs']}")
        print(f"- Using GPU: {config['use_gpu_blurring']}")
        print(f"- Chunk size: {config['chunk_size']}")

    # Other Args
    # blur_sigma_function = lambda x: np.sqrt(x)
    # blur_sigma_function = lambda x: x / 2
    blur_sigma_function = lambda x: x

    # Load config
    output_dir = config["assets"]
    image_output_dir = config["example_image_output_dir"]
    blur_size = config["blur_size"]
    blur_sigma = blur_sigma_function(blur_size)
    min_pixels = (2 * blur_size) ** 2  # Minimum number of pixels for a map to be included following filtering
    
    # Set output format (HDF5 by default for large datasets)
    output_format = config.get("output_format", "auto")
    if output_format == "auto":
        # Automatically choose format based on dataset size
        large_dataset_threshold = 100000  # If more than 100K images, use HDF5
        if total_maps > large_dataset_threshold:
            output_format = "hdf5"
        else:
            output_format = "numpy"
    
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

    # Start processing
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, config["experiment_name"]), exist_ok=True)

    # Process data in chunks to avoid memory issues
    print(f"Processing clickme data in chunks by unique images...")
    
    # Determine chunk size based on unique images, not total maps
    chunk_size = config.get("chunk_size", 100000)  # Configurable chunk size
    
    # Calculate number of chunks based on unique images
    num_chunks = (total_unique_images + chunk_size - 1) // chunk_size
    
    # Process all data in chunks
    all_final_clickmaps = {}
    all_final_keep_index = []
    
    # Prepare HDF5 file if using HDF5 format
    hdf5_path = None
    if output_format == "hdf5":
        hdf5_path = os.path.join(output_dir, f"{config['experiment_name']}.h5")
        print(f"Saving results to HDF5 file: {hdf5_path}")
        # Create the file initially to ensure directory exists
        with h5py.File(hdf5_path, 'w') as f:
            # Create datasets group to store all clickmaps
            f.create_group("clickmaps")
            # Create a metadata group
            meta_grp = f.create_group("metadata")
            # Add some useful metadata
            meta_grp.attrs["total_unique_images"] = total_unique_images
            meta_grp.attrs["creation_date"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            meta_grp.attrs["chunks_completed"] = 0  # Track progress
    
    # Determine start and end chunks based on command line arguments
    start_chunk = max(0, args.start_chunk - 1)  # Convert to 0-indexed
    end_chunk = args.end_chunk if args.end_chunk is not None else num_chunks
    
    # Display restart information if needed
    if start_chunk > 0:
        print(f"\nRestarting from chunk {start_chunk + 1} (skipping chunks 1-{start_chunk})")
    
    # Use a simple progress tracking system with tqdm - prettier hierarchy
    print(f"\nProcessing {total_unique_images} unique images in {num_chunks} chunks (processing chunks {start_chunk + 1}-{end_chunk})...")
    with tqdm(total=end_chunk-start_chunk, desc="├─ Processing chunks", position=0, leave=True, colour="blue") as pbar:
        for chunk_idx in range(start_chunk, end_chunk):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_unique_images)
            
            # Use a clear, hierarchical format for chunk info
            print(f"\n├─ Chunk {chunk_idx + 1}/{num_chunks} ({chunk_start}-{chunk_end})")
            
            # Get the image_paths for this chunk
            chunk_image_paths = unique_images[chunk_start:chunk_end]
            
            # Filter clickme_data to only include maps for images in this chunk
            chunk_data = clickme_data[clickme_data['image_path'].isin(chunk_image_paths)]
            
            l = len(chunk_data)       
            print(f"│  ├─ Debug: Chunk has {l} maps for {len(chunk_image_paths)} unique images")

            # Process chunk data
            print(f"│  ├─ Processing clickmap files...")
            
            # Choose the appropriate method based on dataset size
            parallel_clickmap_processing = config.get("parallel_clickmap_processing", True)
            
            if parallel_clickmap_processing and l > 10000:
                # Use parallel processing for large chunks
                clickmaps, ccounts = utils.process_clickmap_files_parallel(
                    clickme_data=chunk_data,
                    image_path=config["image_path"],
                    file_inclusion_filter=config["file_inclusion_filter"],
                    file_exclusion_filter=config["file_exclusion_filter"],
                    min_clicks=config["min_clicks"],
                    max_clicks=config["max_clicks"],
                    n_jobs=config.get("n_jobs", -1))
            else:
                # Use regular processing for smaller chunks
                clickmaps, ccounts = utils.process_clickmap_files(
                    clickme_data=chunk_data,
                    image_path=config["image_path"],
                    file_inclusion_filter=config["file_inclusion_filter"],
                    file_exclusion_filter=config["file_exclusion_filter"],
                    min_clicks=config["min_clicks"],
                    max_clicks=config["max_clicks"])
            
            print(f"│  ├─ Debug: After processing clickmap files: {len(clickmaps)} unique images with clickmaps")
                
            # Apply all filters to the chunk
            if config["class_filter_file"]:
                print(f"│  ├─ Filtering classes...")
                clickmaps = utils.filter_classes(
                    clickmaps=clickmaps,
                    class_filter_file=config["class_filter_file"])
                print(f"│  ├─ Debug: After class filtering: {len(clickmaps)} images")

            if config["participant_filter"]:
                print(f"│  ├─ Filtering participants...")
                clickmaps = utils.filter_participants(clickmaps)
                print(f"│  ├─ Debug: After participant filtering: {len(clickmaps)} images")
            
            # Get GPU batch size and workers from config or command line args
            gpu_batch_size = args.gpu_batch_size or config.get("gpu_batch_size", 1024)
            n_jobs = args.max_workers or config.get("n_jobs", min(8, os.cpu_count()))
            
            # Add max processing time for GPU batching 
            gpu_timeout = config.get("gpu_timeout", 900)  # 15 minutes timeout for parallel jobs
            
            # Add a safety exit mechanism
            retry_count = 0
            max_retries = config.get("max_retries", 2)
            
            # Use GPU optimized blurring by default, can be disabled with use_gpu_blurring=False
            use_gpu_blurring = False if args.cpu_only else config.get("use_gpu_blurring", True)
            
            # Try different strategies for smaller subsets if dataset is very large
            if len(clickmaps) > 10000 and use_gpu_blurring:
                # Process in smaller sub-chunks for very large datasets
                sub_chunk_size = config.get("sub_chunk_size", 10000)  # Process 10k images at a time
                print(f"│  ├─ Dataset is large, processing in {(len(clickmaps) + sub_chunk_size - 1) // sub_chunk_size} sub-chunks of max {sub_chunk_size} images")
                
                # Initialize sub-chunk results
                all_sub_clickmaps = []
                all_sub_keep_index = []
                
                # Create list of image keys
                img_keys = list(clickmaps.keys())
                
                # Process sub-chunks
                for i in range(0, len(img_keys), sub_chunk_size):
                    sub_keys = img_keys[i:i+sub_chunk_size]
                    sub_clickmaps = {k: clickmaps[k] for k in sub_keys}
                    
                    print(f"│  ├─ Processing sub-chunk {i//sub_chunk_size + 1}/{(len(clickmaps) + sub_chunk_size - 1) // sub_chunk_size} ({len(sub_clickmaps)} images)")
                    
                    if use_gpu_blurring:
                        print(f"│  ├─ Preparing maps with GPU-optimized blurring (batch_size={gpu_batch_size}, n_jobs={n_jobs})...")
                    else:
                        print(f"│  ├─ Preparing maps with CPU processing (n_jobs={n_jobs})...")
                    
                    # Process sub-chunk
                    try:
                        # Use GPU-optimized batched processing with smaller batch size for better throughput
                        sub_final_clickmaps, sub_all_clickmaps, sub_categories, sub_final_keep_index = utils.prepare_maps_with_gpu_batching(
                            final_clickmaps=[sub_clickmaps],  # Wrap in a list as the function expects a list of dictionaries
                            blur_size=blur_size,
                            blur_sigma=blur_sigma,
                            image_shape=config["image_shape"],
                            min_pixels=min_pixels,
                            min_subjects=config["min_subjects"],
                            metadata=metadata,
                            blur_sigma_function=blur_sigma_function,
                            center_crop=False,
                            n_jobs=n_jobs,
                            batch_size=gpu_batch_size,
                            timeout=gpu_timeout,
                            verbose=args.verbose or args.debug,
                            create_clickmap_func=create_clickmap_func,
                            fast_duplicate_detection=fast_duplicate_detection)
                        
                        # Collect results
                        all_sub_clickmaps.extend(sub_all_clickmaps)
                        all_sub_keep_index.extend(sub_final_keep_index)
                        
                        # Update main chunk results
                        all_final_clickmaps.update(sub_final_clickmaps)
                    
                    except Exception as e:
                        print(f"│  ├─ ERROR: Failed to process sub-chunk: {e}")
                        if retry_count < max_retries:
                            retry_count += 1
                            print(f"│  ├─ Trying again with CPU processing...")
                            try:
                                # Fall back to CPU processing
                                sub_final_clickmaps, sub_all_clickmaps, sub_categories, sub_final_keep_index = utils.prepare_maps_with_progress(
                                    final_clickmaps=[sub_clickmaps],  # Wrap in a list as the function expects a list of dictionaries
                                    blur_size=blur_size,
                                    blur_sigma=blur_sigma,
                                    image_shape=config["image_shape"],
                                    min_pixels=min_pixels,
                                    min_subjects=config["min_subjects"],
                                    metadata=metadata,
                                    blur_sigma_function=blur_sigma_function,
                                    center_crop=False,
                                    n_jobs=n_jobs,
                                    create_clickmap_func=create_clickmap_func,
                                    fast_duplicate_detection=fast_duplicate_detection)
                                
                                # Collect results
                                all_sub_clickmaps.extend(sub_all_clickmaps)
                                all_sub_keep_index.extend(sub_final_keep_index)
                                
                                # Update main chunk results
                                all_final_clickmaps.update(sub_final_clickmaps)
                            except Exception as e:
                                print(f"│  ├─ ERROR: CPU processing also failed: {e}")
                                continue
                        else:
                            print(f"│  ├─ Skipping this sub-chunk after {max_retries} retries")
                            continue
                
                # Set results for saving
                chunk_all_clickmaps = all_sub_clickmaps
                chunk_final_keep_index = all_sub_keep_index
                chunk_final_clickmaps = all_final_clickmaps  # This is accumulated from sub-chunks
                chunk_categories = []  # Not used for saving
                
            elif use_gpu_blurring:
                print(f"│  ├─ Preparing maps with GPU-optimized blurring (batch_size={gpu_batch_size}, n_jobs={n_jobs})...")
                
                # Add retry mechanism for GPU batched processing
                while retry_count <= max_retries:
                    try:
                        # Use GPU-optimized batched processing
                        chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.prepare_maps_with_gpu_batching(
                            final_clickmaps=[clickmaps],  # Wrap in a list as the function expects a list of dictionaries
                            blur_size=blur_size,
                            blur_sigma=blur_sigma,
                            image_shape=config["image_shape"],
                            min_pixels=min_pixels,
                            min_subjects=config["min_subjects"],
                            metadata=metadata,
                            blur_sigma_function=blur_sigma_function,
                            center_crop=False,
                            n_jobs=n_jobs,
                            batch_size=gpu_batch_size,
                            timeout=gpu_timeout,
                            verbose=args.verbose or args.debug,  # Use verbose if --verbose or --debug flag is set
                            create_clickmap_func=create_clickmap_func,
                            fast_duplicate_detection=fast_duplicate_detection)
                        # If we get here, processing succeeded
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            print(f"│  ├─ ERROR: Failed to process chunk after {max_retries} retries: {e}")
                            print(f"│  ├─ Trying CPU processing as a last resort...")
                            try:
                                # Fall back to CPU processing
                                chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.prepare_maps_with_progress(
                                    final_clickmaps=[clickmaps],  # Wrap in a list as the function expects a list of dictionaries
                                    blur_size=blur_size,
                                    blur_sigma=blur_sigma,
                                    image_shape=config["image_shape"],
                                    min_pixels=min_pixels,
                                    min_subjects=config["min_subjects"],
                                    metadata=metadata,
                                    blur_sigma_function=blur_sigma_function,
                                    center_crop=False,
                                    n_jobs=n_jobs,
                                    create_clickmap_func=create_clickmap_func,
                                    fast_duplicate_detection=fast_duplicate_detection)
                            except Exception as e:
                                print(f"│  ├─ ERROR: CPU processing also failed: {e}")
                                print(f"│  ├─ Skipping this chunk and continuing to the next one...")
                                # Initialize with empty results to avoid errors in subsequent code
                                chunk_final_clickmaps = {}
                                chunk_all_clickmaps = []
                                chunk_categories = []
                                chunk_final_keep_index = []
                            break
                        else:
                            print(f"│  ├─ WARNING: Error processing chunk: {e}")
                            print(f"│  ├─ Retry {retry_count}/{max_retries} with smaller batch size...")
                            # Reduce batch size for retry
                            gpu_batch_size = max(256, gpu_batch_size // 2)
                            # Continue to retry
            else:
                # Use the original CPU-based processing with better error handling
                print(f"│  ├─ Preparing maps with CPU processing (n_jobs={n_jobs})...")
                try:
                    chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.prepare_maps_with_progress(
                        final_clickmaps=[clickmaps],  # Wrap in a list as the function expects a list of dictionaries
                        blur_size=blur_size,
                        blur_sigma=blur_sigma,
                        image_shape=config["image_shape"],
                        min_pixels=min_pixels,
                        min_subjects=config["min_subjects"],
                        metadata=metadata,
                        blur_sigma_function=blur_sigma_function,
                        center_crop=False,
                        n_jobs=n_jobs,
                        create_clickmap_func=create_clickmap_func,
                        fast_duplicate_detection=fast_duplicate_detection)
                except Exception as e:
                    print(f"│  ├─ ERROR: Failed to process chunk: {e}")
                    print(f"│  ├─ Skipping this chunk and continuing to the next one...")
                    # Initialize with empty results to avoid errors in subsequent code
                    chunk_final_clickmaps = {}
                    chunk_all_clickmaps = []
                    chunk_categories = []
                    chunk_final_keep_index = []

            # Apply mask filtering if needed and if chunk was processed successfully
            if chunk_final_keep_index and config["mask_dir"]:
                print(f"│  ├─ Applying mask filtering...")
                masks = utils.load_masks(config["mask_dir"])
                chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.filter_for_foreground_masks(
                    final_clickmaps=chunk_final_clickmaps,
                    all_clickmaps=chunk_all_clickmaps,
                    categories=chunk_categories,
                    masks=masks,
                    mask_threshold=config["mask_threshold"])
            
            # Save results if we have any
            if chunk_final_keep_index:
                print(f"│  ├─ Saving processed maps ({len(chunk_final_keep_index)} images)...")
            
            # Save results
            print(f"│  ├─ Saving processed maps...")
            
            if output_format == "hdf5":
                # Save to HDF5 file
                saved_count = utils.save_clickmaps_to_hdf5(
                    all_clickmaps=chunk_all_clickmaps,
                    final_keep_index=chunk_final_keep_index,
                    hdf5_path=hdf5_path,
                    n_jobs=n_jobs
                )
                print(f"│  │  └─ Saved {saved_count} datasets to HDF5 file")
            else:
                # Check if parallel saving is enabled (default to True if not specified)
                use_parallel_save = config.get("parallel_save", True)
                if use_parallel_save:
                    # Use parallel saving
                    saved_count = utils.save_clickmaps_parallel(
                        all_clickmaps=chunk_all_clickmaps,
                        final_keep_index=chunk_final_keep_index,
                        output_dir=output_dir,
                        experiment_name=config["experiment_name"],
                        image_path=config["image_path"],
                        n_jobs=n_jobs,
                        file_inclusion_filter=config.get("file_inclusion_filter")
                    )
                    print(f"│  │  └─ Saved {saved_count} files in parallel")
                else:
                    # Use sequential saving with tqdm
                    with tqdm(total=len(chunk_final_keep_index), desc="│  │  ├─ Saving files", 
                             position=1, leave=False, colour="cyan") as save_pbar:
                        for j, img_name in enumerate(chunk_final_keep_index):
                            # Check multiple possible paths for the image
                            # - Standard path
                            # - Path with inclusion filter removed from middle
                            # - Path with inclusion filter removed from beginning
                            image_exists = False
                            possible_paths = [
                                os.path.join(config["image_path"], img_name),  # Standard path
                                os.path.join(config["image_path"].replace(config["file_inclusion_filter"] + os.path.sep, ""), img_name),  # Filter removed from middle
                                os.path.join(config["image_path"].replace(config["file_inclusion_filter"], ""), img_name)  # Filter removed from beginning
                            ]
                            
                            for path in possible_paths:
                                if os.path.exists(path):
                                    image_exists = True
                                    break
                            
                            if not image_exists:
                                if args.debug:
                                    print(f"Warning: Could not find image for {img_name}, tried paths: {possible_paths}")
                                continue
                                
                            hmp = chunk_all_clickmaps[j]
                            # Save directly to disk - don't accumulate in memory
                            np.save(
                                os.path.join(output_dir, config["experiment_name"], f"{img_name.replace('/', '_')}.npy"), 
                                hmp
                            )
                            save_pbar.update(1)
            
            # Merge results (keeping minimal data in memory)
            all_final_clickmaps.update(chunk_final_clickmaps)
            all_final_keep_index.extend(chunk_final_keep_index)
            
            # Free memory
            del chunk_data, clickmaps, ccounts
            del chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index
            
            # Force Python garbage collection
            import gc
            gc.collect()
            
            # Update progress in HDF5 file if using HDF5
            if output_format == "hdf5":
                try:
                    with h5py.File(hdf5_path, 'a') as f:
                        f["metadata"].attrs["chunks_completed"] = chunk_idx + 1
                        f["metadata"].attrs["last_updated"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
                except Exception as e:
                    print(f"Warning: Could not update HDF5 metadata: {e}")
            
            # Update main progress bar
            pbar.update(1)
            print(f"│  └─ Chunk {chunk_idx + 1} complete.\n")
    
    # Get median number of clicks from the combined results
    percentile_thresh = config["percentile_thresh"]
    medians = get_medians(all_final_clickmaps, 'image', thresh=percentile_thresh)
    medians.update(get_medians(all_final_clickmaps, 'category', thresh=percentile_thresh))
    medians.update(get_medians(all_final_clickmaps, 'all', thresh=percentile_thresh))
    medians_json = json.dumps(medians, indent=4)

    # Save medians
    with open(os.path.join(output_dir, config["processed_medians"]), 'w') as f:
        f.write(medians_json)
    
    # Process visualization for display images if needed
    if config["display_image_keys"]:
        if config["display_image_keys"] == "auto":
            sz_dict = {k: len(v) for k, v in all_final_clickmaps.items()}
            arg = np.argsort(list(sz_dict.values()))
            config["display_image_keys"] = np.asarray(list(sz_dict.keys()))[arg[-10:]]
            
        print("Generating visualizations for display images...")
        for img_name in config["display_image_keys"]:
            # Find the corresponding heatmap
            if output_format == "hdf5":
                # Read from HDF5 file
                with h5py.File(hdf5_path, 'r') as f:
                    dataset_name = img_name.replace('/', '_')
                    if dataset_name in f["clickmaps"]:
                        hmp = f["clickmaps"][dataset_name][:]
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
                
            if os.path.exists(os.path.join(config["image_path"], img_name)):
                img = Image.open(os.path.join(config["image_path"], img_name))
            elif os.path.exists(os.path.join(config["image_path"].replace(config["file_inclusion_filter"] + os.path.sep, ""), img_name)):
                img = Image.open(os.path.join(config["image_path"].replace(config["file_inclusion_filter"] + os.path.sep, ""), img_name))
            elif os.path.exists(os.path.join(config["image_path"].replace(config["file_inclusion_filter"], ""), img_name)):
                img = Image.open(os.path.join(config["image_path"].replace(config["file_inclusion_filter"], ""), img_name))
            else:
                raise ValueError(f"Image not found for {img_name}")
            if metadata:
                click_match = [k_ for k_ in all_final_clickmaps.keys() if img_name in k_]
                if click_match:
                    metadata_size = metadata[click_match[0]]
                    img = img.resize(metadata_size)
            
            # Save visualization
            f = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.asarray(img))
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(hmp.mean(0))
            plt.axis("off")
            plt.savefig(os.path.join(image_output_dir, img_name.replace('/', '_')))
            plt.close()

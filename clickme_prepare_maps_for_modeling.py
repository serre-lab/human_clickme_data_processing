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
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor


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


def get_memory_usage():
    """Return current memory usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB


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
    parser.add_argument('--chunk-size', type=int, default=None, help='Override chunk size')
    parser.add_argument('--save-interval', type=int, default=50, help='Save results every N images to avoid memory buildup')
    parser.add_argument('--profile', action='store_true', help='Enable performance profiling')
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
    
    # Load clickme data first to assess dataset size
    print(f"Loading clickme data...")
    clickme_data = utils.process_clickme_data(
        config["clickme_data"],
        config["filter_mobile"])
    total_maps = len(clickme_data)

    # Add data validation to ensure all maps are properly grouped by image_path
    print(f"Validating clickme data structure for {total_maps} maps...")
    image_paths = clickme_data['image_path'].unique()
    total_unique_images = len(image_paths)
    print(f"Found {total_unique_images} unique images")
    
    # Determine optimal parameters for dataset size
    # For a dataset with 300k maps and 4581 images
    
    # Override chunk size if specified
    if args.chunk_size:
        config["chunk_size"] = args.chunk_size
    elif "chunk_size" not in config or total_unique_images < 5000:
        # Optimize chunk size - for 4581 images, use larger chunks
        config["chunk_size"] = min(1000, total_unique_images // 5)
    
    # Set optimal GPU batch size for better throughput
    if args.gpu_batch_size:
        config["gpu_batch_size"] = args.gpu_batch_size
    else:
        # Determine best batch size based on dataset characteristics
        # For 300k maps, smaller batches usually process faster
        if total_maps > 200000:
            config["gpu_batch_size"] = 512
        else:
            config["gpu_batch_size"] = 1024
    
    # Optimize number of workers based on CPU count
    cpu_count = os.cpu_count()
    if args.max_workers:
        config["n_jobs"] = min(args.max_workers, cpu_count)
    else:
        # Leave some cores free for system operations
        config["n_jobs"] = max(1, min(cpu_count - 1, 8))
    
    # Always enable parallel processing for large datasets
    config["parallel_clickmap_processing"] = True
    
    # Use HDF5 for datasets > 100k maps
    if "output_format" not in config or config["output_format"] == "auto":
        config["output_format"] = "hdf5" if total_maps > 100000 else "numpy"
    
    # Force CPU-only if requested
    if args.cpu_only:
        config["use_gpu_blurring"] = False
    elif "use_gpu_blurring" not in config:
        # Default to GPU if available
        try:
            import torch
            config["use_gpu_blurring"] = torch.cuda.is_available()
            if config["use_gpu_blurring"]:
                # Print GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                print(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
        except ImportError:
            config["use_gpu_blurring"] = False
    
    # Enable HDF5 compression for better I/O performance
    config["hdf5_compression"] = "gzip"
    config["hdf5_compression_level"] = 4  # Balance between speed and size
    
    # Ensure all directories exist
    output_dir = config["assets"]
    image_output_dir = config["example_image_output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, config["experiment_name"]), exist_ok=True)
    
    # Print optimization settings
    print("\nOptimized settings for processing:")
    print(f"- Dataset size: {total_maps} maps, {total_unique_images} images")
    print(f"- Chunk size: {config['chunk_size']} images per chunk")
    print(f"- GPU batch size: {config['gpu_batch_size']}")
    print(f"- CPU workers: {config['n_jobs']}")
    print(f"- Using GPU: {config.get('use_gpu_blurring', False)}")
    print(f"- Output format: {config['output_format']}")
    print(f"- Memory usage at start: {get_memory_usage():.2f} MB\n")

    # Other Args
    # blur_sigma_function = lambda x: np.sqrt(x)
    # blur_sigma_function = lambda x: x / 2
    blur_sigma_function = lambda x: x

    # Load config
    blur_size = config["blur_size"]
    blur_sigma = blur_sigma_function(blur_size)
    min_pixels = (2 * blur_size) ** 2  # Minimum number of pixels for a map to be included following filtering
    
    # Set output format (HDF5 by default for large datasets)
    output_format = config["output_format"]
    
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
            meta_grp.attrs["total_maps"] = total_maps
            meta_grp.attrs["creation_date"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            meta_grp.attrs["chunks_completed"] = 0  # Track progress
    
    # Calculate number of chunks based on unique images and chunk size
    chunk_size = config["chunk_size"]
    num_chunks = (total_unique_images + chunk_size - 1) // chunk_size
    
    # Determine start and end chunks based on command line arguments
    start_chunk = max(0, args.start_chunk - 1)  # Convert to 0-indexed
    end_chunk = args.end_chunk if args.end_chunk is not None else num_chunks
    
    # Display restart information if needed
    if start_chunk > 0:
        print(f"\nRestarting from chunk {start_chunk + 1} (skipping chunks 1-{start_chunk})")
    
    # Process all data in chunks
    all_final_clickmaps = {}
    all_final_keep_index = []
    
    # Use a simple progress tracking system with tqdm - prettier hierarchy
    print(f"\nProcessing {total_unique_images} unique images in {num_chunks} chunks (processing chunks {start_chunk + 1}-{end_chunk})...")
    with tqdm(total=end_chunk-start_chunk, desc="├─ Processing chunks", position=0, leave=True, colour="blue") as pbar:
        for chunk_idx in range(start_chunk, end_chunk):
            # Report memory usage at the start of each chunk
            mem_usage = get_memory_usage()
            print(f"\n├─ Memory usage: {mem_usage:.2f} MB")
            
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_unique_images)
            
            # Use a clear, hierarchical format for chunk info
            print(f"├─ Chunk {chunk_idx + 1}/{num_chunks} ({chunk_start}-{chunk_end})")
            
            # Get the image_paths for this chunk
            chunk_image_paths = image_paths[chunk_start:chunk_end]
            
            # Filter clickme_data to only include maps for images in this chunk
            chunk_data = clickme_data[clickme_data['image_path'].isin(chunk_image_paths)]
            
            l = len(chunk_data)       
            print(f"│  ├─ Processing {l} maps for {len(chunk_image_paths)} unique images")

            # Process clickmap files in parallel
            print(f"│  ├─ Processing clickmap files...")
            
            # Always use parallel processing for large datasets
            clickmaps, ccounts = utils.process_clickmap_files_parallel(
                clickme_data=chunk_data,
                image_path=config["image_path"],
                file_inclusion_filter=config["file_inclusion_filter"],
                file_exclusion_filter=config["file_exclusion_filter"],
                min_clicks=config["min_clicks"],
                max_clicks=config["max_clicks"],
                n_jobs=config["n_jobs"])
            
            # Apply filters in parallel if possible
            if config["class_filter_file"]:
                print(f"│  ├─ Filtering classes...")
                clickmaps = utils.filter_classes(
                    clickmaps=clickmaps,
                    class_filter_file=config["class_filter_file"])
            
            if config["participant_filter"]:
                print(f"│  ├─ Filtering participants...")
                clickmaps = utils.filter_participants(clickmaps)
            
            # Get GPU batch size and workers from config
            gpu_batch_size = config.get("gpu_batch_size", 512)
            n_jobs = config.get("n_jobs", 8)
            gpu_timeout = config.get("gpu_timeout", 900)  # 15 minutes timeout
            
            # Process using optimized method based on available resources
            use_gpu_blurring = config.get("use_gpu_blurring", False)
            
            if use_gpu_blurring:
                # Use optimized sub-chunking for better GPU utilization
                sub_chunk_size = min(1000, max(100, len(clickmaps) // 10))
                print(f"│  ├─ Processing with GPU in {(len(clickmaps) + sub_chunk_size - 1) // sub_chunk_size} sub-chunks of {sub_chunk_size} images")
                
                # Initialize sub-chunk results
                all_sub_clickmaps = []
                all_sub_keep_index = []
                chunk_final_clickmaps = {}
                
                # Create list of image keys
                img_keys = list(clickmaps.keys())
                
                # Process sub-chunks
                for i in range(0, len(img_keys), sub_chunk_size):
                    sub_keys = img_keys[i:i+sub_chunk_size]
                    sub_clickmaps = {k: clickmaps[k] for k in sub_keys}
                    
                    print(f"│  │  ├─ Sub-chunk {i//sub_chunk_size + 1}/{(len(clickmaps) + sub_chunk_size - 1) // sub_chunk_size}")
                    
                    # Try GPU processing first
                    try:
                        sub_final_clickmaps, sub_all_clickmaps, sub_categories, sub_final_keep_index = utils.prepare_maps_with_gpu_batching(
                            final_clickmaps=[sub_clickmaps],
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
                        chunk_final_clickmaps.update(sub_final_clickmaps)
                    
                    except Exception as e:
                        print(f"│  │  ├─ GPU processing failed: {e}")
                        print(f"│  │  ├─ Falling back to CPU...")
                        
                        try:
                            # Fall back to CPU processing
                            sub_final_clickmaps, sub_all_clickmaps, sub_categories, sub_final_keep_index = utils.prepare_maps_with_progress(
                                final_clickmaps=[sub_clickmaps],
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
                            chunk_final_clickmaps.update(sub_final_clickmaps)
                        except Exception as e:
                            print(f"│  │  ├─ CPU processing also failed: {e}")
                            continue
                    
                    # Save results periodically to free memory if many images
                    if len(all_sub_clickmaps) > args.save_interval:
                        print(f"│  │  ├─ Intermediate save to free memory...")
                        
                        if output_format == "hdf5":
                            # Save to HDF5 file
                            utils.save_clickmaps_to_hdf5(
                                all_clickmaps=all_sub_clickmaps,
                                final_keep_index=all_sub_keep_index,
                                hdf5_path=hdf5_path,
                                n_jobs=n_jobs,
                                compression=config.get("hdf5_compression"),
                                compression_level=config.get("hdf5_compression_level", 4)
                            )
                        else:
                            # Save to numpy files
                            utils.save_clickmaps_parallel(
                                all_clickmaps=all_sub_clickmaps,
                                final_keep_index=all_sub_keep_index,
                                output_dir=output_dir,
                                experiment_name=config["experiment_name"],
                                image_path=config["image_path"],
                                n_jobs=n_jobs,
                                file_inclusion_filter=config.get("file_inclusion_filter")
                            )
                        
                        # Clear memory
                        all_sub_clickmaps = []
                        all_sub_keep_index = []
                        gc.collect()
                
                # Set results for final saving
                chunk_all_clickmaps = all_sub_clickmaps
                chunk_final_keep_index = all_sub_keep_index
                chunk_categories = []  # Not used for saving
                
            else:
                # Use the CPU-based processing
                print(f"│  ├─ Processing with CPU (n_jobs={n_jobs})...")
                try:
                    chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.prepare_maps_with_progress(
                        final_clickmaps=[clickmaps],
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
                    print(f"│  ├─ Skipping this chunk...")
                    # Initialize with empty results
                    chunk_final_clickmaps = {}
                    chunk_all_clickmaps = []
                    chunk_categories = []
                    chunk_final_keep_index = []

            # Apply mask filtering if needed
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
                print(f"│  ├─ Saving {len(chunk_final_keep_index)} processed maps...")
                
                if output_format == "hdf5":
                    # Use optimized HDF5 saving with compression
                    saved_count = utils.save_clickmaps_to_hdf5(
                        all_clickmaps=chunk_all_clickmaps,
                        final_keep_index=chunk_final_keep_index,
                        hdf5_path=hdf5_path,
                        n_jobs=n_jobs,
                        compression=config.get("hdf5_compression"),
                        compression_level=config.get("hdf5_compression_level", 4)
                    )
                else:
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
                print(f"│  ├─ Saved {saved_count} files")
            
            # Merge minimal results for final statistics
            # Just keep references to image names instead of full data
            for name in chunk_final_clickmaps:
                all_final_clickmaps[name] = chunk_final_clickmaps[name]
            all_final_keep_index.extend(chunk_final_keep_index)
            
            # Free memory
            del chunk_data, clickmaps, ccounts
            del chunk_all_clickmaps, chunk_categories, chunk_final_keep_index
            # Don't delete chunk_final_clickmaps as we need the references
            
            # Force garbage collection
            gc.collect()
            
            # Update progress in HDF5 file
            if output_format == "hdf5":
                try:
                    with h5py.File(hdf5_path, 'a') as f:
                        f["metadata"].attrs["chunks_completed"] = chunk_idx + 1
                        f["metadata"].attrs["last_updated"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
                except Exception as e:
                    print(f"Warning: Could not update HDF5 metadata: {e}")
            
            # Update main progress bar
            pbar.update(1)
            
            # Report memory after chunk processing
            mem_usage_after = get_memory_usage()
            print(f"│  └─ Chunk {chunk_idx + 1} complete. Memory: {mem_usage_after:.2f} MB (delta: {mem_usage_after - mem_usage:.2f} MB)\n")
    
    # Get median number of clicks from the combined results
    print("Calculating medians...")
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
                print(f"Image not found for {img_name}")
                continue
                
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

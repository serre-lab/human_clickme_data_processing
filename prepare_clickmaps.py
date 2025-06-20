import os, sys
import numpy as np
from PIL import Image
import json
import pandas as pd
import argparse
from matplotlib import pyplot as plt
from src import utils
import h5py
import gc
import torch


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
    parser.add_argument('--time_based_bins', action='store_true', help='Enable time based bin threshold instead of count based')
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
    
    # Add time_based_bins to config if not present
    if "time_based_bins" not in config:
        config["time_based_bins"] = args.time_based_bins

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
    
    # Setup HDF5 file for both paths
    hdf5_path = os.path.join(output_dir, f"{config['experiment_name']}.h5")
    print(f"Saving results to file: {hdf5_path}")
    with h5py.File(hdf5_path, 'w') as f:
        f.create_group("clickmaps")
        f.create_group("click_counts")
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
    
    # Determine whether to use batch processing
    use_batch_processing = config.get("use_batch_processing", total_maps > 100000)
    
    # Processing mode decision
    if use_batch_processing:
        # Batch processing for large datasets
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
            batch_hdf5_path = os.path.join(output_dir, f"{config['experiment_name']}{batch_suffix}.h5")
            print(f"Saving batch results to HDF5 file: {batch_hdf5_path}")
            
            with h5py.File(batch_hdf5_path, 'w') as f:
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
            
            # Process this batch of clickmaps with multi-thresh GPU
            print(f"Processing batch with GPU (batch size: {config['gpu_batch_size']})...")
            batch_final_clickmaps, batch_all_clickmaps, batch_categories, batch_final_keep_index, batch_click_counts, batch_clickmap_bins = utils.process_all_maps_multi_thresh_gpu(
                clickmaps=batch_clickmaps,
                config=config,
                metadata=metadata,
                create_clickmap_func=create_clickmap_func,
                fast_duplicate_detection=fast_duplicate_detection,
                time_based_bins = config["time_based_bins"]
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
                with h5py.File(batch_hdf5_path, 'a') as f:
                    for img_name, count in batch_click_counts.items():
                        dataset_name = img_name.replace('/', '_')
                        f["click_counts"].create_dataset(dataset_name, data=count)
                
                # Save with appropriate method based on output format
                if output_format == "hdf5":
                    # Use optimized HDF5 saving with compression
                    saved_count = utils.save_clickmaps_to_hdf5(
                        all_clickmaps=batch_all_clickmaps,
                        final_keep_index=batch_final_keep_index,
                        hdf5_path=batch_hdf5_path,
                        n_jobs=config["n_jobs"],
                        compression=config.get("hdf5_compression"),
                        compression_level=config.get("hdf5_compression_level", 0),
                        clickmap_bins=batch_clickmap_bins
                    )
                    
                    # Also save individual NPY files for compatibility
                    # print("Saving individual NPY files in addition to HDF5...")
                    # npy_saved_count = utils.save_clickmaps_parallel(
                    #     all_clickmaps=batch_all_clickmaps,
                    #     final_keep_index=batch_final_keep_index,
                    #     output_dir=output_dir,
                    #     experiment_name=f"{config['experiment_name']}{batch_suffix}",
                    #     image_path=config["image_path"],
                    #     n_jobs=config["n_jobs"],
                    #     file_inclusion_filter=config.get("file_inclusion_filter")
                    # )
                else:
                    # Use optimized HDF5 saving with compression
                    saved_count = utils.save_clickmaps_to_hdf5(
                        all_clickmaps=batch_all_clickmaps,
                        final_keep_index=batch_final_keep_index,
                        hdf5_path=batch_hdf5_path,
                        n_jobs=config["n_jobs"],
                        compression=config.get("hdf5_compression"),
                        compression_level=config.get("hdf5_compression_level", 0),
                        clickmap_bins=batch_clickmap_bins
                    )
                    # Use parallel saving for non-HDF5 format
                    saved_count = utils.save_clickmaps_parallel(
                        all_clickmaps=batch_all_clickmaps,
                        final_keep_index=batch_final_keep_index,
                        output_dir=output_dir,
                        experiment_name=f"{config['experiment_name']}{batch_suffix}",
                        image_path=config["image_path"],
                        n_jobs=config["n_jobs"],
                        file_inclusion_filter=config.get("file_inclusion_filter")
                    )
                    
                # Save clickmap bins
                batch_bins_path = os.path.join(output_dir, f"{config['processed_clickmap_bins'].replace('.npy', '')}{batch_suffix}.npy")
                np.save(batch_bins_path, batch_clickmap_bins)
                
                print(f"Saved {saved_count} files in batch {batch_num+1}")
                
                # Calculate batch medians and update global medians
                batch_medians = utils.get_medians(batch_final_clickmaps, 'image', thresh=config["percentile_thresh"])
                all_medians['image'].update(batch_medians.get('image', {}))
                
                batch_cat_medians = utils.get_medians(batch_final_clickmaps, 'category', thresh=config["percentile_thresh"])
                for cat, val in batch_cat_medians.get('category', {}).items():
                    if cat not in all_medians['category']:
                        all_medians['category'][cat] = []
                    all_medians['category'][cat].append(val)
                
                batch_all_medians = utils.get_medians(batch_final_clickmaps, 'all', thresh=config["percentile_thresh"])
                for k, v in batch_all_medians.items():
                    if k not in all_medians['all']:
                        all_medians['all'][k] = []
                    all_medians['all'][k].append(v)
            
            # Free memory after each batch
            del batch_clickmaps, batch_final_clickmaps, batch_all_clickmaps, batch_categories, batch_final_keep_index, batch_click_counts
            gc.collect()
            if config["use_gpu_blurring"]:
                torch.cuda.empty_cache()
            
            print(f"Memory usage after batch {batch_num+1}: {utils.get_memory_usage():.2f} MB")
        
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
        


        # Save individual click counts to their own directory
        for img_name, count in all_click_counts.items():
            count_file = os.path.join(click_counts_dir, f"{img_name.replace('/', '_')}.npy")
            np.save(count_file, count)
        
        print(f"Processed and saved a total of {processed_images_count} images across {num_batches} batches")
        
        # Set finals for visualization
        if 'batch_final_clickmaps' in locals():
            final_clickmaps = batch_final_clickmaps
        else:
            final_clickmaps = {}
            
    else:
        # Sequential processing for smaller datasets
        print(f"Processing with GPU in a single batch (batch size: {config['gpu_batch_size']})...")
        final_clickmaps, all_clickmaps, categories, final_keep_index, click_counts, clickmap_bins = utils.process_all_maps_multi_thresh_gpu(
            clickmaps=clickmaps,
            config=config,
            metadata=metadata,
            create_clickmap_func=create_clickmap_func,
            fast_duplicate_detection=fast_duplicate_detection,
            time_based_bins = config["time_based_bins"],
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
            
            # Save click counts to their dedicated directory
            for img_name, count in click_counts.items():
                count_file = os.path.join(click_counts_dir, f"{img_name.replace('/', '_')}.npy")
                np.save(count_file, count)
            # Save with appropriate method based on output format
            if output_format == "hdf5":
                # Use optimized HDF5 saving with compression
                saved_count = utils.save_clickmaps_to_hdf5(
                    all_clickmaps=all_clickmaps,
                    final_keep_index=final_keep_index,
                    hdf5_path=hdf5_path,
                    n_jobs=config["n_jobs"],
                    compression=config.get("hdf5_compression"),
                    compression_level=config.get("hdf5_compression_level", 0),
                    clickmap_bins=clickmap_bins
                )
                
                # Also save individual NPY files for compatibility
                # print("Saving individual NPY files in addition to HDF5...")
                # npy_saved_count = utils.save_clickmaps_parallel(
                #     all_clickmaps=all_clickmaps,
                #     final_keep_index=final_keep_index,
                #     output_dir=output_dir,
                #     experiment_name=config["experiment_name"],
                #     image_path=config["image_path"],
                #     n_jobs=config["n_jobs"],
                #     file_inclusion_filter=config.get("file_inclusion_filter")
                # )
            else:
                # Use parallel saving for non-HDF5 format
                saved_count = utils.save_clickmaps_to_hdf5(
                    all_clickmaps=all_clickmaps,
                    final_keep_index=final_keep_index,
                    hdf5_path=hdf5_path,
                    n_jobs=config["n_jobs"],
                    compression=config.get("hdf5_compression"),
                    compression_level=config.get("hdf5_compression_level", 0),
                    clickmap_bins=clickmap_bins
                )
                saved_count = utils.save_clickmaps_parallel(
                    all_clickmaps=all_clickmaps,
                    final_keep_index=final_keep_index,
                    output_dir=output_dir,
                    experiment_name=config["experiment_name"],
                    image_path=config["image_path"],
                    n_jobs=config["n_jobs"],
                    file_inclusion_filter=config.get("file_inclusion_filter")
                )
            
            print(f"Saved {saved_count} files and their click counts")
            
            # Get median number of clicks from the results
            print("Calculating medians...")
            percentile_thresh = config["percentile_thresh"]
            medians = utils.get_medians(final_clickmaps, 'image', thresh=percentile_thresh)
            medians.update(utils.get_medians(final_clickmaps, 'category', thresh=percentile_thresh))
            medians.update(utils.get_medians(final_clickmaps, 'all', thresh=percentile_thresh))
            
            # Save medians
            medians_json = json.dumps(medians, indent=4)
            with open(os.path.join(output_dir, config["processed_medians"]), 'w') as f:
                f.write(medians_json)

            # Save clickmap bins
            np.save(os.path.join(output_dir, config["processed_clickmap_bins"]), clickmap_bins)
        
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
                            count_path = os.path.join(output_dir, config["experiment_name"], f"{img_name.replace('/', '_')}_count.npy")
                            click_count = np.load(count_path) if os.path.exists(count_path) else None
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
                    
                    # If not found, try the dedicated click counts directory
                    if click_count is None:
                        count_path = os.path.join(click_counts_dir, f"{img_name.replace('/', '_')}.npy")
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
    
    print(f"\nProcessing complete! Final memory usage: {utils.get_memory_usage():.2f} MB")

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


if __name__ == "__main__":
    # Add command line arguments
    parser = argparse.ArgumentParser(description="Process clickme data for modeling")
    parser.add_argument('config', nargs='?', help='Path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable additional debug output')
    parser.add_argument('--verbose', action='store_true', help='Show detailed progress for GPU processing')
    parser.add_argument('--gpu-batch-size', type=int, default=None, help='Override GPU batch size')
    parser.add_argument('--max-workers', type=int, default=None, help='Maximum number of CPU workers')
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
        # Determine best batch size based on dataset characteristics
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
        hdf5_path = os.path.join(output_dir, f"{config['experiment_name']}.h5")
        print(f"Saving results to HDF5 file: {hdf5_path}")
        with h5py.File(hdf5_path, 'w') as f:
            f.create_group("clickmaps")
            meta_grp = f.create_group("metadata")
            meta_grp.attrs["total_unique_images"] = total_unique_images
            meta_grp.attrs["total_maps"] = total_maps
            meta_grp.attrs["creation_date"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Print optimization settings
    print("\nProcessing settings:")
    print(f"- Dataset size: {total_maps} maps, {total_unique_images} images")
    print(f"- GPU batch size: {config['gpu_batch_size']}")
    print(f"- CPU workers: {config['n_jobs']}")
    print(f"- Output format: {config['output_format']}")
    print(f"- Memory usage at start: {get_memory_usage():.2f} MB\n")

    # Processing parameters
    blur_sigma_function = lambda x: x
    blur_size = config["blur_size"]
    blur_sigma = blur_sigma_function(blur_size)
    min_pixels = (2 * blur_size) ** 2
    
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
    import pdb; pdb.set_trace()
    
    # Apply filters if necessary
    if config["class_filter_file"]:
        print("Filtering classes...")
        clickmaps = utils.filter_classes(
            clickmaps=clickmaps,
            class_filter_file=config["class_filter_file"])
    
    if config["participant_filter"]:
        print("Filtering participants...")
        clickmaps = utils.filter_participants(clickmaps)
    
    # GPU processing parameters
    gpu_batch_size = config.get("gpu_batch_size", 512)
    n_jobs = config.get("n_jobs", 8)
    gpu_timeout = config.get("gpu_timeout", 900)  # 15 minutes timeout
    
    print(f"Processing with GPU (batch size: {gpu_batch_size})...")
    try:
        final_clickmaps, all_clickmaps, categories, final_keep_index = utils.prepare_maps_with_gpu_batching(
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
            batch_size=gpu_batch_size,
            timeout=gpu_timeout,
            verbose=args.verbose or args.debug,
            create_clickmap_func=create_clickmap_func,
            fast_duplicate_detection=fast_duplicate_detection)
    except Exception as e:
        print(f"GPU processing failed: {e}")
        print("Exiting...")
        sys.exit(1)

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
    
    # Save results
    if final_keep_index:
        print(f"Saving {len(final_keep_index)} processed maps...")
        
        if output_format == "hdf5":
            # Use optimized HDF5 saving with compression
            saved_count = utils.save_clickmaps_to_hdf5(
                all_clickmaps=all_clickmaps,
                final_keep_index=final_keep_index,
                hdf5_path=hdf5_path,
                n_jobs=n_jobs,
                compression=config.get("hdf5_compression"),
                compression_level=config.get("hdf5_compression_level", 4)
            )
        else:
            # Use parallel saving
            saved_count = utils.save_clickmaps_parallel(
                all_clickmaps=all_clickmaps,
                final_keep_index=final_keep_index,
                output_dir=output_dir,
                experiment_name=config["experiment_name"],
                image_path=config["image_path"],
                n_jobs=n_jobs,
                file_inclusion_filter=config.get("file_inclusion_filter")
            )
        print(f"Saved {saved_count} files")
    
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

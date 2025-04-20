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
            # Note that if we are measuring ceiling we need to keep all maps ^^ change above.
            categories.append(key.split("/")[0])
            keep_index.append(key)
            final_clickmaps[key] = trials
    
    if not all_clickmaps:
        print("No valid clickmaps to process")
        return {}, [], [], []
    
    # Step 2: Prepare for batch blurring on GPU
    total_maps = len(all_clickmaps)
    print(f"Preparing to blur {total_maps} clickmaps using GPU...")
    
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
    return final_clickmaps, all_clickmaps, categories, keep_index


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
        # Process in batches to avoid memory issues
        max_batch_size = config.get("batch_size", 50000)  # Default to 50k images per batch
        num_batches = (total_unique_images + max_batch_size - 1) // max_batch_size
        
        print(f"Processing dataset in {num_batches} batches of up to {max_batch_size} images each")
        
        # Split clickmaps into batches
        all_keys = list(clickme_data.keys())
        
        for batch_num in range(num_batches):
            print(f"\n--- Processing batch {batch_num+1}/{num_batches} ---")
            
            # Calculate batch indices
            start_idx = batch_num * max_batch_size
            end_idx = min(start_idx + max_batch_size, total_unique_images)
            batch_keys = all_keys[start_idx:end_idx]
            
            # Create batch-specific clickmaps dictionary
            batch_clickmaps = {k: clickme_data[k] for k in batch_keys}
            batch_size = len(batch_clickmaps)
            
            # Create batch-specific HDF5 file with suffix
            batch_suffix = f"_batch{batch_num+1:03d}" if num_batches > 1 else ""
            hdf5_path = os.path.join(output_dir, f"{config['experiment_name']}{batch_suffix}.h5")
            print(f"Saving batch results to HDF5 file: {hdf5_path}")
            
            with h5py.File(hdf5_path, 'w') as f:
                f.create_group("clickmaps")
                meta_grp = f.create_group("metadata")
                meta_grp.attrs["batch_number"] = batch_num + 1
                meta_grp.attrs["total_batches"] = num_batches
                meta_grp.attrs["batch_size"] = batch_size
                meta_grp.attrs["total_unique_images"] = total_unique_images
                meta_grp.attrs["total_maps"] = total_maps
                meta_grp.attrs["creation_date"] = np.bytes_(pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # Process this batch of clickmaps
            print(f"Processing with GPU (batch size: {config['gpu_batch_size']})...")
            
            batch_final_clickmaps, batch_all_clickmaps, batch_categories, batch_final_keep_index = process_all_maps_gpu(
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
            
            # Save results for this batch
            if batch_final_keep_index:
                print(f"Saving {len(batch_final_keep_index)} processed maps for batch {batch_num+1}...")
                
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
            
            # Free memory after each batch
            del batch_clickmaps, batch_final_clickmaps, batch_all_clickmaps, batch_categories, batch_final_keep_index
            gc.collect()
            if config["use_gpu_blurring"]:
                torch.cuda.empty_cache()
            
            print(f"Memory usage after batch {batch_num+1}: {get_memory_usage():.2f} MB")
    else:
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
            try:
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

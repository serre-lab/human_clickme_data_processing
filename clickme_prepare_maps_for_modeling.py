import os, sys
import numpy as np
from PIL import Image
import json
import pandas as pd
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

    # Get config file
    config_file = utils.get_config(sys.argv)

    # Other Args
    # blur_sigma_function = lambda x: np.sqrt(x)
    # blur_sigma_function = lambda x: x / 2
    blur_sigma_function = lambda x: x

    # Load config
    config = utils.process_config(config_file)
    clickme_data = utils.process_clickme_data(
        config["clickme_data"],
        config["filter_mobile"])
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
        if len(clickme_data) > large_dataset_threshold:
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
    
    # Group clickme_data by image_path to ensure all maps for an image are processed together
    unique_images = clickme_data['image_path'].unique()
    total_unique_images = len(unique_images)
    
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
    
    # Use a simple progress tracking system with tqdm - prettier hierarchy
    print(f"\nProcessing {total_unique_images} unique images in {num_chunks} chunks...")
    with tqdm(total=num_chunks, desc="├─ Processing chunks", position=0, leave=True, colour="blue") as pbar:
        for chunk_idx in range(num_chunks):
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
            
            # Process maps for this chunk with our custom progress wrapper
            use_parallel = config.get("parallel_prepare_maps", True)
            n_jobs = -1 if use_parallel else 1
            parallel_text = "parallel" if use_parallel else "serial"
            
            # Get GPU batch size from config or use default
            gpu_batch_size = config.get("gpu_batch_size", 32)
            
            # Use GPU optimized blurring by default, can be disabled with use_gpu_blurring=False
            use_gpu_blurring = config.get("use_gpu_blurring", True)
            
            if use_gpu_blurring:
                print(f"│  ├─ Preparing maps with GPU-optimized blurring (batch_size={gpu_batch_size}, n_jobs={n_jobs})...")
            else:
                print(f"│  ├─ Preparing maps ({parallel_text}, n_jobs={n_jobs})...")
            
            # Add debug print to check if chunk_clickmaps is empty
            if not clickmaps:
                raise ValueError(f"│  ├─ WARNING: No images to process after filtering! Check your filter settings.")
            else:
                # Choose the appropriate processing function based on config
                if use_gpu_blurring:
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
                        create_clickmap_func=create_clickmap_func if use_cython else None)
                else:
                    # Use the original CPU-based processing
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
                        create_clickmap_func=create_clickmap_func if use_cython else None)
                
            # Apply mask filtering if needed
            if config["mask_dir"]:
                print(f"│  ├─ Applying mask filtering...")
                masks = utils.load_masks(config["mask_dir"])
                chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.filter_for_foreground_masks(
                    final_clickmaps=chunk_final_clickmaps,
                    all_clickmaps=chunk_all_clickmaps,
                    categories=chunk_categories,
                    masks=masks,
                    mask_threshold=config["mask_threshold"])
            
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
                        n_jobs=n_jobs
                    )
                    print(f"│  │  └─ Saved {saved_count} files in parallel")
                else:
                    # Use sequential saving with tqdm
                    with tqdm(total=len(chunk_final_keep_index), desc="│  │  ├─ Saving files", 
                             position=1, leave=False, colour="cyan") as save_pbar:
                        for j, img_name in enumerate(chunk_final_keep_index):
                            # if not os.path.exists(os.path.join(config["image_path"], img_name)):
                            if (
                                not os.path.join(config["image_path"].replace(config["file_inclusion_filter"] + os.path.sep, ""), img_name)  # New for Co3d train
                                and not os.path.exists(os.path.join(config["image_path"], img_name))  # Legacy
                            ):
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

import os, sys
import numpy as np
from PIL import Image
import json
import pandas as pd
from matplotlib import pyplot as plt
from src import utils
from tqdm import tqdm

def sample_half_pos(point_lists, num_samples=100):
    num_pos = {}
    for image in point_lists:
        sample_nums = []
        for i in range(num_samples):
            clickmaps = point_lists[image]
            all_maps = []
            map_indices = list(range(len(clickmaps)))
            random_indices = np.random.choice(map_indices, int(len(clickmaps)//2))
            s1 = []
            s2 = []
            for j, clickmap in enumerate(clickmaps):
                if j in random_indices:
                    s1 += clickmap
                else:
                    s2 += clickmap
            sample_nums += [len(set(s1)), len(set(s2))]
        num_pos[image] = np.mean(sample_nums)
    return num_pos
    
def get_num_pos(point_lists):
    num_pos = {}
    for image in point_lists:
        clickmaps = point_lists[image]
        all_maps = []
        for clickmap in clickmaps:
            all_maps += clickmap
        all_maps = set(all_maps)
        num_pos[image] = len(all_maps)
    return num_pos

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
    # Instead of loading all 5.3M+ images at once, process in manageable batches
    print(f"Processing clickme data in chunks...")
    
    # Determine whether to use parallel processing
    # For very large datasets, sometimes serial processing with efficient chunking is faster
    # due to reduced overhead and better memory management
    chunk_size = 10000  # Adjust based on available memory
    total_images = len(clickme_data)
    
    # Process all data in chunks
    all_final_clickmaps = {}
    all_final_keep_index = []
    
    # Calculate number of chunks
    num_chunks = (total_images + chunk_size - 1) // chunk_size
    
    # Custom wrapper for prepare_maps_parallel to add progress bar
    def prepare_maps_with_progress(final_clickmaps, **kwargs):
        """
        Create a wrapper that shows progress for prepare_maps_parallel.
        
        Since joblib's parallel processing is difficult to track directly,
        we'll use a simple spinner animation to show that processing is happening.
        """
        total_images = len(final_clickmaps)
        
        # Display information about the processing
        print(f"│  ├─ Processing {total_images} images...")
        
        # Simple spinner animation characters
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        # Use a context manager to handle terminal output and cleanup
        import sys
        import time
        import threading
        from contextlib import contextmanager
        
        @contextmanager
        def spinner_animation():
            # Create a flag to control the spinner animation
            stop_spinner = threading.Event()
            
            # Function to animate the spinner
            def spin():
                i = 0
                while not stop_spinner.is_set():
                    # Print the spinner character and processing message
                    sys.stdout.write(f"\r│  ├─ {spinner[i]} Processing images... ")
                    sys.stdout.flush()
                    time.sleep(0.1)
                    i = (i + 1) % len(spinner)
                
                # Clear the line when done
                sys.stdout.write("\r" + " " * 50 + "\r")
                sys.stdout.flush()
            
            # Start the spinner in a separate thread
            spinner_thread = threading.Thread(target=spin)
            spinner_thread.daemon = True
            spinner_thread.start()
            
            try:
                # Return control to the caller
                yield
            finally:
                # Stop the spinner when the context exits
                stop_spinner.set()
                spinner_thread.join()
        
        # Use the spinner animation while processing
        with spinner_animation():
            # Call the original function
            result = utils.prepare_maps_parallel(final_clickmaps=final_clickmaps, **kwargs)
        
        # Print completion message
        print(f"│  ├─ ✓ Processed {total_images} images")
        
        return result
    
    # Use a simple progress tracking system with tqdm - prettier hierarchy
    print("\nProcessing clickme data in chunks...")
    with tqdm(total=num_chunks, desc="├─ Processing chunks", position=0, leave=True, colour="blue") as pbar:
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, total_images)
            
            # Use a clear, hierarchical format for chunk info
            print(f"\n├─ Chunk {chunk_idx + 1}/{num_chunks} ({chunk_start}-{chunk_end})")
            
            # Create a DataFrame that process_clickmap_files can work with
            chunk_data = {k: clickme_data[k] for k in list(clickme_data.keys())[chunk_start:chunk_end]}
            image_paths = list(chunk_data.keys())
            clicks_data = list(chunk_data.values())
            
            print(f"│  ├─ Debug: Initial chunk has {len(chunk_data)} images")
            
            # Create a DataFrame with explicit 'image_path' column
            chunk_df = pd.DataFrame({
                'image_path': image_paths,
                'clicks': clicks_data
            })
            
            # Process chunk data
            print(f"│  ├─ Processing clickmap files...")
            process_clickmap_files_func = utils.process_clickmap_files
            chunk_clickmaps, chunk_clickmap_counts = process_clickmap_files_func(
                clickme_data=chunk_df,
                image_path=config["image_path"],
                file_inclusion_filter=config["file_inclusion_filter"],
                file_exclusion_filter=config["file_exclusion_filter"],
                min_clicks=config["min_clicks"],
                max_clicks=config["max_clicks"])
                
            print(f"│  ├─ Debug: After processing clickmap files: {len(chunk_clickmaps)} images")
                
            # Apply all filters to the chunk
            if config["class_filter_file"]:
                print(f"│  ├─ Filtering classes...")
                chunk_clickmaps = utils.filter_classes(
                    clickmaps=chunk_clickmaps,
                    class_filter_file=config["class_filter_file"])
                print(f"│  ├─ Debug: After class filtering: {len(chunk_clickmaps)} images")

            if config["participant_filter"]:
                print(f"│  ├─ Filtering participants...")
                chunk_clickmaps = utils.filter_participants(chunk_clickmaps)
                print(f"│  ├─ Debug: After participant filtering: {len(chunk_clickmaps)} images")
            
            # Process maps for this chunk with our custom progress wrapper
            use_parallel = config.get("parallel_prepare_maps", True)
            n_jobs = -1 if use_parallel else 1
            parallel_text = "parallel" if use_parallel else "serial"
            print(f"│  ├─ Preparing maps ({parallel_text}, n_jobs={n_jobs})...")
            
            # Add debug print to check if chunk_clickmaps is empty
            if not chunk_clickmaps:
                print(f"│  ├─ WARNING: No images to process after filtering! Check your filter settings.")
                # Create empty results to avoid errors
                chunk_final_clickmaps = {}
                chunk_all_clickmaps = []
                chunk_categories = []
                chunk_final_keep_index = []
            else:
                # Use our custom progress wrapper
                chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = prepare_maps_with_progress(
                    final_clickmaps=chunk_clickmaps,
                    blur_size=blur_size,
                    blur_sigma=blur_sigma,
                    image_shape=config["image_shape"],
                    min_pixels=min_pixels,
                    min_subjects=config["min_subjects"],
                    metadata=metadata,
                    blur_sigma_function=blur_sigma_function,
                    center_crop=False,
                    n_jobs=n_jobs)
                
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
            with tqdm(total=len(chunk_final_keep_index), desc="│  │  ├─ Saving files", 
                     position=1, leave=False, colour="cyan") as save_pbar:
                for j, img_name in enumerate(chunk_final_keep_index):
                    if not os.path.exists(os.path.join(config["image_path"], img_name)):
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
            del chunk_data, chunk_df, chunk_clickmaps, chunk_clickmap_counts
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
            heatmap_path = os.path.join(output_dir, config["experiment_name"], f"{img_name.replace('/', '_')}.npy")
            if not os.path.exists(heatmap_path):
                print(f"Heatmap not found for {img_name}")
                continue
                
            hmp = np.load(heatmap_path)
            img = Image.open(os.path.join(config["image_path"], img_name))
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

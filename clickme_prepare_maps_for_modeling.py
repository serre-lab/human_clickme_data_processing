import os, sys
import numpy as np
from PIL import Image
import json
import pandas as pd
from matplotlib import pyplot as plt
from src import utils

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
            medians[image] = np.percentile(num_clicks, thresh)
    elif mode == 'category':
        for image in point_lists:
            category = image.split('/')[0]
            if category not in medians.keys():
                medians[category] = []
            clickmaps = point_lists[image]
            for clickmap in clickmaps:
                medians[category].append(len(clickmap))
        for category in medians:
            medians[category] = np.percentile(medians[category], thresh)
    elif mode == 'all':
        num_clicks = []
        for image in point_lists:
            clickmaps = point_lists[image]
            for clickmap in clickmaps:
                num_clicks.append(len(clickmap))
        medians['all'] = np.percentile(num_clicks, thresh)
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
    
    # Create a function to process chunks of data
    def process_chunk(chunk_start, chunk_end):
        chunk_data = {k: clickme_data[k] for k in list(clickme_data.keys())[chunk_start:chunk_end]}
        
        # Convert the dictionary to a DataFrame with the correct column name 'image_path'
        # instead of 'image_name' to match what process_clickmap_files expects
        chunk_df = pd.DataFrame([(k, v) for k, v in chunk_data.items()], 
                               columns=['image_path', 'clicks'])
        
        # Use serial processing for each chunk to avoid joblib overhead
        process_clickmap_files_func = utils.process_clickmap_files
        chunk_clickmaps, chunk_clickmap_counts = process_clickmap_files_func(
            clickme_data=chunk_df,  # Pass DataFrame instead of dictionary
            image_path=config["image_path"],
            file_inclusion_filter=config["file_inclusion_filter"],
            file_exclusion_filter=config["file_exclusion_filter"],
            min_clicks=config["min_clicks"],
            max_clicks=config["max_clicks"])
            
        # Apply all filters to the chunk
        if config["class_filter_file"]:
            chunk_clickmaps = utils.filter_classes(
                clickmaps=chunk_clickmaps,
                class_filter_file=config["class_filter_file"])

        if config["participant_filter"]:
            chunk_clickmaps = utils.filter_participants(chunk_clickmaps)
        
        # Process maps for this chunk
        prepare_maps_func = utils.prepare_maps  # Use serial version for each chunk
        chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = prepare_maps_func(
            final_clickmaps=chunk_clickmaps,
            blur_size=blur_size,
            blur_sigma=blur_sigma,
            image_shape=config["image_shape"],
            min_pixels=min_pixels,
            min_subjects=config["min_subjects"],
            metadata=metadata,
            blur_sigma_function=blur_sigma_function,
            center_crop=False)
            
        # Apply mask filtering if needed
        if config["mask_dir"]:
            masks = utils.load_masks(config["mask_dir"])
            chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = utils.filter_for_foreground_masks(
                final_clickmaps=chunk_final_clickmaps,
                all_clickmaps=chunk_all_clickmaps,
                categories=chunk_categories,
                masks=masks,
                mask_threshold=config["mask_threshold"])
        
        # Process and save directly instead of accumulating
        for j, img_name in enumerate(chunk_final_keep_index):
            if not os.path.exists(os.path.join(config["image_path"], img_name)):
                continue
                
            hmp = chunk_all_clickmaps[j]
            # Save directly to disk - don't accumulate in memory
            np.save(
                os.path.join(output_dir, config["experiment_name"], f"{img_name.replace('/', '_')}.npy"), 
                hmp
            )
        
        return chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index
    
    # Process all data in chunks
    all_final_clickmaps = {}
    all_final_keep_index = []
    
    # Use a simple progress tracking system
    for chunk_start in range(0, total_images, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_images)
        print(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_images + chunk_size - 1)//chunk_size} ({chunk_start}-{chunk_end})")
        
        chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories, chunk_final_keep_index = process_chunk(chunk_start, chunk_end)
        
        # Merge results (keeping minimal data in memory)
        all_final_clickmaps.update(chunk_final_clickmaps)
        all_final_keep_index.extend(chunk_final_keep_index)
        
        # Free memory
        del chunk_final_clickmaps, chunk_all_clickmaps, chunk_categories
        
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

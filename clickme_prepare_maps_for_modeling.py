import os, sys
import numpy as np
from PIL import Image
import json
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

    # Process files in serial
    clickmaps, clickmap_counts = utils.process_clickmap_files(
        clickme_data=clickme_data,
        file_inclusion_filter=config["file_inclusion_filter"],
        file_exclusion_filter=config["file_exclusion_filter"],
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"])

    # Prepare maps
    if config["debug"]:
        new_clickmaps = {}
        for k in config["display_image_keys"]:
            click_match = [k_ for k_ in clickmaps.keys() if k in k_]
            assert len(click_match) == 1, "Clickmap not found"
            new_clickmaps[click_match[0]] = clickmaps[click_match[0]]
        clickmaps = new_clickmaps
        # clickmaps = {k: v for idx, (k, v) in enumerate(clickmaps.items()) if idx < 1000}

    # Filter classes if requested
    if config["class_filter_file"]:
        clickmaps = utils.filter_classes(
            clickmaps=clickmaps,
            class_filter_file=config["class_filter_file"])

    # Filter participants if requested
    if config["participant_filter"]:
        clickmaps = utils.filter_participants(clickmaps)

    # Prepare maps
    final_clickmaps, all_clickmaps, categories, final_keep_index = utils.prepare_maps(
        final_clickmaps=clickmaps,
        blur_size=blur_size,
        blur_sigma=blur_sigma,
        image_shape=config["image_shape"],
        min_pixels=min_pixels,
        min_subjects=config["min_subjects"],
        metadata=metadata,
        blur_sigma_function=blur_sigma_function,
        center_crop=False)

    # Filter for foreground mask overlap if requested
    if config["mask_dir"]:
        masks = utils.load_masks(config["mask_dir"])
        final_clickmaps, all_clickmaps, categories, final_keep_index = utils.filter_for_foreground_masks(
            final_clickmaps=final_clickmaps,
            all_clickmaps=all_clickmaps,
            categories=categories,
            masks=masks,
            mask_threshold=config["mask_threshold"])
    # Visualize if requested
    sz_dict = {k: len(v) for k, v in final_clickmaps.items()}
    arg = np.argsort(list(sz_dict.values()))
    tops = np.asarray(list(sz_dict.keys()))[arg[-10:]]
    if config["display_image_keys"]:
        if config["display_image_keys"] == "auto":
            config["display_image_keys"] = tops
        # Load images
        img_heatmaps = {}
        fck = np.asarray([k for k in final_clickmaps.keys()])
        for image_file in config["display_image_keys"]:
            image_path = os.path.join(config["image_path"], image_file)
            image = Image.open(image_path)
            if metadata:
                click_match = [k_ for k_ in final_clickmaps.keys() if image_file in k_]
                assert len(click_match) == 1, "Clickmap not found"
                metadata_size = metadata[click_match[0]]
                image = image.resize(metadata_size)
            image_name = "_".join(image_path.split('/')[-2:])
            check = fck == image_file
            if check.any():
                find_key = np.where(check)[0][0]
                img_heatmaps[image_file] = {
                    "image": image,
                    "heatmap": all_clickmaps[find_key]
                }
            else:
                print("Image {} not found in final clickmaps".format(image_file))

        # And plot
        for k in img_heatmaps.keys():
            f = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.asarray(img_heatmaps[k]["image"]))
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_heatmaps[k]["heatmap"].mean(0))
            plt.axis("off")
            plt.savefig(os.path.join(image_output_dir, k.split(os.path.sep)[-1]))
            if config["debug"]:
                plt.show()
            plt.close()

    # Get median number of clicks
    percentile_thresh = config["percentile_thresh"]
    medians = get_medians(final_clickmaps, 'image', thresh=percentile_thresh)
    medians.update(get_medians(final_clickmaps, 'category', thresh=percentile_thresh))
    medians.update(get_medians(final_clickmaps, 'all', thresh=percentile_thresh))
    medians_json = json.dumps(medians, indent=4)

    # Save data
    # final_data = {k: v for k, v in zip(final_keep_index, all_clickmaps)}
    # np.save(
    #     os.path.join(output_dir, config["processed_clickme_file"]),
    #     final_data)
    with open(os.path.join(output_dir, config["processed_medians"]), 'w') as f:
        f.write(medians_json)

    img_heatmaps = {}
    for i, img_name in enumerate(final_keep_index):
        if not os.path.exists(os.path.join(config["image_path"], img_name)):
            print(os.path.join(config["image_path"], img_name))
            continue
        if img_name not in final_clickmaps.keys():
            continue
        hmp = all_clickmaps[i]
        img = Image.open(os.path.join(config["image_path"], img_name))
        if metadata:
            click_match = [k_ for k_ in final_clickmaps.keys() if img_name in k_]
            assert len(click_match) == 1, "Clickmap not found"
            metadata_size = metadata[click_match[0]]
            img = img.resize(metadata_size)

        img_heatmaps[img_name] = {"image":img, "heatmap":hmp}
    print(len(img_heatmaps))
    np.savez(os.path.join(output_dir,  config["processed_clickme_file"]), 
            **img_heatmaps
            )
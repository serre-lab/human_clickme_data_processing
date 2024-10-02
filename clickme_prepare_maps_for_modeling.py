import os, sys
import numpy as np
import pandas as pd
from PIL import Image
import json
from matplotlib import pyplot as plt
import utils


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
    debug = False
    output_dir = "assets"
    image_output_dir = "clickme_test_images"
    percentile_thresh = 50
    center_crop = False

    # Load config
    config = utils.process_config(config_file)
    clickme_data = utils.process_clickme_data(config["clickme_data"])
    blur_size = config["blur_size"]
    blur_sigma = np.sqrt(blur_size)
    min_pixels = (2 * blur_size) ** 2  # Minimum number of pixels for a map to be included following filtering
    del config["experiment_name"], config["clickme_data"]

    # Start processing
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Process files in serial
    clickmaps, clickmap_counts = utils.process_clickmap_files(
        clickme_data=clickme_data,
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"])
    # srt = np.argsort(clickmap_counts)[-10:]
    # fls = np.asarray([k for k in clickmaps.keys()])[srt]
    # print(fls)

    # Prepare maps
    final_clickmaps, all_clickmaps, categories, _ = utils.prepare_maps(
        final_clickmaps=clickmaps,
        blur_size=blur_size,
        blur_sigma=blur_sigma,
        image_shape=config["image_shape"],
        min_pixels=min_pixels,
        min_subjects=config["min_subjects"],
        center_crop=center_crop)
        
    # Visualize if requested
    if config["display_image_keys"]:
        # Load images
        # images, image_names = [], []
        img_heatmaps = {}
        # for image_file in final_clickmaps.keys():
        for image_file in config["display_image_keys"]:
            image_path = os.path.join(config["image_dir"], image_file)
            image = Image.open(image_path)
            image_name = "_".join(image_path.split('/')[-2:])
            # images.append(image)
            # image_names.append(image_file)
            img_heatmaps[image_file] = {
                "image": image,
                "heatmap": final_clickmaps[image_file]
            }
            # image_names.append(image_name)
        # Package into legacy format
        # img_heatmaps = {k: {"image": image, "heatmap": heatmap} for (k, image, heatmap) in zip(final_clickmaps.keys(), images, all_clickmaps)}

        # And plot
        for k in config["display_image_keys"]:
            f = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.asarray(img_heatmaps[k]["image"])[:config["image_shape"][0], :config["image_shape"][1]])
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_heatmaps[k]["heatmap"].mean(0))
            plt.axis("off")
            plt.savefig(os.path.join(image_output_dir, k.split(os.path.sep)[-1]))
            if debug:
                plt.show()
            plt.close()

    # Get median number of clicks
    medians = get_medians(final_clickmaps, 'image', thresh=percentile_thresh)
    medians.update(get_medians(final_clickmaps, 'category', thresh=percentile_thresh))
    medians.update(get_medians(final_clickmaps, 'all', thresh=percentile_thresh))
    medians_json = json.dumps(medians, indent=4)

    # Save data
    np.save(os.path.join(output_dir, "co3d_clickmaps_normalized.npy"), img_heatmaps)
    with open("./assets/click_medians.json", 'w') as f:
        f.write(medians_json)

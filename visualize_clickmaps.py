import os, sys
import numpy as np
import pandas as pd
import utils
from skimage import io
from tqdm import tqdm


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
    config_file = os.path.join("configs", "co3d_config.yaml")
    image_dir = "CO3D_ClickMe2/"
    output_dir = "individual_clickme_maps"
    clickmap_saves  = "clickmap_data"
    image_saves = "clickmap_images"
    percentile_thresh = 50
    pos_sample = 20
    center_crop = False

    # Load config
    config = utils.process_config(config_file)

    # Load data
    co3d_clickme_data = utils.process_clickme_data(config["clickme_data"])
    blur_size = config["blur_size"]
    blur_sigma = np.sqrt(blur_size)
    min_pixels = (2 * blur_size) ** 2  # Minimum number of pixels for a map to be included following filtering
    del config["experiment_name"], config["clickme_data"]

    # Start processing
    clickmap_saves = os.path.join(output_dir, clickmap_saves)
    image_saves = os.path.join(output_dir, image_saves)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(clickmap_saves, exist_ok=True)
    os.makedirs(image_saves, exist_ok=True)

    # Process files in serial
    clickmaps, _ = utils.process_clickmap_files(
        clickme_data=co3d_clickme_data,
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"])

    # Prepare maps
    final_clickmaps, all_clickmaps, categories, _ = utils.prepare_maps(
        final_clickmaps=clickmaps,
        blur_size=blur_size,
        blur_sigma=blur_sigma,
        image_shape=config["image_shape"],
        min_pixels=min_pixels,
        min_subjects=config["min_subjects"],
        center_crop=center_crop)
    
    import pdb;pdb.set_trace()
    # Go through maps and let's filter per category 
    # Set positive maps to be the modal maps from each
    # Set negative maps to be outliers.
    # When training a model we will also generate "random walk" versions of the negative maps to equalize +/- classes
    

    # Plot and save all images
    if pos_sample:
        sel_idx = np.random.permutation(len(final_clickmaps))[:pos_sample]
        final_clickmaps = {k: v for idx, (k, v) in enumerate(final_clickmaps.items()) if idx in sel_idx}
        all_clickmaps = [x for idx, x in enumerate(all_clickmaps) if idx in sel_idx]
        print("Using {} clickmaps".format(len(all_clickmaps)))

    for (f, ms), cs in tqdm(zip(final_clickmaps.items(), all_clickmaps), total=len(all_clickmaps), desc="Writing data"):
        f = "{}_{}".format(f.split(os.path.sep)[-2], f.split(os.path.sep)[-1]).split(".")[0]
        for idx, (m, c) in enumerate(zip(ms, cs)):
            c = ((c - c.min()) / (c.max() - c.min()) * 255).astype(np.uint8)
            click_fn = os.path.join(clickmap_saves, "{}_{}.npy".format(f, idx))
            im_fn = os.path.join(image_saves, "{}_{}.png".format(f, idx))
            np.save(click_fn, m)
            io.imsave(im_fn, c)

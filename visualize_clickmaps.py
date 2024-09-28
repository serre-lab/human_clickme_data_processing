import os
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

    # Args
    debug = False
    config_file = os.path.join("configs", "co3d_config.yaml")
    image_dir = "CO3D_ClickMe2/"
    output_dir = "individual_clickme_maps"
    clickmap_saves  = "clickmap_data"
    image_saves = "clickmap_images"
    percentile_thresh = 50
    center_crop = False
    display_image_keys = [
        'mouse/372_41138_81919_renders_00017.png',
        'skateboard/55_3249_9602_renders_00041.png',
        'couch/617_99940_198836_renders_00040.png',
        'microwave/482_69090_134714_renders_00033.png',
        'bottle/601_92782_185443_renders_00030.png',
        'kite/399_51022_100078_renders_00049.png',
        'carrot/405_54110_105495_renders_00039.png',
        'banana/49_2817_7682_renders_00025.png',
        'parkingmeter/429_60366_116962_renders_00032.png'
    ]

    # Load config
    config = utils.process_config(config_file)
    co3d_clickme_data = pd.read_csv(config["clickme_data"])
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
    
    # Plot and save all images
    for (f, ms), cs in tqdm(zip(final_clickmaps.items(), all_clickmaps), total=len(all_clickmaps), desc="Writing data"):
        import pdb;pdb.set_trace()
        a = 2
        f = "{}_{}".format(.split(os.path.sep)[-2], f.split(os.path.sep)[-1])
        for idx, (m, c) in enumerate(zip(ms, cs)):
            click_fn = os.path.join(clickmap_saves, "{}_idx".format(f))
            im_fn = os.path.join(image_saves, "{}_idx".format(f))
            np.save(click_fn, m)
            io.imsave(im_fn, c)

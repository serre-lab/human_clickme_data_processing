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
    debug = False
    percentile_thresh = 50

    # Load config
    config = utils.process_config(config_file)
    clickme_data = utils.process_clickme_data(config["clickme_data"])
    output_dir = config["assets"]
    blur_size = config["blur_size"]
    image_output_dir = config["example_image_output_dir"]
    blur_sigma = np.sqrt(blur_size)
    min_pixels = (2 * blur_size) ** 2  # Minimum number of pixels for a map to be included following filtering

    # Start processing
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Process files in serial
    clickmaps, clickmap_counts = utils.process_clickmap_files(
        clickme_data=clickme_data,
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"])
    
    # Top images according to the number of participants per map
    # srt = np.argsort(clickmap_counts)[-10:]
    # fls = np.asarray([k for k in clickmaps.keys()])[srt]
    # print(fls)

    # Prepare maps
    final_clickmaps, all_clickmaps, categories, final_keep_index = utils.prepare_maps(
        final_clickmaps=clickmaps,
        blur_size=blur_size,
        blur_sigma=blur_sigma,
        image_shape=config["image_shape"],
        min_pixels=min_pixels,
        min_subjects=config["min_subjects"],
        center_crop=False)

    # Visualize if requested
    szs = [len(x) for x in all_clickmaps]
    arg = np.argsort(szs)
    tops = np.asarray(final_keep_index)[arg[-10:]]
    if config["display_image_keys"]:
        # Load images
        # images, image_names = [], []
        img_heatmaps = {}
        fck = np.asarray([k for k in final_clickmaps.keys()])
        tfck = [k.split(os.path.sep)[-1] for k in final_clickmaps.keys()]
        # for image_file in final_clickmaps.keys():
        for image_file in config["display_image_keys"]:
            image_path = os.path.join(config["image_dir"], image_file)
            image = Image.open(image_path)
            image_name = "_".join(image_path.split('/')[-2:])
            # images.append(image)
            # image_names.append(image_file)
            find_key = [idx for idx, k in enumerate(tfck) if k in image_file]
            assert len(find_key) == 1, "Image not found in final clickmaps"
            # find_key = fck[find_key[0]]
            find_key = find_key[0]
            img_heatmaps[image_file] = {
                "image": image,
                "heatmap": all_clickmaps[find_key]
            }
            # image_names.append(image_name)
        # Package into legacy format
        # img_heatmaps = {k: {"image": image, "heatmap": heatmap} for (k, image, heatmap) in zip(final_clickmaps.keys(), images, all_clickmaps)

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
    num_pos = get_num_pos(final_clickmaps)
    half_num_pos = sample_half_pos(final_clickmaps)
    clickmap_stats = {'num_pos': num_pos, 'half_num_pos':half_num_pos, 'medians': medians}
    medians_json = json.dumps(clickmap_stats, indent=4)
    
    # Save data
    final_data = {k: v for k, v in zip(final_keep_index, all_clickmaps)}
    np.save(
        os.path.join(output_dir, config["processed_clickme_file"]),
        final_data)
    with open(os.path.join(output_dir, config["processed_medians"]), 'w') as f:
        f.write(medians_json)

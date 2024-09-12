import re
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
import torch
import json
from utils import gaussian_kernel, gaussian_blur, create_clickmap
from matplotlib import pyplot as plt


BRUSH_SIZE = 11


def get_medians(point_lists, mode='image', thresh=50):
    assert mode in ['image', 'category', 'all']
    medians = {}
    if mode == 'image':
        for image in point_lists:
            clickmaps = point_lists[image]
            num_clicks = []
            for clickmap in clickmaps:
                num_clicks.append(len(clickmap))
            medians[image] = np.percentile(num_clicks, thresh)
    if mode == 'category':
        for image in point_lists:
            category = image.split('/')[0]
            if category not in medians.keys():
                medians[category] = []
            clickmaps = point_lists[image]
            for clickmap in clickmaps:
                medians[category].append(len(clickmap))
        for category in medians:
            medians[category] = np.percentile(medians[category], thresh)
    if mode == 'all':
        num_clicks = []
        for image in point_lists:
            clickmaps = point_lists[image]
            for clickmap in clickmaps:
                num_clicks.append(len(clickmap))
        medians['all'] = np.percentile(num_clicks, thresh)
    return medians

def make_heatmap(image_path, point_lists, gaussian_kernel, image_shape):
    image = Image.open(image_path)
    heatmap = create_clickmap(point_lists, image_shape)
    
    # Blur the mask to create a smooth heatmap
    heatmap = torch.from_numpy(heatmap).float().unsqueeze(0)  # Convert to PyTorch tensor
    heatmap = gaussian_blur(heatmap, gaussian_kernel)
    heatmap = heatmap.squeeze()
    heatmap = heatmap.numpy()  # Convert back to NumPy array         
  
    # Normalize the heatmap
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # heatmap_normalized = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image_name = "_".join(image_path.split('/')[-2:])
    return image_name, image, heatmap_normalized


def process_clickmaps(clickmap_csv):
    clickmaps = {}
    num_maps = []
    processed_maps = {}
    n_empty = 0
    for index, row in clickmap_csv.iterrows():
        image_file_name = row['image_path'].replace("CO3D_ClickMe2/", "")
        if image_file_name not in clickmaps.keys():
            clickmaps[image_file_name] = [row["clicks"]]
        else:
            clickmaps[image_file_name].append(row["clicks"])
    
    for image, maps in clickmaps.items():
        n_maps = 0
        for clickmap in maps:
            if len(clickmap) == 2:
                n_empty += 1
                continue
            n_maps += 1
            clean_string = re.sub(r'[{}"]', '', clickmap)
            # Split the string by commas to separate the tuple strings
            tuple_strings = clean_string.split(', ')
            # Zero indexing here because tuple_strings is a list with a single string
            data_list = tuple_strings[0].strip("()").split("),(")
            tuples_list = [tuple(map(int, pair.split(','))) for pair in data_list]

            if image not in processed_maps.keys():
                processed_maps[image] = []
            
            processed_maps[image].append(tuples_list)
        num_maps.append(n_maps)
    return processed_maps, num_maps


if __name__ == "__main__":
    image_path = "CO3D_ClickMe2/"
    output_dir = "assets"
    image_output_dir = "clickme_test_images"
    img_heatmaps = {}
    co3d_clickme = pd.read_csv("clickme_vCO3D.csv")
    image_shape = [224, 224]
    thresh = 50
    plot_images = False

    # Start processing
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    processed_maps, num_maps = process_clickmaps(co3d_clickme)
    gaussian_kernel = gaussian_kernel(size=BRUSH_SIZE, sigma=BRUSH_SIZE)
    for idx, (image, maps) in enumerate(processed_maps.items()):
        image_name, image, heatmap = make_heatmap(os.path.join(image_path, image), maps, gaussian_kernel, image_shape=image_shape)
        if image_name is None:
            continue
        img_heatmaps[image_name] = {"image":image, "heatmap":heatmap}
    keys = [
        "chair_378_44060_87918_renders_00018.png",
        "hairdryer_506_72958_141814_renders_00044.png",
        "parkingmeter_429_60366_116962_renders_00032.png",
        "cellphone_444_63640_125603_renders_00006.png",
        "backpack_374_42277_84521_renders_00046.png",
        "remote_350_36752_68568_renders_00005.png",
        "toybus_523_75464_147305_renders_00033.png",
        "bicycle_270_28792_57242_renders_00045.png",
        "laptop_606_95066_191413_renders_00006.png",
        "skateboard_579_85705_169395_renders_00039.png",
    ]
    for k in keys:
        f = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.asarray(img_heatmaps[k]["image"])[:image_shape[0], :image_shape[1]])
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(img_heatmaps[k]["heatmap"])
        plt.axis("off")
        plt.savefig(os.path.join(image_output_dir, k))
        if plot_images:
            plt.show()
        plt.close()

    np.save(os.path.join(output_dir, "co3d_clickmaps_normalized.npy"), img_heatmaps)
    medians = get_medians(processed_maps, 'image', thresh=thresh)
    medians.update(get_medians(processed_maps, 'category', thresh=thresh))
    medians.update(get_medians(processed_maps, 'all', thresh=thresh))
    medians_json = json.dumps(medians, indent=4)
    with open("./assets/click_medians.json", 'w') as f:
        f.write(medians_json)

import os 
import numpy as np
from src.utils import process_clickme_data
from tqdm import tqdm
import h5py
import json
from matplotlib import pyplot as plt
from PIL import Image
from src.utils import process_clickme_data

def get_num_subjects():
    clickme_data_file = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/human_clickme_data_processing/clickme_datasets/val_combined_08_27_2025.npz"
    clickme_data = process_clickme_data(clickme_data_file, True)
    total_numbers = {}
    for _, row in tqdm(clickme_data.iterrows(), total=len(clickme_data), desc="Processing clickmaps"):
        image_path = row['image_path']
        image_file_name = row['image_path'].split(os.path.sep)[-1]
        if "ILSVRC2012_val" not in image_path:
            continue
        if image_file_name not in total_numbers:
            total_numbers[image_file_name] = 1
        else:
            total_numbers[image_file_name] += 1
    return total_numbers

def plot_clickmap(img, hmp, score, num_subjects, img_name, image_output_dir):
    f = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.asarray(img))
    title = f"{img_name}\nSpearman: {score}\nNum Subjects: {num_subjects}"
    plt.title(title)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(hmp)
    plt.axis("off")
    plt.savefig(os.path.join(image_output_dir, img_name.replace('/', '_')))
    plt.close()
    return 

if __name__ == "__main__":
    scores_json = "assets/exp_30_subjects_08_27_2025_spearman_ceiling_floor_results.json"
    data_root = "/gpfs/data/shared/imagenet/ILSVRC2012/val"
    image_output_dir = "temp/top_bot_imgs_30"
    os.makedirs(image_output_dir, exist_ok=True)
    with open(scores_json, 'r') as f:
        scores_dict = json.load(f)['all_img_ceilings']
    metadata = np.load("image_metadata/jay_imagenet_val_04_30_2025_dimensions.npy", allow_pickle=True).item()
    val_map_files = ['assets/imgnet_val/jay_imagenet_val_08_27_2025_batch001.h5', 'assets/imgnet_val/jay_imagenet_val_08_27_2025_batch002.h5',
                    'assets/imgnet_val/jay_imagenet_val_08_27_2025_batch003.h5', 'assets/imgnet_val/jay_imagenet_val_08_27_2025_batch004.h5']
    top10 = dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:10])
    bot10 = dict(sorted(scores_dict.items(), key=lambda x: x[1], reverse=False)[:10])
    top10_maps = {}
    bot10_maps = {}
    for map_file in val_map_files:
        map_content = h5py.File(map_file, 'r')['clickmaps']
        for img_name in top10:
            img_name = img_name.replace('/', '_')
            if img_name in map_content:
                top10_maps[img_name] = map_content[img_name]['clickmap'][:].mean(0)
        for bot_img_name in bot10:
            bot_img_name = bot_img_name.replace('/', '_')
            if bot_img_name in map_content:
                bot10_maps[img_name] = map_content[bot_img_name]['clickmap'][:].mean(0)

    top10_paths = []
    bot10_paths = []
    for img_name in top10:
        img_name = img_name.split('/')[1]
        img_path = os.path.join(data_root, f'{img_name}')
        top10_paths.append(img_path)

    for img_name in bot10:
        img_name = img_name.split('/')[1]
        img_path = os.path.join(data_root, f'{img_name}')
        bot10_paths.append(img_path)       

    for i, img_name in enumerate(top10):
        score = scores_dict[img_name]
        metadata_shape = metadata[img_name]

        img_name = img_name.split('/')[1]
        hmp = top10_maps[img_name]
        img = Image.open(top10_paths[i])
        # num_subjects = num_subjects_dict[img_name]
        num_subjects = 1
        img = np.asarray(img)
        if img.shape[0] != hmp.shape[0]:
            print(img_name, img.shape, hmp.shape, metadata_shape)
        plot_clickmap(img, hmp, score, num_subjects, f"top_{img_name}", image_output_dir)


    for i, img_name in enumerate(bot10):
        score = scores_dict[img_name]
        metadata_shape = metadata[img_name]

        img_name = img_name.split('/')[1]
        hmp = bot10_maps[img_name]
        img = Image.open(bot10_paths[i])
        num_subjects = 1
        # num_subjects = num_subjects_dict[img_name]
        img = np.asarray(img)
        if img.shape[0] != hmp.shape[0]:
            print(img_name, img.shape, hmp.shape, metadata_shape)
        plot_clickmap(img, hmp, score, num_subjects, f"bottom_{img_name}", image_output_dir)
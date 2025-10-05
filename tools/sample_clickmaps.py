import os 
import numpy as np
from src.utils import process_clickme_data
from tqdm import tqdm
# Sample clickmaps that have more than 30 subjects while maintaining class distribution
if __name__ == "__main__":
    clickme_data_file = "/cifs/data/tserre_lrs/projects/projects/prj_video_imagenet/human_clickme_data_processing/clickme_datasets/val_combined_08_27_2025.npz"
    clickme_data = process_clickme_data(clickme_data_file, True)
    total_maps = len(clickme_data)
    total_numbers = {}
    target_img_names = ["ILSVRC2012_val_00008676.JPEG", "ILSVRC2012_val_00009263.JPEG", "ILSVRC2012_val_00009305.JPEG", "ILSVRC2012_val_00013865.JPEG"
                        "ILSVRC2012_val_00021166.JPEG", "ILSVRC2012_val_00023616.JPEG", "ILSVRC2012_val_00024753.JPEG", "ILSVRC2012_val_00027647.JPEG",
                        "ILSVRC2012_val_00034111.JPEG", "ILSVRC2012_val_00038455.JPEG"]
    for _, row in tqdm(clickme_data.iterrows(), total=len(clickme_data), desc="Processing clickmaps"):
        image_path = row['image_path']
        image_file_name = os.path.sep.join(row['image_path'].split(os.path.sep)[-2:])
        cls_name = image_file_name.split('/')[0]
        if cls_name not in total_numbers:
            total_numbers[cls_name] = {}
        if "ILSVRC2012_val" not in image_file_name:
            continue
        if image_path not in total_numbers[cls_name]:
            total_numbers[cls_name][image_path] = 1
        else:
            total_numbers[cls_name][image_path] += 1
    sampled_img_paths = {}
    total_counts = []
    for cls_name, image_paths in total_numbers.items():
        numbers = []
        sampled_img_paths[cls_name] = []
        sampled_names = []
        for img_path, number in image_paths.items():
            numbers.append(number)
            img_name = img_path.split('/')[-1]
            if img_name in target_img_names:
            # if number > 20:
                sampled_names.append(img_path)
                sampled_img_paths[cls_name].append(img_path)
        numbers = np.array(numbers)
        larger_than = np.sum(numbers>20)
        if larger_than > 0:
            total_counts.append(larger_than)
       
    for cls_name, img_paths in sampled_img_paths.items():
        sampled_img_paths[cls_name] = img_paths[:5]
    sampled_clickme_data = clickme_data.copy()

    allowed_files = {
        f"{img_path}" for _, img_paths in sampled_img_paths.items() for img_path in img_paths
    }
    print(allowed_files)
    print(sampled_clickme_data)
    print(len(sampled_clickme_data))
    # Keep only rows whose file name is allowed
    sampled_clickme_data = sampled_clickme_data[
        sampled_clickme_data["image_path"].isin(allowed_files)
    ]
    print(len(sampled_clickme_data))
    sampled_clickme_data.to_csv(os.path.join('clickme_datasets', 'bottom_imgnet_val.csv'))

    
        
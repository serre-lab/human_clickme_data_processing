import os
import sys
import numpy as np


if __name__ == "__main__":
    f_name = sys.argv[1]
    human_file = np.load(f_name, allow_pickle=True)
    filtered_imgs = human_file['final_clickmaps'].tolist().keys()
    cats = []
    for img in filtered_imgs:
        cats.append(img.split('/')[0])
    print(len(np.unique(cats)))

    clickme_merged = np.load('clickme_datasets/merged_val_imagenet_11_13_2024.npz', allow_pickle=True)
    img_names = clickme_merged["file_pointer"]
    print(len(img_names))
    cats = []
    for img in img_names:
        if 'imagenet' not in img:
            continue
        cats.append(img.split('/')[2])
    print(len(np.unique(cats)))

    clickme_merged = np.load('clickme_datasets/prj_clickmev2_val_imagenet_10_10_2024.npz', allow_pickle=True)
    img_names = clickme_merged["file_pointer"]
    print(len(img_names))
    cats = []
    for img in img_names:
        if 'imagenet' not in img:
            continue
        cats.append(img.split('/')[2])
    print(len(np.unique(cats)))
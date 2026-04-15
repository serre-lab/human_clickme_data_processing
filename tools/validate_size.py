import os
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    hdf_path = "assets/imgnet"
    metadata = np.load("image_metadata/jay_imagenet_train_04_30_2025_dimensions.npy", allow_pickle=True).item()
    data_root = "/gpfs/data/shared/imagenet/ILSVRC2012/train"
    for i, hdf_file in tqdm(enumerate(os.listdir(hdf_path))):
        if not hdf_file.endswith('.h5'):
            continue
        map_content = h5py.File(os.path.join(hdf_path, hdf_file), 'r')['clickmaps']
        for img_name in map_content:
            metadata_img_name = img_name.replace('_', '/', 1)
            img_cls = metadata_img_name.split('/')[0]
            folder_img_name = metadata_img_name.split('/')[1]
            if not os.path.exists(os.path.join(data_root, img_cls, folder_img_name)) or metadata_img_name not in metadata:
                continue
            img = Image.open(os.path.join(data_root, img_cls, folder_img_name))
            img = np.asarray(img)
            metadata_shape = metadata[metadata_img_name][::-1]
            hmp_shape = map_content[img_name]['clickmap'][:].mean(0).shape
            if metadata_shape != hmp_shape or img.shape[:2] != hmp_shape:
                print(i, img_name, img.shape[:2], metadata_shape, hmp_shape)
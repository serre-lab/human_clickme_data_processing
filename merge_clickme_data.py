from src.utils import merge_clickme_data
import numpy as np

if __name__ == "__main__":
    file_1 = 'clickme_datasets/train_imagenet_11_13_2024.npz'
    file_2 = 'clickme_datasets/prj_clickmev2_train_imagenet_10_10_2024.npz'

    merged = merge_clickme_data(file_1, file_2)
    np.savez('clickme_datasets/merged_train_imagenet_11_13_2024.npz', **merged)
    
    file_1 = 'clickme_datasets/val_imagenet_11_13_2024.npz'
    file_2 = 'clickme_datasets/prj_clickmev2_val_imagenet_10_10_2024.npz'

    merged = merge_clickme_data(file_1, file_2)
    np.savez('clickme_datasets/merged_val_imagenet_11_13_2024.npz', **merged)
import numpy as np


click_data = np.load("clickme_datasets/val_imagenet_09_29_2024.npz", allow_pickle=True)
device_data = np.load("clickme_datasets/prj_clickmev2_val_imagenet_no_clicks_10_06_2024.npz", allow_pickle=True)

click_users = click_data["user_id"]
device_users = device_data["user_id"]
is_mobile = device_data["is_mobile"]

unique_device_users = np.unique(device_users)

mobile_array = np.zeros_like(click_users)
for device_user in unique_device_users:

    cidx = click_users == device_user
    didx = device_users == device_user
    device = np.unique(is_mobile[didx])
    if len(device) > 1:
        print(device_user, device)
    mobile_array[cidx] = device

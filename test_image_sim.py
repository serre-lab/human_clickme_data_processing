import numpy as np
from skimage import io
from matplotlib import pyplot as plt


data = np.load("image_similarity_dict.npy", allow_pickle=True).item()

# new_IN = "/media/data_cifs/projects/prj_video_imagenet/imagenet/ILSVRC/Data/CLS-LOC/val2/n07745940/ILSVRC2012_val_00024155.JPEG"
# old_IN = data[new_IN]

# new_IN_image = io.imread(new_IN)
# old_IN_image = np.load(old_IN)

for k, v in data.items():
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(io.imread(k))
    f.add_subplot(1, 2, 2)
    plt.imshow(np.load(v))
    plt.show()
    plt.close(f)









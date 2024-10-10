import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed


# Load images and read dimensions using Joblib parallelization
def get_image_size(prep_file):
    img_path = os.path.join(image_path, prep_file)
    try:
        with Image.open(img_path) as img:
            return prep_file, img.size
    except Exception as e:
        print(f"Error processing {prep_file}: {e}")
        return prep_file, None


experiment_name = "imagenet_validation_clickme"
click_dump = "clickme_datasets/prj_clickmev2_val_imagenet_10_10_2024.npz"
filter = "imagenet/val"
remove_string = "imagenet/val/"  # in case you want to trim paths
image_path = "/media/data_cifs/projects/prj_video_imagenet/imagenet/ILSVRC/Data/CLS-LOC/val2"
parallel = True  # set to True to enable parallel processing

# Get files
data = np.load(click_dump, allow_pickle=True)
files = [x for x in data["file_pointer"] if filter in x]
if remove_string:
    prep_files = [x.replace(remove_string, "") for x in files]
else:
    prep_files = files
prep_files = np.asarray(prep_files)

if parallel:
    unique_prep_files = np.unique(prep_files)
    results = Parallel(n_jobs=-1)(
        delayed(get_image_size)(pf) for pf in tqdm(unique_prep_files, desc="Reading dimensions")
    )
    dimensions = {pf: size for pf, size in results if size is not None}
else:
    dimensions = {}
    for prep_file in tqdm(np.unique(prep_files), desc="Reading dimensions"):
        img_path = os.path.join(image_path, prep_file)
        try:
            with Image.open(img_path) as img:
                dimensions[prep_file] = img.size
        except Exception as e:
            print(f"Error processing {prep_file}: {e}")

# Save to file
np.save(f"{experiment_name}_dimensions.npy", dimensions)
import os
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed


# Create output directories for each tfrecord file
paths = [
    # "/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_test.tfrecords",
    "/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_train.tfrecords",
    "/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_val.tfrecords",
]

fdict = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
    'heatmap': tf.io.FixedLenFeature([], tf.string), 
    'click_count': tf.io.FixedLenFeature([], tf.int64),
}

def process_record(record, fdict, image_output_dir, hm_output_dir, idx):
    """Process a single record from the dataset."""
    # Parse the record
    features = tf.io.parse_single_example(record, features=fdict)
    
    # Get feature dictionary
    click_count = features["click_count"].numpy()
    label = features["label"].numpy()
    image = tf.io.decode_raw(features["image"], tf.float32)
    image = tf.reshape(image, [256, 256, 3]).numpy().astype(np.uint8)
    heatmap = tf.io.decode_raw(features["heatmap"], tf.float32)
    heatmap = tf.reshape(heatmap, [256, 256, 1]).numpy()

    # Generate unique image name using index
    image_name = f"{label}_{idx}.png"

    # Create output file paths
    image_output_file = os.path.join(image_output_dir, image_name)
    hm_output_file = os.path.join(hm_output_dir, image_name)

    # Save image and HM
    np.save(image_output_file, image)
    np.save(hm_output_file, heatmap)

    return {
        'click_count': click_count,
        'label': label,
        'image_path': features.get("image_path", None),
        'user_id': features.get("user_id", None)
    }

# Process each tfrecord file
for path in paths:
    # Create output directory
    image_output_dir = "{}_images_v1".format(path.split(os.path.sep)[-1].split(".")[0])
    hm_output_dir = "{}_heatmaps_v1".format(path.split(os.path.sep)[-1].split(".")[0])
    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(hm_output_dir, exist_ok=True)
    
    # Read the tfrecord file using a context manager
    with tf.data.TFRecordDataset(path).files() as files:
        dataset = tf.data.TFRecordDataset(files)
        records = list(dataset)  # Convert to list for parallel processing

    # Process records in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_record)(
            record, 
            fdict, 
            image_output_dir, 
            hm_output_dir, 
            idx
        ) for idx, record in enumerate(tqdm(records, desc=f"Processing {path.split(os.path.sep)[-1]}"))
    )

    # Collect results
    clicks = [r['click_count'] for r in results]
    labels = [r['label'] for r in results]
    image_paths = [r['image_path'] for r in results if r['image_path'] is not None]
    user_ids = [r['user_id'] for r in results if r['user_id'] is not None]

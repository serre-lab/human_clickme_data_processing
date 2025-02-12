import os
import tensorflow as tf
from tqdm import tqdm
import cv2


# Create output directories for each tfrecord file
paths = [
    "/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_test.tfrecords",
    "/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_train.tfrecords",
    "/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_val.tfrecords",
]

fdict = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
    'heatmap': tf.io.FixedLenFeature([], tf.string), 
    'click_count': tf.io.FixedLenFeature([], tf.int64),
}

# Process each tfrecord file
for path in paths:
    # Create output directory
    output_dir = "{}_v1".format(path.split(os.path.sep)[-1].split(".")[0])
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the tfrecord file
    dataset = tf.data.TFRecordDataset(path)


    # Store info in lists
    clicks, labels, image_paths, user_ids = [], [], [], []
    image_counts = {}

    # Iterate through records
    for record in tqdm(dataset, desc="Processing record: {}".format(path.split(os.path.sep)[-1].split(".")[0])):
        # Parse the record
        import pdb;pdb.set_trace()
        features = tf.io.parse_single_example(record, features=fdict)
        
        # Get feature dictionary
        click_count = features["click_count"].numpy()
        label = features["label"].numpy()
        image = tf.io.decode_raw(features["image"], tf.float32)
        image = tf.reshape(image, [256, 256, 3]).numpy()
        heatmap = tf.io.decode_raw(features["heatmap"], tf.float32)
        heatmap = tf.reshape(heatmap, [256, 256, 1]).numpy()
        import pdb;pdb.set_trace()

        if label not in image_counts:
            image_counts[label] = 0
        image_counts[label] += 1
        image_name = "{}_{}.png".format(label, image_counts[label])

        # Store data
        clicks.append(heatmap)
        labels.append(label)
        image_paths.append(features["image_path"].bytes_list.value[0])
        user_ids.append(features["user_id"].bytes_list.value[0])

        # Create output file name and save
        output_file = os.path.join(output_dir, image_name)
        
        # Save image
        cv2.imwrite(output_file, image)

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
# Process each tfrecord file
for path in paths:
    # Create output directory
    output_dir = "{}_v1".format(path.split(os.path.sep)[-1].split(".")[0])
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the tfrecord file
    dataset = tf.data.TFRecordDataset(path)
    
    # Iterate through records
    for record in tqdm(dataset, total=len(dataset), desc="Processing record: {}".format(path.split(os.path.sep)[-1].split(".")[0])):
        # Parse the record
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        
        # Get feature dictionary
        features = example.features.feature
        label = features["label"].int64_list.value[0]
        click_count = features["click_count"].int64_list.value[0]
        image = tf.io.decode_raw(features["image"].bytes_list.value[0], tf.float32)
        image = tf.reshape(image, [256, 256, 3]).numpy()

        # Create output file name
        output_file = os.path.join(output_dir, "{}_{}.png".format(label, click_count))
        
        # Save image
        import pdb;pdb.set_trace()
        cv2.imwrite(output_file, image)


import os
import tensorflow as tf
import pdb


# Create output directories for each tfrecord file
paths = [
    "/media/data_cifs/clickme_experiment/tf_records/archive/clickme_test.tfrecords",
    "/media/data_cifs/clickme_experiment/tf_records/archive/clickme_train.tfrecords",
    "/media/data_cifs/clickme_experiment/tf_records/archive/clickme_val.tfrecords",
]

# Process each tfrecord file
for path in paths:
    # Create output directory
    output_dir = "{}_v1".format(path.split(os.path.sep)[-1].split(".")[0])
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the tfrecord file
    dataset = tf.data.TFRecordDataset(path)
    
    # Iterate through records
    for record in dataset:
        # Parse the record
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        
        # Get feature dictionary
        features = example.features.feature
        
        # Add breakpoint to inspect contents
        pdb.set_trace()
        a = 2


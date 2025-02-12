import os
import tensorflow as tf
from tqdm import tqdm
import cv2


# Create output directories for each tfrecord file
paths = [
    # "/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_test.tfrecords",
    "/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_train.tfrecords",
    # "/media/data_cifs/clicktionary/clickme_experiment/tf_records/archive/clickme_val.tfrecords",
]

fdict = {
    'label': tf.FixedLenFeature([], tf.int64),
    'image': tf.FixedLenFeature([], tf.string),
    'heatmap': tf.FixedLenFeature([], tf.string),
    'click_count': tf.FixedLenFeature([], tf.int64),
    'file_path': tf.FixedLenFeature([], tf.string),
}

# Process each tfrecord file
for path in paths:
    # Create output directory
    output_dir = "{}_v1".format(path.split(os.path.sep)[-1].split(".")[0])
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the tfrecord file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(path)
    # features = tf.parse_single_example(serialized_example, features=fdict)

    # dataset = tf.data.TFRecordDataset(path)

    # # Convert from a scalar string tensor (whose single string has
    # image = tf.decode_raw(features[keys['image']], image_type)
    # if image.dtype != tf.float32:
    #     image = tf.cast(image, tf.float32)

    # # Need to reconstruct channels first then transpose channels
    # image = tf.reshape(image, im_size)  # np.asarray(im_size)[[2, 0, 1]])

    # # image = tf.transpose(res_image, [2, 1, 0])
    # image.set_shape(im_size)

    # if return_heatmaps:
    #     # Normalize the heatmap and prep for image augmentations
    #     heatmap = tf.decode_raw(features['heatmap'], tf.float32)
    #     heatmap = tf.reshape(heatmap, [im_size[0], im_size[1], 1])
    #     heatmap /= tf.reduce_max(heatmap)
    #     heatmap = repeat_elements(heatmap, 3, axis=2)
    #     image, heatmap = image_augmentations(
    #         image, heatmap, im_size, data_augmentations,
    #         model_input_shape, return_heatmaps)
    #     heatmap = tf.where(tf.is_nan(heatmap), tf.zeros_like(heatmap), heatmap)

    # Store info in lists
    clicks, labels, image_paths, user_ids = [], [], [], []

    # Iterate through records
    for record in tqdm(dataset, desc="Processing record: {}".format(path.split(os.path.sep)[-1].split(".")[0])):
        # Parse the record
        example = tf.train.Example()
        example.ParseFromString(record.numpy(), features=fdict)
        
        # Get feature dictionary
        features = example.features.feature
        click_data = features["clicks"]
        import pdb;pdb.set_trace()

        label = features["label"].int64_list.value[0]
        import pdb;pdb.set_trace()
        click_count = features["click_count"].int64_list.value[0]
        image = tf.io.decode_raw(features["image"].bytes_list.value[0], tf.float32)
        image = tf.reshape(image, [256, 256, 3]).numpy()

        # Store data
        clicks.append(click_data)
        labels.append(label)
        image_paths.append(features["image_path"].bytes_list.value[0])
        user_ids.append(features["user_id"].bytes_list.value[0])

        # Create output file name
        output_file = os.path.join(output_dir, "{}_{}.png".format(label, click_count))
        
        # Save image
        cv2.imwrite(output_file, image)


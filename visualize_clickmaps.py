import os, sys
import numpy as np
from src import utils
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt


if __name__ == "__main__":

    # Get config file
    config_file = utils.get_config(sys.argv)
    keep_images = 50

    # Other Args
    blur_sigma_function = lambda x: x

    # Load config
    config = utils.process_config(config_file)
    clickme_data = utils.process_clickme_data(
        config["clickme_data"],
        config["filter_mobile"])
    output_dir = config["assets"]
    image_output_dir = config["example_image_output_dir"]
    blur_size = config["blur_size"]
    blur_sigma = blur_sigma_function(blur_size)
    min_pixels = (2 * blur_size) ** 2  # Minimum number of pixels for a map to be included following filtering

    # Load metadata
    if config["metadata_file"]:
        metadata = np.load(config["metadata_file"], allow_pickle=True).item()
    else:
        metadata = None

    # Find files
    processed_clicks = np.load(os.path.join(output_dir, config["processed_clickme_file"]), allow_pickle=True)

    # Count clicks per image
    click_counts = {
        k: v.item()["heatmap"].shape[0]
        for k, v in tqdm(
            processed_clicks.items(), 
            total=len(processed_clicks), 
            desc="Counting clicks per image"
        )
    }
    click_vals = np.sort(list(click_counts.values()))[::-1]
    keep_count = 0
    for c in click_vals:
        keep_count += c
        if keep_count >= keep_images:
            break
    filtered_clicks = {k: v for k, v in processed_clicks.items() if click_counts[k] >= click_vals[keep_count]}

    # Average clicks per image and save in a folder
    output_dir = "{}_images".format(config["processed_clickme_file"])
    os.makedirs(output_dir, exist_ok=True)
    for k, v in tqdm(filtered_clicks.items(), total=len(filtered_clicks), desc="Saving images"):
        v = v.item()
        image = np.asarray(v["image"])
        heatmap = v["heatmap"]
        np.save(os.path.join(output_dir, "{}_clickmap.npy".format(k)), heatmap)
        np.save(os.path.join(output_dir, "{}.npy".format(k)), image)

        f = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(heatmap.mean(0))
        plt.axis("off")
        plt.savefig(os.path.join(output_dir, "{}.png".format(k)))
        if config["debug"]:
            plt.show()
        plt.close()

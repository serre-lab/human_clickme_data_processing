import os, sys
import numpy as np
from src import utils
from skimage import io
from tqdm import tqdm


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
    import pdb;pdb.set_trace()

    # Average clicks per image
    



    # Process files in serial
    clickmaps, clickmap_counts = utils.process_clickmap_files(
        clickme_data=clickme_data,
        image_path=config["image_path"],
        file_inclusion_filter=config["file_inclusion_filter"],
        file_exclusion_filter=config["file_exclusion_filter"],
        min_clicks=config["min_clicks"],
        max_clicks=config["max_clicks"])

    # Prepare maps
    if config["debug"]:
        new_clickmaps = {}
        for k in config["display_image_keys"]:
            click_match = [k_ for k_ in clickmaps.keys() if k in k_]
            assert len(click_match) == 1, "Clickmap not found"
            new_clickmaps[click_match[0]] = clickmaps[click_match[0]]
        clickmaps = new_clickmaps
        # clickmaps = {k: v for idx, (k, v) in enumerate(clickmaps.items()) if idx < 1000}

    # Filter classes if requested
    if config["class_filter_file"]:
        clickmaps = utils.filter_classes(
            clickmaps=clickmaps,
            class_filter_file=config["class_filter_file"])

    # Filter participants if requested
    if config["participant_filter"]:
        clickmaps = utils.filter_participants(clickmaps)
    print(len(clickmaps))
    # Prepare maps
    final_clickmaps, all_clickmaps, categories, final_keep_index = utils.prepare_maps(
        final_clickmaps=clickmaps,
        blur_size=blur_size,
        blur_sigma=blur_sigma,
        image_shape=config["image_shape"],
        min_pixels=min_pixels,
        min_subjects=config["min_subjects"],
        metadata=metadata,
        blur_sigma_function=blur_sigma_function,
        center_crop=False)
    print(len(final_clickmaps), len(all_clickmaps))
    # Filter for foreground mask overlap if requested
    if config["mask_dir"]:
        masks = utils.load_masks(config["mask_dir"])
        final_clickmaps, all_clickmaps, categories, final_keep_index = utils.filter_for_foreground_masks(
            final_clickmaps=final_clickmaps,
            all_clickmaps=all_clickmaps,
            categories=categories,
            masks=masks,
            mask_threshold=config["mask_threshold"])
    # Visualize if requested
    sz_dict = {k: len(v) for k, v in final_clickmaps.items()}
    arg = np.argsort(list(sz_dict.values()))
    tops = np.asarray(list(sz_dict.keys()))[arg[-10:]]
    if config["display_image_keys"]:
        if config["display_image_keys"] == "auto":
            config["display_image_keys"] = tops
        # Load images
        img_heatmaps = {}
        fck = np.asarray([k for k in final_clickmaps.keys()])
        for image_file in config["display_image_keys"]:
            image_path = os.path.join(config["image_path"], image_file)
            image = Image.open(image_path)
            if metadata:
                click_match = [k_ for k_ in final_clickmaps.keys() if image_file in k_]
                assert len(click_match) == 1, "Clickmap not found"
                metadata_size = metadata[click_match[0]]
                image = image.resize(metadata_size)
            image_name = "_".join(image_path.split('/')[-2:])
            check = fck == image_file
            if check.any():
                find_key = np.where(check)[0][0]
                img_heatmaps[image_file] = {
                    "image": image,
                    "heatmap": all_clickmaps[find_key]
                }
            else:
                print("Image {} not found in final clickmaps".format(image_file))

        # And plot
        for k in img_heatmaps.keys():
            f = plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(np.asarray(img_heatmaps[k]["image"]))
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(img_heatmaps[k]["heatmap"].mean(0))
            plt.axis("off")
            plt.savefig(os.path.join(image_output_dir, k.split(os.path.sep)[-1]))
            if config["debug"]:
                plt.show()
            plt.close()


    # Save data
    img_heatmaps = {}
    for i, img_name in enumerate(final_keep_index):
        if not os.path.exists(os.path.join(config["image_path"], img_name)):
            print(os.path.join(config["image_path"], img_name))
            continue
        if img_name not in final_clickmaps.keys():
            continue
        hmp = all_clickmaps[i]
        img = Image.open(os.path.join(config["image_path"], img_name))
        if metadata:
            click_match = [k_ for k_ in final_clickmaps.keys() if img_name in k_]
            assert len(click_match) == 1, "Clickmap not found"
            metadata_size = metadata[click_match[0]]
            img = img.resize(metadata_size)

        img_heatmaps[img_name] = {"image":img, "heatmap":hmp}
    print(len(img_heatmaps))

    # Patch: Sometimes img_heatmaps is too large
    import pdb;pdb.set_trace()
    if len(img_heatmaps) > 10000:
        os.makedirs(os.path.join(output_dir, config["experiment_name"]), exist_ok=True)
        for hn, hm in img_heatmaps.keys():
            np.save(os.path.join(output_dir, config["experiment_name"], "{}.npy".format(hn)), 
                    hm
                )
    else:
        np.savez(os.path.join(output_dir,  config["processed_clickme_file"]), 
            **img_heatmaps
        )

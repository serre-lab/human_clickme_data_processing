import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
import torch
from torch.nn.functional import conv2d


def compute_spearman_correlation(map1, map2):
    """
    Compute the Spearman correlation between two maps.
    """
    filtered_map1 = map1.flatten()
    filtered_map2 = map2.flatten()

    if filtered_map1.size > 1 and filtered_map2.size > 1:
        correlation, _ = spearmanr(filtered_map1, filtered_map2)
        return correlation
    else:
        return float('nan')


def gaussian_blur(heatmap, kernel):
    """
    Blurs a heatmap with a Gaussian kernel.
    """
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    heatmap = heatmap.float()
    kernel = kernel.float()
    blurred_heatmap = conv2d(heatmap, kernel, padding='same')
    return blurred_heatmap[0]


def circle_kernel(size, sigma=None, device='cpu'):
    """
    Create a flat circular kernel normalized to sum to 1.
    
    Args:
        size (int): The diameter of the circle and the size of the kernel.
        sigma (float, optional): Not used for flat kernel. Included for compatibility.
        device (str, optional): Device to place the kernel on. Default is 'cpu'.
    """
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    center = (size - 1) / 2
    radius = (size - 1) / 2
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
    kernel = torch.zeros((size, size), dtype=torch.float32)
    kernel[mask] = 1.0
    kernel /= kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0).to(device)


def generate_heatmap(x_coords, y_coords, image_shape):
    """
    Generate a 2D heatmap from x and y coordinates.
    """
    heatmap = np.zeros(image_shape)
    for x, y in zip(x_coords, y_coords):
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            heatmap[int(y), int(x)] += 1
    return heatmap / heatmap.sum() if heatmap.sum() > 0 else heatmap


def filter_clickmaps(x_coords, y_coords):
    """
    Apply filtering rules to clean clickmaps.

    Steps:
    1. Remove negative values from both x and y.
    2. Remove empty maps (after filtering negative values).
    3. Remove maps that are constant (flat).

    Returns:
        Tuple of filtered x and y coordinates.
    """
    # Remove negative values
    valid_indices = [i for i, (x, y) in enumerate(zip(x_coords, y_coords)) if x >= 0 and y >= 0]
    x_coords = [x_coords[i] for i in valid_indices]
    y_coords = [y_coords[i] for i in valid_indices]

    # Check for empty maps
    if not x_coords or not y_coords:
        return None, None

    # Check for constant maps
    if len(set(x_coords)) == 1 and len(set(y_coords)) == 1:
        return None, None

    return x_coords, y_coords


def eval_alignment(category_data, image_shape, kernel_size=21):
    """
    Evaluate alignment scores for each category using Spearman correlation.
    """
    kernel = circle_kernel(kernel_size, device='cpu')
    category_alignments = []

    for category, data in tqdm(category_data.items(), desc="Evaluating alignments"):
        alignments = []
        for i in range(len(data)):
            x_coords, y_coords = data[i]['x'], data[i]['y']

            # Filter the clickmaps
            x_coords, y_coords = filter_clickmaps(x_coords, y_coords)
            if x_coords is None or y_coords is None:
                continue  # Skip invalid maps

            test_map = generate_heatmap(x_coords, y_coords, image_shape)

            remaining_maps = [
                generate_heatmap(filter_clickmaps(d['x'], d['y'])[0], filter_clickmaps(d['x'], d['y'])[1], image_shape)
                for j, d in enumerate(data) if j != i and filter_clickmaps(d['x'], d['y'])[0] is not None
            ]
            if len(remaining_maps) == 0:
                continue  # Skip if no valid remaining maps

            reference_map = np.mean(remaining_maps, axis=0)

            test_map = torch.tensor(test_map)
            reference_map = torch.tensor(reference_map)

            blurred_test_map = gaussian_blur(test_map.unsqueeze(0).unsqueeze(0), kernel).squeeze().numpy()
            blurred_reference_map = gaussian_blur(reference_map.unsqueeze(0).unsqueeze(0), kernel).squeeze().numpy()

            score = compute_spearman_correlation(blurred_test_map, blurred_reference_map)
            alignments.append(score)

        category_alignments.append((category, alignments))

    return category_alignments


def main():
    # Path for clickmap data
    clickme_data_path = "clickme_datasets/train_combined_11_26_2024.npz"

    # Load data from npz
    data = np.load(clickme_data_path, allow_pickle=True)
    file_pointers = data['file_pointer']
    clickmap_x = data['clickmap_x']
    clickmap_y = data['clickmap_y']

    # Group clickmaps by category
    category_data = {}
    for file_pointer, x_coords, y_coords in zip(file_pointers, clickmap_x, clickmap_y):
        category = file_pointer.split('/')[2]  # Extract synset ID
        if category not in category_data:
            category_data[category] = []
        category_data[category].append({'x': x_coords, 'y': y_coords})

    # TEMP: FILTER TO JUST 2 CATEGORIES FOR SPEED
    category_data = {k: category_data[k] for k in list(category_data.keys())[:2]}
    
    # Evaluate alignment scores
    image_shape = (256, 256)
    kernel_size = 21
    category_alignments = eval_alignment(category_data, image_shape, kernel_size=kernel_size)

    # Prepare data for plotting
    data = {"Category": [], "Alignment": []}
    for category, alignments in category_alignments:
        data["Category"].extend([category] * len(alignments))
        data["Alignment"].extend(alignments)

    df = pd.DataFrame(data)

    # Plot box-and-whisker plot
    plt.figure(figsize=(14, 10))
    df.boxplot(column="Alignment", by="Category", vert=False, showfliers=False, grid=False)
    plt.xlabel("Inter-Human Alignment")
    plt.ylabel("Category")
    plt.title("Inter-Human Alignment by ImageNet Category")
    plt.suptitle("")  # Remove default title from pandas
    plt.tight_layout()

    # Save and display plot
    output_file = "assets/imagenet_alignment.png"
    os.makedirs("assets", exist_ok=True)
    plt.savefig(output_file)
    print(f"Plot saved to: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()

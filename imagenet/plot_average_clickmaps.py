import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import torch
import torch.nn.functional as F
import pandas as pd
import re
import random
from scipy.stats import spearmanr
from PIL import Image
from synset_to_english import synset_to_english


def create_clickmap(click_points, image_shape):
    """
    Create a clickmap from click points.

    Args:
        click_points (list of tuples): List of (x, y) coordinates where clicks occurred.
        image_shape (tuple): Shape of the image (height, width).
        blur_kernel (torch.Tensor, optional): Gaussian kernel for blurring. Default is None.

    Returns:
        np.ndarray or torch.Tensor: A 2D array representing the clickmap, blurred if kernel provided.
    """
    BRUSH_SIZE = 5
    clickmap = np.zeros(image_shape, dtype=int)
    
    for point in click_points:
        if 0 <= point[1] < image_shape[0] and 0 <= point[0] < image_shape[1]:
            clickmap[max(0, point[1] - BRUSH_SIZE):min(image_shape[0], point[1] + BRUSH_SIZE),
                     max(0, point[0] - BRUSH_SIZE):min(image_shape[1], point[0] + BRUSH_SIZE)] += 1
        
    return clickmap

def gaussian_kernel(size, sigma):
    """
    Create a Gaussian kernel.

    Args:
        size (int): Size of the kernel.
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        torch.Tensor: A 2D Gaussian kernel with added batch and channel dimensions.
    """
    x_range = torch.arange(-(size-1)//2, (size-1)//2 + 1, 1)
    y_range = torch.arange((size-1)//2, -(size-1)//2 - 1, -1)

    xs, ys = torch.meshgrid(x_range, y_range, indexing='ij')
    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2)
    
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return kernel

def gaussian_blur(heatmap, kernel):
    """
    Apply Gaussian blur to a heatmap.

    Args:
        heatmap (torch.Tensor): The input heatmap (3D or 4D tensor).
        kernel (torch.Tensor): The Gaussian kernel.

    Returns:
        torch.Tensor: The blurred heatmap (3D tensor).
    """
    heatmap = heatmap.unsqueeze(0) if heatmap.dim() == 3 else heatmap
    blurred_heatmap = F.conv2d(heatmap, kernel, padding='same')
    return blurred_heatmap[0]

# Main function to save the PDF
def save_averaged_clickmaps_pdf(final_clickmaps, clickme_folder, 
    output_pdf_path="averaged_clickmaps.pdf", cmap="viridis", alpha=0.6, p=1.0, s=2.2, 
    blur_size=10, blur_sigma=10):
    
    blur_kernel = gaussian_kernel(size=int(blur_size), sigma=blur_sigma)

    with PdfPages(output_pdf_path) as pdf:
        fig, axes = plt.subplots(5, 2, figsize=(16, 20))  # Create a 5x2 grid per page
        axes = axes.flatten()  # Flatten the axes array for easy iteration
        
        for idx, image_key in enumerate(final_clickmaps.keys()):
            image_path = clickme_folder + image_key
            image_data = Image.open(image_path).convert("RGB")
            
            # Get the original height and width
            image_shape = (image_data.height, image_data.width)

            # Compute the average clickmap
            image_trials = final_clickmaps[image_key]
            avg_clickmap = np.mean([create_clickmap(trials, image_shape) for trials in image_trials], axis=0)

            # Apply Gaussian blur if needed
            avg_clickmap = gaussian_blur(torch.from_numpy(avg_clickmap).float().unsqueeze(0), blur_kernel).numpy()
            
            # Plot the image with the overlay
            ax = axes[idx % 10]  # Select the correct subplot (5x2 grid)
            ax.imshow(image_data)
            if avg_clickmap.sum() > 0:  # Check if the clickmap contains any non-zero values
                ax.imshow(avg_clickmap.squeeze(), cmap=cmap, alpha=alpha)
            class_label = synset_to_english(image_key.split('/')[1])
            ax.axis('off')
            ax.set_title(f"Image Class: {class_label}, N={len(image_trials)}")

            # Save the figure to the PDF every 10 images
            if (idx + 1) % 10 == 0 or (idx + 1) == len(final_clickmaps):
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                fig, axes = plt.subplots(5, 2, figsize=(16, 20))  # Start a new page
                axes = axes.flatten()

    print(f"PDF saved at {output_pdf_path}")

if __name__ == "__main__":
    clickme_folder = "/gpfs/data/tserre/irodri15/DATA/ILSVRC/Data/CLS-LOC/"
    clickme_data = pd.read_csv("clickme_data/clickme_maps_imagenet_20240902.csv")
    clickme_data = clickme_data[~clickme_data['image_path'].str.contains('CO3D')]

    clickmaps = {}
    
    # Step 1: Find image paths that repeat at least 5 times
    repeated_paths = clickme_data['image_path'].value_counts().loc[lambda x: x >= 5].index

    # Step 2: Include all rows with those image paths
    toy_data = clickme_data[clickme_data['image_path'].isin(np.random.choice(repeated_paths, 50, replace=False))].copy()
    filtered_data = clickme_data[clickme_data['image_path'].isin(repeated_paths)].copy()
    
    for index, row in toy_data.iterrows():
        image_file_name = row['image_path'].replace("imagenet/", "")
        if image_file_name not in clickmaps.keys():
            clickmaps[image_file_name] = [row["clicks"]]
        else:
            clickmaps[image_file_name].append(row["clicks"])


    number_of_maps = []

    final_clickmaps = {}

    counters = 0
    n_empty_clickmap = 0

    # Processing the data format
    for image in clickmaps:
        n_clickmaps = 0
        for clickmap in clickmaps[image]:
            
            # Empty Clickmaps
            if len(clickmap) == 2:
                n_empty_clickmap += 1
                continue

            # Increment the number of clickmaps only if the clickmaps aren't empty
            n_clickmaps += 1

            clean_string = re.sub(r'[{}"]', '', clickmap)

            # Split the string by commas to separate the tuple strings
            tuple_strings = clean_string.split(', ')

            # Zero indexing here because tuple_strings is a list with a single string
            data_list = tuple_strings[0].strip("()").split("),(")
            tuples_list = [tuple(map(int, pair.split(','))) for pair in data_list]

            if image not in final_clickmaps.keys():
                final_clickmaps[image] = []
            
            final_clickmaps[image].append(tuples_list)
            # Convert each string to a tuple of integers
            # tuples = [tuple(map(int, s.replace('(', '').replace(')', '').split(','))) for s in tuple_strings]
            # print("Tuples:", tuples)
            # final_clickmaps.append(tuples)

        number_of_maps.append(n_clickmaps)


    save_averaged_clickmaps_pdf(final_clickmaps, clickme_folder)

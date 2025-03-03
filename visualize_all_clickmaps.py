import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_data():
    # Load the clickmap arrays (list), synset IDs (list), and final clickmaps (dict)
    with open("assets/all_clickmaps.pkl", "rb") as f:
        all_clickmaps = pickle.load(f)
    with open("assets/categories.pkl", "rb") as f:
        categories = pickle.load(f)
    with open("assets/final_clickmaps.pkl", "rb") as f:
        final_clickmaps = pickle.load(f)
    return all_clickmaps, categories, final_clickmaps

def resize_and_crop(image, target_size=256):
    """
    Resize the image while preserving aspect ratio so that the smaller dimension equals target_size,
    then center crop to (target_size, target_size).
    """
    w, h = image.size
    scale = target_size / min(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image_resized = image.resize((new_w, new_h), Image.BICUBIC)
    
    # Calculate coordinates for a centered crop
    left = (new_w - target_size) // 2
    upper = (new_h - target_size) // 2
    right = left + target_size
    lower = upper + target_size
    image_cropped = image_resized.crop((left, upper, right, lower))
    return image_cropped

def visualize_overlay(all_clickmaps, categories, final_clickmaps):
    """
    For each clickmap array (of shape (10,256,256)) with its corresponding synset ID, find a unique
    image path from final_clickmaps (skipping if there is not exactly one match). Load and process
    the image, then display 10 overlayed heatmaps and their average on top of the image.
    """
    base_dir = "/Users/jaygopal/Downloads/val/"
    valid_count = 0
    for heatmaps, synset in zip(all_clickmaps, categories):
        # Find all keys in final_clickmaps that contain the synset ID
        matching_keys = [k for k in final_clickmaps.keys() if synset in k]
        # If not exactly one match, skip this image
        if len(matching_keys) != 1:
            continue
        image_key = matching_keys[0]
        image_path = os.path.join(base_dir, image_key)
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue
        
        try:
            # Load and convert the image to RGB
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # Resize and center crop the image to 256 x 256
        img_processed = resize_and_crop(img, target_size=256)
        
        # Create a figure with a 2x6 grid: 10 subplots for the individual heatmaps and 1 for the average
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        axes = axes.flatten()
        
        # Overlay each of the 10 individual heatmaps on the base image
        for i in range(10):
            ax = axes[i]
            ax.imshow(img_processed)
            im_overlay = ax.imshow(heatmaps[i], cmap='viridis', alpha=0.5)
            ax.set_title(f'Heatmap {i+1}')
            ax.axis('off')
            fig.colorbar(im_overlay, ax=ax)
        
        # Compute and overlay the average heatmap
        avg_heatmap = np.mean(heatmaps, axis=0)
        ax = axes[10]
        ax.imshow(img_processed)
        im_overlay = ax.imshow(avg_heatmap, cmap='viridis', alpha=0.5)
        ax.set_title('Average')
        ax.axis('off')
        fig.colorbar(im_overlay, ax=ax)
        
        # Hide any extra subplot(s) if they exist
        for ax in axes[11:]:
            ax.axis('off')
        
        plt.suptitle(f'Image with Synset: {synset}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        valid_count += 1
        if valid_count >= 5:
            break

def main():
    all_clickmaps, categories, final_clickmaps = load_data()
    visualize_overlay(all_clickmaps, categories, final_clickmaps)

if __name__ == '__main__':
    main()

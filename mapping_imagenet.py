import numpy as np
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
from harmonization.common import load_clickme_train
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
import scipy.stats
from skimage.metrics import structural_similarity as ssim


# === BLUR PARAMETERS ===
BLUR_SIZE = 21
BLUR_SIGMA = np.sqrt(BLUR_SIZE)
BLUR_SIGMA_FUNCTION = lambda x: x  # Identity function

def gaussian_kernel(size=BLUR_SIZE, sigma=BLUR_SIGMA):
    """Create a Gaussian kernel."""
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    y = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    xs, ys = torch.meshgrid(x, y, indexing="ij")

    kernel = torch.exp(-(xs**2 + ys**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # Normalize

    return kernel.unsqueeze(0).unsqueeze(0)  # Add batch & channel dimensions

def convolve(heatmap, kernel):
    """Apply convolution to the heatmap using the kernel."""
    heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Add batch & channel dims
    blurred = F.conv2d(heatmap, kernel, padding='same')
    return blurred.squeeze().numpy()  # Remove extra dims

def preprocess_heatmap(clickmap):
    """Convert raw clickmap to a 256x256 float32 heatmap, blur it, and center crop to 224x224."""
    full_size, target_size = 256, 224
    heatmap = np.zeros((full_size, full_size), dtype=np.float32)

    # Handle missing data
    if "x" not in clickmap or "y" not in clickmap or len(clickmap["x"]) == 0:
        print('ZERO MAP')
        return np.zeros((target_size, target_size), dtype=np.float32)

    # Plot clicks
    for x, y in zip(clickmap["x"], clickmap["y"]):
        x, y = int(round(float(x))), int(round(float(y)))
        if 0 <= x < full_size and 0 <= y < full_size:
            heatmap[y, x] += 1

    # Apply blur
    kernel = gaussian_kernel(size=BLUR_SIZE, sigma=BLUR_SIGMA_FUNCTION(BLUR_SIGMA))
    heatmap = convolve(heatmap, kernel)

    # Center crop
    start = (full_size - target_size) // 2
    cropped = heatmap[start:start + target_size, start:start + target_size]

    # Normalize
    final_result = cropped / np.max(cropped) if np.max(cropped) > 0 else cropped

    return final_result


def compute_ssim(map1, map2):
    """Compute SSIM between two heatmaps."""
    map2 = map2.squeeze()
    return ssim(map1, map2, data_range=map1.max() - map1.min())


# Updated plotting function
def plot_match(image, file_pointer, dump_hm, harm_hm, score, metric="Pearson Correlation"):
    """Plots the matched image, DUMP heatmap, and Harmonization heatmap."""
    
    # Un-normalize ImageNet (assuming image is normalized)
    image = image / 256
    # imagenet_mean = np.array([0.485, 0.456, 0.406])
    # imagenet_std = np.array([0.229, 0.224, 0.225])
    # image = (image * imagenet_std) + imagenet_mean  # Undo normalization
    image = np.clip(image, 0, 1)  # Keep values in range [0,1]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Show original image
    axs[0].imshow(image)
    axs[0].set_title(f"Image\n{file_pointer}", fontsize=8, wrap=True)
    axs[0].axis("off")

    # Show DUMP heatmap
    axs[1].imshow(dump_hm, cmap="jet", interpolation="nearest")
    axs[1].set_title("DUMP Heatmap", fontsize=10)
    axs[1].axis("off")

    # Show Harmonization heatmap
    axs[2].imshow(harm_hm.squeeze(), cmap="jet", interpolation="nearest")
    axs[2].set_title("Harmonization Heatmap", fontsize=10)
    axs[2].axis("off")

    # Add score information at the bottom
    fig.suptitle(f"{metric}: {score:.3f}", fontsize=10, y=0.05)

    plt.show()


# === CONFIGURATION ===
CLASS_TO_PROCESS = 260  # Change this for testing different classes
ERROR_THRESHOLD = 0.5  # Flag images below this Pearson correlation threshold
OUTPUT_FILE = f"assets/matched_class_{CLASS_TO_PROCESS}.npz"

# === LOAD DUMP VERSION ===
print("Loading DUMP version...")
clickme_dump = np.load("assets/clickme_dump.npz", allow_pickle=True)
dump_labels = clickme_dump["train_labels"]
dump_heatmaps = clickme_dump["train_heatmaps"]
dump_file_pointers = clickme_dump["train_file_pointers"]

# Extract only heatmaps belonging to the class we are processing
class_mask = dump_labels == CLASS_TO_PROCESS
dump_class_heatmaps = [dump_heatmaps[i] for i in np.where(class_mask)[0]]
dump_class_pointers = [dump_file_pointers[i] for i in np.where(class_mask)[0]]

# Convert all DUMP heatmaps
print("Preprocessing DUMP heatmaps...")
# Group heatmaps by file pointer
grouped_heatmaps = defaultdict(list)
for i, pointer in enumerate(dump_class_pointers):
    grouped_heatmaps[pointer].append(dump_class_heatmaps[i])

# Average heatmaps per file pointer
dump_processed_heatmaps = []
dump_class_pointers_unique = []

for pointer, heatmaps in grouped_heatmaps.items():
    averaged_heatmap = np.mean([preprocess_heatmap(hm) for hm in heatmaps], axis=0)
    dump_processed_heatmaps.append(averaged_heatmap)
    dump_class_pointers_unique.append(pointer)

dump_processed_heatmaps = np.array(dump_processed_heatmaps, dtype=np.float32)
dump_class_pointers = np.array(dump_class_pointers_unique)



# === LOAD HARMONIZATION DATASET ===
print("Loading HARMONIZATION version...")
clickme_dataset_train = load_clickme_train(batch_size=128)

# === MATCHING PROCESS ===
matched_images = []
matched_harmonization_heatmaps = []
matched_labels = []
matched_scores = []
matched_pointers = []

print("Processing HARMONIZATION dataset...")
available_dump_indices = set(range(len(dump_processed_heatmaps)))
for batch in tqdm(clickme_dataset_train):
    images = batch[0].numpy()  # Shape: (128, 224, 224, 3)
    harmonization_heatmaps = batch[1].numpy()  # Shape: (128, 224, 224, 1)
    one_hot_labels = batch[2].numpy()  # Shape: (128, 1000)
    
    # Convert one-hot encoding to class index
    batch_labels = np.argmax(one_hot_labels, axis=1)

    # Filter batch for only the class we are processing
    class_indices = np.where(batch_labels == CLASS_TO_PROCESS)[0]
    if len(class_indices) == 0:
        continue  # Skip this batch if no matching class found

    batch_harmonization_heatmaps = harmonization_heatmaps[class_indices]
    batch_images = images[class_indices]

    print('NUM IMGS IN THIS BATCH IN RIGHT CLASS', len(batch_images))

    # === FIND BEST MATCH ===
    for j, harm_hm in enumerate(batch_harmonization_heatmaps):
        # For each image in harmonization, reset the "best match" metrics
        best_score = -1
        best_index = None

        # Loop through our entire dump and 
        for i in list(available_dump_indices):
            dump_hm = dump_processed_heatmaps[i]
            # Flatten and compute score
            # score = np.corrcoef(dump_hm.flatten(), harm_hm.flatten())[0, 1]
            # score = -compute_emd(dump_hm, harm_hm)  # Negative because lower EMD = better match
            score = compute_ssim(dump_hm, harm_hm)

            if score > best_score:
                best_score = score
                best_index = j  # Store best matching harmonization image index

        # Save only if a valid match is found
        if best_index is not None:
            matched_images.append(batch_images[best_index])
            matched_harmonization_heatmaps.append(batch_harmonization_heatmaps[best_index])
            matched_labels.append(CLASS_TO_PROCESS)
            matched_scores.append(best_score)
            matched_pointers.append(dump_class_pointers[i])

            available_dump_indices.discard(i)  # Remove matched dump index

            # === PLOT DEBUGGING OUTPUT ===
            plot_match(batch_images[best_index], dump_class_pointers[i], dump_hm, batch_harmonization_heatmaps[best_index].squeeze(), best_score, metric="SSIM")
            

# Convert lists to NumPy arrays
matched_images = np.array(matched_images, dtype=np.float32)
matched_harmonization_heatmaps = np.array(matched_harmonization_heatmaps, dtype=np.float32)
matched_labels = np.array(matched_labels, dtype=np.int32)
matched_scores = np.array(matched_scores, dtype=np.float32)
matched_pointers = np.array(matched_pointers)

# Flag low-confidence matches
bad_matches = matched_scores < ERROR_THRESHOLD
print(f"⚠️ Flagging {bad_matches.sum()} images as having no good match (score < {ERROR_THRESHOLD})")

# === SAVE OUTPUT ===
np.savez(
    OUTPUT_FILE,
    images=matched_images,
    harmonization_heatmaps=matched_harmonization_heatmaps,
    labels=matched_labels,
    scores=matched_scores,
    file_pointers=matched_pointers
)
print(f"✅ Saved {len(matched_images)} matched images to {OUTPUT_FILE}")

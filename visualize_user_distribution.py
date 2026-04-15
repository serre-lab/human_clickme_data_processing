import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the train and val datasets
train_data = np.load("clickme_datasets/train_combined_11_26_2024.npz", allow_pickle=True)
val_data = np.load("clickme_datasets/val_combined_11_26_2024.npz", allow_pickle=True)

# Combine the datasets
combined_data = {
    key: np.concatenate([train_data[key], val_data[key]]) for key in train_data.files
}

# Extract user IDs
user_ids = combined_data['user_id']

# Count the number of maps per user
user_counts = pd.Series(user_ids).value_counts()

# Define bin size for thresholds and plotting
threshold_bin_size = 50
bin_sizes = [50, 1500]  # Bin sizes for plotting

# Calculate thresholds for superusers and light users
superuser_threshold = np.ceil(user_counts.quantile(0.95) / threshold_bin_size) * threshold_bin_size  # Round to nearest 50
light_user_threshold = max(threshold_bin_size, np.ceil(user_counts.quantile(0.25) / threshold_bin_size) * threshold_bin_size)  # At least one bin

# Identify superusers and light users
num_superusers = (user_counts > superuser_threshold).sum()
num_light_users = (user_counts < light_user_threshold).sum()

# Print the thresholds and counts
print(f"Superuser threshold: More than {superuser_threshold:.0f} maps per user")
print(f"Light user threshold: Fewer than {light_user_threshold:.0f} maps per user")
print(f"Number of superusers: {num_superusers}")
print(f"Number of light users: {num_light_users}")
print(f"Total number of users: {len(user_counts)}")

# Function to plot histogram
def plot_histogram(bin_size, title_suffix):
    plt.figure(figsize=(12, 6))
    plt.hist(user_counts, bins=np.arange(0, user_counts.max() + bin_size, bin_size), log=True, edgecolor='k', alpha=0.7)
    plt.axvline(superuser_threshold, color='r', linestyle='dashed', linewidth=1, label='Superuser Threshold')
    plt.axvline(light_user_threshold, color='g', linestyle='dashed', linewidth=1, label='Light User Threshold')

    # Add labels and title
    plt.xlabel("Number of Maps per User")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Distribution of Maps per User ({title_suffix})")
    plt.legend()
    plt.show()

# Generate plots for both bin sizes
for bin_size in bin_sizes:
    plot_histogram(bin_size, f"Bin Size {bin_size}")

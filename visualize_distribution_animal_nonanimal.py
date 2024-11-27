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

# Extract file pointers
file_pointers = np.array(combined_data['file_pointer'])

# Count the number of maps per unique image
df = pd.DataFrame({'file_pointer': file_pointers})
counts_df = df['file_pointer'].value_counts().reset_index()
counts_df.columns = ['file_pointer', 'num_maps']

# Extract synset IDs and filter out invalid entries
def extract_synset_id(file_pointer):
    try:
        parts = file_pointer.split('/')
        synset_id = parts[2]  # 'nXXXX'
        if synset_id.startswith('n') and synset_id[1:].isdigit():
            return synset_id
        else:
            return None
    except IndexError:
        return None

# Apply the extraction function
counts_df['synset_id'] = counts_df['file_pointer'].apply(extract_synset_id)

# Drop rows with invalid synset IDs
counts_df = counts_df.dropna(subset=['synset_id'])

# Convert synset_id to numeric and categorize as animal or non-animal
counts_df['synset_number'] = counts_df['synset_id'].str[1:].astype(int)
counts_df['is_animal'] = counts_df['synset_number'] < 2666196

# Split data into animal and non-animal groups
animal_data = counts_df[counts_df['is_animal']]
non_animal_data = counts_df[~counts_df['is_animal']]

# Compute statistics for animal and non-animal images
def compute_statistics(data, label):
    mean_maps = data['num_maps'].mean()
    percentiles = data['num_maps'].quantile([0.25, 0.5, 0.75])
    median_maps = percentiles.loc[0.5]
    percent_gt_5 = (data['num_maps'] > 5).mean() * 100

    print(f"--- {label} Images ---")
    print(f"Mean number of maps per image: {mean_maps:.2f}")
    print(f"25th percentile: {percentiles.loc[0.25]:.2f}")
    print(f"50th percentile (median): {median_maps:.2f}")
    print(f"75th percentile: {percentiles.loc[0.75]:.2f}")
    print(f"Percent of images with >5 maps per image: {percent_gt_5:.2f}%\n")

# Print statistics for each category
compute_statistics(animal_data, "Animal")
compute_statistics(non_animal_data, "Non-Animal")

# Map colors for plotting
counts_df['color'] = counts_df['is_animal'].map({True: 'blue', False: 'orange'})

# Prepare data for plotting
x = counts_df['num_maps']
y = counts_df['synset_number']
colors = counts_df['color']

# Create the scatter plot
plt.figure(figsize=(16, 10))  # Adjust figure size for better visualization
plt.scatter(x, y, c=colors, alpha=0.6, edgecolors='w', s=20)

# Add labels and title
plt.xlabel("Number of Maps per Image")
plt.ylabel("Category (Synset ID)")
plt.title("Number of Maps per Image vs. Categories (Animal vs. Non-Animal)")
plt.grid(True)

# Add a legend
plt.legend(['Animal', 'Non-Animal'], loc='upper right')

# Show the plot
plt.show()

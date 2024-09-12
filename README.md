# ClickMe Heatmap Generator

This project processes ClickMe data for CO3D images, generating heatmaps and analyzing click statistics.

## Features

- Processes ClickMe CSV data
- Generates heatmaps from click data
- Calculates median clicks per image, category, and overall
- Visualizes heatmaps for selected images
- Saves normalized heatmaps and click statistics

## Dependencies

- numpy
- pandas
- Pillow
- scipy
- torch
- matplotlib

## Usage

1. Ensure the `clickme_vCO3D.csv` file and `CO3D_ClickMe2` image directory are in the project root.
2. Run the script:

   ```
   python clickme.py
   ```

3. Output will be saved in the `assets` directory:
   - `co3d_clickmaps_normalized.npy`: Normalized heatmaps
   - `click_medians.json`: Median click statistics

## Main Functions

- `process_clickmaps()`: Processes the CSV data
- `make_heatmap()`: Generates heatmaps from click data
- `get_medians()`: Calculates median clicks for different groupings

## Notes

- The script uses a Gaussian blur for smoothing heatmaps
- Visualization is provided for a predefined set of example images
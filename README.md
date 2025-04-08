# ClickMe Heatmap Generator

This project processes ClickMe data for CO3D images, generating heatmaps and analyzing click statistics. It provides tools to compute various correlation metrics, including AUC, cross-entropy, Spearman, and RSA, to evaluate the quality of generated heatmaps.

Download additional files from [here](https://drive.google.com/drive/folders/1Ha4n2gK0Ze0ezRTDpPT2GuMpSy5TzTdQ)

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Scripts](#running-the-scripts)
    - [Compute Human Ceiling Split Half](#compute-human-ceiling-split-half)
    - [Compute Human Ceiling Hold One Out](#compute-human-ceiling-hold-one-out)
    - [Prepare Clickmaps for Modeling](#prepare-clickmaps-for-modeling)
    - [Visualize Clickmaps](#visualize-clickmaps)
- [Metrics](#metrics)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Processing:** Processes ClickMe CSV or NPZ data to generate clickmaps.
- **Heatmap Generation:** Creates heatmaps from click data with optional Gaussian blurring.
- **Correlation Analysis:** Calculates split-half correlations and null correlations using various metrics.
- **Visualization:** Provides tools for debugging and analysis of clickmaps and heatmaps.
- **Configuration Driven:** Easily configurable through YAML configuration files.

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/clickme-heatmap-generator.git
   cd clickme-heatmap-generator
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Configuration

Before running the scripts, ensure that the configuration file is properly set up. The project uses YAML configuration files located in the `configs/` directory. Below is an example of `configs/co3d_config.yaml`:

```yaml:configs/co3d_config.yaml
experiment_name: co3d
clickme_data: clickme_vCO3D.csv
image_dir: CO3D_ClickMe2
preprocess_db_data: False
blur_size: 21
min_clicks: 10  # Minimum number of clicks for a map to be included
max_clicks: 75  # Maximum number of clicks for a map to be included
min_subjects: 10  # Minimum number of subjects for an image to be included
null_iterations: 10
metric: auc  # Options: AUC, crossentropy, spearman, RSA
image_shape: [256, 256]
center_crop: [224, 224]
display_image_keys:
  - mouse/372_41138_81919_renders_00017.png
  - skateboard/55_3249_9602_renders_00041.png
  - couch/617_99940_198836_renders_00040.png
  - microwave/482_69090_134714_renders_00033.png
  - bottle/601_92782_185443_renders_00030.png
  - kite/399_51022_100078_renders_00049.png
  - carrot/405_54110_105495_renders_00039.png
  - banana/49_2817_7682_renders_00025.png
  - parkingmeter/429_60366_116962_renders_00032.png
```

**Key Configuration Parameters:**

- `experiment_name`: Name of the experiment.
- `clickme_data`: Path to the ClickMe data file (CSV or NPZ).
- `image_dir`: Directory containing the images.
- `preprocess_db_data`: Boolean indicating whether to preprocess database data.
- `blur_size`: Size of the Gaussian blur kernel.
- `min_clicks`: Minimum number of clicks required for a map to be included.
- `max_clicks`: Maximum number of clicks allowed for a map to be included.
- `min_subjects`: Minimum number of subjects required for an image to be included.
- `null_iterations`: Number of iterations for null correlation computations.
- `metric`: Correlation metric to use (`auc`, `crossentropy`, `spearman`, `RSA`).
- `image_shape`: Shape of the images (height, width).
- `center_crop`: Size for center cropping the images.
- `display_image_keys`: List of image keys to visualize.

### Running the Scripts

The repository provides several scripts for processing and analyzing ClickMe data. Below are the primary scripts along with examples using the updated configuration interface.

#### Compute Human Ceiling Split Half

This script performs split-half correlation analysis on the clickmaps.

**Script:** `compute_human_ceiling_split_half.py`

**Usage:**

```bash
python compute_human_ceiling_split_half.py --config configs/co3d_config.yaml
```

**Description:**

- Processes ClickMe data to generate clickmaps.
- Applies Gaussian blur and optional center cropping.
- Computes split-half correlations using the specified metric.
- Optionally visualizes the clickmaps for debugging purposes.
- Saves the results to `human_ceiling_results.npz`.

#### Compute Human Ceiling Hold One Out

This script performs a hold-one-out correlation analysis on the clickmaps using parallel processing to speed up computations.

**Script:** `compute_human_ceiling_hold_one_out.py`

**Usage:**

```bash
python compute_human_ceiling_hold_one_out.py --config configs/co3d_config.yaml
```

**Description:**

- Similar to the split-half script but uses a hold-one-out approach.
- Utilizes parallel processing with Joblib to compute correlations efficiently.
- Computes both split-half and null correlations using the specified metric.
- Saves the results to `human_ceiling_results.npz`.

**Note:** Adjust the `null_iterations` parameter in the configuration file to control the number of null correlation computations.

#### Prepare Clickmaps for Modeling

This script prepares the clickmaps for modeling by processing the data and saving the processed clickmaps and medians.

**Script:** `clickme_prepare_maps_for_modeling.py`

**Usage:**

```bash
python clickme_prepare_maps_for_modeling.py --config configs/co3d_config.yaml
```

**Description:**

- Processes ClickMe data to generate and prepare clickmaps.
- Applies Gaussian blur and center cropping as specified in the configuration.
- Saves the prepared clickmaps and median statistics to the `assets/` directory.

#### Visualize Clickmaps

This script visualizes the processed clickmaps and their corresponding images.

**Script:** `visualize_clickmaps.py`

**Usage:**

```bash
python visualize_clickmaps.py --config configs/co3d_config.yaml
```

**Description:**

- Loads the processed clickmaps.
- Visualizes the heatmaps alongside the original images.
- Saves the visualization images to the specified output directory.

## Metrics

The project supports the following correlation metrics to evaluate the quality of the generated heatmaps:

- **AUC (Area Under the Curve):** Measures the ability of the heatmap to discriminate between relevant and non-relevant regions.
- **Cross-Entropy:** Evaluates the difference between the predicted heatmap distribution and the target distribution.
- **Spearman:** Computes the Spearman rank-order correlation coefficient between two heatmaps.
- **RSA (Representational Similarity Analysis):** Assesses the similarity between two representations by comparing their correlation matrices.

You can specify the desired metric in the configuration file under the `metric` key.

## Project Structure

```plaintext:clickme-heatmap-generator/
├── compute_human_ceiling_hold_one_out.py
├── compute_human_ceiling_split_half.py
├── configs/
│   ├── co3d_config.yaml
│   └── jay_imagenet_0.1_config.yaml
├── utils.py
├── clickme_prepare_maps_for_modeling.py
├── visualize_clickmaps.py
├── requirements.txt
├── .gitignore
└── README.md
```

- **compute_human_ceiling_split_half.py:** Script for split-half correlation analysis.
- **compute_human_ceiling_hold_one_out.py:** Script for hold-one-out correlation analysis with parallel processing.
- **clickme_prepare_maps_for_modeling.py:** Script to prepare clickmaps for modeling.
- **visualize_clickmaps.py:** Script to visualize clickmaps alongside images.
- **configs/co3d_config.yaml:** Configuration file for the CO3D dataset.
- **configs/jay_imagenet_0.1_config.yaml:** Configuration file for the ImageNet dataset.
- **utils.py:** Utility functions for processing data, generating heatmaps, and computing metrics.
- **requirements.txt:** List of Python dependencies.
- **.gitignore:** Specifies files and directories to be ignored by Git.
- **README.md:** Project documentation.

## Dependencies

The project relies on the following Python packages:

- numpy
- pandas
- Pillow
- scipy
- torch
- matplotlib
- tqdm
- joblib
- torchvision
- PyYAML

Install all dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

   Provide a clear description of your changes and submit the pull request for review.

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to reach out if you have any questions or need further assistance!

# Human Clickme Data Processing

## GPU-Accelerated Blurring Feature

A significant performance optimization has been implemented to make the blurring process faster by leveraging GPU acceleration. The script now:

1. Pre-processes clickmaps in parallel on CPU with joblib
2. Runs blurring in batches on GPU for maximum performance
3. Post-processes results in parallel on CPU with joblib

### How to Use GPU Acceleration

The GPU acceleration is enabled by default. You can control it with the following config parameters:

- `use_gpu_blurring`: Boolean to enable/disable GPU acceleration (default: `true`)
- `gpu_batch_size`: Number of images to process in each GPU batch (default: `32`)

Example config:
```json
{
  "experiment_name": "my_experiment",
  "use_gpu_blurring": true,
  "gpu_batch_size": 64,
  "other_params": "..."
}
```

### Performance Considerations

- The optimal batch size depends on your GPU memory. Larger batches generally provide better performance but require more memory.
- You may need to adjust the batch size based on your image dimensions and GPU memory.
- If you experience out-of-memory errors, try reducing the batch size.

### Implementation Details

The implementation splits the work into three phases:
1. CPU-parallel pre-processing: Creates binary clickmaps from click coordinates
2. GPU batch processing: Applies blurring to multiple clickmaps simultaneously
3. CPU-parallel post-processing: Filters and processes the blurred maps

This approach significantly reduces processing time compared to the previous implementation where blurring was done sequentially.

## GPU Acceleration and Batch Size Configuration

The processing pipeline has been optimized for GPU acceleration with large batch processing. By default, the batch size is set to 1024 for both GPU operations and correlation computations.

### Batch Size Settings

You can control batch sizes through the configuration file:

```yaml
# GPU and parallelization settings
gpu_batch_size: 1024        # Batch size for GPU blurring operations
correlation_batch_size: 1024 # Batch size for correlation computations
n_jobs: -1                  # Number of CPU jobs (-1 for all cores)
```

If you experience out-of-memory errors on your GPU, try reducing the batch sizes.

### Updating Existing Configs

To update all your existing config files with these batch size settings, run:

```bash
python update_configs_with_batch_size.py
```

This will add `gpu_batch_size: 1024` and `correlation_batch_size: 1024` to all your YAML configuration files.

### Template Config

A template config file with these settings is available at `configs/batch_size_template.yaml`.
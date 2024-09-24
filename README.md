```markdown:README.md
# ClickMe Heatmap Generator

This project processes ClickMe data for CO3D images, generating heatmaps and analyzing click statistics. It provides tools to compute various correlation metrics, including AUC, cross-entropy, Spearman, and RSA, to evaluate the quality of generated heatmaps.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Scripts](#running-the-scripts)
    - [Compute Human Ceiling Split Half](#compute-human-ceiling-split-half)
    - [Compute Human Ceiling Hold One Out](#compute-human-ceiling-hold-one-out)
- [Metrics](#metrics)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Processing:** Processes ClickMe CSV data to generate clickmaps.
- **Heatmap Generation:** Creates heatmaps from click data with optional Gaussian blurring.
- **Correlation Analysis:** Calculates split-half correlations and null correlations using various metrics.
- **Visualization:** Provides tools for debugging and analysis.
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

Before running the scripts, ensure that the `configs/co3d_config.yaml` file is properly configured. Below is an example configuration:

```yaml:configs/co3d_config.yaml
experiment_name: co3d
co3d_clickme_data: clickme_vCO3D.csv
co3d_clickme_folder: CO3D_ClickMe2
blur_size: 21
min_clicks: 10  # Minimum number of clicks for a map to be included
max_clicks: 75 # Maximum number of clicks for a map to be included
min_subjects: 10  # Minimum number of subjects for an image to be included
null_iterations: 10
metric: auc  # AUC, crossentropy, spearman, RSA
image_shape: [256, 256]
center_crop: [224, 224]
```

Ensure that the paths and parameters match your dataset and desired settings.

### Running the Scripts

There are two main scripts provided in this repository for computing human ceiling correlations:

#### Compute Human Ceiling Split Half

This script performs split-half correlation analysis on the clickmaps.

**Script:** `compute_human_ceiling_split_half.py`

**Usage:**

```bash
python compute_human_ceiling_split_half.py
```

**Description:**

- Processes ClickMe data to generate clickmaps.
- Applies Gaussian blur and optional center cropping.
- Computes split-half correlations using the specified metric.
- Optionally, visualizes the clickmaps for debugging purposes.
- Saves the results to `human_ceiling_results.npz`.

**Example Command:**

```bash
python compute_human_ceiling_split_half.py
```

Ensure that the `co3d_config.yaml` is properly set up before running the script.

#### Compute Human Ceiling Hold One Out

This script performs a hold-one-out correlation analysis on the clickmaps using parallel processing to speed up computations.

**Script:** `compute_human_ceiling_hold_one_out.py`

**Usage:**

```bash
python compute_human_ceiling_hold_one_out.py
```

**Description:**

- Similar to the split-half script but uses a hold-one-out approach.
- Utilizes parallel processing with Joblib to compute correlations efficiently.
- Computes both split-half and null correlations using the specified metric.
- Saves the results to `human_ceiling_results.npz`.

**Example Command:**

```bash
python compute_human_ceiling_hold_one_out.py
```

**Note:** Adjust the `null_iterations` parameter in the configuration file to control the number of null correlation computations.

## Metrics

The project supports the following correlation metrics to evaluate the quality of the generated heatmaps:

- **AUC (Area Under the Curve):** Measures the ability of the heatmap to discriminate between relevant and non-relevant regions.
- **Cross-Entropy:** Evaluates the difference between the predicted heatmap distribution and the target distribution.
- **Spearman:** Computes the Spearman rank-order correlation coefficient between two heatmaps.
- **RSA (Representational Similarity Analysis):** Assess the similarity between two representations by comparing their correlation matrices.

You can specify the desired metric in the `co3d_config.yaml` file under the `metric` key.

## Project Structure

```plaintext:clickme-heatmap-generator/
├── compute_human_ceiling_hold_one_out.py
├── compute_human_ceiling_split_half.py
├── configs/
│   └── co3d_config.yaml
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

- **compute_human_ceiling_split_half.py:** Script for split-half correlation analysis.
- **compute_human_ceiling_hold_one_out.py:** Script for hold-one-out correlation analysis with parallel processing.
- **configs/co3d_config.yaml:** Configuration file containing parameters and paths.
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
```
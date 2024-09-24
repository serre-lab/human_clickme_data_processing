# ClickMe Heatmap Generator

This project processes ClickMe data for CO3D images, generating heatmaps and analyzing click statistics. It provides tools to compute various correlation metrics, including AUC, cross-entropy, Spearman, and RSA, to evaluate the quality of generated heatmaps.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Configuration](#configuration)
  - [Running the Script](#running-the-script)
- [Metrics](#metrics)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Processing:** Processes ClickMe CSV data to generate clickmaps.
- **Heatmap Generation:** Creates heatmaps from click data with optional Gaussian blurring.
- **Correlation Analysis:** Calculates split-half correlations and null correlations using various metrics.
- **Visualization:** Provides visualization tools for debugging and analysis.
- **Configuration Driven:** Easily configurable through YAML configuration files.

## Installation

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

#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=128G
#SBATCH -p gpu --gres=gpu:2
#SBATCH --account=carney-tserre-condo
#SBATCH -J Jay-ClickMe-Processing
#SBATCH -o logs/log-ClickMe-Processing-%j.out

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/oscar/data/tserre/jgopal/human_clickme_data_processing:$PYTHONPATH"

source /gpfs/data/tserre/jgopal/human_clickme_data_processing/jay-venv/bin/activate

python -u ceiling_floor_estimate.py configs/imagenet_val_oscar.yaml --save_json
python -u ceiling_floor_estimate.py configs/imagenet_train_oscar.yaml --save_json
python -u prepare_clickmaps.py configs/imagenet_val_oscar.yaml
python -u prepare_clickmaps.py configs/imagenet_train_oscar.yaml

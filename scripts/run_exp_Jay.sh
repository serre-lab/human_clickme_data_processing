#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem=256G
#SBATCH -p gpu --gres=gpu:1
#SBATCH --account=carney-tserre-condo
#SBATCH -J Jay-ClickMe-Processing
#SBATCH -o logs/log-ClickMe-Processing-%j.out


source /gpfs/data/tserre/jgopal/human_clickme_data_processing/jay-venv/bin/activate

python ceiling_floor_estimate.py configs/exp_configs/imagenet_val_oscar_max_5.yaml
python ceiling_floor_estimate.py configs/exp_configs/imagenet_val_oscar_max_10.yaml
python ceiling_floor_estimate.py configs/exp_configs/imagenet_val_oscar_max_15.yaml
python ceiling_floor_estimate.py configs/exp_configs/imagenet_val_oscar_max_20.yaml
python ceiling_floor_estimate.py configs/exp_configs/imagenet_val_oscar_max_25.yaml
python ceiling_floor_estimate.py configs/exp_configs/imagenet_val_oscar_max_30.yaml

# python ceiling_floor_estimate.py configs/balance_exp_configs/imagenet_val_oscar_max_5.yaml
# python ceiling_floor_estimate.py configs/balance_exp_configs/imagenet_val_oscar_max_10.yaml
# python ceiling_floor_estimate.py configs/balance_exp_configs/imagenet_val_oscar_max_15.yaml
# python ceiling_floor_estimate.py configs/balance_exp_configs/imagenet_val_oscar_max_20.yaml
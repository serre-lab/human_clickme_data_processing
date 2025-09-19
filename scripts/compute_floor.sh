#!/bin/bash
#SBATCH -J ImgNet_Floor
#SBATCH -N 1-1
#SBATCH -n 16
#SBATCH -t 96:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=512G
#SBATCH -p gpu-he
echo Starting execution at `date`
conda run -p ../../gs-perception/venv python ceiling_floor_estimate_large.py configs/imgnet_configs/imagenet_val_spearman_oscar.yaml
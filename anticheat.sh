python train_subject_classifier.py \
    --model-name gru \
    --epochs 20 \
    --output ./logs/gru_new_training.txt \
    --train-data-path ./clickme_datasets/train_imagenet_10_28_2024.npz \
    --val-data-path ./clickme_datasets/val_imagenet_10_28_2024.npz \

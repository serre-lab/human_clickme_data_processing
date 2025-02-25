# python train_subject_classifier.py \
#     --model-name gru \
#     --output ./logs/gru_new_training_eval.txt \
#     --epochs 1 \
#     --train-data-path ./clickme_datasets/train_imagenet_10_28_2024.npz \
#     --val-data-path ./clickme_datasets/val_combined_11_26_2024.npz \
#     --checkpoint-path ./checkpoints/model_epoch_ckpt_19.pth \
#     --evaluate-only

# python inference.py \
#     --model-path ./checkpoints/model_epoch_ckpt_19.pth \
#     --data-path ./clickme_datasets/val_combined_11_26_2024.npz \
#     --output-path ./clickme_datasets/output_filtered_data.npz \
#     --model-name gru \
#     --threshold 0.003

python train_subject_classifier_sequence_v2.py \
    --output ./logs/training_log.txt \
    --epochs 20 \
    --batch-size 1 \
    --train-data-path ./clickme_datasets/train_SANDBOX_01_25_2025.npz \
    --val-data-path ./clickme_datasets/val_SANDBOX_01_25_2025.npz \


# Classification Summary:
# Threshold for bad_ratio: 0.0
# Total users: 1180
# Bad users (bad_ratio > threshold): 119 (10.08%)
# Good users (bad_ratio ≤ threshold): 1061 (89.92%)

# Data Summary:
# Original samples: 174139
# Samples after filtering: 25868
# Filtered data saved to: ./clickme_datasets/output_filtered_data.npz

# Bad threshold:  0.001

# Classification Summary:
# Threshold for bad_ratio: 0.001
# Total users: 1180
# Bad users (bad_ratio > threshold): 114 (9.66%)
# Good users (bad_ratio ≤ threshold): 1066 (90.34%)

# Data Summary:
# Original samples: 174139
# Samples after filtering: 35997

# Bad threshold:  0.003

# Classification Summary:
# Threshold for bad_ratio: 0.003
# Total users: 1180
# Bad users (bad_ratio > threshold): 79 (6.69%)
# Good users (bad_ratio ≤ threshold): 1101 (93.31%)

# Data Summary:
# Original samples: 174139
# Samples after filtering: 111744

# Bad threshold:  0.005

# Classification Summary:
# Threshold for bad_ratio: 0.005
# Total users: 1180
# Bad users (bad_ratio > threshold): 49 (4.15%)
# Good users (bad_ratio ≤ threshold): 1131 (95.85%)

# Data Summary:
# Original samples: 174139
# Samples after filtering: 154146

# Bad threshold:  0.01

# Classification Summary:
# Threshold for bad_ratio: 0.01
# Total users: 1180
# Bad users (bad_ratio > threshold): 28 (2.37%)
# Good users (bad_ratio ≤ threshold): 1152 (97.63%)

# Data Summary:
# Original samples: 174139
# Samples after filtering: 171480
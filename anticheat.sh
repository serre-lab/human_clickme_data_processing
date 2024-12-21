# python train_subject_classifier.py \
#     --model-name gru \
#     --output ./logs/gru_new_training_eval.txt \
#     --epochs 1 \
#     --train-data-path ./clickme_datasets/train_imagenet_10_28_2024.npz \
#     --val-data-path ./clickme_datasets/val_combined_11_26_2024.npz \
#     --checkpoint-path ./checkpoints/model_epoch_ckpt_19.pth \
#     --evaluate-only

python inference.py \
    --model-path ./checkpoints/model_epoch_ckpt_19.pth \
    --data-path ./clickme_datasets/val_combined_11_26_2024.npz \
    --output-path ./clickme_datasets/output_filtered_data.npz \
    --model-name gru \
    --thredhold 0.0
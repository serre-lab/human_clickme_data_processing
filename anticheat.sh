python train_subject_classifier.py \
    --model-name gru \
    --epochs 5 \
    --output ./logs/evaluate_jay_model.txt \
    --checkpoint-path ./checkpoints/model_epoch_5.pth \
    --data-path ./clickme_datasets/prj_clickmev2_train_imagenet_10_10_2024.npz \
    # --evaluate-only true 

python get_file_info_from_dump.py jay_imagenet_val_0.1.yaml
python get_file_info_from_dump.py jay_imagenet_train_0.1.yaml

python clickme_prepare_maps_for_modeling.py jay_imagenet_val_0.1.yaml
python clickme_prepare_maps_for_modeling.py jay_imagenet_for_co3d_val_0.1.yaml
# python clickme_prepare_maps_for_modeling.py co3d_val.yaml

python compute_human_ceiling_split_half.py jay_imagenet_val_0.1.yaml
python compute_human_ceiling_split_half.py jay_imagenet_for_co3d_val_0.1.yaml
# python compute_human_ceiling_split_half.py co3d_val.yaml
python clickme_prepare_maps_for_modeling.py jay_imagenet_train_0.1.yaml
python compute_human_ceiling_split_half.py jay_imagenet_train_0.1.yaml

# python compute_human_ceiling_hold_one_out.py jay_imagenet_train_0.1.yaml
# python compute_human_ceiling_hold_one_out.py jay_imagenet_val_0.1.yaml
# python compute_human_ceiling_hold_one_out.py jay_imagenet_for_co3d_val_0.1.yaml
# python compute_human_ceiling_hold_one_out.py co3d_val.yaml



# Train:
# Mean human correlation full set: 0.08664774537280222
# Null correlations full set: 0.027026184604559984

# Val:
# Mean human correlation full set: 0.09120038285655054
# Null correlations full set: 0.028687646741805756

# Val_imagenet_co3d:
# Mean human correlation full set: 0.09161272922789762
# Null correlations full set: 0.02913157979315741

# co3d:
# Mean human correlation full set: 0.16571727314019297
# Null correlations full set: 0.023708055710504


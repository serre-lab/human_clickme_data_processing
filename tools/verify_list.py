import json
import os

if __name__ == "__main__":
    img_list = "clickme_datasets/missing_imgnet_train.json"
    imgnet_path = "/gpfs/data/shared/imagenet/ILSVRC2012/train"
    with open(img_list, 'r') as f:
        json_content = json.load(f)
    
    for img_cls, img_list in json_content.items():
        for img in img_list:
            if not os.path.exists(os.path.join(imgnet_path, img_cls, img)):
                print("Missing", img)
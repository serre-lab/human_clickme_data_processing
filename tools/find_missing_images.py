import os
import json
from tools.find_top_bottom import get_num_subjects

if __name__ == "__main__":
    num_subjects = get_num_subjects()
    missing_images = {"val":[]}
    print(len(num_subjects))
    for img_name, num_subjects in num_subjects.items():
        if num_subjects < 5:
            missing_images["val"].append((img_name, num_subjects))
            print(img_name, num_subjects)

    with open("missing_val.json", 'w') as f:
        json_content = json.dumps(missing_images, indent=4)
        f.write(json_content)
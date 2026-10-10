"""
@article{fernandez2024uninformed,
  title={Uninformed teacher-student for hard-samples distillation in weakly supervised mitosis localization},
  author={Fernandez-Mart{\'\i}n, Claudio and Silva-Rodriguez, Julio and Kiraz, Umay and Morales, Sandra and Janssen, Emiel AM and Naranjo, Valery},
  journal={Computerized Medical Imaging and Graphics},
  volume={112},
  pages={102328},
  year={2024},
  publisher={Elsevier}
}


cases 41 to 45,91 to 95, and 141 to 145 are used for validation,
and cases 46 to 50, 96 to 100, and 146 to 150 are incorporated into the testing subset.
The rest of the cases are used for the training subset.

train:
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
       31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
       69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105,
       106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
       128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140]
val:
    [41, 42, 43, 44, 45, 91, 92, 93, 94, 95, 141, 142, 143, 144, 145]
test:
    [46, 47, 48, 49, 50, 96, 97, 98, 99, 100, 146, 147, 148, 149, 150]

"""
import json
import os
import random
import shutil

import numpy as np
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor


def do_convert(wsi_file, png_path, label_path):
    image_id = int(wsi_file[:3])
    file_name = wsi_file.split('.')[0]
    file_path = os.path.join(wsi_path, wsi_file)
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    mask = np.zeros((height, width)).astype(np.uint8)

    annotations = [anno for anno in data["annotations"] if
                   anno["image_id"] == image_id and anno["category_id"] == 1]

    if len(annotations) > 0:
        for anno in annotations:
            bbox_arr = anno["bbox"]
            x = bbox_arr[1]
            y = bbox_arr[0]
            x = int(x + 25)
            y = int(y + 25)
            mask[x, y] = 1

    overlap = 0
    for h in range(0, height, patch_size - overlap):
        h_start = h
        h_end = h + patch_size
        if h_end > height:
            h_start = height - patch_size
            h_end = height
        for w in range(0, width, patch_size - overlap):
            w_start = w
            w_end = w + patch_size
            if w_end > width:
                w_start = width - patch_size
                w_end = width

            # 切割块
            png_patch = img[h_start:h_end, w_start:w_end]
            mask_patch = mask[h_start:h_end, w_start:w_end]

            png_name = f"{file_name}_{w_start}_{h_start}.png"
            txt_name = f"{file_name}_{w_start}_{h_start}.txt"

            if mask_patch.max() == 0:
                Image.fromarray(png_patch).save(png_path + png_name)
            else:
                points = np.argwhere(mask_patch == 1)
                for point in points:
                    label = (
                        0,
                        point[1] / patch_size,
                        point[0] / patch_size,
                        react_size_percent,
                        react_size_percent
                    )
                    with open(label_path + txt_name, 'a') as f:
                        f.write(('%g ' * len(label)).rstrip() % label + '\n')
                Image.fromarray(png_patch).save(png_path + png_name)
    print(f"{wsi_file} done")


def move_files(files):
    for file in files:
        file_name = file.split('.')[0]
        png_name = f"{file_name}.png"
        txt_name = f"{file_name}.txt"

        png_src_path = os.path.join(png_path, png_name)
        label_src_path = os.path.join(label_path, txt_name)

        png_dest_path = png_src_path.replace("train", "val")
        label_dest_path = label_src_path.replace("train", "val")

        shutil.move(png_src_path, png_dest_path)
        if os.path.exists(label_src_path):
            shutil.move(label_src_path, label_dest_path)


if __name__ == '__main__':
    seed = 3407
    np.random.seed(seed)
    random.seed(seed)

    wsi_path = "./dataset/MIDOG2021/tiff_sa/"
    midog_json = "./dataset/MIDOG2021/MIDOG.json"
    data = json.load(open(midog_json))
    png_path = f"./dataset/MIDOG2021/det/images/train/"
    label_path = f"./dataset/MIDOG2021/det/labels/train/"
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    patch_size = 640
    react_size = 50
    react_size_percent = react_size / patch_size

    train_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
       31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
       69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105,
       106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
       128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140]

    with ThreadPoolExecutor(max_workers=20) as executor:
        for wsi_file in os.listdir(wsi_path):
            image_id = int(wsi_file[:3])
            if image_id in train_arr:
                executor.submit(do_convert, wsi_file, png_path, label_path)
            
    print("convert train done")

    val_png_path = png_path.replace("train", "val")
    val_label_path = label_path.replace("train", "val")
    os.makedirs(val_png_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)

    train_ratio = 0.8
    png_files = os.listdir(png_path)
    random.shuffle(png_files)
    total_files = len(png_files)
    train_end = int(total_files * train_ratio)
    val_files = png_files[train_end:]
    move_files(val_files)

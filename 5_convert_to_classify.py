"""
将 MIDOG 2021 中  tif 转换成 classify 系列需要的数据集

https://midog.deepmicroscopy.org/download-dataset/

m21
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

from PIL import Image
from concurrent.futures import ThreadPoolExecutor

wsi_path = "./dataset/MIDOG2021/tiff_sa/"
midog_json = "./dataset/MIDOG2021/MIDOG.json"
json_data = json.load(open(midog_json))
p_path = f"./dataset/MIDOG2021/cls/train/p/"
n_path = f"./dataset/MIDOG2021/cls/train/n/"
os.makedirs(p_path, exist_ok=True)
os.makedirs(n_path, exist_ok=True)

patch_size = 64  # 训练 image 大小
patch_offset = patch_size / 2

train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
       31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
       69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105,
       106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
       128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140]

val = [41, 42, 43, 44, 45, 91, 92, 93, 94, 95, 141, 142, 143, 144, 145]


def do_convert(wsi_file):
    file_name = wsi_file.split('.')[0]
    file_path = os.path.join(wsi_path, wsi_file)
    img = Image.open(file_path)
    width, height = img.size
    annotations = [anno for anno in json_data["annotations"] if anno["image_id"] == int(wsi_file[:3])]
    if len(annotations) > 0:
        for anno in annotations:
            bbox_arr = anno["bbox"]
            center_w = int((bbox_arr[0] + bbox_arr[2]) / 2)
            center_h = int((bbox_arr[1] + bbox_arr[3]) / 2)
            h_start = center_h - patch_offset
            h_end = center_h + patch_offset
            if h_start < 0:
                h_start = 0
                h_end = patch_size
            if h_end > height:
                h_start = height - patch_size
                h_end = height

            w_start = center_w - patch_offset
            w_end = center_w + patch_offset
            if w_start < 0:
                w_start = 0
                w_end = patch_size
            if w_end > width:
                w_start = width - patch_size
                w_end = width
            # (left, upper, right, lower)
            patch = img.crop((w_start, h_start, w_end, h_end))
            category_id = anno["category_id"]
            if category_id == 1:
                # 正样本
                patch.save(f'{p_path}/{file_name}_{center_w}_{center_h}.png')
            else:
                patch.save(f'{n_path}/{file_name}_{center_w}_{center_h}.png')
    print(f"{wsi_file} done")


wsi_infer_arr = []
for wsi_file in os.listdir(wsi_path):
    image_id = int(wsi_file[:3])
    if image_id in train:
        wsi_infer_arr.append(wsi_file)


print(f"train num: {len(wsi_infer_arr)}")
with ThreadPoolExecutor(max_workers=20) as executor:
    for wsi_file in wsi_infer_arr:
        executor.submit(do_convert, wsi_file)


print("classify train convert done")

wsi_path = "./dataset/MIDOG2021/tiff_sa/"
p_path = f"./dataset/MIDOG2021/cls/val/p/"
n_path = f"./dataset/MIDOG2021/cls/val/n/"
os.makedirs(p_path, exist_ok=True)
os.makedirs(n_path, exist_ok=True)

wsi_infer_arr = []
for wsi_file in os.listdir(wsi_path):
    image_id = int(wsi_file[:3])
    if image_id in val:
        wsi_infer_arr.append(wsi_file)


print(f"val num: {len(wsi_infer_arr)}")
with ThreadPoolExecutor(max_workers=20) as executor:
    for wsi_file in wsi_infer_arr:
        executor.submit(do_convert, wsi_file)
import numpy as np
import torch
from PIL import Image
from scipy.spatial import KDTree
from torch.backends import cudnn
from tqdm import tqdm

cudnn.enabled = True
import os.path
import cv2
import json
from torch.utils.data import DataLoader, Dataset
from utils.general import non_max_suppression
from models.common import DetectMultiBackend


class BuildDetectDataset(Dataset):
    def __init__(self, wsi, start_points):
        self.wsi = wsi
        self.start_points = start_points

    def __getitem__(self, index):
        [h_start, h_end, w_start, w_end] = self.start_points[index]
        img_item = self.wsi[h_start:h_end, w_start:w_end]
        img_item = img_item.transpose((2, 0, 1))
        return img_item, h_start, w_start

    def __len__(self):
        return len(self.start_points)


def xyxy2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    z = y[:, 0:2]
    return z


def check_gt(pred_x, pred_y, annotations, threshold=30):
    if not annotations:
        return False, -1
    anno_points = np.array([[anno["bbox"][0] + 25, anno["bbox"][1] + 25] for anno in annotations])
    tree = KDTree(anno_points)
    distance, index = tree.query([pred_x, pred_y])
    if distance <= threshold:
        return True, index
    return False, -1


def get_crop_img(w, h, width, height, slide, crop_img_size=64):
    offset = crop_img_size // 2
    h_start = h - offset
    h_end = h + offset
    w_start = w - offset
    w_end = w + offset

    if h_start < 0:
        h_start = 0
        h_end = crop_img_size
    if h_end > height:
        h_start = height - crop_img_size
        h_end = height

    if w_start < 0:
        w_start = 0
        w_end = crop_img_size
    if w_end > width:
        w_start = width - crop_img_size
        w_end = width
    crop_img = slide[h_start:h_end, w_start:w_end, :]
    crop_img = crop_img.astype(np.uint8)
    crop_img = Image.fromarray(crop_img).convert('RGB')
    return crop_img


device = torch.device('cuda:0')
wsi_path = './dataset/dataset/MIDOG2021/tiff_sa/'
midog_json = './dataset/dataset/MIDOG2021/MIDOG.json'
json_data = json.load(open(midog_json))

if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
       31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
       69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 101, 102, 103, 104, 105,
       106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
       128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140]

    batch_size = 64
    stage1_patch_size = 640
    weights_stage1 = f"./det_pretrained.pt"
    model_detect = DetectMultiBackend(weights=weights_stage1, device=device)
    model_detect.eval()
    model_detect.warmup(imgsz=(batch_size, 3, stage1_patch_size, stage1_patch_size))

    out_folder = f"./dataset/cls/train"
    os.makedirs(out_folder, exist_ok=True)

    wsi_infer_arr = []
    for wsi_file in os.listdir(wsi_path):
        image_id = int(wsi_file[:3])
        if image_id in train_arr:
            wsi_infer_arr.append(wsi_file)

    for wsi_file in tqdm(wsi_infer_arr):
        file_path = os.path.join(wsi_path, wsi_file)
        slide = cv2.imread(file_path)
        slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
        height, width, _ = slide.shape
        detect_points = []
        patch_size = stage1_patch_size
        overlap = 60
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
                point = (h_start, h_end, w_start, w_end)
                detect_points.append(point)

        dataset = BuildDetectDataset(slide, detect_points)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        wsi_anno = [anno for anno in json_data["annotations"] if
                    anno["image_id"] == int(wsi_file[:3]) and anno["category_id"] == 1]
        matched = []
        file_name = wsi_file.split('.')[0]
        for imgs, start_h_arr, start_w_arr in data_loader:
            ims = imgs.to(device, non_blocking=True).float() / 255
            preds = model_detect(ims)
            preds = non_max_suppression(preds, conf_thres=0.01)
            for det, img, det1_start_w, det1_start_h in zip(preds, imgs, start_w_arr, start_h_arr):
                if len(det):
                    for *xyxy, conf, cls in reversed(det):
                        xy = (xyxy2xy(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                        w = int(xy[0] + det1_start_w)
                        h = int(xy[1] + det1_start_h)
                        crop_img = get_crop_img(w, h, width, height, slide)
                        gt, match_index = check_gt(w, h, wsi_anno)
                        if not gt:
                            crop_img.save(f"{out_folder}/n/{file_name}_{w}_{h}.png")
        print(f"{wsi_file} done")

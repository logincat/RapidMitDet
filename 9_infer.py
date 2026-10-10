import torch
import torchvision
from PIL import Image
import os.path
import cv2
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
from scipy.spatial import KDTree
import numpy as np


class BuildDetectDataset(Dataset):
    def __init__(self, wsi, start_points, patch_size):
        self.wsi = wsi
        self.start_points = start_points
        self.patch_size = patch_size

    def __getitem__(self, index):
        [h_start, h_end, w_start, w_end] = self.start_points[index]
        img_item = self.wsi[h_start:h_end, w_start:w_end]
        img_item = img_item.transpose((2, 0, 1))
        return img_item, h_start, w_start

    def __len__(self):
        return len(self.start_points)


from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class BuildClassifyDataset(Dataset):
    def __init__(self, wsi, center_points, patch_size):
        self.wsi = wsi
        self.center_points = center_points
        self.patch_size = patch_size

    def __getitem__(self, index):
        [center_h, center_w] = self.center_points[index]
        height, width, _ = self.wsi.shape

        offset = self.patch_size // 2
        h_start = center_h - offset
        h_end = center_h + offset
        if h_start < 0:
            h_start = 0
            h_end = self.patch_size
        if h_end > height:
            h_start = height - self.patch_size
            h_end = height

        w_start = center_w - offset
        w_end = center_w + offset
        if w_start < 0:
            w_start = 0
            w_end = self.patch_size
        if w_end > width:
            w_start = width - self.patch_size
            w_end = width

        patch = self.wsi[h_start:h_end, w_start:w_end, :]
        img_item = Image.fromarray(patch).convert('RGB')
        img_item = transform(img_item)
        return img_item, center_h, center_w

    def __len__(self):
        return len(self.center_points)


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


def remove_repeat_points(arr, threshold=20):
    if len(arr) == 0:
        return []
    arr = np.array(arr)
    tree = KDTree(arr)
    unique_indices = []
    visited = set()
    for i in range(len(arr)):
        if i in visited:
            continue
        unique_indices.append(i)
        neighbors = tree.query_ball_point(arr[i], threshold)
        visited.update(neighbors)
    return arr[unique_indices].tolist()


def run(
        model_det,
        model_cls,
        device,
        wsi_path,
        patch_size=640,
        stage2_patch_size=64,
        overlap=60,
        is_level_20X=False,
        det_conf=0.5,
        cls_conf=0.5,
        batch_size=100,

):
    file_path = wsi_path
    wsi_file = os.path.basename(wsi_path)
    slide = cv2.imread(file_path)
    slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
    height, width, _ = slide.shape
    if is_level_20X:
        slide = cv2.resize(slide, (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
    detect_points = []
    height, width, _ = slide.shape
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

    dataset = BuildDetectDataset(slide, detect_points, patch_size)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    need_classify_arr = []
    for imgs, start_h_arr, start_w_arr in data_loader:
        ims = imgs.to(device, non_blocking=True).float() / 255
        preds = model_det(ims)
        preds = non_max_suppression(preds, conf_thres=det_conf)
        for det, img, det1_start_w, det1_start_h in zip(preds, imgs, start_w_arr, start_h_arr):
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    xy = (xyxy2xy(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    center_w = xy[0] + det1_start_w
                    center_h = xy[1] + det1_start_h
                    if is_level_20X:
                        center_w = center_w * 2
                        center_h = center_h * 2
                    need_classify_arr.append([int(center_h), int(center_w)])
    slide = cv2.imread(file_path)
    slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
    stage2_dataset = BuildClassifyDataset(slide, need_classify_arr, stage2_patch_size)
    stage2_data_loader = DataLoader(stage2_dataset, batch_size=batch_size)
    final_arr = []
    for img_40xs, h_centers, w_centers in stage2_data_loader:
        img_40xs = img_40xs.to(device).float()
        logits = model_cls(img_40xs)
        cls_preds = torch.softmax(logits, 1).cpu().detach().numpy()
        for (img_40x, h_center, w_center, pred) in zip(img_40xs, h_centers, w_centers, cls_preds):
            c_conf = pred[1]
            if c_conf > cls_conf:
                final_arr.append((int(w_center), int(h_center)))

    final_arr = remove_repeat_points(final_arr)
    for x, y in final_arr:
        cv2.circle(slide, (int(x), int(y)), 25, (0, 255, 0), 5)
    slide = cv2.cvtColor(slide, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"./{wsi_file[:3]}.png", slide)


import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--wsi_path', type=str, default='./dataset/MIDOG2021/tiff/XXX.tiff')
    parser.add_argument('--weights_stage1', type=str, default='./ckpts/det.pt')
    parser.add_argument('--weights_stage1_20x', type=str, default='./ckpts/det_align.pt')
    parser.add_argument('--weights_stage2', type=str, default='./ckpts/cls.pt')
    parser.add_argument('--patch_size', type=int, default=640)
    parser.add_argument('--stage2_patch_size', type=int, default=64)
    parser.add_argument('--overlap', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--det_conf', type=float, default=0.007)
    parser.add_argument('--cls_conf', type=float, default=0.32)
    parser.add_argument('--is_level_20X', action='store_true', default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    patch_size = args.patch_size
    overlap = args.overlap
    weights_stage1 = args.weights_stage1
    weights_stage2 = args.weights_stage2

    if args.is_level_20X:
        patch_size = args.patch_size // 2
        overlap = args.overlap // 2
        weights_stage1 = args.weights_stage1_20x

    model_detect = DetectMultiBackend(weights=weights_stage1, device=device)
    model_detect.warmup(imgsz=(args.batch_size, 3, patch_size, patch_size))
    model_detect.eval()

    model_classify = torchvision.models.resnet152(num_classes=2)
    model_classify.maxpool = nn.Identity()
    model_classify.load_state_dict(torch.load(weights_stage2, map_location='cpu'))
    model_classify.to(device)
    model_classify.eval()

    run(
        model_det=model_detect,
        model_cls=model_classify,
        device=device,
        wsi_path=args.wsi_path,
        patch_size=patch_size,
        stage2_patch_size=args.stage2_patch_size,
        is_level_20X=args.is_level_20X,
        overlap=overlap,
        det_conf=args.det_conf,
        cls_conf=args.cls_conf
    )


if __name__ == '__main__':
    main()

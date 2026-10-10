import os
import random
import warnings

import numpy as np
import redis
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torchmetrics import F1Score
from torchvision import transforms
from tqdm import tqdm

warnings.filterwarnings("ignore")

import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x6 = self.avgpool(x5)
        x6 = x6.reshape(x6.size(0), -1)
        x = self.fc(x6)
        return x, x6

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        x = self.avgpool(x5)

        return x


class FocalLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.1, weight: torch.Tensor = None, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_onehot = F.one_hot(target, num_classes=input.size(1))
        target_onehot_labelsmoothing = torch.clamp(target_onehot.float(),
                                                   min=self.label_smoothing / (input.size(1) - 1),
                                                   max=1.0 - self.label_smoothing)
        input_softmax = F.softmax(input, dim=1) + 1e-7
        input_logsoftmax = torch.log(input_softmax)
        ce = -1 * input_logsoftmax * target_onehot_labelsmoothing
        fl = torch.pow((1 - input_softmax), self.gamma) * ce
        fl = fl.sum(1) * self.weight[target.long()]
        return fl.mean()


def setting_optimizer(model):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)
    optimizer = optim.AdamW(g[2], lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    optimizer.add_param_group({'params': g[0], 'weight_decay': 5e-4})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer


class BidirectionalDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_to_keys = {}
        for key, value in self.items():
            self._add_reverse_mapping(key, value)

    def _add_reverse_mapping(self, key, value):
        if value not in self.value_to_keys:
            self.value_to_keys[value] = []
        self.value_to_keys[value].append(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._add_reverse_mapping(key, value)

    def get_keys(self, value):
        return self.value_to_keys.get(value, [])

    def get_keys_not_equal(self, value):
        return [
            key for key, val in self.items()
            if val != value
        ]


class BuildDataset(Dataset):
    def __init__(self, root):
        self.samples, self.targets = self.load_dataset(root)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, idx):
        anchor_path = self.samples[idx]
        anchor_label = self.targets[idx]
        positive_idx = np.where(np.array(self.targets) == anchor_label)[0]
        positive_idx = positive_idx[positive_idx != idx]
        positive = self.samples[torch.randint(len(positive_idx), (1,))]

        key_path = anchor_path.replace(data_root, "")
        negative = rdb.get(key_path)
        if negative is None:
            negative_idx = np.where(np.array(self.targets) != anchor_label)[0]
            negative = self.samples[negative_idx[torch.randint(len(negative_idx), (1,))]]
        else:
            negative = data_root + negative

        anchor_emb = Image.open(anchor_path).convert("RGB")
        anchor_emb = self.transform(anchor_emb)

        positive = Image.open(positive).convert("RGB")
        positive = self.transform(positive)

        negative = Image.open(negative).convert("RGB")
        negative = self.transform(negative)

        return anchor_emb, anchor_path, anchor_label, positive, negative

    def __len__(self) -> int:
        return len(self.samples)

    def load_dataset(self, root_dir):
        dataset = []
        targets = []
        for subdir, dirs, files in os.walk(root_dir):
            if subdir == root_dir:
                continue
            basename = os.path.basename(subdir)
            if basename == "n":
                label = 0
            else:
                label = 1
            for file in files:
                file_path = os.path.join(subdir, file)
                dataset.append(file_path)
                targets.append(label)

        return dataset, targets


if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch_size = 32
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    rdb = redis.Redis(host='localhost', port=6379, db=3, password="abc123456", decode_responses=True)
    rdb.flushdb()
    print("fulshdb")
    model_save_name = f"model_name"
    data_root = "./dataset/cls"
    train_path = f"{data_root}/train"
    val_path = f"{data_root}/val"
    train_set = BuildDataset(root=train_path)
    val_set = datasets.ImageFolder(root=val_path, transform=val_transforms)
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=8)
    device = torch.device("cuda:1")
    model = ResNet(Bottleneck, layers=[3, 8, 36, 3], num_classes=2)
    state_dict = torch.load("./resnet152-f82ba261.pth", map_location='cpu')
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    class_weight = np.sqrt(compute_class_weight('balanced', classes=np.unique(train_set.targets), y=train_set.targets))
    print('class weight: {}'.format(class_weight))
    focalLoss = FocalLoss(weight=torch.from_numpy(class_weight).to(device).float())
    optimizer = setting_optimizer(model)
    best_score = 0
    epochs = 100
    criterion = nn.TripletMarginLoss()
    import torch.optim.lr_scheduler as lr_scheduler

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    for epoch in range(0, epochs):
        model.train()
        train_acc = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}') as pbar:
            X = []
            X_paths = []
            epoch_acc = 0.0
            for batch in train_loader:
                anchors, anchor_paths, anchor_labels, positives, negatives = batch
                anchors = anchors.to(device)
                anchor_labels = anchor_labels.to(device)
                positives = positives.to(device)
                negatives = negatives.to(device)

                optimizer.zero_grad()

                y_pred, anchor_emb = model(anchors)
                _, positive_emb = model(positives)
                _, negative_emb = model(negatives)

                X.append(anchor_emb.cpu().detach().numpy())
                X_paths += anchor_paths

                loss1 = criterion(anchor_emb, positive_emb, negative_emb)
                loss2 = focalLoss(y_pred, anchor_labels)
                loss = loss1 + loss2
                loss.backward()
                optimizer.step()

                y_pred = torch.softmax(y_pred, dim=1).argmax(dim=1).cpu().flatten()
                y_true = anchor_labels.cpu().flatten()
                correct = (y_pred == y_true).float()
                epoch_acc += correct.mean().item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update()
        train_acc = epoch_acc / len(train_loader)
        print(f"Train acc score: {train_acc :.3f}")
        scheduler.step()
        if epoch > 30:
            X = np.concatenate(X, axis=0)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=50, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(X_pca)
            dist_matrix = cdist(X_pca, X_pca, metric="euclidean")
            for i, img_path in tqdm(enumerate(X_paths), total=len(X_paths)):
                anchor = "/p/" if "/p/" in img_path else "/n/"
                diff_class_mask = (labels != labels[i])
                diff_class_indices = np.where(diff_class_mask)[0]
                diff_class_indices = [idx for idx in diff_class_indices
                                      if anchor not in X_paths[idx]]
                nearest_diff_file = None
                if len(diff_class_indices) > 0:
                    diff_class_distances = dist_matrix[i, diff_class_indices]
                    nearest_idx = diff_class_indices[np.argmin(diff_class_distances)]
                    nearest_diff_file = X_paths[nearest_idx]
                if nearest_diff_file is not None:
                    nearest_diff_file = nearest_diff_file.replace(data_root, "")
                    img_path = img_path.replace(data_root, "")
                    rdb.set(img_path, nearest_diff_file)
            model.eval()
            with torch.inference_mode():
                val_f1 = F1Score(task="binary", average='macro')
                for x, y_true in val_loader:
                    logist, _ = model(x.to(device).float())
                    y_pred = torch.softmax(logist, dim=1).argmax(dim=1).cpu().flatten()
                    y_true = y_true.flatten()
                    val_f1.update(y_pred, y_true)
                score = val_f1.compute().item()
                if score > best_score:
                    best_score = score
                    torch.save(model.state_dict(), f'./{model_save_name}_{best_score:.3f}.pt')
                print(f'Validation score: {score:.3f}. best score: {best_score:.3f}')

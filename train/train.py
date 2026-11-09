# -*- coding: utf-8 -*-
"""训练 TrashNet 6 类分类模型并导出 ONNX（无 torchvision 依赖）。

CPU 也可跑：手写 ResNet18 + 手写 transform，加载 torchvision 的
ResNet18 ImageNet 预训练权重（resnet18-f37072fd.pth，raw state_dict）后微调。
10 个 epoch，按验证集 Macro-F1 保存最优权重。
"""
import os
import json
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
HERE = os.path.dirname(os.path.abspath(__file__))   # .../train
ROOT = os.path.dirname(HERE)                         # 项目根目录
SRC = os.path.join(ROOT, "data", "dataset-resized")  # 数据集（相对项目根）
OUT = ROOT
WEIGHTS = os.path.join(ROOT, "weights", "resnet18-f37072fd.pth")  # 预训练权重
ONNX_PATH = os.path.join(OUT, "trashnet.onnx")
PT_PATH = os.path.join(OUT, "trashnet.pt")
META_PATH = os.path.join(OUT, "classes.json")
BATCH = 32
EPOCHS = 10
LR = 1e-4
SEED = 42

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------- 手写 transform（替代 torchvision.transforms） ----------------
def resize_shortest(img, size=256):
    w, h = img.size
    if h < w:
        nw = max(int(round(w * size / h)), 1)
        return img.resize((nw, size), Image.Resampling.BILINEAR)
    nh = max(int(round(h * size / w)), 1)
    return img.resize((size, nh), Image.Resampling.BILINEAR)


def center_crop(img, size=224):
    w, h = img.size
    left = (w - size) // 2
    top = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def random_crop(img, size=224):
    w, h = img.size
    if w < size or h < size:
        img = img.resize((max(w, size), max(h, size)), Image.Resampling.BILINEAR)
        w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return img.crop((x, y, x + size, y + size))


def color_jitter(img, b=0.2, c=0.2, s=0.2):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(1 - b, 1 + b))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(1 - c, 1 + c))
    img = ImageEnhance.Color(img).enhance(random.uniform(1 - s, 1 + s))
    return img


def to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return torch.from_numpy(arr)


def normalize(t, mean, std):
    m = torch.tensor(mean, dtype=t.dtype).view(3, 1, 1)
    s = torch.tensor(std, dtype=t.dtype).view(3, 1, 1)
    return (t - m) / s


class TrainTransform:
    def __call__(self, img):
        img = resize_shortest(img, 256)
        img = random_crop(img, 224)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = color_jitter(img)
        return normalize(to_tensor(img), MEAN, STD)


class EvalTransform:
    def __call__(self, img):
        img = resize_shortest(img, 256)
        img = center_crop(img, 224)
        return normalize(to_tensor(img), MEAN, STD)


# ---------------- 手写 ResNet18（结构与 torchvision 完全一致） ----------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_planes, planes, blocks, stride):
        downsample = None
        if stride != 1 or in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(in_planes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1, None))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_pretrained(model):
    ckpt = torch.load(WEIGHTS, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    # 去掉 fc 权重（预训练是 1000 类，我们是 6 类），其余全部加载
    ckpt = {k: v for k, v in ckpt.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("pretrained loaded. missing fc:", [k for k in missing if k.startswith("fc.")],
          "unexpected:", len(unexpected))


class TrashDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.tf = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), label


def build_items():
    items = []
    for i, c in enumerate(CLASSES):
        d = os.path.join(SRC, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(d, f), i))
    random.Random(SEED).shuffle(items)
    return items


def macro_f1(ys, ps, n=6):
    f1s = []
    for c in range(n):
        tp = sum(1 for y, p in zip(ys, ps) if y == c and p == c)
        fp = sum(1 for y, p in zip(ys, ps) if y != c and p == c)
        fn = sum(1 for y, p in zip(ys, ps) if y == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s))


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            p = logits.argmax(1).cpu().tolist()
            ps.extend(p)
            ys.extend(y.tolist())
    top1 = sum(1 for y, p in zip(ys, ps) if y == p) / len(ys)
    return top1, macro_f1(ys, ps)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    items = build_items()
    n = len(items)
    n_tr = int(n * 0.7)
    n_va = int(n * 0.15)
    tr_items, va_items, te_items = items[:n_tr], items[n_tr:n_tr + n_va], items[n_tr + n_va:]
    print("total/train/val/test:", n, n_tr, n_va, n - n_tr - n_va)

    tr_ds = TrashDataset(tr_items, TrainTransform())
    va_ds = TrashDataset(va_items, EvalTransform())
    te_ds = TrashDataset(te_items, EvalTransform())
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    va_ld = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=2)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    model = ResNet18(num_classes=len(CLASSES))
    load_pretrained(model)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_f1 = -1.0
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total = 0
        correct = 0
        for x, y in tr_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total += y.numel()
            correct += (logits.argmax(1) == y).sum().item()
        v_top1, v_f1 = evaluate(model, va_ld, device)
        print(f"epoch {ep:02d}  train_acc={correct/total:.4f}  val_top1={v_top1:.4f}  val_f1={v_f1:.4f}  {time.time()-t0:.0f}s", flush=True)
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), PT_PATH)

    model.load_state_dict(torch.load(PT_PATH, map_location=device))
    te_top1, te_f1 = evaluate(model, te_ld, device)
    print(f"TEST top1={te_top1:.4f}  macro_f1={te_f1:.4f}")

    # 导出 ONNX
    model.eval()
    model.cpu()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, ONNX_PATH,
                      input_names=["input"], output_names=["logits"],
                      opset_version=13, dynamic_axes=None)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES, "mean": MEAN, "std": STD,
                   "input_size": 224, "test_top1": te_top1, "test_macro_f1": te_f1}, f, ensure_ascii=False, indent=2)
    print("saved", ONNX_PATH)
    print("saved", META_PATH)


if __name__ == "__main__":
    main()

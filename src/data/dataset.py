# src/data/dataset.py
import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class YOLODataset(Dataset):
    def __init__(self, root, split="train", img_size=640, transforms=None, class_names=None):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.paths = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.transforms = transforms
        self.img_size = img_size
        self.class_names = class_names or []
    def __len__(self): return len(self.paths)
    def load_labels(self, path, w, h):
        txt = os.path.join(self.lbl_dir, Path(path).stem + ".txt")
        boxes, labels = [], []
        if os.path.isfile(txt):
            with open(txt, "r") as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    toks = line.split()
                    if len(toks) < 5:
                        # linha inválida, menos de 5 campos
                        continue
                    # use apenas os 5 primeiros campos (classe cx cy w h)
                    t = toks[:5]
                    try:
                        c = int(float(t))
                        xc, yc, bw, bh = map(float, t[1:5])
                    except Exception:
                        # formato inválido nesta linha
                        continue
                    # converter de xywh normalizado ([0,1]) para xyxy em pixels
                    # se não estiver normalizado (valores > 1), assume que já está em pixels e usa direto
                    if max(xc, yc, bw, bh) <= 1.0:
                        x1 = (xc - bw / 2.0) * w
                        y1 = (yc - bh / 2.0) * h
                        x2 = (xc + bw / 2.0) * w
                        y2 = (yc + bh / 2.0) * h
                    else:
                        # se vierem em pixels (cx,cy,w,h), converta do mesmo modo
                        x1 = xc - bw / 2.0
                        y1 = yc - bh / 2.0
                        x2 = xc + bw / 2.0
                        y2 = yc + bh / 2.0
                    boxes.append([x1, y1, x2, y2])
                    labels.append(c)
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
    def __getitem__(self, i):
        p = self.paths[i]
        im = cv2.imread(p)
        assert im is not None, f"Image not found: {p}"
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h0, w0 = im.shape[:2]
        boxes, labels = self.load_labels(p, w0, h0)
        sample = {"image": im, "boxes": boxes, "labels": labels}
        if self.transforms:
            sample = self.transforms(sample, self.img_size)
        img = sample["image"]
        boxes = sample["boxes"]
        labels = sample["labels"]
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "img_id": torch.tensor([i])}
        img = torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        return img, target

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), list(targets)

# src/data/dataset.py
import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

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
        txt = os.path.join(self.lbl_dir, os.path.splitext(os.path.basename(path)) + ".txt")
        boxes, labels = [], []
        if os.path.isfile(txt):
            with open(txt, "r") as f:
                for line in f.read().strip().splitlines():
                    c, xc, yc, bw, bh = map(float, line.split())
                    labels.append(int(c))
                    x1 = (xc - bw/2) * w; y1 = (yc - bh/2) * h
                    x2 = (xc + bw/2) * w; y2 = (yc + bh/2) * h
                    boxes.append([x1, y1, x2, y2])
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

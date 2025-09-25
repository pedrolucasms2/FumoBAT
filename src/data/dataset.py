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

    def __len__(self):
        return len(self.paths)

    def load_labels(self, path, w, h):
        txt_path = os.path.join(self.lbl_dir, Path(path).stem + ".txt")
        boxes, labels = [], []
        if os.path.isfile(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # Formato YOLO: class_id, x_center, y_center, width, height
                            c = int(parts[0])
                            xc, yc, bw, bh = map(float, parts[1:5])
                            
                            # Converter de YOLO normalizado para [x1, y1, x2, y2] em pixels
                            x1 = (xc - bw / 2) * w
                            y1 = (yc - bh / 2) * h
                            x2 = (xc + bw / 2) * w
                            y2 = (yc + bh / 2) * h
                            boxes.append([x1, y1, x2, y2])
                            labels.append(c)
                        except ValueError:
                            continue # Ignora linhas mal formatadas
        
        # O formato para albumentations deve ser [x_min, y_min, x_max, y_max]
        # e as labels precisam estar juntas
        # Para YOLO, o formato é [x_center, y_center, width, height]
        yolo_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            xc = ((x1 + x2) / 2) / w
            yc = ((y1 + y2) / 2) / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            yolo_boxes.append([xc, yc, box_w, box_h])

        if not yolo_boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        
        return np.array(yolo_boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __getitem__(self, i):
        path = self.paths[i]
        image = cv2.imread(path)
        assert image is not None, f"Image not found: {path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h0, w0 = image.shape[:2]
        
        boxes, labels = self.load_labels(path, w0, h0)
        
        sample = {"image": image, "boxes": boxes, "labels": labels}
        
        if self.transforms:
            transformed = self.transforms(sample)
            image = transformed["image"]
            boxes = transformed["boxes"]
            labels = transformed["labels"]

        # A transformação já retorna tensores, então não precisamos converter aqui
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "img_id": torch.tensor([i])
        }
        
        return image, target

def collate_fn(batch):
    imgs, targets = zip(*batch)
    # As imagens já são tensores, então apenas empilhamos
    return torch.stack(imgs, 0), list(targets)
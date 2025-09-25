# src/data/dataset.py
import cv2
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any

def clamp_bbox(bbox):
    """Clamps bounding box coordinates to the range [0.0, 1.0]."""
    x_center, y_center, w, h = bbox
    # Convert to x_min, y_min, x_max, y_max
    x_min = x_center - w / 2
    y_min = y_center - h / 2
    x_max = x_center + w / 2
    y_max = y_center + h / 2
    
    # Clamp the coordinates
    x_min = np.clip(x_min, 0.0, 1.0)
    y_min = np.clip(y_min, 0.0, 1.0)
    x_max = np.clip(x_max, 0.0, 1.0)
    y_max = np.clip(y_max, 0.0, 1.0)
    
    # Convert back to x_center, y_center, w, h
    new_w = x_max - x_min
    new_h = y_max - y_min
    new_x_center = x_min + new_w / 2
    new_y_center = y_min + new_h / 2
    
    return [new_x_center, new_y_center, new_w, new_h]

class SmallObjectDataset(Dataset):
    """Dataset for small object detection."""

    def __init__(self, img_dir: str, label_dir: str, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.label_files = {os.path.splitext(f)[0]: f for f in os.listdir(label_dir) if f.endswith('.txt')}

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_name = os.path.splitext(img_name)[0]
        boxes = []
        labels = []
        
        label_file_name = self.label_files.get(label_name)
        if label_file_name:
            label_path = os.path.join(self.label_dir, label_file_name)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            coords = list(map(float, parts[1:5]))
                            # --- CORRECTION APPLIED HERE ---
                            clamped_coords = clamp_bbox(coords)
                            boxes.append(clamped_coords)
                            labels.append(class_id)
        
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels, dtype=np.int64)

        sample = {
            "image": image,
            "bboxes": boxes,
            "class_labels": labels
        }
        
        transformed = self.transforms(sample)
        
        #print(f"[DEBUG Dataset] Sample keys before transforms: {list(sample.keys())}")
        #print(f"[DEBUG Dataset] Image shape: {image.shape if hasattr(image, 'shape') else type(image)}")
        #print(f"[DEBUG Dataset] Boxes type: {type(boxes)}, length: {len(boxes) if hasattr(boxes, '__len__') else 'no len'}")

        if transformed is None:
            print(f"[ERROR] Transforms returned None for idx {idx}")
            return {
                'image': torch.zeros((3, self.img_size, self.img_size)),
                'boxes': torch.zeros((0, 4)),
                'labels': torch.zeros(0)
            }   
    
        return {
            'image': transformed['image'],
            'boxes': torch.tensor(transformed['boxes'], dtype=torch.float32),
            'labels': torch.tensor(transformed['labels'], dtype=torch.long)
        }

def collate_fn(batch):
    """Custom collate function."""
    images = [item['image'] for item in batch]
    boxes = [torch.tensor(item['boxes']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    
    images = torch.stack(images, 0)
    
    targets = []
    for i in range(len(boxes)):
        target = {}
        target["boxes"] = boxes[i]
        target["labels"] = labels[i]
        targets.append(target)

    return images, targets
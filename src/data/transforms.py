# src/data/transforms.py
import cv2
import numpy as np
import random
import torch
from typing import Tuple, List
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SmallObjectMixUp:
    """MixUp augmentation specifically for small objects"""
    def __init__(self, alpha=0.2, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, image1, boxes1, image2, boxes2):
        if random.random() > self.prob:
            return image1, boxes1
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_image = lam * image1 + (1 - lam) * image2
        mixed_boxes = []
        for box in boxes1:
            new_box = box.copy()
            if len(new_box) > 4: new_box[4] *= lam
            mixed_boxes.append(new_box)
        for box in boxes2:
            new_box = box.copy()
            if len(new_box) > 4: new_box[4] *= (1 - lam)
            mixed_boxes.append(new_box)
        return mixed_image.astype(np.uint8), mixed_boxes

class SmallObjectMosaic:
    """Mosaic augmentation optimized for small objects"""
    def __init__(self, size=640, prob=0.8):
        self.size = size
        self.prob = prob

    def __call__(self, images, all_boxes):
        if random.random() > self.prob or len(images) < 4:
            return images[0], all_boxes[0]
        mosaic_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        mosaic_boxes = []
        h, w = self.size // 2, self.size // 2
        positions = [(0, 0), (w, 0), (0, h), (w, h)]
        for i in range(4):
            img, boxes = images[i % len(images)], all_boxes[i % len(all_boxes)]
            img_resized = cv2.resize(img, (w, h))
            x_offset, y_offset = positions[i]
            mosaic_img[y_offset:y_offset+h, x_offset:x_offset+w] = img_resized
            img_h, img_w = img.shape[:2]
            scale_x, scale_y = w / img_w, h / img_h
            for box in boxes:
                if len(box) >= 4:
                    x_center, y_center = box[0] * img_w, box[1] * img_h
                    box_w, box_h = box[2] * img_w, box[3] * img_h
                    new_x = (x_center * scale_x + x_offset) / self.size
                    new_y = (y_center * scale_y + y_offset) / self.size
                    new_w, new_h = (box_w * scale_x) / self.size, (box_h * scale_y) / self.size
                    if new_w > 0.01 and new_h > 0.01:
                        new_box = [new_x, new_y, new_w, new_h]
                        if len(box) > 4: new_box.extend(box[4:])
                        mosaic_boxes.append(new_box)
        return mosaic_img, mosaic_boxes

class SmallObjectCopyPaste:
    """Copy-paste augmentation for small objects"""
    def __init__(self, prob=0.3, max_objects=5):
        self.prob = prob
        self.max_objects = max_objects

    def __call__(self, image, boxes, object_patches=None):
        if random.random() > self.prob or not object_patches:
            return image, boxes
        img_copy, new_boxes = image.copy(), boxes.copy()
        h, w = image.shape[:2]
        num_paste = random.randint(1, min(self.max_objects, len(object_patches)))
        for _ in range(num_paste):
            patch_data = random.choice(object_patches)
            patch_img, patch_box = patch_data['image'], patch_data['box']
            scale = random.uniform(0.5, 1.5)
            patch_h, patch_w = patch_img.shape[:2]
            new_h, new_w = int(patch_h * scale), int(patch_w * scale)
            if new_h > 0 and new_w > 0:
                patch_resized = cv2.resize(patch_img, (new_w, new_h))
                margin = 20
                if w > new_w + 2*margin and h > new_h + 2*margin:
                    x, y = random.randint(margin, w - new_w - margin), random.randint(margin, h - new_h - margin)
                    try:
                        img_copy[y:y+new_h, x:x+new_w] = patch_resized
                        center_x, center_y = (x + new_w/2) / w, (y + new_h/2) / h
                        box_w, box_h = new_w / w, new_h / h
                        new_box = [center_x, center_y, box_w, box_h]
                        if len(patch_box) > 4: new_box.extend(patch_box[4:])
                        new_boxes.append(new_box)
                    except Exception:
                        continue
        return img_copy, new_boxes

def get_small_object_transforms(img_size=640, training=True):
    """Get optimized transforms for small object detection"""
    bbox_params = A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1)

    if training:
        return A.Compose([
            # --- THIS IS THE CORRECTED LINE ---
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            A.Transpose(p=0.1),
            A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], p=0.7),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                A.ChannelShuffle(p=0.1),
            ], p=0.5),
            A.OneOf([A.GaussNoise(p=0.3), A.ISONoise(intensity=(0.1, 0.5), p=0.2)], p=0.3),
            A.OneOf([A.MotionBlur(blur_limit=3, p=0.2), A.MedianBlur(blur_limit=3, p=0.1), A.Blur(blur_limit=3, p=0.1)], p=0.2),
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=bbox_params)
    else: # Validation/Testing
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], bbox_params=bbox_params)
class MultiScaleTransforms:
    """Multi-scale training class, which can be pickled for multiprocessing."""
    def __init__(self, scales=[480, 512, 544, 576, 608, 640, 672, 704, 736], target_size=640):
        self.scales = scales
        self.target_size = target_size

    def __call__(self, image, bboxes, class_labels):
        scale = random.choice(self.scales)
        h, w = image.shape[:2]
        new_h, new_w = (scale, int(scale * w / h)) if h > w else (int(scale * h / w), scale)

        transforms = A.Compose([
            A.Resize(height=new_h, width=new_w),
            A.PadIfNeeded(min_height=scale, min_width=scale, border_mode=cv2.BORDER_CONSTANT, position='center'),
            A.Resize(height=self.target_size, width=self.target_size),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        return transforms(image=image, bboxes=bboxes, class_labels=class_labels)
class SmallObjectAugmentationPipeline:
    """
    Pipeline de augmentation AGRESSIVO para combater overfitting em datasets pequenos.
    """
    def __init__(self, img_size=1024, training=True):
        self.img_size = img_size
        self.training = training
        
        if training:
            # --- Pipeline de Treinamento Agressivo Corrigido ---
            self.transforms = A.Compose([
                # Transformações espaciais
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.75, 1.0), p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT),

                # Transformações de cor e brilho
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
                A.ToGray(p=0.1),
                A.RandomGamma(p=0.2),

                # Adição de "ruído"
                A.OneOf([
                    A.GaussNoise(p=0.5),
                    A.ISONoise(p=0.5),
                ], p=0.3),
                
                A.OneOf([
                    A.MotionBlur(p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.Blur(blur_limit=3, p=0.5),
                ], p=0.4),
                
                # 
                # 🔧 GARANTIA DE TAMANHO ADICIONADA AQUI 🔧
                # Esta linha força todas as imagens a terem o tamanho final correto, 
                # mesmo que o RandomResizedCrop seja pulado.
                A.Resize(height=img_size, width=img_size),

                # Normalizar e converter para Tensor (sempre no final)
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.1))
        else:
            # --- Pipeline de Validação (sempre esteve correto) ---
            self.transforms = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


    def __call__(self, sample=None, **kwargs):
        if sample is not None:
            image = sample['image']
            boxes = sample['bboxes'] 
            labels = sample['class_labels']
        else:
            image = kwargs.get("image")
            boxes = kwargs.get("boxes", [])
            labels = kwargs.get("labels", [])
        
        class_labels = labels.tolist() if isinstance(labels, np.ndarray) else list(labels)
        
        try:
            # Aplica o pipeline de transformações completo
            result = self.transforms(image=image, bboxes=boxes, class_labels=class_labels)

            # Garante que os tensores retornados tenham o tipo de dado correto
            return {
                "image": result["image"], 
                "boxes": torch.as_tensor(result["bboxes"], dtype=torch.float32), 
                "labels": torch.as_tensor(result["class_labels"], dtype=torch.long)
            }
        except Exception as e:
            # 
            # 🔧 CORREÇÃO APLICADA AQUI 🔧
            # Se a augmentation falhar (ex: remover todos os bboxes), 
            # ainda assim redimensionamos a imagem original antes de retorná-la.
            # 
            print(f"Aviso: Augmentation falhou para uma imagem ({e}). Usando imagem original redimensionada.")
            
            # Cria um pipeline de fallback simples
            fallback_transforms = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
            # Aplica o fallback e retorna os dados originais, mas com o tamanho correto
            result = fallback_transforms(image=image, bboxes=boxes, class_labels=class_labels)

            return {
                "image": result["image"], 
                "boxes": torch.as_tensor(result["bboxes"], dtype=torch.float32), 
                "labels": torch.as_tensor(result["class_labels"], dtype=torch.long)
            }
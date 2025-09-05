# src/data/small_object_transforms.py
import cv2
import numpy as np
import random
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
            
        # Generate lambda for mixing
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # Combine boxes (keeping all boxes but with confidence scaling)
        mixed_boxes = []
        
        # Add boxes from first image with lambda weight
        for box in boxes1:
            new_box = box.copy()
            if len(new_box) > 4:  # If confidence exists
                new_box[4] *= lam
            mixed_boxes.append(new_box)
        
        # Add boxes from second image with (1-lambda) weight
        for box in boxes2:
            new_box = box.copy()
            if len(new_box) > 4:  # If confidence exists
                new_box[4] *= (1 - lam)
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
        
        # Create mosaic canvas
        mosaic_img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        mosaic_boxes = []
        
        # Divide into 4 quadrants
        h, w = self.size // 2, self.size // 2
        
        positions = [(0, 0), (w, 0), (0, h), (w, h)]
        
        for i in range(4):
            img = images[i % len(images)]
            boxes = all_boxes[i % len(all_boxes)]
            
            # Resize image to fit quadrant
            img_resized = cv2.resize(img, (w, h))
            
            # Place in mosaic
            x_offset, y_offset = positions[i]
            mosaic_img[y_offset:y_offset+h, x_offset:x_offset+w] = img_resized
            
            # Adjust box coordinates
            img_h, img_w = img.shape[:2]
            scale_x, scale_y = w / img_w, h / img_h
            
            for box in boxes:
                if len(box) >= 4:
                    # Convert relative to absolute
                    x_center = box[0] * img_w
                    y_center = box[1] * img_h
                    box_w = box[2] * img_w
                    box_h = box[3] * img_h
                    
                    # Scale and translate
                    new_x = (x_center * scale_x + x_offset) / self.size
                    new_y = (y_center * scale_y + y_offset) / self.size
                    new_w = (box_w * scale_x) / self.size
                    new_h = (box_h * scale_y) / self.size
                    
                    # Keep box if it's still visible and not too small
                    if new_w > 0.01 and new_h > 0.01:
                        new_box = [new_x, new_y, new_w, new_h]
                        if len(box) > 4:
                            new_box.extend(box[4:])
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
        
        img_copy = image.copy()
        new_boxes = boxes.copy()
        
        h, w = image.shape[:2]
        
        # Randomly select and paste objects
        num_paste = random.randint(1, min(self.max_objects, len(object_patches)))
        
        for _ in range(num_paste):
            patch_data = random.choice(object_patches)
            patch_img = patch_data['image']
            patch_box = patch_data['box']
            
            # Random scale (0.5x to 1.5x)
            scale = random.uniform(0.5, 1.5)
            patch_h, patch_w = patch_img.shape[:2]
            new_h, new_w = int(patch_h * scale), int(patch_w * scale)
            
            if new_h > 0 and new_w > 0:
                patch_resized = cv2.resize(patch_img, (new_w, new_h))
                
                # Random position (avoid edges)
                margin = 20
                if w > new_w + 2*margin and h > new_h + 2*margin:
                    x = random.randint(margin, w - new_w - margin)
                    y = random.randint(margin, h - new_h - margin)
                    
                    # Paste with blending
                    try:
                        img_copy[y:y+new_h, x:x+new_w] = patch_resized
                        
                        # Add new box
                        center_x = (x + new_w/2) / w
                        center_y = (y + new_h/2) / h
                        box_w = new_w / w
                        box_h = new_h / h
                        
                        new_box = [center_x, center_y, box_w, box_h]
                        if len(patch_box) > 4:
                            new_box.extend(patch_box[4:])
                        new_boxes.append(new_box)
                    except:
                        continue
        
        return img_copy, new_boxes

def get_small_object_transforms(img_size=640, training=True):
    """Get optimized transforms for small object detection"""
    
    if training:
        return A.Compose([
            # Geometric transforms - careful with small objects
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),  # Less aggressive cropping
                ratio=(0.9, 1.1),  # Maintain aspect ratio
                p=0.5
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),  # Less common for pest detection
            A.RandomRotate90(p=0.2),
            A.Transpose(p=0.1),
            
            # Small rotation to avoid losing small objects
            A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
            
            # Photometric transforms - important for small object visibility
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, 
                    contrast_limit=0.3, 
                    p=0.8
                ),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ], p=0.7),
            
            # Color augmentations
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=20, 
                    sat_shift_limit=30, 
                    val_shift_limit=20, 
                    p=0.5
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                A.ChannelShuffle(p=0.1),
            ], p=0.5),
            
            # Noise and blur - simulate real conditions
            A.OneOf([
                A.GaussNoise(noise_scale_factor=0.1, p=0.3),
                A.ISONoise(intensity=(0.1, 0.5), p=0.2),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            
            # Weather effects
            A.OneOf([
                A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
                A.RandomRain(
                    slant_range=(-10, 10),
                    drop_length=1, drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=1, p=0.1
                ),
            ], p=0.1),
            
            # Mandatory resize to ensure consistent batch sizes
            A.Resize(height=img_size, width=img_size),
            
            # Final normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.1  # Keep boxes with at least 10% visibility
        ))
    
    else:  # Validation/Testing
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

def multi_scale_training_transforms(scales=[480, 512, 544, 576, 608, 640, 672, 704, 736], target_size=640):
    """Multi-scale training for small object robustness"""
    def transform(image, boxes):
        # Random scale selection for training variety
        scale = random.choice(scales)
        
        # Resize maintaining aspect ratio
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = scale, int(scale * w / h)
        else:
            new_h, new_w = int(scale * h / w), scale
        
        # Apply transforms with final resize to target_size for consistent batching
        transforms = A.Compose([
            A.Resize(height=new_h, width=new_w),
            A.PadIfNeeded(
                min_height=scale, min_width=scale,
                border_mode=cv2.BORDER_CONSTANT,
                position='center'
            ),
            # Final resize to ensure consistent batch size
            A.Resize(height=target_size, width=target_size),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
        
        return transforms(image=image, bboxes=boxes, class_labels=[0]*len(boxes))
    
    return transform

class SmallObjectAugmentationPipeline:
    """Complete augmentation pipeline for small object detection"""
    
    def __init__(self, img_size=640, training=True):
        self.img_size = img_size
        self.training = training
        
        # Initialize specialized augmentations
        self.mixup = SmallObjectMixUp(alpha=0.2, prob=0.2)
        self.mosaic = SmallObjectMosaic(size=img_size, prob=0.3)
        self.copy_paste = SmallObjectCopyPaste(prob=0.2)
        
        # Standard transforms
        self.transforms = get_small_object_transforms(img_size, training)
        
        # Multi-scale transforms
        self.multi_scale = multi_scale_training_transforms(target_size=img_size)
        
    def __call__(self, sample, img_size=None):
        """Apply augmentation pipeline"""
        
        # Handle both dictionary input (from dataset) and separate parameters
        if isinstance(sample, dict):
            image = sample["image"]
            boxes = sample["boxes"]
            labels = sample.get("labels", [])
            additional_data = None
        else:
            # Legacy support for separate parameters
            image = sample
            boxes = img_size if img_size is not None else []
            labels = []
            additional_data = None
        
        if not self.training:
            result = self.transforms(image=image, bboxes=boxes, class_labels=[0]*len(boxes))
            return {
                "image": result["image"],
                "boxes": result["bboxes"],
                "labels": labels
            }
        
        # Apply specialized augmentations with probability
        if additional_data and random.random() < 0.3:
            if 'mosaic_data' in additional_data and random.random() < 0.3:
                image, boxes = self.mosaic(
                    additional_data['mosaic_data']['images'],
                    additional_data['mosaic_data']['boxes']
                )
            
            elif 'mixup_data' in additional_data and random.random() < 0.2:
                mixup_data = additional_data['mixup_data']
                image, boxes = self.mixup(
                    image, boxes,
                    mixup_data['image'], mixup_data['boxes']
                )
            
            elif 'object_patches' in additional_data and random.random() < 0.2:
                image, boxes = self.copy_paste(
                    image, boxes, 
                    additional_data['object_patches']
                )
        
        # Apply multi-scale training
        if random.random() < 0.5:
            result = self.multi_scale(image, boxes)
            image, boxes = result['image'], result['bboxes']
        
        # Apply standard transforms
        try:
            result = self.transforms(
                image=image, 
                bboxes=boxes, 
                class_labels=[0]*len(boxes)
            )
            return {
                "image": result["image"],
                "boxes": result["bboxes"],
                "labels": labels
            }
        except Exception as e:
            # Fallback to basic transforms if augmentation fails
            basic_transforms = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
            result = basic_transforms(
                image=image, 
                bboxes=boxes, 
                class_labels=[0]*len(boxes)
            )
            return {
                "image": result["image"],
                "boxes": result["bboxes"],
                "labels": labels
            }

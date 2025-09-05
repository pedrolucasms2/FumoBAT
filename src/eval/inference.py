#!/usr/bin/env python3
"""
Single file inference for SmallObjectYOLO
Usage: python inference.py <image_path>
Returns: Bounding boxes with confidence scores
"""

import torch
import cv2
import numpy as np
import yaml
import os
import sys
import argparse
from pathlib import Path

try:
    import torchvision.ops
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
    print("Warning: torchvision not available, NMS will be skipped")

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels from src/eval/
sys.path.append(project_root)

from src.models.model import SmallObjectYOLO


def load_model(model_path, config_path, device):
    """Load trained model"""
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    class_names = list(cfg.get('names', {}).values())
    nc = len(class_names)
    
    # Load model
    model = SmallObjectYOLO(nc=nc, ch=[64, 128, 256, 384])
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded model (epoch: {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded model weights")
    
    model = model.to(device)
    model.eval()
    
    return model, class_names


def preprocess_image(image_path, img_size=640):
    """Load and preprocess image"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img.shape[:2]
    
    # Resize image
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    return img_tensor, (h_orig, w_orig), img


def postprocess_detections(predictions, orig_size, img_size=640, conf_threshold=0.25, nms_threshold=0.45):
    """Convert model predictions to bounding boxes"""
    h_orig, w_orig = orig_size
    scale_x = w_orig / img_size
    scale_y = h_orig / img_size
    
    detections = []
    
    def extract_tensors(obj):
        """Recursively extract tensors from nested structure"""
        if torch.is_tensor(obj):
            return [obj]
        elif isinstance(obj, (list, tuple)):
            tensors = []
            for item in obj:
                tensors.extend(extract_tensors(item))
            return tensors
        else:
            return []
    
    # Extract all tensors from the predictions
    pred_tensors = extract_tensors(predictions)
    
    if not pred_tensors:
        return detections
    
    # Process each tensor separately instead of concatenating
    for tensor in pred_tensors:
        if tensor is None or tensor.numel() == 0:
            continue
            
        # Reshape to [N, features] if needed
        if tensor.dim() > 2:
            tensor = tensor.view(-1, tensor.shape[-1])
        
        if len(tensor) == 0:
            continue
        
        # Apply confidence threshold
        if tensor.shape[1] >= 5:
            conf_mask = tensor[:, 4] >= conf_threshold
            tensor = tensor[conf_mask]
        
        if len(tensor) == 0:
            continue
        
        # Convert predictions to boxes
        boxes = tensor[:, :4].clone()
        scores = tensor[:, 4]
        
        # Handle class predictions
        if tensor.shape[1] > 5:
            classes = tensor[:, 5:].argmax(dim=1)
        else:
            classes = torch.zeros(len(tensor), dtype=torch.long)
        
        # Convert from normalized to pixel coordinates
        # Assume format is [x_center, y_center, width, height] normalized
        x_center = boxes[:, 0] * img_size
        y_center = boxes[:, 1] * img_size
        width = boxes[:, 2] * img_size
        height = boxes[:, 3] * img_size
        
        boxes[:, 0] = (x_center - width / 2) * scale_x   # x1
        boxes[:, 1] = (y_center - height / 2) * scale_y # y1
        boxes[:, 2] = (x_center + width / 2) * scale_x  # x2
        boxes[:, 3] = (y_center + height / 2) * scale_y # y2
        
        # Add detections from this tensor
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            score = scores[i].item()
            cls = classes[i].item()
            
            # Ensure coordinates are valid
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_orig, x2), min(h_orig, y2)
            
            if x2 > x1 and y2 > y1:  # Valid box
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': score,
                    'class': cls
                })
    
    # Apply NMS to all detections if torchvision is available
    if HAS_TORCHVISION and len(detections) > 1:
        boxes = torch.tensor([det['bbox'] for det in detections])
        scores = torch.tensor([det['confidence'] for det in detections])
        
        keep = torchvision.ops.nms(boxes, scores, nms_threshold)
        detections = [detections[i] for i in keep.tolist()]
    
    return detections


def draw_detections(image, detections, class_names):
    """Draw bounding boxes on image"""
    img_with_boxes = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        confidence = det['confidence']
        class_id = det['class']
        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_with_boxes, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(img_with_boxes, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return img_with_boxes


def run_inference(image_path, model_path=None, config_path=None, output_path=None, 
                 conf_threshold=0.25, show_image=False):
    """Run inference on single image"""
    
    # Default paths
    if model_path is None:
        model_path = "runs/improved_train/improved_best_model.pt"
    if config_path is None:
        config_path = "configs/data.yaml"
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, class_names = load_model(model_path, config_path, device)
    
    # Preprocess image
    print("Processing image...")
    img_tensor, orig_size, orig_image = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(img_tensor)
    
    # Postprocess
    print("Processing detections...")
    detections = postprocess_detections(predictions, orig_size, conf_threshold=conf_threshold)
    
    # Print results
    print("\nDETECTION RESULTS:")
    print("=" * 50)
    
    if len(detections) == 0:
        print("No objects detected")
        return []
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        cls = det['class']
        class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        
        print(f"Detection {i+1}:")
        print(f"  Class: {class_name}")
        print(f"  Confidence: {conf:.3f}")
        print(f"  Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"  Size: {x2-x1:.1f} x {y2-y1:.1f}")
        print()
    
    print(f"Total detections: {len(detections)}")
    
    # Save output image if requested
    if output_path or show_image:
        img_with_boxes = draw_detections(orig_image, detections, class_names)
        
        if output_path:
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img_bgr)
            print(f"Output saved to: {output_path}")
        
        if show_image:
            cv2.imshow('Detections', cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return detections


def main():
    parser = argparse.ArgumentParser(description='SmallObjectYOLO Inference')
    parser.add_argument('image', help='Path to input image')
    parser.add_argument('--model', default='runs/improved_train/improved_best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--config', default='configs/data.yaml',
                       help='Path to data config')
    parser.add_argument('--output', help='Path to save output image with detections')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--show', action='store_true',
                       help='Show image with detections')
    
    args = parser.parse_args()
    
    try:
        detections = run_inference(
            image_path=args.image,
            model_path=args.model,
            config_path=args.config,
            output_path=args.output,
            conf_threshold=args.conf,
            show_image=args.show
        )
        
        return detections
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return []


if __name__ == "__main__":
    main()

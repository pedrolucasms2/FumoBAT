import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, eps=1e-7):
    """Calculate IoU variants including EIoU."""
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    
    if CIoU or DIoU or GIoU or EIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        
        if EIoU:
            # EIoU Implementation from the paper
            # Distance loss
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            c2 = cw ** 2 + ch ** 2 + eps
            
            # Width and height difference
            rho_w2 = ((w2 - w1) ** 2) / (cw ** 2 + eps)
            rho_h2 = ((h2 - h1) ** 2) / (ch ** 2 + eps)
            
            return iou - (rho2 / c2 + rho_w2 + rho_h2)  # EIoU
            
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
            
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
        
    return iou

class FocalEIoULoss(nn.Module):
    """
    Focal-EIoU Loss implementation based on the paper:
    "Focal and Efficient IOU Loss for Accurate Bounding Box Regression"
    """
    def __init__(self, alpha=1.0, gamma=0.5, eps=1e-7):
        super(FocalEIoULoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred: [N, 4] predicted boxes (x1, y1, x2, y2)
            target: [N, 4] target boxes (x1, y1, x2, y2)
        """
        # Calculate EIoU
        eiou = bbox_iou(pred, target, xywh=False, EIoU=True, eps=self.eps)
        
        # EIoU Loss
        eiou_loss = 1 - eiou
        
        # Focal mechanism: IoU^gamma * EIoU_loss
        iou = bbox_iou(pred, target, xywh=False, eps=self.eps)
        focal_weight = iou.pow(self.gamma)
        
        # Focal-EIoU Loss
        focal_eiou_loss = focal_weight * eiou_loss
        
        return focal_eiou_loss.mean()

class SmallObjectYOLOLoss(nn.Module):
    """
    Simplified YOLO Loss for Small Object Detection with Focal-EIoU
    """
    def __init__(self, nc=1, device='cpu', focal_gamma=0.5):
        super(SmallObjectYOLOLoss, self).__init__()
        self.nc = nc
        self.device = device
        self.focal_gamma = focal_gamma
        
        # Loss functions
        self.bce_cls = nn.BCEWithLogitsLoss()
        self.bce_obj = nn.BCEWithLogitsLoss()
        self.focal_eiou = FocalEIoULoss(gamma=focal_gamma)
        
        # Loss weights
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
    
    def forward(self, predictions, targets):
        """
        Simplified YOLO Loss with Focal-EIoU
        Args:
            predictions: list of model outputs
            targets: ground truth targets
        """
        device = predictions[0].device
        
        # Initialize losses
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        total_targets = 0
        
        for pred in predictions:
            batch_size, _, height, width = pred.shape
            
            # Simplified approach: use dummy targets for now
            # In a real implementation, you would match predictions to targets
            
            # Box loss (using Focal-EIoU on a subset of predictions)
            if total_targets == 0:  # Dummy implementation
                # Create some dummy box regression targets
                dummy_pred_boxes = torch.rand(10, 4, device=device)
                dummy_target_boxes = torch.rand(10, 4, device=device)
                box_loss = box_loss + self.focal_eiou(dummy_pred_boxes, dummy_target_boxes) * 0.1
            
            # Object loss (simplified)
            obj_targets = torch.zeros(batch_size, height, width, device=device)
            if pred.size(1) >= 5:  # Has objectness prediction
                obj_pred = pred[:, 4, :, :]  # Objectness channel
                obj_loss = obj_loss + self.bce_obj(obj_pred, obj_targets) * 0.1
            
            # Class loss (simplified)
            if self.nc > 1 and pred.size(1) > 5:
                cls_targets = torch.zeros(batch_size, self.nc, height, width, device=device)
                cls_pred = pred[:, 5:5+self.nc, :, :]
                cls_loss = cls_loss + self.bce_cls(cls_pred, cls_targets) * 0.1
            
            total_targets += batch_size * height * width
        
        # Combine losses
        total_loss = (self.lambda_box * box_loss + 
                     self.lambda_obj * obj_loss + 
                     self.lambda_cls * cls_loss)
        
        return total_loss, torch.stack([box_loss, obj_loss, cls_loss, total_loss]).detach()
    

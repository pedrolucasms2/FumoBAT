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
    Loss YOLO REAL com Focal-EIoU para Small Objects
    """
    def __init__(self, nc=1, device='cpu', focal_gamma=0.5):
        super(SmallObjectYOLOLoss, self).__init__()
        self.nc = nc
        self.device = device
        self.focal_gamma = focal_gamma
        
        # Loss functions
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        self.focal_eiou = FocalEIoULoss(gamma=focal_gamma)
        
        # Loss weights otimizados para objetos pequenos
        self.lambda_box = 0.1    # Aumentado para objetos pequenos
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
        
        # Anchors para diferentes escalas (P3, P4, P5)
        self.anchors = [
            torch.tensor([[10, 13], [16, 30], [33, 23]], device=device, dtype=torch.float32) / 8,    # P3
            torch.tensor([[30, 61], [62, 45], [59, 119]], device=device, dtype=torch.float32) / 16,  # P4  
            torch.tensor([[116, 90], [156, 198], [373, 326]], device=device, dtype=torch.float32) / 32 # P5
        ]
        
    def build_targets(self, predictions, targets):
        """Constrói targets para cada escala de predição"""
        tcls, tbox, indices, anchors = [], [], [], []
        
        for i, pred in enumerate(predictions):
            B, _, H, W = pred.shape
            anchors_i = self.anchors[i].to(pred.device)
            
            # Initialize
            tcls_i = torch.zeros(B, 3, H, W, device=pred.device, dtype=torch.long)
            tbox_i = torch.zeros(B, 3, H, W, 4, device=pred.device)
            indices_i = (torch.zeros(0, device=pred.device, dtype=torch.long),
                        torch.zeros(0, device=pred.device, dtype=torch.long),
                        torch.zeros(0, device=pred.device, dtype=torch.long),
                        torch.zeros(0, device=pred.device, dtype=torch.long))
            
            if len(targets) > 0:
                # Process targets for this scale
                for b, target in enumerate(targets):
                    if 'boxes' in target and len(target['boxes']) > 0:
                        boxes = target['boxes']  # [N, 4] in format [x_center, y_center, w, h]
                        labels = target['labels'] if 'labels' in target else torch.zeros(len(boxes), dtype=torch.long)
                        
                        # Scale boxes to grid
                        boxes_scaled = boxes.clone()
                        boxes_scaled[:, [0, 2]] *= W  # x, w
                        boxes_scaled[:, [1, 3]] *= H  # y, h
                        
                        # Find grid cells
                        gxy = boxes_scaled[:, :2]  # grid xy
                        gwh = boxes_scaled[:, 2:4]  # grid wh
                        
                        # Grid indices
                        gi = gxy[:, 0].long().clamp(0, W-1)  # grid x
                        gj = gxy[:, 1].long().clamp(0, H-1)  # grid y
                        
                        # Select anchors based on IoU
                        anchor_ious = []
                        for a_idx, anchor in enumerate(anchors_i):
                            # Calculate IoU between target and anchor
                            anchor_box = torch.cat([torch.zeros(2), anchor])  # [0, 0, w, h]
                            target_box = torch.cat([torch.zeros(2), gwh[0]])   # [0, 0, w, h] 
                            iou = bbox_iou(anchor_box.unsqueeze(0), target_box.unsqueeze(0), xywh=True).item()
                            anchor_ious.append(iou)
                        
                        # Best anchor for each target
                        best_anchor = torch.tensor(anchor_ious).argmax()
                        
                        # Assign targets
                        if len(boxes) > 0:
                            # Take first target for simplicity
                            tcls_i[b, best_anchor, gj[0], gi[0]] = labels[0]
                            tbox_i[b, best_anchor, gj[0], gi[0]] = boxes[0]
                            
                            # Store indices
                            indices_i = (torch.tensor([b]), torch.tensor([best_anchor]), 
                                       torch.tensor([gj[0]]), torch.tensor([gi[0]]))
            
            tcls.append(tcls_i)
            tbox.append(tbox_i)
            indices.append(indices_i)
            anchors.append(anchors_i)  
        
        return tcls, tbox, indices, anchors
    
    def forward(self, predictions, targets):
        """Forward pass com loss YOLO real"""
        device = predictions[0].device
        
        # Build targets
        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)
        
        # Initialize losses 
        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        cls_loss = torch.tensor(0.0, device=device)
        
        for i, pred in enumerate(predictions):
            B, C, H, W = pred.shape
            
            # Reshape prediction: [B, C, H, W] -> [B, 3, C//3, H, W]
            num_anchors = 3
            pred = pred.view(B, num_anchors, -1, H, W).permute(0, 1, 3, 4, 2).contiguous()
            # pred: [B, 3, H, W, features]
            
            # Get indices for this scale
            b_idx, a_idx, gj, gi = indices[i]
            
            # Objectness targets
            tobj = torch.zeros_like(pred[..., 4], device=device)
            
            if len(b_idx) > 0:
                # Select predictions for positive samples
                pred_subset = pred[b_idx, a_idx, gj, gi] # [N, features]
                
                # Box predictions (x, y, w, h)
                pred_xy = torch.sigmoid(pred_subset[:, :2])  # xy
                pred_wh = pred_subset[:, 2:4]  # wh (will be exp transformed)
                pred_obj = pred_subset[:, 4]   # objectness
                
                # Box targets
                target_boxes = tbox[i][b_idx, a_idx, gj, gi]  # [N, 4]
                target_xy = target_boxes[:, :2]
                target_wh = target_boxes[:, 2:4]
                
                # Box loss (using Focal-EIoU)
                if len(pred_subset) > 0:
                    # Convert to xyxy format for IoU calculation
                    pred_boxes_xyxy = torch.cat([
                        pred_xy - torch.exp(pred_wh) * anchors[i][a_idx] / 2,
                        pred_xy + torch.exp(pred_wh) * anchors[i][a_idx] / 2
                    ], dim=1)
                    
                    target_boxes_xyxy = torch.cat([
                        target_xy - target_wh / 2,
                        target_xy + target_wh / 2  
                    ], dim=1)
                    
                    # Apply Focal-EIoU loss
                    box_loss += self.focal_eiou(pred_boxes_xyxy, target_boxes_xyxy)
                
                # Objectness targets (set to 1 for positive samples)
                tobj[b_idx, a_idx, gj, gi] = 1.0
                
                # Class loss
                if self.nc > 1 and pred.size(-1) > 5:
                    target_cls = tcls[i][b_idx, a_idx, gj, gi]
                    pred_cls = pred_subset[:, 5:]
                    
                    # One-hot encoding
                    target_cls_onehot = torch.zeros_like(pred_cls)
                    target_cls_onehot[range(len(target_cls)), target_cls] = 1.0
                    
                    cls_loss += self.bce_cls(pred_cls, target_cls_onehot)
            
            # Objectness loss (all predictions)
            obj_loss += self.bce_obj(pred[..., 4], tobj)
        
        # Combine losses
        total_loss = (self.lambda_box * box_loss + 
                     self.lambda_obj * obj_loss + 
                     self.lambda_cls * cls_loss)
        
        # Return loss components
        loss_items = torch.stack([box_loss, obj_loss, cls_loss, total_loss]).detach()
        
        return total_loss, loss_items

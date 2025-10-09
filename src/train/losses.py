import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, eps=1e-7):
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
    def __init__(self, alpha=1.0, gamma=0.5, eps=1e-7):
        super(FocalEIoULoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, pred, target):
        #pred: [N, 4] predicted boxes (x1, y1, x2, y2)
        #target: [N, 4] target boxes (x1, y1, x2, y2)

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
    def __init__(self, nc=1, device='cpu', focal_gamma=0.5):
        super(SmallObjectYOLOLoss, self).__init__()
        self.nc = nc
        self.device = device
        self.focal_gamma = focal_gamma
        
        # Loss functions
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        self.focal_eiou = FocalEIoULoss(gamma=focal_gamma)
        
        # Loss weights optimized for small objects
        self.lambda_box = 0.1  
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
        
        # Anchors for different scales (P3, P4, P5)
        self.anchors = [
            torch.tensor([[10, 13], [16, 30], [33, 23]], device=device, dtype=torch.float32) / 8,    # P3
            torch.tensor([[30, 61], [62, 45], [59, 119]], device=device, dtype=torch.float32) / 16,  # P4  
            torch.tensor([[116, 90], [156, 198], [373, 326]], device=device, dtype=torch.float32) / 32 # P5
        ]
        
    def build_targets(self, predictions, targets):
        tcls, tbox, indices, anchors = [], [], [], []
        
        for i, pred in enumerate(predictions):
            device = pred.device
            B, _, H, W = pred.shape
            anchors_i = self.anchors[i].to(device)
            
            # Initialize
            tcls_i = torch.zeros(B, 3, H, W, device=device, dtype=torch.long)
            tbox_i = torch.zeros(B, 3, H, W, 4, device=device)
            
            # Create empty lists to gather indices for this prediction scale
            b_indices, a_indices, gj_indices, gi_indices = [], [], [], []

            if len(targets) > 0:
                # Process targets for this scale
                for b, target in enumerate(targets):
                    if 'boxes' in target and len(target['boxes']) > 0:
                        boxes = target['boxes'].to(device)
                        labels = target['labels'].to(device) if 'labels' in target else torch.zeros(len(boxes), dtype=torch.long, device=device)
                        
                        # Scale boxes to the feature map grid
                        boxes_scaled = boxes.clone()
                        boxes_scaled[:, [0, 2]] *= W  # x, w
                        boxes_scaled[:, [1, 3]] *= H  # y, h
                        
                        gwh = boxes_scaled[:, 2:4]  # grid wh
                        
                        # Check whether there are ground-truth boxes
                        if gwh.shape[0] > 0:
                            # Calculate IoU between all targets and anchors for this scale
                            # Expand dims to broadcast: [N, 1, 2] vs [1, A, 2] -> [N, A, 2]
                            iou_matrix = bbox_iou(
                                torch.cat([torch.zeros_like(gwh).unsqueeze(1), gwh.unsqueeze(1)], dim=-1).repeat(1, len(anchors_i), 1),
                                torch.cat([torch.zeros_like(anchors_i).unsqueeze(0), anchors_i.unsqueeze(0)], dim=-1).repeat(gwh.shape[0], 1, 1),
                                xywh=True
                            )
                            # Get the best anchor for each target box
                            best_anchors_for_targets = iou_matrix.argmax(dim=1)

                            # Assign targets to grid cells
                            for t_idx, best_anchor in enumerate(best_anchors_for_targets):
                                gxy = boxes_scaled[t_idx, :2]
                                gi = gxy[0].long().clamp(0, W-1)
                                gj = gxy[1].long().clamp(0, H-1)

                                # Assign targets
                                tcls_i[b, best_anchor, gj, gi] = labels[t_idx]
                                tbox_i[b, best_anchor, gj, gi] = boxes[t_idx]

                                # Store indices
                                b_indices.append(b)
                                a_indices.append(best_anchor)
                                gj_indices.append(gj)
                                gi_indices.append(gi)

            # Convert collected indices to tensors
            indices_i = (
                torch.tensor(b_indices, device=device, dtype=torch.long),
                torch.tensor(a_indices, device=device, dtype=torch.long),
                torch.stack(gj_indices).long() if gj_indices else torch.tensor([], device=device, dtype=torch.long),
                torch.stack(gi_indices).long() if gi_indices else torch.tensor([], device=device, dtype=torch.long)
            )
            
            tcls.append(tcls_i)
            tbox.append(tbox_i)
            indices.append(indices_i)
            anchors.append(anchors_i)  
        
        return tcls, tbox, indices, anchors
    
    def forward(self, predictions, targets):
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
            
            if b_idx.numel() > 0:
                # Select predictions for positive samples
                pred_subset = pred[b_idx, a_idx, gj, gi] # [N, features]
                
                # Box predictions (x, y, w, h)
                pred_xy = torch.sigmoid(pred_subset[:, :2])  # xy
                pred_wh = pred_subset[:, 2:4]  # wh (will be exp transformed)
                
                # Box targets
                target_boxes = tbox[i][b_idx, a_idx, gj, gi]  # [N, 4]
                
                # Transform predicted boxes        
                anchor_wh = anchors[i][a_idx]
                pred_w_transformed = torch.exp(pred_wh[:, 0]) * anchor_wh[:, 0]
                pred_h_transformed = torch.exp(pred_wh[:, 1]) * anchor_wh[:, 1]
                pred_box_transformed = torch.stack([pred_xy[:, 0], pred_xy[:, 1], pred_w_transformed, pred_h_transformed], dim=1)

                # Convert to xyxy format for IoU calculation
                # Box format is center_x, center_y, width, height
                pred_boxes_xyxy = torch.cat((
                    pred_box_transformed[:, :2] - pred_box_transformed[:, 2:] / 2,
                    pred_box_transformed[:, :2] + pred_box_transformed[:, 2:] / 2
                ), 1)

                target_boxes_xyxy = torch.cat((
                    target_boxes[:, :2] - target_boxes[:, 2:] / 2,
                    target_boxes[:, :2] + target_boxes[:, 2:] / 2
                ), 1)
                
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
                    target_cls_onehot.scatter_(1, target_cls.unsqueeze(1), 1.0)
                    
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

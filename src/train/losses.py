# src/train/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """Calculate IoU of box1(1,4) to box2(n,4)."""
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
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in small object detection."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class QualityFocalLoss(nn.Module):
    """Quality Focal Loss for better localization quality estimation."""
    def __init__(self, beta=2.0):
        super(QualityFocalLoss, self).__init__()
        self.beta = beta

    def forward(self, pred, target, score):
        weight = torch.abs(target - pred) ** self.beta
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
        return loss.mean()

class SmallObjectLoss(nn.Module):
    """Enhanced loss function specifically designed for small object detection."""
    
    def __init__(self, nc=1, device='cpu', hyp=None):
        super(SmallObjectLoss, self).__init__()
        if hyp is None:
            hyp = {
                'box': 0.05,
                'cls': 0.5,
                'obj': 1.0,
                'focal_loss_gamma': 1.5,
                'fl_gamma': 0.0,
                'small_obj_weight': 2.0,  
                'iou_type': 'CIoU'
            }
        
        self.hyp = hyp
        self.nc = nc
        self.device = device
        
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp.get('cls_pw', 1.0)], device=device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp.get('obj_pw', 1.0)], device=device))
        self.focal_loss = FocalLoss(gamma=hyp['focal_loss_gamma'])
        
        self.gr = 1.0  
        self.autobalance = False

        
    def build_targets(self, p, targets):
        """Build targets for loss computation with small object emphasis."""
        tcls, tbox, indices, anchors = [], [], [], []
        na, nt = 3, targets.shape[0]  # number of anchors, targets
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                           [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                           ], device=targets.device).float() * g  # offsets

        for i, pred in enumerate(p):
            anchors_i = torch.tensor([[10, 13], [16, 30], [33, 23]]) / 8  # Example anchors for P3
            if i == 1:  # P4
                anchors_i = torch.tensor([[30, 61], [62, 45], [59, 119]]) / 16
            elif i == 2:  # P5
                anchors_i = torch.tensor([[116, 90], [156, 198], [373, 326]]) / 32
                
            anchors_i = anchors_i.to(targets.device)
            
            gain = torch.tensor(pred.shape)[[3, 2, 3, 2]].to(targets.device)  # xyxy gain

            # Match targets to anchors
            t = targets.clone()
            if nt:
                # Scale box coordinates
                t[:, :, 2:6] *= gain
                # Matches
                r = t[:, :, 4:6] / anchors_i[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < 4  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, 0].long(), t[:, 1].long()
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            a = t[:, 6].long()
            
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, gain[3].item() - 1), gi.clamp_(0, gain[2].item() - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anchors.append(anchors_i[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anchors

    # In src/train/losses.py

    def forward(self, p, targets):
        """
        Args:
            p: list of predictions [P3, P4, P5]
            targets: ground truth targets
        """
        device = p[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        # Prepare targets
        targets_list = []
        for i, t in enumerate(targets):
            img_idx = torch.full((t['labels'].shape[0], 1), i, device=device, dtype=torch.float)
            boxes = t['boxes'].to(device)
            labels = t['labels'][:, None].to(device).float()
            targets_list.append(torch.cat([img_idx, labels, boxes], 1))

        if len(targets_list) > 0:
            targets = torch.cat(targets_list, 0)
        else:
            targets = torch.zeros(0, 6, device=device)

        # --- FIX STARTS HERE ---
        # Add a dummy anchor dimension to predictions to match the loss function's expectation
        p_reshaped = []
        for pi in p:
            # pi shape: [batch, features, height, width]
            bs, _, ny, nx = pi.shape
            # Reshape to [batch, 1 (anchor), features, height, width]
            pi = pi.view(bs, 1, -1, ny, nx)
            # Permute to [batch, 1 (anchor), height, width, features]
            pi = pi.permute(0, 1, 3, 4, 2).contiguous()
            p_reshaped.append(pi)
        # --- FIX ENDS HERE ---

        tcls, tbox, indices, anchors = self.build_targets(p_reshaped, targets)  # targets

        # Losses
        for i, pi in enumerate(p_reshaped):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
            
            # 🔧 CORREÇÃO: Indentação corrigida - fica DENTRO do for
            if len(b):
                # Predictions
                pxy = torch.sigmoid(pi[b, a, gj, gi, :2])  # predicted xy
                pwh = pi[b, a, gj, gi, 2:4]  # predicted wh
                pbox = torch.cat((pxy, torch.exp(pwh) * anchors[i]), 1)  # predicted box
                
                # Box loss
                iou = bbox_iou(pbox.T, tbox[i], CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()
                
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)
                
                # Classification
                if self.nc > 1:
                    t = torch.full_like(pi[..., 5:], 0, device=device)
                    t[b, a, gj, gi, tcls[i]] = 1
                    lcls += self.BCEcls(pi[..., 5:], t)

            # Objectness loss (também DENTRO do for, mas fora do if)
            lobj += self.BCEobj(pi[..., 4], tobj)

        # Total loss (FORA do for)
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        
        loss = lbox + lobj + lcls
        
        return loss, torch.cat((lbox, lobj, lcls, loss)).detach()

class EIoULoss(nn.Module):
    """
    Efficient IoU Loss for small object detection
    Paper: https://arxiv.org/abs/2101.08158
    """
    def __init__(self, eps=1e-7):
        super(EIoULoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        """
        pred: [N, 4] (x1, y1, x2, y2)
        target: [N, 4] (x1, y1, x2, y2)
        """
        # Intersection coordinates
        inter_x1 = torch.max(pred[:, 0], target[:, 0])
        inter_y1 = torch.max(pred[:, 1], target[:, 1])
        inter_x2 = torch.min(pred[:, 2], target[:, 2])
        inter_y2 = torch.min(pred[:, 3], target[:, 3])

        # Intersection area
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Prediction and target areas
        pred_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
        target_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])

        # Union area
        union_area = pred_area + target_area - inter_area + self.eps

        # IoU
        iou = inter_area / union_area

        # Enclosing box
        enclose_x1 = torch.min(pred[:, 0], target[:, 0])
        enclose_y1 = torch.min(pred[:, 1], target[:, 1])
        enclose_x2 = torch.max(pred[:, 2], target[:, 2])
        enclose_y2 = torch.max(pred[:, 3], target[:, 3])

        # Enclosing box width and height
        enclose_w = enclose_x2 - enclose_x1 + self.eps
        enclose_h = enclose_y2 - enclose_y1 + self.eps

        # Center distance
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2
        target_cx = (target[:, 0] + target[:, 2]) / 2
        target_cy = (target[:, 1] + target[:, 3]) / 2

        rho2 = ((pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2)
        c2 = enclose_w ** 2 + enclose_h ** 2 + self.eps

        # Width and height differences
        pred_w = pred[:, 2] - pred[:, 0]
        pred_h = pred[:, 3] - pred[:, 1]
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

        w_diff = torch.abs(pred_w - target_w)
        h_diff = torch.abs(pred_h - target_h)

        # EIoU = IoU - rho2/c2 - w_diff^2/cw^2 - h_diff^2/ch^2
        eiou = iou - rho2 / c2 - w_diff ** 2 / enclose_w ** 2 - h_diff ** 2 / enclose_h ** 2

        return 1 - eiou
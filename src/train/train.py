# src/train/train.py
import os, yaml, math, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.model import SmallObjectYOLO
from src.data.dataset import YOLODataset, collate_fn
from src.data.transforms import simple_transforms

def iou_xyxy(a, b, eps=1e-7):
    # a: [N,4], b: [M,4]
    lt = torch.max(a[:, None, :2], b[:, :2])   # [N,M,2]
    rb = torch.min(a[:, None, 2:], b[:, 2:])   # [N,M,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)
    union = area_a[:, None] + area_b - inter + eps
    return inter / union

def build_dataloader(data_yaml, img_size=640, bs=8, workers=0):
    use_cuda = torch.cuda.is_available()
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    root = cfg.get("path", ".")
    train_set = YOLODataset(root, "train", img_size, transforms=simple_transforms, class_names=list(cfg.get("names", {}).values()))
    val_set = YOLODataset(root, "val", img_size, transforms=simple_transforms, class_names=list(cfg.get("names", {}).values()))
    dl_tr = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=use_cuda, collate_fn=collate_fn)
    dl_va = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=use_cuda, collate_fn=collate_fn)
    nc = len(cfg.get("names", {}))
    return dl_tr, dl_va, nc

class SimpleLoss(nn.Module):
    def __init__(self, nc, weights=(1.0, 0.7, 0.5)):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.nc = nc
        self.ws = weights  # pesos P2,P3,P4
    def forward(self, preds, targets, strides):
        # Esta é uma loss placeholder para kickstart; substitua por assigner+loss adequados
        loss = 0.0
        for i, (p, s) in enumerate(zip(preds, strides)):
            B, C, H, W = p.shape
            reg, obj, cls = p[:, :4], p[:, 4:5], p[:, 5:]
            # penalizar objetividade vazia (sem matching real nesta versão mínima)
            loss = loss + self.ws[i] * (obj.sigmoid().mean() + cls.sigmoid().mean() * 0.0 + reg.mean() * 0.0)
        return loss

def train(data_yaml="configs/data.yaml", epochs=10, batch=8, lr=5e-4, img_size=640, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dl_tr, dl_va, nc = build_dataloader(data_yaml, img_size, batch)
    model = SmallObjectYOLO(nc=nc).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = SimpleLoss(nc)
    for ep in range(epochs):
        model.train()
        tl = 0.0
        for ims, targs in dl_tr:
            ims = ims.to(device)
            out, strides = model(ims)
            loss = criterion(out, targs, strides)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item()
        sch.step()
        print(f"Epoch {ep+1}/{epochs} - train_loss: {tl/len(dl_tr):.4f}")
    torch.save(model.state_dict(), "weights_small_object_yolo.pt")
    print("Treino concluído.")

if __name__ == "__main__":
    train()

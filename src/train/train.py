# src/train/train.py
import os
import sys
import yaml
import math
import time
import json
import shutil
import logging
import argparse
import torch.nn.functional as F
from torchvision.ops import nms

# Adicionar o diretório raiz ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Imports relativos corrigidos
try:
    from src.models.model import SmallObjectYOLO
    from src.train.losses import SmallObjectYOLOLoss as SmallObjectLoss
    from src.data.dataset import SmallObjectDataset, collate_fn 
    from src.data.transforms import SmallObjectAugmentationPipeline
    from src.train.losses import bbox_iou
except ImportError:
    # Fallback para imports diretos
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, root_dir)

    from src.models.model import SmallObjectYOLO
    from src.train.losses import SmallObjectLoss
    from src.data.dataset import SmallObjectDataset, collate_fn
    from src.data.transforms import SmallObjectAugmentationPipeline

ANCHORS = torch.tensor([
    [[10, 13], [16, 30], [33, 23]],      # Para P3 (stride 8)
    [[30, 61], [62, 45], [59, 119]],     # Para P4 (stride 16)
    [[116, 90], [156, 198], [373, 326]]  # Para P5 (stride 32)
], dtype=torch.float32)

# (As classes EarlyStopping, WarmupCosineScheduler, etc. continuam aqui, sem alterações)
class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class WarmupCosineScheduler:
    """Warmup + Cosine Annealing LR Scheduler"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def build_enhanced_dataloader(data_yaml, img_size, batch_size, workers=4, training=True):
    """Builds an enhanced dataloader for small object detection."""
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)

    path = data['path']
    nc = data['nc']
    
    split = 'train' if training else 'val'
    
    # --- LÓGICA FINAL E ROBUSTA ---
    # Lê o caminho relativo das imagens diretamente do data.yaml
    # Ex: data['train'] é 'images/train'
    img_dir = os.path.join(path, data[split])
    
    # Deriva o caminho dos labels a partir do caminho das imagens
    # Isso assume que a estrutura é a mesma, trocando 'images' por 'labels'
    label_dir = img_dir.replace('images', 'labels')

    # O resto da função continua igual...
    transforms = SmallObjectAugmentationPipeline(img_size=img_size, training=training)
    
    dataset = SmallObjectDataset(img_dir=img_dir, label_dir=label_dir, transforms=transforms)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader, nc

def calculate_metrics(predictions, targets, iou_threshold=0.5):
    """Calculates precision, recall, and F1 score."""
    true_positives = 0
    num_preds = 0
    total_targets = 0

    # Iterate over each image in the batch
    for i, (preds, target) in enumerate(zip(predictions, targets)):
        target_boxes = target['boxes']
        target_labels = target['labels']
        total_targets += len(target_labels)
        num_preds += len(preds)

        if len(preds) == 0 or len(target_boxes) == 0:
            continue

        # Keep track of which ground truth boxes have been detected
        detected = torch.zeros(target_boxes.shape[0], dtype=torch.bool, device=preds.device)

        # Match predictions to targets
        for j, pred in enumerate(preds):
            pred_box = pred[:4]
            pred_label = pred[5]

            # Find matching class labels in the targets
            class_mask = (target_labels == pred_label)
            if not class_mask.any():
                continue

            # Get the ground truth boxes and their original indices for the matching class
            matching_target_boxes = target_boxes[class_mask]
            matching_target_indices = torch.where(class_mask)[0]

            # Convert target boxes to xyxy for IoU calculation
            tbox_xyxy = torch.cat((
                matching_target_boxes[:, :2] - matching_target_boxes[:, 2:] / 2,
                matching_target_boxes[:, :2] + matching_target_boxes[:, 2:] / 2
            ), 1)

            # Calculate IoU between the current prediction and all matching targets
            overlaps = bbox_iou(pred_box.unsqueeze(0), tbox_xyxy, xywh=False).squeeze(0)

            if overlaps.numel() == 0:
                continue

            # Find the best match
            best_overlap, best_idx = overlaps.max(0)

            # 🚨 CORREÇÃO AQUI 🚨
            # Use .item() to get scalar values for the condition
            # And check if the best matching GT box hasn't been detected yet
            original_target_idx = matching_target_indices[best_idx]
            if best_overlap.item() > iou_threshold and not detected[original_target_idx]:
                true_positives += 1
                detected[original_target_idx] = True # Mark this GT box as detected

    # Calculate metrics
    fp = num_preds - true_positives
    fn = total_targets - true_positives

    precision = true_positives / (num_preds + 1e-16)
    recall = true_positives / (total_targets + 1e-16)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-16)

    return {'precision': precision, 'recall': recall, 'f1': f1}

def setup_logging(save_dir):
    """Configura o logging para salvar em arquivo e no console"""
    log_file = os.path.join(save_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def postprocess_and_nms(outputs, strides, conf_threshold=0.25, iou_threshold=0.45):
    """
    Decodifica a saída do modelo e aplica NMS, de forma consistente com SmallObjectYOLOLoss.
    """
    all_preds_per_image = [[] for _ in range(outputs[0].shape[0])]
    
    for i, (pred, stride) in enumerate(zip(outputs, strides)):
        B, C, H, W = pred.shape
        device = pred.device
        
        # Seleciona as âncoras para a escala atual e move para o dispositivo
        current_anchors = ANCHORS[i].to(device)

        # Reformatar a saída do modelo
        num_anchors = 3
        num_classes = C // num_anchors - 5
        pred = pred.view(B, num_anchors, -1, H, W).permute(0, 1, 3, 4, 2).contiguous()
        
        # Criar a grade de células para adicionar os offsets
        grid_x, grid_y = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
        grid_xy = torch.stack((grid_x, grid_y), 2).view(1, 1, H, W, 2)
        
        # --- DECODIFICAÇÃO ---
        # A lógica aqui agora espelha a lógica inversa da sua função de perda.
        
        # Decodificar coordenadas do centro (x, y)
        # Lógica padrão do YOLOv5/v7, que é consistente com o cálculo de gi, gj em build_targets
        xy = (torch.sigmoid(pred[..., 0:2]) * 2 - 0.5 + grid_xy) * stride
        
        # Decodificar largura e altura (w, h)
        # 🚨 ESTA É A MUDANÇA CRÍTICA 🚨
        # Usamos a transformação exponencial com as âncoras, exatamente como em SmallObjectYOLOLoss
        anchor_grid = current_anchors.view(1, num_anchors, 1, 1, 2)
        wh = (torch.exp(pred[..., 2:4]) * anchor_grid) * stride # A loss usa torch.exp(), então aqui usamos o inverso
        
        # Obter confiança de objeto e probabilidades de classe
        pred_conf = torch.sigmoid(pred[..., 4])
        pred_cls = torch.sigmoid(pred[..., 5:])
        
        # Combinar todas as predições decodificadas
        output = torch.cat((xy, wh, pred_conf.unsqueeze(-1), pred_cls), -1)
        output = output.view(B, -1, 5 + num_classes) # [Batch, N_Boxes, 5+NC]

        # Processar cada imagem do batch separadamente para o NMS
        for b in range(B):
            img_preds = output[b]
            
            # Filtrar predições com baixa confiança
            conf_mask = img_preds[:, 4] >= conf_threshold
            img_preds = img_preds[conf_mask]
            
            if not img_preds.shape[0]:
                continue
            
            # Obter a classe com maior pontuação e a pontuação final
            class_conf, class_pred = torch.max(img_preds[:, 5:], 1, keepdim=True)
            final_conf = img_preds[:, 4].unsqueeze(1) * class_conf
            
            # Converter caixas de [cx, cy, w, h] para [x1, y1, x2, y2] para o NMS
            box_xywh = img_preds[:, :4]
            box_xyxy = torch.empty_like(box_xywh)
            box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2  # x1
            box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2  # y1
            box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2  # x2
            box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2  # y2
            
            # Formato final para NMS: [x1, y1, x2, y2, conf, class]
            detections = torch.cat((box_xyxy, final_conf, class_pred.float()), 1)
            
            # Aplicar NMS por classe
            unique_classes = detections[:, 5].unique()
            nms_detections = []
            for cls in unique_classes:
                cls_mask = detections[:, 5] == cls
                cls_dets = detections[cls_mask]
                keep = nms(cls_dets[:, :4], cls_dets[:, 4], iou_threshold)
                nms_detections.append(cls_dets[keep])
            
            if nms_detections:
                all_preds_per_image[b].append(torch.cat(nms_detections))

    # Concatenar predições de todas as escalas para cada imagem
    final_results = [torch.cat(preds) if preds else torch.empty(0, 6, device=outputs[0].device) for preds in all_preds_per_image]
    
    return final_results

def enhanced_train(
    data_yaml,
    model_yaml,
    epochs,
    batch_size,
    lr,
    img_size,
    device,
    weights,
    save_dir
):
    """FUNÇÃO DE TREINAMENTO CONSOLIDADA E APRIMORADA"""
    # Setup
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logging(save_dir)
    logger.info(f"Using device: {device}")
    logger.info(f"Saving results to: {save_dir}")

    # Salvar configurações para reprodutibilidade
    shutil.copy(data_yaml, os.path.join(save_dir, 'data.yaml'))
    shutil.copy(model_yaml, os.path.join(save_dir, 'model.yaml'))

    # Load config
    with open(model_yaml, "r") as f:
        model_cfg = yaml.safe_load(f)
    
    # Dataloaders
    logger.info("Building dataloaders...")
    train_loader, nc = build_enhanced_dataloader(data_yaml, img_size, batch_size, workers=4, training=True)
    val_loader, _ = build_enhanced_dataloader(data_yaml, img_size, batch_size*2, workers=4, training=False)
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")

    # Model
    logger.info("Initializing model...")
    # Usando a nova arquitetura com Módulo de Relação
    model = SmallObjectYOLO(nc=nc, ch=model_cfg.get('channels', [64, 128, 256, 384]))
    
    # if weights and os.path.exists(weights):
    #     logger.info(f"Loading weights from {weights}")
    #     checkpoint = torch.load(weights, map_location=device)
    #     # Ajuste para carregar pesos mesmo com a nova arquitetura
    #     model_dict = model.state_dict()
    #     pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict)
    #     logger.info(f"Loaded {len(pretrained_dict)} layers from pretrained weights.")
    
    logger.info("Initializing model with random weights - training from scratch")

    model = model.to(device)
    
    # Loss, Optimizer, Scheduler, etc.
    criterion = SmallObjectLoss(nc=nc, device=device, focal_gamma=0.5)
    optimizer = torch.optim.AdamW([{'params': model.parameters()}], lr=lr, weight_decay=5e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=epochs, base_lr=lr, min_lr=lr * 0.01)
    scaler = GradScaler() if device == 'cuda' else None
    early_stopping = EarlyStopping(patience=30, min_delta=0.001)


    # Training loop
    logger.info("Starting training...")
    best_loss = float('inf')
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        current_lr = scheduler.step(epoch)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast():
                    outputs, strides = model(images)
                    loss, loss_items = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, strides = model(images)
                loss, loss_items = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            epoch_loss = epoch_loss + loss.item()
        
        # Validation
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics_sum = {'precision': 0, 'recall': 0, 'f1': 0}
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                # Move targets to the correct device inside the loop
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs, strides = model(images)
                
                # Calculate loss
                loss, loss_items = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate metrics
                predictions = postprocess_and_nms(outputs, strides)
                metrics = calculate_metrics(predictions, targets)
                
                for key in val_metrics_sum:
                    val_metrics_sum[key] += metrics[key]

        
        # Average metrics
        val_loss /= len(val_loader)
        val_metrics = {key: val / len(val_loader) for key, val in val_metrics_sum.items()}

        logger.info(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")

                # Salvar histórico
        epoch_results = {
            'epoch': epoch+1, 
            'lr': float(current_lr), 
            'train_loss': float(epoch_loss), 
            'val_loss': float(val_loss),
            'precision': float(val_metrics['precision']),
            'recall': float(val_metrics['recall']),
            'f1': float(val_metrics['f1'])
        }
        history.append(epoch_results)
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=4)

        # Salvar melhor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))
            logger.info(f"New best model saved with val_loss: {val_loss:.4f}")

        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model_relation.pt'))
    logger.info("Training completed!")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Training for Small Object Detection')
    parser.add_argument('--img-size', type=int, default=1024, help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training (A100 can handle this with 1024px)')
    # 🔧 ALTERAÇÃO B: Aumentar epochs de 150 para 200
    parser.add_argument('--epochs', type=int, default=200, help='Total number of epochs')
    # 🔧 ALTERAÇÃO A: Aumentar learning rate de 5e-4 para 1e-3
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--data', type=str, default='configs/data.yaml', help='Path to data.yaml')
    parser.add_argument('--model', type=str, default='configs/enhanced_model.yaml', help='Path to model.yaml')
    parser.add_argument('--weights', type=str, default='weights_small_object_yolo.pt', help='Path to pretrained weights')
    parser.add_argument('--save-dir', type=str, default='runs/train_relation_A100', help='Directory to save results')
    
    args = parser.parse_args()

    enhanced_train(
        data_yaml=args.data,
        model_yaml=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        device=None,
        weights=args.weights,
        save_dir=args.save_dir
    )


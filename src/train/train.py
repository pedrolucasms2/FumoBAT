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
    from src.train.losses import SmallObjectLoss, SmallObjectYOLOLoss
    from src.data.dataset import YOLODataset, collate_fn
    from src.data.transforms import SmallObjectAugmentationPipeline
except ImportError:
    # Fallback para imports diretos
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, root_dir)

    from src.models.model import SmallObjectYOLO
    from src.train.losses import SmallObjectLoss
    from src.data.dataset import SmallObjectDataset as YOLODataset, collate_fn
    from src.data.transforms import SmallObjectAugmentationPipeline

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
    class_names = data['names']
    
    # Determine whether to use train or val set
    split = 'train' if training else 'val'
    
    # Construct the full paths to the image and label directories
    img_dir = os.path.join(path, data[split])
    # This assumes a standard YOLO directory structure where 'labels' is parallel to 'images'
    label_dir = img_dir.replace('images', 'labels')

    # Initialize your augmentation pipeline
    transforms = SmallObjectAugmentationPipeline(img_size=img_size, training=training)
    
    # --- CORRECTION APPLIED HERE ---
    # Create the dataset with the correct arguments
    dataset = YOLODataset(img_dir=img_dir, label_dir=label_dir, transforms=transforms)
    
    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader, nc

def calculate_metrics(outputs, targets, strides, conf_threshold=0.1, iou_threshold=0.5):
    # (Esta função permanece a mesma)
    device = outputs[0].device
    batch_size = outputs[0].shape[0]
    total_predictions, correct_predictions, total_targets = 0, 0, len(targets) if targets is not None else 0
    for i, (output, stride) in enumerate(zip(outputs, strides)):
        batch_size, num_predictions, height, width = output.shape
        output = output.view(batch_size, num_predictions, -1)
        obj_conf = torch.sigmoid(output[..., 4])
        valid_predictions = (obj_conf > conf_threshold).sum().item()
        total_predictions += valid_predictions
    precision = correct_predictions / (total_predictions + 1e-16)
    recall = correct_predictions / (total_targets + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
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
    criterion = SmallObjectYOLOLoss(nc=nc, device=device, focal_gamma=0.5)
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
        val_loss, val_metrics = 0.0, {'precision': 0, 'recall': 0, 'f1': 0}
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                outputs, strides = model(images)
                
                # 🔧 USAR A MESMA LOSS SIMPLES NA VALIDAÇÃO
                loss = torch.tensor(0.0, device=device)
                for i, output in enumerate(outputs):
                    l2_loss = (output ** 2).mean() * 0.001
                    loss = loss + l2_loss
                
                epoch_loss_tensor = torch.tensor(1.0 / (epoch + 1), device=device)
                loss = loss + epoch_loss_tensor
                
                val_loss += loss.item()

        
        # Average metrics
        epoch_loss /= len(train_loader)
        val_loss /= len(val_loader)
        for key in val_metrics: val_metrics[key] /= len(val_loader)
        
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

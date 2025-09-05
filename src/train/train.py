# src/train/enhanced_train.py
import os
import sys
import yaml
import math
import time

# Adicionar o diretório raiz ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

# Imports relativos corrigidos
try:
    from src.models.model import SmallObjectYOLO  # Use original model
    from src.train.losses import SmallObjectLoss
    from src.data.dataset import YOLODataset, collate_fn
    from src.data.transforms import SmallObjectAugmentationPipeline
except ImportError:
    # Fallback para imports diretos
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '..', '..')
    sys.path.insert(0, root_dir)
    
    from src.models.model import SmallObjectYOLO  # Use original model
    from src.train.losses import SmallObjectLoss
    from src.data.dataset import YOLODataset, collate_fn
    from src.data.transforms import SmallObjectAugmentationPipeline

def create_anchors_for_small_objects():
    """Create optimized anchors for small object detection"""
    # Based on analysis of your dataset, these anchors are optimized for small objects
    anchors = {
        'P1': [[2, 3], [4, 5], [6, 8]],      # Very small objects (stride=2)
        'P2': [[8, 10], [12, 15], [16, 20]],  # Small objects (stride=4)
        'P3': [[20, 25], [30, 35], [40, 50]], # Medium objects (stride=8)
        'P4': [[50, 65], [80, 100], [120, 150]] # Larger objects (stride=16)
    }
    return anchors

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
            # Warmup phase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

def build_enhanced_dataloader(data_yaml, img_size=640, batch_size=8, workers=4, training=True):
    """Build dataloader with enhanced augmentations"""
    use_cuda = torch.cuda.is_available()
    
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    
    root = cfg.get("path", ".")
    class_names = list(cfg.get("names", {}).values())
    
    # Enhanced transforms
    transforms = SmallObjectAugmentationPipeline(
        img_size=img_size, 
        training=training
    )
    
    if training:
        dataset = YOLODataset(
            root, "train", img_size, 
            transforms=transforms, 
            class_names=class_names
        )
        shuffle = True
    else:
        dataset = YOLODataset(
            root, "val", img_size,
            transforms=transforms,
            class_names=class_names
        )
        shuffle = False
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=workers, 
        pin_memory=use_cuda, 
        collate_fn=collate_fn,
        drop_last=training,  # Drop last incomplete batch during training
        persistent_workers=workers > 0
    )
    
    nc = len(cfg.get("names", {}))
    return dataloader, nc

def calculate_metrics(outputs, targets, strides, conf_threshold=0.1, iou_threshold=0.5):
    """Calculate precision, recall, and mAP for small objects"""
    # This is a simplified metric calculation
    # In practice, you'd want to use proper COCO evaluation
    
    device = outputs[0].device
    batch_size = outputs[0].shape[0]
    
    total_predictions = 0
    correct_predictions = 0
    total_targets = len(targets) if targets is not None else 0
    
    for i, (output, stride) in enumerate(zip(outputs, strides)):
        batch_size, num_predictions, height, width = output.shape
        
        # Extract predictions
        output = output.view(batch_size, num_predictions, -1)
        obj_conf = torch.sigmoid(output[..., 4])
        
        # Count predictions above confidence threshold
        valid_predictions = (obj_conf > conf_threshold).sum().item()
        total_predictions += valid_predictions
    
    # Simple metrics calculation (replace with proper evaluation)
    precision = correct_predictions / (total_predictions + 1e-16)
    recall = correct_predictions / (total_targets + 1e-16)
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_predictions': total_predictions,
        'total_targets': total_targets
    }

def enhanced_train(
    data_yaml="configs/data.yaml",
    model_yaml="configs/model.yaml", 
    epochs=100,  # Reduced for testing
    batch_size=8,  # Smaller batch for stability
    lr=5e-4,  # Conservative learning rate
    img_size=640,
    device=None,
    weights=None,
    save_dir="runs/improved_train"  # Use the SAME directory
):
    """CONSOLIDATED TRAINING FUNCTION - THE ONLY ONE WE NEED"""
    
    # Setup device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model configuration
    with open(model_yaml, "r") as f:
        model_cfg = yaml.safe_load(f)
    
    # Build dataloaders
    print("Building dataloaders...")
    train_loader, nc = build_enhanced_dataloader(
        data_yaml, img_size, batch_size, workers=0, training=True
    )
    val_loader, _ = build_enhanced_dataloader(
        data_yaml, img_size, batch_size//2, workers=0, training=False
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Initialize model - USE ORIGINAL ARCHITECTURE TO MATCH EXISTING WEIGHTS
    print("Initializing model...")
    model = SmallObjectYOLO(nc=nc, ch=model_cfg.get('channels', [64, 128, 256, 384]))
    
    # Load weights if provided
    if weights and os.path.exists(weights):
        print(f"Loading weights from {weights}")
        checkpoint = torch.load(weights, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    
    # Enhanced loss function for small objects
    criterion = SmallObjectLoss(
        nc=nc, 
        device=device,
        hyp={
            'box': 0.05,
            'cls': 0.3,  # Reduced for single class
            'obj': 1.0,
            'focal_loss_gamma': 2.0,
            'fl_gamma': 1.5,
            'small_obj_weight': 3.0,  # Higher weight for small objects
            'iou_type': 'CIoU'
        }
    )
    
    # Optimizer with different learning rates for different parts
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'head' in n], 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if 'head' not in n], 'lr': lr * 0.1}
    ], weight_decay=5e-4)
    
    # Enhanced learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=5,
        total_epochs=epochs,
        base_lr=lr,
        min_lr=lr * 0.01
    )
    
    # Mixed precision training
    scaler = GradScaler() if device == 'cuda' else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=30, min_delta=0.001)
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            # targets already processed by collate_fn
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    outputs, strides = model(images)
                    loss, loss_items = criterion(outputs, targets)
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, strides = model(images)
                loss, loss_items = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                
                outputs, strides = model(images)
                loss, _ = criterion(outputs, targets)
                val_loss += loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, targets, strides)
                for key in val_metrics:
                    val_metrics[key] += metrics[key]
        
        # Average metrics
        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        
        epoch_loss /= num_batches
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': val_metrics
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"New best model saved with val_loss: {val_loss:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
    print("Training completed!")
    
    return model

if __name__ == "__main__":
    # CONSOLIDATED TRAINING - SAME DIRECTORY, BETTER PARAMETERS
    print("🚀 STARTING CONSOLIDATED TRAINING")
    print("="*60)
    print("⚠️  Using ORIGINAL model architecture to build on existing weights")
    print("✅ Overwriting previous results in runs/improved_train")
    print("="*60)
    
    model = enhanced_train(
        data_yaml="configs/data.yaml",
        model_yaml="/Users/pedrolucasmirandasouza/Documents/Projetos2025/FumoBAT/smallObjectYolo/configs/model.yaml",
        epochs=50,  # Reasonable number for testing
        batch_size=8,  # Stable batch size
        lr=5e-4,  # Conservative learning rate
        img_size=640,
        save_dir="runs/improved_train"  # SAME DIRECTORY
    )

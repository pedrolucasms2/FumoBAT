import torch
import torch.nn as nn
import numpy as np
import torchvision.ops as ops
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc

class SmallObjectEvaluator:
    def __init__(self, model, device='cpu', num_classes=1):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reinicia as estatísticas"""
        self.all_detections = defaultdict(list)
        self.all_annotations = defaultdict(int)
        self.image_ids = []
    
    def add_batch(self, images, targets, image_ids=None):
        """Adiciona um batch para avaliação"""
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            predictions, strides = self.model(images)
            
            # Converter predições para formato final
            detections = self._parse_predictions(predictions, strides)
            
            for i, (detection, target) in enumerate(zip(detections, targets)):
                img_id = image_ids[i] if image_ids else len(self.image_ids)
                self.image_ids.append(img_id)
                
                # Processar ground truth
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                # Contar anotações por classe
                for label in gt_labels:
                    self.all_annotations[label.item()] += 1
                
                # Processar detecções
                if len(detection['boxes']) > 0:
                    self._match_detections(detection, gt_boxes, gt_labels, img_id)
    
    def _parse_predictions(self, predictions, strides):
        """Converte predições raw em detecções finais"""
        batch_size = predictions[0].shape[0]
        detections = []
        
        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []
            
            for pred, stride in zip(predictions, strides):
                # pred shape: [B, 5+nc, H, W] onde 5 = [x, y, w, h, obj]
                B, C, H, W = pred.shape
                nc = C - 5  # número de classes
                
                pred_b = pred[b].permute(1, 2, 0).contiguous().view(-1, C)  # [H*W, C]
                
                # Extrair componentes
                xy = pred_b[:, :2]  # centro x, y
                wh = pred_b[:, 2:4]  # largura, altura
                obj_conf = pred_b[:, 4].sigmoid()
                cls_conf = pred_b[:, 5:].sigmoid() if nc > 0 else torch.ones(pred_b.shape[0], 1)
                
                # Converter para coordenadas absolutas
                grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                grid = torch.stack([grid_x, grid_y], dim=-1).float().to(pred.device)
                grid = grid.view(-1, 2)
                
                xy = (xy.sigmoid() + grid) * stride
                wh = wh.exp() * stride
                
                # Converter para formato x1,y1,x2,y2
                x1y1 = xy - wh / 2
                x2y2 = xy + wh / 2
                boxes = torch.cat([x1y1, x2y2], dim=1)
                
                # Scores finais
                if nc == 1:
                    scores = obj_conf
                    labels = torch.zeros_like(scores).long()
                else:
                    cls_scores = cls_conf.max(dim=1)[0]
                    cls_labels = cls_conf.max(dim=1)[1]
                    scores = obj_conf * cls_scores
                    labels = cls_labels
                
                # Filtrar por confiança
                keep = scores > 0.1
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)
            
            if len(all_boxes) > 0:
                boxes = torch.cat(all_boxes, dim=0)
                scores = torch.cat(all_scores, dim=0)
                labels = torch.cat(all_labels, dim=0)
                
                # NMS
                keep = ops.nms(boxes, scores, iou_threshold=0.5)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                detections.append({
                    'boxes': boxes.cpu(),
                    'scores': scores.cpu(),
                    'labels': labels.cpu()
                })
            else:
                detections.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty(0),
                    'labels': torch.empty(0).long()
                })
        
        return detections
    
    def _match_detections(self, detection, gt_boxes, gt_labels, img_id):
        """Faz matching entre detecções e ground truth"""
        det_boxes = detection['boxes']
        det_scores = detection['scores']
        det_labels = detection['labels']
        
        for class_id in range(self.num_classes):
            # Filtrar por classe
            gt_mask = gt_labels == class_id
            det_mask = det_labels == class_id
            
            gt_class_boxes = gt_boxes[gt_mask]
            det_class_boxes = det_boxes[det_mask]
            det_class_scores = det_scores[det_mask]
            
            if len(det_class_boxes) == 0:
                continue
                
            # Ordenar por score
            sorted_inds = torch.argsort(det_class_scores, descending=True)
            det_class_boxes = det_class_boxes[sorted_inds]
            det_class_scores = det_class_scores[sorted_inds]
            
            # Matching
            detected = [False] * len(gt_class_boxes)
            
            for det_box, det_score in zip(det_class_boxes, det_class_scores):
                max_iou = 0
                max_idx = -1
                
                if len(gt_class_boxes) > 0:
                    ious = ops.box_iou(det_box.unsqueeze(0), gt_class_boxes).squeeze(0)
                    max_iou, max_idx = torch.max(ious, dim=0)
                    max_iou = max_iou.item()
                    max_idx = max_idx.item()
                
                is_tp = max_iou >= 0.5 and not detected[max_idx] if max_idx >= 0 else False
                
                if is_tp:
                    detected[max_idx] = True
                
                self.all_detections[class_id].append({
                    'score': det_score.item(),
                    'tp': is_tp,
                    'image_id': img_id
                })
    
    def compute_ap(self, class_id, iou_threshold=0.5):
        """Calcula AP para uma classe específica"""
        detections = self.all_detections[class_id]
        n_annotations = self.all_annotations[class_id]
        
        if len(detections) == 0 or n_annotations == 0:
            return 0.0, [], []
        
        # Ordenar por score
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        tp = np.array([d['tp'] for d in detections])
        fp = 1 - tp
        
        # Acumular
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calcular precision e recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / n_annotations
        
        # Suavizar curva precision-recall
        precision = np.concatenate([[1], precision, [0]])
        recall = np.concatenate([[0], recall, [1]])
        
        # Interpolação
        for i in range(len(precision) - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        
        # Calcular AP
        indices = np.where(recall[1:] != recall[:-1])[0] + 1
        ap = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])
        
        return ap, precision[1:-1], recall[1:-1]
    
    def compute_map(self, iou_threshold=0.5):
        """Calcula mAP para todas as classes"""
        aps = []
        results = {}
        
        for class_id in range(self.num_classes):
            ap, precision, recall = self.compute_ap(class_id, iou_threshold)
            aps.append(ap)
            results[f'AP_class_{class_id}'] = ap
            results[f'precision_class_{class_id}'] = precision
            results[f'recall_class_{class_id}'] = recall
        
        map_score = np.mean(aps) if aps else 0.0
        results['mAP'] = map_score
        results['APs'] = aps
        
        return results
    
    def compute_small_object_metrics(self):
        """Calcula métricas específicas para objetos pequenos"""
        # Implementar lógica específica para objetos pequenos
        # Filtrar detecções e anotações por tamanho
        small_results = {}
        
        for class_id in range(self.num_classes):
            detections = self.all_detections[class_id]
            
            # Filtrar objetos pequenos (área < 32²)
            small_detections = []
            small_annotations = 0
            
            # Esta é uma implementação simplificada
            # Na prática, você precisaria filtrar por tamanho real
            small_detections = detections  # Por enquanto, considerar todos
            small_annotations = self.all_annotations[class_id]
            
            if small_annotations > 0:
                # Calcular AP para objetos pequenos
                # ... implementar lógica similar ao compute_ap
                pass
        
        return small_results
    
    def plot_pr_curve(self, class_id=0, save_path=None):
        """Plota curva Precision-Recall"""
        ap, precision, recall = self.compute_ap(class_id)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - Class {class_id}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_summary(self):
        """Imprime resumo da avaliação"""
        results = self.compute_map()
        
        print("=" * 50)
        print("AVALIAÇÃO DO MODELO - OBJETOS PEQUENOS")
        print("=" * 50)
        print(f"mAP@0.5: {results['mAP']:.4f}")
        
        for class_id in range(self.num_classes):
            ap = results[f'AP_class_{class_id}']
            n_annotations = self.all_annotations[class_id]
            n_detections = len(self.all_detections[class_id])
            print(f"Classe {class_id}: AP = {ap:.4f} | GT: {n_annotations} | Det: {n_detections}")
        
        print("=" * 50)
        
    def save_results_to_json(self, results, save_path):
        """Salva o dicionário de resultados em um arquivo JSON."""
        # Converter arrays numpy para listas para serialização JSON
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Resultados de avaliação salvos em: {save_path}")
        
    def save_sample_images(self, loader, class_names, save_dir, num_images=10):
        """Salva imagens de exemplo com detecções e ground truth."""
        import cv2
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        
        count = 0
        with torch.no_grad():
            for images, targets in loader:
                if count >= num_images:
                    break
                
                images = images.to(self.device)
                predictions, _ = self.model(images)
                detections = self._parse_predictions(predictions, [8, 16, 32]) # Strides podem precisar de ajuste

                for i in range(len(images)):
                    if count >= num_images:
                        break
                    
                    # Converter imagem de tensor para numpy (BGR)
                    img = images[i].permute(1, 2, 0).cpu().numpy() * 255
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    # Desenhar Ground Truth (em vermelho)
                    for box, label in zip(targets[i]['boxes'], targets[i]['labels']):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, class_names[label], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # Desenhar Predições (em verde)
                    for box, score, label in zip(detections[i]['boxes'], detections[i]['scores'], detections[i]['labels']):
                        if score > 0.3: # Limiar de confiança para visualização
                           x1, y1, x2, y2 = map(int, box)
                           cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                           label_text = f"{class_names[label]}: {score:.2f}"
                           cv2.putText(img, label_text, (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    save_path = os.path.join(save_dir, f"sample_{count}.jpg")
                    cv2.imwrite(save_path, img)
                    count += 1
        print(f"{count} imagens de exemplo salvas em: {save_dir}")


def run_complete_evaluation():
    """COMPLETE EVALUATION FUNCTION - THE ONLY ONE WE NEED"""
    import os
    import sys
    import yaml
    import cv2
    import glob
    from torch.utils.data import DataLoader, Dataset
    
    # Add project root to path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, '..', '..')
    sys.path.append(project_root)
    
    from src.models.model import SmallObjectYOLO
    
    print("RUNNING COMPLETE EVALUATION")
    print("="*60)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    model_path = os.path.join(project_root, "runs/improved_train/improved_best_model.pt")
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at: {model_path}")
        return
    
    # Load data config
    data_yaml = os.path.join(project_root, "configs/data.yaml")
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    
    root = cfg.get("path", ".")
    if not os.path.isabs(root):
        root = os.path.join(project_root, root)
    
    class_names = list(cfg.get("names", {}).values())
    nc = len(class_names)
    
    print(f"Dataset: {root}")
    print(f"Classes: {class_names}")
    
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
    
    # Simple dataset class
    class EvalDataset(Dataset):
        def __init__(self, img_dir, lbl_dir, img_size=640):
            self.img_dir = img_dir
            self.lbl_dir = lbl_dir
            self.img_size = img_size
            
            # Find images
            self.img_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
            self.img_paths = sorted(self.img_paths)
            print(f"Found {len(self.img_paths)} images")
        
        def __len__(self):
            return len(self.img_paths)
        
        def __getitem__(self, idx):
            img_path = self.img_paths[idx]
            
            # Load image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img.shape[:2]
            
            # Load labels
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(self.lbl_dir, f"{base_name}.txt")
            
            boxes = []
            labels = []
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                cx, cy, w, h = map(float, parts[1:5])
                                
                                # Convert to absolute coordinates
                                x1 = (cx - w/2) * w_orig
                                y1 = (cy - h/2) * h_orig
                                x2 = (cx + w/2) * w_orig
                                y2 = (cy + h/2) * h_orig
                                
                                boxes.append([x1, y1, x2, y2])
                                labels.append(cls_id)
            
            # Resize image
            img = cv2.resize(img, (self.img_size, self.img_size))
            
            # Adjust boxes
            if boxes:
                boxes = np.array(boxes)
                scale_x = self.img_size / w_orig
                scale_y = self.img_size / h_orig
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
            else:
                boxes = np.zeros((0, 4))
                labels = []
            
            # Convert to tensors
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }
            
            return img_tensor, target
    
    def collate_fn(batch):
        images, targets = zip(*batch)
        return torch.stack(images, 0), list(targets)
    
    # Create dataset
    val_dataset = EvalDataset(
        img_dir=os.path.join(root, "images/val"),
        lbl_dir=os.path.join(root, "labels/val"),
        img_size=640
    )
    
    if len(val_dataset) == 0:
        print("ERROR: No validation images found!")
        return
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Initialize evaluator
    evaluator = SmallObjectEvaluator(model, device, nc)
    
    # Run evaluation
    print("Running evaluation...")
    for batch_idx, (images, targets) in enumerate(val_loader):
        print(f"Processing batch {batch_idx + 1}/{len(val_loader)}", end='\r')
        image_ids = [batch_idx * val_loader.batch_size + i for i in range(len(images))]
        evaluator.add_batch(images, targets, image_ids)
    
    print("\nEvaluation completed!")
    
    results = evaluator.compute_map()
    eval_dir = os.path.join(project_root, "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)
    evaluator.save_results_to_json(results, os.path.join(eval_dir, "evaluation_metrics.json"))
    
    # Print results
    evaluator.print_summary()
    
    # Generate plots
    os.makedirs(os.path.join(project_root, "evaluation_results"), exist_ok=True)
    
    try:
        evaluator.plot_pr_curve(
            class_id=0, 
            save_path=os.path.join(project_root, "evaluation_results/pr_curve_class_0.png")
        )
        print("PR curve saved to evaluation_results/pr_curve_class_0.png")
    except Exception as e:
        print(f"WARNING: Could not generate PR curve: {e}")
    

    # Additional analysis
    results = evaluator.compute_map()
    print("\nDETAILED RESULTS:")
    print(f"mAP@0.5: {results['mAP']:.4f}")
    
    total_detections = sum(len(evaluator.all_detections[i]) for i in range(nc))
    total_annotations = sum(evaluator.all_annotations.values())
    
    print(f"Total Detections: {total_detections}")
    print(f"Total Annotations: {total_annotations}")
    
    if total_detections > 10000:
        print("WARNING: Too many detections - model likely has issues!")
    
    if total_annotations == 0:
        print("WARNING: No ground truth found - check dataset!")
    
    return results


if __name__ == "__main__":
    run_complete_evaluation()

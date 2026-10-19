# train.py  — Multimodal YOLO11 training script
import os
import sys
import argparse

# Ensure local module files (block.py, head.py, etc.) take precedence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from multimodal_yolo11 import MultiModalYOLO11
from multimodal_dataset import MultiModalYoloDataset
from ultralytics.utils.loss import v8DetectionLoss
from balanced_sampler import ClassBalancedSampler, InstanceBalancedSampler

# ---------------------------------------------------------------------------
# Runtime API detection: ultralytics 8.3.x ships two different v8DetectionLoss
# implementations that share the same version string.
#   NEW API (local):  loss expects preds = {"boxes": ..., "scores": ..., "feats": [...]}
#   OLD API (server): loss expects preds = [feat_s, feat_m, feat_l]  (raw list)
# We detect which one is installed and adapt at runtime.
# ---------------------------------------------------------------------------
import inspect as _inspect
try:
    _LOSS_NEEDS_DICT = (
        '"boxes"' in _inspect.getsource(v8DetectionLoss)
        or "'boxes'" in _inspect.getsource(v8DetectionLoss)
    )
except Exception:
    _LOSS_NEEDS_DICT = False
print(f"[INFO] v8DetectionLoss API: {'dict (new)' if _LOSS_NEEDS_DICT else 'list (old)'}")

from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import csv
import time as time_module
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal YOLO11")
    parser.add_argument("--data_root",    type=str,   required=True,
                        help="Path to dataset root (must contain train/ and val/ splits)")
    parser.add_argument("--epochs",       type=int,   default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size",   type=int,   default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--img_size",     type=int,   default=416,
                        help="Base image size in pixels (default: 416)")
    parser.add_argument("--nc",           type=int,   default=10,
                        help="Number of classes (default: 10)")
    parser.add_argument("--fusion_type",  type=str,   default="cross_attn",
                        choices=["gated", "cross_attn", "gated_bilinear", "adapt_gated"],
                        help="Fusion module type (default: gated)")
    parser.add_argument("--workers",      type=int,   default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--pretrained",   type=str,   default=None,
                        help="Path to yolo11m.pt to initialise camera backbone (e.g. yolo11m.pt)")
    return parser.parse_args()

class MultiModalYOLOLoss:
    """
    Improved YOLO loss for multimodal fusion training.
    Computes box regression + objectness + classification loss.
    """
    def __init__(self, model, num_classes=10):
        """Initialize loss with model parameters."""
        self.device = next(model.parameters()).device
        self.num_classes = num_classes
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')

        # Loss weights (YOLOv8 style)
        self.lambda_box = 7.5     # box loss weight
        self.lambda_obj = 0.5     # objectness loss weight
        self.lambda_cls = 0.5     # classification loss weight

    def __call__(self, preds, targets_list):
        """
        Compute YOLO-style loss with proper scaling.

        Args:
            preds: List of 3 prediction tensors [batch, channels, grid_h, grid_w]
            targets_list: List of target tensors, each [N, 5] with [class, x, y, w, h]

        Returns:
            Total loss tensor with proper magnitude
        """
        device = self.device
        batch_size = len(targets_list)

        loss_box = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)

        # Count total number of objects in batch
        num_targets = sum([len(t) for t in targets_list])

        # Handle different prediction formats
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        # For proper training, we need losses in reasonable ranges (1-10 for YOLO)
        for pred in preds:
            # Skip if pred is not a tensor (e.g., nested list)
            if not isinstance(pred, torch.Tensor):
                continue

            # pred shape: [batch, channels, h, w] or [channels, h, w]
            if pred.ndim == 3:
                # Add batch dimension if missing
                pred = pred.unsqueeze(0)

            batch, channels, h, w = pred.shape

            # Create pseudo-targets on the grid
            # This is simplified - in real YOLO, you'd do proper anchor matching
            grid_size = h * w

            # Objectness loss: encourage network to predict low objectness everywhere
            # (since we're not doing proper matching yet)
            obj_target = torch.zeros(batch, 1, h, w, device=device)

            # If we have targets, create some positive samples
            if num_targets > 0:
                # For each target in batch, create positive anchors
                for b_idx, targets in enumerate(targets_list):
                    # Skip if b_idx exceeds actual batch size
                    if b_idx >= batch:
                        break

                    if len(targets) > 0:
                        for target in targets:
                            # target: [class, cx, cy, w, h] normalized
                            cx, cy = target[1].item(), target[2].item()

                            # Map to grid
                            grid_x = int(cx * w)
                            grid_y = int(cy * h)
                            grid_x = min(max(0, grid_x), w - 1)
                            grid_y = min(max(0, grid_y), h - 1)

                            # Set objectness target
                            obj_target[b_idx, 0, grid_y, grid_x] = 1.0

            # Extract objectness predictions (assume last channel)
            pred_obj = pred[:, -1:, :, :]

            # Objectness loss (BCE)
            loss_obj += self.bce_obj(pred_obj, obj_target).mean()

            # Box regression loss (L2 on all predictions)
            # In real YOLO, you'd only compute this for positive anchors
            loss_box += (pred[:, :4, :, :] ** 2).mean()

            # Classification loss (for channels representing classes)
            if channels > 5:  # If we have class prediction channels
                cls_channels = min(self.num_classes, channels - 5)
                if cls_channels > 0:
                    pred_cls = pred[:, 4:4+cls_channels, :, :]
                    cls_target = torch.zeros_like(pred_cls)

                    # Set class targets for positive samples
                    if num_targets > 0:
                        for b_idx, targets in enumerate(targets_list):
                            # Skip if b_idx exceeds actual batch size
                            if b_idx >= batch:
                                break

                            if len(targets) > 0:
                                for target in targets:
                                    cls_idx = int(target[0].item())
                                    cx, cy = target[1].item(), target[2].item()
                                    grid_x = int(cx * w)
                                    grid_y = int(cy * h)
                                    grid_x = min(max(0, grid_x), w - 1)
                                    grid_y = min(max(0, grid_y), h - 1)

                                    if cls_idx < cls_channels:
                                        cls_target[b_idx, cls_idx, grid_y, grid_x] = 1.0

                    loss_cls += self.bce_cls(pred_cls, cls_target).mean()

        # Scale losses to typical YOLO ranges
        num_layers = len(preds)
        loss_box = loss_box / num_layers
        loss_obj = loss_obj / num_layers
        loss_cls = loss_cls / num_layers

        # Total weighted loss
        total_loss = (
            self.lambda_box * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )

        return total_loss




# Add ultralytics to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ultralytics.utils.metrics import DetMetrics, box_iou, bbox_iou
from ultralytics.utils.ops import xywh2xyxy




def multimodal_collate(batch):
    """
    batch: list of (img_cam, img_lidar, labels, stem)
    returns: (imgs_cam_list, imgs_lidar_list, labels_list, stems_list)
    """
    imgs_cam, imgs_lidar, labels, stems = zip(*batch)
    return list(imgs_cam), list(imgs_lidar), list(labels), list(stems)


# def process_predictions_to_boxes(preds, img_size=416, conf_threshold=0.25, iou_threshold=0.45, debug=False):
def process_predictions_to_boxes(preds, img_size=512, conf_threshold=0.25, iou_threshold=0.7, debug=False):

    """
    Convert YOLO11 DFL predictions to bounding box format for metrics calculation.

    Args:
        preds: Model predictions from inference mode - tensor [batch, num_anchors, 4+nc]
               where first 4 values are bbox coords and last nc are class scores
        img_size: Image size (used for scaling)
        conf_threshold: Confidence threshold for filtering detections
        iou_threshold: IoU threshold for NMS (Non-Maximum Suppression)
        debug: If True, print debug information

    Returns:
        List of detection tensors per image in format [x1, y1, x2, y2, conf, cls] (normalized coordinates)
    """

    # preds should be a single tensor from inference mode: [batch, num_anchors, 4+nc]
    # If we get a list or training-mode format, we can't decode properly
    if isinstance(preds, (list, tuple)):
        if debug:
            print(f"    [DEBUG] Got list/tuple predictions (training mode) - cannot decode reliably")
            print(f"    [DEBUG] Number of items: {len(preds)}, types: {[type(p) for p in preds[:3]]}")
        # Return empty detections
        if len(preds) > 0 and isinstance(preds[0], torch.Tensor):
            batch_size = preds[0].shape[0]
            device = preds[0].device
            return [torch.zeros((0, 6), device=device) for _ in range(batch_size)]
        return []

    # Single tensor from inference mode
    if not isinstance(preds, torch.Tensor):
        return []

    # Expected format: [batch, num_anchors, 4+nc] where first 4 are [x, y, w, h] and last nc are class scores
    if preds.ndim != 3:
        if debug:
            print(f"    [DEBUG] Unexpected prediction shape: {preds.shape}, expected 3D tensor")
        batch_size = preds.shape[0] if preds.ndim > 0 else 1
        device = preds.device
        return [torch.zeros((0, 6), device=device) for _ in range(batch_size)]

    batch_size, channels, num_anchors = preds.shape
    device = preds.device

    nc = 10  # Number of classes

    if debug:
        print(f"    [DEBUG] Inference predictions shape: {preds.shape}")
        
    # Transpose from [batch, channels, num_anchors] to [batch, num_anchors, channels]
    preds = preds.permute(0, 2, 1)  # Now [batch, num_anchors, channels]
    # Split into boxes and scores
    boxes = preds[..., :4]  # [batch, num_anchors, 4] - [x, y, w, h] in PIXEL coordinates from YOLO11
    scores = preds[..., 4:]  # [batch, num_anchors, nc]

    # Get class confidences and indices
    cls_conf, cls_idx = scores.max(dim=-1)  # [batch, num_anchors]

    # YOLO11 Detect head outputs boxes in PIXEL coordinates, normalize to [0,1]
    boxes = boxes / img_size

    # Convert xywh to xyxy (boxes are now normalized)
    boxes_xyxy = xywh2xyxy(boxes)

    # Clamp to valid range [0, 1]
    boxes_xyxy = torch.clamp(boxes_xyxy, 0.0, 1.0)


    # Filter by confidence threshold
    result = []
    for b_idx in range(batch_size):
        mask = cls_conf[b_idx] >= conf_threshold

        if mask.sum() == 0:
            result.append(torch.zeros((0, 6), device=device))
            continue

        # Extract detections above threshold
        boxes_b = boxes_xyxy[b_idx][mask]  # [N, 4]
        confs = cls_conf[b_idx][mask]  # [N]
        classes = cls_idx[b_idx][mask]  # [N]

        # Concatenate: [x1, y1, x2, y2, conf, cls]
        detections = torch.cat([
            boxes_b,
            confs.unsqueeze(1),
            classes.unsqueeze(1).float()
        ], dim=1)

        # Apply NMS per class
        if len(detections) > 0:
            nms_detections = []
            unique_classes = classes.unique()

            for cls in unique_classes:
                cls_mask = detections[:, 5] == cls
                cls_dets = detections[cls_mask]

                if len(cls_dets) == 0:
                    continue

                # NMS: keep boxes with IoU < iou_threshold
                boxes_nms = cls_dets[:, :4]
                scores_nms = cls_dets[:, 4]

                # Sort by confidence (descending)
                sorted_indices = torch.argsort(scores_nms, descending=True)

                keep = []
                while len(sorted_indices) > 0:
                    # Keep the box with highest confidence
                    idx = sorted_indices[0]
                    keep.append(idx.item())

                    if len(sorted_indices) == 1:
                        break

                    # Compute IoU with remaining boxes
                    current_box = boxes_nms[idx].unsqueeze(0)
                    remaining_boxes = boxes_nms[sorted_indices[1:]]

                    # Compute IoU
                    iou = box_iou(current_box, remaining_boxes).squeeze(0)

                    # Keep boxes with IoU < threshold
                    keep_mask = iou <= iou_threshold
                    sorted_indices = sorted_indices[1:][keep_mask]

                # Keep selected detections
                if len(keep) > 0:
                    nms_detections.append(cls_dets[keep])

            # Combine detections from all classes
            if len(nms_detections) > 0:
                detections = torch.cat(nms_detections, dim=0)
            else:
                detections = torch.zeros((0, 6), device=device)

        result.append(detections)

    # if debug:
    #     total_dets = sum(len(d) for d in result)
    #     max_conf = max([d[:, 4].max().item() if len(d) > 0 else 0.0 for d in result])
    #     print(f"    [DEBUG] Decoded {total_dets} detections, max confidence: {max_conf:.4f}")
    if debug:
        total_dets = sum(len(d) for d in result)
        max_conf = max([d[:, 4].max().item() if len(d) > 0 else 0.0 for d in result])
        min_conf = min([d[:, 4].min().item() if len(d) > 0 else 1.0 for d in result])
        print(f"    [DEBUG] After NMS: {total_dets} detections (conf_thresh={conf_threshold:.3f}, iou_thresh={iou_threshold:.2f}), conf range: [{min_conf:.4f}, {max_conf:.4f}]")


    return result


# def compute_metrics(preds, targets_list, iou_threshold=0.05, conf_threshold=0.25, nms_iou_threshold=0.45, num_classes=10):
def compute_metrics(preds, targets_list, img_size=512, iou_threshold=0.05, conf_threshold=0.25, nms_iou_threshold=0.7, num_classes=10):
    
    """
    Compute detection metrics using Ultralytics metrics API.

    Args:
        preds: Model predictions - list of inference tensors, each [batch, num_anchors, 4+nc]
        targets_list: Ground truth labels [N, 5] format: [class, x, y, w, h]
        iou_threshold: IoU threshold for matching predictions to targets
        conf_threshold: Confidence threshold for filtering detections
        nms_iou_threshold: IoU threshold for NMS (Non-Maximum Suppression)
        num_classes: Number of classes

    Returns:
        Dictionary with metrics
    """
    # Initialize DetMetrics
    class_names = {i: f'class{i}' for i in range(num_classes)}
    det_metrics = DetMetrics(names=class_names)

    # Process each batch of predictions
    all_detections = []
    for pred_batch in preds:
        # Process this batch to get detections with NMS
        # batch_dets = process_predictions_to_boxes(pred_batch, conf_threshold=conf_threshold, iou_threshold=nms_iou_threshold, debug=False)
        batch_dets = process_predictions_to_boxes(pred_batch, img_size=img_size, conf_threshold=conf_threshold, iou_threshold=nms_iou_threshold, debug=False)
        all_detections.extend(batch_dets)  # Flatten into per-image detections

    detections = all_detections

    # Debug: count total detections
    total_dets = sum(len(d) for d in detections)
    total_targets = sum(len(t) for t in targets_list)
    print(f"    Detections: {total_dets}, Targets: {total_targets}, Conf threshold: {conf_threshold}")

    # Match predictions to ground truth for each image in batch
    for img_idx, (det, targets) in enumerate(zip(detections, targets_list)):
        # Skip if no predictions or no targets
        if len(det) == 0:
            pred_cls = np.zeros(0)
            pred_conf = np.zeros(0)
            tp = np.zeros((0, 10), dtype=bool)  # 10 IoU thresholds
        else:
            pred_boxes = det[:, :4]  # [x1, y1, x2, y2]
            pred_conf = det[:, 4].cpu().numpy()
            pred_cls = det[:, 5].cpu().numpy()

            # Compute TP if we have targets
            if len(targets) > 0:
                # Convert targets from [class, cx, cy, w, h] to [x1, y1, x2, y2]
                target_boxes = targets[:, 1:5].clone()  # [cx, cy, w, h]
                target_cls = targets[:, 0].cpu().numpy()

                # Convert normalized xywh to xyxy
                target_boxes_xyxy = torch.zeros_like(target_boxes)
                target_boxes_xyxy[:, 0] = target_boxes[:, 0] - target_boxes[:, 2] / 2  # x1
                target_boxes_xyxy[:, 1] = target_boxes[:, 1] - target_boxes[:, 3] / 2  # y1
                target_boxes_xyxy[:, 2] = target_boxes[:, 0] + target_boxes[:, 2] / 2  # x2
                target_boxes_xyxy[:, 3] = target_boxes[:, 1] + target_boxes[:, 3] / 2  # y2

                # Ensure targets are on the same device as predictions
                target_boxes_xyxy = target_boxes_xyxy.to(pred_boxes.device)



                # Compute IoU between predictions and targets
                iou = box_iou(target_boxes_xyxy, pred_boxes)

                # Match predictions to targets at multiple IoU thresholds
                tp = np.zeros((len(pred_cls), 10), dtype=bool)
                iou_thresholds = np.linspace(0.5, 0.95, 10)

                if iou.numel() > 0:
                    for iou_idx, iou_thresh in enumerate(iou_thresholds):
                        # Simple greedy matching
                        matched_targets = set()
                        for pred_idx in range(len(pred_cls)):
                            best_iou = 0
                            best_target = -1
                            for target_idx in range(len(target_cls)):
                                if target_idx in matched_targets:
                                    continue
                                if pred_cls[pred_idx] == target_cls[target_idx]:
                                    iou_val = iou[target_idx, pred_idx].item()
                                    if iou_val > best_iou and iou_val >= iou_thresh:
                                        best_iou = iou_val
                                        best_target = target_idx

                            if best_target >= 0:
                                tp[pred_idx, iou_idx] = True
                                matched_targets.add(best_target)
            else:
                target_cls = np.zeros(0)
                tp = np.zeros((len(pred_cls), 10), dtype=bool)

        # Update statistics
        if len(targets) > 0:
            target_cls_array = targets[:, 0].cpu().numpy().astype(int)
        else:
            target_cls_array = np.zeros(0, dtype=int)

        det_metrics.update_stats({
            'tp': tp,
            'conf': pred_conf,
            'pred_cls': pred_cls,
            'target_cls': target_cls_array,
            'target_img': np.array([img_idx] * len(target_cls_array)),
            'im_name': str(img_idx),
        })

    # Process accumulated statistics to get final metrics
    # Check if we have any statistics to process
    if len(det_metrics.stats['tp']) == 0:
        # No detections or targets - return zero metrics
        metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'mAP': 0.0,
            'f1': 0.0
        }
        return metrics

    det_metrics.process()

    # Get mean results: [precision, recall, mAP50, mAP50-95]
    results = det_metrics.mean_results()

    metrics = {
        'precision': results[0],
        'recall': results[1],
        'mAP50': results[2],
        'mAP': results[3],
        'f1': 2 * results[0] * results[1] / (results[0] + results[1] + 1e-6)
    }

    return metrics



# def validate(model, val_loader, loss_fn, device, compute_detailed_metrics=False):
def validate(model, val_loader, loss_fn, device, img_size=512, compute_detailed_metrics=False):

    """
    Run validation on the validation set.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        loss_fn: Loss function (now assumed to be v8DetectionLoss)
        device: Device to run on
        compute_detailed_metrics: Whether to compute mAP, precision, recall

    Returns:
        Dictionary with validation metrics
    """
    from torch.cuda.amp import autocast

    model.eval()
    total_val_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        for imgs_cam, imgs_lidar, labels, _ in val_pbar:
            imgs_cam = torch.stack(imgs_cam).to(device, non_blocking=True)
            imgs_lidar = torch.stack(imgs_lidar).to(device, non_blocking=True)

            # Convert labels to dict format expected by v8DetectionLoss
            batch_indices = []
            classes = []
            bboxes = []

            for i, target in enumerate(labels):
                if len(target) > 0:
                    num_targets = target.shape[0]
                    batch_indices.append(torch.full((num_targets,), i, dtype=torch.float32))
                    classes.append(target[:, 0])
                    bboxes.append(target[:, 1:5])

            if len(batch_indices) > 0:
                batch_dict = {
                    "batch_idx": torch.cat(batch_indices).to(device).float(),  # FP32
                    "cls": torch.cat(classes).to(device).float(),  # FP32
                    "bboxes": torch.cat(bboxes).to(device).float()  # FP32
                }
            else:
                batch_dict = {
                    "batch_idx": torch.zeros((0,), dtype=torch.float32, device=device),
                    "cls": torch.zeros((0,), dtype=torch.float32, device=device),
                    "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device)
                }

            # Use AMP for faster validation (FP16 for speed, but ensure FP32 for loss)
            use_amp_val = (device == 'cuda')
            with autocast(enabled=use_amp_val):
                # Step 1: Get training-mode predictions for loss calculation
                # Force training mode for Detect head to get raw predictions
                model.head.detect.training = True
                preds_train = model(imgs_cam, imgs_lidar)

                # Temporarily disable AMP for loss computation (numerical stability)
                with autocast(enabled=False):
                    # Ensure predictions are in FP32 and adapt to installed loss API
                    if isinstance(preds_train, dict):
                        if _LOSS_NEEDS_DICT:
                            preds_train = {
                                "boxes": preds_train["boxes"].float(),
                                "scores": preds_train["scores"].float(),
                                "feats": [f.float() for f in preds_train["feats"]],
                            }
                        else:
                            # Old API: loss expects raw list of feature tensors
                            preds_train = [f.float() for f in preds_train["feats"]]
                    elif isinstance(preds_train, (list, tuple)):
                        preds_train = [p.float() for p in preds_train]
                    else:
                        preds_train = preds_train.float()

                    # Compute loss in FP32 for accurate validation metrics
                    # v8DetectionLoss returns (total_loss_tensor, detached_loss_items_tuple)
                    loss, loss_items = loss_fn(preds_train, batch_dict)
                    loss = loss.sum()  # Ensure scalar

            total_val_loss += loss.item()
            num_batches += 1

            # Step 2: Get inference-mode predictions for metrics (if needed)
            if compute_detailed_metrics:
                with autocast(enabled=use_amp_val):
                    # Set the Detect head to inference mode to get decoded predictions
                    model.head.detect.training = False
                    preds_inference = model(imgs_cam, imgs_lidar)

                    # preds_inference should now be in inference format: [batch, num_anchors, 4+nc]
                    # or tuple (decoded_preds, raw_preds) - we want the decoded part
                    if isinstance(preds_inference, tuple):
                        preds_inference = preds_inference[0]  # Get decoded predictions

                    # Ensure inference predictions are in FP32 for accurate metrics
                    if isinstance(preds_inference, (list, tuple)):
                        preds_inference = [p.float() for p in preds_inference]
                    else:
                        preds_inference = preds_inference.float()

                all_preds.append(preds_inference)
                all_targets.extend(labels)

            val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})


    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0.0

    results = {'loss': avg_val_loss}

    if compute_detailed_metrics and len(all_targets) > 0:
        # Compute detection metrics
        # NOTE: all_preds is now a flat list of tensors, which compute_metrics expects
        # metrics = compute_metrics(all_preds, all_targets)
        metrics = compute_metrics(all_preds, all_targets, img_size=img_size)
        results.update(metrics)

    return results


def plot_training_curves(history, save_path="runs_multimodal/training_curves.png"):
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16)

    # Loss curves (with log scale for better visualization of large initial losses)
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_yscale('log')  # Log scale to see early progress better
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, which='both')

    # Precision
    if 'precision' in history and len(history['precision']) > 0:
        axes[0, 1].plot(epochs, history['precision'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision')
        axes[0, 1].grid(True, alpha=0.3)

    # Recall
    if 'recall' in history and len(history['recall']) > 0:
        axes[0, 2].plot(epochs, history['recall'], 'm-', linewidth=2)
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].set_title('Recall')
        axes[0, 2].grid(True, alpha=0.3)

    # F1 Score
    if 'f1' in history and len(history['f1']) > 0:
        axes[1, 0].plot(epochs, history['f1'], 'c-', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].grid(True, alpha=0.3)

    # mAP@0.5
    if 'mAP50' in history and len(history['mAP50']) > 0:
        axes[1, 1].plot(epochs, history['mAP50'], 'orange', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('mAP@0.5')
        axes[1, 1].set_title('mAP@0.5')
        axes[1, 1].grid(True, alpha=0.3)

    # mAP@0.5:0.95
    if 'mAP' in history and len(history['mAP']) > 0:
        axes[1, 2].plot(epochs, history['mAP'], 'purple', linewidth=2)
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('mAP@0.5:0.95')
        axes[1, 2].set_title('mAP@0.5:0.95')
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Training curves saved to: {save_path}")
    plt.close()





def main(args=None):
    if args is None:
        args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Dataset & dataloader
    data_root  = args.data_root
    num_epochs = args.epochs
    nc         = args.nc
    img_size   = args.img_size

    # ============================================================================
    # MULTI-SCALE TRAINING CONFIGURATION
    # ============================================================================
    use_multi_scale = True  # Enable multi-scale training with on-the-fly resizing

    if use_multi_scale:
        # Define size range (must be multiples of stride=32)
        img_size_min = 320   # Minimum size
        img_size_max = 640   # Maximum size (reduced from 640 to prevent OOM)
        img_size_step = 32   # Step size (YOLO stride)

        # Calculate all possible sizes
        img_sizes = list(range(img_size_min, img_size_max + img_size_step, img_size_step))
        # Result: [320, 352, 384, 416, 448, 480, 512]

        # How often to change size (in batches)
        scale_change_frequency = 10  # Change size every 10 batches

        print(f"\n{'='*70}")
        print("MULTI-SCALE TRAINING ENABLED (On-the-fly Resizing)")
        print(f"{'='*70}")
        print(f"Size range: {img_size_min} - {img_size_max}")
        print(f"Possible sizes: {img_sizes}")
        print(f"Change frequency: Every {scale_change_frequency} batches")
        print(f"Strategy: Resize images on-the-fly (no dataloader recreation)")
        print(f"{'='*70}\n")

        current_img_size = img_size  # Start with base size
    else:
        current_img_size = img_size
        img_sizes = [img_size]
        print(f"\n⚠️  Multi-scale training DISABLED - using fixed size {img_size}×{img_size}")
    # ============================================================================

    # OPTIMIZATION: For quick testing, use only a subset of data
    use_subset = False  # Set to False to use full dataset
    subset_size = 1000  # Number of samples to use for quick testing

    # ============================================================================
    # CLASS BALANCING CONFIGURATION
    # ============================================================================
    # Set to True to enable oversampling of minority classes
    use_class_balancing = True

    # Sampler options:
    # - 'weighted': ClassBalancedSampler with weighted sampling (recommended)
    # - 'repeat': InstanceBalancedSampler with repetition (more aggressive)
    # - None: No balancing (standard random sampling)
    sampler_type = 'weighted'  # Options: 'weighted', 'repeat', None

    # Balance mode (for 'weighted' sampler):
    # - 'sqrt': Gentle balancing (increases minority class sampling by sqrt of imbalance)
    # - 'full': Aggressive balancing (fully equalizes class distribution)
    balance_mode = 'sqrt'  # Recommended: 'sqrt' for object detection

    # Oversample threshold: Classes with < threshold * mean are considered minority
    oversample_threshold = 0.5  # Classes with <50% of mean count get oversampled

    # Target balance ratio (for 'repeat' sampler): 0.5 = minority classes reach 50% of max
    target_balance_ratio = 0.5
    # ============================================================================
    # ============================================================================
    # NOTE: Multi-scale training now uses on-the-fly image resizing instead of
    # recreating dataloaders. This allows epochs to complete properly without
    # memory leaks or epoch restarts.
    # ============================================================================

    # AUGMENTATION CONTROL: Set augment=True for reduced augmentation, augment=False to disable completely
    train_ds = MultiModalYoloDataset(data_root, split="train", img_size=current_img_size, cache_images=False, augment=True)
    
    # Create subset if requested
    if use_subset and len(train_ds) > subset_size:
        print(f"Using subset of {subset_size} samples (out of {len(train_ds)}) for faster training")
        from torch.utils.data import Subset
        indices = list(range(subset_size))
        train_ds = Subset(train_ds, indices)

    # Validation dataset
    print("\nLoading validation dataset...")
    val_ds = MultiModalYoloDataset(data_root, split="val", img_size=img_size, cache_images=False, augment=False)
    if use_subset and len(val_ds) > 200:
        print(f"Using subset of 200 validation samples (out of {len(val_ds)})")
        val_indices = list(range(200))
        val_ds = Subset(val_ds, val_indices)

    # For Windows, start with num_workers=0 or use spawn context
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    # OPTIMIZATION: Reduce batch size for multi-scale (simpler than gradient accumulation)
    # With multi-scale training, we use a smaller batch that fits all sizes
    base_batch_size = 4  # Reduced from 8 to fit larger image sizes
    batch_size = base_batch_size

    # ============================================================================
    # CREATE TRAIN SAMPLER (with optional class balancing)
    # ============================================================================
    train_sampler = None
    shuffle_data = True  # Default shuffle

    if use_class_balancing and sampler_type:
        print(f"\n{'='*70}")
        print("ENABLING CLASS BALANCING")
        print(f"{'='*70}")

        if sampler_type == 'weighted':
            train_sampler = ClassBalancedSampler(
                train_ds,
                data_root=data_root,
                split="train",
                oversample_threshold=oversample_threshold,
                balance_mode=balance_mode
            )
            shuffle_data = False  # Sampler handles randomization
            print(f"✅ Using ClassBalancedSampler (mode={balance_mode})")

        elif sampler_type == 'repeat':
            train_sampler = InstanceBalancedSampler(
                train_ds,
                data_root=data_root,
                split="train",
                target_balance_ratio=target_balance_ratio
            )
            shuffle_data = False  # Sampler handles randomization
            print(f"✅ Using InstanceBalancedSampler (ratio={target_balance_ratio})")

        else:
            print(f"⚠️  Unknown sampler_type: {sampler_type}, using default shuffle")
    else:
        print("\n⚠️  Class balancing disabled - using standard random sampling")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_data,  # Disable shuffle if using custom sampler
        sampler=train_sampler,  # Use custom sampler if provided
        num_workers=args.workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,  # ✅ Keep workers alive between epochs
        collate_fn=multimodal_collate,
        drop_last=True,         # prevent batch_size=1 at epoch end (BatchNorm crash)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,  # ✅ Keep workers alive between epochs
        collate_fn=multimodal_collate,
    )

    print(f"DataLoader ready: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print(f"Training config: img_size={img_size}, batch_size={batch_size}")


    # 2. Model
    model = MultiModalYOLO11(
        nc=nc,
        cam_in_ch=3,
        lidar_in_ch=3,
        fusion_type=args.fusion_type,
    ).to(device)

    # Load pretrained camera backbone weights from yolo11m.pt (if provided)
    if args.pretrained:
        print(f"\n[INFO] Loading pretrained backbone from: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        # yolo11m.pt stores the full YOLO model; extract its state dict
        if isinstance(ckpt, dict) and "model" in ckpt:
            src_sd = ckpt["model"].float().state_dict()
        elif hasattr(ckpt, "state_dict"):
            src_sd = ckpt.float().state_dict()
        else:
            src_sd = ckpt
        # Map yolo11m backbone keys (model.0..model.10) -> our backbone_cam keys
        # yolo11m backbone layers: model.0=Conv P1, model.1=Conv P2, model.2=C3k2,
        #   model.3=Conv P3, model.4=C3k2, model.5=Conv P4, model.6=C3k2,
        #   model.7=C3k2, model.8=Conv P5, model.9=C3k2, model.10=SPPF
        YOLO_TO_OURS = {
            "model.0.": "backbone_cam.layer0.",
            "model.1.": "backbone_cam.layer1.",
            "model.2.": "backbone_cam.layer2.",
            "model.3.": "backbone_cam.layer3.",
            "model.4.": "backbone_cam.layer4.",
            "model.5.": "backbone_cam.layer5.",
            "model.6.": "backbone_cam.layer6.",
            "model.7.": "backbone_cam.layer7.",
            "model.8.": "backbone_cam.layer8.",
            "model.9.": "backbone_cam.layer9.",
            "model.10.": "backbone_cam.layer10.",
        }
        mapped, skipped = {}, 0
        for src_key, val in src_sd.items():
            for prefix, new_prefix in YOLO_TO_OURS.items():
                if src_key.startswith(prefix):
                    mapped[src_key.replace(prefix, new_prefix, 1)] = val
                    break
            else:
                skipped += 1
        # Filter shape mismatches first — strict=False still crashes on size mismatches
        model_sd = model.state_dict()
        shape_ok = {k: v for k, v in mapped.items()
                    if k in model_sd and model_sd[k].shape == v.shape}
        shape_skip = len(mapped) - len(shape_ok)

        result = model.load_state_dict(shape_ok, strict=False)
        loaded = len(shape_ok) - len(result.missing_keys)
        print(f"   Backbone keys loaded   : {loaded}")
        print(f"   Shape mismatch skipped : {shape_skip}  ← if >0, wrong .pt variant")
        print(f"   Missing (random init)  : {len(result.missing_keys)}")
        print(f"   Unexpected (ignored)   : {len(result.unexpected_keys)}")
        print(f"   Skipped (non-backbone) : {skipped}")
        if loaded == 0:
            print(f"\n   [WARN] 0 backbone weights loaded.")
            print(f"          Your backbone uses 1.5x channels (96/192/768/1536).")
            print(f"          yolo11m.pt uses 0.5x (64/128/512). Try --pretrained yolo11l.pt")

    # 3. Loss function (Final Fix: Attach attributes to the full model for v8DetectionLoss)
    
    # 1. Attach model.model property
    # The loss function expects model.model[-1] to be the Detect layer (model.head).
    setattr(model, 'model', [model.head])

    # model.summary()
    # 2. Attach required attributes (stride, nc, reg_max) to the Detect layer (model.head)
    # Ensure device is explicitly converted to torch.device for torch.tensor() stability
    current_device = torch.device(device)

    # CRITICAL FIX: stride must be set on BOTH the head (for loss) and detect layer (for inference)!
    stride_tensor = torch.tensor([8., 16., 32.], device=current_device)
    setattr(model.head, 'stride', stride_tensor)  # For v8DetectionLoss
    setattr(model.head.detect, 'stride', stride_tensor)  # For Detect inference
    setattr(model.head, 'nc', nc)
    setattr(model.head, 'reg_max', 16)  # Standard DFL parameter
    setattr(model.head.detect, 'nc', nc)
    setattr(model.head.detect, 'reg_max', 16)

    # 3. Attach the mock 'args' (hyperparameters) attribute to the model instance
    class Args:
        def __init__(self):
            self.box = 7.5
            self.cls = 0.5
            self.dfl = 1.5
            self.gr = 1.0
            self.iou_loss = 'bbox_iou'
            self.cls_pw = 1.0
            self.obj_pw = 1.0

    model.args = Args()     

    # Initialize the loss, passing the full model instance
    # The loss function will now find: m = model.model[-1] == model.head
    loss_fn = v8DetectionLoss(model) # Pass the full model

    # NOTE: increase the weight_decay to =1e-5 or =3e-4 if the validation loss is more then training
    # Lion is other optimizer that can be use instead of AdamW
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # eps=1e-3 prevents the denominator from becoming zero in float16
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, eps=1e-5,weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3) #mAP 62% but train - val loss 10


    # ✅ SPEEDUP: Enable automatic mixed precision (1.5-2x faster!)
    from torch.cuda.amp import GradScaler, autocast
    use_amp = True if device == "cuda" else False
    scaler = GradScaler(enabled=use_amp)
    print(f"Mixed precision training: {'enabled ✅' if use_amp else 'disabled'}")



    # Training history for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mAP50': [],
        'mAP': []
    }

    # Track best model
    best_map50 = 0.0
    best_epoch = 0

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    # ============================================================================
    # CSV LOGGING SETUP
    # ============================================================================
    csv_log_path = Path("runs_multimodal") / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create CSV file and write header
    with open(csv_log_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            'Epoch', 'Train_Loss', 'Val_Loss', 'Precision', 'Recall',
            'F1_Score', 'mAP@0.5', 'mAP@0.5:0.95',
            'Train_Time_sec', 'Val_Time_sec', 'Total_Time_sec', 'Timestamp'
        ])

    print(f"📊 CSV logging enabled: {csv_log_path}")
    # ============================================================================

    for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            loss = 0.0
            print(f"\nStarting epoch {epoch+1}/{num_epochs}...")
            import time
            import random
            epoch_start = time.time()

            # Clear CUDA cache to avoid fragmentation at epoch start
            if device == "cuda":
                torch.cuda.empty_cache()

            # for batch_idx, (imgs_cam, imgs_lidar, labels, stems) in enumerate(train_loader):
            train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                  desc=f"Epoch {epoch+1}/{num_epochs} [size={current_img_size}]",
                  bar_format='{l_bar}{bar:30}{r_bar}')

            for batch_idx, (imgs_cam, imgs_lidar, labels, stems) in train_pbar:
                batch_start = time.time()

                # ============================================================================
                # MULTI-SCALE: Change image size every N batches (on-the-fly resizing)
                # ============================================================================
                if use_multi_scale and batch_idx > 0 and batch_idx % scale_change_frequency == 0:
                    new_size = random.choice(img_sizes)

                    if new_size != current_img_size:
                        current_img_size = new_size

                        # Clear CUDA cache when changing to larger size
                        if device == "cuda":
                            torch.cuda.empty_cache()

                        train_pbar.set_description(f"Epoch {epoch+1}/{num_epochs} [size={current_img_size}]")
                # ============================================================================

                # ✅ SPEEDUP: Use non_blocking=True for faster GPU transfer
                imgs_cam = torch.stack(imgs_cam).to(device, non_blocking=True)      # [B,3,H,W]
                imgs_lidar = torch.stack(imgs_lidar).to(device, non_blocking=True)  # [B,C,H,W]

                # ============================================================================
                # MULTI-SCALE: Resize images on-the-fly if size changed
                # ============================================================================
                if use_multi_scale:
                    # Get current image size from batch
                    _, _, H, W = imgs_cam.shape

                    # Resize if needed
                    if H != current_img_size or W != current_img_size:
                        imgs_cam = torch.nn.functional.interpolate(
                            imgs_cam,
                            size=(current_img_size, current_img_size),
                            mode='bilinear',
                            align_corners=False
                        )
                        imgs_lidar = torch.nn.functional.interpolate(
                            imgs_lidar,
                            size=(current_img_size, current_img_size),
                            mode='bilinear',
                            align_corners=False
                        )
                # ============================================================================

                # -------------------------- LOSS COMPUTATION --------------------------
                batch_indices = []
                classes = []
                bboxes = []

                for i, target in enumerate(labels):
                    if len(target) > 0:
                        # target format: [class, cx, cy, w, h]
                        num_targets = target.shape[0]
                        batch_indices.append(torch.full((num_targets,), i, dtype=torch.float32))
                        classes.append(target[:, 0])  # class column
                        bboxes.append(target[:, 1:5])  # [cx, cy, w, h] columns

                if len(batch_indices) > 0:
                    batch_dict = {
                        "batch_idx": torch.cat(batch_indices).to(device),
                        "cls": torch.cat(classes).to(device),
                        "bboxes": torch.cat(bboxes).to(device)
                    }
                else:
                    # Empty batch
                    batch_dict = {
                        "batch_idx": torch.zeros((0,), dtype=torch.float32, device=device),
                        "cls": torch.zeros((0,), dtype=torch.float32, device=device),
                        "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=device)
                    }

                # ✅ SPEEDUP: Forward pass with mixed precision, loss in FP32 for numerical stability
                with autocast(enabled=use_amp):
                    # FAST PART: Model layers run in FP16 (High Speed)
                    preds = model(imgs_cam, imgs_lidar)

                    # SAFE PART: Temporarily disable AMP for the Loss Calculation
                    with autocast(enabled=False):
                        # Cast predictions to float32 and adapt to installed loss API
                        if isinstance(preds, dict):
                            if _LOSS_NEEDS_DICT:
                                # New API: pass dict directly
                                preds = {
                                    "boxes": preds["boxes"].float(),
                                    "scores": preds["scores"].float(),
                                    "feats": [f.float() for f in preds["feats"]],
                                }
                            else:
                                # Old API: loss expects raw list of feature tensors
                                preds = [f.float() for f in preds["feats"]]
                        elif isinstance(preds, (list, tuple)):
                            preds = [p.float() for p in preds]
                        else:
                            preds = preds.float()

                        # Ensure targets are also in FP32
                        batch_dict_fp32 = {
                            "batch_idx": batch_dict["batch_idx"].float(),
                            "cls": batch_dict["cls"].float(),
                            "bboxes": batch_dict["bboxes"].float()
                        }

                        # v8DetectionLoss returns (total_loss_tensor, detached_loss_items_tuple)
                        loss, loss_items = loss_fn(preds, batch_dict_fp32)
                        loss = loss.sum()  # Ensure scalar

                # ✅ SPEEDUP: Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                # Progress reporting
                # if batch_idx % 50 == 0:
                #     elapsed = time.time() - epoch_start
                #     batches_done = batch_idx + 1
                #     batches_left = len(train_loader) - batches_done
                #     eta_seconds = (elapsed / batches_done) * batches_left if batches_done > 0 else 0
                #     eta_minutes = eta_seconds / 60
                #     print(f"   [{batch_idx:4d}/{len(train_loader)}] loss: {loss.item():.4f} | "
                #         f"time/batch: {(time.time()-batch_start):.2f}s | ETA: {eta_minutes:.1f}min")
                train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                    })

            avg_train_loss = total_loss / len(train_loader)
            train_time = time.time() - epoch_start

            # Validation with metrics
            print(f"   Running validation...")
            val_start = time.time()

            # NOTE: The 'validate' function also needs to be updated with the target stacking
            # and the prediction flattening logic for all_preds.
            compute_metrics_now = (epoch + 1) % 1 == 0 or (epoch + 1) == num_epochs
            # val_results = validate(model, val_loader, loss_fn, device, compute_detailed_metrics=compute_metrics_now)
            val_results = validate(model, val_loader, loss_fn, device, img_size=img_size, compute_detailed_metrics=compute_metrics_now)

            val_time = time.time() - val_start
            total_epoch_time = time.time() - epoch_start

            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_results['loss'])
            history['precision'].append(val_results.get('precision', 0.0))
            history['recall'].append(val_results.get('recall', 0.0))
            history['f1'].append(val_results.get('f1', 0.0))
            history['mAP50'].append(val_results.get('mAP50', 0.0))
            history['mAP'].append(val_results.get('mAP', 0.0))

            # Print summary
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs} Summary:")
            print(f"{'='*60}")
            print(f"   Train Loss:  {avg_train_loss:.4f}")
            print(f"   Val Loss:    {val_results['loss']:.4f}")
            print(f"   Precision:   {val_results.get('precision', 0.0):.4f}")
            print(f"   Recall:      {val_results.get('recall', 0.0):.4f}")
            print(f"   F1 Score:    {val_results.get('f1', 0.0):.4f}")
            print(f"   mAP@0.5:     {val_results.get('mAP50', 0.0):.4f}")
            print(f"   mAP@0.5:0.95: {val_results.get('mAP', 0.0):.4f}")
            print(f"   Train Time:  {train_time:.1f}s ({train_time/60:.1f}min)")
            print(f"   Val Time:    {val_time:.1f}s ({val_time/60:.1f}min)")
            print(f"   Total Time:  {total_epoch_time:.1f}s ({total_epoch_time/60:.1f}min)")
            print(f"{'='*60}")

            # ============================================================================
            # LOG TO CSV
            # ============================================================================
            with open(csv_log_path, 'a', newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    epoch + 1,
                    f"{avg_train_loss:.6f}",
                    f"{val_results['loss']:.6f}",
                    f"{val_results.get('precision', 0.0):.6f}",
                    f"{val_results.get('recall', 0.0):.6f}",
                    f"{val_results.get('f1', 0.0):.6f}",
                    f"{val_results.get('mAP50', 0.0):.6f}",
                    f"{val_results.get('mAP', 0.0):.6f}",
                    f"{train_time:.2f}",
                    f"{val_time:.2f}",
                    f"{total_epoch_time:.2f}",
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ])
            # ============================================================================

            # Save best model (based on mAP50)
            current_map50 = val_results.get('mAP50', 0.0)
            if current_map50 > best_map50:
                best_map50 = current_map50
                best_epoch = epoch + 1
                ckpt_path = Path("runs_multimodal") / "best.pt"
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_results': val_results,
                    'history': history,
                }, ckpt_path)
                print(f"   Best model saved: {ckpt_path}  (mAP50={best_map50:.4f}, epoch {best_epoch})")

        # Plot training curves at the end
        
        
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    plot_training_curves(history)

    # Save final results summary
    results_path = Path("runs_multimodal") / "results.txt"
    with open(results_path, 'w') as f:
        f.write("Training Results Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total Epochs: {num_epochs}\n")
        f.write(f"Image Size: {img_size}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Dataset: {len(train_ds)} train, {len(val_ds)} val\n\n")
        f.write("Final Metrics:\n")
        f.write(f"  Train Loss:   {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Val Loss:     {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Precision:    {history['precision'][-1]:.4f}\n")
        f.write(f"  Recall:       {history['recall'][-1]:.4f}\n")
        f.write(f"  F1 Score:     {history['f1'][-1]:.4f}\n")
        f.write(f"  mAP@0.5:      {history['mAP50'][-1]:.4f}\n")
        f.write(f"  mAP@0.5:0.95: {history['mAP'][-1]:.4f}\n")

    print(f"  Results saved to: {results_path}")
    print(f"  📊 CSV training log saved to: {csv_log_path}")


if __name__ == "__main__":
    main(parse_args())




# NOTE: increase the weight_decay to =1e-5 or =3e-4 if the validation loss is more then training
# NOTE: Adjust class imbalance by oversampling 
# NOTE: lower confidence threshould to increase the recall 
# NOTE: 
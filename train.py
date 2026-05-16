"""
DPF-Trans training script.

Training configuration mirrors the paper:
- AdamW, lr=1e-4, weight_decay=1e-4
- Cosine decay after 5-epoch warmup
- 300 epochs, batch_size=16
- Mosaic augmentation in early stages, disabled in final 20 epochs
- Gumbel-Softmax temperature annealed from 1.0 to 0.1

Usage:
    python train.py --config configs/default.py --data_root ./data/MS-LarchPest
"""

import os
import sys
import math
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpf_trans.models import DPFTrans, DPFTransLoss
from dpf_trans.data import build_dataloader, build_augmentation
from dpf_trans.utils.matcher import HungarianMatcher
from dpf_trans.configs.config import DPFTransConfig, TrainConfig


def parse_args():
    parser = argparse.ArgumentParser(description='DPF-Trans Training')
    parser.add_argument('--data_root', type=str, default='./data/MS-LarchPest',
                        help='MS-LarchPest dataset root')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate_only', action='store_true',
                        help='Only run validation')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision')

    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, criterion, matcher, optimizer,
                    scaler, epoch: int, total_epochs: int, device,
                    gumbel_init_temp: float = 1.0, gumbel_min_temp: float = 0.1,
                    use_amp: bool = True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0
    total_prune_loss = 0.0
    start_time = time.time()

    # Compute Gumbel temperature for this epoch (cosine annealing)
    progress = epoch / max(total_epochs - 1, 1)
    temperature = gumbel_min_temp + 0.5 * (gumbel_init_temp - gumbel_min_temp) * \
                  (1 + math.cos(math.pi * progress))
    model.set_gumbel_temperature(temperature)

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        targets = batch['targets']

        # Move targets to device
        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images, targets)
                pred_logits = outputs['pred_logits']
                pred_boxes = outputs['pred_boxes']

                # Hungarian matching
                indices = matcher(pred_logits.detach(), pred_boxes.detach(), targets)

                # Compute loss
                keep_masks = [
                    outputs['keep_masks']['s3'],
                    outputs['keep_masks']['s4'],
                ]
                target_ratios = [0.7, 0.6]

                loss_dict = criterion(
                    pred_logits, pred_boxes, targets, indices,
                    keep_masks, target_ratios,
                )

            scaler.scale(loss_dict['total_loss']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, targets)
            pred_logits = outputs['pred_logits']
            pred_boxes = outputs['pred_boxes']

            indices = matcher(pred_logits.detach(), pred_boxes.detach(), targets)

            keep_masks = [
                outputs['keep_masks']['s3'],
                outputs['keep_masks']['s4'],
            ]
            target_ratios = [0.7, 0.6]

            loss_dict = criterion(
                pred_logits, pred_boxes, targets, indices,
                keep_masks, target_ratios,
            )

            loss_dict['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss_dict['total_loss'].item()
        total_cls_loss += loss_dict['cls_loss'].item()
        total_box_loss += (loss_dict['l1_loss'].item() + loss_dict['giou_loss'].item())
        total_prune_loss += loss_dict.get('prune_loss', torch.tensor(0.0)).item()

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss_dict['total_loss'].item():.4f} | "
                  f"Temp: {temperature:.4f}")

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_cls = total_cls_loss / num_batches
    avg_box = total_box_loss / num_batches
    avg_prune = total_prune_loss / num_batches
    elapsed = time.time() - start_time

    return {
        'loss': avg_loss,
        'cls_loss': avg_cls,
        'box_loss': avg_box,
        'prune_loss': avg_prune,
        'time': elapsed,
        'temperature': temperature,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, matcher, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0

    # Metrics
    all_pred_boxes = []
    all_pred_scores = []
    all_pred_labels = []
    all_gt_boxes = []
    all_gt_labels = []

    for batch in dataloader:
        images = batch['image'].to(device)
        targets = batch['targets']

        for t in targets:
            t['boxes'] = t['boxes'].to(device)
            t['labels'] = t['labels'].to(device)

        outputs = model(images)
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        # Hungarian matching
        indices = matcher(pred_logits, pred_boxes, targets)

        # Loss
        loss_dict = criterion(
            pred_logits, pred_boxes, targets, indices,
            keep_masks=None, target_retention_ratios=None,
        )
        total_loss += loss_dict['total_loss'].item()

        # Collect predictions for mAP computation
        for b in range(len(targets)):
            pred_idx, tgt_idx = indices[b]

            if len(tgt_idx) > 0:
                matched_logits = pred_logits[b, pred_idx]
                matched_scores = matched_logits.sigmoid().max(dim=-1)[0]
                matched_boxes = pred_boxes[b, pred_idx]

                all_pred_boxes.append(matched_boxes.cpu())
                all_pred_scores.append(matched_scores.cpu())
                all_pred_labels.append(matched_logits.sigmoid().argmax(-1).cpu())
                all_gt_boxes.append(targets[b]['boxes'][tgt_idx].cpu())
                all_gt_labels.append(targets[b]['labels'][tgt_idx].cpu())

    avg_loss = total_loss / len(dataloader)

    # Simple mAP@0.5 computation
    mAP50 = compute_map50(
        all_pred_boxes, all_pred_scores, all_pred_labels,
        all_gt_boxes, all_gt_labels,
    )

    return {'loss': avg_loss, 'mAP50': mAP50}


def compute_map50(pred_boxes, pred_scores, pred_labels,
                   gt_boxes, gt_labels, iou_threshold: float = 0.5):
    """Compute mAP@0.5 using simple IoU matching."""
    if not pred_boxes:
        return 0.0

    from dpf_trans.utils.matcher import box_cxcywh_to_xyxy

    ap_per_class = []

    for cls_id in range(3):  # 3 classes
        # Collect predictions for this class
        cls_preds = []
        for pb, ps, pl in zip(pred_boxes, pred_scores, pred_labels):
            mask = pl == cls_id
            for i in range(len(pb)):
                if mask[i]:
                    cls_preds.append({
                        'box': pb[i],
                        'score': ps[i].item(),
                        'matched': False,
                    })

        # Collect ground truths for this class
        cls_gts = []
        for gb, gl in zip(gt_boxes, gt_labels):
            mask = gl == cls_id
            for i in range(len(gb)):
                if mask[i]:
                    cls_gts.append({'box': gb[i], 'matched': False})

        if not cls_gts:
            continue
        if not cls_preds:
            ap_per_class.append(0.0)
            continue

        # Sort by confidence
        cls_preds.sort(key=lambda x: x['score'], reverse=True)

        # Match
        tp = []
        fp = []
        for pred in cls_preds:
            pred_xyxy = box_cxcywh_to_xyxy(pred['box'].unsqueeze(0)).squeeze(0)
            best_iou = 0.0
            best_gt_idx = -1

            for j, gt in enumerate(cls_gts):
                if gt['matched']:
                    continue
                gt_xyxy = box_cxcywh_to_xyxy(gt['box'].unsqueeze(0)).squeeze(0)

                # IoU
                inter_x1 = max(pred_xyxy[0], gt_xyxy[0])
                inter_y1 = max(pred_xyxy[1], gt_xyxy[1])
                inter_x2 = min(pred_xyxy[2], gt_xyxy[2])
                inter_y2 = min(pred_xyxy[3], gt_xyxy[3])
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter = inter_w * inter_h

                area_pred = (pred_xyxy[2] - pred_xyxy[0]) * (pred_xyxy[3] - pred_xyxy[1])
                area_gt = (gt_xyxy[2] - gt_xyxy[0]) * (gt_xyxy[3] - gt_xyxy[1])
                union = area_pred + area_gt - inter

                iou = inter / (union + 1e-8)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp.append(1)
                fp.append(0)
                cls_gts[best_gt_idx]['matched'] = True
            else:
                tp.append(0)
                fp.append(1)

        # Compute AP from precision-recall
        tp = np.array(tp)
        fp = np.array(fp)
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        recall = cum_tp / max(len(cls_gts), 1)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

        # 11-point interpolation
        ap = 0.0
        for r in np.linspace(0, 1, 11):
            p_at_r = precision[recall >= r].max() if (recall >= r).any() else 0.0
            ap += p_at_r
        ap /= 11.0

        ap_per_class.append(ap)

    return np.mean(ap_per_class) if ap_per_class else 0.0


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Build model
    model_cfg = DPFTransConfig()
    model = DPFTrans(
        num_classes=3,
        input_size=model_cfg.input_size,
        backbone_config=model_cfg.backbone.__dict__ if hasattr(model_cfg.backbone, '__dict__') else {},
        fgdp_config=model_cfg.fgdp.__dict__ if hasattr(model_cfg.fgdp, '__dict__') else {},
        asfi_config=model_cfg.asfi.__dict__ if hasattr(model_cfg.asfi, '__dict__') else {},
        ccff_config=model_cfg.ccff.__dict__ if hasattr(model_cfg.ccff, '__dict__') else {},
        vcbl_config=model_cfg.vcbl.__dict__ if hasattr(model_cfg.vcbl, '__dict__') else {},
    ).to(device)

    n_params = model.count_parameters() / 1e6
    print(f"DPF-Trans parameters: {n_params:.1f}M")
    print(f"Expected: 28.5M (paper); Actual: {n_params:.1f}M")

    # Build dataloaders
    augment = build_augmentation(model_cfg.input_size, use_mosaic=True)

    train_loader = build_dataloader(
        args.data_root, 'train', args.batch_size, args.num_workers,
        model_cfg.input_size, augment,
    )
    val_loader = build_dataloader(
        args.data_root, 'val', args.batch_size, 2,
        model_cfg.input_size, None,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Loss and matcher
    criterion = DPFTransLoss(
        cls_weight=model_cfg.cls_loss_weight,
        l1_weight=model_cfg.l1_loss_weight,
        giou_weight=model_cfg.giou_loss_weight,
        prune_weight=model_cfg.prune_loss_weight,
    ).to(device)

    matcher = HungarianMatcher(
        cls_cost=model_cfg.cls_loss_weight,
        l1_cost=model_cfg.l1_loss_weight,
        giou_cost=model_cfg.giou_loss_weight,
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler: cosine with warmup
    warmup_epochs = 5
    total_epochs = args.epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs - 1, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # AMP
    scaler = GradScaler() if args.amp else None

    # Resume
    start_epoch = 0
    best_map = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint.get('best_map', 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Validate only
    if args.validate_only:
        val_metrics = validate(model, val_loader, criterion, matcher, device)
        print(f"Validation: Loss={val_metrics['loss']:.4f}, mAP@0.5={val_metrics['mAP50']:.4f}")
        return

    # Training loop
    print(f"\nStarting training for {total_epochs} epochs...")
    for epoch in range(start_epoch, total_epochs):
        print(f"\n--- Epoch {epoch+1}/{total_epochs} ---")

        # Check if mosaic should be disabled
        if epoch >= total_epochs - 20:  # Last 20 epochs without mosaic
            train_loader.dataset.mosaic_enabled = False

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, matcher, optimizer,
            scaler, epoch, total_epochs, device,
            gumbel_init_temp=1.0, gumbel_min_temp=0.1,
            use_amp=args.amp,
        )

        print(f"Train Loss: {train_metrics['loss']:.4f} | "
              f"CLS: {train_metrics['cls_loss']:.4f} | "
              f"Box: {train_metrics['box_loss']:.4f} | "
              f"Prune: {train_metrics['prune_loss']:.4f} | "
              f"Temp: {train_metrics['temperature']:.4f} | "
              f"Time: {train_metrics['time']:.1f}s")

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR: {current_lr:.2e}")

        # Validate
        if (epoch + 1) % 1 == 0 or epoch == total_epochs - 1:
            val_metrics = validate(model, val_loader, criterion, matcher, device)
            print(f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"mAP@0.5: {val_metrics['mAP50']:.4f}")

            # Save best
            if val_metrics['mAP50'] > best_map:
                best_map = val_metrics['mAP50']
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_map': best_map,
                    'config': model_cfg,
                }
                torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
                print(f"  → Best model saved (mAP@0.5: {best_map:.4f})")

        # Save checkpoint periodically
        if (epoch + 1) % 30 == 0 or epoch == total_epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_map': best_map,
                'config': model_cfg,
            }
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'),
            )

    print(f"\nTraining complete. Best mAP@0.5: {best_map:.4f}")
    print(f"Expected: 89.2% mAP@0.5 (paper); Actual: {best_map:.4f}")


if __name__ == '__main__':
    main()

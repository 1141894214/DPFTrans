"""
Loss functions for DPF-Trans.

L = lambda_cls * L_vl_cls + lambda_L1 * L1 + lambda_giou * L_GIoU + lambda_prune * L_prune
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from ..utils.matcher import box_cxcywh_to_xyxy, generalized_box_iou


class FocalLoss(nn.Module):
    """Focal loss for vision-language contrastive classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, num_classes] classification logits
            targets: [N] class labels (long)
        Returns:
            scalar loss
        """
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, F.one_hot(targets, inputs.shape[-1]).float(), reduction='none')

        p_t = prob * F.one_hot(targets, inputs.shape[-1]) + (1 - prob) * (
            1 - F.one_hot(targets, inputs.shape[-1]))
        p_t = p_t.sum(-1)

        loss = ce_loss.sum(-1)
        loss = self.alpha * (1 - p_t) ** self.gamma * loss

        return loss.mean()


class L1BoxLoss(nn.Module):
    """L1 loss for bounding box regression."""

    def forward(self, pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(pred_boxes, tgt_boxes, reduction='none')
        return loss.sum(-1).mean()


class GIoULoss(nn.Module):
    """Generalized IoU loss for bounding box regression."""

    def forward(self, pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor) -> torch.Tensor:
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
        giou = generalized_box_iou(pred_xyxy, tgt_xyxy)
        # giou is [N, N] (diagonal for matched pairs)
        loss = 1 - torch.diag(giou)
        return loss.mean()


class DPFTransLoss(nn.Module):
    """
    Complete DPF-Trans loss with Hungarian matching.

    L = lambda_cls * L_vl_cls + lambda_L1 * L1 + lambda_giou * L_GIoU + lambda_prune * L_prune
    """

    def __init__(self, cls_weight: float = 2.0, l1_weight: float = 5.0,
                 giou_weight: float = 2.0, prune_weight: float = 0.1,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        super().__init__()
        self.cls_weight = cls_weight
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.prune_weight = prune_weight

        self.focal_loss = FocalLoss(focal_alpha, focal_gamma)
        self.l1_loss = L1BoxLoss()
        self.giou_loss = GIoULoss()

    def forward(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                targets: list, indices: list,
                keep_masks: List[torch.Tensor] = None,
                target_retention_ratios: List[float] = None) -> dict:
        """
        Args:
            pred_logits: [B, N_q, num_classes]
            pred_boxes: [B, N_q, 4]
            targets: list of dicts with 'labels' and 'boxes'
            indices: Hungarian matching indices
            keep_masks: list of [B, N_w] pruning masks per level
            target_retention_ratios: target rho per level
        Returns:
            loss_dict with keys: cls, l1, giou, prune, total
        """
        B = pred_logits.shape[0]
        device = pred_logits.device

        cls_losses = []
        l1_losses = []
        giou_losses = []

        for b in range(B):
            pred_idx, tgt_idx = indices[b]

            if len(tgt_idx) == 0:
                continue

            matched_logits = pred_logits[b, pred_idx]   # [N_m, num_classes]
            matched_boxes = pred_boxes[b, pred_idx]      # [N_m, 4]
            matched_labels = targets[b]['labels'][tgt_idx]
            matched_gt_boxes = targets[b]['boxes'][tgt_idx]

            cls_losses.append(self.focal_loss(matched_logits, matched_labels))
            l1_losses.append(self.l1_loss(matched_boxes, matched_gt_boxes))
            giou_losses.append(self.giou_loss(matched_boxes, matched_gt_boxes))

        loss_dict = {
            'cls_loss': torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device),
            'l1_loss': torch.stack(l1_losses).mean() if l1_losses else torch.tensor(0.0, device=device),
            'giou_loss': torch.stack(giou_losses).mean() if giou_losses else torch.tensor(0.0, device=device),
        }

        # Pruning regularization loss
        prune_loss = torch.tensor(0.0, device=device)
        if keep_masks is not None and target_retention_ratios is not None:
            for mask, rho in zip(keep_masks, target_retention_ratios):
                actual_ratio = mask.mean(dim=-1)  # [B]
                prune_loss = prune_loss + torch.abs(actual_ratio - rho).mean()

        loss_dict['prune_loss'] = prune_loss

        # Total loss
        total = (self.cls_weight * loss_dict['cls_loss'] +
                 self.l1_weight * loss_dict['l1_loss'] +
                 self.giou_weight * loss_dict['giou_loss'] +
                 self.prune_weight * prune_loss)

        loss_dict['total_loss'] = total

        return loss_dict

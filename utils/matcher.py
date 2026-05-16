"""
Hungarian matcher for end-to-end set prediction.

Computes bipartite matching between predictions and ground-truth objects
using a combined cost matrix.
"""

import torch
import torch.nn.functional as F
from typing import Tuple

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU between two sets of boxes.
    boxes1: [N, 4], boxes2: [M, 4], xyxy format.
    Returns: [N, M] pairwise GIoU.
    """
    assert boxes1.dim() == 2 and boxes2.dim() == 2

    # Area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]

    # Union
    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-8)

    # Enclosing box
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_enc = (rb_enc - lt_enc).clamp(min=0)
    area_enc = wh_enc[:, :, 0] * wh_enc[:, :, 1]

    giou = iou - (area_enc - union) / (area_enc + 1e-8)

    return giou


class HungarianMatcher:
    """
    Hungarian matcher for bipartite matching.

    Cost = lambda_cls * L_vl_cls + lambda_l1 * L1 + lambda_giou * GIoU
    """

    def __init__(self, cls_cost: float = 2.0, l1_cost: float = 5.0,
                 giou_cost: float = 2.0):
        self.cls_cost = cls_cost
        self.l1_cost = l1_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def __call__(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                 targets: list) -> list:
        """
        Args:
            pred_logits: [B, N_q, num_classes] classification logits
            pred_boxes: [B, N_q, 4] predicted boxes (cxcywh, normalized)
            targets: list of dicts with 'labels' [N_gt] and 'boxes' [N_gt, 4]
        Returns:
            indices: list of (pred_idx, gt_idx) tuples per image
        """
        B, N_q = pred_logits.shape[:2]
        bs_indices = []

        for b in range(B):
            tgt_labels = targets[b]['labels']  # [N_gt]
            tgt_boxes = targets[b]['boxes']    # [N_gt, 4] cxcywh

            N_gt = len(tgt_labels)
            if N_gt == 0:
                bs_indices.append((torch.tensor([], dtype=torch.long),
                                    torch.tensor([], dtype=torch.long)))
                continue

            # Classification cost (focal sigmoid cross-entropy)
            pred_logit_b = pred_logits[b]  # [N_q, num_classes]
            cls_cost = self._focal_cost(pred_logit_b, tgt_labels)

            # L1 cost
            pred_box_b = pred_boxes[b]
            l1_cost = torch.cdist(pred_box_b, tgt_boxes, p=1)

            # GIoU cost
            pred_xyxy = box_cxcywh_to_xyxy(pred_box_b)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            giou_cost = -generalized_box_iou(pred_xyxy, tgt_xyxy)

            # Combined cost
            cost = (self.cls_cost * cls_cost + self.l1_cost * l1_cost +
                    self.giou_cost * giou_cost)

            cost_np = cost.cpu().numpy()
            pred_idx, tgt_idx = self._solve_assignment(cost_np)
            bs_indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(tgt_idx, dtype=torch.long),
            ))

        return bs_indices

    @staticmethod
    def _solve_assignment(cost: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
        """Solve linear sum assignment, with fallback if scipy unavailable."""
        import numpy as np
        if _HAS_SCIPY:
            return linear_sum_assignment(cost)
        # Greedy fallback: iteratively match lowest cost
        cost = cost.copy()
        N, M = cost.shape
        row_ind = []
        col_ind = []
        for _ in range(min(N, M)):
            r, c = np.unravel_index(cost.argmin(), cost.shape)
            row_ind.append(r)
            col_ind.append(c)
            cost[r, :] = 1e10
            cost[:, c] = 1e10
        return np.array(row_ind), np.array(col_ind)

    def _focal_cost(self, pred_logits: torch.Tensor,
                    tgt_labels: torch.Tensor) -> torch.Tensor:
        """
        Focal loss based classification cost.
        pred_logits: [N_q, num_classes]
        tgt_labels: [N_gt]
        Returns: [N_q, N_gt]
        """
        alpha = 0.25
        gamma = 2.0

        prob = pred_logits.sigmoid()  # [N_q, num_classes]
        neg_prob = 1 - prob

        ce_loss = torch.zeros(pred_logits.shape[0], len(tgt_labels),
                              device=pred_logits.device)
        for j, label in enumerate(tgt_labels):
            p = prob[:, label]
            ce_loss[:, j] = -alpha * (1 - p) ** gamma * p.clamp(min=1e-8).log()

        return ce_loss

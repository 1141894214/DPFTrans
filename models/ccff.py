"""
CCFF: Cross-Scale Contextual Feature Fusion.

Aggregates sparse-enhanced pathological representations across scales.
Uses bidirectional aggregation:
- Top-down: injects deep semantics from C5 into S4, S3
- Bottom-up: propagates fine boundary cues from S3 to deeper levels

S_fusion = CCFF(S3, S4, C5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CCFF(nn.Module):
    """
    Cross-Scale Contextual Feature Fusion.

    Fuses S3, S4 (sparse, after ASFI) and C5 (dense, from backbone)
    into a single fused visual memory for the decoder.
    """

    def __init__(self, dim: int = 256, c5_channels: int = 640):
        super().__init__()

        # Project backbone C5 features to target dim
        self.proj_c5 = nn.Conv2d(c5_channels, dim, kernel_size=1)

        # Top-down fusion: semantic injection
        self.td_conv4 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.td_conv3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        # Bottom-up fusion: detail propagation
        self.bu_conv4 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )
        self.bu_conv5 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

        # Output refinement
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, S3: torch.Tensor, S4: torch.Tensor, C5: torch.Tensor,
                pos3: torch.Tensor, pos4: torch.Tensor,
                spatial3: Tuple[int, int], spatial4: Tuple[int, int],
                spatial5: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            S3: [B, N3, D] sparse tokens from ASFI level 3
            S4: [B, N4, D] sparse tokens from ASFI level 4
            C5: [B, 512, H5, W5] dense features from backbone
        Returns:
            S_fusion: [B, H3*W3, dim] fused visual memory
        """
        B = C5.shape[0]
        H3, W3 = spatial3
        H4, W4 = spatial4

        # Ensure sparse tokens are in target dim
        D = 256
        if S3.shape[-1] != D:
            if not hasattr(self, 's3_proj'):
                self.s3_proj = nn.Linear(S3.shape[-1], D, bias=False).to(S3.device)
            S3 = self.s3_proj(S3)
        if S4.shape[-1] != D:
            if not hasattr(self, 's4_proj'):
                self.s4_proj = nn.Linear(S4.shape[-1], D, bias=False).to(S4.device)
            S4 = self.s4_proj(S4)

        # Project C5
        C5_proj = self.proj_c5(C5)  # [B, D, H5, W5]

        # Scatter sparse tokens to 2D grids
        S3_2d = self._scatter(S3, pos3, H3, W3)  # [B, D, H3, W3]
        S4_2d = self._scatter(S4, pos4, H4, W4)  # [B, D, H4, W4]

        # ---- Top-down: semantic injection ----
        # C5 → S4
        c5_up = F.interpolate(C5_proj, size=(H4, W4), mode='bilinear',
                               align_corners=False)
        s4_td = self.td_conv4(torch.cat([S4_2d, c5_up], dim=1))

        # C5-enhanced S4 → S3
        s4_td_up = F.interpolate(s4_td, size=(H3, W3), mode='bilinear',
                                  align_corners=False)
        s3_td = self.td_conv3(torch.cat([S3_2d, s4_td_up], dim=1))

        # ---- Bottom-up: detail propagation ----
        # S3 → S4
        s3_down = F.avg_pool2d(s3_td, kernel_size=2, stride=2)
        if s3_down.shape[2:] != (H4, W4):
            s3_down = F.interpolate(s3_down, size=(H4, W4), mode='bilinear',
                                     align_corners=False)
        s4_bu = self.bu_conv4(torch.cat([s4_td, s3_down], dim=1))

        # S4 → C5
        s4_down = F.avg_pool2d(s4_bu, kernel_size=2, stride=2)
        H5, W5 = spatial5
        if s4_down.shape[2:] != (H5, W5):
            s4_down = F.interpolate(s4_down, size=(H5, W5), mode='bilinear',
                                     align_corners=False)
        s5_bu = self.bu_conv5(torch.cat([C5_proj, s4_down], dim=1))

        # ---- Gather outputs at S3 resolution ----
        s4_bu_up = F.interpolate(s4_bu, size=(H3, W3), mode='bilinear',
                                  align_corners=False)
        s5_bu_up = F.interpolate(s5_bu, size=(H3, W3), mode='bilinear',
                                  align_corners=False)

        # Concatenate and refine
        fused = torch.cat([s3_td, s4_bu_up, s5_bu_up], dim=1)
        out = self.out_conv(fused)  # [B, D, H3, W3]

        # Flatten for decoder
        out = out.reshape(B, D, -1).permute(0, 2, 1)  # [B, H3*W3, D]

        return out

    def _scatter(self, x: torch.Tensor, positions: torch.Tensor,
                 H: int, W: int) -> torch.Tensor:
        """Scatter sparse tokens [B, N, D] to dense grid [B, D, H, W]."""
        B, N, D = x.shape

        px = (positions[..., 0] * (W - 1)).long().clamp(0, W - 1)
        py = (positions[..., 1] * (H - 1)).long().clamp(0, H - 1)

        flat_idx = py * W + px  # [B, N]
        feat = torch.zeros(B, D, H * W, device=x.device, dtype=x.dtype)
        feat.scatter_add_(2, flat_idx.unsqueeze(1).expand(-1, D, -1),
                          x.permute(0, 2, 1))
        feat = feat.view(B, D, H, W)

        # Light smoothing
        if N < H * W * 0.8:
            kernel = torch.ones(D, 1, 3, 3, device=x.device) / 9.0
            feat = F.conv2d(F.pad(feat, (1, 1, 1, 1), mode='reflect'),
                            kernel, groups=D)
        return feat

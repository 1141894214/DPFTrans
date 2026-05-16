"""
PA-HGNet: Pathology-Aware HGNet Backbone.

Enhances three types of pruning-relevant cues:
- High-frequency lesion textures
- Canopy boundary discontinuities
- Morphological structural variations

Key operators:
- PA GhostBlock: Ghost feature generation + pathology-aware spatial perception
- MP-Down: Morphology-preserving downsampling (dual-track pooling)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class PA_GhostBlock(nn.Module):
    """
    Pathology-Aware GhostBlock.

    Combines Ghost-style efficient feature generation with a pathology-aware
    spatial perception branch that produces pruning-oriented response maps.
    """

    def __init__(self, in_channels: int, out_channels: int, ghost_ratio: float = 0.5):
        super().__init__()
        self.ghost_ratio = ghost_ratio
        intrinsic_channels = int(out_channels * ghost_ratio)
        ghost_channels = out_channels - intrinsic_channels

        # Primary convolution: generates intrinsic features
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, intrinsic_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intrinsic_channels),
            nn.SiLU(),
        )

        # Cheap depth-wise transformations to generate ghost features
        self.ghost_conv = nn.Sequential(
            nn.Conv2d(intrinsic_channels, ghost_channels, kernel_size=3,
                       padding=1, groups=intrinsic_channels, bias=False),
            nn.BatchNorm2d(ghost_channels),
        )

        # Pathology-aware spatial perception branch
        # Takes concatenated [MaxPool(X); AvgPool(X)] with 2*out_channels → 1
        self.pathology_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Ghost feature generation
        intrinsic = self.primary_conv(x)
        ghost = self.ghost_conv(intrinsic)
        x_primary = torch.cat([intrinsic, ghost], dim=1)

        # Pathology-aware spatial perception
        T = self._compute_pathology_response(x_primary)

        # Adaptive feature recalibration: X_enhanced = X_primary ⊗ (1 + T)
        x_enhanced = x_primary * (1.0 + T)

        # Output projection
        out = self.out_conv(x_enhanced)
        return out, T

    def _compute_pathology_response(self, x: torch.Tensor) -> torch.Tensor:
        """T = sigma(f_3x3([MaxPool(X); AvgPool(X)]))"""
        max_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        avg_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        pooled = torch.cat([max_pool, avg_pool], dim=1)
        T = self.pathology_conv(pooled)
        return T


class MP_Down(nn.Module):
    """
    Morphology-Preserving Downsampling.

    Uses a dual-track pooling strategy instead of stride-2 convolution:
    - High-frequency track: stride-2 max pooling (sharp lesion responses)
    - Low-frequency track: stride-2 average pooling (smooth canopy context)

    X_out = SiLU(BN(f_1x1([MaxPool(X_in); AvgPool(X_in)])))
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_feat = F.max_pool2d(x, kernel_size=2, stride=2)
        avg_feat = F.avg_pool2d(x, kernel_size=2, stride=2)
        fused = torch.cat([max_feat, avg_feat], dim=1)
        return self.fusion_conv(fused)


class PA_HGNetStage(nn.Module):
    """A single stage of PA-HGNet: MP_Down then multiple PA_GhostBlocks."""

    def __init__(self, in_channels: int, out_channels: int, depth: int,
                 ghost_ratio: float = 0.5, return_priors: bool = False):
        super().__init__()
        self.return_priors = return_priors

        # Downsample first to get features at target resolution
        self.downsample = MP_Down(in_channels, out_channels)

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = PA_GhostBlock(out_channels, out_channels, ghost_ratio)
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Downsample first
        x = self.downsample(x)

        T_accum = None
        for block in self.blocks:
            x, T = block(x)
            if self.return_priors:
                T_accum = T if T_accum is None else T_accum + T

        # Average the accumulated pathology response
        if T_accum is not None:
            T_accum = T_accum / len(self.blocks)

        # Return: feature at this level and pathology prior
        return None, x, T_accum


class PA_HGNet(nn.Module):
    """
    Pathology-Aware HGNet Backbone.

    Produces:
    - Multi-scale visual features: {C3, C4, C5}
    - Pathology-aware spatial priors: {T3, T4}

    C3: fine-grained lesion textures and weak canopy boundaries
    C4: meso-level canopy structures
    C5: high-level pathological semantics
    T3, T4: used by FGDP to estimate pathological importance of local windows
    """

    def __init__(self, in_channels: int = 3, stem_channels: int = 32,
                 stage_channels: List[int] = None, stage_depths: List[int] = None,
                 ghost_ratio: float = 0.5, prior_levels: List[int] = None):
        super().__init__()

        if stage_channels is None:
            stage_channels = [128, 256, 512]
        if stage_depths is None:
            stage_depths = [3, 5, 5]
        if prior_levels is None:
            prior_levels = [3, 4]

        self.prior_levels = prior_levels

        # Stem: initial feature extraction (1/4 resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2,
                       padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=3, stride=2,
                       padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(),
        )

        # Build stages
        self.stages = nn.ModuleList()
        in_ch = stem_channels
        for i, (out_ch, depth) in enumerate(zip(stage_channels, stage_depths)):
            stage_level = i + 3  # levels 3, 4, 5
            return_priors = stage_level in prior_levels
            stage = PA_HGNetStage(in_ch, out_ch, depth, ghost_ratio, return_priors)
            self.stages.append(stage)
            in_ch = out_ch  # next stage input is this stage's output channels

        # Store multi-scale feature channels
        self.out_channels = stage_channels

    def forward(self, x: torch.Tensor) -> Tuple[dict, dict]:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            features: {'C3': tensor, 'C4': tensor, 'C5': tensor}
            priors: {'T3': tensor, 'T4': tensor}
        """
        features = {}
        priors = {}

        x = self.stem(x)

        for i, stage in enumerate(self.stages):
            level = i + 3
            _, x, T = stage(x)
            features[f'C{level}'] = x
            if T is not None:
                priors[f'T{level}'] = T

        return features, priors

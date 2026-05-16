"""
DPF-Trans configuration.
Reference: DPF-Trans: Dynamic Pathology-Aware Feature Pruning Transformer
for Efficient UAV-Based Forest Pest Detection (IEEE JSTARS).
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PA_HGNetConfig:
    """Pathology-Aware HGNet backbone configuration."""
    # Input
    in_channels: int = 3

    # Stem
    stem_channels: int = 32

    # Stage channel configs: [C3, C4, C5]
    stage_channels: List[int] = field(default_factory=lambda: [160, 320, 640])

    # Stage depth configs: number of PA GhostBlocks per stage
    stage_depths: List[int] = field(default_factory=lambda: [3, 5, 5])

    # Ghost ratio: intrinsic channels ratio
    ghost_ratio: float = 0.5

    # Downsampling
    down_sample_strides: List[int] = field(default_factory=lambda: [2, 2, 2])

    # Pathology prior output levels: which stages produce T maps
    prior_levels: List[int] = field(default_factory=lambda: [3, 4])


@dataclass
class FGDPConfig:
    """Feature-Guided Dynamic Pruning configuration."""
    window_size: int = 8
    target_retention_ratio_s3: float = 0.7
    target_retention_ratio_s4: float = 0.6
    gumbel_init_temp: float = 1.0
    gumbel_min_temp: float = 0.1

    # Dual-prior scoring
    texture_channels: int = 64
    morphology_channels: int = 64

    # Pruning levels
    prune_levels: List[int] = field(default_factory=lambda: [3, 4])


@dataclass
class ASFIConfig:
    """Asymmetric Sparse Feature Interaction configuration."""
    hidden_dim: int = 256
    n_heads: int = 8
    n_deformable_heads: int = 8
    n_deformable_points: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1


@dataclass
class CCFFConfig:
    """Cross-Scale Contextual Feature Fusion configuration."""
    hidden_dim: int = 256
    num_fusion_blocks: int = 2


@dataclass
class VCBLConfig:
    """Vision-to-Class/Box Contrastive Learning decoder configuration."""
    hidden_dim: int = 256
    n_heads: int = 8
    n_deformable_heads: int = 8
    n_deformable_points: int = 4
    ffn_dim: int = 1024
    num_decoder_layers: int = 3
    num_queries: int = 300
    dropout: float = 0.1

    # Text prototype
    text_dim: int = 512
    learnable_temperature: bool = True
    init_temperature: float = 0.07

    # Category prompts
    category_prompts: List[str] = field(default_factory=lambda: [
        "healthy tree crown",
        "lightly damaged tree crown",
        "heavily damaged tree crown",
    ])


@dataclass
class DPFTransConfig:
    """Complete DPF-Trans model configuration."""
    # Input
    input_size: Tuple[int, int] = (640, 640)

    # Sub-module configs
    backbone: PA_HGNetConfig = field(default_factory=PA_HGNetConfig)
    fgdp: FGDPConfig = field(default_factory=FGDPConfig)
    asfi: ASFIConfig = field(default_factory=ASFIConfig)
    ccff: CCFFConfig = field(default_factory=CCFFConfig)
    vcbl: VCBLConfig = field(default_factory=VCBLConfig)

    # Number of classes (H, LD, HD)
    num_classes: int = 3

    # Loss weights
    cls_loss_weight: float = 2.0
    l1_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0
    prune_loss_weight: float = 0.1

    # Focal loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class TrainConfig:
    """Training configuration."""
    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    # Scheduler
    warmup_epochs: int = 5
    total_epochs: int = 300
    lr_min: float = 1e-6

    # Data
    batch_size: int = 16
    num_workers: int = 4

    # Augmentation
    enable_mosaic: bool = True
    mosaic_disable_epochs: int = 20  # disable mosaic in last N epochs

    # Hardware
    device: str = "cuda"

    # Logging
    log_interval: int = 50
    eval_interval: int = 1
    save_interval: int = 10
    output_dir: str = "./output"

    # Dataset
    dataset_root: str = "./data/MS-LarchPest"

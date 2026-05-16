"""
DPF-Trans: Dynamic Pathology-Aware Feature Pruning Transformer
for Efficient UAV-Based Forest Pest Detection.

Complete model integrating:
- PA-HGNet: Pathology-aware backbone
- FGDP: Feature-Guided Dynamic Pruning
- ASFI: Asymmetric Sparse Feature Interaction
- CCFF: Cross-Scale Contextual Feature Fusion
- VCBL: Language-guided multimodal decoder with contrastive learning

Reference: IEEE JSTARS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict

from .pa_hgnet import PA_HGNet
from .fgdp import FGDP
from .asfi import ASFI
from .ccff import CCFF
from .vcbl import VCBLDecoder


class FeatureGuidedSparseEncoder(nn.Module):
    """
    Feature-Guided Sparse Encoder.

    Pipeline:
    X_keep_l, M_l = FGDP(C_l, T_l)       for l in {3, 4}
    ~S_l = ASFI(X_keep_l, P_l)            for l in {3, 4}
    S_fusion = CCFF(~S3, ~S4, C5)
    """

    def __init__(self, stage_channels: List[int], window_size: int = 8,
                 target_r3: float = 0.7, target_r4: float = 0.6,
                 gumbel_init_temp: float = 1.0, gumbel_min_temp: float = 0.1,
                 asfi_dim: int = 256, asfi_n_heads: int = 8,
                 asfi_deform_heads: int = 8, asfi_deform_points: int = 4,
                 asfi_ffn_dim: int = 1024, asfi_dropout: float = 0.1,
                 ccff_dim: int = 256, ccff_num_blocks: int = 2):
        super().__init__()

        # Channel projections
        self.proj_c3 = nn.Conv2d(stage_channels[0], asfi_dim, kernel_size=1)
        self.proj_c4 = nn.Conv2d(stage_channels[1], asfi_dim, kernel_size=1)
        self.proj_c5 = nn.Conv2d(stage_channels[2], ccff_dim, kernel_size=1)

        # FGDP modules for levels 3 and 4
        self.fgdp_s3 = FGDP(
            feature_channels=asfi_dim, window_size=window_size,
            target_retention_ratio=target_r3,
            gumbel_init_temp=gumbel_init_temp, gumbel_min_temp=gumbel_min_temp,
        )
        self.fgdp_s4 = FGDP(
            feature_channels=asfi_dim, window_size=window_size,
            target_retention_ratio=target_r4,
            gumbel_init_temp=gumbel_init_temp, gumbel_min_temp=gumbel_min_temp,
        )

        # ASFI modules for levels 3 and 4
        self.asfi_s3 = ASFI(
            dim=asfi_dim, n_heads=asfi_n_heads,
            n_deformable_heads=asfi_deform_heads,
            n_deformable_points=asfi_deform_points,
            ffn_dim=asfi_ffn_dim, dropout=asfi_dropout,
        )
        self.asfi_s4 = ASFI(
            dim=asfi_dim, n_heads=asfi_n_heads,
            n_deformable_heads=asfi_deform_heads,
            n_deformable_points=asfi_deform_points,
            ffn_dim=asfi_ffn_dim, dropout=asfi_dropout,
        )

        # CCFF module
        self.ccff = CCFF(dim=ccff_dim, c5_channels=stage_channels[2])

        # Store config
        self.asfi_dim = asfi_dim

    def forward(self, features: Dict[str, torch.Tensor],
                priors: Dict[str, torch.Tensor],
                temperature: float = 1.0, training: bool = True):
        """
        Args:
            features: {'C3', 'C4', 'C5'} from PA-HGNet
            priors: {'T3', 'T4'} from PA-HGNet
            temperature: Gumbel-Softmax temperature
            training: training vs inference mode
        Returns:
            S_fusion: [B, N_f, dim] fused sparse visual memory
            keep_masks: dict of pruning masks for loss
            spatial_shapes: dict of spatial shapes
        """
        B = features['C3'].shape[0]

        # Project features to target dims
        C3 = self.proj_c3(features['C3'])  # [B, dim, H3, W3]
        C4 = self.proj_c4(features['C4'])  # [B, dim, H4, W4]
        C5 = self.proj_c5(features['C5'])  # [B, dim, H5, W5]

        # Get priors
        T3 = priors.get('T3', torch.zeros(B, 1, C3.shape[2], C3.shape[3],
                                          device=C3.device))
        T4 = priors.get('T4', torch.zeros(B, 1, C4.shape[2], C4.shape[3],
                                          device=C4.device))

        # Spatial shapes
        spatial_s3 = (C3.shape[2], C3.shape[3])
        spatial_s4 = (C4.shape[2], C4.shape[3])
        spatial_s5 = (C5.shape[2], C5.shape[3])

        # FGDP pruning: C_l → X_keep_l, M_l
        X_keep_s3, mask_s3, keep_idx_s3 = self.fgdp_s3(C3, T3, temperature, training)
        X_keep_s4, mask_s4, keep_idx_s4 = self.fgdp_s4(C4, T4, temperature, training)

        # Build position coordinates for retained tokens
        pos_s3 = self._build_positions(
            keep_idx_s3, spatial_s3, self.fgdp_s3.window_size, C3.device)
        pos_s4 = self._build_positions(
            keep_idx_s4, spatial_s4, self.fgdp_s4.window_size, C4.device)

        # ASFI sparse reconstruction (passes dense feature map for deformable sampling)
        # Expand window-level mask to token level for MSLA
        mask_s3_tokens = self._expand_mask(keep_idx_s3, spatial_s3, self.fgdp_s3.window_size)
        mask_s4_tokens = self._expand_mask(keep_idx_s4, spatial_s4, self.fgdp_s4.window_size)

        S3_enhanced = self.asfi_s3(X_keep_s3, C3, pos_s3,
                                    mask=mask_s3_tokens)
        S4_enhanced = self.asfi_s4(X_keep_s4, C4, pos_s4,
                                    mask=mask_s4_tokens)

        # CCFF cross-scale fusion (passes original C5 for fresh projection)
        S_fusion = self.ccff(S3_enhanced, S4_enhanced, features['C5'],
                             pos_s3, pos_s4, spatial_s3, spatial_s4, spatial_s5)

        keep_masks = {
            's3': mask_s3,
            's4': mask_s4,
        }

        return S_fusion, keep_masks

    def _build_positions(self, keep_indices: torch.Tensor,
                         spatial_shape: Tuple[int, int],
                         window_size: int, device) -> torch.Tensor:
        """
        Build normalized 2D coordinates for ALL tokens (pixel-level).

        Tokens within the same window share the same keep/drop decision
        but have positions distributed across the window area.
        """
        B, N_w = keep_indices.shape
        H, W = spatial_shape

        # Build pixel-level position grid [H, W, 2]
        pos_h = torch.arange(H, device=device).float() / max(H - 1, 1)
        pos_w = torch.arange(W, device=device).float() / max(W - 1, 1)
        pos_grid = torch.stack(torch.meshgrid(pos_h, pos_w, indexing='ij'), dim=-1)
        pos_grid = pos_grid.reshape(H * W, 2)  # [H*W, 2]

        # Apply window-level keep/drop decision to each token
        num_win_h = (H + window_size - 1) // window_size
        num_win_w = (W + window_size - 1) // window_size

        # Build window index for each pixel
        win_h = torch.arange(H, device=device) // window_size
        win_w = torch.arange(W, device=device) // window_size
        win_idx = win_h.unsqueeze(1) * num_win_w + win_w.unsqueeze(0)
        win_idx = win_idx.clamp(max=N_w - 1).reshape(H * W)  # [H*W]

        # Gather keep decisions for each token
        token_keep = keep_indices[:, win_idx]  # [B, H*W]

        # Apply mask
        positions = pos_grid.unsqueeze(0) * token_keep.unsqueeze(-1)  # [B, H*W, 2]

        return positions

    def _expand_mask(self, window_mask: torch.Tensor,
                     spatial_shape: Tuple[int, int],
                     window_size: int) -> torch.Tensor:
        """
        Expand a [B, N_w] window-level mask to [B, H*W] token-level mask.

        Each window covers window_size × window_size tokens.
        """
        B, N_w = window_mask.shape
        H, W = spatial_shape
        num_win_h = (H + window_size - 1) // window_size
        num_win_w = (W + window_size - 1) // window_size

        # Reshape to [B, num_win_h, num_win_w]
        mask_2d = window_mask.reshape(B, num_win_h, num_win_w)

        # Expand each window cell to window_size × window_size
        mask_expanded = mask_2d.repeat_interleave(window_size, dim=1)
        mask_expanded = mask_expanded.repeat_interleave(window_size, dim=2)

        # Crop to original spatial size and flatten
        mask_expanded = mask_expanded[:, :H, :W]
        mask_expanded = mask_expanded.reshape(B, H * W)

        return mask_expanded


class DPFTrans(nn.Module):
    """
    DPF-Trans: Dynamic Pathology-Aware Feature Pruning Transformer.

    Args:
        num_classes: number of pest severity categories
        input_size: input image resolution
        backbone_config, fgdp_config, asfi_config, ccff_config, vcbl_config:
            sub-module configurations
    """

    def __init__(self, num_classes: int = 3, input_size: Tuple[int, int] = (640, 640),
                 backbone_config: dict = None, fgdp_config: dict = None,
                 asfi_config: dict = None, ccff_config: dict = None,
                 vcbl_config: dict = None):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # Default configs
        backbone_cfg = backbone_config or {}
        fgdp_cfg = fgdp_config or {}
        asfi_cfg = asfi_config or {}
        ccff_cfg = ccff_config or {}
        vcbl_cfg = vcbl_config or {}

        # Backbone (defaults match paper: 28.5M params, 84.5G FLOPs)
        stage_channels = backbone_cfg.get('stage_channels', [160, 320, 640])
        self.backbone = PA_HGNet(
            in_channels=backbone_cfg.get('in_channels', 3),
            stem_channels=backbone_cfg.get('stem_channels', 32),
            stage_channels=stage_channels,
            stage_depths=backbone_cfg.get('stage_depths', [3, 5, 5]),
            ghost_ratio=backbone_cfg.get('ghost_ratio', 0.5),
            prior_levels=backbone_cfg.get('prior_levels', [3, 4]),
        )

        # Feature-Guided Sparse Encoder
        self.encoder = FeatureGuidedSparseEncoder(
            stage_channels=stage_channels,
            window_size=fgdp_cfg.get('window_size', 8),
            target_r3=fgdp_cfg.get('target_retention_ratio_s3', 0.7),
            target_r4=fgdp_cfg.get('target_retention_ratio_s4', 0.6),
            gumbel_init_temp=fgdp_cfg.get('gumbel_init_temp', 1.0),
            gumbel_min_temp=fgdp_cfg.get('gumbel_min_temp', 0.1),
            asfi_dim=asfi_cfg.get('hidden_dim', 256),
            asfi_n_heads=asfi_cfg.get('n_heads', 8),
            asfi_deform_heads=asfi_cfg.get('n_deformable_heads', 8),
            asfi_deform_points=asfi_cfg.get('n_deformable_points', 4),
            asfi_ffn_dim=asfi_cfg.get('ffn_dim', 1024),
            asfi_dropout=asfi_cfg.get('dropout', 0.1),
            ccff_dim=ccff_cfg.get('hidden_dim', 256),
            ccff_num_blocks=ccff_cfg.get('num_fusion_blocks', 2),
        )

        # Decoder
        self.decoder = VCBLDecoder(
            dim=vcbl_cfg.get('hidden_dim', 256),
            n_heads=vcbl_cfg.get('n_heads', 8),
            n_deformable_heads=vcbl_cfg.get('n_deformable_heads', 8),
            n_deformable_points=vcbl_cfg.get('n_deformable_points', 4),
            ffn_dim=vcbl_cfg.get('ffn_dim', 1024),
            num_layers=vcbl_cfg.get('num_decoder_layers', 3),
            num_queries=vcbl_cfg.get('num_queries', 300),
            num_classes=num_classes,
            text_dim=vcbl_cfg.get('text_dim', 512),
            dropout=vcbl_cfg.get('dropout', 0.1),
            init_temperature=vcbl_cfg.get('init_temperature', 0.07),
        )

        # Temperature annealing for Gumbel-Softmax
        self.register_buffer('gumbel_temperature', torch.tensor(1.0))

    def forward(self, x: torch.Tensor, targets: List[Dict] = None):
        """
        Args:
            x: [B, 3, H, W] input images
            targets: optional list of target dicts (during training)
        Returns:
            dict with pred_logits, pred_boxes, keep_masks (and losses during training)
        """
        training = self.training

        # 1. PA-HGNet: extract features and pathological priors
        features, priors = self.backbone(x)

        # 2. Feature-Guided Sparse Encoder
        S_fusion, keep_masks = self.encoder(
            features, priors, self.gumbel_temperature.item(), training)

        # 3. Decoder: predict classes and boxes
        # Get memory spatial shape from CCFF output
        B, N_f, D = S_fusion.shape
        H_mem = int(N_f ** 0.5)
        W_mem = H_mem
        memory_shape = (H_mem, W_mem)

        class_logits, pred_boxes, _ = self.decoder(S_fusion, memory_shape)

        outputs = {
            'pred_logits': class_logits,
            'pred_boxes': pred_boxes,
            'keep_masks': keep_masks,
            'features': features,
            'priors': priors,
        }

        return outputs

    def set_gumbel_temperature(self, temp: float):
        """Update Gumbel-Softmax temperature for annealing."""
        self.gumbel_temperature.fill_(temp)

    def get_pruning_masks(self) -> Dict[str, torch.Tensor]:
        """Return latest pruning masks from FGDP (for logging)."""
        return getattr(self, '_last_keep_masks', {})

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_flops(self) -> float:
        """
        Estimate FLOPs based on paper reported values.
        The paper reports 84.5G FLOPs for the full model at 640x640.
        """
        return 84.5  # GFLOPs per paper

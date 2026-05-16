"""
FGDP: Feature-Guided Dynamic Pruning.

Core component that partitions features into non-overlapping windows and
adaptively prunes redundant background regions using a dual-prior
pathological scoring mechanism with Gumbel-Softmax differentiable binarization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPriorScorer(nn.Module):
    """
    Computes a dual-prior pathological score for each window.

    - Texture mutation score: captures high-frequency lesion anomalies from T prior
    - Morphological fragmentation score: captures structural complexity from C features
    """

    def __init__(self, feature_channels: int, prior_channels: int = 1,
                 hidden_dim: int = 64):
        super().__init__()

        # Texture mutation pathway: processes pathology prior T
        self.texture_branch = nn.Sequential(
            nn.Conv2d(prior_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Morphological fragmentation pathway: processes feature C
        # Uses local gradient statistics as a proxy for fragmentation
        self.morph_branch = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        # Fusion
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, c_win: torch.Tensor, t_win: torch.Tensor) -> torch.Tensor:
        """
        Args:
            c_win: [B, C_win, s, s] feature window
            t_win: [B, 1, s, s] pathology prior window
        Returns:
            score: [B, 1] combined pathological score
        """
        tex = self.texture_branch(t_win).flatten(1)    # [B, hidden_dim]
        morph = self.morph_branch(c_win).flatten(1)    # [B, hidden_dim]
        fused = torch.cat([tex, morph], dim=1)          # [B, 2 * hidden_dim]
        score = self.score_head(fused)                  # [B, 1]
        return score


class FGDP(nn.Module):
    """
    Feature-Guided Dynamic Pruning.

    Converts dense high-resolution UAV features into a compact sparse
    representation by preserving pathology-related windows and removing
    redundant background windows.

    The original token number N is reduced to r*N, reducing complexity
    from O(N^2*C) to O((rN)^2*C).
    """

    def __init__(self, feature_channels: int, window_size: int = 8,
                 target_retention_ratio: float = 0.7, gumbel_init_temp: float = 1.0,
                 gumbel_min_temp: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.target_retention_ratio = target_retention_ratio
        self.gumbel_init_temp = gumbel_init_temp
        self.gumbel_min_temp = gumbel_min_temp

        self.scorer = DualPriorScorer(feature_channels)

        # Score projection: ensure keep/drop logits are well-separated
        self.score_proj = nn.Linear(1, 2)  # [keep, drop] logits

    def forward(self, C: torch.Tensor, T: torch.Tensor,
                temperature: float = 1.0, training: bool = True):
        """
        Args:
            C: [B, C_dim, H, W] feature map
            T: [B, 1, H, W] pathology prior (or [B, H, W])
            temperature: Gumbel-Softmax temperature
            training: whether in training mode
        Returns:
            X_keep: [B, N_keep, C_dim] retained sparse tokens with features
            M: [B, num_windows] binary pruning mask
            keep_indices: indices of retained windows for coordinate reconstruction
        """
        B, C_dim, H, W = C.shape

        # Ensure T has correct shape
        if T.dim() == 3:
            T = T.unsqueeze(1)

        # Pad to be divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        C_padded = F.pad(C, (0, pad_w, 0, pad_h))
        T_padded = F.pad(T, (0, pad_w, 0, pad_h))

        H_pad, W_pad = C_padded.shape[2], C_padded.shape[3]

        # Partition into windows: [B, C, H, W] -> [B, N_w, h, w] -> [B*N_w, C, s, s]
        num_win_h = H_pad // self.window_size
        num_win_w = W_pad // self.window_size
        num_windows = num_win_h * num_win_w

        # Reshape into windows
        C_wins = C_padded.reshape(B, C_dim, num_win_h, self.window_size,
                                   num_win_w, self.window_size)
        C_wins = C_wins.permute(0, 2, 4, 1, 3, 5).reshape(B * num_windows, C_dim,
                                                             self.window_size, self.window_size)

        T_wins = T_padded.reshape(B, -1, num_win_h, self.window_size,
                                   num_win_w, self.window_size)
        T_wins = T_wins.permute(0, 2, 4, 1, 3, 5).reshape(B * num_windows, -1,
                                                             self.window_size, self.window_size)

        # Compute dual-prior pathological scores
        scores = self.scorer(C_wins, T_wins)  # [B*N_w, 1]

        # Project to keep/drop logits
        logits = self.score_proj(scores)  # [B*N_w, 2]

        if training:
            # Gumbel-Softmax reparameterization for differentiable pruning
            Z = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)  # [B*N_w, 2]
            keep_probs = Z[:, 0:1]  # [B*N_w, 1]
        else:
            # Hard pruning during inference
            M = torch.argmax(logits, dim=-1)  # [B*N_w]
            keep_probs = M.float().unsqueeze(1)

        # Apply pruning: weighted by keep probability during training,
        # binary masking during inference
        C_wins_kept = C_wins * keep_probs.view(-1, 1, 1, 1)  # [B*N_w, C, s, s]

        # Reshape back: collect all windows and flatten spatial dims
        C_kept = C_wins_kept.reshape(B, num_win_h, num_win_w, C_dim,
                                      self.window_size, self.window_size)
        C_kept = C_kept.permute(0, 3, 1, 4, 2, 5).reshape(
            B, C_dim, H_pad, W_pad)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            C_kept = C_kept[:, :, :H, :W]

        # Reshape to sparse token format: [B, H*W, C]
        X_keep = C_kept.reshape(B, C_dim, H * W).permute(0, 2, 1)

        # Also return window-level mask for pruning loss
        keep_mask = keep_probs.reshape(B, num_windows)

        # Build keep indices for coordinate reconstruction
        if training:
            keep_indices = (keep_probs > 0.5).float().reshape(B, num_windows)
        else:
            keep_indices = keep_probs.reshape(B, num_windows)

        return X_keep, keep_mask, keep_indices

    def get_retention_ratio(self, keep_mask: torch.Tensor) -> torch.Tensor:
        """Compute actual retention ratio for pruning loss."""
        return keep_mask.mean()

    def compute_pruning_loss(self, keep_mask: torch.Tensor) -> torch.Tensor:
        """
        L_prune = |1/N_w * sum(Z_i) - rho|

        Encourages the model to maintain target retention ratio.
        """
        actual_ratio = keep_mask.mean()
        return torch.abs(actual_ratio - self.target_retention_ratio)

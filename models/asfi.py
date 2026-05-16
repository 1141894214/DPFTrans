"""
ASFI: Asymmetric Sparse Feature Interaction.

Reconstructs meaningful dependencies among retained pathological tokens
after FGDP background suppression. Combines:
- Global branch: Masked Sparse Linear Attention (MSLA) operating on sparse tokens
- Local branch: Deformable attention sampling from the original dense feature map

Final: ~S = X_sparse + FFN(X_global + X_local)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MaskedSparseLinearAttention(nn.Module):
    """
    MSLA: Masked Sparse Linear Attention.

    Uses non-negative kernel feature map ELU(x)+1 to reduce complexity from
    O(N^2*C) to O(N*C^2).

    X_global = phi(Q) * (phi(K)^T * V) / (phi(Q) * (phi(K)^T * 1))
    """

    def __init__(self, dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _kernel_feature_map(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        residual = x

        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim)

        q = self._kernel_feature_map(q)
        k = self._kernel_feature_map(k)

        # kv = K^T * V -> [B, H, d, d]
        kv = torch.einsum('bnhd,bnhe->bhde', k, v)
        # z = K^T * 1 -> [B, H, d]
        z = k.sum(dim=1)
        # q * kv -> [B, N, H, d]
        out = torch.einsum('bnhd,bhde->bnhe', q, kv)
        # q * z -> [B, N, H, 1]
        out_z = torch.einsum('bnhd,bhd->bnh', q, z).unsqueeze(-1)
        out = out / (out_z + 1e-8)

        if mask is not None:
            out = out * mask.unsqueeze(-1).unsqueeze(-1)

        out = out.reshape(B, N, D)
        out = self.dropout(self.out_proj(out))
        return out + residual


class DeformableFeatureSampler(nn.Module):
    """
    Deformable feature sampler.

    For each sparse token at position p_q, predicts sampling offsets
    and aggregates features from the dense feature map around each
    reference point via grid_sample.
    """

    def __init__(self, dim: int, n_points: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(dim, n_points * 2)
        self.attention_weights = nn.Linear(dim, n_points)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, dense_feat: torch.Tensor,
                ref_points: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N_s, D = query.shape
        _, C, H, W = dense_feat.shape
        residual = query

        # Predict sampling offsets [B, N_s, P, 2] and weights [B, N_s, P]
        offsets = self.sampling_offsets(query).reshape(B, N_s, self.n_points, 2)
        offsets = offsets * 0.25

        attn_weights = F.softmax(
            self.attention_weights(query).reshape(B, N_s, self.n_points), dim=-1)

        # Sampling positions: ref_points + offsets, in [-1, 1]
        ref = ref_points.unsqueeze(2)  # [B, N_s, 1, 2]
        sample_pos = ref + offsets  # [B, N_s, P, 2]
        sample_pos[..., 0] = sample_pos[..., 0] * 2 - 1
        sample_pos[..., 1] = sample_pos[..., 1] * 2 - 1
        sample_pos = sample_pos.reshape(B, N_s * self.n_points, 1, 2)

        # Sample from dense feature map
        sampled = F.grid_sample(dense_feat, sample_pos, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)
        sampled = sampled.squeeze(-1)  # [B, C, N_s*P]
        sampled = sampled.reshape(B, C, N_s, self.n_points)
        sampled = sampled.permute(0, 2, 3, 1)  # [B, N_s, P, C]

        # Weighted aggregation across points
        attn_weights = attn_weights.unsqueeze(-1)  # [B, N_s, P, 1]
        out = (sampled * attn_weights).sum(dim=2)  # [B, N_s, C]

        # Project to dim if channel dimensions don't match
        if C != D:
            if not hasattr(self, 'channel_proj'):
                self.channel_proj = nn.Linear(C, D, bias=False).to(query.device)
            out = self.channel_proj(out)

        if mask is not None:
            out = out * mask.unsqueeze(-1)

        out = self.dropout(self.out_proj(out))
        return out + residual


class ASFI(nn.Module):
    """
    Asymmetric Sparse Feature Interaction.

    Reconstructs dense-like semantic and geometric dependencies from
    sparse pathological tokens after FGDP.
    """

    def __init__(self, dim: int = 256, n_heads: int = 8, n_deformable_heads: int = 8,
                 n_deformable_points: int = 4, ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dim = dim

        # Global branch: Masked Sparse Linear Attention on sparse tokens
        self.global_norm = nn.LayerNorm(dim)
        self.msla = MaskedSparseLinearAttention(dim, n_heads, dropout)

        # Local branch: Deformable sampling from dense feature map
        self.local_norm = nn.LayerNorm(dim)
        self.deform_sampler = DeformableFeatureSampler(dim, n_deformable_points, dropout)

        # FFN for final fusion
        self.fusion_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, X_sparse: torch.Tensor, dense_feat: torch.Tensor,
                positions: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            X_sparse: [B, N_s, dim] retained sparse tokens after FGDP
            dense_feat: [B, dim, H, W] original dense feature map (for deformable sampling)
            positions: [B, N_s, 2] normalized coordinates of retained tokens
            mask: [B, N_s] optional binary keep mask
        Returns:
            ~S: [B, N_s, dim] enhanced sparse features
        """
        # Global semantic branch (operates on sparse tokens)
        X_global = self.msla(self.global_norm(X_sparse), mask=mask)

        # Local geometric branch (samples from dense feature map)
        X_local = self.deform_sampler(
            self.local_norm(X_sparse), dense_feat, positions, mask=mask)

        # Fuse: ~S = X_sparse + FFN(X_global + X_local)
        fused = X_global + X_local
        out = X_sparse + self.ffn(self.fusion_norm(fused))

        return out

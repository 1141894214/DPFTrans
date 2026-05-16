"""
VCBL: Vision-to-Class/Box Contrastive Learning Decoder.

Language-guided multimodal decoder that jointly exploits visual and
textual pathological representations. Includes:
- Multi-head self-attention on queries
- Deformable cross-attention on fused visual memory
- Text cross-attention for pathological semantic injection
- Contrastive classification in shared vision-language space
- MLP-based bounding box regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from typing import Tuple, Optional


class DeformableCrossAttention(nn.Module):
    """Deformable cross-attention for extracting visual evidence."""

    def __init__(self, dim: int, n_heads: int = 8, n_points: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = dim // n_heads

        self.q_proj = nn.Linear(dim, dim)
        self.sampling_offsets = nn.Linear(dim, n_points * 2)
        self.attention_weights = nn.Linear(dim, n_points)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                memory_spatial_shape: Tuple[int, int],
                ref_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N_q, D = query.shape

        if ref_points is None:
            ref_points = torch.rand(B, N_q, 2, device=query.device)

        # Predict offsets [B, N_q, n_points, 2] and weights [B, N_q, n_points]
        offsets = self.sampling_offsets(query).reshape(B, N_q, self.n_points, 2)
        offsets = offsets * 0.25

        attn_weights = F.softmax(
            self.attention_weights(query).reshape(B, N_q, self.n_points), dim=-1)

        # Prepare memory as 2D spatial grid
        H_m, W_m = memory_spatial_shape
        D_m = memory.shape[-1]
        mem_2d = memory.reshape(B, H_m, W_m, D_m).permute(0, 3, 1, 2)

        # Sampling positions in [-1, 1]
        ref = ref_points.unsqueeze(2)  # [B, N_q, 1, 2]
        sample_pos = ref + offsets  # [B, N_q, P, 2]
        sample_pos[..., 0] = sample_pos[..., 0] * 2 - 1
        sample_pos[..., 1] = sample_pos[..., 1] * 2 - 1
        sample_pos = sample_pos.reshape(B, N_q * self.n_points, 1, 2)

        # Sample from memory grid
        sampled = F.grid_sample(mem_2d, sample_pos, mode='bilinear',
                                 padding_mode='zeros', align_corners=False)
        sampled = sampled.squeeze(-1)  # [B, D_m, N_q*P]
        sampled = sampled.reshape(B, D_m, N_q, self.n_points)
        sampled = sampled.permute(0, 2, 3, 1)  # [B, N_q, P, D_m]

        # Weighted aggregation
        attn_weights = attn_weights.unsqueeze(-1)  # [B, N_q, P, 1]
        out = (sampled * attn_weights).sum(dim=2)  # [B, N_q, D_m]

        # Project to target dim if needed
        if D_m != D:
            if not hasattr(self, 'mem_proj'):
                self.mem_proj = nn.Linear(D_m, D, bias=False).to(query.device)
            out = self.mem_proj(out)

        out = self.dropout(self.out_proj(out))
        return out + query


class DecoderLayer(nn.Module):
    """
    A single layer of the language-guided multimodal decoder.

    Q_l₁ = LN(Q_{l-1} + MSA(Q_{l-1}))
    Q_l₂ = LN(Q_l₁ + DeformCrossAttn(Q_l₁, S_fusion))
    Q_l₃ = LN(Q_l₂ + CrossAttn(Q_l₂, E_text))
    Q_l   = LN(Q_l₃ + FFN(Q_l₃))
    """

    def __init__(self, dim: int = 256, n_heads: int = 8,
                 n_deformable_heads: int = 8, n_deformable_points: int = 4,
                 ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__()

        # Self-attention
        self.self_attn_norm = nn.LayerNorm(dim)
        self.self_attn = MultiheadAttention(dim, n_heads, dropout=dropout,
                                              batch_first=True)

        # Deformable cross-attention to visual memory
        self.deform_cross_norm = nn.LayerNorm(dim)
        self.deform_cross = DeformableCrossAttention(
            dim, n_deformable_heads, n_deformable_points, dropout,
        )

        # Text cross-attention
        self.text_cross_norm = nn.LayerNorm(dim)
        self.text_cross = MultiheadAttention(dim, n_heads, dropout=dropout,
                                              batch_first=True)

        # FFN
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, visual_memory: torch.Tensor,
                text_embeds: torch.Tensor,
                memory_spatial_shape: Tuple[int, int]) -> torch.Tensor:
        # Self-attention
        q = self.self_attn_norm(query)
        q = self.self_attn(q, q, q)[0] + query

        # Deformable cross-attention
        q2 = self.deform_cross_norm(q)
        q = self.deform_cross(q2, visual_memory, memory_spatial_shape) + q

        # Text cross-attention
        q3 = self.text_cross_norm(q)
        q = self.text_cross(q3, text_embeds, text_embeds)[0] + q

        # FFN
        q = self.ffn(self.ffn_norm(q)) + q

        return q


class TextPrototypeEncoder(nn.Module):
    """
    Encodes category prompt strings into pathological class prototype embeddings.

    In practice, this can be a frozen text encoder (e.g., CLIP text encoder)
    or a learned embedding table.
    """

    def __init__(self, num_classes: int, text_dim: int = 512, vision_dim: int = 256):
        super().__init__()
        # Learnable text prototypes (one per category)
        self.prototypes = nn.Parameter(torch.randn(num_classes, text_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # Project to vision space
        self.proj = nn.Sequential(
            nn.Linear(text_dim, vision_dim),
            nn.LayerNorm(vision_dim),
        )

    def forward(self) -> torch.Tensor:
        return self.proj(self.prototypes)  # [num_classes, vision_dim]


class VCBLDecoder(nn.Module):
    """
    Vision-to-Class/Box Contrastive Learning Decoder.

    After L decoder layers:
    - Classification: similarity between Q_L and text prototypes
    - Box regression: MLP on Q_L
    """

    def __init__(self, dim: int = 256, n_heads: int = 8,
                 n_deformable_heads: int = 8, n_deformable_points: int = 4,
                 ffn_dim: int = 1024, num_layers: int = 3, num_queries: int = 300,
                 num_classes: int = 3, text_dim: int = 512, dropout: float = 0.1,
                 init_temperature: float = 0.07):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, dim)
        nn.init.xavier_uniform_(self.query_embed.weight)

        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(dim, n_heads, n_deformable_heads, n_deformable_points,
                         ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        # Text prototype encoder
        self.text_encoder = TextPrototypeEncoder(num_classes, text_dim, dim)

        # Learnable temperature for contrastive classification
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1.0 / init_temperature)),
        )

        # Box regression head
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 4),
        )
        self.bbox_head[-1].weight.data.zero_()
        self.bbox_head[-1].bias.data.copy_(torch.tensor([0.0, 0.0, 1.0, 1.0]))

    def forward(self, visual_memory: torch.Tensor,
                memory_spatial_shape: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_memory: [B, N_m, dim] fused sparse visual memory
            memory_spatial_shape: (H, W) of visual memory
        Returns:
            class_logits: [B, N_q, num_classes]
            boxes: [B, N_q, 4] (cx, cy, w, h) normalized
            queries: [B, N_q, dim] final query embeddings
        """
        B = visual_memory.shape[0]

        # Initialize queries
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        # Get text prototypes
        text_embeds = self.text_encoder()  # [num_classes, dim]

        # Expand text for batch
        text_embeds_batch = text_embeds.unsqueeze(0).expand(B, -1, -1)

        # Decoder layers
        for layer in self.layers:
            query = layer(query, visual_memory, text_embeds_batch,
                         memory_spatial_shape)

        # Classification via contrastive similarity
        class_logits = self._contrastive_classify(query, text_embeds)

        # Box regression
        boxes = self.bbox_head(query).sigmoid()

        return class_logits, boxes, query

    def _contrastive_classify(self, queries: torch.Tensor,
                               text_protos: torch.Tensor) -> torch.Tensor:
        r"""
        Compute classification logits via normalized dot product.

        S = \hat{Q} \hat{T}^T / tau
        """
        # L2-normalize
        Q_norm = F.normalize(queries, p=2, dim=-1)
        T_norm = F.normalize(text_protos, p=2, dim=-1)

        # Scaled similarity
        tau = self.logit_scale.exp()
        logits = torch.matmul(Q_norm, T_norm.t()) / tau  # [B, N_q, num_classes]

        return logits

    def get_text_prototypes(self) -> torch.Tensor:
        return self.text_encoder()

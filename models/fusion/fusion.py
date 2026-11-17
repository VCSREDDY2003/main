"""
Multimodal feature fusion (Section 3.6, Eq. 3.15-3.19; Eq. 4.10).

    F_v   = f_CNN(X_i)                                  (3.15)  visual features
    H_c   = Encoder(e_1, ..., e_L)                      (3.16)  caption encoding
    A_vc  = Softmax(F_v W_q (H_c W_k)^T / sqrt(d))      (3.17)  cross-modal attention
    F_cv  = A_vc (H_c W_v)                              (3.18)  text-conditioned visual context
    F_fuse = LayerNorm(F_v + F_cv)                      (3.19)  fused static representation

    Z_i = F_fuse (+) G(z_i, y_i) (+) h_T                (4.10)  full multimodal state
                                                                  (visual+text, GAN, temporal)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """A_vc = Softmax(F_v W_q (H_c W_k)^T / sqrt(d)) ; F_cv = A_vc (H_c W_v)  (Eq. 3.17-3.18)."""

    def __init__(self, visual_dim: int, text_dim: int, proj_dim: int = 256):
        super().__init__()
        self.proj_dim = proj_dim
        self.q_proj = nn.Linear(visual_dim, proj_dim)
        self.k_proj = nn.Linear(text_dim, proj_dim)
        self.v_proj = nn.Linear(text_dim, proj_dim)
        self.out_proj = nn.Linear(proj_dim, visual_dim)

    def forward(self, f_v: torch.Tensor, h_c: torch.Tensor,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        f_v: (B, visual_dim)        pooled visual feature
        h_c: (B, L, text_dim)       per-token caption encoding
        text_mask: (B, L) 1 for real tokens, 0 for padding
        returns F_cv: (B, visual_dim)
        """
        q = self.q_proj(f_v).unsqueeze(1)               # (B, 1, P)
        k = self.k_proj(h_c)                            # (B, L, P)
        v = self.v_proj(h_c)                            # (B, L, P)

        scores = torch.matmul(q, k.transpose(1, 2)) / (self.proj_dim ** 0.5)  # (B, 1, L)
        if text_mask is not None:
            scores = scores.masked_fill(text_mask.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)            # (B, 1, L)

        f_cv = torch.matmul(attn, v).squeeze(1)         # (B, P)
        return self.out_proj(f_cv)                      # (B, visual_dim)


class MultimodalFusion(nn.Module):
    """F_fuse = LayerNorm(F_v + F_cv)  (Eq. 3.19)."""

    def __init__(self, visual_dim: int = 512, text_dim: int = 256, proj_dim: int = 256):
        super().__init__()
        self.cross_attn = CrossModalAttention(visual_dim, text_dim, proj_dim)
        self.norm = nn.LayerNorm(visual_dim)

    def forward(self, f_v: torch.Tensor, h_c: torch.Tensor,
                text_mask: torch.Tensor | None = None) -> torch.Tensor:
        f_cv = self.cross_attn(f_v, h_c, text_mask)
        return self.norm(f_v + f_cv)


class MultimodalStateBuilder(nn.Module):
    """
    Z_i = F_fuse (+) G_feat (+) h_T  (Eq. 4.10)

    Projects and concatenates the fused visual-text representation, an
    (optional) GAN-derived feature vector, and the LSTM temporal summary
    h_T into a single fusion-dimensional state Z_i used by the LMM decoder.
    """

    def __init__(self, fuse_dim: int = 512, gan_dim: int = 512,
                 lstm_dim: int = 256, fusion_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        # Stored so callers (e.g. the disable_gan/disable_lstm ablation
        # switches in IATGAMModel.forward) can build a correctly-shaped
        # zero tensor for a disabled branch without duplicating these dims.
        self.fuse_dim = fuse_dim
        self.gan_dim = gan_dim
        self.lstm_dim = lstm_dim
        self.proj = nn.Sequential(
            nn.Linear(fuse_dim + gan_dim + lstm_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
        )

    def forward(self, f_fuse: torch.Tensor, gan_feat: torch.Tensor,
                h_T: torch.Tensor) -> torch.Tensor:
        z = torch.cat([f_fuse, gan_feat, h_T], dim=-1)
        return self.proj(z)

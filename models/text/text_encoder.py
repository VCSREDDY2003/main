"""
Caption representation & semantic encoding (Section 3.4, Eq. 3.2-3.5).

    C_i = {w_1, ..., w_L}                (3.2)
    e_t = phi(w_t) in R^d                (3.3)
    H_c = Encoder(e_1, ..., e_L)         (3.4)   <- Transformer encoder
    L_cap = -sum_t log P(w_t | w_<t)     (3.5)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encoding p_t (Eq. 3.9: e~_t = e_t + p_t)."""

    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                              * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class CaptionEncoder(nn.Module):
    """Transformer-based caption encoder producing token-level embeddings H_c."""

    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 8,
                 max_len: int = 32, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = PositionalEncoding(embed_dim, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, token_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        token_ids:      (B, L) integer token ids (Eq. 3.2)
        attention_mask: (B, L) 1 for real tokens, 0 for padding
        returns H_c:    (B, L, embed_dim)  (Eq. 3.4)
        """
        x = self.token_embed(token_ids)          # e_t = phi(w_t)  (Eq. 3.3)
        x = self.pos_embed(x)                     # e~_t = e_t + p_t (Eq. 3.9)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def caption_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                      ignore_index: int = 0) -> torch.Tensor:
        """L_cap (Eq. 3.5): cross-entropy over the vocabulary for next-token prediction."""
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
            ignore_index=ignore_index,
        )

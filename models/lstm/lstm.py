"""
Temporal LSTM module (Section 3.5, Eq. 3.20-3.27).

Standard LSTM gating applied over a sequence of per-frame fused features
Z_1, ..., Z_T (Eq. 3.15-3.19 -> Eq. 4.10):

    i_t = sigma(W_i [h_{t-1}, Z_t] + b_i)        (3.20)
    f_t = sigma(W_f [h_{t-1}, Z_t] + b_f)        (3.21)
    o_t = sigma(W_o [h_{t-1}, Z_t] + b_o)        (3.22)
    g_t = tanh(W_g [h_{t-1}, Z_t] + b_g)         (3.23)
    c_t = f_t * c_{t-1} + i_t * g_t              (3.24)
    h_t = o_t * tanh(c_t)                        (3.25)

The final hidden state h_T (Eq. 3.26) is the temporal summary fed into the
fusion/LMM stage (Eq. 4.10):

    h_T = LSTM(Z_1, ..., Z_T)                    (3.26)

and the per-step outputs support a temporal consistency loss (Eq. 3.27):

    L_temporal = (1/(T-1)) * sum_t || h_t - h_{t-1} ||^2
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalLSTM(nn.Module):
    """Multi-layer LSTM over per-frame fused features (Eq. 3.20-3.26)."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True, bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_dim = out_dim

    def forward(self, z_seq: torch.Tensor):
        """
        z_seq: (B, T, input_dim) sequence of per-frame fused features Z_t
        returns:
          h_seq: (B, T, output_dim)  per-step hidden states h_1..h_T
          h_T:   (B, output_dim)     final hidden state (Eq. 3.26)
        """
        h_seq, _ = self.lstm(z_seq)
        h_T = h_seq[:, -1, :]
        return h_seq, h_T

    def temporal_consistency_loss(self, h_seq: torch.Tensor) -> torch.Tensor:
        """L_temporal (Eq. 3.27): penalize abrupt changes between consecutive
        hidden states to encourage smooth temporal dynamics."""
        if h_seq.size(1) < 2:
            return torch.zeros((), device=h_seq.device, dtype=h_seq.dtype)
        diffs = h_seq[:, 1:, :] - h_seq[:, :-1, :]
        return diffs.pow(2).sum(dim=-1).mean()

"""
GAM Memory Bank -- the mechanism that makes "GAM" (Generative Adversarial
Memory) an accurate name instead of a plain conditional GAN with no memory.

Design (see plan discussed 2026-09-16):
    e_i = 1 - sigmoid(D(X_i, y_i))          per-sample discriminator "unrealism"
                                              error for the REAL image of class y_i
    M[c] <- (1-gamma) * M[c] + gamma * mean(e_i for i with y_i == c)   (EMA)

    alpha_scale_i = sigmoid(M[y_i])          fed into IAT as `class_weight`
                                              (IAT already exposes this hook)

M is a real persistent buffer carried across training steps -- not
recomputed from the current batch alone -- so classes the discriminator has
found persistently hard accumulate a higher alpha_scale over time, and IAT's
invasive-perturbation strength for that class grows accordingly. This is the
literal coupling between the "GAN" and "Memory" halves of the module name.

gamma -> 0 freezes the memory (no coupling); gamma = 1 makes M track only the
current batch (no real memory, reduces to the previous memoryless design).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GAMMemoryBank(nn.Module):
    def __init__(self, num_classes: int = 5, gamma: float = 0.1, init_value: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.register_buffer("memory", torch.full((num_classes,), init_value))

    @torch.no_grad()
    def update(self, d_real_logits: torch.Tensor, labels: torch.Tensor) -> None:
        """d_real_logits: (B,) raw discriminator logits on REAL images.
        labels: (B,) integer class ids in [0, num_classes)."""
        error = 1.0 - torch.sigmoid(d_real_logits.detach()).flatten()
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                batch_mean = error[mask].mean()
                self.memory[c] = (1 - self.gamma) * self.memory[c] + self.gamma * batch_mean

    def alpha_scale(self, labels: torch.Tensor) -> torch.Tensor:
        """Returns (B,) multiplier in (0,1), one per sample, from its class's memory."""
        return torch.sigmoid(self.memory[labels])

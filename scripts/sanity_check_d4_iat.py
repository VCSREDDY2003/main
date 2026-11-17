"""Phase B synthetic sanity check (Phase A redesign, 2026-11, corrected).

First attempt (preserved in git history / session log) tested whether the
new attention module is more stable under D4 rotation/reflection than the
old one. That test returned 0.0 for BOTH modules at random init -- token-
wise self-attention with no positional encoding is already exactly
permutation-equivariant, so there was never an equivariance bug to fix.
That diagnosis is wrong and is not used below.

Corrected test: the real re-diagnosed bottleneck is spatial SCATTER, not
rotational instability -- the old attention map can select disconnected,
anatomically-incoherent pixels. This script checks:
  1. A scattered (multi-disconnected-blob) saliency pattern produces a
     meaningfully nonzero total-variation (TV) penalty under
     CoherentSaliencyMap.consistency_loss -- i.e. the penalty actually
     fires on the failure mode it's meant to catch.
  2. Directly minimizing that penalty via a short gradient-descent loop
     (no task loss, just the penalty itself) measurably smooths the
     resulting attention map -- i.e. the loss is actually optimizable
     toward a more coherent region, not just nonzero-but-inert.
  3. The OLD ImageSelfAttentionMap has no such mechanism at all (no
     consistency_loss method exists on it), so nothing in the old design
     could ever correct a scattered map.

Cheap, no GPU/dataset required -- run before touching real data.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.iat.iat import CoherentSaliencyMap, ImageSelfAttentionMap


def make_scattered_grid(batch=4, channels=16, h=8, w=8, seed=0):
    """Two small, far-apart hot patches -- a scattered/disconnected
    salient pattern, the failure mode the coherence penalty should flag."""
    torch.manual_seed(seed)
    grid = torch.randn(batch, channels, h, w) * 0.1
    grid[:, :, 0:1, 0:1] += 5.0   # corner patch
    grid[:, :, 6:7, 6:7] += 5.0   # opposite corner patch
    return grid


def main():
    feat_grid = make_scattered_grid()
    b, c, h, w = feat_grid.shape
    tokens = feat_grid.flatten(2).transpose(1, 2)

    torch.manual_seed(42)
    new_attn = CoherentSaliencyMap(visual_dim=c)

    print("=== Phase B synthetic sanity check (corrected: spatial coherence, not equivariance) ===")

    # Check 1: penalty fires on a scattered pattern.
    new_attn.eval()
    with torch.no_grad():
        tv_before = new_attn.consistency_loss(tokens, (h, w)).item()
    print(f"TV coherence penalty on scattered input (untrained weights): {tv_before:.6f}")

    # Check 2: the penalty is actually optimizable -- minimize it directly
    # (no task loss) for a short loop and confirm it decreases.
    new_attn.train()
    opt = torch.optim.Adam(new_attn.parameters(), lr=0.05)
    losses = []
    for step in range(50):
        opt.zero_grad()
        loss = new_attn.consistency_loss(tokens, (h, w))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"TV penalty after 50 optimization steps: {losses[-1]:.6f} "
          f"(started at {losses[0]:.6f})")
    improved = losses[-1] < losses[0] * 0.5
    print(f"Penalty decreased by >50%: {improved}")

    # Check 3: old design has no mechanism to do this at all.
    old_attn = ImageSelfAttentionMap(visual_dim=c)
    has_consistency_method = hasattr(old_attn, "consistency_loss")
    print(f"Old ImageSelfAttentionMap has a consistency_loss method: {has_consistency_method}")

    print()
    if tv_before > 1e-4 and improved and not has_consistency_method:
        print("PASS: the penalty (a) fires on a real scattered pattern, "
              "(b) is actually minimizable by gradient descent, and "
              "(c) has no equivalent in the old design. This is a real, "
              "testable mechanism difference -- proceed to Phase C/D.")
    else:
        print("FAIL: one or more conditions did not hold -- do not proceed "
              "to real-dataset training on this design yet.")


if __name__ == "__main__":
    main()

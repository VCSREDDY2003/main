"""
Composite training objective (Eq. 3.28-3.31).

    L_cap      = -sum_t log P(w_t | w_<t)                          (3.5 / 3.28a)
    L_summary  = -sum_t log P(s_t | s_<t)                          (3.28b)
    L_cls      = CrossEntropy(y_hat, y)                            (3.29)
    L_GAN      = E[log D(X|y)] + E[log(1 - D(G(z|y), y))]          (3.13 / 3.30a)
    L_temporal = (1/(T-1)) sum_t || h_t - h_{t-1} ||^2             (3.27 / 3.30b)

    L_total = L_cap + L_summary + L_cls
              + lambda_1 * L_GAN + lambda_2 * L_temporal
              + lambda_3 * L_attn_entropy                          (3.31, v3 update)

lambda_1, lambda_2 come from config.LossWeights / Hyperparameters
(lambda_gan, lambda_temporal -- Table 3.1/3.2).

L_attn_entropy (v3, multi-head spatial attention regularizer):
    negative mean entropy of the per-head, per-timestep spatial attention
    distribution produced by models/lmm/lmm.py:SpatialAttention, i.e.

        L_attn_entropy = -mean_{b,t,h} [ -sum_n a_{b,t,h,n} log a_{b,t,h,n} ]

    Minimising L_attn_entropy maximises attention entropy, discouraging the
    attention weights from collapsing onto a single spatial location (or an
    identical location across heads/timesteps) -- the failure mode
    diagnosed for the beam-5 regression with the pretrained-CNN backbone
    (see lmm.py:SpatialAttention docstring). A small weight is used since
    this is a regularizer, not a primary objective: it should smooth out
    collapse without preventing the model from confidently attending to a
    genuinely informative region.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def caption_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 0,
                  label_smoothing: float = 0.0) -> torch.Tensor:
    """L_cap / L_summary (Eq. 3.5, 3.28a/b): token-level cross-entropy.

    label_smoothing caps the per-token loss so a single near-zero-probability
    prediction on a rare/heterogeneous validation batch can't produce a
    triple-digit per-token CE that dominates the batch mean.
    """
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


def classification_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1,
                         weight: torch.Tensor | None = None, label_smoothing: float = 0.0,
                         focal_gamma: float = 0.0) -> torch.Tensor:
    """L_cls (Eq. 3.29). Samples with label == -1 (no ground-truth class) are ignored.

    cross_entropy with ignore_index returns nan when *every* sample in the
    batch is ignored (mean over zero elements), which happens for batches
    drawn entirely from unlabeled sources (ROCO/ROCOv2). Fall back to a
    zero loss (still graph-connected) in that case.

    label_smoothing bounds the per-sample loss (avoids the unbounded
    -log(p) blow-up when an overfit head is confidently wrong on a
    validation sample), and `weight` (inverse-class-frequency) counters
    the heavy skew toward class 4 ("Other/video", ~59% of training data).

    focal_gamma > 0 applies the focal-loss modulating factor (1-p_t)^gamma
    on top of the (still class-weighted) cross-entropy (2026-11, binary-
    minority-recall fix): class weighting alone still lets easy, already-
    confident majority-class examples dominate the gradient by sheer
    volume; focal loss down-weights those and keeps gradient pressure on
    the examples -- disproportionately minority-class -- the model is
    still getting wrong. 0.0 (default) reproduces plain weighted CE
    exactly, so this is backward-compatible with every existing run.
    """
    mask = targets != ignore_index
    if not mask.any():
        return logits.sum() * 0.0
    if focal_gamma <= 0:
        return nn.functional.cross_entropy(logits[mask], targets[mask], weight=weight, label_smoothing=label_smoothing)
    ce = nn.functional.cross_entropy(logits[mask], targets[mask], weight=weight,
                                       label_smoothing=label_smoothing, reduction="none")
    pt = torch.exp(-ce.detach())  # model's probability on the true class, detached (focal factor is not learned)
    focal_factor = (1.0 - pt).clamp_min(0.0) ** focal_gamma
    return (focal_factor * ce).mean()


def attention_entropy_loss(attn: torch.Tensor | None, eps: float = 1e-8) -> torch.Tensor | None:
    """Negative mean entropy of a spatial attention tensor (B, T, heads, N).

    Returns None if attn is None (e.g. the no-spatial-attention ablation,
    which has nothing to regularize). Minimising the returned value
    maximises attention entropy -> discourages collapse to a single
    location/head pattern.
    """
    if attn is None:
        return None
    entropy = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(dim=-1)  # (B, T, heads)
    return -entropy.mean()


class CompositeLoss(nn.Module):
    """L_total (Eq. 3.31, v3: + lambda_3 * L_attn_entropy)."""

    def __init__(self, lambda_gan: float = 0.5, lambda_temporal: float = 0.5, pad_id: int = 0,
                 class_weights: dict[str, torch.Tensor] | None = None, label_smoothing: float = 0.1,
                 lambda_attn_entropy: float = 0.01, lambda_iat_consistency: float = 0.1,
                 focal_gamma: float = 0.0):
        super().__init__()
        self.lambda_gan = lambda_gan
        self.lambda_temporal = lambda_temporal
        self.pad_id = pad_id
        self.class_weights = class_weights or {}
        self.label_smoothing = label_smoothing
        self.lambda_attn_entropy = lambda_attn_entropy
        # Phase A redesign (2026-11): weight for the D4-equivariance penalty
        # on IAT's saliency map (models/iat/iat.py:CoherentSaliencyMap).
        self.lambda_iat_consistency = lambda_iat_consistency
        # Binary-minority-recall fix (2026-11): see classification_loss's
        # focal_gamma docstring.
        self.focal_gamma = focal_gamma

    def forward(self, outputs: dict, targets: dict, gan_loss: torch.Tensor | None = None,
                temporal_loss: torch.Tensor | None = None) -> dict:
        """
        outputs: dict from LMMDecoder.forward (caption_logits, summary_logits, class_logits)
        targets: dict with caption_targets, summary_targets, class_targets
        gan_loss: scalar discriminator/generator loss (Eq. 3.13), or None to skip
        temporal_loss: scalar from TemporalLSTM.temporal_consistency_loss (Eq. 3.27), or None to skip

        returns dict of individual losses plus 'total'
        """
        l_cap = caption_loss(outputs["caption_logits"], targets["caption_targets"], self.pad_id,
                             label_smoothing=self.label_smoothing)
        l_summary = caption_loss(outputs["summary_logits"], targets["summary_targets"], self.pad_id,
                                  label_smoothing=self.label_smoothing)

        # Classification loss trains the IMAGE-ONLY heads (img_class_logits_*,
        # models/iatgam_model.py), not the old Z-based classifier_* heads.
        # Diagnosed 2026-09: Z is built from F_fuse, which mixes in caption
        # cross-attention, and since the 5-class labels are themselves
        # keyword-derived from captions, a classifier trained on Z learns to
        # read the caption instead of the image (verified: caption-shuffle
        # and caption-mask probes). img_class_logits_* read F_v directly,
        # before any caption cross-attention, so training on them from the
        # start (instead of bolting on a separate fine-tune after the fact)
        # gives a real image-only classifier out of the base training run.
        # Bug fix (2026-11, found via 3 separate real collapse runs): when
        # focal_gamma > 0, do NOT also pass the inverse-frequency class
        # weight -- focal loss is a replacement for hard class weighting,
        # not an addition to it. Stacking both corrections compounded the
        # push toward the minority class every time it was tried (sampler+
        # weight, weight+sampler+focal+capacity, and weight+focal alone
        # all independently collapsed to >90% recall on the former-
        # minority class within 2-3 epochs). Only one correction
        # mechanism is active at a time now.
        effective_weight = (lambda key: None if self.focal_gamma > 0 else self.class_weights.get(key))
        l_cls_binary = classification_loss(
            outputs["img_class_logits_binary"], targets["class_targets_binary"],
            weight=effective_weight("binary"), label_smoothing=self.label_smoothing,
            focal_gamma=self.focal_gamma)
        l_cls_3class = classification_loss(
            outputs["img_class_logits_3class"], targets["class_targets_3class"],
            weight=effective_weight("3class"), label_smoothing=self.label_smoothing,
            focal_gamma=self.focal_gamma)
        l_cls_5class = classification_loss(
            outputs["img_class_logits_5class"], targets["class_targets_5class"],
            weight=effective_weight("5class"), label_smoothing=self.label_smoothing,
            focal_gamma=self.focal_gamma)
        l_cls = l_cls_binary + l_cls_3class + l_cls_5class

        total = l_cap + l_summary + l_cls

        losses = {
            "caption": l_cap, "summary": l_summary, "classification": l_cls,
            "classification_binary": l_cls_binary,
            "classification_3class": l_cls_3class,
            "classification_5class": l_cls_5class,
        }

        if gan_loss is not None:
            total = total + self.lambda_gan * gan_loss
            losses["gan"] = gan_loss
        if temporal_loss is not None:
            total = total + self.lambda_temporal * temporal_loss
            losses["temporal"] = temporal_loss

        cap_attn_loss = attention_entropy_loss(outputs.get("caption_attn"))
        summ_attn_loss = attention_entropy_loss(outputs.get("summary_attn"))
        attn_losses = [x for x in (cap_attn_loss, summ_attn_loss) if x is not None]
        if attn_losses and self.lambda_attn_entropy > 0:
            l_attn_entropy = sum(attn_losses) / len(attn_losses)
            total = total + self.lambda_attn_entropy * l_attn_entropy
            losses["attn_entropy"] = l_attn_entropy

        iat_consistency = outputs.get("iat_consistency_loss")
        if iat_consistency is not None and self.lambda_iat_consistency > 0:
            total = total + self.lambda_iat_consistency * iat_consistency
            losses["iat_consistency"] = iat_consistency

        losses["total"] = total
        return losses

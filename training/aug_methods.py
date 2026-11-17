"""
Alternative image-augmentation methods for the IAT-vs-augmentation
comparison (Phase 2, classification-only scope, 2026-11).

Each function takes a batch of images (B, C, H, W) in [0, 1] and the batch's
5-class labels, and returns (aug_images, labels_a, labels_b, lam) so a single
training loop can handle both label-preserving methods (crop_flip,
randaugment, none) and label-mixing methods (mixup, cutmix) uniformly:
loss = lam * CE(logits, labels_a) + (1 - lam) * CE(logits, labels_b)
collapses to plain CE when lam == 1 and labels_a == labels_b.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms import v2


def no_aug(images: torch.Tensor, labels: torch.Tensor):
    return images, labels, labels, 1.0


def crop_flip(images: torch.Tensor, labels: torch.Tensor):
    """Standard random-resized-crop + horizontal flip, the conventional
    augmentation baseline requested for the IAT comparison table."""
    h, w = images.shape[-2:]
    tf = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(size=(h, w), scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
    ])
    return tf(images), labels, labels, 1.0


_RANDAUGMENT = v2.RandAugment(num_ops=2, magnitude=9)


def randaugment(images: torch.Tensor, labels: torch.Tensor):
    """RandAugment operates on uint8 tensors; convert, augment, convert back."""
    images_u8 = (images.clamp(0, 1) * 255).to(torch.uint8)
    aug_u8 = _RANDAUGMENT(images_u8)
    aug = aug_u8.to(images.dtype) / 255.0
    return aug, labels, labels, 1.0


def mixup(images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2):
    """MixUp (Zhang et al. 2018): convex-combine two samples and their labels."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    perm = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[perm]
    return mixed, labels, labels[perm], lam


def cutmix(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """CutMix (Yun et al. 2019): paste a random box from a second sample."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    perm = torch.randperm(images.size(0), device=images.device)
    h, w = images.shape[-2:]
    cut_rat = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(h * cut_rat), int(w * cut_rat)
    cy, cx = torch.randint(h, (1,)).item(), torch.randint(w, (1,)).item()
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, h)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, w)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
    # recompute lam to match the actual pasted area (standard CutMix practice)
    lam_adjusted = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))
    return mixed, labels, labels[perm], lam_adjusted


def mixed_cross_entropy(logits: torch.Tensor, labels_a: torch.Tensor,
                         labels_b: torch.Tensor, lam: float,
                         weight: torch.Tensor | None = None) -> torch.Tensor:
    """lam * CE(logits, a) + (1 - lam) * CE(logits, b); collapses to plain CE
    when lam == 1 and a == b (no_aug/crop_flip/randaugment)."""
    if lam == 1.0:
        return F.cross_entropy(logits, labels_a, weight=weight)
    return (lam * F.cross_entropy(logits, labels_a, weight=weight)
            + (1 - lam) * F.cross_entropy(logits, labels_b, weight=weight))


AUG_METHODS = {
    "none": no_aug,
    "cropflip": crop_flip,
    "mixup": mixup,
    "cutmix": cutmix,
    "randaugment": randaugment,
}

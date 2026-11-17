"""
Invasive Augmented Transform (IAT) module.

Implements Eq. (3.10), Eq. (4.10) and Algorithm 1 (image-only IAT, revised
2026-11 from the original Caption-Guided IAT) from the IAT-GAM paper.

    X_IAT = X + alpha * F(X, R)

where R are salient regions derived from an image self-attention map
(ImageSelfAttentionMap -- no caption/text input), F(.) injects controlled
synthetic perturbations into those regions, and alpha is adaptively shrunk
until SSIM(X, X_IAT) >= tau.

The caption-conditioned attention (CaptionAttentionMap, kept below for
reference/backward-compat with any code still importing it) required the
reference caption at inference, which every eval script supplied from
ground truth -- a confirmed test-time leak into the downstream image-only
classifier heads. InvasiveAugmentedTransform now uses ImageSelfAttentionMap
exclusively, which depends on nothing but the image.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_kernel(channels: int, kernel_size: int = 11, sigma: float = 1.5,
                      device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    kernel_2d = g.t() @ g
    kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    return kernel_2d


def ssim(x: torch.Tensor, y: torch.Tensor, c1: float = 0.01 ** 2,
         c2: float = 0.03 ** 2) -> torch.Tensor:
    """Per-sample SSIM between two batches of images in [0, 1], shape (B,C,H,W)."""
    channels = x.shape[1]
    kernel = _gaussian_kernel(channels, device=x.device, dtype=x.dtype)
    pad = kernel.shape[-1] // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=channels)

    sigma_x = F.conv2d(x * x, kernel, padding=pad, groups=channels) - mu_x ** 2
    sigma_y = F.conv2d(y * y, kernel, padding=pad, groups=channels) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=channels) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )
    return ssim_map.flatten(1).mean(dim=1)


class CaptionAttentionMap(nn.Module):
    """Computes A_i = Softmax(H_c . F_v^T) (Eq. 4.9, Algorithm-1 steps 3-5)."""

    def __init__(self, visual_dim: int, text_dim: int, proj_dim: int = 256):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)

    def forward(self, visual_tokens: torch.Tensor, text_tokens: torch.Tensor,
                spatial_hw: tuple[int, int]) -> torch.Tensor:
        """
        visual_tokens: (B, N, visual_dim)  flattened spatial feature map (N = H*W)
        text_tokens:   (B, L, text_dim)    encoded caption tokens H_c
        returns:       (B, 1, H, W) spatial attention map in [0, 1]
        """
        v = self.visual_proj(visual_tokens)               # (B, N, P)
        t = self.text_proj(text_tokens)                    # (B, L, P)
        scores = torch.matmul(t, v.transpose(1, 2))        # (B, L, N)
        attn = torch.softmax(scores, dim=-1)               # (B, L, N)
        attn = attn.mean(dim=1)                            # (B, N)  pooled over tokens
        h, w = spatial_hw
        attn_map = attn.view(-1, 1, h, w)
        # normalize to [0,1]
        attn_map = (attn_map - attn_map.amin(dim=(2, 3), keepdim=True)) / (
            attn_map.amax(dim=(2, 3), keepdim=True)
            - attn_map.amin(dim=(2, 3), keepdim=True) + 1e-8
        )
        return attn_map


class ImageSelfAttentionMap(nn.Module):
    """Image-only replacement for CaptionAttentionMap (classification-only
    scope, 2026-11): computes a spatial saliency map from the visual
    feature map alone via single-head self-attention (query/key/value all
    projected from visual_tokens), with no text/caption input anywhere.

    Fixes a confirmed test-time leak: the previous CaptionAttentionMap
    required the reference caption's encoded tokens (H_c) to compute IAT's
    region mask, including at inference (generate()), where every eval
    script passed in the dataset's ground-truth caption. That let the
    downstream image-only classifier heads be indirectly caption-
    influenced via the IAT-augmented image they read, even though they
    never see caption text directly. This module removes that dependency
    structurally so IAT's region-selection is deployment-valid (image in,
    no side-channel text/label required) at both train and test time.
    """

    def __init__(self, visual_dim: int, proj_dim: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(visual_dim, proj_dim)
        self.key_proj = nn.Linear(visual_dim, proj_dim)
        self.scale = proj_dim ** -0.5

    def forward(self, visual_tokens: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        """
        visual_tokens: (B, N, visual_dim)  flattened spatial feature map (N = H*W)
        returns:       (B, 1, H, W) spatial attention/saliency map in [0, 1]
        """
        q = self.query_proj(visual_tokens)                 # (B, N, P)
        k = self.key_proj(visual_tokens)                    # (B, N, P)
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale   # (B, N, N)
        attn = torch.softmax(scores, dim=-1)                # (B, N, N)
        saliency = attn.mean(dim=1)                          # (B, N) each token's average incoming attention
        h, w = spatial_hw
        attn_map = saliency.view(-1, 1, h, w)
        attn_map = (attn_map - attn_map.amin(dim=(2, 3), keepdim=True)) / (
            attn_map.amax(dim=(2, 3), keepdim=True)
            - attn_map.amin(dim=(2, 3), keepdim=True) + 1e-8
        )
        return attn_map


class CoherentSaliencyMap(nn.Module):
    """Spatially-coherent replacement for ImageSelfAttentionMap (Phase A,
    2026-11 redesign; corrected 2026-11 after the synthetic sanity check in
    scripts/sanity_check_d4_iat.py).

    Diagnosis history: the original hypothesis was that plain dot-product
    self-attention over flattened visual tokens is unstable under rotation/
    reflection of the image (an "equivariance" bug). The synthetic sanity
    check falsified that: token-wise self-attention with no positional
    encoding is ALREADY exactly permutation-equivariant (rotating the grid
    just permutes which token is which; attention depends only on token
    content), so both the old and new module scored 0.0 consistency loss
    at random init -- there was never an equivariance bug to fix.

    The real, re-diagnosed bottleneck: the attention map has no SPATIAL
    LOCALITY/COHERENCE constraint. It can select scattered, disconnected
    pixels instead of one contiguous anatomical region -- that scattering
    is what makes it behave like a soft, structure-blind version of CutMix
    (consistent with CutMix being the worst-performing augmentation in our
    real Phase 2 results: pasting/perturbing across a true anatomical
    boundary destroys diagnostic signal). ACE-Net's "attention instability
    under weak supervision" finding is consistent with this: nothing in an
    unconstrained attention map stops it from being spatially scattered.

    Fix: `consistency_loss` below is now a **spatial coherence** penalty
    (total-variation of the saliency map), not an equivariance check --
    it penalizes the map for jumping between unrelated pixels and rewards
    one smooth, contiguous region. `forward()` is unchanged in cost from
    ImageSelfAttentionMap; the penalty is training-time only.
    """

    def __init__(self, visual_dim: int, proj_dim: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(visual_dim, proj_dim)
        self.key_proj = nn.Linear(visual_dim, proj_dim)
        self.scale = proj_dim ** -0.5

    def _saliency_from_grid(self, feat_grid: torch.Tensor) -> torch.Tensor:
        """feat_grid: (B, C, H, W) -> (B, 1, H, W) saliency in [0, 1]."""
        b, c, h, w = feat_grid.shape
        tokens = feat_grid.flatten(2).transpose(1, 2)          # (B, N, C)
        q = self.query_proj(tokens)
        k = self.key_proj(tokens)
        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        saliency = attn.mean(dim=1).view(b, 1, h, w)
        saliency = (saliency - saliency.amin(dim=(2, 3), keepdim=True)) / (
            saliency.amax(dim=(2, 3), keepdim=True)
            - saliency.amin(dim=(2, 3), keepdim=True) + 1e-8
        )
        return saliency

    def forward(self, visual_tokens: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        """Canonical (untransformed) saliency map -- same signature/cost as
        ImageSelfAttentionMap.forward, used for both training and inference."""
        h, w = spatial_hw
        feat_grid = visual_tokens.transpose(1, 2).view(-1, visual_tokens.shape[-1], h, w)
        return self._saliency_from_grid(feat_grid)

    def consistency_loss(self, visual_tokens: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        """Spatial-coherence (total-variation) penalty on the saliency map
        (training-time only, not used at inference; see class docstring
        for why this replaced an earlier, falsified equivariance penalty).

        TV(S) = mean(|S[:, :, 1:, :] - S[:, :, :-1, :]|) +
                mean(|S[:, :, :, 1:] - S[:, :, :, :-1]|)

        Minimising this encourages the saliency map to form one smooth,
        contiguous region instead of scattering across disconnected
        pixels -- the property that actually distinguishes a coherent
        anatomical-region selector from an arbitrary unconstrained map.
        """
        h, w = spatial_hw
        c = visual_tokens.shape[-1]
        feat_grid = visual_tokens.transpose(1, 2).view(-1, c, h, w)  # (B, C, H, W)
        sal = self._saliency_from_grid(feat_grid)  # (B, 1, h, w)

        tv_vertical = torch.mean(torch.abs(sal[:, :, 1:, :] - sal[:, :, :-1, :])) if h > 1 else sal.new_zeros(())
        tv_horizontal = torch.mean(torch.abs(sal[:, :, :, 1:] - sal[:, :, :, :-1])) if w > 1 else sal.new_zeros(())
        return tv_vertical + tv_horizontal


class _TransformLayer(nn.Module):
    """One IAT "transform layer": a depthwise 3x3 conv + pointwise 1x1 conv
    (channel-preserving, residual) refining the perturbation field. Stacking
    `num_layers` of these is the configurable-depth knob (7 vs 14) for the
    IAT-depth ablation."""

    def __init__(self, channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                                    groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pointwise(self.depthwise(x))
        out = self.act(self.norm(out))
        return x + out


class InvasivePerturbation(nn.Module):
    """F(X, R): learnable multi-scale perturbation generator (pixel/patch/global).

    `num_layers` controls the depth of the patch-level transform stack (the
    IAT-depth ablation: 7-layer vs 14-layer IAT). Each layer is a residual
    depthwise-separable conv (_TransformLayer); the pixel-level and global
    branches are unchanged so alpha/SSIM-gating behaviour stays comparable
    across depths.
    """

    def __init__(self, channels: int = 3, num_layers: int = 7):
        super().__init__()
        self.num_layers = num_layers
        self.pixel = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Tanh(),
        )
        self.patch_in = nn.Conv2d(channels, channels, kernel_size=7, padding=3,
                                   groups=channels)
        self.transform_layers = nn.ModuleList(
            [_TransformLayer(channels) for _ in range(num_layers)]
        )
        self.global_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pixel_noise = torch.randn_like(x) * 0.05
        f_pixel = self.pixel(x + pixel_noise)
        f_patch = torch.tanh(self.patch_in(x))
        for layer in self.transform_layers:
            f_patch = layer(f_patch)
        f_global = x * torch.tanh(self.global_scale)
        return f_pixel + f_patch + f_global


class InvasiveAugmentedTransform(nn.Module):
    """
    Full IAT module (Eq. 3.10 / 4.10 + Algorithm 1), image-only variant
    (classification-only scope, 2026-11; see ImageSelfAttentionMap above
    for why the caption-conditioned attention branch was removed).

    Given an image batch X and its visual feature map, produces X_aug with
    region-targeted, adaptively-bounded perturbations. No text/caption
    input anywhere, so the procedure is identical at train and test time.
    """

    def __init__(self, channels: int = 3, visual_dim: int = 512,
                 alpha_init: float = 0.3, ssim_threshold: float = 0.85,
                 beta: float = 0.9, max_adjust_steps: int = 6, num_layers: int = 7,
                 ssim_upper: float = 0.97, alpha_max: float = 1.5):
        super().__init__()
        self.attention = CoherentSaliencyMap(visual_dim)
        self.perturb = InvasivePerturbation(channels, num_layers=num_layers)
        self.alpha_init = alpha_init
        self.tau = ssim_threshold
        self.beta = beta
        self.max_adjust_steps = max_adjust_steps
        # 2026-11 fix: `forward`'s adjust loop below used to be one-directional
        # (alpha could only ever shrink), which real eval data showed produces
        # a near no-op (ssim_iat_vs_original=0.9797 on the real test set) --
        # any perturbation strong enough to dip below tau even once in the
        # loop gets ratcheted down and never recovers. ssim_upper/alpha_max
        # add the missing other direction: if SSIM is ABOVE ssim_upper (the
        # augmentation is barely perturbing anything), alpha is grown, not
        # just shrunk, up to alpha_max -- so IAT converges to the strongest
        # perturbation that still respects the [ssim_upper, 1] floor on
        # faithfulness implied by tau, instead of collapsing toward alpha=0.
        self.ssim_upper = ssim_upper
        self.alpha_max = alpha_max

    def forward(self, x: torch.Tensor, visual_tokens: torch.Tensor,
                spatial_hw: tuple[int, int],
                class_weight: torch.Tensor | None = None):
        """
        x:             (B, C, H, W) input images, values in [0, 1]
        visual_tokens: (B, N, visual_dim)
        spatial_hw:    (h, w) of the visual token grid
        class_weight:  optional (B,) per-sample augmentation intensity
                       scaling (Algorithm 1, step 8: g(Y_i, class_imbalance))
        returns: x_aug (B, C, H, W), attention_map (B, 1, H, W)
        """
        attn_map = self.attention(visual_tokens, spatial_hw)
        mask = F.interpolate(attn_map, size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        mask = (mask >= mask.mean(dim=(2, 3), keepdim=True)).float()

        alpha = torch.full((x.shape[0], 1, 1, 1), self.alpha_init,
                            device=x.device, dtype=x.dtype)
        if class_weight is not None:
            alpha = alpha * class_weight.view(-1, 1, 1, 1)

        x_aug = x
        for _ in range(self.max_adjust_steps):
            perturbation = self.perturb(x) * mask
            x_aug = torch.clamp(x + alpha * perturbation, 0.0, 1.0)
            scores = ssim(x, x_aug)
            too_different = scores < self.tau
            too_similar = scores > self.ssim_upper
            if not (too_different.any() or too_similar.any()):
                break
            grow = 1.0 / self.beta
            adjust = torch.ones_like(scores)
            adjust = torch.where(too_different, torch.full_like(scores, self.beta), adjust)
            adjust = torch.where(too_similar, torch.full_like(scores, grow), adjust)
            alpha = (alpha * adjust.view(-1, 1, 1, 1)).clamp(min=1e-3, max=self.alpha_max)

        return x_aug, attn_map

    def attention_consistency_loss(self, visual_tokens: torch.Tensor,
                                     spatial_hw: tuple[int, int]) -> torch.Tensor:
        """D4-equivariance auxiliary loss for the saliency map (Phase A).
        Training-time only; the caller decides whether/how to weight it."""
        return self.attention.consistency_loss(visual_tokens, spatial_hw)

"""
Visual feature extractor: DL-7 / DL-17 / DL-39 layer CNN backbones
(Section 4.3.1 - 4.3.3, Eq. 4.12, Eq. 3.15).

    F_i = sigma(W * X_i + b)            (Eq. 4.12)
    F_v = f_CNN(X_i)                    (Eq. 3.15)

The 39-layer "GAM" backbone is the deepest variant and is the one combined
with the GAN + LSTM modules (Eq. 4.10):

    Z_i = f_CNN^39(X_i) (+) G(z_i, y_i) (+) h_T
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class ResidualBlock(nn.Module):
    """Two 3x3 convs with a residual/shortcut connection (Eq. 4.8 layer-wise form)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class AttentionGate(nn.Module):
    """Lightweight channel-attention used by the iterative multimodal-fusion
    layers described in Section 4.3.2 (semantic embeddings fed at multiple
    depths)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)), nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        s = self.pool(x).view(b, c)
        scale = self.fc(s).view(b, c, 1, 1)
        return x * scale


# Depth -> (stage_channels, blocks_per_stage)
# Each ResidualBlock contributes 2 conv layers; stem contributes 1.
_DEPTH_CONFIGS = {
    7: ([32, 64], [1, 1]),                       # 1 + 2*(1+1)  = 5  -> "7-layer" incl. fc layers
    17: ([32, 64, 128, 256], [1, 1, 2, 2]),       # 1 + 2*(1+1+2+2) = 13 -> "17-layer" incl. fc layers
    39: ([32, 64, 128, 256, 512], [2, 2, 4, 4, 6]),  # deep GAM backbone
}


class MultiDepthCNN(nn.Module):
    """Configurable CNN backbone with DL-7 / DL-17 / DL-39 presets."""

    def __init__(self, depth: int = 39, in_channels: int = 3,
                 feature_dim: int = 512, with_attention_gates: bool = True):
        super().__init__()
        if depth not in _DEPTH_CONFIGS:
            raise ValueError(f"Unsupported depth {depth}; choose from {list(_DEPTH_CONFIGS)}")
        channels, blocks_per_stage = _DEPTH_CONFIGS[depth]
        self.depth = depth

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        stages = []
        gates = []
        in_ch = channels[0]
        for stage_idx, (out_ch, n_blocks) in enumerate(zip(channels, blocks_per_stage)):
            blocks = []
            for b in range(n_blocks):
                stride = 2 if (b == 0 and stage_idx > 0) else 1
                blocks.append(ResidualBlock(in_ch, out_ch, stride=stride))
                in_ch = out_ch
            stages.append(nn.Sequential(*blocks))
            gates.append(AttentionGate(out_ch) if with_attention_gates else nn.Identity())
        self.stages = nn.ModuleList(stages)
        self.gates = nn.ModuleList(gates)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, feature_dim)
        # Raw spatial feature-map channel count (last stage's out_ch), which
        # only equals `feature_dim` by coincidence at depth=39 (both 512).
        # Modules that consume the raw map (IAT's visual attention, LMM's
        # spatial cross-attention decoder) must use this, not `feature_dim`,
        # or depth=7/17 configs mismatch (bug found 2026-09, small_capacity
        # ablation).
        self.raw_channels = in_ch

    def forward(self, x: torch.Tensor, return_map: bool = False):
        x = self.stem(x)
        for stage, gate in zip(self.stages, self.gates):
            x = stage(x)
            x = gate(x)
        feature_map = x                              # (B, C, H, W)
        pooled = self.pool(feature_map).flatten(1)    # (B, C)
        feats = self.proj(pooled)                     # (B, feature_dim) -- F_v (Eq. 3.15)
        if return_map:
            return feats, feature_map
        return feats


class _DepthwiseSeparableConv(nn.Module):
    """MobileNet-style depthwise-separable conv block: depthwise 3x3 + pointwise 1x1."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                                    groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class LightCNN(nn.Module):
    """Lightweight, efficient from-scratch CNN backbone (MobileNet-style
    depthwise-separable convolutions), used for the "efficient/lightweight
    computing" ablation point of comparison.

    Motivation: the 2025-2026 field has shifted toward lightweight/efficient
    vision-language approaches (MedBLIP, LoRA-adapted VLMs, Swin-BART with
    regional attention) rather than heavy from-scratch or fully fine-tuned
    CNN/Transformer backbones. LightCNN gives the proposed architecture an
    honest efficiency-focused counterpart: a small (~6-layer) from-scratch
    backbone, not pretrained, designed to be fast and cheap to train while
    still producing a usable spatial feature map for the spatial
    cross-attention decoder.

    Architecture (6 conv "layers" total, standard conv stem + 5 depthwise-
    separable blocks, matching the DL-7/DL-17/DL-39 naming convention used
    by MultiDepthCNN):
      1. stem:    3x3 conv, stride 2, 3 -> 16 channels           (H/2)
      2. DS block: depthwise-sep, stride 1, 16 -> 32             (H/2)
      3. DS block: depthwise-sep, stride 2, 32 -> 64             (H/4)
      4. DS block: depthwise-sep, stride 2, 64 -> 128            (H/8)
      5. DS block: depthwise-sep, stride 2, 128 -> 256           (H/16)
      6. DS block: depthwise-sep, stride 1, 256 -> 256           (H/16)
      -> 1x1 projection conv: 256 -> feature_dim (matches
         cfg.cnn_feature_dim so it's a drop-in for MultiDepthCNN /
         PretrainedResNetCNN downstream: IAT visual_dim, fusion visual_dim,
         LMM cnn_feature_dim all stay unchanged).

    Interface matches MultiDepthCNN / PretrainedResNetCNN exactly:
    input images in [0, 1] as (B, 3, H, W); forward(x, return_map=True)
    returns (feats (B, feature_dim), feature_map (B, feature_dim, H', W')).
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
        )
        self.block1 = _DepthwiseSeparableConv(16, 32, stride=1)
        self.block2 = _DepthwiseSeparableConv(32, 64, stride=2)
        self.block3 = _DepthwiseSeparableConv(64, 128, stride=2)
        self.block4 = _DepthwiseSeparableConv(128, 256, stride=2)
        self.block5 = _DepthwiseSeparableConv(256, 256, stride=1)

        self.feature_proj = (nn.Conv2d(256, feature_dim, 1, bias=False)
                              if feature_dim != 256 else nn.Identity())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(feature_dim, feature_dim)
        # Unlike MultiDepthCNN, LightCNN's raw feature_map is already
        # projected to feature_dim, so raw_channels == feature_dim always.
        self.raw_channels = feature_dim

    def forward(self, x: torch.Tensor, return_map: bool = False):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        feature_map = self.feature_proj(x)             # (B, feature_dim, H', W')
        pooled = self.pool(feature_map).flatten(1)      # (B, feature_dim)
        feats = self.proj(pooled)                       # (B, feature_dim)
        if return_map:
            return feats, feature_map
        return feats


class PretrainedResNetCNN(nn.Module):
    """ImageNet-pretrained ResNet-34 backbone, drop-in replacement for
    MultiDepthCNN with the same call signature: forward(x, return_map=True)
    -> (feats (B, feature_dim), feature_map (B, C, H, W)).

    Motivation (proposed-model CNN backbone fix): the from-scratch 39-layer
    MultiDepthCNN, trained on only ~2500 images, produces weak/undertrained
    early-layer feature maps -- exactly the maps the per-step spatial
    cross-attention decoder (models/lmm/lmm.py) attends over. Starting from
    ImageNet-pretrained weights gives the attention mechanism meaningful
    edge/texture/part-level features from step 0, instead of forcing it to
    learn both "what to look at" and "what a useful feature even looks like"
    from a tiny dataset simultaneously.

    Design:
      - stem + layer1 + layer2 frozen (generic low-level features transfer
        as-is and are the most prone to catastrophic forgetting on a small
        dataset);
      - layer3 + layer4 fine-tuned at the model's normal LR (mid/high-level
        features need to adapt to gait/medical imagery);
      - ImageNet mean/std normalisation is applied internally so the rest
        of the pipeline (IAT augmentation, GAN, dataloaders) is untouched
        and keeps operating on raw [0, 1] images;
      - final feature map is layer4's output (B, 512, H/32, W/32), so
        feature_dim=512 matches cfg.cnn_feature_dim with no changes needed
        elsewhere (IAT visual_dim, fusion visual_dim, LMM cnn_feature_dim).
    """

    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, feature_dim: int = 512, freeze_until_layer: str = "layer2"):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet34(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        frozen_modules = {"stem": [self.stem], "layer1": [self.stem, self.layer1],
                           "layer2": [self.stem, self.layer1, self.layer2]}[freeze_until_layer]
        for mod in frozen_modules:
            for p in mod.parameters():
                p.requires_grad = False
            mod.eval()  # keep frozen BN stats fixed regardless of outer .train()

        self.register_buffer("_mean", torch.tensor(self._IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor(self._IMAGENET_STD).view(1, 3, 1, 1))

        out_ch = 512  # resnet34 layer4 output channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(out_ch, feature_dim) if feature_dim != out_ch else nn.Identity()
        # raw feature_map channels = resnet34's fixed layer4 output (512),
        # NOT feature_dim (proj only applies to the pooled vector).
        self.raw_channels = out_ch

        self._frozen_prefixes = frozen_modules

    def train(self, mode: bool = True):
        super().train(mode)
        # Re-freeze BN stats in the frozen prefix every time .train() is
        # called elsewhere in the codebase (e.g. model.train() each epoch).
        for mod in self._frozen_prefixes:
            mod.eval()
        return self

    def forward(self, x: torch.Tensor, return_map: bool = False):
        x = (x - self._mean) / self._std
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_map = self.layer4(x)               # (B, 512, H, W)
        pooled = self.pool(feature_map).flatten(1)  # (B, 512)
        feats = self.proj(pooled)
        if return_map:
            return feats, feature_map
        return feats

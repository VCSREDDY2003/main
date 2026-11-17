"""
Param-matched (~2.4M, tolerance 2.0-2.8M) baseline classifier architectures,
built to compare against CNN+IATnew's micro_capacity (~2.4M params) on the
same 5-class modality-classification task. Hand-rolled rather than bare
torchvision defaults because the standard torchvision factories don't land
near 2.4M (mobilenet_v3_small=1.52M, shufflenet_v2_x1_0=1.26M,
squeezenet1_1=0.73M, mnasnet1_0=3.11M, efficientnet_b0=4.01M, resnet18=11.2M
-- all checked directly, none in band without modification).

Each builder takes no width-tuning args in the public API (sized by the
constants below) and prints its own param count; use `python
feature_pipeline/baseline_archs.py` to self-check all 8 counts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------
# 1. ResNet-style (scaled-down basic-block ResNet) -- stand-in for
#    "ResNet50" at matched capacity, not the literal ResNet50 architecture.
# ---------------------------------------------------------------------
class _BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = None
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNetSmall(nn.Module):
    """4-stage basic-block ResNet, channels [40,80,160,256], 2 blocks/stage."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        chs = [48, 96, 192, 320]
        self.stem = nn.Sequential(
            nn.Conv2d(3, chs[0], 3, 2, 1, bias=False), nn.BatchNorm2d(chs[0]), nn.ReLU())
        stages = []
        c_in = chs[0]
        for i, c_out in enumerate(chs):
            stride = 1 if i == 0 else 2
            stages.append(_BasicBlock(c_in, c_out, stride))
            c_in = c_out
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(chs[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------
# 2. EfficientNet-style (MBConv blocks, no SE to save params) -- stand-in
#    for "EfficientNet-B4" at matched capacity.
# ---------------------------------------------------------------------
class _MBConv(nn.Module):
    def __init__(self, c_in, c_out, expand=4, stride=1):
        super().__init__()
        c_mid = c_in * expand
        self.expand = nn.Sequential(
            nn.Conv2d(c_in, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid), nn.ReLU6())
        self.dw = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, 3, stride, 1, groups=c_mid, bias=False),
            nn.BatchNorm2d(c_mid), nn.ReLU6())
        self.project = nn.Sequential(nn.Conv2d(c_mid, c_out, 1, bias=False), nn.BatchNorm2d(c_out))
        self.use_residual = stride == 1 and c_in == c_out

    def forward(self, x):
        out = self.project(self.dw(self.expand(x)))
        return x + out if self.use_residual else out


class EfficientNetSmall(nn.Module):
    """MBConv stack, channels [32,64,128,192,256], expand=4."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        chs = [48, 96, 192, 320, 448]
        self.stem = nn.Sequential(
            nn.Conv2d(3, chs[0], 3, 2, 1, bias=False), nn.BatchNorm2d(chs[0]), nn.ReLU6())
        blocks = []
        c_in = chs[0]
        for i, c_out in enumerate(chs):
            stride = 2 if i > 0 else 1
            blocks.append(_MBConv(c_in, c_out, expand=6, stride=stride))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(chs[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        return self.blocks(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------
# 3. MLP on downsampled pixels -- stand-in for "2-layer MLP", properly
#    param-matched (the literature MLP baseline was not).
# ---------------------------------------------------------------------
class MLPClassifier(nn.Module):
    """Flatten 56x56x3 input -> 2 hidden layers -> num_classes."""

    IMAGE_SIZE = 56

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        in_dim = 3 * self.IMAGE_SIZE * self.IMAGE_SIZE
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        if x.shape[-1] != self.IMAGE_SIZE:
            x = F.interpolate(x, size=(self.IMAGE_SIZE, self.IMAGE_SIZE), mode="bilinear", align_corners=False)
        return self.net(x.flatten(1))


# ---------------------------------------------------------------------
# 5. MobileNet-style (depthwise-separable stack)
# ---------------------------------------------------------------------
class _DWSepBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, stride, 1, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in), nn.ReLU6())
        self.pw = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU6())

    def forward(self, x):
        return self.pw(self.dw(x))


class MobileNetStyleSmall(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        chs = [144, 288, 576, 960, 1344]
        self.stem = nn.Sequential(nn.Conv2d(3, chs[0], 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(chs[0]), nn.ReLU6())
        blocks, c_in = [], chs[0]
        for i, c_out in enumerate(chs):
            stride = 2 if i > 0 else 1
            blocks.append(_DWSepBlock(c_in, c_out, stride))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(chs[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        return self.blocks(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.fc(self.pool(x).flatten(1))


# ---------------------------------------------------------------------
# 6. ShuffleNet-style (grouped conv + channel shuffle)
# ---------------------------------------------------------------------
def _channel_shuffle(x, groups):
    b, c, h, w = x.shape
    x = x.view(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)


class _ShuffleBlock(nn.Module):
    def __init__(self, c_in, c_out, groups=4, stride=1):
        super().__init__()
        self.groups = groups
        c_mid = c_out // 2
        self.gconv1 = nn.Conv2d(c_in, c_mid, 1, groups=min(groups, c_in), bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.dw = nn.Conv2d(c_mid, c_mid, 3, stride, 1, groups=c_mid, bias=False)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.gconv2 = nn.Conv2d(c_mid, c_out, 1, groups=min(groups, c_mid), bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)
        self.shortcut = None
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = F.relu(self.bn1(self.gconv1(x)))
        out = _channel_shuffle(out, self.groups)
        out = self.bn2(self.dw(out))
        out = self.bn3(self.gconv2(out))
        return F.relu(out + identity)


class ShuffleNetStyleSmall(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        chs = [144, 288, 576, 864, 1152]
        self.stem = nn.Sequential(nn.Conv2d(3, chs[0], 3, 2, 1, bias=False),
                                   nn.BatchNorm2d(chs[0]), nn.ReLU())
        blocks, c_in = [], chs[0]
        for i, c_out in enumerate(chs):
            stride = 2 if i > 0 else 1
            blocks.append(_ShuffleBlock(c_in, c_out, groups=4, stride=stride))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(chs[-1], num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        return self.blocks(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.fc(self.pool(x).flatten(1))


# ---------------------------------------------------------------------
# 7. SqueezeNet-style (fire modules: squeeze 1x1 -> expand 1x1/3x3)
# ---------------------------------------------------------------------
class _FireModule(nn.Module):
    def __init__(self, c_in, squeeze, expand):
        super().__init__()
        self.squeeze = nn.Sequential(nn.Conv2d(c_in, squeeze, 1), nn.ReLU())
        self.expand1x1 = nn.Sequential(nn.Conv2d(squeeze, expand, 1), nn.ReLU())
        self.expand3x3 = nn.Sequential(nn.Conv2d(squeeze, expand, 3, padding=1), nn.ReLU())

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1x1(s), self.expand3x3(s)], dim=1)


class SqueezeNetStyleSmall(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 96, 3, 2, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.fire1 = _FireModule(96, 128, 256)     # -> 512
        self.pool1 = nn.MaxPool2d(2)
        self.fire2 = _FireModule(512, 192, 384)    # -> 768
        self.pool2 = nn.MaxPool2d(2)
        self.fire3 = _FireModule(768, 256, 512)    # -> 1024
        self.pool3 = nn.MaxPool2d(2)
        self.classifier = nn.Conv2d(1024, num_classes, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.pool1(self.fire1(x))
        x = self.pool2(self.fire2(x))
        return self.pool3(self.fire3(x))  # (B, 1024, H, W) pre-classifier feature map

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return self.pool(x).flatten(1)


# ---------------------------------------------------------------------
# 8. DenseNet-style (shallow dense blocks, small growth rate)
# ---------------------------------------------------------------------
class _DenseLayer(nn.Module):
    def __init__(self, c_in, growth):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, growth, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(self, c_in, growth, n_layers):
        super().__init__()
        layers, c = [], c_in
        for _ in range(n_layers):
            layers.append(_DenseLayer(c, growth))
            c += growth
        self.layers = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.layers(x)


class _Transition(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(x))


class DenseNetStyleSmall(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, growth=48, block_layers=6):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 48, 3, 2, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU())
        c = 48
        stages = []
        for i in range(3):
            block = _DenseBlock(c, growth, block_layers)
            c = block.out_channels
            stages.append(block)
            if i < 2:
                c_out = c // 2
                stages.append(_Transition(c, c_out))
                c = c_out
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c, num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.fc(self.pool(x).flatten(1))


# Final feature-map channel count per arch (for attaching a captioning
# head on top of the backbone's spatial feature map). "mlp" has no conv
# feature map (operates on flattened pixels), so it cannot get a caption
# head -- excluded from FEATURE_DIMS.
FEATURE_DIMS = {
    "resnet_small": 320,
    "efficientnet_small": 448,
    "mobilenet_style": 1344,
    "shufflenet_style": 1152,
    "squeezenet_style": 1024,
    "densenet_style": 516,  # 48 stem -> +6*48=336 -> /2=168 -> +6*48=456 -> /2=228 -> +6*48=516
}


BUILDERS = {
    "resnet_small": ResNetSmall,
    "efficientnet_small": EfficientNetSmall,
    "mlp": MLPClassifier,
    "mobilenet_style": MobileNetStyleSmall,
    "shufflenet_style": ShuffleNetStyleSmall,
    "squeezenet_style": SqueezeNetStyleSmall,
    "densenet_style": DenseNetStyleSmall,
}


def build_model(arch: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    return BUILDERS[arch](num_classes=num_classes)


if __name__ == "__main__":
    for name, cls in BUILDERS.items():
        m = cls()
        print(f"{name:20s} {count_params(m):,} params")

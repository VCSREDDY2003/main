"""
Conditional GAN module (Eq. 3.11-3.14, Algorithm 2).

    X_hat = G(z, y)
    L_GAN = E[log D(X|y)] + E[log(1 - D(G(z|y), y))]

Used for:
  - synthesizing additional samples for rare/imbalanced classes (3.8)
  - generating sequence-congruent synthetic frames for MedVideoCap-55K
    temporal-sequence augmentation (Algorithm 2)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """G(z, y) -> synthetic image, conditioned on class label embedding."""

    def __init__(self, latent_dim: int = 128, num_classes: int = 10,
                 out_channels: int = 3, img_size: int = 64,
                 class_embed_dim: int = 64, base_channels: int = 256):
        super().__init__()
        self.img_size = img_size
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)

        in_dim = latent_dim + class_embed_dim
        init_size = img_size // 16  # after 4 upsampling stages
        c1, c2, c3, c4 = base_channels, base_channels // 2, base_channels // 4, base_channels // 8

        self.fc = nn.Linear(in_dim, c1 * init_size * init_size)
        self.init_size = init_size
        self.base_channels = c1

        self.net = nn.Sequential(
            nn.InstanceNorm2d(c1, affine=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.InstanceNorm2d(c2, affine=True), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c2, c3, 3, padding=1), nn.InstanceNorm2d(c3, affine=True), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c3, c4, 3, padding=1), nn.InstanceNorm2d(c4, affine=True), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c4, out_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.class_embed(y)
        x = torch.cat([z, y_emb], dim=1)
        x = self.fc(x).view(-1, self.base_channels, self.init_size, self.init_size)
        return self.net(x)


class Discriminator(nn.Module):
    """D(X, y) -> real/fake probability, conditioned on class label.

    `light_head=False` (default) preserves the ORIGINAL flatten-based
    classifier exactly (needed so existing/in-flight experiments that
    import this module keep getting the identical architecture they were
    launched with -- do not change this default). `light_head=True` is a
    new, strictly opt-in alternative that global-average-pools before the
    classifier head instead of flattening the full spatial feature map, so
    parameter count depends only on `base_channels`, not on img_size**2 --
    the flatten-based design makes the classifier/label_proj Linear layers
    the dominant cost at 224x224 (12.8M+ params from
    flatten_dim=256*14*14=50176 alone). Use light_head=True + a small
    base_channels for a low-parameter-count GAN variant.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 img_size: int = 64, class_embed_dim: int = 64, base_channels: int = 256,
                 light_head: bool = False):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.img_size = img_size
        self.light_head = light_head
        c1, c2, c3, c4 = base_channels // 8, base_channels // 4, base_channels // 2, base_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c2, 4, stride=2, padding=1), nn.InstanceNorm2d(c2, affine=True), nn.LeakyReLU(0.2),
            nn.Conv2d(c2, c3, 4, stride=2, padding=1), nn.InstanceNorm2d(c3, affine=True), nn.LeakyReLU(0.2),
            nn.Conv2d(c3, c4, 4, stride=2, padding=1), nn.InstanceNorm2d(c4, affine=True), nn.LeakyReLU(0.2),
        )

        if light_head:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.pooled_dim = c4
            self.label_proj = nn.Linear(class_embed_dim, self.pooled_dim)
            self.classifier = nn.Sequential(
                nn.Linear(self.pooled_dim, max(self.pooled_dim // 2, 8)), nn.LeakyReLU(0.2),
                nn.Linear(max(self.pooled_dim // 2, 8), 1),
            )
        else:
            feat_size = img_size // 16
            self.flatten_dim = c4 * feat_size * feat_size
            self.label_proj = nn.Linear(class_embed_dim, self.flatten_dim)
            self.classifier = nn.Sequential(
                nn.Linear(self.flatten_dim, 256), nn.LeakyReLU(0.2),
                nn.Linear(256, 1),
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        if self.light_head:
            feat = self.pool(feat).flatten(1)
        else:
            feat = feat.flatten(1)
        y_proj = self.label_proj(self.class_embed(y))
        joint = feat * torch.sigmoid(y_proj)
        return self.classifier(joint)


class ConditionalGAN(nn.Module):
    """Wrapper bundling generator + discriminator with the adversarial loss."""

    def __init__(self, latent_dim: int = 128, num_classes: int = 10,
                 channels: int = 3, img_size: int = 64, base_channels: int = 256,
                 light_head: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, num_classes, channels, img_size, base_channels=base_channels)
        self.discriminator = Discriminator(num_classes, channels, img_size, base_channels=base_channels,
                                            light_head=light_head)

    def sample_noise(self, batch_size: int, device) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)

    def generator_loss(self, fake_logits: torch.Tensor) -> torch.Tensor:
        """Non-saturating generator loss: -E[log D(G(z|y))]."""
        return nn.functional.binary_cross_entropy_with_logits(
            fake_logits, torch.ones_like(fake_logits)
        )

    def discriminator_loss(self, real_logits: torch.Tensor,
                            fake_logits: torch.Tensor) -> torch.Tensor:
        """L_GAN (Eq. 3.13)."""
        real_loss = nn.functional.binary_cross_entropy_with_logits(
            real_logits, torch.ones_like(real_logits)
        )
        fake_loss = nn.functional.binary_cross_entropy_with_logits(
            fake_logits, torch.zeros_like(fake_logits)
        )
        return real_loss + fake_loss

    def forward(self, x_real: torch.Tensor, y: torch.Tensor):
        b = x_real.shape[0]
        z = self.sample_noise(b, x_real.device)
        x_fake = self.generator(z, y)
        d_real = self.discriminator(x_real, y)
        d_fake = self.discriminator(x_fake.detach(), y)
        g_fake = self.discriminator(x_fake, y)
        return {
            "x_fake": x_fake,
            "d_real": d_real,
            "d_fake": d_fake,
            "g_fake": g_fake,
        }

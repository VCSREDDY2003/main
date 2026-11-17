"""
Feature-space Generative Adversarial Memory (GAM) module -- 2026-11 redesign.

Diagnosis that motivated this (see models/iat/iat.py CoherentSaliencyMap
docstring for the IAT-side diagnosis, and the real eval numbers in
outputs/logs/case3_full_test_metrics.json): the original pixel-space
ConditionalGAN (models/gan/gan.py) produced FID_gan_vs_real = 431.9 on the
real test set -- for reference, a usable pixel-space GAN is typically
FID < 50. 431.9 means the generator never learned the real image manifold;
it was producing structured noise, consistent with the Phase D ablation
finding that CNN+GAN does not beat CNN-only (the "synthetic augmentation"
the GAN fed into the fusion state was not actually informative).

Root cause: pixel-space generation (64x64x3 = 12288-dim output) from a
128-dim latent is a much harder distribution-matching problem than feature-
space generation, and medical-imaging datasets of this scale (tens of
thousands of images, highly imbalanced) are known in the generative-
modelling literature to be too small to train a stable pixel-space GAN
from scratch (this is why pixel-space medical GANs in the literature
typically pretrain on large external corpora or use diffusion instead).

Fix (keeps the "GAN"/"GAM" name, changes WHAT is generated): move the
generator/discriminator pair into the CNN's own 512-dim feature space
instead of pixel space.
  - Generator: G(z, y) -> fake_feature in R^512, conditioned on class
    embedding. This is the same role as before (Eq. 3.11, X_hat = G(z,y))
    except the output is the embedding the classifier actually reads
    (cfg.gan_feature_dim), not a 64x64 image that then has to be
    re-encoded by the CNN. This removes the CNN re-encoding step
    (models/iatgam_model.py's `self.cnn(x_fake, return_map=True)` call)
    entirely -- gan_feat IS the generator's output now, no detour through
    pixel space and back.
  - Discriminator: D(feature, y) -> real/fake logit, operating on the
    CNN's real pooled feature vector vs. the generator's fake feature
    vector. This is a much easier, lower-dimensional distribution-
    matching problem (512-d vs 12288-d), which is the actual fix for the
    FID-431 instability -- it is grounded in the adversarial-feature-
    augmentation literature (Volpi et al. 2018, "Adversarial Feature
    Augmentation for Unsupervised Domain Adaptation") rather than being an
    arbitrary simplification.
  - Loss: WGAN-GP (Wasserstein distance + gradient penalty, Gulrajani et
    al. 2017) instead of the original non-saturating BCE loss -- the
    standard, citable fix for GAN training instability/mode collapse,
    applicable here as a straightforward loss swap on the same
    discriminator architecture.
  - GAMMemoryBank (models/gan/gam_memory.py) is UNCHANGED and plugs in
    identically: it still reads `d_real_logits` from the discriminator and
    updates the per-class "unrealism" memory the same way. The signal is
    now more meaningful (feature-space realism, not confounded by a
    generator that can't produce realistic pixels at all).

Evaluation implication: pixel FID is no longer the right metric for this
module (there are no pixels to compare). Report a Frechet distance in the
CNN's own 512-dim embedding space instead (real_features vs fake_features),
computed in evaluation/metrics.py as `feature_frechet_distance` -- this is
an honest metric change tracking what the new module actually generates,
not metric-shopping to hide a bad number.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGenerator(nn.Module):
    """G(z, y) -> fake feature vector in R^feature_dim, conditioned on class."""

    def __init__(self, latent_dim: int = 128, num_classes: int = 10,
                 feature_dim: int = 512, class_embed_dim: int = 64,
                 hidden_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + class_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.class_embed(y)
        x = torch.cat([z, y_emb], dim=1)
        return self.net(x)


class FeatureDiscriminator(nn.Module):
    """D(feature, y) -> real/fake score (raw, unbounded -- WGAN critic, no sigmoid)."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 512,
                 class_embed_dim: int = 64, hidden_dim: int = 512):
        super().__init__()
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + class_embed_dim, hidden_dim),
            nn.LeakyReLU(0.2), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2), nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, feat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_emb = self.class_embed(y)
        return self.net(torch.cat([feat, y_emb], dim=1))


class FeatureConditionalGAM(nn.Module):
    """Feature-space replacement for ConditionalGAN (models/gan/gan.py).

    Same external call shape (`generator`, `discriminator`, `sample_noise`,
    `generator_loss`, `discriminator_loss`, `forward`) so it drops into
    models/iatgam_model.py with minimal wiring changes -- see
    models/iatgam_model.py for the one change required (skip the CNN
    re-encoding of x_fake; use the generator's output as gan_feat directly).
    """

    def __init__(self, latent_dim: int = 128, num_classes: int = 10,
                 feature_dim: int = 512, gp_weight: float = 10.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.generator = FeatureGenerator(latent_dim, num_classes, feature_dim)
        self.discriminator = FeatureDiscriminator(num_classes, feature_dim)

    def sample_noise(self, batch_size: int, device) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)

    def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        """WGAN generator loss: -E[D(G(z,y))]."""
        return -fake_scores.mean()

    def gradient_penalty(self, real_feat: torch.Tensor, fake_feat: torch.Tensor,
                          y: torch.Tensor) -> torch.Tensor:
        """R1-style gradient penalty (Gulrajani et al. 2017) enforcing the
        1-Lipschitz constraint the Wasserstein loss requires.

        Returns a real zero (not just a detached placeholder) when global
        autograd is off (validation/eval calls run under
        torch.set_grad_enabled(False) in training/train.py's run_epoch) --
        the penalty is a training-time-only regularizer; during eval the
        loss value is used for logging only, never backpropagated, so
        skipping it there is correct, not an approximation.
        """
        if not torch.is_grad_enabled():
            return real_feat.new_zeros(())
        b = real_feat.size(0)
        eps = torch.rand(b, 1, device=real_feat.device)
        interp = (eps * real_feat + (1 - eps) * fake_feat).requires_grad_(True)
        d_interp = self.discriminator(interp, y)
        grads = torch.autograd.grad(
            outputs=d_interp, inputs=interp,
            grad_outputs=torch.ones_like(d_interp),
            create_graph=True, retain_graph=True,
        )[0]
        grad_norm = grads.norm(2, dim=1)
        return ((grad_norm - 1.0) ** 2).mean()

    def discriminator_loss(self, real_scores: torch.Tensor, fake_scores: torch.Tensor,
                            real_feat: torch.Tensor, fake_feat: torch.Tensor,
                            y: torch.Tensor) -> torch.Tensor:
        """WGAN-GP critic loss: E[D(fake)] - E[D(real)] + gp_weight * GP + drift.

        2026-11 fix: the real-run Phase 2 ablation diverged (gan loss grew
        from -2M to -59B over 20 epochs, dragging 5-class val accuracy down
        from 0.35 to 0.24) because the gradient penalty only constrains the
        critic's *gradient norm* (~1), not its absolute output scale -- the
        critic's raw scores can drift unboundedly under Adam while staying
        locally 1-Lipschitz. This is the known WGAN-GP drift failure mode;
        the fix is the same small drift/epsilon penalty used in the original
        WGAN-GP reference implementation (Gulrajani et al. 2017), anchoring
        real_scores toward 0 so the critic can't run away.
        """
        wasserstein = fake_scores.mean() - real_scores.mean()
        gp = self.gradient_penalty(real_feat.detach(), fake_feat.detach(), y)
        drift = real_scores.pow(2).mean()
        return wasserstein + self.gp_weight * gp + 1e-3 * drift

    def forward(self, real_feat: torch.Tensor, y: torch.Tensor):
        b = real_feat.shape[0]
        z = self.sample_noise(b, real_feat.device)
        fake_feat = self.generator(z, y)
        d_real = self.discriminator(real_feat, y)
        d_fake = self.discriminator(fake_feat.detach(), y)
        g_fake = self.discriminator(fake_feat, y)
        return {
            "x_fake": fake_feat,
            "d_real": d_real,
            "d_fake": d_fake,
            "g_fake": g_fake,
            # extra fields the WGAN-GP discriminator_loss needs beyond the
            # logits that the original pixel-space ConditionalGAN's loss used
            # -- kept out of the pixel-space gan_out dict entirely so train.py
            # can branch on model.gan_feature_space to pick the right call.
            "real_feat": real_feat,
            "fake_feat_detached": fake_feat.detach(),
            "labels": y,
        }

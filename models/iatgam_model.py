"""
Full IAT-GAM-LSTM model: wires together every module into the pipeline
described by Eq. 3.10-3.31 and Eq. 4.1-4.13.

Pipeline (single frame / single sample):

    X --CNN(depth=39)--> F_v0, feature_map            (Eq. 3.15, initial pass)
    C --CaptionEncoder--> H_c                          (Eq. 3.2-3.5)
    (F_v0, H_c) --IAT--> X_aug, attn_map               (Eq. 3.10/4.10, Algorithm 1)
    X_aug --CNN--> F_v                                 (Eq. 3.15, final visual feature)
    (F_v, H_c) --MultimodalFusion--> F_fuse            (Eq. 3.17-3.19)
    (X, y) --ConditionalGAN--> x_fake; CNN(x_fake) --> G_feat   (Eq. 3.11-3.14, 4.10)
    [F_fuse]_t --TemporalLSTM--> h_seq, h_T            (Eq. 3.20-3.27)
    (F_fuse, G_feat, h_T) --MultimodalStateBuilder--> Z (Eq. 4.10)
    Z --LMMDecoder--> caption, summary, class logits   (Eq. 4.1-4.9, Algorithm 3)

For single-image samples (ROCO/ROCOv2) the temporal window is T=1
(the LSTM degenerates to a single step, h_T = h_1).
"""

from __future__ import annotations

import contextlib
import torch
import torch.nn as nn

import config


@contextlib.contextmanager
def _freeze_bn(module: nn.Module):
    """Temporarily put every BatchNorm2d inside *module* into eval mode.

    Used when passing GAN-generated images through the shared CNN backbone:
    fake images have different pixel statistics from real/augmented images,
    and allowing them to update BatchNorm running mean/var contaminates the
    statistics that the backbone relies on for normalising real images in
    eval mode, causing catastrophic loss explosions on the validation set.
    Freezing BN during the fake-image CNN pass keeps running stats clean
    while still letting gradients flow through the normalisation layers.
    """
    bn_layers = [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]
    was_training = [m.training for m in bn_layers]
    for m in bn_layers:
        m.eval()
    try:
        yield
    finally:
        for m, s in zip(bn_layers, was_training):
            m.train(s)
from models.cnn.cnn import LightCNN, MultiDepthCNN, PretrainedResNetCNN
from models.fusion.fusion import MultimodalFusion, MultimodalStateBuilder
from models.gan.gan import ConditionalGAN
from models.gan.feature_gan import FeatureConditionalGAM
from models.gan.gam_memory import GAMMemoryBank
from models.iat.iat import InvasiveAugmentedTransform
from models.lmm.lmm import LMMDecoder
from models.lstm.lstm import TemporalLSTM
from models.text.text_encoder import CaptionEncoder


class IATGAMModel(nn.Module):
    def __init__(self, cfg: config.ModelConfig = config.DEFAULT_MODEL_CONFIG):
        super().__init__()
        self.cfg = cfg

        if getattr(cfg, "cnn_light", False):
            self.cnn = LightCNN(feature_dim=cfg.cnn_feature_dim)
        elif getattr(cfg, "cnn_pretrained", False):
            self.cnn = PretrainedResNetCNN(feature_dim=cfg.cnn_feature_dim)
        else:
            self.cnn = MultiDepthCNN(depth=cfg.cnn_depth, feature_dim=cfg.cnn_feature_dim)

        self.caption_encoder = CaptionEncoder(
            vocab_size=cfg.vocab_size, embed_dim=cfg.embedding_dim,
            num_layers=cfg.transformer_layers, num_heads=cfg.transformer_heads,
            max_len=cfg.max_caption_len, dropout=cfg.dropout,
        )

        # visual_dim for IAT's attention over the raw spatial feature map =
        # self.cnn.raw_channels, which only equals cfg.cnn_feature_dim by
        # coincidence at the default depth=39 (both 512) -- using
        # cfg.cnn_feature_dim directly breaks for cnn_depth=7/17 (bug found
        # 2026-09, small_capacity ablation).
        raw_ch = self.cnn.raw_channels
        self.iat = InvasiveAugmentedTransform(
            channels=3, visual_dim=raw_ch,
            alpha_init=cfg.iat_alpha_init, ssim_threshold=cfg.iat_ssim_threshold,
            beta=cfg.iat_beta, num_layers=getattr(cfg, "iat_num_layers", 7),
        )

        img_size = config.DEFAULT_IMAGE_SIZE[0]
        self.gan_feature_space = getattr(cfg, "gan_feature_space", False)
        if self.gan_feature_space:
            self.gan = FeatureConditionalGAM(
                latent_dim=cfg.gan_latent_dim, num_classes=cfg.num_classes_5class,
                feature_dim=cfg.gan_feature_dim,
            )
        else:
            self.gan = ConditionalGAN(
                latent_dim=cfg.gan_latent_dim, num_classes=cfg.num_classes_5class,
                channels=3, img_size=img_size,
                base_channels=getattr(cfg, "gan_base_channels", 256),
                light_head=getattr(cfg, "gan_light_head", False),
            )

        self.fusion = MultimodalFusion(
            visual_dim=cfg.cnn_feature_dim, text_dim=cfg.embedding_dim,
        )

        self.lstm = TemporalLSTM(
            input_dim=cfg.cnn_feature_dim, hidden_dim=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers, dropout=cfg.dropout,
        )

        self.state_builder = MultimodalStateBuilder(
            fuse_dim=cfg.cnn_feature_dim, gan_dim=cfg.gan_feature_dim,
            lstm_dim=self.lstm.output_dim, fusion_dim=cfg.fusion_dim, dropout=cfg.dropout,
        )

        # GAM Memory Bank: real coupling between the GAN's discriminator and
        # IAT's perturbation strength (see models/gan/gam_memory.py). Opt-in
        # via cfg.gam_memory_coupling so existing runs/checkpoints that don't
        # set this flag are unaffected (default False preserves prior behavior).
        self.gam_memory_coupling = getattr(cfg, "gam_memory_coupling", False)
        if self.gam_memory_coupling:
            self.gam_memory = GAMMemoryBank(
                num_classes=cfg.num_classes_5class,
                gamma=getattr(cfg, "gam_memory_gamma", 0.1),
            )

        # Image-only classification heads (bypass Z / F_fuse entirely, which
        # mix in caption cross-attention -- see models/fusion/fusion.py
        # CrossModalAttention). Diagnosed 2026-09: since the 5-class labels
        # are themselves keyword-derived from captions, and the caption is a
        # real model input via H_c, the Z-based classifier_5class in
        # LMMDecoder learns to read the caption instead of the image
        # (verified: shuffling captions across a batch changes predictions
        # to match the shuffled caption, not the unchanged image; masking
        # the caption collapses every checkpoint to a single constant-class
        # prediction). These heads take F_v -- the CNN's pooled image
        # features -- directly, before any caption cross-attention touches
        # it, so they cannot shortcut through the caption.
        def _img_head(num_classes: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(cfg.cnn_feature_dim, cfg.cnn_feature_dim), nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.cnn_feature_dim, num_classes),
            )
        self.img_classifier_binary = _img_head(cfg.num_classes_binary)
        self.img_classifier_3class = _img_head(cfg.num_classes_3class)
        self.img_classifier_5class = _img_head(cfg.num_classes_5class)

        self.lmm = LMMDecoder(
            fusion_dim=cfg.fusion_dim, vocab_size=cfg.vocab_size,
            embed_dim=cfg.embedding_dim, hidden_dim=cfg.lstm_hidden,
            num_classes_binary=cfg.num_classes_binary,
            num_classes_3class=cfg.num_classes_3class,
            num_classes_5class=cfg.num_classes_5class,
            dropout=cfg.dropout,
            # same raw-channel fix as IAT above: LMM's spatial cross-
            # attention decoder reads the raw feature map (dec_feat_map),
            # not the pooled cfg.cnn_feature_dim vector.
            cnn_feature_dim=raw_ch,
            use_spatial_attention=getattr(cfg, "use_spatial_attention", True),
        )

    def forward(self, images: torch.Tensor, caption_ids: torch.Tensor,
                caption_mask: torch.Tensor, label_5class: torch.Tensor,
                caption_input: torch.Tensor, summary_input: torch.Tensor,
                apply_iat: bool = True, disable_lstm: bool = False,
                disable_gan: bool = False):
        """
        images:        (B, 3, H, W) in [0, 1]
        caption_ids:   (B, L) caption token ids (for the encoder, Eq. 3.2-3.4)
        caption_mask:  (B, L) 1 for real tokens, 0 for padding
        label_5class:  (B,) 5-class modality label, used to condition the GAN (Eq. 3.11-3.14)
        caption_input: (B, Lc) teacher-forcing caption decoder input
        summary_input: (B, Ls) teacher-forcing summary decoder input
        apply_iat:     whether to run the IAT augmentation branch (Algorithm 1)
        disable_lstm:  ablation switch -- if True, h_T is zeroed before the
                       state builder (module still exists/trains normally
                       when False; this only controls the ablation matrix)
        disable_gan:   ablation switch -- if True, gan_feat is zeroed before
                       the state builder, and the GAN itself is not run
                       (saves compute in the disabled-GAN ablation rows)

        returns dict with lmm outputs, attn_map, gan outputs, h_seq (for L_temporal)
        """
        f_v0, feat_map = self.cnn(images, return_map=True)        # Eq. 3.15 (initial)
        h_c = self.caption_encoder(caption_ids, caption_mask)      # Eq. 3.4

        attn_map = None
        iat_consistency_loss = torch.zeros((), device=images.device, dtype=f_v0.dtype)
        if apply_iat:
            b, c, h, w = feat_map.shape
            visual_tokens = feat_map.flatten(2).transpose(1, 2)    # (B, H*W, C)

            class_weight = None
            if self.gam_memory_coupling and not disable_gan:
                # Real GAN<->IAT coupling: use the discriminator's CURRENT
                # realism read on the real images of this batch's classes to
                # drive IAT's per-sample perturbation strength, then update
                # the persistent per-class memory with the same signal.
                # Feature-space GAM's discriminator reads f_v0 (the CNN's
                # real pooled feature), not raw pixels -- see
                # models/gan/feature_gan.py for why this is the fix for the
                # pixel-space GAN's FID-431.9 failure.
                with torch.no_grad():
                    d_real_input = f_v0 if self.gan_feature_space else images
                    d_real_now = self.gan.discriminator(d_real_input, label_5class)
                class_weight = self.gam_memory.alpha_scale(label_5class)
                self.gam_memory.update(d_real_now, label_5class)

            x_aug, attn_map = self.iat(images, visual_tokens, (h, w),
                                        class_weight=class_weight)
            if self.training:
                # D4-equivariance penalty (Phase A redesign): only computed
                # during training, since it's an auxiliary regularizer, not
                # something inference needs.
                iat_consistency_loss = self.iat.attention_consistency_loss(visual_tokens, (h, w))
            f_v, dec_feat_map = self.cnn(x_aug, return_map=True)   # Eq. 3.15 (augmented)
        else:
            f_v = f_v0
            dec_feat_map = feat_map

        f_fuse = self.fusion(f_v, h_c, caption_mask)                # Eq. 3.17-3.19

        if disable_gan:
            gan_out = None
            gan_feat = torch.zeros(images.size(0), self.state_builder.gan_dim,
                                    device=images.device, dtype=f_fuse.dtype)
        elif self.gan_feature_space:
            # Feature-space GAM (2026-11 redesign): the generator's output IS
            # gan_feat directly -- no pixel generation, no CNN re-encoding.
            # This removes the pixel-space FID-431.9 failure mode entirely by
            # construction (there is no pixel realism requirement left to fail).
            gan_out = self.gan(f_v0, label_5class)                  # L_GAN (WGAN-GP) on real features
            dummy_label = torch.zeros_like(label_5class)
            fusion_noise = self.gan.sample_noise(images.size(0), images.device)
            gan_feat = self.gan.generator(fusion_noise, dummy_label)
        else:
            gan_out = self.gan(images, label_5class)                # Eq. 3.11-3.14 (L_GAN only)

            # gan_feat feeds state_builder -> z -> classification heads, so it must
            # NOT be conditioned on the ground-truth label_5class (the classifier
            # would just decode the generator's class embedding instead of learning
            # from the image). Use a fixed dummy label for this branch only.
            dummy_label = torch.zeros_like(label_5class)
            fusion_noise = self.gan.sample_noise(images.size(0), images.device)
            x_fake_fusion = self.gan.generator(fusion_noise, dummy_label)
            # Freeze BN running stats so fake-image statistics don't contaminate
            # the running mean/var used to normalise real images in eval mode.
            with _freeze_bn(self.cnn):
                gan_feat, _ = self.cnn(x_fake_fusion, return_map=True)   # G(z, 0) -> CNN feature (Eq. 4.10)

        z_seq = f_fuse.unsqueeze(1)                                  # (B, 1, D); T=1 for single-frame samples
        if disable_lstm:
            h_seq = None
            h_T = torch.zeros(images.size(0), self.lstm.output_dim,
                               device=images.device, dtype=f_fuse.dtype)
        else:
            h_seq, h_T = self.lstm(z_seq)                            # Eq. 3.20-3.26

        z = self.state_builder(f_fuse, gan_feat, h_T)                # Eq. 4.10

        lmm_out = self.lmm(z, caption_input, summary_input, dec_feat_map,
                            saliency_map=attn_map)  # Eq. 4.1-4.9

        # Image-only classification (bypasses Z/F_fuse -- see class comment
        # on img_classifier_* above): f_v is the CNN's pooled feature on the
        # (possibly IAT-augmented) image, with no caption cross-attention.
        img_class_logits_binary = self.img_classifier_binary(f_v)
        img_class_logits_3class = self.img_classifier_3class(f_v)
        img_class_logits_5class = self.img_classifier_5class(f_v)

        return {
            **lmm_out,
            "attn_map": attn_map,
            "gan_out": gan_out,
            "h_seq": h_seq,
            "z": z,
            "img_class_logits_binary": img_class_logits_binary,
            "img_class_logits_3class": img_class_logits_3class,
            "img_class_logits_5class": img_class_logits_5class,
            "iat_consistency_loss": iat_consistency_loss,
        }

    @torch.no_grad()
    def generate(self, images: torch.Tensor, caption_ids: torch.Tensor,
                  caption_mask: torch.Tensor, label_5class: torch.Tensor,
                  bos_id: int = 1, eos_id: int = 2, max_len: int | None = None,
                  beam_width: int = 1, unk_id: int | None = 3,
                  apply_iat: bool = True, disable_lstm: bool = False,
                  disable_gan: bool = False):
        """Algorithm 3: inference-time caption/summary/prediction generation.

        beam_width=1 -> greedy decoding (default); beam_width>1 -> beam
        search (used for evaluation/report generation).

        apply_iat/disable_lstm/disable_gan: same ablation switches as
        forward() -- must match the flags the checkpoint was trained with
        for the ablation matrix's eval pass to be meaningful.
        """
        max_len = max_len or self.cfg.max_caption_len
        f_v0, feat_map = self.cnn(images, return_map=True)
        h_c = self.caption_encoder(caption_ids, caption_mask)

        attn_map = None
        x_aug = images
        if apply_iat:
            b, c, h, w = feat_map.shape
            visual_tokens = feat_map.flatten(2).transpose(1, 2)
            # class_weight is left at its alpha_init default (no gam_memory
            # read) at inference: scaling by the true class label is not
            # available for a real inference input and was a confirmed
            # test-time leak (2026-11 fix) into the
            # downstream image-only classifier via the IAT-augmented image.
            class_weight = None
            x_aug, attn_map = self.iat(images, visual_tokens, (h, w), class_weight=class_weight)
            f_v, dec_feat_map = self.cnn(x_aug, return_map=True)
        else:
            f_v = f_v0
            dec_feat_map = feat_map

        f_fuse = self.fusion(f_v, h_c, caption_mask)

        if disable_gan:
            x_fake = None
            gan_feat = torch.zeros(images.size(0), self.state_builder.gan_dim,
                                    device=images.device, dtype=f_fuse.dtype)
        elif self.gan_feature_space:
            # Feature-space GAM: generator's output is gan_feat directly, no
            # pixel image to report -- x_fake stays None (nothing to visualize).
            dummy_label = torch.zeros_like(label_5class)
            noise = self.gan.sample_noise(images.size(0), images.device)
            gan_feat = self.gan.generator(noise, dummy_label)
            x_fake = None
        else:
            # Same dummy-label substitution as forward() -- gan_feat must not
            # depend on label_5class (no ground-truth label is available at
            # inference time anyway).
            dummy_label = torch.zeros_like(label_5class)
            noise = self.gan.sample_noise(images.size(0), images.device)
            x_fake = self.gan.generator(noise, dummy_label)
            gan_feat, _ = self.cnn(x_fake, return_map=True)

        if disable_lstm:
            h_seq = None
            h_T = torch.zeros(images.size(0), self.lstm.output_dim,
                               device=images.device, dtype=f_fuse.dtype)
        else:
            h_seq, h_T = self.lstm(f_fuse.unsqueeze(1))
        z = self.state_builder(f_fuse, gan_feat, h_T)

        out = self.lmm.generate(z, dec_feat_map, bos_id=bos_id, eos_id=eos_id,
                                 max_len=max_len, beam_width=beam_width, unk_id=unk_id,
                                 saliency_map=attn_map)
        out["attn_map"] = attn_map
        out["x_aug"] = x_aug
        out["x_fake"] = x_fake

        # Image-only classification -- see forward()'s comment on
        # img_classifier_*; f_v never passes through caption cross-attention.
        img_logits_binary = self.img_classifier_binary(f_v)
        img_logits_3class = self.img_classifier_3class(f_v)
        img_logits_5class = self.img_classifier_5class(f_v)
        out["img_class_logits_binary"] = img_logits_binary
        out["img_class_logits_3class"] = img_logits_3class
        out["img_class_logits_5class"] = img_logits_5class
        out["img_class_pred_binary"] = img_logits_binary.argmax(dim=-1)
        out["img_class_pred_3class"] = img_logits_3class.argmax(dim=-1)
        out["img_class_pred_5class"] = img_logits_5class.argmax(dim=-1)
        return out

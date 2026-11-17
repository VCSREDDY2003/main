"""
Central configuration for the IAT-GAM-LSTM framework.

IAT-GAM-LSTM = Invasive Augmented Transform driven
               Generative Adversarial Memory (GAN + LSTM) framework
               for real-time medical video captioning & semantic labelling.

This file holds:
  - dataset path configuration (ROCO, ROCOv2, MedVideoCap-55K)
  - baseline vs. tuned hyperparameters (Table 3.1 / 3.2 of the paper)
  - architecture dimensions (Eq. 3.10-3.27, Eq. 4.10-4.13)

Dataset folder layout expected by `data/datasets.py`
-----------------------------------------------------
Each dataset is described by a *manifest* CSV/JSON with one row per sample:

    id, image_path, caption, label, video_id, frame_index

- ROCO / ROCOv2   -> static samples: image_path + caption (+ label)
                      (data/prepare_roco.py, data/prepare_rocov2.py)
- MedVideoCap-55K -> video samples: a `video_id` groups multiple rows that
                      share the same `video_id`, ordered by `frame_index`.
                      (data/prepare_medvideocap.py)

Point `ROCO_ROOT`, `ROCOV2_ROOT`, `MEDVIDEOCAP_ROOT` at the dataset folders
and `*_MANIFEST` at the manifest files. `MEDICAT_ROOT` is kept as an
optional 4th source if/when that data becomes available.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# Dataset paths (placeholders -- update once real paths are provided)
# ------------------------------------------------------------------
DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets")

MEDICAT_ROOT = os.path.join(DATA_ROOT, "medicat")
MEDICAT_MANIFEST = os.path.join(MEDICAT_ROOT, "manifest.csv")

ROCO_ROOT = os.path.join(DATA_ROOT, "roco")
ROCO_MANIFEST = os.path.join(ROCO_ROOT, "manifest.csv")

ROCOV2_ROOT = os.path.join(DATA_ROOT, "rocov2")
ROCOV2_MANIFEST = os.path.join(ROCOV2_ROOT, "manifest.csv")

MEDVIDEOCAP_ROOT = os.path.join(DATA_ROOT, "medvideocap55k")
MEDVIDEOCAP_MANIFEST = os.path.join(MEDVIDEOCAP_ROOT, "manifest.csv")

# Dataset split ratios (Table 5.1: 70/15/15)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

RANDOM_SEED = 42


# ------------------------------------------------------------------
# Image / temporal preprocessing (Eq. 3.6-3.9)
# ------------------------------------------------------------------
IMAGE_SIZES = [(224, 224), (160, 160), (128, 128)]
DEFAULT_IMAGE_SIZE = (224, 224)

TEMPORAL_WINDOW = 8          # k in Eq. 3.7 (sliding window size)
MAX_CAPTION_LEN = 32         # L in Eq. 3.2
VOCAB_SIZE = 16000  # Bug fix (2026-09): 8000 caused 46.4%/33.9% of GT captions (full/limit=20k
# splits, respectively) to contain >=1 OOV token, a structural BLEU ceiling. 16000 was
# measured to be the smallest of {12000, 16000} that gets token-level OOV into the low
# single digits (1.18% at 16000 vs. 1.87% at 12000 vs. 3.07% at 8000); see
# data/vocab.py OOV measurement in the phase-A diagnostic notes.


# ------------------------------------------------------------------
# Hyperparameters (Table 3.1: baseline / Table 3.2: tuned)
# ------------------------------------------------------------------
@dataclass
class Hyperparameters:
    learning_rate: float = 0.001     # eta
    batch_size: int = 32             # B
    epochs: int = 30                 # E
    lambda_gan: float = 0.5          # lambda_1
    lambda_temporal: float = 0.5     # lambda_2
    embedding_dim: int = 256         # d
    lstm_hidden: int = 256           # h
    dropout: float = 0.3
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    # IAT augmentation strength (Eq. 3.10, Algorithm 1) -- overrides
    # ModelConfig.iat_alpha_init / iat_ssim_threshold when set.
    iat_alpha_init: float = 0.3
    iat_ssim_threshold: float = 0.85


BASELINE_HPARAMS = Hyperparameters(
    learning_rate=0.001, batch_size=32, epochs=30,
    lambda_gan=0.5, lambda_temporal=0.5,
    embedding_dim=256, lstm_hidden=256, dropout=0.3,
    weight_decay=0.0, grad_clip_norm=5.0,
    iat_alpha_init=0.3, iat_ssim_threshold=0.85,
)

TUNED_HPARAMS = Hyperparameters(
    learning_rate=0.0003, batch_size=64, epochs=50,
    lambda_gan=0.7, lambda_temporal=0.6,
    embedding_dim=512, lstm_hidden=512, dropout=0.5,
    weight_decay=1e-4, grad_clip_norm=5.0,
    iat_alpha_init=0.3, iat_ssim_threshold=0.85,
)

# Stronger-regularization preset to address the train/val gap observed with
# BASELINE_HPARAMS: higher dropout + weight decay for the LMM/fusion/CNN
# heads, gradient clipping for GAN stability, and a stronger/more frequent
# IAT perturbation (lower SSIM threshold -> larger alpha tolerated, higher
# initial alpha) so X_aug differs more from X each step (Algorithm 1).
AUG_STRONG_HPARAMS = Hyperparameters(
    learning_rate=0.001, batch_size=32, epochs=30,
    lambda_gan=0.5, lambda_temporal=0.5,
    embedding_dim=256, lstm_hidden=256, dropout=0.45,
    weight_decay=1e-4, grad_clip_norm=5.0,
    iat_alpha_init=0.45, iat_ssim_threshold=0.75,
)


# ------------------------------------------------------------------
# Architecture configuration
# ------------------------------------------------------------------
@dataclass
class ModelConfig:
    # CNN feature extractor (Eq. 3.15, 4.12); depth: 7 / 17 / 39 layers
    cnn_depth: int = 39
    cnn_feature_dim: int = 512
    # Backbone-fix flag: use an ImageNet-pretrained ResNet-34 (see
    # models/cnn/cnn.py:PretrainedResNetCNN) instead of the from-scratch
    # MultiDepthCNN. Off by default (scratch), preserving the ablation and
    # baseline exactly as-is; only the proposed model's training run turns
    # this on via train.py --pretrained_cnn.
    cnn_pretrained: bool = False
    # Lightweight-efficiency ablation: use the from-scratch MobileNet-style
    # depthwise-separable LightCNN (models/cnn/cnn.py:LightCNN) instead of
    # MultiDepthCNN/PretrainedResNetCNN. Mutually exclusive with
    # cnn_pretrained; wired via train.py --light_cnn.
    cnn_light: bool = False

    # IAT perturbation-generator depth ablation (7-layer vs 14-layer);
    # wired via train.py --iat_layers 7|14.
    iat_num_layers: int = 7

    # Transformer caption encoder (Eq. 3.4, 3.16)
    vocab_size: int = VOCAB_SIZE
    embedding_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    max_caption_len: int = MAX_CAPTION_LEN

    # LSTM temporal module (Eq. 3.20-3.27)
    lstm_hidden: int = 256
    lstm_layers: int = 2
    temporal_window: int = TEMPORAL_WINDOW

    # GAN (Eq. 3.11-3.14)
    gan_latent_dim: int = 128
    gan_feature_dim: int = 512
    # Low-param-count GAN variant (2026-09, for the ~5M-param capacity
    # ablation): base_channels=256/light_head=False reproduces the original
    # architecture exactly (do not change these defaults -- in-flight
    # experiments depend on them staying put). Set gan_base_channels lower
    # and gan_light_head=True together to shrink the GAN drastically (its
    # flatten-based discriminator head dominates full-capacity param count).
    gan_base_channels: int = 256
    gan_light_head: bool = False
    # Opt-in feature-space GAM redesign (2026-11, see models/gan/feature_gan.py):
    # the original pixel-space GAN scored FID_gan_vs_real=431.9 on the real
    # test set (unusable -- a working pixel GAN is typically FID<50), so it
    # was replaced with a generator/discriminator pair operating directly in
    # the CNN's 512-dim feature space (WGAN-GP loss) instead of pixel space.
    # Default False preserves the original pixel-space ConditionalGAN exactly
    # for any in-flight/existing experiments; set True to use the redesign.
    gan_feature_space: bool = False

    # IAT (Eq. 3.10, Algorithm 1)
    iat_alpha_init: float = 0.3
    iat_ssim_threshold: float = 0.85   # tau
    iat_beta: float = 0.9              # adjustment factor for alpha

    # Fusion / LMM (Eq. 3.17-3.19, 4.2-4.3)
    fusion_dim: int = 512
    # Multi-granularity classification heads (Eq. 3.29 generalized to three
    # targets): binary (static image vs video/other), 3-class (modality
    # group), 5-class (imaging modality). The GAN (Eq. 3.11-3.14) is
    # conditioned on the 5-class label.
    num_classes_binary: int = 2
    num_classes_3class: int = 3
    num_classes_5class: int = 5

    dropout: float = 0.3

    # Proposed update vs. the original IAT-GAM-LSTM: per-step spatial
    # cross-attention over the CNN feature map (see models/lmm/lmm.py).
    # Set False to reproduce the original pooled single-vector bottleneck
    # decoder for the "no spatial attention" ablation.
    use_spatial_attention: bool = True

    # GAM Memory Bank (models/gan/gam_memory.py): opt-in real coupling
    # between the GAN discriminator's per-class realism signal and IAT's
    # perturbation strength. Default False preserves prior behavior for
    # every existing run/checkpoint that doesn't set this explicitly.
    gam_memory_coupling: bool = False
    gam_memory_gamma: float = 0.1


DEFAULT_MODEL_CONFIG = ModelConfig()
NO_SPATIAL_ATTN_MODEL_CONFIG = ModelConfig(use_spatial_attention=False)


# ------------------------------------------------------------------
# Loss weighting (Eq. 3.28)
# ------------------------------------------------------------------
@dataclass
class LossWeights:
    lambda_gan: float = 0.5     # lambda_1
    lambda_temporal: float = 0.5  # lambda_2


# ------------------------------------------------------------------
# Output paths
# ------------------------------------------------------------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "model")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

for _d in (OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)


# ------------------------------------------------------------------
# Device
# ------------------------------------------------------------------
def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

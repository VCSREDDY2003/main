"""
Staged training loop for IAT-GAM-LSTM (Table 3.1/3.2 hyperparameters,
Eq. 3.31 composite loss).

Stage A (per step):
    1. Forward pass through IATGAMModel (IAT -> CNN -> GAN -> Fusion -> LSTM -> LMM)
    2. L_GAN (Eq. 3.13) from the conditional GAN branch
    3. L_temporal (Eq. 3.27) from the LSTM hidden-state sequence
    4. L_total = L_cap + L_summary + L_cls + lambda_1 L_GAN + lambda_2 L_temporal (Eq. 3.31)
    5. Single optimizer updates the full model jointly (generator/discriminator
       included) -- a simplified single-stage variant of Algorithm 2's
       alternating GAN updates, suitable as the baseline training loop.

Usage:
    python training/train.py --hparams baseline --epochs 1 --limit 200
"""

from __future__ import annotations

import argparse
import os
import random

import csv

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

import config
from data.datasets import build_datasets, build_leakage_safe_split, split_indices_grouped
from data.vocab import PAD, Vocabulary, VOCAB_PATH, build_vocab
from evaluation.metrics import classification_metrics
from models.iatgam_model import IATGAMModel
from training.aug_methods import crop_flip
from training.losses import CompositeLoss


# Inverse-class-frequency weights ((n_total/n_classes)/n_k), recomputed
# 2026-09-03 after the classification-label bug fix (data/build_classification_labels.py:
# angiography/ERCP/venogram/fluoroscopy/mammography/urogram/myelogram/barium-study and
# PET/nuclear-medicine keywords added so they no longer fall into class 4 "Other").
# Computed from the --limit 20000/source train split (seed=42, first 20000 rows/source,
# ROCO+ROCOv2 only):
#   5class: {0: 6639, 1: 7907, 2: 3253, 3: 2325, 4: 7876}  (class 4 share dropped from
#   this recompute's prior ~48% to 40.6% of train rows; still large because ROCO/ROCOv2
#   contain many genuinely non-X-ray/CT/MRI/US images -- endoscopy, clinical photos,
#   histology -- that legitimately have no dedicated class under this 5-class taxonomy)
CLASS_WEIGHTS_5CLASS = [0.8435, 0.7082, 1.7215, 2.4086, 0.7110]
# 3class: {0: 6639, 1: 13485, 2: 7876}
CLASS_WEIGHTS_3CLASS = [1.4058, 0.6921, 1.1850]
# binary: {0: 20124, 1: 7876}
CLASS_WEIGHTS_BINARY = [0.6957, 1.7776]


def get_hparams(name: str) -> config.Hyperparameters:
    if name == "tuned":
        return config.TUNED_HPARAMS
    if name == "aug_strong":
        return config.AUG_STRONG_HPARAMS
    return config.BASELINE_HPARAMS


def build_dataloaders(vocab: Vocabulary, batch_size: int, limit: int | None, num_workers: int = 0,
                       balance_sources: bool = True, balance_classes: bool = False,
                       leakage_safe_audit_path: str | None = None,
                       dataset_sources: list[str] | None = None,
                       roco_manifest_override: str | None = None,
                       rocov2_manifest_override: str | None = None):
    sources = build_datasets(vocab, roco_manifest_override=roco_manifest_override,
                              rocov2_manifest_override=rocov2_manifest_override)
    if dataset_sources:
        sources = {k: v for k, v in sources.items() if k in dataset_sources}
    if not sources:
        raise RuntimeError("No dataset manifests found under datasets/. Run the data/prepare_*.py scripts first.")

    # Cap each source to the same size (smallest source) so no single domain
    # dominates batches. MedVideoCap (all class-4/video) is 55K vs ROCOv2's
    # 31K; without balancing, class-4 is 59% of every batch, which causes
    # the 5-class classifier to collapse to always predicting the minority
    # class (mode collapse driven by class-weight + batch imbalance).
    source_sizes = {name: (len(ds) if limit is None else min(limit, len(ds)))
                    for name, ds in sources.items()}
    if balance_sources and limit is None:
        min_size = min(source_sizes.values())
        source_sizes = {name: min_size for name in source_sizes}

    train_sets, val_sets, test_sets = [], [], []
    if leakage_safe_audit_path is not None:
        # Global, cross-source leakage-safe split (Phase C, 2026-11): the
        # old per-source split_indices_grouped below misses duplicate/near-
        # duplicate images that cross the ROCO<->ROCOv2 boundary (the
        # duplicate audit found 9,495 exact-phash and 1,050 near-duplicate
        # groups, mostly cross-source) -- this computes ONE global split
        # over all sources together so no such pair can land on opposite
        # sides of train/val/test. Computed on the FULL (un-truncated)
        # dataset so duplicate matching is correct, then each source's
        # resulting index arrays are capped to the same `n` the old code
        # used, to preserve the balance_sources/limit behavior.
        full_splits = build_leakage_safe_split(sources, audit_json_path=leakage_safe_audit_path)
        for (name, ds), n in zip(sources.items(), source_sizes.values()):
            train_idx, val_idx, test_idx = full_splits[name]
            train_idx = train_idx[train_idx < n]
            val_idx = val_idx[val_idx < n]
            test_idx = test_idx[test_idx < n]
            train_sets.append(Subset(ds, train_idx))
            val_sets.append(Subset(ds, val_idx))
            test_sets.append(Subset(ds, test_idx))
    else:
        for (name, ds), n in zip(sources.items(), source_sizes.values()):
            # Group-aware split: keeps all frames of a given video_id in the
            # same split (see split_indices_grouped docstring) -- prevents
            # the train/test leakage that a plain row-level split would
            # cause on MedVideoCap-55K. Static ROCO/ROCOv2 rows have no
            # video_id, so this reduces to a normal per-row split for them.
            # NOTE: does not catch cross-source duplicates -- use
            # leakage_safe_audit_path for that.
            group_ids = ds.df["video_id"].iloc[:n].tolist()
            train_idx, val_idx, test_idx = split_indices_grouped(group_ids)
            train_sets.append(Subset(ds, train_idx))
            val_sets.append(Subset(ds, val_idx))
            test_sets.append(Subset(ds, test_idx))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_concat = ConcatDataset(train_sets)
    if balance_classes:
        # Oversample rare / undersample common labels via a
        # WeightedRandomSampler, on top of the existing class-weighted loss
        # (CLASS_WEIGHTS_5CLASS/BINARY/3CLASS below) -- two complementary
        # balancing mechanisms, not a replacement for each other.
        #
        # Bug fix (2026-11): this used to weight ONLY by label_5class. Since
        # binary class 1 maps exactly onto 5-class "class 4" (confirmed:
        # both have identical counts in the real data), balancing 5-class
        # alone was assumed to also balance binary for free -- but the
        # sampler's inverse-frequency weights are computed per-5-class-
        # label, so a sample doesn't get the EXTRA boost it would need from
        # being examined as "minority in the binary view" too; measured
        # result was only 14.4% recall on the minority binary class despite
        # this sampler running. Now computes sample weight as the PRODUCT
        # of inverse frequency in both the label_5class view AND the
        # label_binary view, so a sample rare in either grouping is
        # oversampled -- both granularities get equal effective importance,
        # not just whichever one happens to dominate.
        #
        # Read labels directly from each Subset's underlying manifest
        # DataFrame (vectorized pandas indexing) instead of calling
        # __getitem__ per sample -- the naive per-sample loop was loading
        # and transforming every training image from disk just to read its
        # label column, which took 1h20m+ of wall-clock time on a ~20K-row
        # subset alone (found 2026-09 while fine-tuning the image-only
        # classifier heads). This reads the label columns with no image I/O.
        all_labels_5class, all_labels_binary = [], []
        for subset in train_sets:
            df_subset = subset.dataset.df.iloc[subset.indices]
            all_labels_5class.append(df_subset["label_5class"].to_numpy())
            all_labels_binary.append(df_subset["label_binary"].to_numpy())
        labels_5class = np.concatenate(all_labels_5class).astype(np.int64)
        labels_binary = np.concatenate(all_labels_binary).astype(np.int64)

        counts_5class = np.zeros(5, dtype=np.int64)
        for label in labels_5class:
            if 0 <= label < 5:
                counts_5class[label] += 1
        counts_binary = np.zeros(2, dtype=np.int64)
        for label in labels_binary:
            if 0 <= label < 2:
                counts_binary[label] += 1

        weight_5class = np.array([1.0 / max(counts_5class[int(l)], 1) if 0 <= l < 5 else 1.0
                                    for l in labels_5class])
        weight_binary = np.array([1.0 / max(counts_binary[int(l)], 1) if 0 <= l < 2 else 1.0
                                    for l in labels_binary])
        # Normalize each view to mean 1 before combining.
        #
        # Bug fix (2026-11, found via the real binary-fix validation run):
        # a plain product of the two normalized weights double-counts a
        # sample's rarity when the two groupings overlap -- binary class 1
        # IS exactly 5-class class 4 (confirmed: identical counts in the
        # real data), so that one group's rarity boost gets applied TWICE
        # multiplicatively, overshooting into the opposite failure mode:
        # the real run measured 87-99% recall on the former-minority class
        # but only 30-38% overall accuracy (worse than the original
        # 69.7%), i.e. it now over-predicts that class instead of
        # under-predicting it. Geometric mean (sqrt of the product)
        # softens the combination instead of compounding it -- equivalent
        # to averaging the two corrections in log-space rather than
        # stacking them.
        weight_5class = weight_5class / weight_5class.mean()
        weight_binary = weight_binary / weight_binary.mean()
        sample_weights = np.sqrt(weight_5class * weight_binary)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_concat), replacement=True)
        train_loader = DataLoader(train_concat, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_concat, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(ConcatDataset(test_sets), batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def run_epoch(model: IATGAMModel, loader: DataLoader, criterion: CompositeLoss,
               optimizer: torch.optim.Optimizer | None, device: torch.device,
               pad_id: int, grad_clip_norm: float = 0.0,
               use_amp: bool = False, apply_iat: bool = True,
               disable_lstm: bool = False, disable_gan: bool = False,
               augment_train: bool = False) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = use_amp and device.type == "cuda"

    totals = {}
    n_batches = 0
    preds = {"binary": [], "3class": [], "5class": []}
    targets_acc = {"binary": [], "3class": [], "5class": []}
    for batch in loader:
        images = batch["image"].to(device)
        if is_train and augment_train:
            # Real pixel-level augmentation (crop-flip; see
            # training/aug_methods.py), not just the WeightedRandomSampler's
            # oversampling above. Oversampling alone repeats byte-identical
            # copies of rare-class images -- the minority class can overfit
            # to those specific images without ever seeing a varied view of
            # them. crop_flip is the one augmentation method that matched
            # or beat IAT in the real Phase 2 comparison (2026-11), so it's
            # the natural choice here rather than introducing a new,
            # unvalidated augmentation.
            images, _, _, _ = crop_flip(images, torch.zeros(images.size(0), dtype=torch.long, device=device))
        caption_ids = batch["caption_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)
        label_binary = batch["label_binary"].to(device)
        label_3class = batch["label_3class"].to(device)
        label_5class = batch["label_5class"].to(device)
        caption_mask = (caption_ids != pad_id).long()

        # teacher forcing: decoder input = target shifted right by one
        caption_input = torch.roll(caption_ids, shifts=1, dims=1)
        caption_input[:, 0] = caption_ids[0, 0]  # <bos>
        summary_input = torch.roll(summary_ids, shifts=1, dims=1)
        summary_input[:, 0] = summary_ids[0, 0]

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(images, caption_ids, caption_mask, label_5class, caption_input, summary_input,
                             apply_iat=apply_iat, disable_lstm=disable_lstm, disable_gan=disable_gan)

                if disable_gan:
                    gan_loss = torch.zeros((), device=device)
                elif getattr(model, "gan_feature_space", False):
                    # Feature-space GAM (WGAN-GP): discriminator_loss needs
                    # the real/fake feature vectors themselves (for the
                    # gradient penalty), not just the logits -- see
                    # models/gan/feature_gan.py.
                    gan_loss = model.gan.discriminator_loss(
                        out["gan_out"]["d_real"], out["gan_out"]["d_fake"],
                        out["gan_out"]["real_feat"], out["gan_out"]["fake_feat_detached"],
                        out["gan_out"]["labels"],
                    ) + model.gan.generator_loss(out["gan_out"]["g_fake"])
                else:
                    gan_loss = model.gan.discriminator_loss(out["gan_out"]["d_real"], out["gan_out"]["d_fake"]) \
                        + model.gan.generator_loss(out["gan_out"]["g_fake"])
                if disable_lstm:
                    temporal_loss = torch.zeros((), device=device)
                else:
                    temporal_loss = model.lstm.temporal_consistency_loss(out["h_seq"])

                losses = criterion(
                    out,
                    {
                        "caption_targets": caption_ids, "summary_targets": summary_ids,
                        "class_targets_binary": label_binary,
                        "class_targets_3class": label_3class,
                        "class_targets_5class": label_5class,
                    },
                    gan_loss=gan_loss, temporal_loss=temporal_loss,
                )

            if is_train:
                optimizer.zero_grad()
                losses["total"].backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
        n_batches += 1

        # Image-only heads (see losses.py comment) -- these are what the
        # classification loss actually trains now, so report accuracy on
        # them, not the old caption-contaminated class_logits_*.
        preds["binary"].append(out["img_class_logits_binary"].detach().argmax(-1).cpu().numpy())
        preds["3class"].append(out["img_class_logits_3class"].detach().argmax(-1).cpu().numpy())
        preds["5class"].append(out["img_class_logits_5class"].detach().argmax(-1).cpu().numpy())
        targets_acc["binary"].append(label_binary.cpu().numpy())
        targets_acc["3class"].append(label_3class.cpu().numpy())
        targets_acc["5class"].append(label_5class.cpu().numpy())

    num_classes = {"binary": 2, "3class": 3, "5class": 5}
    metrics = {k: v / max(n_batches, 1) for k, v in totals.items()}
    for granularity in ("binary", "3class", "5class"):
        y_pred = np.concatenate(preds[granularity]) if preds[granularity] else np.array([])
        y_true = np.concatenate(targets_acc[granularity]) if targets_acc[granularity] else np.array([])
        cls_metrics = classification_metrics(y_pred, y_true, num_classes=num_classes[granularity])
        for k, v in cls_metrics.items():
            if k.endswith("_per_class"):
                base = k[: -len("_per_class")]
                for ci, val in enumerate(v):
                    metrics[f"acc_{granularity}_{base}_cls{ci}"] = val
            else:
                metrics[f"acc_{granularity}_{k}"] = v
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", choices=["baseline", "tuned", "aug_strong"], default="baseline")
    parser.add_argument("--dataset_sources", type=str, default=None,
                         help="comma-separated subset of {roco,rocov2} to train on (default: both)")
    parser.add_argument("--roco_manifest_override", type=str, default=None,
                         help="use an alternate manifest csv for the roco source (same image root)")
    parser.add_argument("--rocov2_manifest_override", type=str, default=None,
                         help="use an alternate manifest csv for the rocov2 source (same image root)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                         help="cap samples per dataset source (for quick smoke runs)")
    parser.add_argument("--seed", type=int, default=None,
                         help="model-init/shuffle seed for multi-seed runs (Section 5.6 statistical "
                              "analysis); appends '_seed<N>' to log/checkpoint filenames so it does "
                              "not collide with the unsuffixed run")
    parser.add_argument("--batch_size", type=int, default=None,
                         help="override hp.batch_size (e.g. to use more GPU RAM for higher throughput)")
    parser.add_argument("--num_workers", type=int, default=4,
                         help="DataLoader worker processes for image loading/augmentation")
    parser.add_argument("--amp", action="store_true", default=True,
                         help="use automatic mixed precision (fp16) on CUDA (default: on)")
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--no_spatial_attn", action="store_true", default=False,
                         help="disable the spatial cross-attention decoder (Eq. 4.1-4.5 proposed "
                              "update) and fall back to the original pooled single-vector "
                              "bottleneck decoder, for the 'original IAT-GAM-LSTM' ablation")
    parser.add_argument("--pretrained_cnn", action="store_true", default=False,
                         help="use an ImageNet-pretrained ResNet-34 backbone (models/cnn/cnn.py:"
                              "PretrainedResNetCNN) instead of the from-scratch MultiDepthCNN, "
                              "with stem/layer1/layer2 frozen and layer3/layer4 fine-tuned")
    parser.add_argument("--light_cnn", action="store_true", default=False,
                         help="use the lightweight from-scratch MobileNet-style depthwise-"
                              "separable backbone (models/cnn/cnn.py:LightCNN) instead of "
                              "MultiDepthCNN/PretrainedResNetCNN, for the efficiency ablation. "
                              "Mutually exclusive with --pretrained_cnn.")
    parser.add_argument("--iat_layers", type=int, default=7, choices=[7, 14],
                         help="number of stacked transform layers in the IAT perturbation "
                              "generator (models/iat/iat.py:InvasivePerturbation), for the "
                              "IAT-depth ablation (7-layer vs 14-layer)")
    parser.add_argument("--small_capacity", action="store_true", default=False,
                         help="shrink cnn_depth=17, embedding_dim/lstm_hidden=192 (fusion_dim/"
                              "gan_feature_dim left at defaults) -> ~45.3M total params, matching "
                              "the ~42-43M scale of the SPAN/U2DP/ClipCap++ literature baselines, "
                              "for a parameter-count-controlled comparison (isolates architecture "
                              "from raw capacity). Requires the raw_channels fix in models/cnn/"
                              "cnn.py + models/iatgam_model.py (2026-09) -- cnn_depth=7 with "
                              "fusion_dim/gan_feature_dim also overridden was tried first and hit "
                              "a second, unresolved shape mismatch in fusion/state_builder; this "
                              "cnn_depth=17-only variant avoids that by leaving those dims alone.")
    parser.add_argument("--tiny_capacity", action="store_true", default=False,
                         help="shrink to ~5.4M total params (cnn_depth=7, embedding_dim/"
                              "lstm_hidden=64, fusion_dim=128, gan_base_channels=32 + light "
                              "GAP-based discriminator head) -- matches Tiny TR-CAP's scale "
                              "(2.8M). Mutually exclusive with --small_capacity. Uses the new "
                              "opt-in GAN light_head (models/gan/gan.py, 2026-09); default "
                              "GAN behaviour is unchanged for every other flag combination.")
    parser.add_argument("--macro_capacity", action="store_true", default=False,
                         help="~12.6M total params (verified, cnn_depth=7, embedding_dim/lstm_hidden=96, "
                              "fusion_dim=192, gan_base_channels=48 + light GAP-based discriminator "
                              "head) -- between --tiny_capacity (5.4M) and --small_capacity (45.3M). "
                              "2026-11: added to test whether micro_capacity (2.4M) is an "
                              "uncontrolled capacity ceiling on the real classification numbers, "
                              "before concluding the architecture itself is the limit.")
    parser.add_argument("--micro_capacity", action="store_true", default=False,
                         help="shrink to ~2.4M total params (cnn_depth=7, embedding_dim/"
                              "lstm_hidden=16, fusion_dim=32, gan_base_channels=8 + light "
                              "GAP-based discriminator head) -- fastest verified variant; true "
                              "1M requires fixing a separate cnn_feature_dim coupling bug in "
                              "fusion/LMM, not attempted here (2026-09). Mutually exclusive "
                              "with --small_capacity/--tiny_capacity.")
    parser.add_argument("--balance_classes", action="store_true", default=False,
                         help="oversample rare / undersample common 5-class labels via a "
                              "WeightedRandomSampler on the training loader (complements the "
                              "existing CLASS_WEIGHTS_5CLASS-weighted loss, doesn't replace it)")
    parser.add_argument("--attn_entropy_weight", type=float, default=0.01,
                         help="lambda_attn_entropy in CompositeLoss (training/losses.py); default "
                              "0.01 was found (2026-09) to be ~800x weaker than the caption loss "
                              "it's meant to balance, letting the spatial-attention decoder "
                              "collapse onto a fixed, uninformative region. Try 0.5-2.0.")
    parser.add_argument("--soft_class_weights", action="store_true", default=False,
                         help="use sqrt(inverse-frequency) instead of full inverse-frequency for "
                              "CLASS_WEIGHTS_5CLASS/3CLASS/BINARY -- reduces gradient-variance "
                              "amplification from heavily-upweighted rare classes on small "
                              "datasets/batches (found 2026-09: val classification loss/accuracy "
                              "swings ~20+ points epoch-to-epoch, traced to this).")
    parser.add_argument("--binary_weight_power", type=float, default=None,
                         help="override the exponent applied to CLASS_WEIGHTS_BINARY only (3class/"
                              "5class untouched), as weight_i = raw_weight_i ** power. "
                              "--soft_class_weights uses power=0.5 for all three heads; this lets "
                              "binary use a milder power (e.g. 0.25) than 3class/5class without "
                              "disturbing those two heads' already-working results (2026-11: every "
                              "attempt to fix binary minority recall via sampler/focal collapsed to "
                              "0% recall on one class or the other -- testing whether a strictly "
                              "milder correction than the existing sqrt/soft weight avoids that).")
    parser.add_argument("--focal_gamma", type=float, default=0.0,
                         help="focal-loss modulating exponent on top of class-weighted CE "
                              "(training/losses.py:classification_loss) -- 0.0 (default) = plain "
                              "weighted CE, unchanged from every prior run. 2.0 is the standard "
                              "focal-loss default; try that first for the binary-minority-recall fix.")
    parser.add_argument("--augment_train", action="store_true", default=False,
                         help="apply real crop-flip augmentation to training images every batch "
                              "(training/aug_methods.py:crop_flip), on top of the WeightedRandomSampler's "
                              "oversampling -- gives oversampled minority-class images varied views "
                              "instead of byte-identical repeats (2026-11 binary-minority-recall fix).")
    parser.add_argument("--leakage_safe_audit_path", type=str, default=None,
                         help="path to duplicate_audit_full.json (scripts/audit_duplicates.py "
                              "output); when set, uses the global cross-source leakage-safe "
                              "split (data/datasets.py:build_leakage_safe_split) instead of the "
                              "per-source split_indices_grouped, so ROCO<->ROCOv2 duplicate/"
                              "near-duplicate images can't cross a train/val/test boundary.")
    parser.add_argument("--disable_lstm", action="store_true", default=False,
                         help="ablation switch: zero out the LSTM's h_T contribution to the "
                              "fusion state (module still constructed; see IATGAMModel.forward). "
                              "Part of the 2x2x2 {LSTM, GAN, IAT} ablation matrix.")
    parser.add_argument("--disable_gan", action="store_true", default=False,
                         help="ablation switch: skip the GAN forward pass entirely and zero out "
                              "its gan_feat contribution to the fusion state. Part of the "
                              "2x2x2 {LSTM, GAN, IAT} ablation matrix.")
    parser.add_argument("--disable_iat", action="store_true", default=False,
                         help="ablation switch: skip the IAT augmentation branch (apply_iat=False "
                              "in IATGAMModel.forward). Part of the 2x2x2 {LSTM, GAN, IAT} "
                              "ablation matrix -- this is the flag that reproduces the existing "
                              "'without IAT' comparison (Table 5.11) when LSTM/GAN stay enabled.")
    parser.add_argument("--gam_memory_coupling", action="store_true", default=False,
                         help="enable the GAM Memory Bank (models/gan/gam_memory.py): couples the "
                              "GAN discriminator's per-class realism signal into IAT's per-sample "
                              "perturbation strength via a persistent EMA memory, so 'GAM' "
                              "(Generative Adversarial Memory) has an actual memory mechanism "
                              "instead of being a plain conditional GAN. No effect if "
                              "--disable_gan or --disable_iat is also set (nothing to couple).")
    parser.add_argument("--gan_feature_space", action="store_true", default=False,
                         help="use the feature-space GAM redesign (models/gan/feature_gan.py, "
                              "WGAN-GP) instead of the original pixel-space ConditionalGAN. "
                              "2026-11: the pixel-space GAN scored FID_gan_vs_real=431.9 on the "
                              "real test set (unusable); this redesign generates/discriminates "
                              "directly in the CNN's 512-dim feature space instead, which is a "
                              "much easier distribution to match and removes the pixel-realism "
                              "failure mode by construction. Default False preserves the original "
                              "pixel-space GAN exactly for any in-flight/existing experiments.")
    args = parser.parse_args()
    if args.light_cnn and args.pretrained_cnn:
        raise ValueError("--light_cnn and --pretrained_cnn are mutually exclusive")
    if sum([args.tiny_capacity, args.small_capacity, args.micro_capacity]) > 1:
        raise ValueError("--tiny_capacity, --small_capacity, --micro_capacity are mutually exclusive")

    hp = get_hparams(args.hparams)
    epochs = args.epochs or hp.epochs
    device = config.get_device()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    tag = args.hparams if args.seed is None else f"{args.hparams}_seed{args.seed}"
    if args.no_spatial_attn:
        tag = f"{tag}_nospatial"
    if args.pretrained_cnn:
        tag = f"{tag}_pretrainedcnn"
    if args.light_cnn:
        tag = f"{tag}_lightcnn"
    if args.iat_layers != 7:
        tag = f"{tag}_iat{args.iat_layers}"
    if args.small_capacity:
        tag = f"{tag}_small46m"
    if args.tiny_capacity:
        tag = f"{tag}_tiny5m"
    if args.macro_capacity:
        tag = f"{tag}_macro13m"
    if args.micro_capacity:
        tag = f"{tag}_micro2m"
    if args.attn_entropy_weight != 0.01:
        tag = f"{tag}_ae{args.attn_entropy_weight}"
    if args.soft_class_weights:
        tag = f"{tag}_softcw"
    if args.balance_classes:
        tag = f"{tag}_balcls"
    if args.augment_train:
        tag = f"{tag}_augtrain"
    if args.focal_gamma > 0:
        tag = f"{tag}_focal{args.focal_gamma}"
    if args.binary_weight_power is not None:
        tag = f"{tag}_binpow{args.binary_weight_power}"
    if args.disable_lstm:
        tag = f"{tag}_noLSTM"
    if args.disable_gan:
        tag = f"{tag}_noGAN"
    if args.disable_iat:
        tag = f"{tag}_noIAT"
    if args.gam_memory_coupling:
        tag = f"{tag}_gammem"
    if args.gan_feature_space:
        tag = f"{tag}_featgan"

    if not os.path.exists(VOCAB_PATH):
        # MedVideoCap-55K dropped 2026-09-02 -- ROCO + ROCOv2 only.
        manifests = [config.ROCO_MANIFEST, config.ROCOV2_MANIFEST]
        vocab = build_vocab(manifests)
        vocab.save(VOCAB_PATH)
    else:
        vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    batch_size = args.batch_size or hp.batch_size
    dataset_sources = args.dataset_sources.split(",") if args.dataset_sources else None
    train_loader, val_loader, _ = build_dataloaders(vocab, batch_size, args.limit,
                                                     num_workers=args.num_workers, balance_sources=True,
                                                     balance_classes=args.balance_classes,
                                                     leakage_safe_audit_path=args.leakage_safe_audit_path,
                                                     dataset_sources=dataset_sources,
                                                     roco_manifest_override=args.roco_manifest_override,
                                                     rocov2_manifest_override=args.rocov2_manifest_override)

    model_cfg_kwargs = dict(
        vocab_size=len(vocab), embedding_dim=hp.embedding_dim, lstm_hidden=hp.lstm_hidden,
        dropout=hp.dropout,
        iat_alpha_init=hp.iat_alpha_init, iat_ssim_threshold=hp.iat_ssim_threshold,
        use_spatial_attention=not args.no_spatial_attn,
        cnn_pretrained=args.pretrained_cnn,
        cnn_light=args.light_cnn,
        iat_num_layers=args.iat_layers,
        gam_memory_coupling=args.gam_memory_coupling,
        gan_feature_space=args.gan_feature_space,
    )
    if args.small_capacity:
        # ~45.3M total params, matching the ~42-43M scale of SPAN/U2DP/
        # ClipCap++ for a capacity-controlled comparison against the
        # default 85.5M full-capacity model. Verified forward-pass-clean;
        # see --small_capacity help text for why fusion_dim/gan_feature_dim
        # are left at their defaults.
        model_cfg_kwargs.update(cnn_depth=17, embedding_dim=192, lstm_hidden=192)
    if args.tiny_capacity:
        # ~5.4M total params, matching Tiny TR-CAP's scale (2.8M). Uses the
        # opt-in GAN light_head (models/gan/gan.py) since the original
        # flatten-based discriminator alone would exceed this budget at
        # any embedding size. Forward-pass verified clean.
        model_cfg_kwargs.update(cnn_depth=7, embedding_dim=64, lstm_hidden=64, fusion_dim=128,
                                 gan_base_channels=32, gan_light_head=True)
    if args.macro_capacity:
        # ~12.6M total params (verified via instantiation) -- between tiny (5.4M) and small (45.3M).
        model_cfg_kwargs.update(cnn_depth=7, embedding_dim=96, lstm_hidden=96, fusion_dim=192,
                                 gan_base_channels=48, gan_light_head=True)
    if args.micro_capacity:
        # ~2.37M total params -- fastest verified variant. Forward-pass
        # tested clean. True ~1M would need cnn_feature_dim shrunk too, but
        # that hits an unrelated, unfixed dimension-coupling bug in
        # fusion/LMM (2026-09) -- not attempted, so cnn_feature_dim stays
        # at its default 512 here.
        model_cfg_kwargs.update(cnn_depth=7, embedding_dim=16, lstm_hidden=16, fusion_dim=32,
                                 gan_base_channels=8, gan_light_head=True)
    model_cfg = config.ModelConfig(**model_cfg_kwargs)
    model = IATGAMModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=hp.learning_rate * 0.01)

    def _weights(vals, power_override=None):
        power = power_override if power_override is not None else (0.5 if args.soft_class_weights else 1.0)
        return [v ** power for v in vals]

    class_weights = {
        "binary": torch.tensor(_weights(CLASS_WEIGHTS_BINARY, args.binary_weight_power), device=device),
        "3class": torch.tensor(_weights(CLASS_WEIGHTS_3CLASS), device=device),
        "5class": torch.tensor(_weights(CLASS_WEIGHTS_5CLASS), device=device),
    }
    print("class_weights (soft=%s, binary_power_override=%s): binary=%s 5class=%s" %
          (args.soft_class_weights, args.binary_weight_power,
           class_weights["binary"].tolist(), class_weights["5class"].tolist()))
    criterion = CompositeLoss(lambda_gan=hp.lambda_gan, lambda_temporal=hp.lambda_temporal, pad_id=pad_id,
                               class_weights=class_weights, label_smoothing=0.1,
                               lambda_attn_entropy=args.attn_entropy_weight,
                               focal_gamma=args.focal_gamma)

    log_path = os.path.join(config.LOGS_DIR, f"train_log_{tag}.csv")
    log_rows = []
    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, pad_id, hp.grad_clip_norm,
                                    use_amp=args.amp, apply_iat=not args.disable_iat,
                                    disable_lstm=args.disable_lstm, disable_gan=args.disable_gan,
                                    augment_train=args.augment_train)
        val_metrics = run_epoch(model, val_loader, criterion, None, device, pad_id, use_amp=args.amp,
                                  apply_iat=not args.disable_iat,
                                  disable_lstm=args.disable_lstm, disable_gan=args.disable_gan)
        scheduler.step()
        print(f"[epoch {epoch}/{epochs}] lr={scheduler.get_last_lr()[0]:.6f} train={train_metrics} val={val_metrics}")

        ckpt_path = os.path.join(config.MODEL_DIR, f"iatgam_{tag}_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "config": model_cfg, "epoch": epoch,
                    "ablation": {"disable_lstm": args.disable_lstm, "disable_gan": args.disable_gan,
                                 "disable_iat": args.disable_iat}}, ckpt_path)

        # Best-checkpoint selection: track val composite loss (not just save
        # every epoch's checkpoint) so a late-epoch instability spike (as
        # observed in train_log_baseline.csv, epoch 30) doesn't silently
        # become the checkpoint used for downstream evaluation.
        val_loss = val_metrics.get("total", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(config.MODEL_DIR, f"iatgam_{tag}_best.pt")
            torch.save({"model": model.state_dict(), "config": model_cfg, "epoch": epoch,
                        "val_loss": val_loss,
                        "ablation": {"disable_lstm": args.disable_lstm, "disable_gan": args.disable_gan,
                                     "disable_iat": args.disable_iat}}, best_path)
            print(f"  -> new best (val_total={val_loss:.4f}), saved {best_path}")

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        log_rows.append(row)

        fieldnames = list(log_rows[0].keys())
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(log_rows)

    print(f"Best epoch: {best_epoch} (val_total={best_val_loss:.4f}) -> iatgam_{tag}_best.pt")


if __name__ == "__main__":
    main()

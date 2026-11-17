"""
Step 1 — Image + Text Classification  (CNN DL-39 + CaptionEncoder)

Full tensor pipeline:
  HDD  ->  RAM cache  (once at startup, ~15 min for 97K images)
  RAM  ->  GPU        (every batch, num_workers=4 + pin_memory)
  GPU  ->  AMP fp16   (2x speed, half VRAM)

After pre-load, HDD is never read again across all 100 epochs.
Expected total: ~7-10 hours for 100 epochs on full 153K dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import (ConcatDataset, DataLoader, Dataset, Subset,
                               WeightedRandomSampler)
from torchvision import transforms
from torchvision.transforms import functional as TF
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import config
from data.datasets import build_datasets, split_indices
from data.vocab import PAD, Vocabulary, VOCAB_PATH, build_vocab
from models.cnn.cnn import MultiDepthCNN
from models.text.text_encoder import CaptionEncoder

# ---------------------------------------------------------------------------
# Augmentation (tensor-level — applied after RAM cache serves float32 image)
# ---------------------------------------------------------------------------
_TRAIN_AUG = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

CLASS_NAMES_5 = ["X-ray", "CT", "MRI", "Ultrasound", "Video/Other"]
CLASS_NAMES_3 = ["X-ray", "CT+MRI+US", "Video"]
CLASS_NAMES_B = ["Medical", "Non-medical"]


# ---------------------------------------------------------------------------
# HDD → RAM image cache
# ---------------------------------------------------------------------------

class ImageRAMCache:
    """
    Pre-loads every real image from HDD into RAM as uint8 tensors (3x less
    memory than float32).  Zero-images (MedVideoCap, no image_path) are NOT
    stored — they are returned as zero tensors on demand.

    Memory usage (uint8):
      97K images × 3×224×224 × 1 byte ≈ 14.7 GB
      97K images × 3×112×112 × 1 byte ≈  3.7 GB  (with --img_size 112)
    """

    def __init__(self, img_size=(224, 224)):
        self._cache: dict[str, torch.Tensor] = {}
        self._img_size = img_size
        self._resize = transforms.Resize(img_size)
        self._zeros  = torch.zeros(3, *img_size)

    def preload(self, datasets: dict):
        """Load all images from all ManifestDataset sources into RAM."""
        t0 = time.time()
        total_loaded = 0
        for name, ds in datasets.items():
            source_loaded = 0
            for _, row in ds.df.iterrows():
                img_path = str(row["image_path"])
                if not img_path:
                    continue
                abs_path = os.path.join(ds.root_dir, img_path)
                if abs_path in self._cache:
                    source_loaded += 1
                    continue
                if not os.path.exists(abs_path):
                    continue
                try:
                    img = Image.open(abs_path).convert("RGB")
                    t   = TF.to_tensor(self._resize(img))   # float32 [0,1]
                    self._cache[abs_path] = (t * 255).byte()  # uint8 saves 4x RAM
                    source_loaded += 1
                except Exception:
                    pass
            total_loaded += source_loaded
            print(f"    {name}: {source_loaded:,} images cached")

        elapsed  = time.time() - t0
        cache_gb = sum(v.nbytes for v in self._cache.values()) / 1e9
        print(f"  Pre-load done: {total_loaded:,} images | "
              f"RAM used = {cache_gb:.1f} GB | time = {elapsed:.0f}s")

    def get(self, root_dir: str, image_path: str) -> torch.Tensor:
        if not image_path:
            return self._zeros.clone()
        abs_path = os.path.join(root_dir, image_path)
        t = self._cache.get(abs_path)
        if t is None:
            return self._zeros.clone()
        return t.float() / 255.0   # back to float32 [0,1]


# ---------------------------------------------------------------------------
# Dataset wrapper that reads images from RAM cache
# ---------------------------------------------------------------------------

class _ManifestSlice:
    """Picklable slice of a ManifestDataset (needed for num_workers > 0 on Windows)."""
    def __init__(self, df, root_dir, vocab, max_len):
        self.df       = df
        self.root_dir = root_dir
        self.vocab    = vocab
        self.max_len  = max_len


class CachedDataset(Dataset):
    """
    Replaces ManifestDataset's disk-based _load_image with an O(1) RAM lookup.
    Augmentation (aug) is applied only for the training split.
    """

    def __init__(self, manifest_ds, cache: ImageRAMCache, aug=None):
        self.ds    = manifest_ds
        self.cache = cache
        self.aug   = aug

    def __len__(self):
        return len(self.ds.df)

    def __getitem__(self, idx: int):
        row = self.ds.df.iloc[idx]
        img_path = str(row["image_path"])

        # Image from RAM (no HDD read)
        image = self.cache.get(self.ds.root_dir, img_path)
        if self.aug is not None:
            image = self.aug(image)

        caption_ids  = self.ds.vocab.encode(str(row["caption"]), self.ds.max_len)
        summary_text = str(row.get("summary", "")) or str(row["caption"])
        summary_ids  = self.ds.vocab.encode(summary_text, self.ds.max_len)
        label        = int(row["label"]) if str(row["label"]) not in ("", "nan") else -1

        return {
            "id":           str(row["id"]),
            "image":        image,
            "has_image":    bool(img_path),
            "caption_ids":  torch.tensor(caption_ids, dtype=torch.long),
            "summary_ids":  torch.tensor(summary_ids, dtype=torch.long),
            "label":        torch.tensor(label,              dtype=torch.long),
            "label_binary": torch.tensor(int(row["label_binary"]), dtype=torch.long),
            "label_3class": torch.tensor(int(row["label_3class"]), dtype=torch.long),
            "label_5class": torch.tensor(int(row["label_5class"]), dtype=torch.long),
            "video_id":     str(row["video_id"]),
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Step1Classifier(nn.Module):
    def __init__(self, vocab_size: int,
                 cnn_depth=39, cnn_feature_dim=512,
                 embed_dim=256, num_layers=4, num_heads=8, dropout=0.2):
        super().__init__()
        self.cnn             = MultiDepthCNN(depth=cnn_depth, feature_dim=cnn_feature_dim)
        self.caption_encoder = CaptionEncoder(vocab_size, embed_dim, num_layers,
                                              num_heads, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim + embed_dim, 256),
            nn.LayerNorm(256), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.head_binary = nn.Linear(256, 2)
        self.head_3class = nn.Linear(256, 3)
        self.head_5class = nn.Linear(256, 5)

    def forward(self, images, caption_ids, caption_mask):
        f_v     = self.cnn(images)
        h_c_seq = self.caption_encoder(caption_ids, caption_mask)
        mask_f  = caption_mask.float().unsqueeze(-1)
        h_c     = (h_c_seq * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        z       = self.fusion(torch.cat([f_v, h_c], dim=1))
        return {"logits_binary": self.head_binary(z),
                "logits_3class": self.head_3class(z),
                "logits_5class": self.head_5class(z)}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def build_loaders(vocab, batch_size, limit, cache: ImageRAMCache):
    sources = build_datasets(vocab)
    # Pre-load ALL images into RAM (HDD read happens only here)
    print("\nPre-loading images from HDD into RAM ...")
    cache.preload(sources)

    train_sets, val_sets, test_sets = [], [], []
    lbl_b, lbl_3, lbl_5 = [], [], []

    for name, ds in sources.items():
        n = len(ds) if limit is None else min(limit, len(ds))
        tr_idx, va_idx, te_idx = split_indices(n)

        train_sets.append(CachedDataset(
            _ManifestSlice(ds.df.iloc[list(tr_idx)].reset_index(drop=True),
                           ds.root_dir, ds.vocab, ds.max_len),
            cache, aug=_TRAIN_AUG))
        val_sets.append(CachedDataset(
            _ManifestSlice(ds.df.iloc[list(va_idx)].reset_index(drop=True),
                           ds.root_dir, ds.vocab, ds.max_len),
            cache, aug=None))
        test_sets.append(CachedDataset(
            _ManifestSlice(ds.df.iloc[list(te_idx)].reset_index(drop=True),
                           ds.root_dir, ds.vocab, ds.max_len),
            cache, aug=None))

        df_tr = ds.df.iloc[list(tr_idx)]
        lbl_b.extend(df_tr["label_binary"].astype(int).tolist())
        lbl_3.extend(df_tr["label_3class"].astype(int).tolist())
        lbl_5.extend(df_tr["label_5class"].astype(int).tolist())
        print(f"    {name}: {n:,}  train={len(tr_idx):,} "
              f"val={len(va_idx):,} test={len(te_idx):,}")

    cnt_b, cnt_3, cnt_5 = Counter(lbl_b), Counter(lbl_3), Counter(lbl_5)

    # WeightedRandomSampler — fine now that images come from RAM
    total5  = len(lbl_5)
    cw5     = {c: total5 / max(cnt_5[c], 1) for c in range(5)}
    samp_w  = [cw5[l] for l in lbl_5]
    sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)
    print(f"  5-class counts: {dict(sorted(cnt_5.items()))}")

    # num_workers=4 + pin_memory: images pre-fetched in parallel while GPU trains
    nw = min(4, os.cpu_count() or 1)
    kw_tr = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)
    kw_ev = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0)

    return (
        DataLoader(ConcatDataset(train_sets), batch_size=batch_size,
                   sampler=sampler, **kw_tr),
        DataLoader(ConcatDataset(val_sets),   batch_size=batch_size,
                   shuffle=False, **kw_ev),
        DataLoader(ConcatDataset(test_sets),  batch_size=batch_size,
                   shuffle=False, **kw_ev),
        (cnt_b, cnt_3, cnt_5),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _acc(logits, targets):
    return (logits.argmax(-1) == targets).float().mean().item()


def run_epoch(model, loader, optimiser, device, w_b, w_3, w_5,
              pad_id, scaler, is_train=True, log_every=0):
    model.train(is_train)
    ce = nn.CrossEntropyLoss
    total_loss = total_b = total_3 = total_5 = n = 0
    intra = []

    for bi, batch in enumerate(loader):
        images   = batch["image"].to(device, non_blocking=True)
        cap_ids  = batch["caption_ids"].to(device, non_blocking=True)
        cap_mask = (cap_ids != pad_id).long()
        lb = batch["label_binary"].to(device, non_blocking=True)
        l3 = batch["label_3class"].to(device, non_blocking=True)
        l5 = batch["label_5class"].to(device, non_blocking=True)

        if (l5 < 0).all():
            continue

        with torch.set_grad_enabled(is_train):
            # AMP: forward in fp16, loss in fp32
            with autocast("cuda", enabled=(scaler is not None)):
                out  = model(images, cap_ids, cap_mask)
                loss = (ce(weight=w_b, label_smoothing=0.05)(out["logits_binary"], lb) +
                        ce(weight=w_3, label_smoothing=0.05)(out["logits_3class"], l3) +
                        ce(weight=w_5, label_smoothing=0.05)(out["logits_5class"], l5))

            if is_train:
                optimiser.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_b    += _acc(out["logits_binary"], lb) * bs
        total_3    += _acc(out["logits_3class"], l3) * bs
        total_5    += _acc(out["logits_5class"], l5) * bs
        n          += bs

        if log_every > 0 and is_train and (bi + 1) % log_every == 0:
            ab = total_b / n; a3 = total_3 / n; a5 = total_5 / n
            print(f"    batch {bi+1:>5}  Binary={ab*100:.2f}%  "
                  f"3cls={a3*100:.2f}%  5cls={a5*100:.2f}%", flush=True)
            intra.append((bi + 1, ab, a3, a5))

    return {"loss": total_loss / max(n, 1),
            "acc_binary": total_b / max(n, 1),
            "acc_3class": total_3 / max(n, 1),
            "acc_5class": total_5 / max(n, 1),
            "intra": intra}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_full(model, loader, device, pad_id):
    model.eval()
    pb, p3, p5, tb, t3, t5 = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            images  = batch["image"].to(device, non_blocking=True)
            cap_ids = batch["caption_ids"].to(device, non_blocking=True)
            mask    = (cap_ids != pad_id).long()
            with autocast("cuda"):
                out = model(images, cap_ids, mask)
            pb.extend(out["logits_binary"].argmax(-1).cpu().tolist())
            p3.extend(out["logits_3class"].argmax(-1).cpu().tolist())
            p5.extend(out["logits_5class"].argmax(-1).cpu().tolist())
            tb.extend(batch["label_binary"].tolist())
            t3.extend(batch["label_3class"].tolist())
            t5.extend(batch["label_5class"].tolist())
    return (tb, pb), (t3, p3), (t5, p5)


def _report(y_true, y_pred, names, title):
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    labels = list(range(len(names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    _, _, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    print(f"\n{'='*64}\n  {title}\n{'='*64}")
    print(f"  Accuracy : {acc*100:.2f}%   Macro F1 : {mf*100:.2f}%")
    print(f"\n  {'Class':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Supp':>6}")
    print(f"  {'-'*48}")
    for i, nm in enumerate(names):
        print(f"  {nm:<18} {p[i]*100:>6.2f}% {r[i]*100:>6.2f}% "
              f"{f[i]*100:>6.2f}% {s[i]:>6}")
    return {"accuracy": float(acc), "macro_f1": float(mf),
            "per_class": {names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                     "f1": float(f[i]), "support": int(s[i])}
                          for i in range(len(names))}}


def _conf_matrix_plot(y_true, y_pred, names, title, path):
    cm   = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(len(names)*1.6+1, len(names)*1.4+1))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i,j]}\n{norm[i,j]*100:.1f}%",
                    ha="center", va="center", fontsize=7,
                    color="white" if norm[i,j] > 0.55 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved -> {path}")


def _plot_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, key, ylabel in zip(
        axes,
        ["loss", "acc_binary", "acc_3class", "acc_5class"],
        ["Loss", "Binary Acc", "3-Class Acc", "5-Class Acc"],
    ):
        ax.plot(epochs, [h["train_"+key] for h in history], "b-o", ms=3, label="train")
        ax.plot(epochs, [h["val_"+key]   for h in history], "r-s", ms=3, label="val")
        if "acc" in key:
            ax.axhline(0.987, color="green", ls="--", lw=1.2, label="98.7% target")
            ax.set_ylim(0.3, 1.02)
        ax.set_title(ylabel, fontweight="bold"); ax.set_xlabel("Epoch")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Step 1: CNN(DL-39) + CaptionEncoder  [RAM-cached, AMP fp16]",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Curves -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--limit",      type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--dropout",    type=float, default=0.2)
    parser.add_argument("--log_every",  type=int,   default=200)
    parser.add_argument("--img_size",   type=int,   default=224,
                        help="image resize (112=4x faster CNN, 224=default)")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="path to checkpoint to resume from")
    parser.add_argument("--no_amp",     action="store_true",
                        help="disable AMP (use full float32)")
    args = parser.parse_args()

    device   = config.get_device()
    use_amp  = (not args.no_amp) and torch.cuda.is_available()
    scaler   = GradScaler("cuda") if use_amp else None  # torch.amp.GradScaler
    img_size = (args.img_size, args.img_size)
    print(f"Device : {device}   AMP : {use_amp}   ImgSize : {img_size}")

    # Vocab
    if not os.path.exists(VOCAB_PATH):
        vocab = build_vocab([config.ROCO_MANIFEST, config.ROCOV2_MANIFEST,
                             config.MEDVIDEOCAP_MANIFEST])
        vocab.save(VOCAB_PATH)
    else:
        vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]
    print(f"Vocab  : {len(vocab):,} tokens")

    # RAM cache (HDD → RAM, done once)
    cache = ImageRAMCache(img_size=img_size)

    # Data
    train_loader, val_loader, test_loader, (cnt_b, cnt_3, cnt_5) = build_loaders(
        vocab, args.batch_size, args.limit, cache)
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    print(f"\nTrain={n_train:,}  Val={n_val:,}  Test={n_test:,}")
    print(f"Batches/epoch: {len(train_loader):,}  (batch_size={args.batch_size})", flush=True)

    # Model
    cfg   = config.DEFAULT_MODEL_CONFIG
    model = Step1Classifier(
        vocab_size=len(vocab),
        cnn_depth=cfg.cnn_depth, cnn_feature_dim=cfg.cnn_feature_dim,
        embed_dim=cfg.embedding_dim, num_layers=cfg.transformer_layers,
        num_heads=cfg.transformer_heads, dropout=args.dropout,
    ).to(device)
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params : {n_p:,}")

    # Class weights from CSV (instant)
    def _w(cnt, nc):
        tot = sum(cnt.values())
        return torch.tensor([(tot / (nc * max(cnt.get(c,1),1))) for c in range(nc)],
                            dtype=torch.float32, device=device)
    w_b = _w(cnt_b, 2); w_3 = _w(cnt_3, 3); w_5 = _w(cnt_5, 5)
    print(f"Weights(5cls): {[round(x,3) for x in w_5.tolist()]}", flush=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    best_ckpt  = os.path.join(config.MODEL_DIR, "step1_image_text_cls_best.pt")
    best_val5  = 0.0
    history    = []
    start_epoch = 1

    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimiser" in ckpt:
            optimiser.load_state_dict(ckpt["optimiser"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "best_val5" in ckpt:
            best_val5 = ckpt["best_val5"]
        start_epoch = ckpt.get("epoch", 0) + 1
        # Fast-forward scheduler to correct LR if no scheduler state saved
        if "scheduler" not in ckpt:
            for _ in range(start_epoch - 1):
                scheduler.step()
        print(f"Resumed from epoch {start_epoch-1}  "
              f"(best_val5={best_val5*100:.2f}%  lr={scheduler.get_last_lr()[0]:.2e})", flush=True)

    print(f"\n{'Ep':>4} {'Time':>7} {'Loss':>8} "
          f"{'Tr-B':>7} {'Tr-3':>7} {'Tr-5':>7} "
          f"{'Va-B':>7} {'Va-3':>7} {'Va-5':>7}")
    print("-" * 78)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimiser, device,
                       w_b, w_3, w_5, pad_id, scaler,
                       is_train=True, log_every=args.log_every)
        vl = run_epoch(model, val_loader, None, device,
                       w_b, w_3, w_5, pad_id, None,
                       is_train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"{epoch:>4} {elapsed:>6.0f}s {tr['loss']:>8.4f} "
              f"{tr['acc_binary']*100:>6.2f}% {tr['acc_3class']*100:>6.2f}% "
              f"{tr['acc_5class']*100:>6.2f}% "
              f"{vl['acc_binary']*100:>6.2f}% {vl['acc_3class']*100:>6.2f}% "
              f"{vl['acc_5class']*100:>6.2f}%", flush=True)

        history.append({"epoch": epoch, "epoch_time_s": round(elapsed, 1),
                        **{f"train_{k}": v for k, v in tr.items() if k != "intra"},
                        **{f"val_{k}":   v for k, v in vl.items() if k != "intra"}})

        if vl["acc_5class"] > best_val5:
            best_val5 = vl["acc_5class"]
            torch.save({"model":     model.state_dict(),
                        "optimiser": optimiser.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch":     epoch,
                        "best_val5": best_val5,
                        "vocab_size": len(vocab),
                        "args":      vars(args)}, best_ckpt)
            print(f"  ** Best checkpoint saved (val_5class={best_val5*100:.2f}%)", flush=True)

    # Final test evaluation
    print(f"\nLoading best checkpoint (val_5class={best_val5*100:.2f}%) ...")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    print("\n[Test Set Evaluation]")
    (tb, pb), (t3, p3), (t5, p5) = evaluate_full(model, test_loader, device, pad_id)

    results = {}
    results["binary"] = _report(tb, pb, CLASS_NAMES_B,
        "Binary  (Medical vs Non-medical)")
    results["3class"] = _report(t3, p3, CLASS_NAMES_3,
        "3-Class (X-ray / CT+MRI+US / Video)")
    results["5class"] = _report(t5, p5, CLASS_NAMES_5,
        "5-Class (X-ray / CT / MRI / US / Video)")

    print("\n--- Gap to 98.7% target ---")
    for key, label in [("binary","Binary"), ("3class","3-Class"), ("5class","5-Class")]:
        acc = results[key]["accuracy"] * 100
        gap = 98.7 - acc
        status = "ACHIEVED" if gap <= 0 else f"need +{gap:.2f}%"
        print(f"  {label:<10}: {acc:.2f}%  ({status})")

    # Save outputs
    os.makedirs(config.LOGS_DIR,  exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    results["history"] = history
    results["best_val_5"] = best_val5

    log_path = os.path.join(config.LOGS_DIR, "step1_image_text_cls_results.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics    -> {log_path}")
    print(f"  Checkpoint -> {best_ckpt}")

    _plot_curves(history,
        os.path.join(config.PLOTS_DIR, "fig_step1_image_text_curves.png"))
    _conf_matrix_plot(tb, pb, CLASS_NAMES_B, "Binary",
        os.path.join(config.PLOTS_DIR, "fig_step1_binary_cm.png"))
    _conf_matrix_plot(t3, p3, CLASS_NAMES_3, "3-Class",
        os.path.join(config.PLOTS_DIR, "fig_step1_3class_cm.png"))
    _conf_matrix_plot(t5, p5, CLASS_NAMES_5, "5-Class",
        os.path.join(config.PLOTS_DIR, "fig_step1_5class_cm.png"))
    print("\nDone.")


if __name__ == "__main__":
    main()

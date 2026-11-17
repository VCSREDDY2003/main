"""
Step 1 (IAT-GAM-LSTM architecture): Caption-text classification using the
CaptionEncoder (4-layer Transformer) from the IAT-GAM-LSTM model.

This is the correct architecture-aligned version of the Step 1 analysis:
  - Uses the SAME CaptionEncoder class as IATGAMModel (4 layers, 8 heads)
  - Adds three lightweight classification heads on top
  - Trains ONLY on caption text -> binary / 3-class / 5-class label
  - No image loading, no GAN, no LSTM — text-branch only

Architecture:
    caption_ids (B, L)
        -> CaptionEncoder (Transformer 4L x 8H, embed_dim=256)
        -> H_c  (B, L, 256)   [same as in full IAT-GAM-LSTM forward()]
        -> mean-pool over non-pad positions  -> z  (B, 256)
        -> Linear(256, 2)   -> binary logits
        -> Linear(256, 3)   -> 3-class logits
        -> Linear(256, 5)   -> 5-class logits

Composite loss:
    L = L_binary + L_3class + L_5class  (cross-entropy, class_weight balanced)

Usage:
    python evaluation/train_caption_cls_transformer.py --epochs 10
    python evaluation/train_caption_cls_transformer.py --epochs 30 --limit 10000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import config
from data.vocab import PAD, Vocabulary, VOCAB_PATH, build_vocab
from models.text.text_encoder import CaptionEncoder


# ---------------------------------------------------------------------------
# Dataset: caption text -> multi-granularity labels
# ---------------------------------------------------------------------------

import re
import pandas as pd

_TO_3 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
_TO_B = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}

# Explicit modality keywords — when mask_keywords=True these are replaced with
# "<MOD>" so the model cannot simply match "ultrasound" -> class 3 and must
# learn semantic context instead. This gives a realistic 60-70% start.
_KW_MASK = re.compile(
    r"\b(x[-\s]?ray|xray|radiograph\w*|roentgen|anteroposterior|ap view|pa view"
    r"|computed tomography|tomograph\w*|\bct\b|hounsfield"
    r"|magnetic resonance|mri\b|\bmr\b|t1|t2|flair|dwi|dti|bold|weighted image"
    r"|ultrasound|ultrasonograph\w*|echocardiogram|doppler|sonograph\w*"
    r"|hypoechoic|echogenicit\w*"
    r"|video|depicts?|clinical setting|medical setting|short clip"
    r"|medical procedure|scene|patient appears?)\b",
    re.IGNORECASE,
)

CLASS_NAMES_5 = ["X-ray", "CT", "MRI", "Ultrasound", "Video/Other"]
CLASS_NAMES_3 = ["X-ray", "CT+MRI+US", "Video"]
CLASS_NAMES_B = ["Non-video", "Video"]

_MANIFESTS = [
    config.ROCO_MANIFEST,
    config.ROCOV2_MANIFEST,
    config.MEDVIDEOCAP_MANIFEST,
]


class CaptionTextDataset(Dataset):
    """Load (caption_ids, label_*) from manifest CSVs using the shared vocab.

    mask_keywords=True: replaces explicit modality terms with '<MOD>' token so
    the model learns from semantic context rather than keyword memorisation.
    This gives a realistic starting accuracy of ~60-70% instead of 96%.
    """

    def __init__(self, rows: list[dict], vocab: Vocabulary,
                 max_len: int = config.MAX_CAPTION_LEN,
                 mask_keywords: bool = False):
        self.rows         = rows
        self.vocab        = vocab
        self.max_len      = max_len
        self.mask_keywords = mask_keywords

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        cap  = str(r.get("caption", "")).strip()
        summ = str(r.get("summary", "")).strip()
        text = (cap + " " + summ).strip() if (summ and summ != cap) else cap

        if self.mask_keywords:
            text = _KW_MASK.sub("<MOD>", text)

        ids = self.vocab.encode(text, self.max_len)
        lbl5 = int(float(r["label_5class"]))
        return {
            "caption_ids":    torch.tensor(ids, dtype=torch.long),
            "label_binary":   torch.tensor(_TO_B[lbl5], dtype=torch.long),
            "label_3class":   torch.tensor(_TO_3[lbl5], dtype=torch.long),
            "label_5class":   torch.tensor(lbl5,        dtype=torch.long),
        }


def _load_rows(manifests, limit):
    """Split per-source 80/10/10 then concatenate — prevents source ordering bias.
    Uses df.to_dict('records') instead of iterrows() — ~100x faster for large CSVs."""
    train_rows, val_rows, test_rows = [], [], []
    for path in manifests:
        if not os.path.exists(path):
            print(f"  WARNING: not found: {path}")
            continue
        df = pd.read_csv(path, keep_default_na=False)
        if limit:
            df = df.iloc[:limit]

        # Filter labelled rows using vectorised pandas (fast)
        mask = ~df["label_5class"].astype(str).isin(["", "nan", "-1"])
        valid = df[mask].to_dict("records")   # list of dicts, 100x faster than iterrows

        n = len(valid)
        t = int(n * 0.8)
        v = int(n * 0.9)
        train_rows.extend(valid[:t])
        val_rows.extend(valid[t:v])
        test_rows.extend(valid[v:])
        print(f"    {os.path.basename(os.path.dirname(path))}: "
              f"{n:,} rows  (train={t:,} val={v-t:,} test={n-v:,})")

    return train_rows, val_rows, test_rows


# ---------------------------------------------------------------------------
# Text Classification model (IAT-GAM-LSTM CaptionEncoder + heads)
# ---------------------------------------------------------------------------

class CaptionClassifier(nn.Module):
    """
    IAT-GAM-LSTM text branch: CaptionEncoder -> mean-pool -> 3 cls heads.
    The encoder weights are identical in architecture to IATGAMModel.caption_encoder.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 8,
                 max_len: int = config.MAX_CAPTION_LEN,
                 dropout: float = 0.1):
        super().__init__()
        # Same CaptionEncoder used in IATGAMModel (Eq. 3.2-3.5)
        self.encoder = CaptionEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout,
        )
        # Classification heads (one per granularity)
        self.head_binary = nn.Linear(embed_dim, 2)
        self.head_3class = nn.Linear(embed_dim, 3)
        self.head_5class = nn.Linear(embed_dim, 5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, caption_ids: torch.Tensor, caption_mask: torch.Tensor):
        """
        caption_ids:   (B, L) token ids
        caption_mask:  (B, L) 1=real token, 0=pad
        returns dict with binary / 3class / 5class logits
        """
        # H_c: (B, L, embed_dim)  — same output shape as IATGAMModel.caption_encoder
        h_c = self.encoder(caption_ids, caption_mask)   # (B, L, embed_dim)

        # Mean-pool over non-padding token positions -> (B, embed_dim)
        # caption_mask: 1=real token, 0=pad
        mask_f = caption_mask.float().unsqueeze(-1)          # (B, L, 1)
        h_c = (h_c * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)  # (B, embed_dim)

        z = self.dropout(h_c)

        return {
            "logits_binary": self.head_binary(z),   # (B, 2)
            "logits_3class": self.head_3class(z),   # (B, 3)
            "logits_5class": self.head_5class(z),   # (B, 5)
        }


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def _class_weights(labels, n_classes, device):
    from collections import Counter
    counts = Counter(labels)
    total  = sum(counts.values())
    w = torch.tensor(
        [(total / (n_classes * max(counts.get(c, 1), 1))) for c in range(n_classes)],
        dtype=torch.float, device=device,
    )
    return w


def _acc(logits, targets):
    return (logits.argmax(-1) == targets).float().mean().item()


def run_epoch(model, loader, optimiser, device, w_b, w_3, w_5, pad_id,
              is_train=True, log_every=0):
    """
    log_every > 0: print running accuracy every N batches (shows intra-epoch curve).
    """
    model.train(is_train)
    ce = nn.CrossEntropyLoss

    total_loss = total_acc_b = total_acc_3 = total_acc_5 = n = 0
    intra = []   # (batch_idx, acc_binary, acc_3class, acc_5class)

    for batch_idx, batch in enumerate(loader):
        ids   = batch["caption_ids"].to(device)
        mask  = (ids != pad_id).long()
        lb    = batch["label_binary"].to(device)
        l3    = batch["label_3class"].to(device)
        l5    = batch["label_5class"].to(device)

        with torch.set_grad_enabled(is_train):
            out = model(ids, mask)
            loss = (ce(weight=w_b, label_smoothing=0.05)(out["logits_binary"], lb) +
                    ce(weight=w_3, label_smoothing=0.05)(out["logits_3class"], l3) +
                    ce(weight=w_5, label_smoothing=0.05)(out["logits_5class"], l5))

            if is_train:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

        bs = ids.size(0)
        total_loss  += loss.item() * bs
        total_acc_b += _acc(out["logits_binary"], lb) * bs
        total_acc_3 += _acc(out["logits_3class"], l3) * bs
        total_acc_5 += _acc(out["logits_5class"], l5) * bs
        n += bs

        if log_every > 0 and is_train and (batch_idx + 1) % log_every == 0:
            ab = total_acc_b / n; a3 = total_acc_3 / n; a5 = total_acc_5 / n
            print(f"    batch {batch_idx+1:>4}  Binary={ab*100:.2f}%  "
                  f"3cls={a3*100:.2f}%  5cls={a5*100:.2f}%")
            intra.append((batch_idx+1, ab, a3, a5))

    return {
        "loss":       total_loss  / max(n, 1),
        "acc_binary": total_acc_b / max(n, 1),
        "acc_3class": total_acc_3 / max(n, 1),
        "acc_5class": total_acc_5 / max(n, 1),
        "intra":      intra,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _report(y_true, y_pred, names, title):
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    labels = list(range(len(names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Macro F1  : {mf*100:.2f}%")
    print(f"\n  {'Class':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Supp':>6}")
    print(f"  {'-'*50}")
    for i, nm in enumerate(names):
        print(f"  {nm:<18} {p[i]*100:>6.2f}% {r[i]*100:>6.2f}% {f[i]*100:>6.2f}% {s[i]:>6}")
    return {"accuracy": float(acc), "macro_f1": float(mf),
            "per_class": {names[i]: {"p": float(p[i]), "r": float(r[i]),
                                      "f1": float(f[i]), "support": int(s[i])}
                          for i in range(len(names))}}


def evaluate_full(model, loader, device, pad_id):
    model.eval()
    pb, p3, p5 = [], [], []
    tb, t3, t5 = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids   = batch["caption_ids"].to(device)
            mask  = (ids != pad_id).long()
            out   = model(ids, mask)
            pb.extend(out["logits_binary"].argmax(-1).cpu().tolist())
            p3.extend(out["logits_3class"].argmax(-1).cpu().tolist())
            p5.extend(out["logits_5class"].argmax(-1).cpu().tolist())
            tb.extend(batch["label_binary"].tolist())
            t3.extend(batch["label_3class"].tolist())
            t5.extend(batch["label_5class"].tolist())
    return (tb, pb), (t3, p3), (t5, p5)


def _plot_curves(history, path):
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, key, ylabel in zip(
        axes,
        ["loss", "acc_binary", "acc_3class", "acc_5class"],
        ["Loss", "Binary Acc", "3-Class Acc", "5-Class Acc"],
    ):
        ax.plot(epochs, [h["train_"+key] for h in history], label="train", marker="o", markersize=3)
        ax.plot(epochs, [h["val_"+key]   for h in history], label="val",   marker="s", markersize=3)
        if "acc" in key:
            ax.axhline(0.987, color="red", ls="--", lw=1, label="target 98.7%")
            ax.set_ylim(0.5, 1.02)
        ax.set_title(ylabel); ax.set_xlabel("Epoch")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("IAT-GAM-LSTM CaptionEncoder — Caption Classification",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Curves -> {path}")


def _conf_matrix_plot(y_true, y_pred, names, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
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
    print(f"  Saved  -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--limit",      type=int,   default=None,
                        help="max rows per manifest (None=full dataset)")
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--embed_dim",  type=int,   default=256,
                        help="must match config.DEFAULT_MODEL_CONFIG.embedding_dim")
    parser.add_argument("--num_layers", type=int,   default=4)
    parser.add_argument("--num_heads",  type=int,   default=8)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--mask_keywords", action="store_true", default=False,
                        help="Replace explicit modality keywords with <MOD> so model "
                             "learns from context, not keyword memorisation. "
                             "Gives realistic 60-70%% start instead of 96%%.")
    parser.add_argument("--log_every",    type=int, default=0,
                        help="Print intra-epoch accuracy every N batches (0=off)")
    args = parser.parse_args()

    device = config.get_device()
    print(f"Device: {device}")

    # Vocab
    if not os.path.exists(VOCAB_PATH):
        vocab = build_vocab([config.ROCO_MANIFEST, config.ROCOV2_MANIFEST,
                             config.MEDVIDEOCAP_MANIFEST])
        vocab.save(VOCAB_PATH)
    else:
        vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]
    print(f"Vocab size: {len(vocab)}")

    # Data
    print("Loading rows from manifests (per-source 80/10/10 split) ...")
    train_rows, val_rows, test_rows = _load_rows(_MANIFESTS, args.limit)
    print(f"  Train={len(train_rows):,}  Val={len(val_rows):,}  Test={len(test_rows):,}")

    if args.mask_keywords:
        print("  [keyword masking ON] — explicit modality terms replaced with <MOD>")
        print("  Expected start: ~60-70%  Target: 98.7%")
    def _ds(rows): return CaptionTextDataset(rows, vocab,
                                              mask_keywords=args.mask_keywords)
    train_ds = _ds(train_rows); val_ds = _ds(val_rows); test_ds = _ds(test_rows)

    kw = dict(batch_size=args.batch_size, num_workers=0, pin_memory=device.type=="cuda")
    train_loader = DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    # Class weights (from train split)
    train_labels_b = [_TO_B[int(float(r["label_5class"]))] for r in train_rows]
    train_labels_3 = [_TO_3[int(float(r["label_5class"]))] for r in train_rows]
    train_labels_5 = [int(float(r["label_5class"])) for r in train_rows]
    w_b = _class_weights(train_labels_b, 2, device)
    w_3 = _class_weights(train_labels_3, 3, device)
    w_5 = _class_weights(train_labels_5, 5, device)
    print(f"  Class weights 5-class: {w_5.tolist()}")

    # Model
    model = CaptionClassifier(
        vocab_size=len(vocab), embed_dim=args.embed_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel params: {n_params:,}")
    print(f"  CaptionEncoder (4L x 8H Transformer): same as IATGAMModel.caption_encoder")

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training loop
    history = []
    best_val_acc5 = 0.0
    best_ckpt = os.path.join(config.MODEL_DIR, "caption_cls_transformer_best.pt")

    print(f"\n{'Epoch':<6} {'Time':>6} {'TrainLoss':>10} {'TrainB':>8} {'Train3':>8} "
          f"{'Train5':>8} {'ValB':>8} {'Val3':>8} {'Val5':>8}")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimiser, device, w_b, w_3, w_5,
                       pad_id, is_train=True, log_every=args.log_every)
        vl = run_epoch(model, val_loader,   None,       device, w_b, w_3, w_5,
                       pad_id, is_train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"{epoch:<6} {elapsed:>5.0f}s {tr['loss']:>10.4f} "
              f"{tr['acc_binary']*100:>7.2f}% {tr['acc_3class']*100:>7.2f}% "
              f"{tr['acc_5class']*100:>7.2f}% "
              f"{vl['acc_binary']*100:>7.2f}% {vl['acc_3class']*100:>7.2f}% "
              f"{vl['acc_5class']*100:>7.2f}%")

        history.append({"epoch": epoch, "epoch_time_s": round(elapsed, 1),
                         **{f"train_{k}": v for k,v in tr.items()},
                         **{f"val_{k}":   v for k,v in vl.items()}})

        if vl["acc_5class"] > best_val_acc5:
            best_val_acc5 = vl["acc_5class"]
            torch.save({"model": model.state_dict(), "epoch": epoch,
                         "vocab_size": len(vocab), "embed_dim": args.embed_dim,
                         "num_layers": args.num_layers, "num_heads": args.num_heads},
                        best_ckpt)

    # Load best checkpoint for final eval
    print(f"\nLoading best checkpoint (val_5class={best_val_acc5*100:.2f}%) ...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Test evaluation
    print("\n[Test Set Evaluation]")
    (tb, pb), (t3, p3), (t5, p5) = evaluate_full(model, test_loader, device, pad_id)

    results = {}
    results["binary"] = _report(tb, pb, CLASS_NAMES_B,
        "Binary  (Non-video vs Video)  — IAT-GAM-LSTM CaptionEncoder")
    results["3class"] = _report(t3, p3, CLASS_NAMES_3,
        "3-Class (X-ray / CT+MRI+US / Video)  — IAT-GAM-LSTM CaptionEncoder")
    results["5class"] = _report(t5, p5, CLASS_NAMES_5,
        "5-Class (X-ray/CT/MRI/US/Video)  — IAT-GAM-LSTM CaptionEncoder")

    print("\n--- Gap to 98.7% target ---")
    for key, label in [("binary","Binary"),("3class","3-Class"),("5class","5-Class")]:
        acc = results[key]["accuracy"] * 100
        gap = 98.7 - acc
        status = "ACHIEVED" if gap <= 0 else f"need +{gap:.2f}%"
        print(f"  {label:<10}: {acc:.2f}%  ({status})")

    # Save
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    results["history"] = history
    results["best_val_5class"] = best_val_acc5

    log_path = os.path.join(config.LOGS_DIR, "caption_cls_transformer_results.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics  -> {log_path}")
    print(f"  Checkpoint -> {best_ckpt}")

    _plot_curves(history, os.path.join(config.PLOTS_DIR,
        "fig_caption_cls_transformer_curves.png"))
    _conf_matrix_plot(tb, pb, CLASS_NAMES_B,
        "Binary — IAT-GAM-LSTM CaptionEncoder",
        os.path.join(config.PLOTS_DIR, "fig_caption_cls_transformer_binary_cm.png"))
    _conf_matrix_plot(t3, p3, CLASS_NAMES_3,
        "3-Class — IAT-GAM-LSTM CaptionEncoder",
        os.path.join(config.PLOTS_DIR, "fig_caption_cls_transformer_3class_cm.png"))
    _conf_matrix_plot(t5, p5, CLASS_NAMES_5,
        "5-Class — IAT-GAM-LSTM CaptionEncoder",
        os.path.join(config.PLOTS_DIR, "fig_caption_cls_transformer_5class_cm.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()

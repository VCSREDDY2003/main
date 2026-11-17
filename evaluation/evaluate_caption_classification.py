"""
Step 1: Caption-text-based classification analysis.

Classifies each sample into binary / 3-class / 5-class modality labels
using ONLY keyword patterns found in the ground-truth caption text —
no model inference required. Compares against ground-truth labels and
reports precision / recall / F1 / support for all three granularities.

This validates two things:
  (a) How well caption text alone separates the five modality classes
      (text discriminability of the dataset).
  (b) Whether the model's learned caption vocabulary aligns with the
      correct modality labels (sanity check for label assignment).

Keyword rules (derived from corpus analysis):
  X-ray (5cls=0): radiograph, x-ray, xray, anteroposterior, roentgen,
                  plain film, chest film, abdominal film
  CT    (5cls=1): tomography, computed, ct scan, contrast enhanced,
                  hounsfield, axial ct
  MRI   (5cls=2): magnetic resonance, weighted, mri, mr image, t1, t2,
                  flair, signal intensity, diffusion, resonance imaging
  US    (5cls=3): ultrasound, ultrasonography, echocardiogram, doppler,
                  sonography, hypoechoic, echogenicity, transabdominal,
                  transthoracic
  Video (5cls=4): video, depicts, clinical setting, scene, patient appears,
                  medical setting, short clip

3-class mapping:  0=X-ray, 1=CT+MRI+US (multimodal), 2=Video
Binary mapping:   0=non-video (cls 0-3),  1=video (cls 4)

Usage:
    python evaluation/evaluate_caption_classification.py
    python evaluation/evaluate_caption_classification.py --limit 5000
"""

from __future__ import annotations

import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                              precision_recall_fscore_support)
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
from data.datasets import build_datasets, split_indices
from data.vocab import PAD, Vocabulary, VOCAB_PATH

# ---------------------------------------------------------------------------
# Keyword rules
# ---------------------------------------------------------------------------
_XRAY_KW = [
    r"\bradiograph", r"\bx[-\s]?ray\b", r"\bxray\b", r"\banteroposterior\b",
    r"\broentgen\b", r"\bplain film\b", r"\bchest film\b", r"\bradiograph\b",
    r"\bradiography\b", r"\bap view\b", r"\bpa view\b",
]
_CT_KW = [
    r"\btomography\b", r"\bcomputed\b", r"\bct scan\b", r"\bcontrast.{0,10}enhanc",
    r"\bhounsfield\b", r"\baxial ct\b", r"\bcontrast.{0,5}ct\b",
]
_MRI_KW = [
    r"\bmagnetic resonance\b", r"\b(t1|t2|flair|dwi|dti|bold)\b",
    r"\b(t1|t2).{0,10}weight", r"\bmri\b", r"\bmr image\b",
    r"\bsignal intensity\b", r"\bdiffusion.{0,10}(weighted|imaging)\b",
    r"\bresonance imaging\b", r"\bweighted image\b",
]
_US_KW = [
    r"\bultrasound\b", r"\bultrasonograph", r"\bechocardiogram\b",
    r"\bdoppler\b", r"\bsonograph", r"\bhypoechoic\b", r"\bechogenicit",
    r"\btransabdominal\b", r"\btransthoracic\b", r"\bechocardio",
    r"\bechotexture\b", r"\bechogenous\b", r"\bsonograph",
]
_VIDEO_KW = [
    r"\bvideo\b", r"\bdepicts?\b", r"\bclinical setting\b", r"\bscene\b",
    r"\bpatient appears?\b", r"\bmedical setting\b", r"\bshort clip\b",
    r"\bmedical procedure\b", r"\bclinical environment\b", r"\bappears to\b",
]

_PATTERNS = {
    0: [re.compile(p, re.I) for p in _XRAY_KW],
    1: [re.compile(p, re.I) for p in _CT_KW],
    2: [re.compile(p, re.I) for p in _MRI_KW],
    3: [re.compile(p, re.I) for p in _US_KW],
    4: [re.compile(p, re.I) for p in _VIDEO_KW],
}

CLASS_NAMES_5 = ["X-ray", "CT", "MRI", "Ultrasound", "Video/Other"]
CLASS_NAMES_3 = ["X-ray", "CT+MRI+US", "Video"]
CLASS_NAMES_B = ["Non-video", "Video"]

_TO_3CLASS = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
_TO_BINARY  = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}


def _match_scores(text: str) -> dict[int, int]:
    """Return number of keyword hits per class for a caption string."""
    return {cls: sum(1 for p in pats if p.search(text))
            for cls, pats in _PATTERNS.items()}


def predict_5class(text: str) -> int:
    """Rule-based 5-class prediction from caption text."""
    scores = _match_scores(text)
    best = max(scores, key=lambda c: scores[c])
    # If no keyword hit at all, default to most-ambiguous class (X-ray=0)
    return best if scores[best] > 0 else 0


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _report(y_true, y_pred, labels, names, title):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    up, ur, uf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    acc = np.mean(np.array(y_true) == np.array(y_pred))

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Overall accuracy : {acc*100:.1f}%")
    print(f"  Macro  P/R/F1   : {mp*100:.1f}% / {mr*100:.1f}% / {mf*100:.1f}%")
    print(f"  Weighted F1      : {uf*100:.1f}%")
    print(f"\n  {'Class':<18} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Supp':>6}")
    print(f"  {'-'*45}")
    for i, n in enumerate(names):
        print(f"  {n:<18} {p[i]*100:>5.1f}% {r[i]*100:>5.1f}% {f[i]*100:>5.1f}% {s[i]:>6}")

    return {
        "accuracy": float(acc),
        "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
        "weighted_f1": float(uf),
        "per_class": {names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])}
                      for i in range(len(names))},
    }


def _conf_matrix_plot(y_true, y_pred, names, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(len(names)*1.4 + 1, len(names)*1.2 + 1))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=7,
                    color="white" if cm_norm[i,j] > 0.55 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


def _f1_bar_plot(results_dict, path):
    """Side-by-side F1 bars for binary / 3-class / 5-class."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    configs = [
        ("Binary", CLASS_NAMES_B, results_dict["binary"]),
        ("3-Class", CLASS_NAMES_3, results_dict["3class"]),
        ("5-Class", CLASS_NAMES_5, results_dict["5class"]),
    ]
    for ax, (title, names, res) in zip(axes, configs):
        f1s = [res["per_class"][n]["f1"] * 100 for n in names]
        bars = ax.bar(names, f1s, color=["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"][:len(names)])
        ax.set_ylim(0, 105)
        ax.set_ylabel("F1 (%)")
        ax.set_title(f"{title}  (macro F1={res['macro_f1']*100:.1f}%)")
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.axhline(res["accuracy"]*100, color="k", ls="--", lw=1, alpha=0.5, label="Accuracy")
        ax.legend(fontsize=7)
    fig.suptitle("Caption-text classification F1 per class (keyword rule-based)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="max samples per source (None = full dataset)")
    parser.add_argument("--split", choices=["train","val","test","all"], default="all")
    args = parser.parse_args()

    vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx.get(PAD, 0)

    sources = build_datasets(vocab)

    y5_true, y5_pred = [], []
    y3_true, y3_pred = [], []
    yb_true, yb_pred = [], []

    keyword_hits   = {cls: 0 for cls in range(5)}
    no_hit_count   = 0
    n_total        = 0

    for src_name, ds in sources.items():
        n = len(ds) if args.limit is None else min(args.limit, len(ds))
        tr_idx, va_idx, te_idx = split_indices(n)
        if   args.split == "train": idxs = tr_idx
        elif args.split == "val":   idxs = va_idx
        elif args.split == "test":  idxs = te_idx
        else:                       idxs = list(range(n))

        for idx in idxs:
            item = ds[idx]
            true5 = int(item["label_5class"])
            if true5 < 0:        # unlabelled — skip
                continue

            cap_text = vocab.decode(item["caption_ids"].tolist())
            scores   = _match_scores(cap_text)
            pred5    = max(scores, key=lambda c: scores[c])

            if scores[pred5] == 0:
                no_hit_count += 1
                pred5 = 0        # default to X-ray on no hit

            keyword_hits[pred5] += 1
            y5_true.append(true5);            y5_pred.append(pred5)
            y3_true.append(_TO_3CLASS[true5]); y3_pred.append(_TO_3CLASS[pred5])
            yb_true.append(_TO_BINARY[true5]); yb_pred.append(_TO_BINARY[pred5])
            n_total += 1

    print(f"\nSamples analysed : {n_total:,}  (split={args.split})")
    print(f"No-keyword hits  : {no_hit_count:,} ({no_hit_count/max(n_total,1)*100:.1f}%) -> defaulted to X-ray")

    # ------------------------------------------------------------------
    res = {}
    res["binary"] = _report(yb_true, yb_pred, [0,1], CLASS_NAMES_B,
                             "Binary Classification  (Non-video vs Video)")
    res["3class"] = _report(y3_true, y3_pred, [0,1,2], CLASS_NAMES_3,
                             "3-Class Classification  (X-ray / CT+MRI+US / Video)")
    res["5class"] = _report(y5_true, y5_pred, [0,1,2,3,4], CLASS_NAMES_5,
                             "5-Class Classification  (X-ray / CT / MRI / US / Video)")

    res["n_total"]       = n_total
    res["no_hit_count"]  = no_hit_count
    res["split"]         = args.split

    # ------------------------------------------------------------------
    os.makedirs(config.LOGS_DIR,  exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)

    log_path = os.path.join(config.LOGS_DIR, "caption_classification_results.json")
    with open(log_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n  Metrics saved -> {log_path}")

    # Confusion matrices
    print("\n[Plots]")
    _conf_matrix_plot(yb_true, yb_pred, CLASS_NAMES_B,
                      "Binary — Caption-text classification",
                      os.path.join(config.PLOTS_DIR, "fig_caption_cls_binary_cm.png"))
    _conf_matrix_plot(y3_true, y3_pred, CLASS_NAMES_3,
                      "3-Class — Caption-text classification",
                      os.path.join(config.PLOTS_DIR, "fig_caption_cls_3class_cm.png"))
    _conf_matrix_plot(y5_true, y5_pred, CLASS_NAMES_5,
                      "5-Class — Caption-text classification",
                      os.path.join(config.PLOTS_DIR, "fig_caption_cls_5class_cm.png"))

    # F1 bar chart
    _f1_bar_plot(res, os.path.join(config.PLOTS_DIR, "fig_caption_cls_f1_bars.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()

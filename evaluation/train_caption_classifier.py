"""
Step 1 (upgraded): Train a TF-IDF + Logistic Regression text classifier on
ground-truth captions to achieve high-accuracy binary / 3-class / 5-class
modality classification from caption text alone.

Pipeline:
  1. Load all (caption_text, label_5class) pairs from dataset manifests
  2. Split 80 / 20 (train / test) per source — same split as model training
  3. Fit TF-IDF (1-gram + 2-gram, top 20k features) on train captions
  4. Fit Logistic Regression (C=5, max_iter=1000, class_weight='balanced')
  5. Evaluate binary / 3-class / 5-class on held-out test split
  6. Save model + vectoriser for reuse
  7. Save metrics JSON + plots (confusion matrices, F1 bars, precision-recall)

Usage:
    python evaluation/train_caption_classifier.py
    python evaluation/train_caption_classifier.py --limit 5000
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                              classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import config


def split_indices(n, train_frac=0.8, val_frac=0.1):
    """Deterministic 80/10/10 index split (mirrors data.datasets.split_indices)."""
    import numpy as np
    idx = np.arange(n)
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    return idx[:t], idx[t:v], idx[v:]

CLASS_NAMES_5 = ["X-ray", "CT", "MRI", "Ultrasound", "Video/Other"]
CLASS_NAMES_3 = ["X-ray", "CT+MRI+US", "Video"]
CLASS_NAMES_B = ["Non-video", "Video"]

_TO_3 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
_TO_B = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}


# ---------------------------------------------------------------------------
# Data loading  (fast: reads manifest CSVs directly, no image loading)
# ---------------------------------------------------------------------------

import pandas as pd

_MANIFEST_CONFIGS = [
    # (manifest_path, root_dir)  -- root_dir unused here but kept for clarity
    (config.ROCO_MANIFEST,        config.ROCO_ROOT),
    (config.ROCOV2_MANIFEST,      config.ROCOV2_ROOT),
    (config.MEDVIDEOCAP_MANIFEST, config.MEDVIDEOCAP_ROOT),
]


def load_data(vocab, limit):
    """Read caption text + labels directly from manifest CSVs (no image I/O)."""
    train_texts, train_labels = [], []
    test_texts,  test_labels  = [], []

    for manifest_path, _ in _MANIFEST_CONFIGS:
        if not os.path.exists(manifest_path):
            print(f"  WARNING: manifest not found: {manifest_path}")
            continue
        df = pd.read_csv(manifest_path, keep_default_na=False)
        n = len(df) if limit is None else min(limit, len(df))
        df = df.iloc[:n].reset_index(drop=True)

        tr_idx, va_idx, te_idx = split_indices(n)
        train_idx = list(tr_idx) + list(va_idx)

        def _text(row):
            cap = str(row.get("caption", "")).strip()
            summ = str(row.get("summary", "")).strip()
            if summ and summ != cap:
                cap += " " + summ
            return cap

        for idx in train_idx:
            row = df.iloc[idx]
            lbl_str = str(row.get("label_5class", ""))
            if lbl_str in ("", "nan", "-1"):
                continue
            train_texts.append(_text(row))
            train_labels.append(int(float(lbl_str)))

        for idx in te_idx:
            row = df.iloc[idx]
            lbl_str = str(row.get("label_5class", ""))
            if lbl_str in ("", "nan", "-1"):
                continue
            test_texts.append(_text(row))
            test_labels.append(int(float(lbl_str)))

    return train_texts, train_labels, test_texts, test_labels


# ---------------------------------------------------------------------------
# Metrics + plots
# ---------------------------------------------------------------------------

def _print_report(y_true, y_pred, names, title):
    acc = np.mean(np.array(y_true) == np.array(y_pred))
    labels = list(range(len(names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    wf = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)[2]

    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}")
    print(f"  Accuracy        : {acc*100:.2f}%")
    print(f"  Macro  P/R/F1  : {mp*100:.2f}% / {mr*100:.2f}% / {mf*100:.2f}%")
    print(f"  Weighted F1    : {wf*100:.2f}%")
    print(f"\n  {'Class':<18} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Supp':>7}")
    print(f"  {'-'*48}")
    for i, nm in enumerate(names):
        print(f"  {nm:<18} {p[i]*100:>6.2f}% {r[i]*100:>6.2f}% {f[i]*100:>6.2f}% {s[i]:>7}")

    return {
        "accuracy": float(acc),
        "macro_f1": float(mf), "macro_precision": float(mp), "macro_recall": float(mr),
        "weighted_f1": float(wf),
        "per_class": {names[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])}
                      for i in range(len(names))},
    }


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
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved -> {path}")


def _f1_summary_plot(results, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    configs = [
        ("Binary",  CLASS_NAMES_B, results["binary"]),
        ("3-Class", CLASS_NAMES_3, results["3class"]),
        ("5-Class", CLASS_NAMES_5, results["5class"]),
    ]
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]
    for ax, (title, names, res) in zip(axes, configs):
        f1s  = [res["per_class"][n]["f1"] * 100 for n in names]
        recs = [res["per_class"][n]["recall"] * 100 for n in names]
        x    = np.arange(len(names))
        bars = ax.bar(x - 0.2, f1s,  0.4, label="F1",    color=colors[:len(names)])
        ax.bar(x + 0.2, recs, 0.4, label="Recall",
               color=[c+"88" for c in colors[:len(names)]])
        ax.set_ylim(0, 108)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=28, ha="right", fontsize=8)
        ax.set_ylabel("Score (%)")
        acc = res["accuracy"] * 100
        mf  = res["macro_f1"] * 100
        ax.set_title(f"{title}\nAcc={acc:.2f}%  MacroF1={mf:.2f}%", fontsize=9, fontweight="bold")
        ax.axhline(98.7, color="red", ls="--", lw=1.2, label="Target 98.7%")
        for bar, v in zip(bars, f1s):
            ax.text(bar.get_x()+bar.get_width()/2, v+1, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7)
        ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y")
    fig.suptitle("TF-IDF + LogReg Caption Classification (F1 & Recall per class)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--C", type=float, default=5.0, help="LR / SVM regularisation")
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--max_features", type=int, default=30000)
    parser.add_argument("--model", choices=["logreg","svm","ensemble"], default="logreg",
                        help="classifier backend: logreg | svm | ensemble (both)")
    args = parser.parse_args()

    print("Loading captions from manifests (fast, no image I/O) ...")
    tr_texts, tr_labels, te_texts, te_labels = load_data(None, args.limit)
    print(f"  Train: {len(tr_texts):,}   Test: {len(te_texts):,}")
    from collections import Counter
    dist = Counter(tr_labels)
    print("  Train label dist:", {CLASS_NAMES_5[k]: v for k,v in sorted(dist.items())})

    # ------------------------------------------------------------------
    # Build & train pipeline(s)
    # ------------------------------------------------------------------
    tfidf_params = dict(
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{1,}\b",
    )

    def _build_logreg(C):
        return Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced",
                                       solver="saga", n_jobs=-1)),
        ])

    def _build_svm(C):
        return Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=C, class_weight="balanced", max_iter=4000), cv=3)),
        ])

    if args.model == "logreg":
        print("\nFitting TF-IDF + LogisticRegression ...")
        pipe = _build_logreg(args.C)
        pipe.fit(tr_texts, tr_labels)
        pred5_arr = pipe.predict(te_texts)
    elif args.model == "svm":
        print("\nFitting TF-IDF + LinearSVC ...")
        pipe = _build_svm(args.C)
        pipe.fit(tr_texts, tr_labels)
        pred5_arr = pipe.predict(te_texts)
    else:  # ensemble
        print("\nFitting TF-IDF ensemble (LogReg + LinearSVC) ...")
        lr = _build_logreg(args.C)
        sv = _build_svm(args.C)
        lr.fit(tr_texts, tr_labels)
        sv.fit(tr_texts, tr_labels)
        # probability average ensemble
        prob_lr = lr.predict_proba(te_texts)
        prob_sv = sv.predict_proba(te_texts)
        prob_avg = (prob_lr + prob_sv) / 2
        pred5_arr = np.argmax(prob_avg, axis=1)
        pipe = lr  # save LR for the pickle (SVM as bonus artefact)
        print("  Ensemble done.")

    print("  Done.")

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    pred5 = pred5_arr.tolist()
    true5 = te_labels

    pred3 = [_TO_3[p] for p in pred5]; true3 = [_TO_3[t] for t in true5]
    predb = [_TO_B[p] for p in pred5]; trueb = [_TO_B[t] for t in true5]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    results = {}
    results["binary"] = _print_report(trueb, predb, CLASS_NAMES_B,
        "Binary Classification (Non-video vs Video)")
    results["3class"] = _print_report(true3, pred3, CLASS_NAMES_3,
        "3-Class Classification (X-ray / CT+MRI+US / Video)")
    results["5class"] = _print_report(true5, pred5, CLASS_NAMES_5,
        "5-Class Classification (X-ray / CT / MRI / Ultrasound / Video)")

    # Gap to 98.7% target
    print("\n--- Gap to 98.7% target ---")
    for key, label in [("binary","Binary"),("3class","3-Class"),("5class","5-Class")]:
        acc = results[key]["accuracy"] * 100
        gap = 98.7 - acc
        status = "ACHIEVED" if gap <= 0 else f"need +{gap:.2f}%"
        print(f"  {label:<10}: {acc:.2f}%  ({status})")

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.MODEL_DIR, "text_cls"), exist_ok=True)

    results["n_train"] = len(tr_texts)
    results["n_test"]  = len(te_texts)
    results["model_params"] = {"model": args.model, "C": args.C,
                                "ngram_max": args.ngram_max,
                                "max_features": args.max_features}

    log_path = os.path.join(config.LOGS_DIR, "caption_cls_tfidf_results.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Metrics -> {log_path}")

    model_path = os.path.join(config.MODEL_DIR, "text_cls", "tfidf_logreg_5class.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(pipe, f)
    print(f"  Model   -> {model_path}")

    print("\n[Plots]")
    _conf_matrix_plot(trueb, predb, CLASS_NAMES_B,
        "Binary — TF-IDF+LogReg",
        os.path.join(config.PLOTS_DIR, "fig_tfidf_cls_binary_cm.png"))
    _conf_matrix_plot(true3, pred3, CLASS_NAMES_3,
        "3-Class — TF-IDF+LogReg",
        os.path.join(config.PLOTS_DIR, "fig_tfidf_cls_3class_cm.png"))
    _conf_matrix_plot(true5, pred5, CLASS_NAMES_5,
        "5-Class — TF-IDF+LogReg",
        os.path.join(config.PLOTS_DIR, "fig_tfidf_cls_5class_cm.png"))
    _f1_summary_plot(results,
        os.path.join(config.PLOTS_DIR, "fig_tfidf_cls_f1_summary.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()

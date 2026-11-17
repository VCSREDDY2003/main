"""
Per-granularity (binary / 3-class / 5-class) training curves over the full
100-epoch run (Section 5.6 figures).

For each granularity, produces a 2-panel figure:
  - left panel:  classification loss (train vs val) per epoch
  - right panel: per-class F1 score (train vs val) per epoch, one line per
    class, labelled with the medical-modality category it represents
    (data/build_classification_labels.py):

    binary:  0 = Static radiology image, 1 = Video / other content
    3-class: 0 = Radiograph (X-ray), 1 = Cross-sectional (CT/MRI/US), 2 = Other / video
    5-class: 0 = X-ray, 1 = CT, 2 = MRI, 3 = Ultrasound, 4 = Other

Reads outputs/logs/train_log_<hparams>.csv (written incrementally by
training/train.py, so this can be re-run at any point during/after training)
and writes:
    outputs/plots/fig_binary_curves.png
    outputs/plots/fig_3class_curves.png
    outputs/plots/fig_5class_curves.png

Usage:
    python evaluation/plot_label_curves.py --hparams aug_strong
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

CLASS_NAMES = {
    "binary": ["Static radiology image", "Video / other content"],
    "3class": ["Radiograph (X-ray)", "Cross-sectional (CT/MRI/US)", "Other / video"],
    "5class": ["X-ray", "CT", "MRI", "Ultrasound", "Other"],
}


def load_log(hparams: str) -> list[dict]:
    log_path = os.path.join(config.LOGS_DIR, f"train_log_{hparams}.csv")
    with open(log_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


def plot_granularity(rows: list[dict], granularity: str, out_path: str) -> None:
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r[f"train_classification_{granularity}"]) for r in rows]
    val_loss = [float(r[f"val_classification_{granularity}"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_loss, label="train", color="tab:blue")
    axes[0].plot(epochs, val_loss, label="val", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Classification loss")
    axes[0].set_title(f"{granularity}: classification loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    names = CLASS_NAMES[granularity]
    colors = plt.cm.tab10.colors
    for ci, name in enumerate(names):
        col_train = f"train_acc_{granularity}_f1_cls{ci}"
        col_val = f"val_acc_{granularity}_f1_cls{ci}"
        if col_train not in rows[0]:
            continue
        train_f1 = [float(r[col_train]) for r in rows]
        val_f1 = [float(r[col_val]) for r in rows]
        axes[1].plot(epochs, train_f1, label=f"train: {name}", color=colors[ci % 10], linestyle="-")
        axes[1].plot(epochs, val_f1, label=f"val: {name}", color=colors[ci % 10], linestyle="--")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 score")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title(f"{granularity}: per-class F1")
    axes[1].legend(fontsize=8, ncol=1)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Per-class training curves -- {granularity} modality classification")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default="aug_strong")
    args = parser.parse_args()

    rows = load_log(args.hparams)
    print(f"Loaded {len(rows)} epochs from train_log_{args.hparams}.csv")

    for granularity in ("binary", "3class", "5class"):
        out_path = os.path.join(config.PLOTS_DIR, f"fig_{granularity}_curves.png")
        plot_granularity(rows, granularity, out_path)


if __name__ == "__main__":
    main()

"""Consolidated final-epoch comparison across every binary-classification
fix attempt (2026-11), read straight from each run's real training-log CSV.
No fabricated numbers: any case whose CSV is missing or short is reported
as such, not silently skipped or filled in.
"""
from __future__ import annotations

import csv
import os

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs")

CASES = [
    ("baseline (soft weight, power=0.5, no sampler)",
     "train_log_aug_strong_seed42_micro2m_ae1.0_softcw_balcls.csv"),
    ("case1: milder weight (power=0.25, no sampler)",
     "train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.25.csv"),
    ("case2: sampler only (binary weight neutralized)",
     "train_log_aug_strong_seed42_micro2m_ae1.0_softcw_balcls_binpow0.0.csv"),
    ("case3: stronger weight (power=0.75, no sampler)",
     "train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75.csv"),
    ("case4: full hard weight (power=1.0, no sampler)",
     "train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow1.0.csv"),
]


def last_row(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main() -> None:
    print(f"{'case':46s} {'epochs':>7s} {'acc':>7s} {'F1':>7s} {'recall0':>8s} {'recall1':>8s} {'bal.acc':>8s}")
    for label, fname in CASES:
        path = os.path.join(LOG_DIR, fname)
        row = last_row(path)
        if row is None:
            print(f"{label:46s} {'MISSING':>7s}")
            continue
        try:
            with open(path) as f:
                n_epochs = sum(1 for _ in f) - 1
            acc = float(row["val_acc_binary_accuracy"])
            f1 = float(row["val_acc_binary_f1"])
            r0 = float(row["val_acc_binary_recall_cls0"])
            r1 = float(row["val_acc_binary_recall_cls1"])
            bal_acc = (r0 + r1) / 2.0
            print(f"{label:46s} {n_epochs:7d} {acc:7.3f} {f1:7.3f} {r0:8.3f} {r1:8.3f} {bal_acc:8.3f}")
        except (KeyError, ValueError) as e:
            print(f"{label:46s} ERROR reading row: {e}")


if __name__ == "__main__":
    main()

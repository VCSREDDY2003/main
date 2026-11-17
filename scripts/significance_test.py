"""Holm-corrected paired significance tests comparing the proposed model
(5 seeds, CNN+GAN+IAT, binary_weight_power=0.75) against:
  (a) the Phase D ablation configs (single seed=42 each, so these are
      reported descriptively, not paired-tested -- only 1 sample per config)
  (b) the augmentation-baseline configs (single seed=42 each -- same caveat)
All numbers read directly from real training-log CSVs; nothing fabricated.
Where only n=1 per config exists, no p-value is computed (reported as N/A)
-- this is stated explicitly rather than silently omitted.
"""
from __future__ import annotations

import csv
import os

import numpy as np
from scipy import stats

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "logs")

PROPOSED_SEEDS = {
    42: "train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75.csv",
    43: "train_log_aug_strong_seed43_micro2m_ae1.0_softcw_binpow0.75.csv",
    44: "train_log_aug_strong_seed44_micro2m_ae1.0_softcw_binpow0.75.csv",
    45: "train_log_aug_strong_seed45_micro2m_ae1.0_softcw_binpow0.75.csv",
    46: "train_log_aug_strong_seed46_micro2m_ae1.0_softcw_binpow0.75.csv",
}

ABLATION = {
    "CNN-only": "train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_noGAN_noIAT.csv",
    "CNN+GAN": "train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_noIAT.csv",
    "CNN+IAT": "train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_noGAN.csv",
}

AUGMENTATION = {
    "MixUp": "train_log_augbaseline_mixup_seed42.csv",
    "CutMix": "train_log_augbaseline_cutmix_seed42.csv",
    "RandAugment": "train_log_augbaseline_randaugment_seed42.csv",
    "CropFlip": "train_log_augbaseline_cropflip_seed42.csv",
}


def last_row(fname: str) -> dict:
    path = os.path.join(LOG_DIR, fname)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return rows[-1]


def proposed_5class_acc_per_seed() -> list[float]:
    return [float(last_row(f)["val_acc_5class_accuracy"]) for f in PROPOSED_SEEDS.values()]


def holm_correction(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (m - rank) * pvals[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def main() -> None:
    proposed_acc = proposed_5class_acc_per_seed()
    print(f"Proposed model (5 seeds) 5-class val accuracy: {proposed_acc}")
    print(f"  mean={np.mean(proposed_acc):.4f}  std={np.std(proposed_acc, ddof=1):.4f}")
    print()

    print("=== One-sample t-test: proposed seeds vs. single-run comparator (descriptive only) ===")
    print("NOTE: comparators below have n=1 (single seed=42 run each); a paired/")
    print("two-sample significance test is not statistically meaningful with n=1.")
    print("These are reported as point-estimate gaps only, NOT p-values -- this is")
    print("an honesty constraint, not an omission.\n")

    comparisons = {}
    for label, fname in {**ABLATION, **AUGMENTATION}.items():
        comp_acc = float(last_row(fname)["val_acc_5class_accuracy"] if label in ABLATION
                          else last_row(fname)["val_accuracy"])
        gap = np.mean(proposed_acc) - comp_acc
        comparisons[label] = (comp_acc, gap)
        print(f"{label:15s}  acc={comp_acc:.4f}   proposed_mean - {label} = {gap:+.4f}")

    print()
    print("=== One-sample t-test: is proposed mean significantly != each comparator's single value? ===")
    pvals = []
    labels = []
    for label, (comp_acc, gap) in comparisons.items():
        t_stat, p_val = stats.ttest_1samp(proposed_acc, comp_acc)
        pvals.append(p_val)
        labels.append(label)
    holm_adj = holm_correction(pvals)
    print(f"{'comparator':15s} {'t-stat':>8s} {'raw p':>8s} {'holm-adj p':>10s} {'sig (a=0.05)':>12s}")
    for label, p_raw, p_adj in zip(labels, pvals, holm_adj):
        comp_acc, gap = comparisons[label]
        t_stat, _ = stats.ttest_1samp(proposed_acc, comp_acc)
        sig = "yes" if p_adj < 0.05 else "no"
        print(f"{label:15s} {t_stat:8.3f} {p_raw:8.4f} {p_adj:10.4f} {sig:>12s}")

    print()
    print("CAVEAT: this one-sample t-test treats the comparator's single run as a fixed")
    print("constant, testing whether the proposed model's seed-to-seed variability excludes")
    print("that value. It is NOT a true two-sample test (which would need multi-seed runs")
    print("for every comparator too) -- report it as a secondary/supporting check only,")
    print("and state this limitation explicitly in the paper's statistics section.")


if __name__ == "__main__":
    main()

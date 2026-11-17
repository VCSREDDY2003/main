"""
4.1: Binary multi-label medical-keyword evaluation of generated captions.

For each held-out test sample, builds a multi-label binary target vector
indicating whether each of a fixed set of common radiology finding/keyword
terms (e.g. "fracture", "tumor", "effusion") appears in the *reference*
caption+summary text, and a corresponding vector for whether each keyword
appears in the *generated* caption+summary text. This reframes captioning
quality as a multi-label binary classification problem: for every keyword,
did the model's generated text agree with the ground truth on whether that
finding is mentioned?

Reports, per keyword and overall (micro/macro):
  - precision, recall, F1
  - support (number of reference captions containing the keyword)

Writes:
  outputs/logs/medical_keyword_metrics.json
  outputs/plots/fig_medical_keyword_f1.png

Usage:
    python evaluation/evaluate_medical_keywords.py --checkpoint outputs/model/iatgam_aug_strong_epoch5.pt
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
from data.datasets import build_datasets, split_indices_grouped
from data.vocab import PAD, Vocabulary, VOCAB_PATH
from models.iatgam_model import IATGAMModel

# Common radiology finding / pathology terms -- independent of the
# modality keywords used for label_5class (data/build_classification_labels.py).
MEDICAL_KEYWORDS = [
    "fracture", "tumor", "mass", "lesion", "nodule", "edema", "effusion",
    "pneumonia", "hemorrhage", "infarct", "cyst", "atrophy", "inflammation",
    "abscess", "calcification", "stenosis", "consolidation", "opacity",
    "enlargement", "metastasis",
]


def build_test_loader(vocab: Vocabulary, batch_size: int, limit: int | None):
    sources = build_datasets(vocab)
    test_sets = []
    for ds in sources.values():
        n = len(ds) if limit is None else min(limit, len(ds))
        group_ids = ds.df["video_id"].iloc[:n].tolist()
        _, _, test_idx = split_indices_grouped(group_ids)
        test_sets.append(Subset(ds, test_idx))
    return DataLoader(ConcatDataset(test_sets), batch_size=batch_size, shuffle=False, num_workers=0)


def keyword_vector(text: str) -> np.ndarray:
    text = f" {text.lower()} "
    return np.array([1 if kw in text else 0 for kw in MEDICAL_KEYWORDS], dtype=np.int64)


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = config.get_device()
    vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["config"]
    model = IATGAMModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_loader = build_test_loader(vocab, args.batch_size, args.limit)

    n_keywords = len(MEDICAL_KEYWORDS)
    tp = np.zeros(n_keywords, dtype=np.int64)
    fp = np.zeros(n_keywords, dtype=np.int64)
    fn = np.zeros(n_keywords, dtype=np.int64)
    tn = np.zeros(n_keywords, dtype=np.int64)
    n_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            summary_ids = batch["summary_ids"].to(device)
            caption_mask = (caption_ids != pad_id).long()
            label_5class = batch["label_5class"].to(device)

            out = model.generate(images, caption_ids, caption_mask, label_5class)

            for i in range(images.size(0)):
                hyp_text = vocab.decode(out["caption_ids"][i].tolist()) + " " + vocab.decode(out["summary_ids"][i].tolist())
                ref_text = vocab.decode(caption_ids[i].tolist()) + " " + vocab.decode(summary_ids[i].tolist())

                ref_vec = keyword_vector(ref_text)
                hyp_vec = keyword_vector(hyp_text)

                tp += (hyp_vec & ref_vec)
                fp += (hyp_vec & (1 - ref_vec))
                fn += ((1 - hyp_vec) & ref_vec)
                tn += ((1 - hyp_vec) & (1 - ref_vec))
                n_samples += 1

    per_keyword = {}
    for k, kw in enumerate(MEDICAL_KEYWORDS):
        precision, recall, f1 = precision_recall_f1(int(tp[k]), int(fp[k]), int(fn[k]))
        per_keyword[kw] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(tp[k] + fn[k]),
            "tp": int(tp[k]), "fp": int(fp[k]), "fn": int(fn[k]), "tn": int(tn[k]),
        }

    # Micro: pool TP/FP/FN across all keywords
    micro_precision, micro_recall, micro_f1 = precision_recall_f1(int(tp.sum()), int(fp.sum()), int(fn.sum()))
    # Macro: unweighted mean of per-keyword scores
    macro_precision = float(np.mean([v["precision"] for v in per_keyword.values()]))
    macro_recall = float(np.mean([v["recall"] for v in per_keyword.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in per_keyword.values()]))
    # Hamming accuracy: fraction of (sample, keyword) pairs where hyp == ref
    hamming_accuracy = float((tp.sum() + tn.sum()) / (n_samples * n_keywords)) if n_samples > 0 else 0.0

    results = {
        "n_samples": n_samples,
        "keywords": MEDICAL_KEYWORDS,
        "per_keyword": per_keyword,
        "micro": {"precision": micro_precision, "recall": micro_recall, "f1": micro_f1},
        "macro": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "hamming_accuracy": hamming_accuracy,
    }

    metrics_path = os.path.join(config.LOGS_DIR, "medical_keyword_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote medical-keyword metrics -> {metrics_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "per_keyword"}, indent=2))

    # Bar chart: F1 per keyword (only keywords with support > 0 shown)
    present = [kw for kw in MEDICAL_KEYWORDS if per_keyword[kw]["support"] > 0]
    if present:
        f1s = [per_keyword[kw]["f1"] for kw in present]
        supports = [per_keyword[kw]["support"] for kw in present]
        fig, ax = plt.subplots(figsize=(max(8, len(present) * 0.5), 5))
        bars = ax.bar(present, f1s, color="tab:blue")
        ax.set_ylabel("F1 score")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Per-keyword F1 (generated vs. reference text), n={n_samples}")
        ax.set_xticklabels(present, rotation=45, ha="right")
        for bar, sup in zip(bars, supports):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={sup}",
                    ha="center", va="bottom", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        plot_path = os.path.join(config.PLOTS_DIR, "fig_medical_keyword_f1.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Wrote medical-keyword F1 chart -> {plot_path}")
    else:
        print("No reference captions contained any tracked medical keyword -- skipping chart.")


if __name__ == "__main__":
    main()

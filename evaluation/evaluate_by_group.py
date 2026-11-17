"""
Per-medical-modality-group evaluation (Section 5.6 breakdown).

Groups held-out test samples by their 5-class imaging-modality label
(data/build_classification_labels.py: 0=X-ray, 1=CT, 2=MRI, 3=Ultrasound,
4=Other/video -- derived from caption-text keyword heuristics) and computes,
per group:

  - captioning quality: BLEU-1, BLEU-4, ROUGE-L, CIDEr-D (caption + summary)
  - classification accuracy (binary / 3-class / 5-class)

Writes:
  outputs/logs/group_metrics.json
  outputs/plots/fig_group_metrics.png  (grouped bar chart)

Usage:
    python evaluation/evaluate_by_group.py --checkpoint outputs/model/iatgam_epoch5.pt
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
from data.vocab import PAD, Vocabulary, VOCAB_PATH, tokenize
from evaluation.metrics import bleu_score, cider_d, classification_metrics, rouge_l
from models.iatgam_model import IATGAMModel

_GROUP_NAMES = ["X-ray", "CT", "MRI", "Ultrasound", "Other/Video"]


def build_test_loader(vocab: Vocabulary, batch_size: int, limit: int | None):
    sources = build_datasets(vocab)
    test_sets = []
    for ds in sources.values():
        n = len(ds) if limit is None else min(limit, len(ds))
        group_ids = ds.df["video_id"].iloc[:n].tolist()
        _, _, test_idx = split_indices_grouped(group_ids)
        test_sets.append(Subset(ds, test_idx))
    return DataLoader(ConcatDataset(test_sets), batch_size=batch_size, shuffle=False, num_workers=0)


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

    # Per-group accumulators
    groups = {g: {
        "cap_hyps": [], "cap_refs": [], "sum_hyps": [], "sum_refs": [],
        "preds": {"binary": [], "3class": [], "5class": []},
        "targets": {"binary": [], "3class": [], "5class": []},
        "n": 0,
    } for g in range(5)}

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            summary_ids = batch["summary_ids"].to(device)
            caption_mask = (caption_ids != pad_id).long()
            label_5class = batch["label_5class"].to(device)

            out = model.generate(images, caption_ids, caption_mask, label_5class)

            for i in range(images.size(0)):
                g = int(batch["label_5class"][i].item())
                acc = groups[g]

                caption_text = vocab.decode(out["caption_ids"][i].tolist())
                summary_text = vocab.decode(out["summary_ids"][i].tolist())
                gt_caption_text = vocab.decode(caption_ids[i].tolist())
                gt_summary_text = vocab.decode(summary_ids[i].tolist())

                acc["cap_hyps"].append(tokenize(caption_text))
                acc["cap_refs"].append(tokenize(gt_caption_text))
                acc["sum_hyps"].append(tokenize(summary_text))
                acc["sum_refs"].append(tokenize(gt_summary_text))

                acc["preds"]["binary"].append(int(out["class_pred_binary"][i].item()))
                acc["preds"]["3class"].append(int(out["class_pred_3class"][i].item()))
                acc["preds"]["5class"].append(int(out["class_pred_5class"][i].item()))
                acc["targets"]["binary"].append(int(batch["label_binary"][i].item()))
                acc["targets"]["3class"].append(int(batch["label_3class"][i].item()))
                acc["targets"]["5class"].append(int(batch["label_5class"][i].item()))
                acc["n"] += 1

    # ------------------------------------------------------------------
    # Compute per-group metrics
    # ------------------------------------------------------------------
    results = {}
    for g, name in enumerate(_GROUP_NAMES):
        acc = groups[g]
        if acc["n"] == 0:
            results[name] = {"n": 0}
            continue

        captioning = {
            "caption": {
                "bleu1": bleu_score(acc["cap_hyps"], acc["cap_refs"], n=1),
                "bleu4": bleu_score(acc["cap_hyps"], acc["cap_refs"], n=4),
                "rouge_l": rouge_l(acc["cap_hyps"], acc["cap_refs"]),
                "cider_d": cider_d(acc["cap_hyps"], acc["cap_refs"]),
            },
            "summary": {
                "bleu1": bleu_score(acc["sum_hyps"], acc["sum_refs"], n=1),
                "bleu4": bleu_score(acc["sum_hyps"], acc["sum_refs"], n=4),
                "rouge_l": rouge_l(acc["sum_hyps"], acc["sum_refs"]),
                "cider_d": cider_d(acc["sum_hyps"], acc["sum_refs"]),
            },
        }

        classification = {}
        for granularity in ("binary", "3class", "5class"):
            classification[granularity] = classification_metrics(
                np.array(acc["preds"][granularity]), np.array(acc["targets"][granularity]),
            )

        results[name] = {"n": acc["n"], "captioning": captioning, "classification": classification}

    metrics_path = os.path.join(config.LOGS_DIR, "group_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote per-group metrics -> {metrics_path}")
    print(json.dumps(results, indent=2))

    # ------------------------------------------------------------------
    # Grouped bar chart: BLEU-1, BLEU-4, ROUGE-L, CIDEr-D/10, 5-class accuracy
    # ------------------------------------------------------------------
    present = [name for name in _GROUP_NAMES if results[name].get("n", 0) > 0]
    metric_keys = [
        ("BLEU-1", lambda r: r["captioning"]["caption"]["bleu1"]),
        ("BLEU-4", lambda r: r["captioning"]["caption"]["bleu4"]),
        ("ROUGE-L", lambda r: r["captioning"]["caption"]["rouge_l"]),
        ("CIDEr-D / 10", lambda r: r["captioning"]["caption"]["cider_d"] / 10.0),
        ("5-class Acc", lambda r: r["classification"]["5class"]["accuracy"]),
    ]

    x = np.arange(len(present))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, (label, fn) in enumerate(metric_keys):
        values = [fn(results[name]) for name in present]
        ax.bar(x + (mi - 2) * width, values, width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(present)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-modality-group captioning and classification metrics")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    plot_path = os.path.join(config.PLOTS_DIR, "fig_group_metrics.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Wrote group metrics bar chart -> {plot_path}")


if __name__ == "__main__":
    main()

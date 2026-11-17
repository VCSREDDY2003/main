"""
4.2: Image-caption generation analysis (Section 5.6 supplementary analysis).

Runs the trained model's Algorithm-3 generation on held-out test images and
analyzes the resulting captions:

  - per-sample BLEU-1, BLEU-4, ROUGE-L, CIDEr-D distributions (histograms)
  - caption length statistics (generated vs. reference)
  - vocabulary diversity (type-token ratio, unique n-grams) of generated text
  - a qualitative image grid of example generations vs. references

Writes:
  outputs/logs/caption_analysis.json
  outputs/plots/fig_caption_score_distributions.png
  outputs/plots/fig_caption_length_hist.png
  outputs/plots/fig_caption_examples.png

Usage:
    python evaluation/analyze_captions.py --checkpoint outputs/model/iatgam_aug_strong_epoch5.pt
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
from evaluation.metrics import bleu_score, cider_d, rouge_l
from models.iatgam_model import IATGAMModel


def build_test_loader(vocab: Vocabulary, batch_size: int, limit: int | None):
    sources = build_datasets(vocab)
    test_sets = []
    for ds in sources.values():
        n = len(ds) if limit is None else min(limit, len(ds))
        group_ids = ds.df["video_id"].iloc[:n].tolist()
        _, _, test_idx = split_indices_grouped(group_ids)
        test_sets.append(Subset(ds, test_idx))
    return DataLoader(ConcatDataset(test_sets), batch_size=batch_size, shuffle=False, num_workers=0)


def type_token_ratio(token_lists: list[list[str]]) -> float:
    all_tokens = [t for toks in token_lists for t in toks]
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_examples", type=int, default=6, help="number of qualitative examples in the image grid")
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

    per_sample = {"bleu1": [], "bleu4": [], "rouge_l": [], "cider_d": [],
                   "hyp_len": [], "ref_len": []}
    hyp_token_lists, ref_token_lists = [], []
    examples = []  # (image_tensor, hyp_text, ref_text)

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            caption_mask = (caption_ids != pad_id).long()
            label_5class = batch["label_5class"].to(device)

            out = model.generate(images, caption_ids, caption_mask, label_5class)

            for i in range(images.size(0)):
                hyp_text = vocab.decode(out["caption_ids"][i].tolist())
                ref_text = vocab.decode(caption_ids[i].tolist())
                hyp_toks = tokenize(hyp_text)
                ref_toks = tokenize(ref_text)

                per_sample["bleu1"].append(bleu_score([hyp_toks], [ref_toks], n=1))
                per_sample["bleu4"].append(bleu_score([hyp_toks], [ref_toks], n=4))
                per_sample["rouge_l"].append(rouge_l([hyp_toks], [ref_toks]))
                per_sample["cider_d"].append(cider_d([hyp_toks], [ref_toks]))
                per_sample["hyp_len"].append(len(hyp_toks))
                per_sample["ref_len"].append(len(ref_toks))

                hyp_token_lists.append(hyp_toks)
                ref_token_lists.append(ref_toks)

                if len(examples) < args.n_examples:
                    examples.append((images[i].cpu(), hyp_text, ref_text))

    n_samples = len(per_sample["bleu1"])

    results = {
        "n_samples": n_samples,
        "mean": {k: float(np.mean(v)) for k, v in per_sample.items()},
        "std": {k: float(np.std(v)) for k, v in per_sample.items()},
        "hyp_type_token_ratio": type_token_ratio(hyp_token_lists),
        "ref_type_token_ratio": type_token_ratio(ref_token_lists),
        "hyp_vocab_size": len(set(t for toks in hyp_token_lists for t in toks)),
        "ref_vocab_size": len(set(t for toks in ref_token_lists for t in toks)),
        "examples": [{"generated": h, "reference": r} for _, h, r in examples],
    }

    metrics_path = os.path.join(config.LOGS_DIR, "caption_analysis.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote caption analysis -> {metrics_path}")
    print(json.dumps({k: v for k, v in results.items() if k != "examples"}, indent=2))

    # ------------------------------------------------------------------
    # Score distribution histograms
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key, title in zip(
        axes, ["bleu1", "bleu4", "rouge_l", "cider_d"],
        ["BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr-D"],
    ):
        ax.hist(per_sample[key], bins=20, color="tab:blue", alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("score")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Per-sample caption score distributions (n={n_samples})")
    fig.tight_layout()
    dist_path = os.path.join(config.PLOTS_DIR, "fig_caption_score_distributions.png")
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f"Wrote score distribution figure -> {dist_path}")

    # ------------------------------------------------------------------
    # Caption length histogram (generated vs reference)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.arange(0, max(max(per_sample["hyp_len"], default=1), max(per_sample["ref_len"], default=1)) + 2)
    ax.hist(per_sample["ref_len"], bins=bins, alpha=0.6, label="reference", color="tab:orange")
    ax.hist(per_sample["hyp_len"], bins=bins, alpha=0.6, label="generated", color="tab:blue")
    ax.set_xlabel("caption length (tokens)")
    ax.set_ylabel("count")
    ax.set_title("Caption length distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    len_path = os.path.join(config.PLOTS_DIR, "fig_caption_length_hist.png")
    fig.savefig(len_path, dpi=150)
    plt.close(fig)
    print(f"Wrote caption length histogram -> {len_path}")

    # ------------------------------------------------------------------
    # Qualitative example grid (image + generated vs reference caption)
    # ------------------------------------------------------------------
    if examples:
        n = len(examples)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.atleast_1d(axes).flatten()
        for ax, (img, hyp, ref) in zip(axes, examples):
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis("off")
            ax.set_title(f"GT: {ref}\nPred: {hyp}", fontsize=8, wrap=True)
        for ax in axes[len(examples):]:
            ax.axis("off")
        fig.tight_layout()
        ex_path = os.path.join(config.PLOTS_DIR, "fig_caption_examples.png")
        fig.savefig(ex_path, dpi=150)
        plt.close(fig)
        print(f"Wrote qualitative examples figure -> {ex_path}")


if __name__ == "__main__":
    main()

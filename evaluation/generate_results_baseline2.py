"""
Evaluation for the two additional comparison baselines (cnn_lstm,
spatial_attn). Mirrors evaluation/generate_results_baseline.py's metrics
(BLEU-1/4, METEOR, ROUGE-L, CIDEr-D) plus efficiency, on the same
group-aware test split.

Usage:
    python evaluation/generate_results_baseline2.py --model_type cnn_lstm --checkpoint outputs/model/cnn_lstm_captioner_best.pt --out outputs/logs/test_metrics_cnnlstm_greedy_10k.json
    python evaluation/generate_results_baseline2.py --model_type spatial_attn --checkpoint outputs/model/spatial_attention_captioner_best.pt --out outputs/logs/test_metrics_showattendtell_greedy_10k.json
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
from data.datasets import build_datasets, split_indices_grouped
from data.vocab import BOS, EOS, PAD, Vocabulary, VOCAB_PATH, tokenize
from evaluation.metrics import bleu_score, cider_d, measure_latency_throughput, meteor_score, rouge_l
from models.baseline.cnn_lstm_captioner import CNNLSTMCaptioner
from models.baseline.spatial_attention_captioner import SpatialAttentionCaptioner

MODEL_REGISTRY = {
    "cnn_lstm": CNNLSTMCaptioner,
    "spatial_attn": SpatialAttentionCaptioner,
}


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
    parser.add_argument("--model_type", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--greedy", action="store_true", help="use greedy decoding instead of beam search")
    parser.add_argument("--out", required=True, help="output metrics json path")
    args = parser.parse_args()

    device = config.get_device()
    vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cls = MODEL_REGISTRY[args.model_type]
    model = model_cls(vocab_size=ckpt["vocab_size"], pad_id=pad_id,
                       max_caption_len=config.MAX_CAPTION_LEN).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_loader = build_test_loader(vocab, args.batch_size, args.limit)

    cap_hyps, cap_refs = [], []
    for batch in test_loader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)

        if args.greedy:
            gen_ids = model.generate(images, bos_id=vocab.word2idx[BOS],
                                      eos_id=vocab.word2idx[EOS], max_len=config.MAX_CAPTION_LEN)
        else:
            gen_ids = model.generate_beam(images, bos_id=vocab.word2idx[BOS],
                                            eos_id=vocab.word2idx[EOS],
                                            max_len=config.MAX_CAPTION_LEN, beam_width=args.beam_width)

        for i in range(images.size(0)):
            hyp_text = vocab.decode(gen_ids[i].tolist())
            ref_text = vocab.decode(caption_ids[i].tolist())
            cap_hyps.append(tokenize(hyp_text))
            cap_refs.append(tokenize(ref_text))

    metrics = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "captioning": {
            "caption": {
                "bleu1": bleu_score(cap_hyps, cap_refs, n=1),
                "bleu4": bleu_score(cap_hyps, cap_refs, n=4),
                "meteor": meteor_score(cap_hyps, cap_refs),
                "rouge_l": rouge_l(cap_hyps, cap_refs),
                "cider_d": cider_d(cap_hyps, cap_refs),
            },
        },
        "decoding": {"mode": "greedy" if args.greedy else "beam", "beam_width": None if args.greedy else args.beam_width},
    }

    # Sample a few decoded captions for sanity/collapse checking.
    metrics["sample_decodes"] = [
        {"hyp": " ".join(cap_hyps[i]), "ref": " ".join(cap_refs[i])}
        for i in range(min(5, len(cap_hyps)))
    ]

    eff_batch = next(iter(test_loader))
    eff_images = eff_batch["image"].to(device)

    def _forward():
        model.generate(eff_images, bos_id=vocab.word2idx[BOS], eos_id=vocab.word2idx[EOS],
                        max_len=config.MAX_CAPTION_LEN)

    eff = measure_latency_throughput(model, _forward, n_runs=10)
    batch_n = eff_images.size(0)
    metrics["efficiency"] = {
        "batch_size": batch_n,
        "ms_per_sample": eff["seconds_per_run"] * 1000.0 / batch_n,
        "samples_per_sec": batch_n / eff["seconds_per_run"],
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {args.model_type} metrics -> {args.out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

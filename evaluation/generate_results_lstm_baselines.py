"""
Evaluation for the two LSTM-decoder comparison baselines (cnn_lstm,
spatial_attn -- see training/train_lstm_baselines.py). Mirrors
evaluation/generate_results.py's captioning metrics (BLEU-1/4, METEOR,
ROUGE-L, CIDEr-D) plus efficiency, on the same group-aware test split, so
results are directly comparable to the proposed model's
outputs/logs/test_metrics.json and to the other baselines' metrics files.

Usage (NOT executed by this change -- scaffolding only):
    python evaluation/generate_results_lstm_baselines.py --arch cnn_lstm --checkpoint outputs/model/lstmbaseline_cnn_lstm_seed42_best.pt
    python evaluation/generate_results_lstm_baselines.py --arch spatial_attn --checkpoint outputs/model/lstmbaseline_spatial_attn_seed42_best.pt
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

ARCHS = {
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
    parser.add_argument("--arch", required=True, choices=list(ARCHS.keys()))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beam_width", type=int, default=5)
    args = parser.parse_args()

    device = config.get_device()
    vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cls = ARCHS[args.arch]
    model = model_cls(vocab_size=ckpt["vocab_size"], pad_id=pad_id,
                       max_caption_len=config.MAX_CAPTION_LEN).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_loader = build_test_loader(vocab, args.batch_size, args.limit)

    cap_hyps, cap_refs = [], []
    for batch in test_loader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)

        gen_ids = model.generate_beam(images, bos_id=vocab.word2idx[BOS],
                                        eos_id=vocab.word2idx[EOS],
                                        max_len=config.MAX_CAPTION_LEN, beam_width=args.beam_width)

        for i in range(images.size(0)):
            hyp_text = vocab.decode(gen_ids[i].tolist())
            ref_text = vocab.decode(caption_ids[i].tolist())
            cap_hyps.append(tokenize(hyp_text))
            cap_refs.append(tokenize(ref_text))

    metrics = {
        "arch": args.arch,
        "captioning": {
            "caption": {
                "bleu1": bleu_score(cap_hyps, cap_refs, n=1),
                "bleu4": bleu_score(cap_hyps, cap_refs, n=4),
                "meteor": meteor_score(cap_hyps, cap_refs),
                "rouge_l": rouge_l(cap_hyps, cap_refs),
                "cider_d": cider_d(cap_hyps, cap_refs),
            },
        },
        "decoding": {"beam_width": args.beam_width},
    }

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

    metrics_path = os.path.join(config.LOGS_DIR, f"test_metrics_lstmbaseline_{args.arch}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {args.arch} metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

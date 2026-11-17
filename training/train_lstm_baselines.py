"""
Training loop for the two LSTM-decoder comparison baselines:

  cnn_lstm      -- Baseline 5, models/baseline/cnn_lstm_captioner.py
                   (CNNLSTMCaptioner: pooled global feature, inject-once,
                   no attention at any decoding step)
  spatial_attn  -- Baseline 6, models/baseline/spatial_attention_captioner.py
                   (SpatialAttentionCaptioner: single-head additive/Bahdanau
                   spatial attention re-applied at every decoding step,
                   "Show, Attend and Tell" style)

Reuses the exact same manifests, group-aware split, and vocabulary as the
proposed model, the SOTA Transformer baseline (training/train_baseline.py),
and the 5 literature baselines (training/train_lit_baselines.py), so all
comparisons in conference/comparison_plan.md stay apples-to-apples.

Usage (NOT executed by this change -- scaffolding only):
    python training/train_lstm_baselines.py --arch cnn_lstm --epochs 15 --limit 2500 --batch_size 32 --seed 42
    python training/train_lstm_baselines.py --arch spatial_attn --epochs 15 --limit 2500 --batch_size 32 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import config
from data.datasets import build_datasets, split_indices_grouped
from data.vocab import PAD, Vocabulary, VOCAB_PATH, build_vocab
from models.baseline.cnn_lstm_captioner import CNNLSTMCaptioner
from models.baseline.spatial_attention_captioner import SpatialAttentionCaptioner

ARCHS = {
    "cnn_lstm": CNNLSTMCaptioner,
    "spatial_attn": SpatialAttentionCaptioner,
}


def build_dataloaders(vocab: Vocabulary, batch_size: int, limit: int | None, num_workers: int = 0):
    sources = build_datasets(vocab)
    if not sources:
        raise RuntimeError("No dataset manifests found under datasets/. Run the data/prepare_*.py scripts first.")

    source_sizes = {name: (len(ds) if limit is None else min(limit, len(ds)))
                    for name, ds in sources.items()}
    if limit is None:
        min_size = min(source_sizes.values())
        source_sizes = {name: min_size for name in source_sizes}

    train_sets, val_sets, test_sets = [], [], []
    for (name, ds), n in zip(sources.items(), source_sizes.values()):
        group_ids = ds.df["video_id"].iloc[:n].tolist()
        train_idx, val_idx, test_idx = split_indices_grouped(group_ids)
        train_sets.append(Subset(ds, train_idx))
        val_sets.append(Subset(ds, val_idx))
        test_sets.append(Subset(ds, test_idx))

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(ConcatDataset(test_sets), batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def run_epoch(model, loader: DataLoader, optimizer, device: torch.device,
              pad_id: int, grad_clip_norm: float = 5.0, use_amp: bool = False) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = use_amp and device.type == "cuda"

    total_loss, n_batches = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)

        caption_input = torch.roll(caption_ids, shifts=1, dims=1)
        caption_input[:, 0] = caption_ids[0, 0]  # <bos>

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images, caption_input)
                loss = model.caption_loss(logits, caption_ids)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=list(ARCHS.keys()))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = config.get_device()
    print("arch:", args.arch, "device:", device)

    if not os.path.exists(VOCAB_PATH):
        manifests = [config.ROCO_MANIFEST, config.ROCOV2_MANIFEST, config.MEDVIDEOCAP_MANIFEST]
        vocab = build_vocab(manifests)
        vocab.save(VOCAB_PATH)
    else:
        vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    train_loader, val_loader, _ = build_dataloaders(vocab, args.batch_size, args.limit,
                                                     num_workers=args.num_workers)

    model_cls = ARCHS[args.arch]
    model = model_cls(vocab_size=len(vocab), pad_id=pad_id,
                       max_caption_len=config.MAX_CAPTION_LEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    tag = f"{args.arch}_seed{args.seed}"
    log_path = os.path.join(config.LOGS_DIR, f"train_log_lstmbaseline_{tag}.csv")
    log_rows = []
    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, pad_id, use_amp=args.amp)
        val_loss = run_epoch(model, val_loader, None, device, pad_id, use_amp=args.amp)
        scheduler.step()
        print(f"[{args.arch} epoch {epoch}/{args.epochs}] lr={scheduler.get_last_lr()[0]:.6f} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        ckpt_path = os.path.join(config.MODEL_DIR, f"lstmbaseline_{tag}_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "vocab_size": len(vocab), "epoch": epoch, "arch": args.arch},
                   ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(config.MODEL_DIR, f"lstmbaseline_{tag}_best.pt")
            torch.save({"model": model.state_dict(), "vocab_size": len(vocab), "epoch": epoch,
                        "arch": args.arch, "val_loss": val_loss}, best_path)
            print(f"  -> new best (val_loss={val_loss:.4f}), saved {best_path}")

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(log_rows)

    print(f"[{args.arch}] Best epoch: {best_epoch} (val_loss={best_val_loss:.4f}) -> lstmbaseline_{tag}_best.pt")


if __name__ == "__main__":
    main()

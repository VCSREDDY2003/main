"""
Training loop for the SOTA-style TransformerCaptioner comparison baseline
(models/baseline/transformer_captioner.py). Reuses the same manifests,
group-aware split, and vocabulary as the proposed GAM-LSTM-IAT model so the
comparison in conference/comparison_plan.md is apples-to-apples.

Usage:
    python training/train_baseline.py --epochs 30 --limit 200
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
from models.baseline.transformer_captioner import TransformerCaptioner


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


def run_epoch(model: TransformerCaptioner, loader: DataLoader, optimizer, device: torch.device,
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
    parser.add_argument("--epochs", type=int, default=30)
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

    if not os.path.exists(VOCAB_PATH):
        manifests = [config.ROCO_MANIFEST, config.ROCOV2_MANIFEST, config.MEDVIDEOCAP_MANIFEST]
        vocab = build_vocab(manifests)
        vocab.save(VOCAB_PATH)
    else:
        vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    train_loader, val_loader, _ = build_dataloaders(vocab, args.batch_size, args.limit,
                                                     num_workers=args.num_workers)

    model = TransformerCaptioner(vocab_size=len(vocab), pad_id=pad_id,
                                   max_caption_len=config.MAX_CAPTION_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    log_path = os.path.join(config.LOGS_DIR, "train_log_baseline_transformer.csv")
    log_rows = []
    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, pad_id, use_amp=args.amp)
        val_loss = run_epoch(model, val_loader, None, device, pad_id, use_amp=args.amp)
        scheduler.step()
        print(f"[epoch {epoch}/{args.epochs}] lr={scheduler.get_last_lr()[0]:.6f} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        ckpt_path = os.path.join(config.MODEL_DIR, f"transformer_captioner_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "vocab_size": len(vocab), "epoch": epoch}, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(config.MODEL_DIR, "transformer_captioner_best.pt")
            torch.save({"model": model.state_dict(), "vocab_size": len(vocab), "epoch": epoch,
                        "val_loss": val_loss}, best_path)
            print(f"  -> new best (val_loss={val_loss:.4f}), saved {best_path}")

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
            writer.writeheader()
            writer.writerows(log_rows)

    print(f"Best epoch: {best_epoch} (val_loss={best_val_loss:.4f}) -> transformer_captioner_best.pt")


if __name__ == "__main__":
    main()

"""
Generic training loop for the 5 literature-baseline architectures
(models/baseline/lit_baselines.py), reusing the exact same data pipeline,
group-aware split, and vocabulary as the proposed model and the SOTA
Transformer baseline (training/train_baseline.py) for a same-config,
apples-to-apples comparison.

Usage:
    python training/train_lit_baselines.py --arch vitgpt --epochs 15 --limit 2500 --batch_size 32 --seed 42
    python training/train_lit_baselines.py --arch span --epochs 15 --limit 2500 --batch_size 32 --seed 42
    python training/train_lit_baselines.py --arch u2dp --epochs 15 --limit 2500 --batch_size 32 --seed 42
    python training/train_lit_baselines.py --arch tinytrcap --epochs 15 --limit 2500 --batch_size 32 --seed 42
    python training/train_lit_baselines.py --arch clipcappp --epochs 15 --limit 2500 --batch_size 32 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler

import config
from data.datasets import build_datasets, split_indices_grouped
from data.vocab import PAD, Vocabulary, VOCAB_PATH, build_vocab
from models.baseline.lit_baselines import ARCHS, U2DPCaptioner


def compute_class_weights(train_concat_ds, num_classes=5):
    """Inverse-frequency class weights (weight_k = N / (C * n_k)), computed
    from the actual training-split label_5class distribution -- same
    formula the proposed model uses (training/train.py CLASS_WEIGHTS_5CLASS),
    but computed dynamically here so it's correct for any --limit."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for i in range(len(train_concat_ds)):
        label = int(train_concat_ds[i]["label_5class"].item())
        if 0 <= label < num_classes:
            counts[label] += 1
    n = counts.sum()
    weights = n / (num_classes * np.maximum(counts, 1))
    return counts, torch.tensor(weights, dtype=torch.float32)


def build_dataloaders(vocab, batch_size, limit, num_workers=0, balance_sampler=False):
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

    train_concat = ConcatDataset(train_sets)
    class_counts, class_weights = compute_class_weights(train_concat)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if balance_sampler:
        # Oversample rare classes / undersample common ones: each sample's
        # draw probability is proportional to 1/class_count[its label], so
        # over an epoch of len(train_concat) draws (with replacement) all
        # 5 classes are seen roughly equally often, instead of following
        # the natural ~29%-Other/~8%-Ultrasound imbalance.
        sample_weights = np.zeros(len(train_concat), dtype=np.float64)
        for i in range(len(train_concat)):
            label = int(train_concat[i]["label_5class"].item())
            sample_weights[i] = 1.0 / max(class_counts[label], 1)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_concat), replacement=True)
        train_loader = DataLoader(train_concat, batch_size=batch_size, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_concat, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(ConcatDataset(val_sets), batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(ConcatDataset(test_sets), batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, class_weights


def _augmented_view(images):
    """Second augmented view for U2DP's consistency loss: horizontal flip +
    mild brightness jitter (cheap, deterministic-shape augmentation)."""
    flipped = TF.hflip(images)
    return TF.adjust_brightness(flipped, brightness_factor=1.1)


def run_epoch(model, loader, optimizer, device, pad_id, grad_clip_norm=5.0,
              use_amp=False, consistency_weight=0.0, class_weights=None):
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = use_amp and device.type == "cuda"

    total_loss, total_cap_loss, total_cons_loss, total_cls_loss, n_batches = 0.0, 0.0, 0.0, 0.0, 0
    correct5, n5 = 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        label_5class = batch["label_5class"].to(device)

        caption_input = torch.roll(caption_ids, shifts=1, dims=1)
        caption_input[:, 0] = caption_ids[0, 0]

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(images, caption_input)
                cap_loss = model.caption_loss(logits, caption_ids)
                cls_logits = model.classify_logits(images)
                cls_loss = model.classification_loss(cls_logits, label_5class, weight=class_weights)
                loss = cap_loss + cls_loss
                cons_loss = torch.tensor(0.0, device=device)
                if consistency_weight > 0.0 and isinstance(model, U2DPCaptioner):
                    view2 = _augmented_view(images)
                    cons_loss = model.consistency_loss(images, view2)
                    loss = loss + consistency_weight * cons_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        total_loss += float(loss.item())
        total_cap_loss += float(cap_loss.item())
        total_cons_loss += float(cons_loss.item())
        total_cls_loss += float(cls_loss.item())
        correct5 += int((cls_logits.detach().argmax(-1) == label_5class).sum().item())
        n5 += label_5class.size(0)
        n_batches += 1

    n = max(n_batches, 1)
    acc5 = correct5 / max(n5, 1)
    return total_loss / n, total_cap_loss / n, total_cons_loss / n, total_cls_loss / n, acc5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, choices=list(ARCHS.keys()))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--consistency_weight", type=float, default=0.5,
                         help="U2DP-only: weight of the semi-supervised consistency loss")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--balance_sampler", action="store_true", default=False,
                         help="oversample rare / undersample common 5-class labels via a "
                              "WeightedRandomSampler on the training loader, instead of plain "
                              "shuffle=True (class imbalance found: Other ~29%%, Ultrasound ~8%%)")
    parser.add_argument("--class_weighted_loss", action="store_true", default=False,
                         help="apply inverse-class-frequency weights to the classification "
                              "cross-entropy loss (same treatment the proposed model already "
                              "gets in training/train.py, previously missing here)")
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

    train_loader, val_loader, _, class_weights = build_dataloaders(
        vocab, args.batch_size, args.limit, num_workers=args.num_workers,
        balance_sampler=args.balance_sampler)
    class_weights = class_weights.to(device) if args.class_weighted_loss else None
    print(f"balance_sampler={args.balance_sampler} class_weighted_loss={args.class_weighted_loss} "
          f"class_weights={class_weights.tolist() if class_weights is not None else None}")

    model_cls = ARCHS[args.arch]
    model = model_cls(vocab_size=len(vocab), pad_id=pad_id,
                       max_caption_len=config.MAX_CAPTION_LEN).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    tag = f"{args.arch}_seed{args.seed}"
    if args.balance_sampler:
        tag = f"{tag}_balsamp"
    if args.class_weighted_loss:
        tag = f"{tag}_clsw"
    log_path = os.path.join(config.LOGS_DIR, f"train_log_litbaseline_{tag}.csv")
    log_rows = []
    best_val_loss = float("inf")
    best_epoch = None

    cw = args.consistency_weight if args.arch == "u2dp" else 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_cap, train_cons, train_cls, train_acc5 = run_epoch(
            model, train_loader, optimizer, device, pad_id, use_amp=args.amp, consistency_weight=cw,
            class_weights=class_weights)
        val_loss, val_cap, val_cons, val_cls, val_acc5 = run_epoch(
            model, val_loader, None, device, pad_id, use_amp=args.amp, consistency_weight=cw,
            class_weights=class_weights)
        scheduler.step()
        print(f"[{args.arch} epoch {epoch}/{args.epochs}] lr={scheduler.get_last_lr()[0]:.6f} "
              f"train_loss={train_loss:.4f} (cap={train_cap:.4f} cons={train_cons:.4f} cls={train_cls:.4f} acc5={train_acc5:.3f}) "
              f"val_loss={val_loss:.4f} (cap={val_cap:.4f} cons={val_cons:.4f} cls={val_cls:.4f} acc5={val_acc5:.3f})")

        ckpt_path = os.path.join(config.MODEL_DIR, f"litbaseline_{tag}_epoch{epoch}.pt")
        torch.save({"model": model.state_dict(), "vocab_size": len(vocab), "epoch": epoch, "arch": args.arch},
                   ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_path = os.path.join(config.MODEL_DIR, f"litbaseline_{tag}_best.pt")
            torch.save({"model": model.state_dict(), "vocab_size": len(vocab), "epoch": epoch,
                        "arch": args.arch, "val_loss": val_loss}, best_path)
            print(f"  -> new best (val_loss={val_loss:.4f}), saved {best_path}")

        log_rows.append({"epoch": epoch, "train_loss": train_loss, "train_cap": train_cap,
                          "train_cons": train_cons, "train_cls": train_cls, "train_acc5": train_acc5,
                          "val_loss": val_loss, "val_cap": val_cap,
                          "val_cons": val_cons, "val_cls": val_cls, "val_acc5": val_acc5})
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)

    print(f"[{args.arch}] Best epoch: {best_epoch} (val_loss={best_val_loss:.4f}) -> litbaseline_{tag}_best.pt")


if __name__ == "__main__":
    main()

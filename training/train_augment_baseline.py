"""
IAT-vs-alternative-augmentation comparison (Phase 2, classification-only
scope, 2026-11). Trains the same CNN backbone + image-only 5-class
classification head used throughout this project, with IAT, GAN and LSTM
all disabled (apply_iat=False, disable_gan=True, disable_lstm=True) so the
only thing that changes between runs is which augmentation is applied to
the raw image batch before the CNN forward pass. This isolates IAT's
specific contribution from "any augmentation helps."

Augmentation methods (training/aug_methods.py): none, cropflip, mixup,
cutmix, randaugment. IAT itself is compared via run_phase2_ablation.sh's
CNN-only vs CNN+IAT rows (apply_iat=True there, with the real IAT module,
not a generic augmentation function) -- this script covers the four
*alternatives* only, so the two result sets together give the full
IAT-vs-N-alternatives table.

Usage:
    python training/train_augment_baseline.py --aug_method mixup --epochs 20 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import os

import torch

import config
from data.vocab import PAD, VOCAB_PATH, Vocabulary
from evaluation.metrics import classification_metrics
from models.iatgam_model import IATGAMModel
from training.aug_methods import AUG_METHODS, mixed_cross_entropy
from training.train import CLASS_WEIGHTS_5CLASS, build_dataloaders


def run_epoch(model, loader, optimizer, device, class_weights, aug_fn, pad_id):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, n = 0.0, 0
    all_preds, all_targets = [], []
    for batch in loader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)
        caption_mask = (caption_ids != pad_id).long()
        label_5class = batch["label_5class"].to(device)

        aug_images, labels_a, labels_b, lam = aug_fn(images, label_5class)

        with torch.set_grad_enabled(is_train):
            out = model(aug_images, caption_ids, caption_mask, label_5class, caption_ids, summary_ids,
                        apply_iat=False, disable_lstm=True, disable_gan=True)
            logits = out["img_class_logits_5class"]
            loss = mixed_cross_entropy(logits, labels_a, labels_b, lam, weight=class_weights)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        # report accuracy against the dominant (lam-weighted) label, consistent
        # with standard MixUp/CutMix evaluation convention
        preds = logits.argmax(-1)
        ref = labels_a if lam >= 0.5 else labels_b
        all_preds.append(preds.detach().cpu())
        all_targets.append(ref.detach().cpu())
        n += label_5class.size(0)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    metrics = classification_metrics(all_preds, all_targets, num_classes=5)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_method", required=True, choices=list(AUG_METHODS.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leakage_safe_audit_path", type=str, default=None,
                         help="same global cross-source leakage-safe split as training/train.py "
                              "-- pass the same path here for Phase D so the augmentation-"
                              "comparison baselines use an identical split to the ablation runs.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = config.get_device()
    vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    train_loader, val_loader, _ = build_dataloaders(vocab, args.batch_size, args.limit,
                                                     num_workers=args.num_workers, balance_sources=True,
                                                     balance_classes=True,
                                                     leakage_safe_audit_path=args.leakage_safe_audit_path)

    # Same micro-capacity config used throughout Phase 2's ablation, for a
    # like-for-like comparison against the CNN-only / CNN+IAT rows.
    model_cfg = config.ModelConfig(
        vocab_size=len(vocab), embedding_dim=16, lstm_hidden=16, fusion_dim=32,
        dropout=0.1, iat_alpha_init=0.5, iat_ssim_threshold=0.9,
        use_spatial_attention=True, cnn_pretrained=False, cnn_light=False,
        iat_num_layers=7, gam_memory_coupling=False, cnn_depth=7,
        gan_base_channels=8, gan_light_head=True,
    )
    model = IATGAMModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                             eta_min=args.lr * 0.01)

    import math
    class_weights = torch.tensor([math.sqrt(v) for v in CLASS_WEIGHTS_5CLASS], device=device)
    aug_fn = AUG_METHODS[args.aug_method]

    tag = f"augbaseline_{args.aug_method}_seed{args.seed}"
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    log_path = os.path.join(config.LOGS_DIR, f"train_log_{tag}.csv")
    log_rows = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, device, class_weights, aug_fn, pad_id)
        val_m = run_epoch(model, val_loader, None, device, class_weights, aug_fn, pad_id)
        scheduler.step()
        print(f"[{args.aug_method} epoch {epoch}/{args.epochs}] "
              f"train_loss={train_m['loss']:.4f} train_acc={train_m['accuracy']:.4f} "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1']:.4f}")

        if val_m["accuracy"] >= best_val_acc:
            best_val_acc = val_m["accuracy"]
            out_path = os.path.join(config.MODEL_DIR, f"{tag}_best.pt")
            torch.save({"model": model.state_dict(), "config": model_cfg, "epoch": epoch,
                        "val_acc": val_m["accuracy"], "aug_method": args.aug_method}, out_path)

        log_rows.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
                          **{f"val_{k}": v for k, v in val_m.items()}})
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader(); writer.writerows(log_rows)

    print(f"[{args.aug_method}] Done. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()

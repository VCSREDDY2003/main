"""
Fine-tunes ONLY the new image-only classification heads (img_classifier_*,
models/iatgam_model.py) on an existing trained checkpoint.

Why this exists (diagnosed 2026-09): the original classifier_5class (in
LMMDecoder) reads Z, which is built from F_fuse = LayerNorm(F_v + F_cv),
where F_cv comes from cross-attention over the caption text. Since the
5-class labels are themselves keyword-derived from captions, that classifier
learned to read the caption instead of the image (verified: shuffling
captions across a batch changes predictions to match the shuffled caption;
masking the caption collapses every checkpoint to a single constant-class
prediction). The new img_classifier_* heads read F_v directly, before any
caption cross-attention touches it, so they cannot take that shortcut.

This script does NOT retrain the whole model. The CNN backbone already
received real gradient signal from the captioning and GAN losses during the
original run, so F_v should already carry real visual features -- this only
needs to teach a new head to read them for classification, and optionally
lightly fine-tune the CNN. Backbone is frozen by default (--finetune_cnn to
allow light CNN updates too); GAN/LSTM/caption-decoder/old classifier_*
weights are always frozen -- this script trains only the new heads (and
optionally the CNN) on the classification loss.

Usage:
    python training/finetune_image_classifier.py \
        --checkpoint outputs/model/iatgam_aug_strong_seed42_micro2m_ae1.0_softcw_balcls_best.pt \
        --epochs 10 --tag micro_withIAT
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import torch
import torch.nn as nn

import config
from data.vocab import PAD, VOCAB_PATH, Vocabulary
from evaluation.metrics import classification_metrics
from models.iatgam_model import IATGAMModel
from training.train import CLASS_WEIGHTS_3CLASS, CLASS_WEIGHTS_5CLASS, CLASS_WEIGHTS_BINARY, build_dataloaders


def run_epoch(model, loader, optimizer, device, class_weights, apply_iat, disable_lstm, disable_gan, pad_id):
    is_train = optimizer is not None
    model.train(is_train)
    # Keep every frozen submodule in eval mode even during "training" so
    # BatchNorm/Dropout there don't drift from the loaded checkpoint's
    # statistics -- only the new heads (and CNN, if --finetune_cnn) should
    # actually change.
    for name, module in model.named_modules():
        if not any(name.startswith(p) for p in ("img_classifier_",) ) and name != "" and not name.startswith("cnn"):
            module.eval()

    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_targets = [], []
    for batch in loader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)
        caption_mask = (caption_ids != pad_id).long()
        label_5class = batch["label_5class"].to(device)

        with torch.set_grad_enabled(is_train):
            out = model(images, caption_ids, caption_mask, label_5class, caption_ids, summary_ids,
                        apply_iat=apply_iat, disable_lstm=disable_lstm, disable_gan=disable_gan)
            logits = out["img_class_logits_5class"]
            loss = nn.functional.cross_entropy(logits, label_5class, weight=class_weights)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item())
        preds = logits.argmax(-1)
        correct += int((preds == label_5class).sum().item())
        n += label_5class.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(label_5class.detach().cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    metrics = classification_metrics(all_preds, all_targets, num_classes=5)
    metrics["loss"] = total_loss / max(len(loader), 1)
    metrics["accuracy_check"] = correct / max(n, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help="AdamW weight decay -- increase (e.g. 1e-2) to fight overfitting when "
                              "training the CNN end-to-end on a small --limit (diagnosed 2026-09: "
                              "micro/2.4M config overfit badly with only 1e-4 on a 2000/source limit "
                              "-- train_acc climbed to 0.82 while val_acc peaked at epoch 4 then "
                              "degraded/oscillated for the remaining epochs).")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--finetune_cnn", action="store_true", default=False,
                         help="also allow the CNN backbone to update (default: frozen, only the new "
                              "image-only classifier heads train)")
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    device = config.get_device()
    vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["config"]
    ablation = ckpt.get("ablation", {"disable_lstm": False, "disable_gan": False, "disable_iat": False})
    apply_iat = not ablation.get("disable_iat", False)
    disable_lstm = ablation.get("disable_lstm", False)
    disable_gan = ablation.get("disable_gan", False)

    model = IATGAMModel(model_cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    # img_classifier_* are new -- expected to be "missing" (randomly initialized) here.
    real_missing = [k for k in missing if not k.startswith("img_classifier_") and "iat.perturb" not in k]
    if real_missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint mismatch: missing={real_missing} unexpected={unexpected}")
    print(f"Loaded {args.checkpoint}; img_classifier_* heads randomly initialized "
          f"({len(missing)} new params), everything else restored from checkpoint.")

    # Freeze everything except the new image-only heads (and CNN, if asked).
    trainable_prefixes = ("img_classifier_binary", "img_classifier_3class", "img_classifier_5class")
    if args.finetune_cnn:
        trainable_prefixes = trainable_prefixes + ("cnn",)
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(p) for p in trainable_prefixes)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} "
          f"({'CNN + new heads' if args.finetune_cnn else 'new heads only'})")

    train_loader, val_loader, _ = build_dataloaders(vocab, args.batch_size, args.limit,
                                                     num_workers=args.num_workers, balance_sources=True,
                                                     balance_classes=True)

    class_weights = torch.tensor(CLASS_WEIGHTS_5CLASS, device=device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    os.makedirs(config.LOGS_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    log_path = os.path.join(config.LOGS_DIR, f"finetune_imgcls_{args.tag}.csv")
    log_rows = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, device, class_weights, apply_iat, disable_lstm, disable_gan, pad_id)
        val_m = run_epoch(model, val_loader, None, device, class_weights, apply_iat, disable_lstm, disable_gan, pad_id)
        scheduler.step()
        print(f"[epoch {epoch}/{args.epochs}] train_loss={train_m['loss']:.4f} train_acc={train_m['accuracy']:.4f} "
              f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1']:.4f}")

        if val_m["accuracy"] >= best_val_acc:
            best_val_acc = val_m["accuracy"]
            out_path = os.path.join(config.MODEL_DIR, f"iatgam_{args.tag}_imgcls_best.pt")
            torch.save({"model": model.state_dict(), "config": model_cfg, "epoch": epoch,
                        "val_acc": val_m["accuracy"], "ablation": ablation,
                        "source_checkpoint": args.checkpoint, "finetune_cnn": args.finetune_cnn},
                       out_path)
            print(f"  -> new best image-only classifier (val_acc={val_m['accuracy']:.4f}), saved {out_path}")

        log_rows.append({"epoch": epoch, **{f"train_{k}": v for k, v in train_m.items()},
                          **{f"val_{k}": v for k, v in val_m.items()}})
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader(); writer.writerows(log_rows)

    print(f"Done. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()

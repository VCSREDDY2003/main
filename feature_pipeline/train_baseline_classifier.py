"""
Trains ONE param-matched (~2.4M) pure classifier baseline (no IAT, no GAN,
no LSTM caption head -- image -> backbone -> 5-class softmax) on the
combined ROCO+ROCOv2 4-class-balanced manifests, reusing train.py's own
data loading (build_dataloaders) so the split/manifest-override logic is
identical to the main IAT-GAM pipeline's.

Usage:
    python feature_pipeline/train_baseline_classifier.py --arch resnet_small --epochs 50
    python feature_pipeline/train_baseline_classifier.py --arch resnet_small --eval_only \
        --checkpoint outputs/model/baseline_resnet_small_combined_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from data.vocab import VOCAB_PATH, Vocabulary
from evaluation.metrics import classification_metrics, confusion_matrix_report
from feature_pipeline.baseline_archs import build_model, count_params
from training.train import build_dataloaders

ROCO_M = "feature_pipeline/data/roco_manifest_4class_balanced.csv"
ROCOV2_M = "feature_pipeline/data/rocov2_manifest_4class_balanced.csv"


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_targets = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label_5class"].to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            n += images.size(0)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(labels.detach().cpu().numpy())
    return (total_loss / n, correct / n,
            np.concatenate(all_preds), np.concatenate(all_targets))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=[
        "resnet_small", "efficientnet_small", "mlp", "mobilenet_style",
        "shufflenet_style", "squeezenet_style", "densenet_style"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--eval_only", action="store_true", default=False)
    p.add_argument("--checkpoint", type=str, default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vocab is unused by the classifier itself but required by build_dataloaders'
    # dataset loader (ManifestDataset encodes captions regardless); reuse the
    # already-built vocab file rather than rebuilding it.
    vocab = Vocabulary.load(VOCAB_PATH)
    train_loader, val_loader, test_loader = build_dataloaders(
        vocab, args.batch_size, limit=None, num_workers=args.num_workers,
        balance_sources=True, balance_classes=False,
        dataset_sources=["roco", "rocov2"],
        roco_manifest_override=ROCO_M, rocov2_manifest_override=ROCOV2_M)

    model = build_model(args.arch).to(device)
    n_params = count_params(model)
    print(f"[{args.arch}] params: {n_params:,}")

    ckpt_path = args.checkpoint or os.path.join(
        config.MODEL_DIR, f"baseline_{args.arch}_combined_best.pt")

    if args.eval_only:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        _, test_acc, preds, targets = run_epoch(model, test_loader, device, optimizer=None)
        cls = classification_metrics(preds, targets, num_classes=5)
        cm = confusion_matrix_report(preds, targets, num_classes=5)
        out = {
            "arch": args.arch, "n_params": n_params, "test_acc": test_acc,
            "classification": {"5class": {**cls, **cm}},
        }
        out_path = os.path.join(config.LOGS_DIR, f"test_metrics_baseline_{args.arch}_combined.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {out_path}")
        print(json.dumps({"test_acc": test_acc, **cls, **cm}, indent=2, default=str))
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_acc, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, device, optimizer=None)
        print(f"[{args.arch}] epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt_path)

    print(f"[{args.arch}] Best epoch: {best_epoch} (val_acc={best_val_acc:.4f}) -> {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, test_acc, preds, targets = run_epoch(model, test_loader, device, optimizer=None)
    cls = classification_metrics(preds, targets, num_classes=5)
    cm = confusion_matrix_report(preds, targets, num_classes=5)
    out = {
        "arch": args.arch, "n_params": n_params, "best_epoch": best_epoch,
        "best_val_acc": best_val_acc, "test_acc": test_acc,
        "classification": {"5class": {**cls, **cm}},
    }
    out_path = os.path.join(config.LOGS_DIR, f"test_metrics_baseline_{args.arch}_combined.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

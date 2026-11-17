"""
Extends the param-matched classifier baselines (feature_pipeline/baseline_archs.py)
with the SAME caption-decoder head used by the main IAT-GAM pipeline
(models/lmm/lmm.py:LMMDecoder), so every baseline architecture gets the full
metric set (classification + BLEU/METEOR/ROUGE/CIDEr/SPICE/BERTScore), not
just classification. This supersedes the classification-only numbers from
train_baseline_classifier.py for these 6 archs (classification numbers may
shift slightly since the loss now also includes the caption/summary terms).

"mlp" is EXCLUDED: it has no convolutional feature map (operates on
flattened pixels, see MLPClassifier in baseline_archs.py), so there is no
spatial feature map to attach LMMDecoder's cross-attention caption decoder
to. Its classification-only result from the earlier run stands as-is.

XGBoost is also excluded (separate reason, see outputs/logs/test_metrics_baseline_xgboost_combined.json):
it is gradient-boosted trees on hand-picked keyword/text features, with no
learned image representation or decoder architecture to attach a caption
head to at all. No captioning metrics exist or will ever exist for it in
this project.

Classification head stays the backbone's OWN fc (image-only, F_v-based),
NOT the LMMDecoder's Z-based classifier head -- same leakage-avoidance
design already used by the main pipeline (see training/losses.py
CompositeLoss comment: Z mixes in caption cross-attention, and 5-class
labels are themselves keyword-derived from captions, so classifying on Z
lets the model read the caption instead of the image). LMMDecoder is used
here ONLY for its caption_decoder/summary_decoder (captioning loss/metrics).

Usage:
    python feature_pipeline/train_baseline_with_caption.py --arch resnet_small --epochs 50
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
from data.vocab import VOCAB_PATH, Vocabulary, tokenize
from evaluation.metrics import (bert_score_f1, bleu_score, cider_d, classification_metrics,
                                 confusion_matrix_report, meteor_score, rouge_1, rouge_l, spice_score)
from feature_pipeline.baseline_archs import FEATURE_DIMS, build_model, count_params
from models.lmm.lmm import LMMDecoder
from training.losses import caption_loss
from training.train import build_dataloaders

ROCO_M = "feature_pipeline/data/roco_manifest_4class_balanced.csv"
ROCOV2_M = "feature_pipeline/data/rocov2_manifest_4class_balanced.csv"

PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
CAPTIONABLE_ARCHS = ["resnet_small", "efficientnet_small", "mobilenet_style",
                     "shufflenet_style", "squeezenet_style", "densenet_style"]


class BaselineWithCaption(nn.Module):
    """backbone (frozen classifier head, image-only) + LMMDecoder (caption/summary only)."""

    def __init__(self, arch: str, vocab_size: int, num_classes: int = 5):
        super().__init__()
        self.backbone = build_model(arch, num_classes=num_classes)
        feat_dim = FEATURE_DIMS[arch]
        self.feat_dim = feat_dim
        self.lmm = LMMDecoder(
            fusion_dim=feat_dim, vocab_size=vocab_size, embed_dim=256, hidden_dim=256,
            num_classes_binary=2, num_classes_3class=3, num_classes_5class=num_classes,
            dropout=0.3, pad_id=PAD_ID, cnn_feature_dim=feat_dim, use_spatial_attention=True,
        )

    def forward(self, images, caption_input, summary_input):
        feat_map = self.backbone.forward_features(images)
        class_logits = self.backbone.fc(F.adaptive_avg_pool2d(feat_map, 1).flatten(1)) \
            if hasattr(self.backbone, "fc") else self.backbone.classifier(feat_map).mean(dim=(2, 3))
        z = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        lmm_out = self.lmm(z, caption_input, summary_input, feat_map)
        return class_logits, lmm_out

    @torch.no_grad()
    def generate(self, images, max_len=32):
        feat_map = self.backbone.forward_features(images)
        z = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        out = self.lmm.generate(z, feat_map, BOS_ID, EOS_ID, max_len=max_len, beam_width=1, unk_id=UNK_ID)
        class_logits = self.backbone.fc(z) if hasattr(self.backbone, "fc") else self.backbone.classifier(feat_map).mean(dim=(2, 3))
        return out["caption_ids"], out["summary_ids"], class_logits


def shift_right(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """teacher-forcing input (ids[:-1]) / target (ids[1:])."""
    return ids[:, :-1], ids[:, 1:]


def run_train_epoch(model, loader, device, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label_5class"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)
        cap_in, cap_tgt = shift_right(caption_ids)
        sum_in, sum_tgt = shift_right(summary_ids)

        optimizer.zero_grad()
        class_logits, lmm_out = model(images, cap_in, sum_in)
        l_cls = F.cross_entropy(class_logits, labels)
        l_cap = caption_loss(lmm_out["caption_logits"], cap_tgt, PAD_ID, label_smoothing=0.1)
        l_sum = caption_loss(lmm_out["summary_logits"], sum_tgt, PAD_ID, label_smoothing=0.1)
        loss = l_cls + l_cap + l_sum
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (class_logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def run_val_epoch(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label_5class"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)
        cap_in, cap_tgt = shift_right(caption_ids)
        sum_in, sum_tgt = shift_right(summary_ids)
        class_logits, lmm_out = model(images, cap_in, sum_in)
        l_cls = F.cross_entropy(class_logits, labels)
        l_cap = caption_loss(lmm_out["caption_logits"], cap_tgt, PAD_ID, label_smoothing=0.1)
        l_sum = caption_loss(lmm_out["summary_logits"], sum_tgt, PAD_ID, label_smoothing=0.1)
        loss = l_cls + l_cap + l_sum
        total_loss += loss.item() * images.size(0)
        correct += (class_logits.argmax(1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_full(model, loader, device, vocab):
    model.eval()
    all_preds, all_targets = [], []
    cap_hyps, cap_refs, cap_hyp_text, cap_ref_text = [], [], [], []
    sum_hyps, sum_refs = [], []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label_5class"].to(device)
        caption_ids = batch["caption_ids"]
        summary_ids = batch["summary_ids"]
        cap_gen, sum_gen, class_logits = model.generate(images)
        preds = class_logits.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(labels.cpu().numpy())
        for i in range(images.size(0)):
            hyp_text = vocab.decode(cap_gen[i].tolist())
            ref_text = vocab.decode(caption_ids[i].tolist())
            cap_hyps.append(tokenize(hyp_text))
            cap_refs.append(tokenize(ref_text))
            cap_hyp_text.append(hyp_text)
            cap_ref_text.append(ref_text)
            sum_hyp_text = vocab.decode(sum_gen[i].tolist())
            sum_ref_text = vocab.decode(summary_ids[i].tolist())
            sum_hyps.append(tokenize(sum_hyp_text))
            sum_refs.append(tokenize(sum_ref_text))

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    cls = classification_metrics(preds, targets, num_classes=5)
    cm = confusion_matrix_report(preds, targets, num_classes=5)

    captioning = {
        "caption": {
            "bleu1": bleu_score(cap_hyps, cap_refs, n=1),
            "bleu4": bleu_score(cap_hyps, cap_refs, n=4),
            "meteor": meteor_score(cap_hyps, cap_refs),
            "rouge_1": rouge_1(cap_hyps, cap_refs),
            "rouge_l": rouge_l(cap_hyps, cap_refs),
            "cider_d": cider_d(cap_hyps, cap_refs),
            "spice": spice_score(cap_hyps, cap_refs),
            "bert_score_f1": bert_score_f1(cap_hyp_text, cap_ref_text),
        },
        "summary": {
            "bleu1": bleu_score(sum_hyps, sum_refs, n=1),
            "bleu4": bleu_score(sum_hyps, sum_refs, n=4),
            "meteor": meteor_score(sum_hyps, sum_refs),
            "rouge_1": rouge_1(sum_hyps, sum_refs),
            "rouge_l": rouge_l(sum_hyps, sum_refs),
            "cider_d": cider_d(sum_hyps, sum_refs),
            "spice": spice_score(sum_hyps, sum_refs),
        },
    }
    return {"classification": {"5class": {**cls, **cm}}, "captioning": captioning}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=CAPTIONABLE_ARCHS)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary.load(VOCAB_PATH)
    train_loader, val_loader, test_loader = build_dataloaders(
        vocab, args.batch_size, limit=None, num_workers=args.num_workers,
        balance_sources=True, balance_classes=False,
        dataset_sources=["roco", "rocov2"],
        roco_manifest_override=ROCO_M, rocov2_manifest_override=ROCOV2_M)

    model = BaselineWithCaption(args.arch, vocab_size=len(vocab)).to(device)
    n_backbone_params = count_params(model.backbone)
    n_total_params = count_params(model)
    print(f"[{args.arch}+caption] backbone_params: {n_backbone_params:,}  total_params(+LMMDecoder): {n_total_params:,}")

    ckpt_path = os.path.join(config.MODEL_DIR, f"baseline_{args.arch}_withcap_combined_best.pt")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_acc, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_train_epoch(model, train_loader, device, optimizer)
        val_loss, val_acc = run_val_epoch(model, val_loader, device)
        print(f"[{args.arch}+caption] epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), ckpt_path)

    print(f"[{args.arch}+caption] Best epoch: {best_epoch} (val_acc={best_val_acc:.4f}) -> {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    result = evaluate_full(model, test_loader, device, vocab)
    result.update({
        "arch": args.arch, "n_backbone_params": n_backbone_params,
        "n_total_params_with_caption_head": n_total_params,
        "best_epoch": best_epoch, "best_val_acc": best_val_acc,
    })
    out_path = os.path.join(config.LOGS_DIR, f"test_metrics_baseline_{args.arch}_withcap_combined.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

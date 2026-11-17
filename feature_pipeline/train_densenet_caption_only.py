"""
densenet_style-only fix: DenseNetStyleSmall.forward_features returns an
unnormalized raw feature map (no final BatchNorm, unlike the other 5
captionable archs), which is fine for plain classification but explodes
LMMDecoder's spatial cross-attention when trained jointly (observed:
epoch-1 train_loss=282509 vs ~6-11 for every other arch).

Fix: freeze the already-trained densenet_style classification backbone
(outputs/model/baseline_densenet_style_combined_best.pt, 92.52% test acc,
from feature_pipeline/train_baseline_classifier.py) untouched, insert a
FRESH trainable BatchNorm2d to normalize feat_map, and train only that
BatchNorm2d + LMMDecoder on captioning loss. Classification numbers are
reused unchanged from the existing run (backbone frozen -> cannot shift).
"""
from __future__ import annotations

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
ARCH = "densenet_style"


class DenseNetCaptionOnly(nn.Module):
    def __init__(self, vocab_size, num_classes=5):
        super().__init__()
        self.backbone = build_model(ARCH, num_classes=num_classes)
        ckpt = os.path.join(config.MODEL_DIR, "baseline_densenet_style_combined_best.pt")
        self.backbone.load_state_dict(torch.load(ckpt, map_location="cpu"))
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        feat_dim = FEATURE_DIMS[ARCH]
        self.feat_dim = feat_dim
        self.norm = nn.BatchNorm2d(feat_dim)  # fresh, trainable -- fixes unnormalized magnitude
        self.lmm = LMMDecoder(
            fusion_dim=feat_dim, vocab_size=vocab_size, embed_dim=256, hidden_dim=256,
            num_classes_binary=2, num_classes_3class=3, num_classes_5class=num_classes,
            dropout=0.3, pad_id=PAD_ID, cnn_feature_dim=feat_dim, use_spatial_attention=True,
        )

    def forward(self, images, caption_input, summary_input):
        with torch.no_grad():
            raw_feat = self.backbone.forward_features(images)
        feat_map = self.norm(raw_feat)
        # densenet's native feature map is 28x28=784 positions, vs 7x7=49 for
        # the other 5 captionable archs -- spatial cross-attention cost scales
        # with position count, so this made epoch 1 take 45-60+ minutes
        # (confirmed: still running at ~920 CPU-seconds / 48 min wall with no
        # epoch completed). Downsample to 7x7 to match the other archs' scale
        # -- fair comparison (same attention resolution budget) and removes
        # the bottleneck.
        feat_map = F.adaptive_avg_pool2d(feat_map, 7)
        z = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        return self.lmm(z, caption_input, summary_input, feat_map)

    @torch.no_grad()
    def generate(self, images, max_len=32):
        raw_feat = self.backbone.forward_features(images)
        feat_map = self.norm(raw_feat)
        feat_map = F.adaptive_avg_pool2d(feat_map, 7)
        z = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        out = self.lmm.generate(z, feat_map, BOS_ID, EOS_ID, max_len=max_len, beam_width=1, unk_id=UNK_ID)
        class_logits = self.backbone.fc(F.adaptive_avg_pool2d(raw_feat, 1).flatten(1))
        return out["caption_ids"], out["summary_ids"], class_logits


def shift_right(ids):
    return ids[:, :-1], ids[:, 1:]


def run_train_epoch(model, loader, device, optimizer):
    model.train()
    model.backbone.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)
        cap_in, cap_tgt = shift_right(caption_ids)
        sum_in, sum_tgt = shift_right(summary_ids)

        optimizer.zero_grad()
        lmm_out = model(images, cap_in, sum_in)
        l_cap = caption_loss(lmm_out["caption_logits"], cap_tgt, PAD_ID, label_smoothing=0.1)
        l_sum = caption_loss(lmm_out["summary_logits"], sum_tgt, PAD_ID, label_smoothing=0.1)
        loss = l_cap + l_sum
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


@torch.no_grad()
def run_val_epoch(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        caption_ids = batch["caption_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)
        cap_in, cap_tgt = shift_right(caption_ids)
        sum_in, sum_tgt = shift_right(summary_ids)
        lmm_out = model(images, cap_in, sum_in)
        l_cap = caption_loss(lmm_out["caption_logits"], cap_tgt, PAD_ID, label_smoothing=0.1)
        l_sum = caption_loss(lmm_out["summary_logits"], sum_tgt, PAD_ID, label_smoothing=0.1)
        loss = l_cap + l_sum
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / n


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
    import argparse
    p = argparse.ArgumentParser()
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

    model = DenseNetCaptionOnly(vocab_size=len(vocab)).to(device)
    n_backbone_params = count_params(model.backbone)
    n_total_params = count_params(model)
    print(f"[{ARCH}+caption(frozen-backbone)] backbone_params: {n_backbone_params:,}  "
          f"total_params(+norm+LMMDecoder): {n_total_params:,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-3)
    ckpt_path = os.path.join(config.MODEL_DIR, "baseline_densenet_style_withcap_combined_best.pt")
    best_val_loss, best_epoch = float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        train_loss = run_train_epoch(model, train_loader, device, optimizer)
        val_loss = run_val_epoch(model, val_loader, device)
        print(f"[{ARCH}+caption(frozen)] epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), ckpt_path)

    print(f"[{ARCH}+caption(frozen)] Best epoch: {best_epoch} (val_loss={best_val_loss:.4f}) -> {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    result = evaluate_full(model, test_loader, device, vocab)

    # Reuse the ALREADY-VALID classification-only numbers (backbone frozen,
    # unchanged from feature_pipeline/train_baseline_classifier.py's run) --
    # do not let this script's own classification eval (run through the
    # frozen backbone anyway, should match closely) silently override the
    # authoritative number if they differ due to eval-time stochasticity.
    orig_path = os.path.join(config.LOGS_DIR, "test_metrics_baseline_densenet_style_combined.json")
    with open(orig_path) as f:
        orig = json.load(f)
    result["classification"] = orig["classification"]
    result["n_backbone_params"] = n_backbone_params
    result["n_total_params_with_caption_head"] = n_total_params
    result["best_epoch"] = best_epoch
    result["best_val_loss"] = best_val_loss
    result["arch"] = ARCH
    result["note"] = ("backbone frozen (loaded from classification-only checkpoint); "
                       "only a fresh BatchNorm2d + LMMDecoder trained on captioning loss, "
                       "to avoid the loss-explosion bug when densenet's unnormalized "
                       "forward_features() output was fed into LMMDecoder's spatial "
                       "cross-attention under full joint backprop.")

    out_path = os.path.join(config.LOGS_DIR, "test_metrics_baseline_densenet_style_withcap_combined.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

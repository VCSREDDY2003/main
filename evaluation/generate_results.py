"""
Qualitative + quantitative results generator (Algorithm 3 inference).

Loads a trained checkpoint, runs the model on a sample of the held-out test
split, and writes:

  - outputs/plots/results_grid.png : image grids with generated caption,
    summary, and predicted vs. ground-truth binary/3-class/5-class labels
  - outputs/logs/test_results.json : per-sample generated text + labels
  - outputs/logs/test_metrics.json : full metric set over the test split --
    classification (accuracy/precision/recall/F1 for binary/3-class/5-class),
    captioning (BLEU-1/4, ROUGE-L, CIDEr-D for caption + summary),
    generative (SSIM of IAT-augmented / GAN outputs vs. original), and
    efficiency (ms/sample, samples/sec)

Usage:
    python evaluation/generate_results.py --checkpoint outputs/model/iatgam_epoch1.pt --num_samples 12
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from data.datasets import build_datasets, split_indices_grouped
from data.vocab import BOS, EOS, PAD, Vocabulary, VOCAB_PATH, tokenize
from evaluation.metrics import (
    auroc_multiclass, bert_score_f1, bleu_score, cider_d, classification_metrics, confusion_matrix_report,
    fid_score, measure_latency_throughput, meteor_score,
    rouge_1, rouge_l, spice_score, ssim_batch,
)
from models.iatgam_model import IATGAMModel
from torch.utils.data import ConcatDataset, Subset

_LABEL_NAMES_5CLASS = ["X-ray", "CT", "MRI", "Ultrasound", "Other/Video"]
_LABEL_NAMES_3CLASS = ["Radiograph", "Cross-sectional", "Other/Video"]
_LABEL_NAMES_BINARY = ["Static image", "Video/Other"]


def build_test_loader(vocab: Vocabulary, batch_size: int, limit: int | None,
                       dataset_sources: list[str] | None = None,
                       roco_manifest_override: str | None = None,
                       rocov2_manifest_override: str | None = None):
    sources = build_datasets(vocab, roco_manifest_override=roco_manifest_override,
                              rocov2_manifest_override=rocov2_manifest_override)
    if dataset_sources:
        sources = {k: v for k, v in sources.items() if k in dataset_sources}
    test_sets = []
    for ds in sources.values():
        n = len(ds) if limit is None else min(limit, len(ds))
        group_ids = ds.df["video_id"].iloc[:n].tolist()
        _, _, test_idx = split_indices_grouped(group_ids)
        test_sets.append(Subset(ds, test_idx))
    return DataLoader(ConcatDataset(test_sets), batch_size=batch_size, shuffle=False, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=12, help="images to render in the qualitative grid")
    parser.add_argument("--limit", type=int, default=None, help="cap samples per source (match training run)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beam_width", type=int, default=5,
                         help="beam width for evaluation decoding (1 = greedy)")
    parser.add_argument("--noise_std", type=float, default=0.0,
                         help="std of additive Gaussian noise injected into test images (post-normalization, "
                              "clamped to [0,1]) to test robustness to corrupted input -- separate from IAT's "
                              "own training-time perturbation. 0.0 = clean test set (default).")
    parser.add_argument("--mask_caption", action="store_true", default=False,
                         help="replace every input caption with a minimal <bos><eos> sequence (no real "
                              "keyword content) before it reaches the caption encoder, so classification "
                              "metrics reflect genuine image-only understanding rather than a caption-"
                              "keyword shortcut (5-class labels are keyword-derived from captions -- "
                              "diagnosed 2026-09: with the real caption visible, classification predictions "
                              "track the caption, not the image, at any noise level up to pure random pixels).")
    parser.add_argument("--qualitative_samples", type=int, default=10,
                         help="fixed sample count for the greedy-vs-beam qualitative dump")
    parser.add_argument("--dataset_sources", type=str, default=None,
                         help="comma-separated subset of {roco,rocov2} to evaluate on (default: both)")
    parser.add_argument("--roco_manifest_override", type=str, default=None,
                         help="use an alternate manifest csv for the roco source (match training)")
    parser.add_argument("--rocov2_manifest_override", type=str, default=None,
                         help="use an alternate manifest csv for the rocov2 source (match training)")
    parser.add_argument("--metrics_out", type=str, default=None,
                         help="override output path for test_metrics.json (default: config.LOGS_DIR)")
    args = parser.parse_args()

    device = config.get_device()
    vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt["config"]
    # Ablation switches (--disable_lstm/--disable_gan/--disable_iat) are baked
    # into the checkpoint at save time (training/train.py) so eval always
    # matches how this specific checkpoint was trained, without needing to
    # re-specify them here and risk a mismatch.
    ablation = ckpt.get("ablation", {"disable_lstm": False, "disable_gan": False, "disable_iat": False})
    apply_iat = not ablation.get("disable_iat", False)
    disable_lstm = ablation.get("disable_lstm", False)
    disable_gan = ablation.get("disable_gan", False)
    if ablation.get("disable_lstm") or ablation.get("disable_gan") or ablation.get("disable_iat"):
        print(f"[ablation] evaluating with disable_lstm={disable_lstm} disable_gan={disable_gan} "
              f"disable_iat={ablation.get('disable_iat', False)} (read from checkpoint)")
    model = IATGAMModel(model_cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    non_perturb_missing = [k for k in missing if "iat.perturb" not in k]
    non_perturb_unexpected = [k for k in unexpected if "iat.perturb" not in k]
    if non_perturb_missing or non_perturb_unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch outside iat.perturb submodule: "
            f"missing={non_perturb_missing} unexpected={non_perturb_unexpected}"
        )
    if missing or unexpected:
        print(f"[warn] iat.perturb submodule architecture changed on disk since this checkpoint "
              f"was trained (models/iat/iat.py edited mid-run 2026-09-03); loaded with strict=False. "
              f"{len(missing)} missing / {len(unexpected)} unexpected keys, all within iat.perturb. "
              f"Caption/classification decode path does not depend on iat.perturb, so generated "
              f"captions and classification outputs are unaffected. GAN/perturbation-branch SSIM "
              f"metrics (ssim_iat, ssim_gan) from this run should be treated as unreliable/skipped.")
    model.eval()

    dataset_sources = args.dataset_sources.split(",") if args.dataset_sources else None
    test_loader = build_test_loader(vocab, args.batch_size, args.limit,
                                      dataset_sources=dataset_sources,
                                      roco_manifest_override=args.roco_manifest_override,
                                      rocov2_manifest_override=args.rocov2_manifest_override)

    all_results = []
    preds = {"binary": [], "3class": [], "5class": []}
    targets_acc = {"binary": [], "3class": [], "5class": []}
    probs_acc = {"binary": [], "3class": [], "5class": []}

    cap_hyps, cap_refs = [], []
    sum_hyps, sum_refs = [], []
    ssim_iat, ssim_gan = [], []
    fid_real_imgs, fid_fake_imgs = [], []

    grid_images, grid_texts = [], []
    qualitative_rows = []

    bos_id, eos_id = vocab.word2idx[BOS], vocab.word2idx[EOS]

    for batch in test_loader:
        images = batch["image"].to(device)
        if args.noise_std > 0:
            images = torch.clamp(images + torch.randn_like(images) * args.noise_std, 0.0, 1.0)
        caption_ids = batch["caption_ids"].to(device)
        summary_ids = batch["summary_ids"].to(device)
        caption_mask = (caption_ids != pad_id).long()
        label_5class = batch["label_5class"].to(device)

        if args.mask_caption:
            # <bos><eos><pad>...<pad> -- no real keyword content reaches the
            # caption encoder, so classification can't shortcut through it.
            b, L = caption_ids.shape
            enc_caption_ids = torch.full((b, L), pad_id, dtype=caption_ids.dtype, device=device)
            enc_caption_ids[:, 0] = bos_id
            enc_caption_ids[:, 1] = eos_id
            enc_caption_mask = torch.zeros((b, L), dtype=caption_mask.dtype, device=device)
            enc_caption_mask[:, :2] = 1
        else:
            enc_caption_ids, enc_caption_mask = caption_ids, caption_mask

        out = model.generate(images, enc_caption_ids, enc_caption_mask, label_5class,
                              beam_width=args.beam_width, apply_iat=apply_iat,
                              disable_lstm=disable_lstm, disable_gan=disable_gan)

        if len(qualitative_rows) < args.qualitative_samples:
            with torch.no_grad():
                greedy_out = model.generate(images, enc_caption_ids, enc_caption_mask, label_5class,
                                             beam_width=1, apply_iat=apply_iat,
                                             disable_lstm=disable_lstm, disable_gan=disable_gan)
            for i in range(images.size(0)):
                if len(qualitative_rows) >= args.qualitative_samples:
                    break
                qualitative_rows.append({
                    "id": batch["id"][i],
                    "reference": vocab.decode(caption_ids[i].tolist()),
                    "greedy_caption": vocab.decode(greedy_out["caption_ids"][i].tolist()),
                    "beam_caption": vocab.decode(out["caption_ids"][i].tolist()),
                })

        has_image_mask = batch["has_image"]
        if has_image_mask.any():
            img_idx = has_image_mask.nonzero(as_tuple=True)[0]
            real_imgs = images[img_idx]
            if apply_iat and out["x_aug"] is not None:
                ssim_iat.append(ssim_batch(real_imgs, out["x_aug"][img_idx]))
            if not disable_gan and out["x_fake"] is not None:
                ssim_gan.append(ssim_batch(real_imgs, out["x_fake"][img_idx]))
                # Cap collected samples for FID (Inception features are not
                # free) -- ~500 real/500 fake is enough for a stable Frechet
                # distance estimate without re-running the whole test split.
                if sum(t.size(0) for t in fid_real_imgs) < 500:
                    fid_real_imgs.append(real_imgs.detach().cpu())
                    fid_fake_imgs.append(out["x_fake"][img_idx].detach().cpu())

        for i in range(images.size(0)):
            caption_text = vocab.decode(out["caption_ids"][i].tolist())
            summary_text = vocab.decode(out["summary_ids"][i].tolist())
            # Bug fix (2026-11): this used to read out["class_pred_binary"]
            # etc. -- the OLD Z-based classifier heads inside LMMDecoder,
            # which CompositeLoss (training/losses.py) never trains (only
            # img_class_logits_* receive the classification loss; see the
            # comment on IATGAMModel.img_classifier_* for why). Those heads
            # stay at random init the whole run, so they collapse to one
            # fixed class -- this was producing a below-random-chance
            # "classification accuracy" purely from evaluating an untrained
            # head, not a real model failure. Reading the correct, actually
            # trained img_class_pred_* fields instead.
            pred_binary = int(out["img_class_pred_binary"][i].item())
            pred_3class = int(out["img_class_pred_3class"][i].item())
            pred_5class = int(out["img_class_pred_5class"][i].item())
            prob_binary = torch.softmax(out["img_class_logits_binary"][i], dim=-1).detach().cpu().numpy()
            prob_3class = torch.softmax(out["img_class_logits_3class"][i], dim=-1).detach().cpu().numpy()
            prob_5class = torch.softmax(out["img_class_logits_5class"][i], dim=-1).detach().cpu().numpy()
            probs_acc["binary"].append(prob_binary)
            probs_acc["3class"].append(prob_3class)
            probs_acc["5class"].append(prob_5class)

            gt_caption_text = vocab.decode(caption_ids[i].tolist())
            gt_summary_text = vocab.decode(summary_ids[i].tolist())

            result = {
                "id": batch["id"][i],
                "has_image": bool(batch["has_image"][i]),
                "gt_caption": gt_caption_text,
                "gen_caption": caption_text,
                "gt_summary": gt_summary_text,
                "gen_summary": summary_text,
                "gt_label_binary": int(batch["label_binary"][i].item()),
                "pred_label_binary": pred_binary,
                "gt_label_3class": int(batch["label_3class"][i].item()),
                "pred_label_3class": pred_3class,
                "gt_label_5class": int(batch["label_5class"][i].item()),
                "pred_label_5class": pred_5class,
            }
            all_results.append(result)

            cap_hyps.append(tokenize(caption_text))
            cap_refs.append(tokenize(gt_caption_text))
            sum_hyps.append(tokenize(summary_text))
            sum_refs.append(tokenize(gt_summary_text))

            preds["binary"].append(pred_binary)
            preds["3class"].append(pred_3class)
            preds["5class"].append(pred_5class)
            targets_acc["binary"].append(result["gt_label_binary"])
            targets_acc["3class"].append(result["gt_label_3class"])
            targets_acc["5class"].append(result["gt_label_5class"])

            if result["has_image"] and len(grid_images) < args.num_samples:
                grid_images.append(images[i].detach().cpu())
                grid_texts.append(
                    f"gen: {caption_text[:60]}\n"
                    f"5cls: {_LABEL_NAMES_5CLASS[pred_5class]} "
                    f"(gt: {_LABEL_NAMES_5CLASS[result['gt_label_5class']]})\n"
                    f"3cls: {_LABEL_NAMES_3CLASS[pred_3class]} | "
                    f"bin: {_LABEL_NAMES_BINARY[pred_binary]}"
                )

    # ------------------------------------------------------------------
    # Classification metrics over the full test split (binary/3-class/5-class)
    # ------------------------------------------------------------------
    metrics = {"classification": {}}
    _NUM_CLASSES = {"binary": 2, "3class": 3, "5class": 5}
    for granularity in ("binary", "3class", "5class"):
        n_cls = _NUM_CLASSES[granularity]
        y_pred = np.array(preds[granularity])
        y_true = np.array(targets_acc[granularity])
        y_prob = np.stack(probs_acc[granularity], axis=0)
        granularity_metrics = classification_metrics(y_pred, y_true, num_classes=n_cls)
        granularity_metrics.update(confusion_matrix_report(y_pred, y_true, num_classes=n_cls))
        granularity_metrics["auroc_macro_ovr"] = auroc_multiclass(y_prob, y_true)
        metrics["classification"][granularity] = granularity_metrics

    # ------------------------------------------------------------------
    # Captioning metrics (Eq. 3.5/3.28a, Table 5.x): BLEU-1/4, ROUGE-L, CIDEr-D
    # ------------------------------------------------------------------
    # BERTScore (2026-11 addition, per standing metrics checklist): credits
    # a medically-correct caption phrased differently from the reference,
    # which the n-gram metrics above penalize. Uses the raw (untokenized)
    # strings already stored in all_results.
    cap_hyp_text = [r["gen_caption"] for r in all_results]
    cap_ref_text = [r["gt_caption"] for r in all_results]
    sum_hyp_text = [r["gen_summary"] for r in all_results]
    sum_ref_text = [r["gt_summary"] for r in all_results]

    metrics["captioning"] = {
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
            "bert_score_f1": bert_score_f1(sum_hyp_text, sum_ref_text),
        },
    }
    metrics["decoding"] = {"beam_width": args.beam_width}

    # ------------------------------------------------------------------
    # Generative quality (IAT augmentation / GAN outputs): SSIM (Eq. 3.10-3.14)
    # + FID (real substitute for FVD -- see evaluation/metrics.py:fid_score
    # docstring for why FVD itself cannot be computed on this codebase's
    # video branch)
    # ------------------------------------------------------------------
    fid_value = None
    if fid_real_imgs and fid_fake_imgs:
        try:
            fid_value = fid_score(torch.cat(fid_real_imgs).to(device), torch.cat(fid_fake_imgs).to(device))
        except Exception as e:
            print(f"[warn] FID computation failed ({e}); reporting null rather than a fabricated value.")
    metrics["generative"] = {
        "ssim_iat_vs_original": float(np.mean(ssim_iat)) if ssim_iat else 0.0,
        "ssim_gan_vs_original": float(np.mean(ssim_gan)) if ssim_gan else 0.0,
        "fid_gan_vs_real": fid_value,
        "fvd": None,  # not computable: no real video frames in this codebase's data (Limitations)
    }

    # ------------------------------------------------------------------
    # Efficiency: latency (ms/sample) and throughput (samples/sec)
    # ------------------------------------------------------------------
    eff_batch = next(iter(test_loader))
    eff_images = eff_batch["image"].to(device)
    eff_caption_ids = eff_batch["caption_ids"].to(device)
    eff_caption_mask = (eff_caption_ids != pad_id).long()
    eff_label_5class = eff_batch["label_5class"].to(device)

    def _forward():
        model.generate(eff_images, eff_caption_ids, eff_caption_mask, eff_label_5class,
                        apply_iat=apply_iat, disable_lstm=disable_lstm, disable_gan=disable_gan)

    eff = measure_latency_throughput(model, _forward, n_runs=10)
    batch_n = eff_images.size(0)
    metrics["efficiency"] = {
        "batch_size": batch_n,
        "ms_per_sample": eff["seconds_per_run"] * 1000.0 / batch_n,
        "samples_per_sec": batch_n / eff["seconds_per_run"],
    }
    metrics["ablation"] = ablation
    metrics["noise_std"] = args.noise_std
    metrics["mask_caption"] = args.mask_caption

    metrics_path = args.metrics_out or os.path.join(config.LOGS_DIR, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote full results metrics -> {metrics_path}")
    print(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------
    # Per-sample generated captions/summaries/labels
    # ------------------------------------------------------------------
    results_path = os.path.join(config.LOGS_DIR, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Wrote {len(all_results)} per-sample results -> {results_path}")

    # ------------------------------------------------------------------
    # Qualitative greedy-vs-beam comparison on a fixed sample set
    # ------------------------------------------------------------------
    qual_path = os.path.join(config.LOGS_DIR, "qualitative_greedy_vs_beam.json")
    with open(qual_path, "w", encoding="utf-8") as f:
        json.dump(qualitative_rows, f, indent=2)
    print(f"Wrote {len(qualitative_rows)} qualitative greedy-vs-beam samples -> {qual_path}")

    # ------------------------------------------------------------------
    # Qualitative image grid
    # ------------------------------------------------------------------
    if grid_images:
        n = len(grid_images)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4.5))
        axes = np.array(axes).reshape(-1)
        for i in range(rows * cols):
            ax = axes[i]
            ax.axis("off")
            if i < n:
                img = grid_images[i].permute(1, 2, 0).numpy()
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(grid_texts[i], fontsize=8)
        plt.tight_layout()
        grid_path = os.path.join(config.PLOTS_DIR, "results_grid.png")
        plt.savefig(grid_path, dpi=120)
        plt.close(fig)
        print(f"Wrote qualitative result grid -> {grid_path}")


if __name__ == "__main__":
    main()

"""
Computational-complexity profiling for IAT-GAM-LSTM (Section 5.7).

Reports, per module (IAT, GAN generator/discriminator, CNN backbone,
Transformer caption encoder, LSTM temporal module, LMM decoder) and for the
full model:

  - parameter count (total / trainable)
  - FLOPs for one forward pass (via torch.profiler, batch size 1)
  - peak GPU memory during a forward+backward step
  - training time per epoch (read from outputs/logs/train_log_<hparams>.csv
    if available, else measured directly)
  - inference latency / throughput at a sweep of batch sizes

Writes outputs/logs/complexity_metrics.json.

Usage:
    python evaluation/profile_complexity.py --hparams aug_strong
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn

import config
from data.vocab import PAD, Vocabulary, VOCAB_PATH
from models.iatgam_model import IATGAMModel


def count_params(module: nn.Module) -> dict:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def estimate_flops(model: IATGAMModel, sample: dict, device: torch.device) -> float:
    """Single-forward FLOPs (batch size 1) via torch.profiler."""
    model.eval()
    with torch.profiler.profile(with_flops=True) as prof:
        with torch.no_grad():
            model(sample["image"], sample["caption_ids"], sample["caption_mask"],
                  sample["label_5class"], sample["caption_input"], sample["summary_input"])
    total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
    return float(total_flops)


def make_dummy_batch(model_cfg: config.ModelConfig, batch_size: int, device: torch.device, pad_id: int) -> dict:
    images = torch.rand(batch_size, 3, *config.DEFAULT_IMAGE_SIZE, device=device)
    L = model_cfg.max_caption_len
    caption_ids = torch.randint(low=3, high=model_cfg.vocab_size, size=(batch_size, L), device=device)
    caption_ids[:, 0] = pad_id
    caption_mask = (caption_ids != pad_id).long()
    label_5class = torch.randint(0, model_cfg.num_classes_5class, (batch_size,), device=device)
    caption_input = torch.roll(caption_ids, shifts=1, dims=1)
    summary_input = torch.roll(caption_ids, shifts=1, dims=1)
    return {
        "image": images, "caption_ids": caption_ids, "caption_mask": caption_mask,
        "label_5class": label_5class, "caption_input": caption_input, "summary_input": summary_input,
    }


def measure_latency(model: IATGAMModel, model_cfg: config.ModelConfig, device: torch.device,
                     pad_id: int, batch_sizes: list[int], n_runs: int = 10) -> dict:
    model.eval()
    results = {}
    for bs in batch_sizes:
        batch = make_dummy_batch(model_cfg, bs, device, pad_id)
        with torch.no_grad():
            for _ in range(3):  # warmup
                model.generate(batch["image"], batch["caption_ids"], batch["caption_mask"], batch["label_5class"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                model.generate(batch["image"], batch["caption_ids"], batch["caption_mask"], batch["label_5class"])
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        per_run = elapsed / n_runs
        results[str(bs)] = {
            "ms_per_sample": per_run * 1000.0 / bs,
            "samples_per_sec": bs / per_run,
        }
    return results


def measure_peak_memory(model: IATGAMModel, model_cfg: config.ModelConfig, device: torch.device,
                          pad_id: int, batch_size: int, criterion) -> float:
    if device.type != "cuda":
        return 0.0
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    batch = make_dummy_batch(model_cfg, batch_size, device, pad_id)
    out = model(batch["image"], batch["caption_ids"], batch["caption_mask"],
                 batch["label_5class"], batch["caption_input"], batch["summary_input"])
    gan_loss = model.gan.discriminator_loss(out["gan_out"]["d_real"], out["gan_out"]["d_fake"]) \
        + model.gan.generator_loss(out["gan_out"]["g_fake"])
    temporal_loss = model.lstm.temporal_consistency_loss(out["h_seq"])
    targets = {
        "caption_targets": batch["caption_ids"], "summary_targets": batch["caption_ids"],
        "class_targets_binary": batch["label_5class"].clamp(max=1),
        "class_targets_3class": batch["label_5class"].clamp(max=2),
        "class_targets_5class": batch["label_5class"],
    }
    losses = criterion(out, targets, gan_loss=gan_loss, temporal_loss=temporal_loss)
    losses["total"].backward()
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB


def epoch_time_from_log(hparams: str) -> dict:
    log_path = os.path.join(config.LOGS_DIR, f"train_log_{hparams}.csv")
    if not os.path.exists(log_path):
        return {}
    import csv
    rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
    return {"n_epochs_logged": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default="aug_strong")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 16, 32])
    args = parser.parse_args()

    device = config.get_device()
    vocab = Vocabulary.load(VOCAB_PATH)
    pad_id = vocab.word2idx[PAD]

    hp = config.AUG_STRONG_HPARAMS if args.hparams == "aug_strong" else (
        config.TUNED_HPARAMS if args.hparams == "tuned" else config.BASELINE_HPARAMS)
    model_cfg = config.ModelConfig(
        vocab_size=len(vocab), embedding_dim=hp.embedding_dim, lstm_hidden=hp.lstm_hidden,
        dropout=hp.dropout, iat_alpha_init=hp.iat_alpha_init, iat_ssim_threshold=hp.iat_ssim_threshold,
    )
    model = IATGAMModel(model_cfg).to(device)

    # ------------------------------------------------------------
    # Parameter counts per module
    # ------------------------------------------------------------
    module_params = {
        "iat": count_params(model.iat),
        "gan_generator": count_params(model.gan.generator),
        "gan_discriminator": count_params(model.gan.discriminator),
        "cnn_backbone": count_params(model.cnn),
        "caption_encoder": count_params(model.caption_encoder),
        "fusion": count_params(model.fusion),
        "lstm": count_params(model.lstm),
        "state_builder": count_params(model.state_builder),
        "lmm_decoder": count_params(model.lmm),
        "full_model": count_params(model),
    }

    # ------------------------------------------------------------
    # FLOPs (batch size 1, forward only)
    # ------------------------------------------------------------
    sample = make_dummy_batch(model_cfg, 1, device, pad_id)
    try:
        flops = estimate_flops(model, sample, device)
    except Exception as e:  # profiler FLOPs counting unsupported for some ops
        flops = None
        print(f"FLOPs estimation failed: {e}")

    # ------------------------------------------------------------
    # Inference latency / throughput sweep
    # ------------------------------------------------------------
    latency = measure_latency(model, model_cfg, device, pad_id, args.batch_sizes)

    # ------------------------------------------------------------
    # Peak training memory (forward + backward, batch size = hp.batch_size)
    # ------------------------------------------------------------
    from training.losses import CompositeLoss
    criterion = CompositeLoss(lambda_gan=hp.lambda_gan, lambda_temporal=hp.lambda_temporal, pad_id=pad_id)
    peak_mem_mb = measure_peak_memory(model, model_cfg, device, pad_id, hp.batch_size, criterion)

    results = {
        "hparams": args.hparams,
        "device": str(device),
        "module_parameters": module_params,
        "flops_per_sample": flops,
        "inference_latency": latency,
        "peak_train_memory_mb": peak_mem_mb,
        "train_batch_size": hp.batch_size,
        **epoch_time_from_log(args.hparams),
    }

    out_path = os.path.join(config.LOGS_DIR, "complexity_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote complexity metrics -> {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

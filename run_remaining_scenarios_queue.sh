#!/bin/bash
# Remaining pending-scenarios queue (2026-11), run unattended and sequentially
# on the single shared GPU. Every step here waits for ANY python process
# (including unrelated ones on this machine) to clear before using the GPU,
# per the single-GPU/no-parallel-jobs rule.
#
# Order:
#   0. Real eval pass (confusion matrix + AUROC + BERTScore) on the winning
#      Case3 (binary_weight_power=0.75) checkpoint -- quick, inference only.
#   1. Phase D ablation: CNN-only, CNN+GAN, CNN+IAT (3 runs), all with the
#      same binary_weight_power=0.75 fix, same seed, same 20 epochs. (The
#      4th config, CNN+GAN+IAT, is already Case3 -- not re-run.)
#   2. Augmentation comparison: mixup, cutmix, randaugment, cropflip (4 runs)
#      vs IAT, matching the original Phase-2 design (5-class, apply_iat=False
#      for all four, isolating "any augmentation" from IAT specifically).
#   3. Multi-seed: 4 additional seeds (43,44,45,46) of the full proposed
#      model (CNN+GAN+IAT, binary_weight_power=0.75) for mean/std reporting.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE

wait_for_gpu_free() {
  while true; do
    alive=$(powershell -Command "(Get-Process python -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r')
    if [ "$alive" = "0" ] || [ -z "$alive" ]; then break; fi
    sleep 60
  done
}

wait_for_csv() {
  local csv="$1"
  while true; do
    if [ -f "$csv" ]; then
      n=$(wc -l < "$csv")
      if [ "$n" -ge 21 ]; then break; fi
    fi
    alive=$(powershell -Command "(Get-Process python -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r')
    if [ "$alive" = "0" ] || [ -z "$alive" ]; then
      echo "WARNING: $csv not complete and no python process alive -- proceeding anyway"
      break
    fi
    sleep 60
  done
}

echo "=== Waiting for GPU to be free (other python jobs currently running) ==="
wait_for_gpu_free

echo "=== Step 0: real eval pass on Case3 checkpoint (confusion matrix + AUROC + BERTScore) ==="
python evaluation/generate_results.py \
  --checkpoint outputs/model/iatgam_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_best.pt \
  --beam_width 1 --num_samples 8 --qualitative_samples 8 \
  > outputs/logs/case3_eval_full.log 2>&1 || echo "Step 0 FAILED, see outputs/logs/case3_eval_full.log"
wait_for_gpu_free

echo "=== Step 1a: Phase D -- CNN-only (disable_gan + disable_iat) ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --disable_gan --disable_iat \
  --epochs 20 --seed 42 --num_workers 2
wait_for_csv "outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_noGAN_noIAT.csv"

echo "=== Step 1b: Phase D -- CNN+GAN (disable_iat only) ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --disable_iat \
  --epochs 20 --seed 42 --num_workers 2
wait_for_csv "outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_noIAT.csv"

echo "=== Step 1c: Phase D -- CNN+IAT (disable_gan only) ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --disable_gan \
  --epochs 20 --seed 42 --num_workers 2
wait_for_csv "outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_noGAN.csv"

echo "=== Step 2a: augmentation comparison -- mixup ==="
python training/train_augment_baseline.py --aug_method mixup --epochs 20 --seed 42
echo "=== Step 2b: augmentation comparison -- cutmix ==="
python training/train_augment_baseline.py --aug_method cutmix --epochs 20 --seed 42
echo "=== Step 2c: augmentation comparison -- randaugment ==="
python training/train_augment_baseline.py --aug_method randaugment --epochs 20 --seed 42
echo "=== Step 2d: augmentation comparison -- cropflip ==="
python training/train_augment_baseline.py --aug_method cropflip --epochs 20 --seed 42

echo "=== Step 3a: multi-seed -- seed 43 ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --epochs 20 --seed 43 --num_workers 2
echo "=== Step 3b: multi-seed -- seed 44 ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --epochs 20 --seed 44 --num_workers 2
echo "=== Step 3c: multi-seed -- seed 45 ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --epochs 20 --seed 45 --num_workers 2
echo "=== Step 3d: multi-seed -- seed 46 ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --epochs 20 --seed 46 --num_workers 2

echo "=== ALL REMAINING SCENARIOS DONE ==="

#!/bin/bash
# Properly isolated single-variable test (2026-11): the real known-good
# baseline for binary classification (0.697 acc / 0.516 F1) came from the
# FULL JOINT model (captioning+GAN+LSTM all active, --soft_class_weights,
# no binary-aware sampler). Every "fix" attempt since then used
# --disable_lstm, a different base config -- confounding the comparison.
# This waits for the currently-running disable_lstm sampler test to
# finish, then re-runs the EXACT original baseline command with only
# --focal_gamma 2.0 added, so the comparison against 0.697/0.516 is
# actually valid (one variable changed, not several at once).
set -e
cd "$(dirname "$0")"

CSV1="outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_balcls_noLSTM.csv"
echo "=== Waiting for the currently-running disable_lstm sampler test to finish ==="
while true; do
  n=$(wc -l < "$CSV1" 2>/dev/null || echo 0)
  if [ "$n" -ge 21 ]; then
    break
  fi
  alive=$(powershell -Command "(Get-Process python -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r')
  if [ "$alive" = "0" ] || [ -z "$alive" ]; then
    echo "WARNING: no python process alive and run not complete -- proceeding anyway"
    break
  fi
  sleep 60
done
echo "=== Starting isolated focal-loss test from the real known-good baseline ==="

PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE python training/train.py \
  --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --balance_classes --focal_gamma 2.0 \
  --epochs 20 --seed 42 --num_workers 2

echo "=== ISOLATED FOCAL TEST COMPLETE ==="
# Resulting files: outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_softcw_balcls_focal2.0.csv

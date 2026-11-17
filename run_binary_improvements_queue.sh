#!/bin/bash
# Sequential queue for the binary-classification-improvement validation
# runs (2026-11): waits for the already-running micro-capacity fix-
# validation run to finish, then runs the macro-capacity + focal-loss
# variant, so both runs never compete for the single GPU.
set -e
cd "$(dirname "$0")"

CSV1="outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_balcls_noLSTM.csv"
echo "=== Waiting for the already-running micro-capacity fix-validation run to finish ==="
while true; do
  if [ -f "$CSV1" ]; then
    n=$(wc -l < "$CSV1")
    if [ "$n" -ge 21 ]; then
      break
    fi
  fi
  if ! powershell -Command "(Get-Process python -ErrorAction SilentlyContinue).Count" | grep -qv "^0$"; then
    echo "WARNING: no python process alive and run not complete -- check for a crash before continuing"
    break
  fi
  sleep 60
done
echo "=== Micro-capacity run done, starting macro-capacity + focal-loss run ==="

PYTHONPATH=. KMP_DUPLICATE_LIB_OK=TRUE python training/train.py \
  --hparams aug_strong --macro_capacity --attn_entropy_weight 1.0 \
  --balance_classes --augment_train --focal_gamma 2.0 --disable_lstm \
  --epochs 20 --seed 42 --num_workers 2

echo "=== QUEUE COMPLETE ==="

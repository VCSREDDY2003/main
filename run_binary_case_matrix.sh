#!/bin/bash
# Full binary-classification fix matrix (2026-11): one variable changed per
# case vs. the real baseline, run sequentially on the single GPU. No manual
# status checks needed -- this script itself waits for each case to finish
# before starting the next, and a trailing python call prints one
# consolidated final-epoch comparison table across every case once the
# whole matrix is done.
#
# Real baseline (already run, not repeated here):
#   --soft_class_weights only (binary weight power=0.5), no sampler
#   -> val binary acc=0.697 F1=0.516 recall_cls1=14.4%
#
# Case already running when this script starts (binpow0.25, power=0.25,
# no sampler) is left to finish on its own -- early epochs already show it
# drifting toward majority-collapse (recall_cls1 bouncing 0.6-22%), the
# wrong direction, but it's cheap to let finish for a complete record.
#
# Cases this script runs, in order, each a SINGLE variable change from the
# plain baseline (no stacking two correction mechanisms at once, per the
# 2026-11 double-counting lesson):
#   Case 2: sampler only         (--balance_classes, no soft_class_weights)
#   Case 3: stronger weight only (--soft_class_weights --binary_weight_power 0.75, no sampler)
#   Case 4: full hard weight only (--binary_weight_power 1.0, no sampler)
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE

wait_for() {
  local csv="$1"
  while true; do
    if [ -f "$csv" ]; then
      n=$(wc -l < "$csv")
      if [ "$n" -ge 21 ]; then break; fi
    fi
    alive=$(powershell -Command "(Get-Process python -ErrorAction SilentlyContinue).Count" 2>/dev/null | tr -d '\r')
    if [ "$alive" = "0" ] || [ -z "$alive" ]; then
      echo "WARNING: $csv not complete (rows=$n) and no python process alive -- proceeding anyway"
      break
    fi
    sleep 60
  done
}

echo "=== Waiting for in-progress binpow0.25 run to finish ==="
wait_for "outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.25.csv"

echo "=== Case 2: sampler only (balance_classes, binary loss-weight neutralized to isolate the sampler's effect; 3class/5class keep soft weight, matching baseline) ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.0 --balance_classes --epochs 20 --seed 42 --num_workers 2
wait_for "outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_softcw_balcls_binpow0.0.csv"

echo "=== Case 3: stronger weight only (binary_weight_power=0.75) ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --epochs 20 --seed 42 --num_workers 2
wait_for "outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75.csv"

echo "=== Case 4: full hard weight only (binary_weight_power=1.0) ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 1.0 --epochs 20 --seed 42 --num_workers 2
wait_for "outputs/logs/train_log_aug_strong_seed42_micro2m_ae1.0_softcw_binpow1.0.csv"

echo "=== MATRIX COMPLETE -- building consolidated comparison table ==="
python scripts/summarize_binary_matrix.py

echo "=== ALL CASES DONE ==="

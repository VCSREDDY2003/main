#!/bin/bash
# Retry Step 0 (full eval pass on the winning Case3 checkpoint) after the
# bert_score_f1 fix (2026-11: bert_score pkg incompatible with
# transformers>=5, replaced with a direct transformers implementation).
# Waits for the main remaining-scenarios queue to fully finish before using
# the GPU, so it never competes with the Phase D / augmentation / seed runs.
set -e
cd "$(dirname "$0")"
export PYTHONPATH=.
export KMP_DUPLICATE_LIB_OK=TRUE

while true; do
  if grep -q "ALL REMAINING SCENARIOS DONE" outputs/logs/remaining_scenarios_queue.log 2>/dev/null; then
    break
  fi
  sleep 120
done

echo "=== Retrying Step 0: real eval pass on Case3 checkpoint (bert_score_f1 fixed) ==="
python evaluation/generate_results.py \
  --checkpoint outputs/model/iatgam_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_best.pt \
  --beam_width 1 --num_samples 8 --qualitative_samples 8 \
  > outputs/logs/case3_eval_full_retry.log 2>&1
cp outputs/logs/test_metrics.json outputs/logs/case3_full_test_metrics.json
echo "=== CASE3 EVAL RETRY DONE ==="

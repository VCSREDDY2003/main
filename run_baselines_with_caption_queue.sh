#!/usr/bin/env bash
set -e
cd "C:\Users\kalya\PD\IMAGECAPTION CODE\conference\proposed_model\new_IAT" 2>/dev/null || cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

for arch in resnet_small efficientnet_small mobilenet_style shufflenet_style squeezenet_style densenet_style; do
  echo "=== TRAIN+EVAL (with caption head): ${arch} (combined) ==="
  python -u feature_pipeline/train_baseline_with_caption.py \
    --arch "$arch" --epochs 50 --seed 42 --num_workers 2 --batch_size 32
done

echo "=== baselines-with-caption queue complete ==="

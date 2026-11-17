#!/usr/bin/env bash
set -e
cd "C:\Users\kalya\PD\IMAGECAPTION CODE\conference\proposed_model\new_IAT" 2>/dev/null || cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

ROCO_M=feature_pipeline/data/roco_manifest_4class_balanced.csv
ROCOV2_M=feature_pipeline/data/rocov2_manifest_4class_balanced.csv

echo "=== CNN+IATnew micro (no GAN) - ROCOv2 ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --disable_gan \
  --dataset_sources rocov2 --rocov2_manifest_override $ROCOV2_M \
  --epochs 50 --seed 42 --num_workers 2

echo "=== CNN+IATnew micro (no GAN) - combined ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --disable_gan \
  --dataset_sources roco,rocov2 --roco_manifest_override $ROCO_M --rocov2_manifest_override $ROCOV2_M \
  --epochs 50 --seed 42 --num_workers 2

echo "=== IAT+GAM (GAN + memory coupling) - ROCOv2 ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --gam_memory_coupling \
  --dataset_sources rocov2 --rocov2_manifest_override $ROCOV2_M \
  --epochs 50 --seed 42 --num_workers 2

echo "=== IAT+GAM (GAN + memory coupling) - combined ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --gam_memory_coupling \
  --dataset_sources roco,rocov2 --roco_manifest_override $ROCO_M --rocov2_manifest_override $ROCOV2_M \
  --epochs 50 --seed 42 --num_workers 2

echo "=== cross-dataset queue complete ==="

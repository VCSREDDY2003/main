#!/usr/bin/env bash
set -e
cd "C:\Users\kalya\PD\IMAGECAPTION CODE\conference\proposed_model\new_IAT" 2>/dev/null || cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

MANIFEST=feature_pipeline/data/roco_manifest_4class_balanced.csv

echo "=== IAT + light CNN (no GAN) ==="
python training/train.py --hparams aug_strong --light_cnn --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --disable_gan \
  --dataset_sources roco --roco_manifest_override $MANIFEST \
  --epochs 50 --seed 42 --num_workers 2

echo "=== IAT + deep/macro CNN (no GAN) ==="
python training/train.py --hparams aug_strong --macro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --disable_gan \
  --dataset_sources roco --roco_manifest_override $MANIFEST \
  --epochs 50 --seed 42 --num_workers 2

echo "=== IAT + GAM (GAN + memory coupling) ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --gam_memory_coupling \
  --dataset_sources roco --roco_manifest_override $MANIFEST \
  --epochs 50 --seed 42 --num_workers 2

echo "=== all 3 IAT variant runs complete ==="

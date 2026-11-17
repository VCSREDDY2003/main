#!/usr/bin/env bash
set -e
cd "C:\Users\kalya\PD\IMAGECAPTION CODE\conference\proposed_model\new_IAT" 2>/dev/null || cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

echo "=== Phase 2a: CNN+GAN(featgan) -- disable_iat + gan_feature_space ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --disable_iat --gan_feature_space \
  --epochs 20 --seed 42 --num_workers 2

echo "=== Phase 2b: CNN+GAN+IAT(featgan) -- full model, gan_feature_space ==="
python training/train.py --hparams aug_strong --micro_capacity --attn_entropy_weight 1.0 \
  --soft_class_weights --binary_weight_power 0.75 --gan_feature_space \
  --epochs 20 --seed 42 --num_workers 2

echo "=== Phase 2 ablation training complete ==="

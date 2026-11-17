#!/usr/bin/env bash
set -e
cd "C:\Users\kalya\PD\IMAGECAPTION CODE\conference\proposed_model\new_IAT" 2>/dev/null || cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=.

ROCO_M=feature_pipeline/data/roco_manifest_4class_balanced.csv
ROCOV2_M=feature_pipeline/data/rocov2_manifest_4class_balanced.csv

echo "=== EVAL: IAT+lightCNN - ROCO v1 ==="
python evaluation/generate_results.py \
  --checkpoint outputs/model/iatgam_aug_strong_seed42_lightcnn_ae1.0_softcw_binpow0.75_noGAN_best.pt \
  --dataset_sources roco --roco_manifest_override $ROCO_M \
  --metrics_out outputs/logs/test_metrics_lightcnn_roco.json \
  --num_samples 8 --qualitative_samples 5

echo "=== EVAL: IAT+macroCNN(deep) - ROCO v1 ==="
python evaluation/generate_results.py \
  --checkpoint outputs/model/iatgam_aug_strong_seed42_macro13m_ae1.0_softcw_binpow0.75_noGAN_best.pt \
  --dataset_sources roco --roco_manifest_override $ROCO_M \
  --metrics_out outputs/logs/test_metrics_macrocnn_roco.json \
  --num_samples 8 --qualitative_samples 5

echo "=== EVAL: CNN+IATnew (micro, noGAN) - combined ==="
python evaluation/generate_results.py \
  --checkpoint outputs/model/iatgam_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_noGAN_best.pt \
  --dataset_sources roco,rocov2 --roco_manifest_override $ROCO_M --rocov2_manifest_override $ROCOV2_M \
  --metrics_out outputs/logs/test_metrics_iatnew_combined.json \
  --num_samples 8 --qualitative_samples 5

echo "=== EVAL: IAT+GAM (micro, gammem) - combined ==="
python evaluation/generate_results.py \
  --checkpoint outputs/model/iatgam_aug_strong_seed42_micro2m_ae1.0_softcw_binpow0.75_gammem_best.pt \
  --dataset_sources roco,rocov2 --roco_manifest_override $ROCO_M --rocov2_manifest_override $ROCOV2_M \
  --metrics_out outputs/logs/test_metrics_gam_combined.json \
  --num_samples 8 --qualitative_samples 5

echo "=== full metrics eval queue complete ==="

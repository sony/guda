#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

DATA_ROOT="${DATA_ROOT:-data/UnlearnCanvas}"
PROMPTS="${PROMPTS:-data/train_prompts_ffsd_very_relaxed.jsonl}"
ALLCLASS="${ALLCLASS:-outputs/allclass_sd15_ffsd_very_relaxed/checkpoint_step_10000.pt}"
PROTO="${PROTO:-cache/clip_style_prototypes_very_relaxed.pt}"
DESCRIPTORS="${DESCRIPTORS:-data/ffsd_very_relaxed_descriptors.json}"
STYLE="${STYLE:-Abstractionism}"

python machine_unlearning/retrack_latent/src/train_retrack_ffsd.py \
  --allclass_checkpoint "${ALLCLASS}" \
  --data_root "${DATA_ROOT}" \
  --prompts_file "${PROMPTS}" \
  --forget_style "${STYLE}" \
  --anchor_strategy weighted_style_select \
  --clip_prototypes_file "${PROTO}" \
  --descriptors_file "${DESCRIPTORS}" \
  --beta "${AWSS_BETA:-2.0}" \
  --eta_uniform "${AWSS_ETA:-0.3}" \
  --style_sampling_mode weighted \
  --num_anchor_descriptors "${ANCHOR_DESCRIPTORS:-3}" \
  --output_dir "outputs/ablations/anchor/${STYLE}/awss" \
  --num_steps "${UNLEARN_STEPS:-5000}" \
  --learning_rate "${UNLEARN_LR:-2e-6}"

python machine_unlearning/retrack_latent/src/train_retrack_ffsd.py \
  --allclass_checkpoint "${ALLCLASS}" \
  --data_root "${DATA_ROOT}" \
  --prompts_file "${PROMPTS}" \
  --forget_style "${STYLE}" \
  --anchor_strategy weighted_style_select \
  --clip_prototypes_file "${PROTO}" \
  --descriptors_file "${DESCRIPTORS}" \
  --beta "${AWSS_BETA:-2.0}" \
  --eta_uniform "${AWSS_ETA:-0.3}" \
  --style_sampling_mode uniform \
  --num_anchor_descriptors "${ANCHOR_DESCRIPTORS:-3}" \
  --output_dir "outputs/ablations/anchor/${STYLE}/uniform" \
  --num_steps "${UNLEARN_STEPS:-5000}" \
  --learning_rate "${UNLEARN_LR:-2e-6}"

python machine_unlearning/retrack_latent/src/train_retrack_ffsd.py \
  --allclass_checkpoint "${ALLCLASS}" \
  --data_root "${DATA_ROOT}" \
  --prompts_file "${PROMPTS}" \
  --forget_style "${STYLE}" \
  --anchor_strategy style_removed \
  --output_dir "outputs/ablations/anchor/${STYLE}/style_removed" \
  --num_steps "${UNLEARN_STEPS:-5000}" \
  --learning_rate "${UNLEARN_LR:-2e-6}"

#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

DATA_ROOT="${DATA_ROOT:-data/UnlearnCanvas}"
PROMPTS="${PROMPTS:-data/train_prompts_ffsd_very_relaxed.jsonl}"
EVAL_PROMPTS="${EVAL_PROMPTS:-data/eval_prompts_ffsd_very_relaxed.jsonl}"
ALLCLASS_DIR="${ALLCLASS_DIR:-outputs/allclass_sd15_ffsd_very_relaxed}"
ALLCLASS="${ALLCLASS:-${ALLCLASS_DIR}/checkpoint_step_10000.pt}"
LOGO_DIR="${LOGO_DIR:-outputs/logo_sd15_ffsd_very_relaxed}"
RETRACK_DIR="${RETRACK_DIR:-outputs/guda_awss}"
PROTO="${PROTO:-cache/clip_style_prototypes_very_relaxed.pt}"
DESCRIPTORS="${DESCRIPTORS:-data/ffsd_very_relaxed_descriptors.json}"
IMAGE_DIR="${IMAGE_DIR:-outputs/evaluation_images_ffsd_very_relaxed}"

DEFAULT_STYLES=(
  "Abstractionism"
  "Artist Sketch"
  "Blossom Season"
  "Blue Blooming"
  "Bricks"
  "Byzantine"
  "Cartoon"
  "Cold Warm"
  "Color Fantasy"
  "Comic Etch"
  "Crayon"
  "Crypto Punks"
  "Cubism"
  "Dadaism"
  "Dapple"
  "Defoliation"
)

if [[ -n "${STYLES:-}" ]]; then
  readarray -t STYLE_LIST < <(printf '%s\n' "${STYLES}" | tr ',' '\n' | sed 's/^ *//;s/ *$//' | sed '/^$/d')
else
  STYLE_LIST=("${DEFAULT_STYLES[@]}")
fi

python evaluation/precompute_clip_prototypes.py \
  --descriptors_file "${DESCRIPTORS}" \
  --output_file "${PROTO}" \
  --clip_model "${CLIP_MODEL:-openai/clip-vit-large-patch14}" \
  --device "${DEVICE:-cuda}"

python machine_unlearning/retrack_latent/src/train_allclass_prompts_full.py \
  --model_id "${MODEL_ID:-runwayml/stable-diffusion-v1-5}" \
  --data_root "${DATA_ROOT}" \
  --prompt_file "${PROMPTS}" \
  --output_dir "${ALLCLASS_DIR}" \
  --num_steps "${ALLCLASS_STEPS:-10000}" \
  --lr "${ALLCLASS_LR:-1e-6}"

for style in "${STYLE_LIST[@]}"; do
  python machine_unlearning/retrack_latent/src/train_logo_prompts_full.py \
    --model_id "${MODEL_ID:-runwayml/stable-diffusion-v1-5}" \
    --data_root "${DATA_ROOT}" \
    --prompt_file "${PROMPTS}" \
    --exclude_style "${style}" \
    --output_dir "${LOGO_DIR}/${style}" \
    --num_steps "${LOGO_STEPS:-10000}" \
    --lr "${LOGO_LR:-1e-6}"
done

python evaluation/generate_ffsd_evaluation_images.py \
  --allclass_dir "${ALLCLASS_DIR}" \
  --checkpoint_step "${ALLCLASS_STEP:-10000}" \
  --eval_prompts_file "${EVAL_PROMPTS}" \
  --output_dir "${IMAGE_DIR}" \
  --styles "${STYLE_LIST[@]}" \
  --seed "${SEED:-42}"

for style in "${STYLE_LIST[@]}"; do
  python machine_unlearning/retrack_latent/src/train_retrack_ffsd.py \
    --allclass_checkpoint "${ALLCLASS}" \
    --data_root "${DATA_ROOT}" \
    --prompts_file "${PROMPTS}" \
    --forget_style "${style}" \
    --anchor_strategy weighted_style_select \
    --clip_prototypes_file "${PROTO}" \
    --descriptors_file "${DESCRIPTORS}" \
    --beta "${AWSS_BETA:-2.0}" \
    --eta_uniform "${AWSS_ETA:-0.3}" \
    --style_sampling_mode weighted \
    --num_anchor_descriptors "${ANCHOR_DESCRIPTORS:-3}" \
    --output_dir "${RETRACK_DIR}/${style}" \
    --num_steps "${UNLEARN_STEPS:-5000}" \
    --learning_rate "${UNLEARN_LR:-2e-6}"
done

python evaluation/compute_logoa_ffsd.py \
  --allclass_dir "${ALLCLASS_DIR}" \
  --checkpoint_step "${ALLCLASS_STEP:-10000}" \
  --logo_dir "${LOGO_DIR}" \
  --image_dir "${IMAGE_DIR}" \
  --eval_prompts_file "${EVAL_PROMPTS}" \
  --output_dir outputs/logoa_scores_ffsd_very_relaxed

python evaluation/compute_una_ffsd.py \
  --allclass_dir "${ALLCLASS_DIR}" \
  --checkpoint_step "${ALLCLASS_STEP:-10000}" \
  --retrack_dir "${RETRACK_DIR}" \
  --retrack_checkpoint_step "${GUDA_STEP:-2000}" \
  --image_dir "${IMAGE_DIR}" \
  --eval_prompts_file "${EVAL_PROMPTS}" \
  --output_dir outputs/guda_awss_scores

python evaluation/analyze_una_retrack_correlation.py \
  --logoa_dir outputs/logoa_scores_ffsd_very_relaxed \
  --una_dir outputs/guda_awss_scores \
  --output_dir outputs/ranking_agreement_awss

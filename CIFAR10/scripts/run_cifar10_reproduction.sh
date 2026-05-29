#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

DATA_DIR="${DATA_DIR:-data}"
BATCH_SIZE="${BATCH_SIZE:-256}"
FULL_DIR="${FULL_DIR:-outputs/models/cifar10/full}"
COND_DIR="${COND_DIR:-outputs/models/cifar10/conditional}"
LOGO_DIR="${LOGO_DIR:-outputs/models/cifar10/logo}"
RETRACK_DIR="${RETRACK_DIR:-outputs/models/cifar10/guda_retrack}"
ESD_DIR="${ESD_DIR:-outputs/models/cifar10/guda_esd}"
QUERY_DIR="${QUERY_DIR:-results/cifar10/generated_queries/ddpm4000}"
NOISE_FILE="${NOISE_FILE:-results/cifar10/fixed_noise/query_2048.pt}"
FULL_EVAL_CKPT="${FULL_EVAL_CKPT:-${FULL_DIR}/epoch_2400}"
FULL_TEACHER_CKPT="${FULL_TEACHER_CKPT:-${FULL_DIR}/latest}"
COND_CKPT="${COND_CKPT:-${COND_DIR}/latest}"

DEFAULT_CLASSES=(0 1 2 3 4 5 6 7 8 9)
if [[ -n "${CLASSES:-}" ]]; then
  read -r -a CLASS_LIST <<< "${CLASSES}"
else
  CLASS_LIST=("${DEFAULT_CLASSES[@]}")
fi

python scripts/train_cifar10_unconditional.py \
  --dataset cifar10 \
  --data_dir "${DATA_DIR}" \
  --output_dir "${FULL_DIR}" \
  --epochs "${FULL_EPOCHS:-2600}" \
  --batch_size "${BATCH_SIZE}" \
  --timesteps 4000 \
  --beta_schedule squaredcos_cap_v2

if [[ "${RUN_ESD:-0}" == "1" ]]; then
  python scripts/train_cifar10_conditional.py \
    --dataset cifar10 \
    --data_dir "${DATA_DIR}" \
    --output_dir "${COND_DIR}" \
    --epochs "${COND_EPOCHS:-400}" \
    --batch_size "${BATCH_SIZE}" \
    --timesteps 4000 \
    --beta_schedule squaredcos_cap_v2
fi

for cls in "${CLASS_LIST[@]}"; do
  python scripts/train_cifar10_unconditional.py \
    --dataset cifar10 \
    --data_dir "${DATA_DIR}" \
    --output_dir "${LOGO_DIR}/exclude_${cls}" \
    --exclude_classes "${cls}" \
    --epochs "${LOGO_EPOCHS:-2600}" \
    --batch_size "${BATCH_SIZE}" \
    --timesteps 4000 \
    --beta_schedule squaredcos_cap_v2
done

python tools/create_fixed_noise.py \
  --output "${NOISE_FILE}" \
  --num_images "${QUERY_IMAGES:-2048}" \
  --channels 3 \
  --height 32 \
  --width 32 \
  --seed "${SEED:-42}"

python tools/sample.py \
  --model_type unconditional \
  --ckpt_dir "${FULL_EVAL_CKPT}" \
  --dataset cifar10 \
  --output_dir "${QUERY_DIR}" \
  --num_images "${QUERY_IMAGES:-2048}" \
  --batch_size "${BATCH_SIZE}" \
  --num_inference_steps 4000 \
  --scheduler ddpm \
  --noise_file "${NOISE_FILE}" \
  --seed "${SEED:-42}"

for cls in "${CLASS_LIST[@]}"; do
  python scripts/train_cifar10_retrack.py \
    --dataset cifar10 \
    --data_dir "${DATA_DIR}" \
    --teacher_model_path "${FULL_TEACHER_CKPT}" \
    --unlearn_class "${cls}" \
    --output_dir "${RETRACK_DIR}/unlearn_class_${cls}" \
    --epochs "${UNLEARN_EPOCHS:-50}" \
    --lr "${RETRACK_LR:-1e-5}" \
    --lambda_forget "${RETRACK_LAMBDA:-0.03}" \
    --k_neighbors "${K_NEIGHBORS:-10}" \
    --fast_retrack
done

if [[ "${RUN_ESD:-0}" == "1" ]]; then
  for cls in "${CLASS_LIST[@]}"; do
    python scripts/train_cifar10_esd.py \
      --dataset cifar10 \
      --data_dir "${DATA_DIR}" \
      --teacher_model_path "${FULL_TEACHER_CKPT}" \
      --conditional_model_path "${COND_CKPT}" \
      --unlearn_class "${cls}" \
      --output_dir "${ESD_DIR}/unlearn_class_${cls}" \
      --epochs "${ESD_EPOCHS:-50}" \
      --lr "${ESD_LR:-3e-5}" \
      --lambda_forget "${ESD_LAMBDA:-0.3}"
  done
fi

python evaluation/compute_delta_elbo_logo.py \
  --dataset cifar10 \
  --gen_dir "${QUERY_DIR}" \
  --ckpt_all "${FULL_EVAL_CKPT}" \
  --ckpt_base "${LOGO_DIR}" \
  --ckpt_epoch "${LOGO_CKPT_EPOCH:-epoch_2400}" \
  --output_dir results/cifar10/logoa \
  --timesteps 4000 \
  --skip 10 \
  --min_t 1

python evaluation/compute_delta_elbo_unlearned.py \
  --dataset cifar10 \
  --gen_dir "${QUERY_DIR}" \
  --ckpt_base "${FULL_EVAL_CKPT}" \
  --ckpt_unlearned "${RETRACK_DIR}" \
  --ckpt_epoch "${UNLEARN_CKPT_EPOCH:-epoch_0030}" \
  --output_dir results/cifar10/guda_retrack_scores \
  --timesteps 4000 \
  --skip 10 \
  --min_t 1

python evaluation/plot_multi_correlation_analysis.py \
  --reference_csv results/cifar10/logoa/delta_elbo_loo_sorted.csv \
  --test_csvs results/cifar10/guda_retrack_scores/delta_elbo_unlearned_sorted.csv \
  --labels GUDA_ReTrack \
  --output_dir results/cifar10/ranking_agreement \
  --exp_name logoa_vs_guda

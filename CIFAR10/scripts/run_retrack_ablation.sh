#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-.}"

DATA_DIR="${DATA_DIR:-data}"
FULL_CKPT="${FULL_CKPT:-outputs/models/cifar10/full/latest}"
CLASS_ID="${CLASS_ID:-0}"

for epochs in ${EPOCH_GRID:-10 20 30 40 50}; do
  for lr in ${LR_GRID:-1e-4 3e-5 1e-5}; do
    for lambda in ${LAMBDA_GRID:-0.003 0.01 0.03}; do
      for k in ${K_GRID:-1 5 10 20 50}; do
        out="outputs/ablations/retrack/class_${CLASS_ID}/epochs${epochs}_lr${lr}_lambda${lambda}_k${k}"
        python scripts/train_cifar10_retrack.py \
          --dataset cifar10 \
          --data_dir "${DATA_DIR}" \
          --teacher_model_path "${FULL_CKPT}" \
          --unlearn_class "${CLASS_ID}" \
          --output_dir "${out}" \
          --epochs "${epochs}" \
          --lr "${lr}" \
          --lambda_forget "${lambda}" \
          --k_neighbors "${k}" \
          --fast_retrack
      done
    done
  done
done


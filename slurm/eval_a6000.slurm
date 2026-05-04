#!/bin/bash
#SBATCH --job-name=heartseg-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=eval-%j.out
#SBATCH --error=eval-%j.err
#
#   export REPO_ROOT=/fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
#   export MEDIA_ROOT=/fs/nexus-scratch/anant04/heart-mri-data
#   export CKPT=/fs/nexus-scratch/anant04/runs/segmentation_exp01/checkpoint_best.pt
#   export OUT_DIR=/fs/nexus-scratch/anant04/reports/eval_val_001
#   export SPLIT=val          # optional: train | val | test
#   sbatch slurm/eval_a6000.slurm

set -euo pipefail
: "${REPO_ROOT:?}"
: "${MEDIA_ROOT:?}"
: "${CKPT:?}"
: "${OUT_DIR:?}"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.1 || module load cuda || true
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"
mkdir -p "${OUT_DIR}"

python scripts/monai_eval_segmentation.py \
  --data-root "${REPO_ROOT}" \
  --media-root "${MEDIA_ROOT}" \
  --checkpoint "${CKPT}" \
  --split "${SPLIT:-val}" \
  --out-dir "${OUT_DIR}" \
  --device cuda

echo "Wrote reports under ${OUT_DIR}"

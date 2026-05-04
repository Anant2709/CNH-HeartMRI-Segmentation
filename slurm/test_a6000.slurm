#!/bin/bash
#SBATCH --job-name=heartseg-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
#
# External TEST split only (frozen protocol).
#
#   export REPO_ROOT=/fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
#   export MEDIA_ROOT=/fs/nexus-scratch/anant04/heart-mri-data
#   export CKPT=/fs/nexus-scratch/anant04/runs/segmentation_exp01/checkpoint_best.pt
#   export OUT_DIR=/fs/nexus-scratch/anant04/reports/test_external_001
#   sbatch slurm/test_a6000.slurm

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

python scripts/monai_test_segmentation.py \
  --data-root "${REPO_ROOT}" \
  --media-root "${MEDIA_ROOT}" \
  --checkpoint "${CKPT}" \
  --out-dir "${OUT_DIR}" \
  --device cuda

echo "Wrote test reports under ${OUT_DIR}"

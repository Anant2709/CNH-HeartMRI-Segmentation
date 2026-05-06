#!/bin/bash
#SBATCH --job-name=heartseg-intsub
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=intsub-%j.out
#SBATCH --error=intsub-%j.err
#
# Internal-only subset experiment:
# - train + val + test are ALL internal (patient-level split).
#
# Usage:
#   export REPO_ROOT=/fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
#   export MEDIA_ROOT=/fs/nexus-scratch/anant04/heart-mri-data
#   export RUN_DIR=/fs/nexus-scratch/anant04/runs/internal_subset_01
#   python "${REPO_ROOT}/scripts/build_internal_subset_split.py" --data-root "${REPO_ROOT}"
#   sbatch slurm/train_internal_subset.slurm

set -euo pipefail

: "${REPO_ROOT:?export REPO_ROOT=/path/to/CNH-HeartMRI-Segmentation}"
: "${MEDIA_ROOT:?export MEDIA_ROOT=/path/to/heart-mri-data}"
: "${RUN_DIR:?export RUN_DIR=/path/to/runs/internal_subset_01}"

SPLIT_CSV="${SPLIT_CSV:-${REPO_ROOT}/reports/splits/internal_train_val_internal_test.csv}"
if [[ ! -f "${SPLIT_CSV}" ]]; then
  echo "Missing split CSV: ${SPLIT_CSV}" >&2
  echo "Generate with: python scripts/build_internal_subset_split.py --data-root ." >&2
  exit 1
fi

if command -v module >/dev/null 2>&1; then
  module unload cuda cuda/12.1 cuda/12.1.1 cuda/13.1.1 2>/dev/null || true
  module load "${CUDA_MODULE:-cuda/12.1.1}" \
    || module load cuda/12.1 \
    || module load cuda/13.1.1 \
    || module load cuda \
    || true
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"
mkdir -p "${RUN_DIR}"
export PYTHONUNBUFFERED=1

python scripts/monai_train_segmentation.py \
  --data-root "${REPO_ROOT}" \
  --media-root "${MEDIA_ROOT}" \
  --split-csv "${SPLIT_CSV}" \
  --out-dir "${RUN_DIR}" \
  --device cuda \
  --amp \
  --epochs "${EPOCHS:-100}" \
  --val-interval "${VAL_INTERVAL:-5}" \
  --final-test

echo "Done. Internal-subset test is in ${RUN_DIR}/test_summary.json"

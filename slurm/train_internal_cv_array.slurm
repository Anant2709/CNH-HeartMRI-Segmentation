#!/bin/bash
#SBATCH --job-name=heartseg-icv
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-4
#SBATCH --output=icv-%A_%a.out
#SBATCH --error=icv-%A_%a.err
#
# Internal-only K-fold CV training (patient-level folds from build_splits.py).
#
# Usage:
#   export REPO_ROOT=/fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
#   export MEDIA_ROOT=/fs/nexus-scratch/anant04/heart-mri-data
#   export RUN_ROOT=/fs/nexus-scratch/anant04/runs/internal_cv
#   export CV_FOLDS=5
#   python "${REPO_ROOT}/scripts/build_splits.py" --data-root "${REPO_ROOT}" --internal-cv-folds "${CV_FOLDS}"
#   sbatch --array=0-$((CV_FOLDS-1)) slurm/train_internal_cv_array.slurm

set -euo pipefail

: "${REPO_ROOT:?export REPO_ROOT=/path/to/CNH-HeartMRI-Segmentation}"
: "${MEDIA_ROOT:?export MEDIA_ROOT=/path/to/heart-mri-data}"
: "${RUN_ROOT:?export RUN_ROOT=/path/to/runs/internal_cv}"

FOLD_ID="${SLURM_ARRAY_TASK_ID}"
FOLD_PADDED="$(printf "%02d" "${FOLD_ID}")"
SPLIT_CSV="${REPO_ROOT}/reports/splits/internal_cv_fold_${FOLD_PADDED}.csv"
OUT_DIR="${RUN_ROOT}/fold_${FOLD_PADDED}"

if [[ ! -f "${SPLIT_CSV}" ]]; then
  echo "Missing fold split CSV: ${SPLIT_CSV}" >&2
  echo "Generate with: python scripts/build_splits.py --data-root . --internal-cv-folds ${CV_FOLDS:-5}" >&2
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
mkdir -p "${OUT_DIR}"
export PYTHONUNBUFFERED=1

python scripts/monai_train_segmentation.py \
  --data-root "${REPO_ROOT}" \
  --media-root "${MEDIA_ROOT}" \
  --split-csv "${SPLIT_CSV}" \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --amp \
  --epochs "${EPOCHS:-100}" \
  --val-interval "${VAL_INTERVAL:-5}"

echo "Fold ${FOLD_PADDED} done. See ${OUT_DIR}/summary.json"

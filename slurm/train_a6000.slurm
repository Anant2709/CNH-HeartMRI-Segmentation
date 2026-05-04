#!/bin/bash
#SBATCH --job-name=heartseg-train
# UMIACS Nexus GPU pool is typically "tron" (not "gpu"). See: wiki SLURM/JobSubmission. Run: sinfo -o "%P %a" | head
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --gres=gpu:1
# Nexus default QoS often caps CPUs per job at 4; raise only if your association allows it.
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err
#
# Optional RTX A6000 (uncomment the style your cluster documents):
# #SBATCH --gres=gpu:a6000:1
# #SBATCH --constraint=a6000
#
# Submit from repo root, after exporting:
#   export REPO_ROOT=/fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
#   export MEDIA_ROOT=/fs/nexus-scratch/anant04/heart-mri-data
#   export RUN_DIR=/fs/nexus-scratch/anant04/runs/segmentation_exp01
#   export EPOCHS=100          # optional
#   export VAL_INTERVAL=5    # optional
#   sbatch slurm/train_a6000.slurm

set -euo pipefail
: "${REPO_ROOT:?export REPO_ROOT=/path/to/CNH-HeartMRI-Segmentation}"
: "${MEDIA_ROOT:?export MEDIA_ROOT=/path/to/heart-mri-data}"
: "${RUN_DIR:?export RUN_DIR=/path/to/runs/segmentation_exp01}"

if command -v module >/dev/null 2>&1; then
  # Prolog or login may preload CUDA; loading another default causes "Conflicting cuda is loaded".
  module unload cuda cuda/12.1 cuda/12.1.1 cuda/13.1.1 2>/dev/null || true
  # Use `module avail cuda` on Nexus; many nodes expose cuda/12.1.1 (not cuda/12.1). Override: export CUDA_MODULE=cuda/X.Y.Z
  module load "${CUDA_MODULE:-cuda/12.1.1}" \
    || module load cuda/12.1 \
    || module load cuda/13.1.1 \
    || module load cuda \
    || true
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv/bin/activate"
mkdir -p "${RUN_DIR}"

python scripts/monai_train_segmentation.py \
  --data-root "${REPO_ROOT}" \
  --media-root "${MEDIA_ROOT}" \
  --out-dir "${RUN_DIR}" \
  --device cuda \
  --amp \
  --epochs "${EPOCHS:-100}" \
  --val-interval "${VAL_INTERVAL:-5}"

echo "Done. Artifacts under ${RUN_DIR}"

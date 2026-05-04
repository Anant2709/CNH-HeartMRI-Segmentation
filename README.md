# Heart MRI Segmentation

Pediatric cardiac MRI **3D multiclass segmentation** (five classes including background). This repo keeps **curated CSVs** (manifest + splits) and a **MONAI** training script; NRRDs stay on disk outside git.

## Train (MONAI)

```bash
cd CNH-HeartMRI-Segmentation
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-training.txt
```

Point `--media-root` at the folder that contains `External/` and `Internal/` if the default search does not find your NRRDs.

```bash
python scripts/monai_train_segmentation.py --data-root . --epochs 100 --device auto
```

Useful flags:

- `--media-root /path/to/data` — folder with `External/`, `Internal/`
- `--patch-size 96 96 96` — 3D crop for training and sliding-window ROI
- `--spacing-mm 1.2 1.2 1.2` — optional resample to fixed spacing (omit for native spacing)
- `--amp` — mixed precision on CUDA
- `--split-csv path/to.csv` — override split file (default: internal train/val + external test)
- `--final-test` — after training, score the external **test** split once (no tuning on test); skip for internal-only CV CSVs (no `test` rows)

Quick sanity check (small subset, short run):

```bash
python scripts/monai_train_segmentation.py --data-root . --epochs 1 \\
  --max-train-cases 2 --max-val-cases 1 --val-interval 1
```

Outputs go to `runs/segmentation/` (`config.json`, `checkpoint_best.pt`, `checkpoint_last.pt`, `history.csv`, `summary.json`).

Shared helpers live in `scripts/monai_segmentation_common.py`. Training stack includes **`nibabel`** (used by MONAI `Orientationd`).

## Eval and test (no training)

```bash
# Validation split (or use --split train)
python scripts/monai_eval_segmentation.py \
  --data-root . --media-root /path/to/data \
  --checkpoint runs/segmentation/checkpoint_best.pt \
  --split val --out-dir reports/eval_val_01 --device cuda

# External test only (wrapper; do not tune on these metrics)
python scripts/monai_test_segmentation.py \
  --data-root . --media-root /path/to/data \
  --checkpoint runs/segmentation/checkpoint_best.pt \
  --out-dir reports/test_external_01 --device cuda
```

Optional: `--save-predictions-dir /path/to/preds` writes one `*_pred.nrrd` per case (class map in **preprocessed** RAS grid; use for rough Slicer QA).

## Slurm (GPU cluster)

Templates under `slurm/` (`train_a6000.slurm`, `eval_a6000.slurm`, `test_a6000.slurm`). Set `REPO_ROOT`, `MEDIA_ROOT`, `RUN_DIR` / `CKPT` / `OUT_DIR` as in the file headers, then `sbatch slurm/train_a6000.slurm`. See [`docs/COMPUTE_NEXUS.md`](docs/COMPUTE_NEXUS.md) for `rsync`/`scp` from laptop to scratch.

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/DATASET.md`](docs/DATASET.md) | Manifests, splits, media root, regeneration commands |
| [`docs/MONAI_TRAIN_SEGMENTATION.md`](docs/MONAI_TRAIN_SEGMENTATION.md) | Full walkthrough of `monai_train_segmentation.py` (math + code) |
| [`docs/progress_log.md`](docs/progress_log.md) | Timeline and decisions |
| [`docs/COMPUTE_NEXUS.md`](docs/COMPUTE_NEXUS.md) | UMIACS Nexus scratch + GPU + Slurm |

## Dataset CSVs and splits

See [`docs/DATASET.md`](docs/DATASET.md). Summary: `reports/dataset_manifest.csv` + `reports/splits/internal_train_val_external_test.csv` drive training.

## Data QA scripts (optional maintenance)

| Script | Role |
|--------|------|
| `scripts/build_pairing_audit.py` | Scan tree → pairing CSVs |
| `scripts/build_baseline_manifest.py` | Strict / geometry manifests |
| `scripts/build_dataset_manifest.py` | Stage B manifest + summary |
| `scripts/build_splits.py` | Train/val/test CSV; optional **`--internal-cv-folds K`** (internal k-fold) or **`--cv-folds K`** (mixed-site k-fold) |
| `scripts/inspect_nrrd_dataset.py` | NRRD metadata dump |
| `scripts/inspect_torch_checkpoint.py` | Inspect `.pt` checkpoints |

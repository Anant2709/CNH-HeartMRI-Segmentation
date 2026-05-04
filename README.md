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
- `--final-test` — after training, score the external **test** split once (no tuning on test)

Quick sanity check (small subset, short run):

```bash
python scripts/monai_train_segmentation.py --data-root . --epochs 1 \\
  --max-train-cases 2 --max-val-cases 1 --val-interval 1
```

Outputs go to `runs/segmentation/` (`config.json`, `checkpoint_best.pt`, `checkpoint_last.pt`, `history.csv`, `summary.json`).

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
| `scripts/build_splits.py` | Train/val/test CSV (`--cv-folds 5` optional) |
| `scripts/inspect_nrrd_dataset.py` | NRRD metadata dump |
| `scripts/inspect_torch_checkpoint.py` | Inspect `.pt` checkpoints |

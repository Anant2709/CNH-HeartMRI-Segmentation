# Dataset and splits (reference)

This document describes **what the CSVs mean**, **how paths resolve to NRRD files**, and **how to regenerate** everything after data or policy changes. Training code (`scripts/monai_train_segmentation.py`) consumes the **split CSV** plus on-disk NRRDs; it does not re-run pairing logic at runtime.

---

## 1. Label convention

Multiclass segmentation masks use integer voxel values:

| Value | Structure |
|------:|-----------|
| 0 | Background |
| 1 | Atrium |
| 2 | Ventricle |
| 3 | CPC |
| 4 | Arch |

Clinical legend CSVs (if present in your tree, e.g. under `model/`) align names with IDs 1–4. Training assumes **exactly** this five-class setup (including background).

---

## 2. Where NRRD files live (media root)

CSV columns `image_path` and `mask_path` are **relative** to a single directory we call the **media root** — the folder that contains `External/` and `Internal/` trees (e.g. `External/BCH-….nrrd`).

**Default search order** (when you do not pass `--media-root`) is the same for `build_dataset_manifest.py`, `monai_train_segmentation.py`, and `monai_eval_segmentation.py`:

1. `<repo>/../data`
2. `<repo>/data`
3. `<repo>` (repository root)
4. Parent of `<repo>`

If files are not found, pass an explicit absolute path:

```bash
python scripts/build_dataset_manifest.py --data-root . --media-root /abs/path/to/nrrd_root
python scripts/monai_train_segmentation.py --data-root . --media-root /abs/path/to/nrrd_root
python scripts/monai_eval_segmentation.py --data-root . --media-root /abs/path/to/nrrd_root --checkpoint ... --out-dir ...
```

---

## 3. Files under `reports/` (data lineage)

### 3.1 Pairing QA

| File | Purpose |
|------|---------|
| `reports/pairing_audit.csv` | One row per candidate image; geometry and heuristic mask pairing |
| `reports/pairing_mismatches.csv` | Subset flagged for human review (optional Slicer QA) |

**Producer:** `scripts/build_pairing_audit.py --root <media_root>`

---

### 3.2 Baseline manifests (strict vs geometry-only)

| File | Purpose |
|------|---------|
| `reports/baseline_manifest_v1.csv` | **Strict** list: geometry OK, pairing OK, mask labels exactly `{0,1,2,3,4}`, default excludes ambiguous 4D unless you pass flags to builder |
| `reports/baseline_manifest_v1_excluded.csv` | Rows excluded from strict list (with reasons in the CSV) |
| `reports/baseline_manifest_v1_geometry_only.csv` | Geometry-valid rows allowing **any** label set (broader exploratory list) |
| `reports/baseline_manifest_v1_geometry_only_excluded.csv` | Excluded from geometry-only list |

**Producer:** `scripts/build_baseline_manifest.py --data-root .` (uses pairing audit under `reports/` by default; see script `--help`).

Stage B manifest builder **reads the strict** `baseline_manifest_v1.csv` by default.

---

### 3.3 Dataset manifest (Stage B)

| File | Purpose |
|------|---------|
| `reports/dataset_manifest.csv` | One row per training sample: `sample_id`, `site`, `patient_id`, paths, `is_cropped`, `is_4d`, `n_frames`, `size_xyz`, `spacing_xyz`, `labels_present`, `pairing_status`, `quality_flag` |
| `reports/dataset_manifest_summary.md` | Counts and which media roots were checked for file existence |

**Current snapshot (regenerate after changes):**

- **132** rows, all `quality_flag == ok_automated` in the last generated summary.
- **113** internal, **19** external (site column).
- **patient_id** is a **filename-derived** grouping key for leakage-safe splitting (not necessarily a hospital MRN).

**Producer:**

```bash
python scripts/build_dataset_manifest.py --data-root .
```

Only rows with paths that exist under a resolved media root should show `ok_automated`. Fix paths or pass `--media-root` if not.

---

### 3.4 Splits (Stage C)

| File | Purpose |
|------|---------|
| `reports/splits/internal_train_val_external_test.csv` | Every manifest row with columns `sample_id`, `patient_id`, `site`, `split`, `image_path`, `mask_path` where `split` ∈ {`train`, `val`, `test`} |
| `reports/splits/split_summary.txt` | Human-readable counts |

**Policy (primary split):**

- **Internal** site rows: split by **patient_id** into **train** and **val** (default 20% of internal *patients* to val, at least one patient in val when possible).
- **External** site rows: **all** assigned to **test** (holdout for generalization; do not tune hyperparameters on test).

**Optional split variants** (not written unless requested):

1. **Internal-only k-fold** — `internal_cv_fold_00.csv`, … — **internal rows only** (external partner cases are omitted). Each file is one fold: **val** = internal patients assigned to that fold; **train** = remaining internal patients. Same columns as the primary CSV (`split` is only `train` or `val`). Use for **internal benchmarking** without contaminating val with external appearance. Requires at least as many internal **patients** as folds.

   ```bash
   python scripts/build_splits.py --data-root . --internal-cv-folds 5
   ```

2. **Mixed-site k-fold** — `mixed_site_cv_fold_00.csv`, … — pools **internal + external** patients into folds (each row is `train` or `val` for that fold; extra column `fold_id`). Use when you explicitly want cross-site / mixed-pool CV, not a pure internal-only estimate.

   ```bash
   python scripts/build_splits.py --data-root . --cv-folds 5
   ```

Defaults: `--internal-cv-folds 0`, `--cv-folds 0` (primary split only). You can pass **both** flags in one run if you want both file families.

**Training / eval:** Point `monai_train_segmentation.py` and `monai_eval_segmentation.py` at a fold CSV with **`--split-csv reports/splits/internal_cv_fold_00.csv`** (repeat per fold). **`--final-test`** has no effect on internal-only fold files (there is no `test` split in those rows).

**Producer:**

```bash
python scripts/build_splits.py --data-root .
```

---

## 4. End-to-end regeneration (copy-paste)

From the repository root (`CNH-HeartMRI-Segmentation`):

```bash
# 1) Refresh pairing from live NRRD tree
python scripts/build_pairing_audit.py --root "/path/to/media_root"

# 2) Strict + geometry manifests
python scripts/build_baseline_manifest.py --data-root .

# 3) Dataset manifest + summary
python scripts/build_dataset_manifest.py --data-root .

# 4) Splits (primary only; add --internal-cv-folds 5 and/or --cv-folds 5 if needed)
python scripts/build_splits.py --data-root .
```

Order matters: each stage consumes the previous stage’s outputs (except pairing audit, which scans the filesystem directly).

---

## 5. Relationship to training and evaluation

`scripts/monai_train_segmentation.py`:

- Loads **`reports/splits/internal_train_val_external_test.csv`** (override with `--split-csv`).
- Resolves `image_path` / `mask_path` to absolute files using the media root rules above.
- Uses rows with `split == train` for optimization and `split == val` for periodic full-volume validation. Optional `--final-test` runs the **test** split once at the end using the best validation checkpoint.

`scripts/monai_eval_segmentation.py` scores any **split** (`train`, `val`, or `test`) from the same CSV and writes `eval_<split>_per_case.csv`, `eval_<split>_summary.json`, and `eval_<split>_summary.md`.

`scripts/monai_test_segmentation.py` is a thin wrapper that always evaluates **`split=test`** (external holdout) and reminds you not to tune on test metrics.

It does **not** read `dataset_manifest.csv` directly today; the split file is expected to stay **in sync** with the manifest used when `build_splits.py` was last run. If you edit the manifest, **re-run** `build_splits.py` before long training jobs.

---

## 6. Further reading

- **Training script (math, transforms, CLI):** [`docs/MONAI_TRAIN_SEGMENTATION.md`](MONAI_TRAIN_SEGMENTATION.md)
- **Progress and decisions:** [`docs/progress_log.md`](progress_log.md)
- **Cluster / GPU runbook:** [`docs/COMPUTE_NEXUS.md`](COMPUTE_NEXUS.md)

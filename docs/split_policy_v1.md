# Split policy v1 (Stage C)

## Primary baseline split (recommended first)

| Split | Source | Rule |
|-------|--------|------|
| **train** | `site == internal` | Random **patient-level** partition; see `--val-fraction` and `--seed`. |
| **val** | `site == internal` | Remaining internal patients (no patient appears in both train and val). |
| **test** | `site == external` | **All** external rows are test holdout (not used for hyperparameter tuning in the primary workflow). |

**File:** `reports/splits/internal_train_val_external_test.csv`

**Rationale:** Tuning on internal validation mimics “same institution distribution”; reporting on external test estimates **cross-site generalization** without contaminating tuning with external statistics.

## Patient ID and leakage

- Splits are keyed by `patient_id` from `dataset_manifest.csv` (filename heuristic).
- If the same `patient_id` appears under both internal and external, the split script **exits with an error** so you must disambiguate (e.g. `internal:prefontan_78` vs `external:bch-foo`) before training.

## Reproducibility

- `--seed` (default `42`) controls shuffling of **internal patient ids** and the mixed-site CV patient order.
- Changing seed or `val_fraction` changes assignments; record both in experiment notes.

## Mixed-site k-fold (secondary experiment)

- **Files:** `reports/splits/mixed_site_cv_fold_00.csv` … `fold_04.csv` (for `--cv-folds 5`).
- **Rule:** All patients (internal + external) are shuffled with the same seed, ordered, and assigned fold index `i % K`. For fold `f`, patients with index `f` are **val**; others **train**.
- **Note:** This is **not** the same as the primary internal-train/val + external-test story; use it when you intentionally want cross-site CV.

## CLI reference

```bash
python scripts/build_splits.py --data-root .
python scripts/build_splits.py --data-root . --val-fraction 0.15 --seed 1
python scripts/build_splits.py --data-root . --cv-folds 0   # primary only, no CV files
```

## Gate (Stage C)

- No `patient_id` appears in more than one of `train`, `val`, `test` in the primary CSV.
- External rows are exclusively `test` in the primary CSV.

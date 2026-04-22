# Baseline v1 protocol (Stage D)

## Data contract (inputs and labels)

- Source manifest: `reports/dataset_manifest.csv`
- Split file: `reports/splits/internal_train_val_external_test.csv`
- Label contract (strict): background + 4 structures
  - `0`: background
  - `1`: atrium
  - `2`: ventricle
  - `3`: CPC
  - `4`: arch
- Stage B/C status used for v1:
  - total strict rows: 132
  - internal rows: 113
  - external rows: 19
  - primary split counts: train 90, val 23, test 19

### Inclusion rules for v1 training/evaluation

- Use only rows where:
  - `quality_flag == ok_automated`
  - strict labels were enforced in baseline manifest creation
  - split assignment exists in Stage C CSV
- For primary baseline:
  - train + val: internal only
  - test: external only

### Exclusion rules for v1

- Any row outside strict label contract `{0..4}`
- Any row with unresolved path/data quality flags
- Any row that would cause patient leakage across splits

---

## Modeling choice (v1)

### Task formulation

- Multiclass 3D semantic segmentation
- One row in manifest = one 3D volume + one 3D mask

### Architecture family

- Start with a standard 3D U-Net style model (MONAI or equivalent implementation).
- Keep architecture simple for v1; avoid temporal modules here.

### Output channels

- 5 channels (classes 0..4), softmax over class dimension.

---

## Preprocessing pipeline (v1 frozen plan)

Order is important:

1. **Load image and mask**
  - Read from `image_path`, `mask_path` in split CSV.
2. **Canonical orientation**
  - Convert to a consistent orientation convention.
  - Reason: different institutions/scanners may store axes differently.
3. **Intensity normalization (image only)**
  - Per-volume normalization (e.g., z-score with robust clipping if needed).
  - Reason: MRI intensity scales are not standardized.
4. **Spacing handling**
  - v1 default: keep native spacing for first run if memory allows.
  - If unstable across scanners, switch to fixed isotropic/near-isotropic target spacing and log this change as v1.1.
5. **Patch extraction / spatial size**
  - Use fixed-size 3D patches for GPU feasibility.
  - Start conservative and adjust after first memory profiling run.
6. **Mask handling**
  - Preserve integer labels exactly (nearest-neighbor interpolation when resampling masks).

---

## Training configuration (v1 defaults)

- Loss:
  - Combined Dice + Cross-Entropy (common for class imbalance and stable optimization).
- Optimizer:
  - AdamW (default baseline choice).
- Learning rate:
  - fixed initial LR with scheduler (document exact value in run config).
- Batch size:
  - as large as fits GPU memory (start low, then scale).
- Epochs:
  - fixed max epoch budget + early stopping on internal validation mean Dice.
- Seed:
  - fixed seed for reproducibility and deterministic split reuse.

> Note: precise numeric hyperparameters should be captured in the first runnable training script config (`runs/baseline_v1/config.yaml` or equivalent).

---

## Evaluation protocol

### Validation (during training)

- Evaluate on internal validation split (`split == val`) each epoch.
- Primary monitor metric: mean Dice across foreground classes (1..4).

### Final reporting

- Internal validation summary (best checkpoint epoch)
- External test summary (single pass, no hyperparameter tuning)

### Required outputs

- Per-class Dice (`1..4`)
- Mean foreground Dice
- Optional background Dice reported separately (not used as primary quality signal)
- Site-stratified results:
  - internal val
  - external test

---

## Runtime constraints and guardrails

- Keep first run small enough to complete end-to-end in one session.
- Save:
  - best checkpoint by validation metric,
  - per-epoch metrics,
  - inference outputs for at least a small validation subset.
- Never tune hyperparameters on external test.
- If a major preprocessing change is made, bump protocol version (v1 -> v1.1) and record reason.

---

## Stage D gate checklist

Protocol is considered complete when all are true:

- Dataset and split files are fixed and versioned in reports.
- Label contract is explicit (0..4).
- Preprocessing order is specified.
- Training objective/loss family is specified.
- Evaluation metrics and reporting structure are specified.
- Leakage and test-holdout rules are explicit.

---

## Planned next step (Stage E)

Implement:

- `scripts/train_baseline_v1.py`
- `scripts/eval_baseline_v1.py`

using this protocol without changing assumptions mid-run.
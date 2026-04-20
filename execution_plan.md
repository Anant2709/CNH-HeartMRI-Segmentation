# Heart MRI Segmentation: Execution Plan (v1)

## Purpose
Define a quality-gated, reproducible path from dataset QA to a first baseline model, then prepare transition to temporal (4D) modeling.

## Executive Outcome
By the end of Stages A-D, we should have:
- a clean, approved trainable sample list,
- leakage-safe split definitions,
- a frozen baseline experiment protocol,
- no ambiguity about what "trainable data" means.

This ensures that even modest baseline performance is scientifically valid and reproducible.

---

## Stage A - Finalize Pairing Policy (Highest Priority)
**Objective:** Decide exactly which files are trainable and how image-mask pairing is defined.

### Tasks
- Review `reports/pairing_mismatches.csv` row by row.
- Classify each mismatch into:
  - **expected** (crop vs full mismatch by design),
  - **recoverable** (can map to the correct mask),
  - **unusable** (missing mask, bad metadata).
- Validate 10-15 representative cases in 3D Slicer:
  - 5 clean pairs,
  - 5 mismatched geometry,
  - all label-anomaly cases.

### Decisions to Lock
- Baseline input policy: cropped-only vs mixed.
- Keep original volumes as archive/reference (recommended), even if excluded from baseline inputs.
- Label policy for v1:
  - strict `[0,1,2,3,4]`, or
  - mapped exceptions with explicit rules.

### Deliverables
- `docs/pairing_policy_v1.md`
- Updated mismatch report column: `status = approved/rejected/review`

### Gate to Pass
- At least **90%** of selected baseline candidates have trusted image-mask alignment by visual QA.

---

## Stage B - Build Trainable Dataset Manifest
**Objective:** Produce one clean source of truth for training/evaluation samples.

### Tasks
- Create `dataset_manifest.csv` with fields:
  - `sample_id`, `site`, `patient_id` (if derivable), `image_path`, `mask_path`,
  - `is_cropped`, `is_4d`, `n_frames`, `size_xyz`, `spacing_xyz`,
  - `labels_present`, `pairing_status`, `quality_flag`.
- Exclude rows with unresolved pairing.
- Add deterministic sample IDs and stable sorting.

### Deliverables
- `reports/dataset_manifest.csv`
- `reports/dataset_manifest_summary.md` (counts by site, label set, 3D/4D, cropped/full)

### Gate to Pass
Every manifest row has:
- existing image path,
- existing mask path,
- approved pairing status,
- labels within policy.

---

## Stage C - Define Split Strategy (Leakage-Safe)
**Objective:** Create reproducible split files before training.

### Tasks
- Baseline split strategy (primary):
  - train/val on internal data,
  - external data as untouched test holdout.
- Alternate experiment:
  - mixed-site cross-validation (start 5-fold; consider 10-fold if sample size supports).
- Enforce patient-level grouping:
  - no patient overlap across train/val/test.

### Deliverables
- `reports/splits/internal_train_val_external_test.csv`
- `reports/splits/mixed_site_cv_fold_*.csv`
- `docs/split_policy_v1.md`

### Gate to Pass
- Zero patient leakage across splits/folds.
- External test set untouched during baseline tuning.

---

## Stage D - Baseline v1 Spec (Per-Frame 3D)
**Objective:** Freeze a simple baseline experiment specification before full training implementation.

### Baseline Choice (Recommended)
- Per-frame 3D segmentation:
  - for 4D scans: split into independent 3D frames,
  - for 3D scans: one sample equals one volume.

### Tasks
- Define preprocessing:
  - orientation harmonization,
  - intensity normalization,
  - spacing handling (resample or patch-based strategy).
- Define labels:
  - multiclass with background + 4 structures.
- Define metrics:
  - Dice per class, mean Dice, site-stratified Dice.
- Define runtime constraints:
  - patch size, batch size, epoch count, early stopping rule.

### Deliverables
- `docs/baseline_v1_protocol.md` (complete experiment card)
- `docs/metrics_definition.md`

### Gate to Pass
- Protocol is complete enough for independent reproduction without clarification.

---

## Stage E - Implement Minimal Training Pipeline
**Objective:** Run one full end-to-end train/eval cycle, even with modest initial performance.

### Tasks
- Implement:
  - data loader from manifest,
  - preprocessing transform chain,
  - training loop,
  - validation loop,
  - metric logging.
- Save:
  - checkpoints,
  - validation predictions for visual inspection,
  - run config and random seed.

### Deliverables
- `scripts/train_baseline_v1.py`
- `scripts/eval_baseline_v1.py`
- `runs/baseline_v1/...` artifacts
- `reports/baseline_v1_results.md`

### Gate to Pass
- One complete train+eval cycle runs without manual intervention and outputs class-wise Dice.

---

## Stage F - Transition Plan (3D Aggregate -> 4D)
**Objective:** Define next milestone after baseline stabilization.

### Tasks
- 3D aggregate options:
  - majority/average over frame predictions,
  - temporal feature summaries after per-frame segmentation.
- 4D shortlist:
  - sequence model over frame embeddings,
  - full 4D spatiotemporal network (higher complexity).
- Select one low-risk temporal extension for v2.

### Deliverables
- `docs/temporal_extension_v2_plan.md`

### Gate to Pass
- Baseline v1 is stable and externally evaluated before 4D modeling starts.

---

## Suggested Timeline
- **Day 1-2:** Stage A (pairing policy + Slicer verification)
- **Day 3:** Stage B (manifest) + Stage C (splits)
- **Day 4:** Stage D (baseline protocol freeze)
- **Day 5-7:** Stage E (first training run + results summary)
- **Afterward:** Stage F planning + incremental temporal prototype

---

## Risk Register (Watch-Outs)
- Label anomalies (e.g., value 5 or missing class 2) -> decide remap/drop policy explicitly.
- 4D handling inconsistency -> define one frame extraction rule once.
- Crop/full confusion -> enforce one baseline data rule first.
- Data leakage in CV -> strict patient/site grouping required.
- External data used for tuning -> keep strict final holdout.

---

## Immediate Focus
Current execution priority:
1. Stage A (pairing policy + visual QA)
2. Stage B (dataset manifest creation)

## Status Note
No new code is required for this planning milestone. The next implementation action can begin with Stage B manifest automation.

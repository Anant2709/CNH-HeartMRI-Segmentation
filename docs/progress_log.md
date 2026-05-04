# Pediatric cardiac MRI segmentation — progress log

## Master project context (read this first)

This document is both a **progress tracker** and a **master knowledge note** for the CNH pediatric cardiac MRI project. If you lose context, start from this section.

### Project in plain language

- We are building a pipeline to automatically analyze pediatric heart MRI scans.
- Core first task: **segment heart structures** (atrium, ventricle, CPC, arch) from MRI volumes.
- Longer-term goal: combine segmentation with **time-domain analysis** (heart dynamics across frames) and downstream prediction.
- Why this matters: manual workflows are slow and fragmented; automation can make analysis faster and more consistent.

### Current understanding of the workflow

- Existing clinical workflow is roughly:
  1. 3D Slicer manual segmentation
  2. Separate software for temporal/time-based measurements
  3. ML predictions using extracted features
- Target future workflow:
  - Input MRI -> segmentation -> temporal features -> predictions (end-to-end or tightly integrated).

### Data setup we are working with

- **Internal data:** CNH pediatric cohort (includes 4D cine cases).
- **External data:** partner hospitals (useful for robustness/generalization testing).
- Files are in `.nrrd` format with original/cropped/segmentation variants.
- Ground truth masks are **manual clinician annotations** (high-trust supervision target).

### Label mapping (important)

- Expected segmentation classes:
  - `0`: background
  - `1`: atrium
  - `2`: ventricle
  - `3`: CPC
  - `4`: arch
- Source files: `model/labels.csv` and `model/labels-USSERIKAPRISE-A.csv` (currently identical).

### 3D vs 4D in this project

- **3D MRI:** single static volume.
- **4D/cine MRI:** sequence of 3D volumes over time (cardiac phases).
- Practical plan: establish a stable **3D baseline first**, then move to temporal modeling.

### Project stage (where we are now) — updated May 2026

- **Data curation (Stages A–C)** is in place: pairing audit → baseline manifest → **dataset manifest** → **leakage-aware splits** (see [`docs/DATASET.md`](DATASET.md)).
- **Model training:** primary entrypoint is **`scripts/monai_train_segmentation.py`** (MONAI 3D U-Net, Dice+CE, patch training + sliding-window validation). Deep walkthrough: [`docs/MONAI_TRAIN_SEGMENTATION.md`](MONAI_TRAIN_SEGMENTATION.md).
- **Eval / test (no training):** **`scripts/monai_eval_segmentation.py`** (any split, CSV + JSON + MD summaries, optional NRRD preds); **`scripts/monai_test_segmentation.py`** (external **`test`** only).
- **Cluster / GPU:** runbook for UMIACS Nexus scratch: [`docs/COMPUTE_NEXUS.md`](COMPUTE_NEXUS.md); repo **`slurm/`** templates (`train` / `eval` / `test`) use **`partition=tron`**, **`account=nexus`**, **`PYTHONUNBUFFERED=1`**, CUDA unload/load.
- **Latest baseline numbers (single primary split, Nexus-style run):** internal **val** mean foreground Dice **~0.72**; external **test** mean foreground Dice **~0.30** — large **domain shift**; treat external as honest generalization check, not for hyperparameter tuning.
- **Next focus:** **internal k-fold benchmarking** via **`build_splits.py --internal-cv-folds K`** + repeated training with **`--split-csv`** per fold; aggregate val Dice (mean ± std across folds) for model comparison.

### Quick re-entry checklist (when you forget context)

1. Read this section.
2. Open [`docs/DATASET.md`](DATASET.md) for CSV roles and regeneration commands.
3. Open [`docs/MONAI_TRAIN_SEGMENTATION.md`](MONAI_TRAIN_SEGMENTATION.md) for training math and flags.
4. Open [`docs/COMPUTE_NEXUS.md`](COMPUTE_NEXUS.md) if running on Slurm / scratch.
5. Check `reports/pairing_mismatches.csv` if you are revisiting pairing QA.

## What we know

- **Task (working hypothesis):** Learn or apply algorithms that segment pediatric cardiac structures from MRI, likely with interest in **time-resolved (cine)** data and single-ventricle anatomy. **Segmentation** = assigning each voxel (3D pixel) to a class (e.g. background, atrium, ventricle). **Ground truth** = expert-drawn masks used to train or evaluate models.
- **Internal vs external:** **Internal** usually means data from the primary institution’s cohort (here: standardized paths like `imaging/` + `labels_standardized/`). **External** often means other sites or formats (e.g. `TCH-…`, `ARK-…`) for generalization testing. Confirm definitions with the hospital.
- **`model.pt`:** Typically a **PyTorch checkpoint** (network weights, sometimes optimizer state). It is **not** self-explanatory: you need matching code (architecture + preprocessing) to run inference. **Not present in this Cursor workspace** as of last check — copy it into `model/` or point scripts at its path.
- **`labels.csv` / `labels-USSERIKAPRISE-A.csv`:** Both list **label IDs 1–4** with names **Atrium, Ventricle, CPC, Arch** and DICOM/SNOMED-style coding columns. Likely a **label legend** for segmentations and/or training. The two files are **identical** in the current copy.
- **Supervision:** If training uses these masks as targets, that is **supervised segmentation**. If you only run a pre-trained model, it is still *conceptually* supervised learning, but your immediate task may be **inference + evaluation** rather than training.
- **3D vs 4D:** **3D** = one static 3D volume. **4D** = 3D + **time** (cine), e.g. 30 cardiac phases. Example inspected: `Internal/imaging/PreFontan 58.nrrd` is **4D NRRD** (`dimension: 4`, `sizes: 30 120 160 56`, `kinds: list domain domain domain`) — **30** time frames, **120×160×56** spatial grid. SimpleITK reports this as a **vector image** with **30 components per pixel**.
- **Dataset inventory (this workspace):** **302** `.nrrd` files under the project root (mostly `External/`). **Internal** currently has **one** imaging file in `Internal/imaging/`. Many external **segmentations** use labels **[0,1,2,3,4]** (0 = background; 1–4 match the CSV). **Cropped** variants may **not** match full-resolution grid (verify pairs before training).

## What we still need to find out

- Exact **study goals** and **deliverable** (paper? clinical tool? benchmark? report on temporal models?).
- Whether the pipeline should be **3D per frame**, **3D aggregated**, or true **4D** modeling.
- **Canonical pairing rules** for image ↔ mask (naming, resampling, cropping).
- **Source and format** of `model.pt` and **repository** or script that defines the network.
- **Data use / IRB / sharing** constraints and whether external data may be merged for training.

## Repository layout (suggested)

```text
CNH-HeartMRI-Segmentation/
  docs/
    DATASET.md
    MONAI_TRAIN_SEGMENTATION.md
    progress_log.md
    COMPUTE_NEXUS.md
  scripts/
    monai_train_segmentation.py
    monai_eval_segmentation.py
    monai_test_segmentation.py
    monai_segmentation_common.py
    build_pairing_audit.py
    build_baseline_manifest.py
    build_dataset_manifest.py
    build_splits.py
    inspect_nrrd_dataset.py
    inspect_torch_checkpoint.py
  reports/
    dataset_manifest.csv
    splits/internal_train_val_external_test.csv
    ...
  requirements-training.txt
  README.md
```

## Commands run

```bash
# Inspect all NRRD under project root
python scripts/inspect_nrrd_dataset.py --root "/path/to/Heart MRI Segmentation"

# After model.pt is available (trusted file only if using --unsafe-full-pickle)
python scripts/inspect_torch_checkpoint.py model/model.pt
```

## Meeting prep (short)

- **Understanding:** Pediatric cardiac MRI segmentation with possible **cine/4D** focus; internal + external NRRD; label set {Atrium, Ventricle, CPC, Arch}; exploring baselines (existing weights vs nnU-Net/MONAI) under hospital guidance.
- **First-week plan:** Inventory + pairing QA in Slicer; document spacing/orientation; run checkpoint inspection on `model.pt`; optional baseline: repeat 3D Slicer visual overlap; then decide nnU-Net vs MONAI vs provided model.
- **Questions:** Definition of internal/external; target deliverable; required structures beyond the four labels; whether 4D is in scope; canonical preprocessing; where `model.pt` was trained and which code loads it.

# 15th April

### Decision alignment update

- Confirmed target direction: temporal/4D analysis is final deliverable.
- Agreed staged plan: per-frame 3D baseline -> 3D aggregate -> 4D modeling.
- Agreed to deprioritize legacy `model.pt` and develop a fresh reproducible model.
- Pairing hypothesis: masks may correspond to cropped images; must validate by overlay before filtering originals.
- Data strategy (proposed): internal baseline training + external holdout test first, then optional mixed/cross-site CV experiments.
- Script comprehension completed for:
  - `inspect_nrrd_dataset.py`
  - `inspect_torch_checkpoint.py`
  - `build_pairing_audit.py`

# 16th April

### Step A kickoff — pairing policy and visual QA

**Goal for Step A**

- Finalize which image-mask pairs are valid for baseline training.
- Resolve whether segmentations correspond to original volumes or cropped volumes.
- Tag each mismatch case as approved/rejected/review with evidence.

**Progress completed today**

- Started Step A execution.
- Updated `reports/pairing_mismatches.csv` with manual-review tracking fields:
  - `status` (default: `pending_review`)
  - `review_notes`
  - `slicer_checked` (default: `False`)
- Mismatch rows currently requiring review: **31**

**How we will review each case**

1. Load image + candidate mask in 3D Slicer.
2. Check anatomical overlap in axial/sagittal/coronal views.
3. Confirm whether mask aligns with cropped image, original image, or neither.
4. Update CSV:
  - `status = approved` if pairing is correct for baseline policy
  - `status = rejected` if pairing is incorrect/unusable
  - `status = review` if uncertain and needs clinical confirmation
  - `slicer_checked = True` after manual check
  - `review_notes` with short justification

**Step A policy draft (to validate)**

- Baseline v1 candidate policy: prioritize **cropped image + matching segmentation** pairs.
- Keep original images in dataset storage, but treat as non-baseline unless pairing is explicitly validated.
- Do not include rows with unresolved label anomalies until label mapping is confirmed.

**Immediate next actions**

- Manually review the first 10 rows in `reports/pairing_mismatches.csv` in Slicer.
- Confirm at least one internal case path for masks (internal currently appears mostly unpaired).
- After first 10 reviews, summarize acceptance/rejection counts and update policy.

# 21st April

### Baseline manifest (Step A deferred; strict labels default)

- **Decision:** Proceed with automated manifests; defer full Slicer Step A for later.
- **Geometry rules:** `has_pair`, `size_match`, `spacing_match` all true; `needs_review` false; exclude multi-component (4D/cine) volumes unless `--include-4d`.
- **Default label contract:** masks must contain **exactly** labels `{0,1,2,3,4}` (background + four structures from `model/labels.csv`). Rationale: `labels.csv` defines IDs **1–4** for anatomy and **0** is standard background; anything else is out-of-contract until CNH confirms (e.g. we saw a rare `5` in the audit).
- **Script:** `scripts/build_baseline_manifest.py` — each run writes **two included lists** plus excluded sidecars:
  - `reports/baseline_manifest_v1.csv` — strict **0–4** only (**132** rows on current audit)
  - `reports/baseline_manifest_v1_excluded.csv` — not in strict list (geometry fails **or** label contract fails)
  - `reports/baseline_manifest_v1_geometry_only.csv` — geometry-valid, **any** mask labels (**134** rows)
  - `reports/baseline_manifest_v1_geometry_only_excluded.csv` — geometry failures only (**31** rows)
- **Caveat:** Heuristic pairing + geometry match does not replace Slicer QA; revisit manifest after clinical review.

```bash
cd CNH-HeartMRI-Segmentation
python scripts/build_baseline_manifest.py --data-root .
```

### Stage B — dataset manifest (done)

- **Script:** `scripts/build_dataset_manifest.py` (reads strict `reports/baseline_manifest_v1.csv` by default).
- **Deliverables:**
  - `reports/dataset_manifest.csv` — **132** rows; columns: `sample_id`, `site`, `patient_id` (filename heuristic), `image_path`, `mask_path`, `is_cropped`, `is_4d`, `n_frames`, `size_xyz`, `spacing_xyz`, `labels_present`, `pairing_status`, `quality_flag`.
  - `reports/dataset_manifest_summary.md` — counts by site / cropped / 4D / labels / quality.
- **Command:** `python scripts/build_dataset_manifest.py --data-root .`
- **Paths (NRRD root):** NRRDs live in sibling **`../data`** (i.e. `Heart MRI Segmentation/data/` with `External/` and `Internal/`). The manifest script now tries **`../data` first**, then `./data`, then the repo root, then the parent folder. Override anytime: `--media-root "/absolute/path/to/data"`.
- **Documentation:** Stage B + C + path rules are summarized in **`docs/DATASET.md`** (May 2026 refresh).

### Stage C — splits (done)

- **Script:** `scripts/build_splits.py`
- **Outputs:**
  - `reports/splits/internal_train_val_external_test.csv` — internal → train/val by **patient_id**; external → **test** (19 rows).
  - `reports/splits/split_summary.txt` — quick counts.
- **CV outputs (optional):** `python scripts/build_splits.py --data-root . --cv-folds 5` writes **`mixed_site_cv_fold_*.csv`** (internal + external pooled). **`--internal-cv-folds 5`** writes **`internal_cv_fold_*.csv`** (internal only; for internal benchmarking). Defaults **`0`** for both unless requested.
- **Command:** `python scripts/build_splits.py --data-root .`
- **Convention:** New scripts get a long module docstring + section comments; chat walkthrough stays **very detailed** for each new file as we add code.

### Stage D — baseline protocol + metrics (historical note)

- Earlier repo versions committed **`docs/baseline_v1_protocol.md`** and **`docs/metrics_definition.md`** to freeze preprocessing and Dice reporting. Those files were **removed in the May 2026 simplification**; the substance is now carried by **`docs/DATASET.md`** (data contract) and **`docs/MONAI_TRAIN_SEGMENTATION.md`** (training + validation metric math).
- **Design principles retained:** label contract `{0..4}`, internal train/val vs external test, no test-driven tuning, reproducible split CSV + seed.

# 2 May 2026

### Repo simplification + MONAI training path

**Motivation:** Move from scattered “stage” docs and smoke artifacts to a **single reproducible training entrypoint** aligned with the existing manifest and split CSVs.

**What changed**

- **Removed** (cleanup): TotalSegmentator baseline scripts and reports; old `runs/baseline_v1` smoke tree; long-form docs (`PROJECT_HANDOFF`, `execution_plan`, `baseline_v1_protocol`, `metrics_definition`, `split_policy_v1`, `START_HERE`, TotalSeg guide). **`docs/progress_log.md`** was recreated from the last committed version and extended (this file).
- **Added / refreshed:**
  - `scripts/monai_train_segmentation.py` — MONAI **3D U-Net** (`in_channels=1`, `out_channels=5`), **DiceCELoss**, **AdamW**, optional **CUDA AMP**, **RandCropByPosNegLabeld** + **SpatialPadd** for fixed patch batches, **sliding_window_inference** on val (and test if `--final-test`), mean **foreground Dice (classes 1–4)** with explicit empty-set rule (imports **`monai_segmentation_common.py`**).
  - `scripts/monai_segmentation_common.py` — shared path resolution, transforms, U-Net builder, sliding-window + Dice helpers.
  - `scripts/monai_eval_segmentation.py` / `scripts/monai_test_segmentation.py` — checkpoint eval on val/train/test (CSV + JSON + MD); test wrapper forces **split=test**.
  - `slurm/train_a6000.slurm`, `slurm/eval_a6000.slurm`, `slurm/test_a6000.slurm` — GPU batch templates (edit `#SBATCH` / CUDA module for Nexus).
  - `requirements-training.txt` — `torch`, `monai`, `itk`, `pandas`, `numpy`.
  - `docs/DATASET.md` — full remake: label table, media root, every `reports/` artifact, regeneration order, link to training.
  - `docs/MONAI_TRAIN_SEGMENTATION.md` — line-by-line / math-heavy explanation of the training script.
  - `docs/COMPUTE_NEXUS.md` — how to run on **UMIACS Nexus** scratch (`/fs/nexus-scratch/...`), git + venv + Slurm sketch.
  - `.gitignore` extended for `runs/**/*.pt` etc.
- **`scripts/build_splits.py`:** default **`--cv-folds 0`** so mixed-site fold CSVs are opt-in.
- **Regenerated** `reports/splits/internal_train_val_external_test.csv` and `split_summary.txt` after the default change (counts unchanged: train 90, val 23, test 19).

**Smoke verification**

- Ran `monai_train_segmentation.py` locally for **1 epoch**, **2 train / 1 val** cases on CPU after fixing batch stacking (SpatialPadd + fixed crop size). Removed local `runs/segmentation` afterward so the repo stays clean until a real GPU run.

**Immediate next steps (engineering)**

1. **Push** this branch to the remote; on Nexus **clone or pull** into `/fs/nexus-scratch/anant04/<project-dir>` (see `docs/COMPUTE_NEXUS.md`).
2. **Rsync or stage NRRDs** to cluster-visible storage; pass `--media-root` to training.
3. **Done:** `scripts/monai_segmentation_common.py` (shared train/eval), **`scripts/monai_eval_segmentation.py`** (per-case CSV + summaries + optional NRRD preds), **`scripts/monai_test_segmentation.py`** (test-only wrapper), **`slurm/train_a6000.slurm`**, **`slurm/eval_a6000.slurm`**, **`slurm/test_a6000.slurm`**. Optional later: bootstrap CIs on external test Dice; training still supports **`--final-test`** for a quick one-shot test pass without Slurm.

# 3 May 2026

### Documentation refresh + internal CV splits

- **Nexus / Slurm lessons** (partition **`tron`**, QoS **≤4 CPUs**, CUDA module **`cuda/12.1.1`** after explicit unloads, **`PYTHONUNBUFFERED=1`**, correct **`REPO_ROOT`**) are reflected in [`docs/COMPUTE_NEXUS.md`](COMPUTE_NEXUS.md) and the **`slurm/*.slurm`** headers.
- **Dependencies:** `nibabel` is required for **`Orientationd`**; it is listed in **`requirements-training.txt`** (install via `pip install -r requirements-training.txt`).
- **Results logged above:** one full training run on the primary split → strong **internal val**, weak **external test**; external remains a no-tuning benchmark.
- **`scripts/build_splits.py`:** added **`--internal-cv-folds K`** to emit **`reports/splits/internal_cv_fold_*.csv`** (internal patients only, patient-level k-fold). Distinct from **`--cv-folds`** (**mixed_site** pool). See [`docs/DATASET.md`](DATASET.md) §3.4.
- **Next engineering step:** run **K** trainings (or Slurm array) with **`--split-csv`** per fold, record **`best_val_mean_fg_dice`** per fold from each run’s `summary.json` (or re-run `monai_eval_segmentation.py --split val`), then report mean ± std for internal benchmarking.

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

### Project stage (where we are now)

- We are in **data understanding + QA phase**, not full model training yet.
- We already have scripts for:
  - dataset inspection (`scripts/inspect_nrrd_dataset.py`)
  - checkpoint inspection (`scripts/inspect_torch_checkpoint.py`)
  - image-mask pairing audit (`scripts/build_pairing_audit.py`)
- Immediate focus: pairing correctness, geometry consistency, and baseline protocol definition.

### Quick re-entry checklist (when you forget context)

1. Read this section.
2. Check **Latest execution snapshot**.
3. Open `reports/pairing_mismatches.csv` and review unresolved cases.
4. Continue with **Pre-next-meeting execution plan**.

## What we know

- **Task (working hypothesis):** Learn or apply algorithms that segment pediatric cardiac structures from MRI, likely with interest in **time-resolved (cine)** data and single-ventricle anatomy. **Segmentation** = assigning each voxel (3D pixel) to a class (e.g. background, atrium, ventricle). **Ground truth** = expert-drawn masks used to train or evaluate models.
- **Internal vs external:** **Internal** usually means data from the primary institution’s cohort (here: standardized paths like `imaging/` + `labels_standardized/`). **External** often means other sites or formats (e.g. `TCH-…`, `ARK-…`) for generalization testing. Confirm definitions with the hospital.
- `**model.pt`:** Typically a **PyTorch checkpoint** (network weights, sometimes optimizer state). It is **not** self-explanatory: you need matching code (architecture + preprocessing) to run inference. **Not present in this Cursor workspace** as of last check — copy it into `model/` or point scripts at its path.
- `**labels.csv` / `labels-USSERIKAPRISE-A.csv`:** Both list **label IDs 1–4** with names **Atrium, Ventricle, CPC, Arch** and DICOM/SNOMED-style coding columns. Likely a **label legend** for segmentations and/or training. The two files are **identical** in the current copy.
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
Heart MRI Segmentation/
  scripts/
    inspect_nrrd_dataset.py    # scan .nrrd, shapes, spacing, label uniques for *seg* files
    inspect_torch_checkpoint.py
  notebooks/                   # optional exploratory visualization
  model/
    labels*.csv
    model.pt                   # when available
  Internal/ …  External/ …
  progress_log.md
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


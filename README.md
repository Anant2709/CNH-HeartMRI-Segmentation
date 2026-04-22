# Heart MRI Segmentation

Quality-controlled baseline pipeline for cardiac MRI segmentation, with a staged roadmap from dataset curation to reproducible model training and later temporal (4D) extension.

## Project Goal
Build a scientifically valid and reproducible segmentation baseline by first fixing data quality and split integrity, then implementing a minimal end-to-end training pipeline.

## Current Status
- Planning and quality gates are documented in `execution_plan.md`.
- Pairing audit artifacts exist in `reports/` and support Stage A (pairing policy finalization).
- Dataset and training pipeline implementation are in progress.

## Repository Structure
- `execution_plan.md` - manager-facing staged execution roadmap (Stages A-F).
- `progress_log.md` - running project notes and progress updates.
- `reports/`
  - `pairing_mismatches.csv` - mismatch list for manual review/classification.
  - `pairing_audit.csv` - pairing audit output.
- `scripts/`
  - `build_pairing_audit.py` - generates/updates pairing audit artifacts.
  - `inspect_nrrd_dataset.py` - dataset inspection utilities.
  - `inspect_torch_checkpoint.py` - checkpoint inspection utility.
- `model/`
  - `labels.csv` and related label mapping/reference files.
- `Internal/imaging/`
  - local imaging data samples (not intended as portable GitHub benchmark assets).

## Staged Roadmap (v1)
1. **Stage A - Finalize Pairing Policy**
   - classify mismatches (expected/recoverable/unusable),
   - verify representative cases in 3D Slicer,
   - lock baseline pairing and label policy.
2. **Stage B - Build Trainable Dataset Manifest**
   - create deterministic `dataset_manifest.csv`,
   - include metadata, labels, and pairing status,
   - exclude unresolved rows.
3. **Stage C - Define Leakage-Safe Splits**
   - primary: internal train/val + external holdout test,
   - alternate: mixed-site CV with strict patient-level grouping.
4. **Stage D - Freeze Baseline v1 Protocol**
   - per-frame 3D baseline design,
   - preprocessing, labels, metrics, runtime constraints.
5. **Stage E - Implement Minimal Train/Eval Pipeline**
   - loader, preprocessing, train/val loops, logging,
   - checkpoints + evaluation report.
6. **Stage F - Plan 3D Aggregate -> 4D Transition**
   - shortlist low-risk temporal extension after baseline stability.

## Quality Gates
Progression between stages is gated by explicit criteria, including:
- trusted image-mask alignment by visual QA,
- manifest rows with valid paths and approved pairing,
- zero patient leakage across splits,
- reproducible baseline protocol and first full train+eval run.

## Immediate Next Milestones
- Complete Stage A pairing decisions and visual QA sign-off.
- Implement Stage B manifest builder and summary report generator.

## Notes for GitHub
- This repository is under active development; interfaces and scripts may evolve quickly.
- Large/private medical imaging data should remain outside public version control and be referenced via manifests.

---

For detailed deliverables, timeline, and risk register, see `execution_plan.md`.

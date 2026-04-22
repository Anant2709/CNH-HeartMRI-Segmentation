#!/usr/bin/env python3
"""
================================================================================
Stage C: leakage-aware dataset splits
================================================================================

WHAT this script does (high level)
---------------------------------
Reads ``reports/dataset_manifest.csv`` (Stage B output) and writes CSV files
that assign every sample to a split: **train**, **val**, or **test**, in a way
that avoids **patient leakage**.

WHY patient-level splitting matters
-----------------------------------
If two rows are different scans or crops from the *same patient* and one row
goes to **train** while another goes to **val**, the model can indirectly
"memorize" that patient. Validation scores would look **optimistic** and would
not reflect true generalization. We therefore assign splits by **patient_id**
(Stage B heuristic), not by individual rows.

WHY two strategies are produced
--------------------------------
1) **Primary baseline (execution plan):**
   - **internal** → split into **train** + **val** only.
   - **external** → **all test** (holdout). You tune on internal val only;
     external is reserved for a later honest benchmark.

2) **Mixed-site k-fold (secondary):**
   - Pool **internal + external** patients, partition into k folds.
   - Useful later for "mixed training + cross-site validation" experiments.
   - Each fold file marks rows as **train** or **val** *for that fold only*;
     there is no separate test column in those files (typical CV pattern).

Outputs (under ``reports/splits/`` by default)
----------------------------------------------
- ``internal_train_val_external_test.csv`` — one row per manifest row;
  column ``split`` in {train, val, test}.
- ``mixed_site_cv_fold_00.csv`` … ``mixed_site_cv_fold_{k-1}.csv`` —
  for each fold, which rows are train vs val for that fold.

No third-party deps (stdlib only).

Example
-------
  cd CNH-HeartMRI-Segmentation
  python scripts/build_splits.py --data-root .
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_manifest(path: Path) -> list[dict]:
    """Load all rows from dataset_manifest.csv as dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    # -------------------------------------------------------------------------
    # CLI: where inputs/outputs live and how aggressively to split internal val
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Build Stage C train/val/test and optional mixed-site CV split CSVs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository root (contains reports/).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Input dataset_manifest.csv (default: <data-root>/reports/dataset_manifest.csv).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-root>/reports/splits).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of *internal* patients (not rows) assigned to val (default 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed so splits are reproducible.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds for mixed_site_cv outputs (default 5). Use 0 to skip CV files.",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else data_root / "reports" / "dataset_manifest.csv"
    )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else data_root / "reports" / "splits"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        print("Run: python scripts/build_dataset_manifest.py --data-root .", file=sys.stderr)
        sys.exit(1)

    rows = read_manifest(manifest_path)

    # -------------------------------------------------------------------------
    # Keep only rows that passed automated file checks in Stage B
    # -------------------------------------------------------------------------
    usable = [r for r in rows if r.get("quality_flag", "").strip() == "ok_automated"]
    if not usable:
        print("No rows with quality_flag==ok_automated; fix paths or manifest.", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Partition rows by site (string from manifest: "internal" / "external")
    # -------------------------------------------------------------------------
    internal = [r for r in usable if r.get("site", "").lower() == "internal"]
    external = [r for r in usable if r.get("site", "").lower() == "external"]
    unknown = [r for r in usable if r.get("site", "").lower() not in ("internal", "external")]
    if unknown:
        print(f"Warning: {len(unknown)} rows with unexpected site values; excluded from splits.")

    # -------------------------------------------------------------------------
    # Primary split: internal patients -> train/val; external -> all test
    #
    # We collect unique patient_id values *within internal only*, shuffle them
    # with a fixed seed, then assign the first (1 - val_fraction) fraction to
    # train and the rest to val. Every row for that patient_id inherits the same
    # split (so two volumes from same patient never land in different splits).
    # -------------------------------------------------------------------------
    internal_patients = sorted({r["patient_id"] for r in internal})
    rng = random.Random(args.seed)
    rng.shuffle(internal_patients)

    n_pat = len(internal_patients)
    if n_pat == 0:
        print("No internal rows in manifest; cannot form internal train/val.", file=sys.stderr)
        sys.exit(1)

    # At least one patient in val when we have 2+ patients and val_fraction > 0
    n_val = int(round(args.val_fraction * n_pat))
    if args.val_fraction > 0 and n_pat >= 2:
        n_val = max(1, min(n_val, n_pat - 1))
    else:
        n_val = 0

    val_set = set(internal_patients[:n_val])
    train_set = set(internal_patients[n_val:])

    patient_split_primary: dict[str, str] = {}
    for pid in train_set:
        patient_split_primary[pid] = "train"
    for pid in val_set:
        patient_split_primary[pid] = "val"

    # External rows always go to test. Refuse if patient_id collides with internal (same key would leak).
    ext_pids = {r["patient_id"] for r in external}
    collision = ext_pids & (train_set | val_set)
    if collision:
        print(
            "ERROR: patient_id appears in both internal and external under the same id. "
            "Disambiguate in Stage B (e.g. prefix patient_id with site) before splitting.",
            file=sys.stderr,
        )
        print(f"Colliding ids (sample): {sorted(collision)[:20]}", file=sys.stderr)
        sys.exit(1)
    for r in external:
        patient_split_primary[r["patient_id"]] = "test"

    primary_rows: list[dict] = []
    for r in internal + external:
        pid = r["patient_id"]
        primary_rows.append(
            {
                "sample_id": r["sample_id"],
                "patient_id": pid,
                "site": r["site"],
                "split": patient_split_primary[pid],
                "image_path": r["image_path"],
                "mask_path": r["mask_path"],
            }
        )
    primary_rows.sort(key=lambda x: (x["split"], x["site"], x["patient_id"], x["sample_id"]))

    primary_path = out_dir / "internal_train_val_external_test.csv"
    fields_primary = ["sample_id", "patient_id", "site", "split", "image_path", "mask_path"]
    with primary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields_primary)
        w.writeheader()
        w.writerows(primary_rows)

    # -------------------------------------------------------------------------
    # Sanity checks: no patient appears in more than one split (internal+external pooled check)
    # -------------------------------------------------------------------------
    by_patient: dict[str, set[str]] = defaultdict(set)
    for row in primary_rows:
        by_patient[row["patient_id"]].add(row["split"])
    leaks = [pid for pid, splits in by_patient.items() if len(splits) > 1]
    if leaks:
        print(f"ERROR: patient leakage detected for: {leaks[:20]}", file=sys.stderr)
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Mixed-site k-fold: pool ALL usable patients (internal + external),
    # sort+shuffle for reproducibility, assign fold = index % k.
    # For fold f, rows whose patient_fold == f are "val"; others "train".
    # -------------------------------------------------------------------------
    if args.cv_folds and args.cv_folds > 1:
        all_patients = sorted({r["patient_id"] for r in internal + external})
        rng_cv = random.Random(args.seed)
        rng_cv.shuffle(all_patients)
        patient_fold_index = {pid: i % args.cv_folds for i, pid in enumerate(all_patients)}

        for fold_id in range(args.cv_folds):
            fold_rows: list[dict] = []
            for r in internal + external:
                pid = r["patient_id"]
                pf = patient_fold_index[pid]
                role = "val" if pf == fold_id else "train"
                fold_rows.append(
                    {
                        "sample_id": r["sample_id"],
                        "patient_id": pid,
                        "site": r["site"],
                        "fold_id": str(fold_id),
                        "split": role,
                        "image_path": r["image_path"],
                        "mask_path": r["mask_path"],
                    }
                )
            fold_rows.sort(key=lambda x: (x["split"], x["site"], x["patient_id"], x["sample_id"]))
            fp = out_dir / f"mixed_site_cv_fold_{fold_id:02d}.csv"
            fields_cv = [
                "sample_id",
                "patient_id",
                "site",
                "fold_id",
                "split",
                "image_path",
                "mask_path",
            ]
            with fp.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields_cv)
                w.writeheader()
                w.writerows(fold_rows)

    # -------------------------------------------------------------------------
    # Human-readable summary for logs / meetings
    # -------------------------------------------------------------------------
    summary_path = out_dir / "split_summary.txt"

    c = Counter(r["split"] for r in primary_rows)
    lines = [
        "Stage C split summary",
        f"manifest: {manifest_path}",
        f"seed: {args.seed}",
        f"internal val_fraction (patients): {args.val_fraction}",
        f"internal patients: {n_pat} (train {len(train_set)}, val {len(val_set)})",
        f"external rows (all test): {len(external)}",
        "primary split counts:",
    ]
    for k in sorted(c.keys()):
        lines.append(f"  {k}: {c[k]}")
    if args.cv_folds and args.cv_folds > 1:
        lines.append(f"mixed_site_cv: {args.cv_folds} fold files written")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {primary_path}")
    print(f"Wrote {summary_path}")
    if args.cv_folds and args.cv_folds > 1:
        for fold_id in range(args.cv_folds):
            print(f"Wrote {out_dir / f'mixed_site_cv_fold_{fold_id:02d}.csv'}")


if __name__ == "__main__":
    main()

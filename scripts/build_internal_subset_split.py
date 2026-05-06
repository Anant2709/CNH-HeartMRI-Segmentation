#!/usr/bin/env python3
"""
Create an internal-only train/val/test split CSV (patient-level, leakage-safe).

Purpose:
- Isolate whether performance drop is external-data specific by testing on a
  held-out subset of INTERNAL patients only.

Output columns match training/eval expectations:
sample_id, patient_id, site, split, image_path, mask_path

Example:
  python scripts/build_internal_subset_split.py --data-root . \
    --test-fraction 0.2 --val-fraction 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser(description="Build internal-only train/val/test split (patient-level).")
    p.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Default: <data-root>/reports/dataset_manifest.csv",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Default: <data-root>/reports/splits/internal_train_val_internal_test.csv",
    )
    p.add_argument("--test-fraction", type=float, default=0.2, help="Internal patient fraction for test.")
    p.add_argument("--val-fraction", type=float, default=0.2, help="From remaining internal patients, fraction for val.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_root = args.data_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else data_root / "reports" / "dataset_manifest.csv"
    )
    out_csv = (
        args.out_csv.expanduser().resolve()
        if args.out_csv
        else data_root / "reports" / "splits" / "internal_train_val_internal_test.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file():
        print(f"Missing manifest: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    rows = _read_csv(manifest_path)
    usable_internal = [
        r
        for r in rows
        if r.get("quality_flag", "").strip() == "ok_automated" and r.get("site", "").strip().lower() == "internal"
    ]
    if not usable_internal:
        print("No usable internal rows found in manifest.", file=sys.stderr)
        sys.exit(1)

    pids = sorted({r["patient_id"] for r in usable_internal})
    n_pat = len(pids)
    if n_pat < 3:
        print("Need at least 3 internal patients to form train/val/test.", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(pids)

    n_test = int(round(args.test_fraction * n_pat))
    n_test = max(1, min(n_test, n_pat - 2))

    remaining = n_pat - n_test
    n_val = int(round(args.val_fraction * remaining))
    n_val = max(1, min(n_val, remaining - 1))

    test_pids = set(pids[:n_test])
    val_pids = set(pids[n_test : n_test + n_val])
    train_pids = set(pids[n_test + n_val :])

    patient_split: dict[str, str] = {}
    for pid in train_pids:
        patient_split[pid] = "train"
    for pid in val_pids:
        patient_split[pid] = "val"
    for pid in test_pids:
        patient_split[pid] = "test"

    out_rows: list[dict[str, str]] = []
    for r in usable_internal:
        pid = r["patient_id"]
        out_rows.append(
            {
                "sample_id": r["sample_id"],
                "patient_id": pid,
                "site": "internal",
                "split": patient_split[pid],
                "image_path": r["image_path"],
                "mask_path": r["mask_path"],
            }
        )
    out_rows.sort(key=lambda x: (x["split"], x["patient_id"], x["sample_id"]))

    fields = ["sample_id", "patient_id", "site", "split", "image_path", "mask_path"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    c = Counter(r["split"] for r in out_rows)
    print(f"Wrote {out_csv}")
    print(
        f"Internal patients: {n_pat} (train={len(train_pids)}, val={len(val_pids)}, test={len(test_pids)}) | "
        f"rows: train={c.get('train', 0)}, val={c.get('val', 0)}, test={c.get('test', 0)}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build train-ready manifests from reports/pairing_audit.csv.

Every run writes **two included manifests**:
  1. baseline_manifest_v1.csv
     Geometry-valid rows AND mask labels exactly {0,1,2,3,4} (default training list).
  2. baseline_manifest_v1_geometry_only.csv
     Same geometry rules but **any** mask label values (for debugging / later relabeling).

Also writes excluded rows + reasons for each list:
  - baseline_manifest_v1_excluded.csv
  - baseline_manifest_v1_geometry_only_excluded.csv

Geometry rules (from build_pairing_audit.py):
  has_pair, size_match, spacing_match true; needs_review false.
  By default excludes multi-component (cine) volumes; pass --include-4d to keep them.

Why {0..4} is the default label contract:
  model/labels.csv defines structure IDs 1–4; voxels labeled 0 are background. That is
  five classes total 0–4. Any other integer in a mask is treated as out-of-contract until
  the team confirms it (we saw rare cases like 5 in the audit).

No pandas dependency.

Example:
  python scripts/build_baseline_manifest.py --data-root .
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def norm_label_str(s: str) -> str:
    """Normalize mask_labels cell for comparison, e.g. '[0, 1, 2, 3, 4]' -> '01234'."""
    return re.sub(r"[^\d]", "", str(s))


# Expected multiclass mask: background + 4 structures from labels.csv
STRICT_LABELS_NORM = "01234"


def geometry_exclude_reasons(row: dict, include_4d: bool) -> list[str]:
    reasons: list[str] = []
    try:
        ncomp = int(float(row.get("img_components", "1")))
    except ValueError:
        ncomp = 1

    if not parse_bool(row.get("has_pair", "False")):
        reasons.append("no_pair")
    if not parse_bool(row.get("size_match", "False")):
        reasons.append("size_mismatch")
    if not parse_bool(row.get("spacing_match", "False")):
        reasons.append("spacing_mismatch")
    if parse_bool(row.get("needs_review", "True")):
        reasons.append("needs_review")
    if not include_4d and ncomp > 1:
        reasons.append("multi_component_skip_4d")
    return reasons


def manifest_row(i: int, case: str, row: dict) -> dict:
    try:
        ncomp = int(float(row.get("img_components", "1")))
    except ValueError:
        ncomp = 1
    sample_id = f"v1_{i:04d}_{Path(case).stem.replace(' ', '_')}"
    return {
        "sample_id": sample_id,
        "image_path": case,
        "mask_path": row.get("mask_path") or "",
        "site": row.get("site", ""),
        "is_cropped_image": row.get("is_cropped_image", ""),
        "img_components": ncomp,
        "mask_labels": row.get("mask_labels", "") or "",
        "img_size_xyz": row.get("img_size_xyz", ""),
        "img_spacing_xyz": row.get("img_spacing_xyz", ""),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build baseline training manifests from pairing audit.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Dataset root (paths in CSV are relative to this)",
    )
    parser.add_argument(
        "--pairing-audit",
        type=Path,
        default=None,
        help="Path to pairing_audit.csv (default: <data-root>/reports/pairing_audit.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <data-root>/reports)",
    )
    parser.add_argument(
        "--include-4d",
        action="store_true",
        help="Include images with img_components > 1 (cine); default excludes for 3D scalar v1",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root.expanduser().resolve()
    audit_path = (
        args.pairing_audit.expanduser().resolve()
        if args.pairing_audit
        else data_root / "reports" / "pairing_audit.csv"
    )
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else data_root / "reports"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    path_strict = out_dir / "baseline_manifest_v1.csv"
    path_strict_excl = out_dir / "baseline_manifest_v1_excluded.csv"
    path_geom = out_dir / "baseline_manifest_v1_geometry_only.csv"
    path_geom_excl = out_dir / "baseline_manifest_v1_geometry_only_excluded.csv"

    if not audit_path.is_file():
        print(f"Missing pairing audit: {audit_path}", file=sys.stderr)
        print("Run: python scripts/build_pairing_audit.py --root <data-root>", file=sys.stderr)
        sys.exit(1)

    with audit_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    included_strict: list[dict] = []
    included_geom: list[dict] = []
    excluded_strict: list[dict] = []
    excluded_geom: list[dict] = []

    for i, row in enumerate(rows):
        case = row.get("case_image", "")
        greasons = geometry_exclude_reasons(row, args.include_4d)
        labels_norm = norm_label_str(row.get("mask_labels", "") or "")

        if greasons:
            r = {**row, "exclude_reason": ";".join(greasons)}
            excluded_strict.append(r)
            excluded_geom.append(r)
            continue

        out = manifest_row(i, case, row)
        included_geom.append(out)

        if labels_norm == STRICT_LABELS_NORM:
            included_strict.append(out)
        else:
            excluded_strict.append(
                {
                    **row,
                    "exclude_reason": "labels_not_exactly_0_4",
                }
            )

    field_inc = [
        "sample_id",
        "image_path",
        "mask_path",
        "site",
        "is_cropped_image",
        "img_components",
        "mask_labels",
        "img_size_xyz",
        "img_spacing_xyz",
    ]
    write_csv(path_strict, field_inc, included_strict)
    write_csv(path_geom, field_inc, included_geom)

    ex_fields_s = list(excluded_strict[0].keys()) if excluded_strict else list(rows[0].keys()) + ["exclude_reason"]
    ex_fields_g = list(excluded_geom[0].keys()) if excluded_geom else list(rows[0].keys()) + ["exclude_reason"]
    write_csv(path_strict_excl, ex_fields_s, excluded_strict)
    write_csv(path_geom_excl, ex_fields_g, excluded_geom)

    print(f"Data root: {data_root}")
    print(f"Audit: {audit_path}")
    print(f"Strict manifest (labels 0–4 only): {len(included_strict)} rows -> {path_strict}")
    print(f"Strict excluded: {len(excluded_strict)} rows -> {path_strict_excl}")
    print(f"Geometry-only manifest: {len(included_geom)} rows -> {path_geom}")
    print(f"Geometry-only excluded: {len(excluded_geom)} rows -> {path_geom_excl}")


if __name__ == "__main__":
    main()

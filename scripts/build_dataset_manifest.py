#!/usr/bin/env python3
"""
Stage B: build reports/dataset_manifest.csv — single source of truth for trainable rows.

Reads the strict baseline list (default: reports/baseline_manifest_v1.csv) produced by
build_baseline_manifest.py, adds Stage-B columns, verifies files exist, stable-sorts rows,
and writes reports/dataset_manifest_summary.md.

Usage:
  python scripts/build_dataset_manifest.py --data-root .

Default NRRD search locations (unless --media-root is set), in order:
  1. <data-root>/../data   (sibling `data/` folder — typical layout)
  2. <data-root>/data
  3. <data-root>
  4. <data-root>/..
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def parse_bool(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def patient_group_id(image_rel: str) -> str:
    """
    Heuristic ID for leakage-safe splitting: same case should map to one key.
    Strips a trailing ' cropped' token so full-FOV and crop variants may share a key.
    """
    stem = Path(image_rel).stem.lower().strip()
    if stem.endswith(" cropped"):
        stem = stem[: -len(" cropped")].strip()
    stem = stem.replace(" ", "_")
    return stem or Path(image_rel).stem.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage B dataset_manifest.csv")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Dataset root; image_path and mask_path are relative to this",
    )
    parser.add_argument(
        "--from-manifest",
        type=Path,
        default=None,
        help="Input CSV (default: <data-root>/reports/baseline_manifest_v1.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: <data-root>/reports)",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=None,
        help="Folder containing External/ and Internal/ (default: try ../data, ./data, data-root, parent)",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root.expanduser().resolve()
    media_root_explicit = args.media_root.expanduser().resolve() if args.media_root else None
    src = (
        args.from_manifest.expanduser().resolve()
        if args.from_manifest
        else data_root / "reports" / "baseline_manifest_v1.csv"
    )
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else data_root / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "dataset_manifest.csv"
    out_md = out_dir / "dataset_manifest_summary.md"

    if not src.is_file():
        print(f"Missing input manifest: {src}", file=sys.stderr)
        print("Run: python scripts/build_baseline_manifest.py --data-root <data-root>", file=sys.stderr)
        sys.exit(1)

    with src.open(newline="", encoding="utf-8") as f:
        rows_in = list(csv.DictReader(f))

    def resolve_media_roots() -> list[Path]:
        """Paths to try for on-disk NRRD checks (manifest paths are relative to this folder)."""
        if media_root_explicit is not None:
            return [media_root_explicit]
        candidates = [
            data_root.parent / "data",
            data_root / "data",
            data_root,
            data_root.parent,
        ]
        seen: set[Path] = set()
        out: list[Path] = []
        for p in candidates:
            try:
                r = p.resolve()
            except OSError:
                continue
            if r in seen:
                continue
            seen.add(r)
            if p.is_dir():
                out.append(p)
        return out if out else [data_root]

    media_roots = resolve_media_roots()

    def exists_pair(img_rel: str, mask_rel: str) -> tuple[bool, Path | None]:
        for root in media_roots:
            ia, ma = root / img_rel, root / mask_rel
            if ia.is_file() and ma.is_file():
                return True, root
        return False, None

    rows_out: list[dict] = []
    for r in rows_in:
        img_rel = r.get("image_path", "").strip()
        mask_rel = r.get("mask_path", "").strip()
        ok_files, found_root = exists_pair(img_rel, mask_rel)

        try:
            ncomp = int(float(r.get("img_components", "1")))
        except ValueError:
            ncomp = 1

        is_cropped = parse_bool(r.get("is_cropped_image", "False"))
        is_4d = ncomp > 1
        n_frames = max(1, ncomp)

        if ok_files:
            quality = "ok_automated"
        else:
            quality = "image_or_mask_missing_under_media_roots"

        rows_out.append(
            {
                "sample_id": r.get("sample_id", ""),
                "site": r.get("site", ""),
                "patient_id": patient_group_id(img_rel),
                "image_path": img_rel,
                "mask_path": mask_rel,
                "is_cropped": is_cropped,
                "is_4d": is_4d,
                "n_frames": n_frames,
                "size_xyz": r.get("img_size_xyz", ""),
                "spacing_xyz": r.get("img_spacing_xyz", ""),
                "labels_present": r.get("mask_labels", ""),
                "pairing_status": "automated_strict_v1",
                "quality_flag": quality,
            }
        )

    rows_out.sort(key=lambda x: (x["site"], x["patient_id"], x["image_path"]))

    fields = [
        "sample_id",
        "site",
        "patient_id",
        "image_path",
        "mask_path",
        "is_cropped",
        "is_4d",
        "n_frames",
        "size_xyz",
        "spacing_xyz",
        "labels_present",
        "pairing_status",
        "quality_flag",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    # Summary markdown
    n = len(rows_out)
    by_site = Counter(r["site"] for r in rows_out)
    by_crop = Counter(str(r["is_cropped"]) for r in rows_out)
    by_4d = Counter(str(r["is_4d"]) for r in rows_out)
    by_quality = Counter(r["quality_flag"] for r in rows_out)
    by_labels = Counter(r["labels_present"] for r in rows_out)

    try:
        src_rel = src.relative_to(data_root)
    except ValueError:
        src_rel = src
    try:
        out_rel = out_csv.relative_to(data_root)
    except ValueError:
        out_rel = out_csv

    lines = [
        "# Dataset manifest summary (Stage B)",
        "",
        f"- **Source:** `{src_rel}`",
        f"- **Output:** `{out_rel}`",
        f"- **Total rows:** {n}",
        "",
        "## Paths checked for NRRD existence",
        "",
    ]
    for p in media_roots:
        lines.append(f"- `{p}`")
    lines.append("")
    lines.append("## By site")
    lines.append("")
    for k, v in sorted(by_site.items()):
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(["", "## Cropped vs not (is_cropped)", ""])
    for k, v in sorted(by_crop.items()):
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(["", "## 4D / multi-frame (is_4d)", ""])
    for k, v in sorted(by_4d.items()):
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(["", "## labels_present (unique strings in manifest)", ""])
    for k, v in by_labels.most_common():
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(["", "## quality_flag", ""])
    for k, v in sorted(by_quality.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `patient_id` is a **heuristic** grouping key from the image filename (not necessarily a hospital MRN). Use for split logic until official IDs are provided.",
            "- `pairing_status` reflects that rows came from automated strict v1 filtering (geometry + labels 0–4).",
            "- File existence is checked under `--media-root` if set, else in order: **`../data`**, `./data`, `--data-root`, parent of `--data-root` (manifest paths are like `External/...` under that folder).",
            "- Rows with `quality_flag` other than `ok_automated` need path fixes or `--media-root` before training.",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_csv} ({len(rows_out)} rows)")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()

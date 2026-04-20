#!/usr/bin/env python3
"""
Build a practical image-mask pairing audit CSV for NRRD datasets.

Outputs:
  - reports/pairing_audit.csv
  - reports/pairing_mismatches.csv

Usage:
  python scripts/build_pairing_audit.py --root "/path/to/Heart MRI Segmentation"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import SimpleITK as sitk
except ImportError:
    print("Please install SimpleITK: pip install SimpleITK", file=sys.stderr)
    sys.exit(1)


def is_seg_file(path: Path) -> bool:
    n = path.name.lower()
    return ("seg" in n) or ("segmentation" in n) or ("label" in n)


def normalize_stem(path: Path) -> str:
    s = path.stem.lower()
    s = s.replace("segmentation", "")
    s = s.replace("seg", "")
    s = s.replace("cropped", "")
    s = re.sub(r"[\s_\-]+", " ", s)
    return s.strip()


def get_site(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    if "internal" in parts:
        return "internal"
    if "external" in parts:
        return "external"
    return "unknown"


def image_meta(path: Path) -> dict:
    img = sitk.ReadImage(str(path))
    arr_shape = tuple(int(x) for x in sitk.GetArrayFromImage(img).shape)
    return {
        "size_xyz": tuple(float(x) for x in img.GetSize()),
        "spacing_xyz": tuple(float(x) for x in img.GetSpacing()),
        "n_components": int(img.GetNumberOfComponentsPerPixel()),
        "pixel_type": img.GetPixelIDTypeAsString(),
        "array_shape": arr_shape,
    }


def seg_labels(path: Path) -> list[int]:
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    vals = np.unique(arr)
    vals = vals[(vals >= 0) & (vals <= 255)]
    return [int(v) for v in vals.tolist()]


def choose_mask(image_path: Path, masks: list[Path]) -> Path | None:
    if not masks:
        return None
    image_site = get_site(image_path)
    site_masks = [m for m in masks if get_site(m) == image_site]
    if not site_masks:
        return None

    img_norm = normalize_stem(image_path)
    ranked = []
    for m in site_masks:
        m_norm = normalize_stem(m)
        score = 0
        if img_norm == m_norm:
            score += 100
        if img_norm in m_norm or m_norm in img_norm:
            score += 20
        if ("cropped" in image_path.stem.lower()) == ("cropped" in m.stem.lower()):
            score += 10
        if image_path.parent == m.parent:
            score += 5
        ranked.append((score, m))
    ranked.sort(key=lambda x: x[0], reverse=True)
    # Require non-trivial name overlap; avoids unrelated pairings.
    return ranked[0][1] if ranked[0][0] >= 20 else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create image-mask pairing audit CSV.")
    parser.add_argument("--root", type=Path, required=True, help="Dataset root directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Output folder for CSV reports",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    all_nrrd = sorted(root.rglob("*.nrrd"))
    masks = [p for p in all_nrrd if is_seg_file(p)]
    images = [p for p in all_nrrd if not is_seg_file(p)]

    rows = []
    for img_path in images:
        cand = choose_mask(img_path, masks)
        img_info = image_meta(img_path)

        row = {
            "case_image": str(img_path.relative_to(root)),
            "site": get_site(img_path),
            "is_cropped_image": "cropped" in img_path.stem.lower(),
            "img_size_xyz": img_info["size_xyz"],
            "img_spacing_xyz": img_info["spacing_xyz"],
            "img_array_shape": img_info["array_shape"],
            "img_components": img_info["n_components"],
            "img_pixel_type": img_info["pixel_type"],
            "mask_path": None,
            "mask_size_xyz": None,
            "mask_spacing_xyz": None,
            "mask_array_shape": None,
            "mask_labels": None,
            "size_match": None,
            "spacing_match": None,
            "has_pair": False,
            "needs_review": True,
        }

        if cand is not None:
            mask_info = image_meta(cand)
            labels = seg_labels(cand)
            size_match = tuple(mask_info["size_xyz"]) == tuple(img_info["size_xyz"])
            spacing_match = np.allclose(
                np.array(mask_info["spacing_xyz"]), np.array(img_info["spacing_xyz"]), atol=1e-6
            )
            row.update(
                {
                    "mask_path": str(cand.relative_to(root)),
                    "mask_size_xyz": mask_info["size_xyz"],
                    "mask_spacing_xyz": mask_info["spacing_xyz"],
                    "mask_array_shape": mask_info["array_shape"],
                    "mask_labels": labels,
                    "size_match": bool(size_match),
                    "spacing_match": bool(spacing_match),
                    "has_pair": True,
                    "needs_review": not (size_match and spacing_match),
                }
            )

        rows.append(row)

    out_dir = args.out_dir if args.out_dir.is_absolute() else (root / args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["site", "case_image"])
    df.to_csv(out_dir / "pairing_audit.csv", index=False)

    mismatch_df = df[
        (~df["has_pair"])
        | (df["size_match"] == False)  # noqa: E712
        | (df["spacing_match"] == False)  # noqa: E712
    ].copy()
    mismatch_df.to_csv(out_dir / "pairing_mismatches.csv", index=False)

    print(f"Total .nrrd files: {len(all_nrrd)}")
    print(f"Images (non-seg): {len(images)}")
    print(f"Masks (seg-like): {len(masks)}")
    print(f"Paired images: {int(df['has_pair'].sum())}")
    print(f"Needs review: {int(df['needs_review'].sum())}")
    print(f"Saved: {out_dir / 'pairing_audit.csv'}")
    print(f"Saved: {out_dir / 'pairing_mismatches.csv'}")


if __name__ == "__main__":
    main()

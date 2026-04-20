#!/usr/bin/env python3
"""
Walk a data root, summarize every NRRD: dimensions, spacing, vector vs scalar,
and basic label statistics for likely segmentation files.

Requires: pip install SimpleITK numpy
Run from anywhere:
  python scripts/inspect_nrrd_dataset.py --root "/path/to/dataforAnwar"
Default root is the parent of the scripts/ directory (project root).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    print("Install SimpleITK: pip install SimpleITK", file=sys.stderr)
    sys.exit(1)


def looks_like_segmentation(path: Path) -> bool:
    """Check if the file name looks like a segmentation file based on naming."""
    name = path.name.lower()
    return "seg" in name or "segmentation" in name or "label" in name


def nrrd_header_snippet(path: Path, max_bytes: int = 8000) -> str:
    """Return readable ASCII from the start of the file (NRRD header is text)."""
    with path.open("rb") as f:
        chunk = f.read(max_bytes)
    # Decode permissively; binary data becomes replacement chars — fine for header peek.
    return chunk.decode("utf-8", errors="replace")


def parse_nrrd_dimension_sizes(snippet: str) -> tuple[int | None, list[int] | None]:
    """Parse 'dimension:' and 'sizes:' from NRRD header text if present."""
    dim, sizes = None, None
    for line in snippet.splitlines():
        if line.startswith("dimension:"):
            try:
                dim = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        if line.startswith("sizes:"):
            parts = line.split(":", 1)[1].strip().split()
            try:
                sizes = [int(p) for p in parts]
            except ValueError:
                pass
    return dim, sizes


def unique_label_values(arr: np.ndarray, max_samples: int = 2_000_000) -> np.ndarray:
    """Unique values with a cap on scanned elements for huge arrays."""
    flat = arr.ravel()
    if flat.size > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(flat.size, size=max_samples, replace=False)
        flat = flat[idx]
    return np.unique(flat)


def summarize_image(path: Path) -> None:
    print(f"\n{'=' * 72}\nFILE: {path}\n{'=' * 72}")
    snippet = nrrd_header_snippet(path)
    dim_hdr, sizes_hdr = parse_nrrd_dimension_sizes(snippet)
    if dim_hdr is not None:
        print(f"NRRD header: dimension={dim_hdr}, sizes={sizes_hdr}")
    # Short hints from NRRD header (avoid dumping huge MultiVolume frame lists)
    printed = 0
    for line in snippet.splitlines():
        s = line.strip()
        if s.startswith(("kinds:", "space:", "dimension:", "sizes:", "DataNodeClassName:")):
            print(f"  {s[:220]}")
            printed += 1
        elif s.startswith("MultiVolume.DICOM.") and "FrameFileList" not in s:
            print(f"  {s[:220]}")
            printed += 1
        if printed >= 12:
            break

    img = sitk.ReadImage(str(path))
    size = img.GetSize()  # order is X,Y,Z for 3D; for vector image, components are separate
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    ncomp = img.GetNumberOfComponentsPerPixel()
    pixel_type = img.GetPixelIDTypeAsString()

    print("SimpleITK summary:")
    print(f"  Size (X,Y,Z): {size}")
    print(f"  Spacing: {spacing}")
    print(f"  Origin: {origin}")
    print(f"  Components per pixel: {ncomp}")
    print(f"  Pixel type: {pixel_type}")

    arr = sitk.GetArrayFromImage(img)
    # SimpleITK array is often (Z,Y,X) or (frame,Z,Y,X) depending on image; print shape clearly
    print(f"  numpy array shape (memory order from SimpleITK): {arr.shape}")

    # Label statistics only when the filename suggests a mask (scalar MRIs are also often integer).
    if looks_like_segmentation(path):
        u = unique_label_values(arr)
        print(f"  Unique label values (possibly subsampled): {u}")
        print(f"  Count of unique labels: {len(u)}")
    elif ncomp > 1:
        print("  Multi-component volume (e.g. cine/time); not reporting label uniques from filename.")


def suggest_pairs(nrrd_files: list[Path]) -> None:
    """Heuristic: match imaging volumes to segmentations by stem."""
    segs = [p for p in nrrd_files if looks_like_segmentation(p)]
    imgs = [p for p in nrrd_files if p not in segs]
    print("\n" + "#" * 72)
    print("PAIRING HEURISTIC (name-based — verify in Slicer)")
    print("#" * 72)
    print(f"Candidate images: {len(imgs)}, candidate segmentations: {len(segs)}")
    for im in sorted(imgs):
        stem = re.sub(r"\s+", " ", im.stem.lower())
        for sg in sorted(segs):
            sstem = re.sub(r"\s+", " ", sg.stem.lower())
            if stem in sstem or sstem.startswith(stem):
                print(f"  possible pair: {im.name}  <->  {sg.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect NRRD cardiac MRI dataset layout.")
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Root folder to scan recursively for .nrrd files",
    )
    args = parser.parse_args()
    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    nrrd_files = sorted(root.rglob("*.nrrd"))
    if not nrrd_files:
        print(f"No .nrrd files under {root}")
        sys.exit(0)

    print(f"Scanned root: {root}")
    print(f"Found {len(nrrd_files)} NRRD file(s)")
    for p in nrrd_files:
        summarize_image(p)
    suggest_pairs(nrrd_files)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Summarize K-fold training outputs from runs/*/summary.json files.

Expected JSON key from monai_train_segmentation.py:
  best_val_mean_fg_dice
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize fold best_val_mean_fg_dice values.")
    p.add_argument("--runs-root", type=Path, required=True, help="e.g. /fs/.../runs/internal_cv")
    p.add_argument("--glob", type=str, default="fold_*/summary.json")
    args = p.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    files = sorted(runs_root.glob(args.glob))
    if not files:
        raise SystemExit(f"No summary files matched: {runs_root / args.glob}")

    rows: list[tuple[str, float]] = []
    for fp in files:
        data = json.loads(fp.read_text(encoding="utf-8"))
        v = float(data.get("best_val_mean_fg_dice", math.nan))
        rows.append((fp.parent.name, v))

    valid = [v for _, v in rows if not math.isnan(v)]
    print("fold,best_val_mean_fg_dice")
    for fold, v in rows:
        print(f"{fold},{'' if math.isnan(v) else f'{v:.6f}'}")
    if valid:
        print()
        print(f"n={len(valid)}")
        print(f"mean={float(np.mean(valid)):.6f}")
        print(f"std={float(np.std(valid)):.6f}")


if __name__ == "__main__":
    main()

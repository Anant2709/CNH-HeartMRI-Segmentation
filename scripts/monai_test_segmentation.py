#!/usr/bin/env python3
"""
Frozen **external test** evaluation only (split=test).

Same implementation as monai_eval_segmentation.py but forces `--split test` and
prints a reminder not to use test metrics for hyperparameter tuning.

Example:
  python scripts/monai_test_segmentation.py \\
    --data-root . \\
    --media-root /fs/nexus-scratch/anant04/heart-mri-data \\
    --checkpoint runs/segmentation/checkpoint_best.pt \\
    --out-dir reports/test_external_run1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import monai_eval_segmentation as ev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate checkpoint on the external TEST split only (holdout)."
    )
    p.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--media-root", type=Path, default=None)
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--patch-size", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"))
    p.add_argument("--spacing-mm", type=float, nargs=3, default=None, metavar=("SZ", "SY", "SX"))
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--save-predictions-dir", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    if "--split" in sys.argv:
        print("monai_test_segmentation does not accept --split (always test).", file=sys.stderr)
        sys.exit(2)
    args = parse_args()
    args.split = "test"
    print("NOTE: Test split is for final reporting only — do not tune on these numbers.", file=sys.stderr)
    ev.run_eval(args)


if __name__ == "__main__":
    main()

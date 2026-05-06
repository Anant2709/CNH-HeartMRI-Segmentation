#!/usr/bin/env python3
"""
Report worst cases for class-3 Dice from eval_<split>_per_case.csv.

Class mapping in this project:
  class 3 -> CPC
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Show worst class-3 (CPC) Dice cases.")
    p.add_argument("--csv", type=Path, required=True, help="eval_<split>_per_case.csv")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    rows = []
    with args.csv.expanduser().resolve().open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                d3 = float(r.get("dice_3", "nan"))
            except ValueError:
                continue
            rows.append((d3, r.get("sample_id", ""), r.get("patient_id", ""), r.get("site", "")))

    rows.sort(key=lambda x: x[0])  # worst first
    print("rank,dice_3_cpc,sample_id,patient_id,site")
    for i, (d3, sid, pid, site) in enumerate(rows[: args.top_k], start=1):
        print(f"{i},{d3:.6f},{sid},{pid},{site}")


if __name__ == "__main__":
    main()

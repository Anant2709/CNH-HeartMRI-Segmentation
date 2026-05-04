#!/usr/bin/env python3
"""
Evaluate a trained MONAI checkpoint on train / val / test split.

Writes per-case CSV (Dice per foreground class + mean), aggregate JSON, markdown summary,
and optionally saves class-map NRRDs (preprocessed RAS grid — see --save-predictions-dir).

Example:
  python scripts/monai_eval_segmentation.py \\
    --data-root . \\
    --checkpoint runs/segmentation/checkpoint_best.pt \\
    --split val \\
    --out-dir reports/eval_val_run1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import monai_segmentation_common as msc


def _save_pred_nrrd(logits: torch.Tensor, out_path: Path) -> None:
    import itk

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint16)
    img = itk.image_from_array(pred)
    itk.imwrite(img, str(out_path))


def run_eval(args: argparse.Namespace) -> None:
    data_root = args.data_root.expanduser().resolve()
    split_csv = (
        args.split_csv.expanduser().resolve()
        if args.split_csv
        else data_root / "reports" / "splits" / "internal_train_val_external_test.csv"
    )
    ckpt_path = args.checkpoint.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not split_csv.is_file():
        print(f"Missing split CSV: {split_csv}", file=sys.stderr)
        sys.exit(1)
    if not ckpt_path.is_file():
        print(f"Missing checkpoint: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    roots = msc.resolve_media_roots(data_root, args.media_root.expanduser().resolve() if args.media_root else None)
    all_rows = msc.read_split_csv(split_csv)
    data_list = msc.build_data_list(all_rows, args.split, roots, args.max_cases, with_meta=True)
    if not data_list:
        print(f"No cases for split={args.split} with files on disk.", file=sys.stderr)
        sys.exit(1)

    device = msc.pick_device(args.device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print("Checkpoint must be a dict with key 'model' (same format as training script).", file=sys.stderr)
        sys.exit(1)

    patch_size = msc.patch_size_from_checkpoint(ckpt, tuple(args.patch_size) if args.patch_size else None)
    spacing_mm = msc.spacing_from_checkpoint(ckpt, tuple(args.spacing_mm) if args.spacing_mm else None)

    val_tf = msc.val_transforms(spacing_mm)

    model = msc.build_unet(device)
    model.load_state_dict(ckpt["model"])

    rows_out: list[dict[str, Any]] = []
    pred_dir = args.save_predictions_dir.expanduser().resolve() if args.save_predictions_dir else None
    if pred_dir is not None:
        pred_dir.mkdir(parents=True, exist_ok=True)

    for entry in data_list:
        sample_id = entry.get("sample_id", "")
        try:
            batch = msc.batch_from_val_dict(val_tf, entry)
            logits, label = msc.sliding_window_logits(
                model,
                batch,
                device,
                roi_size=patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.overlap,
            )
            pc = msc.per_class_fg_dice_from_logits(logits, label)
            d1, d2, d3, d4 = (float(x) for x in pc.cpu())
            mean_fg = float(pc.mean().cpu())
            rows_out.append(
                {
                    "sample_id": sample_id,
                    "patient_id": entry.get("patient_id", ""),
                    "site": entry.get("site", ""),
                    "dice_1": d1,
                    "dice_2": d2,
                    "dice_3": d3,
                    "dice_4": d4,
                    "mean_fg_dice": mean_fg,
                }
            )
            if pred_dir is not None:
                safe = sample_id.replace("/", "_").replace(" ", "_") or "case"
                _save_pred_nrrd(logits, pred_dir / f"{safe}_pred.nrrd")
        except Exception as e:
            print(f"Case failed {sample_id}: {e}", file=sys.stderr)
            rows_out.append(
                {
                    "sample_id": sample_id,
                    "patient_id": entry.get("patient_id", ""),
                    "site": entry.get("site", ""),
                    "dice_1": math.nan,
                    "dice_2": math.nan,
                    "dice_3": math.nan,
                    "dice_4": math.nan,
                    "mean_fg_dice": math.nan,
                    "error": str(e),
                }
            )

    csv_path = out_dir / f"eval_{args.split}_per_case.csv"
    fields = ["sample_id", "patient_id", "site", "dice_1", "dice_2", "dice_3", "dice_4", "mean_fg_dice", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow(
                {
                    "sample_id": r.get("sample_id", ""),
                    "patient_id": r.get("patient_id", ""),
                    "site": r.get("site", ""),
                    "dice_1": r.get("dice_1", ""),
                    "dice_2": r.get("dice_2", ""),
                    "dice_3": r.get("dice_3", ""),
                    "dice_4": r.get("dice_4", ""),
                    "mean_fg_dice": r.get("mean_fg_dice", ""),
                    "error": r.get("error", ""),
                }
            )

    valid = [r for r in rows_out if not math.isnan(r.get("mean_fg_dice", math.nan))]
    agg: dict[str, Any] = {
        "split": args.split,
        "n_cases": len(rows_out),
        "n_evaluated_ok": len(valid),
        "checkpoint": str(ckpt_path),
        "patch_size": list(patch_size),
        "spacing_mm": list(spacing_mm) if spacing_mm else None,
    }
    if valid:
        for ci, key in enumerate(["dice_1", "dice_2", "dice_3", "dice_4"], start=1):
            vals = [r[key] for r in valid]
            agg[f"class_{ci}_mean"] = float(np.mean(vals))
            agg[f"class_{ci}_std"] = float(np.std(vals))
        mfs = [r["mean_fg_dice"] for r in valid]
        agg["mean_fg_dice_mean"] = float(np.mean(mfs))
        agg["mean_fg_dice_std"] = float(np.std(mfs))

    (out_dir / f"eval_{args.split}_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    md_lines = [
        f"# Eval summary — split `{args.split}`",
        "",
        f"- **Checkpoint:** `{ckpt_path}`",
        f"- **Cases (rows):** {len(rows_out)}",
        f"- **OK evaluations:** {len(valid)}",
        "",
        "## Aggregate foreground Dice",
        "",
        "| Class | Mean | Std |",
        "|------:|-----:|----:|",
    ]
    if valid:
        for ci, key in enumerate(["dice_1", "dice_2", "dice_3", "dice_4"], start=1):
            vals = [r[key] for r in valid]
            md_lines.append(f"| {ci} (structure {ci}) | {np.mean(vals):.4f} | {np.std(vals):.4f} |")
        mfs = [r["mean_fg_dice"] for r in valid]
        md_lines.extend(
            [
                "",
                f"**Mean of per-case mean foreground Dice (1–4):** {float(np.mean(mfs)):.4f} ± {float(np.std(mfs)):.4f}",
                "",
                f"Per-case CSV: `{csv_path.name}`",
            ]
        )
    else:
        md_lines.append("_No successful evaluations._")

    (out_dir / f"eval_{args.split}_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / f'eval_{args.split}_summary.md'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate MONAI segmentation checkpoint.")
    p.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--media-root", type=Path, default=None)
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, required=True, help="e.g. runs/segmentation/checkpoint_best.pt")
    p.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Which split from the CSV to score",
    )
    p.add_argument("--out-dir", type=Path, required=True, help="Directory for CSV/MD/JSON outputs")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--sw-batch-size", type=int, default=1)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--patch-size", type=int, nargs=3, default=None, metavar=("Z", "Y", "X"), help="Override ROI (default from checkpoint)")
    p.add_argument(
        "--spacing-mm",
        type=float,
        nargs=3,
        default=None,
        metavar=("SZ", "SY", "SX"),
        help="Override val spacing (default from checkpoint config)",
    )
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument(
        "--save-predictions-dir",
        type=Path,
        default=None,
        help="If set, write one *_pred.nrrd per case (class map in preprocessed grid; ITK)",
    )
    return p.parse_args()


def main() -> None:
    run_eval(parse_args())


if __name__ == "__main__":
    main()

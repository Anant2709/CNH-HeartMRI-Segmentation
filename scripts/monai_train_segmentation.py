#!/usr/bin/env python3
"""
3D cardiac MRI segmentation — MONAI training on manifest + split CSVs.

Reads absolute NRRD paths resolved from --media-root (same search order as
build_dataset_manifest.py), trains a 3D U-Net with Dice+CE, validates with
sliding-window inference on full volumes.

Example:
  pip install -r requirements-training.txt
  cd CNH-HeartMRI-Segmentation
  python scripts/monai_train_segmentation.py --data-root . --epochs 50

Optional smoke check (tiny subset, 1 epoch):
  python scripts/monai_train_segmentation.py --data-root . --epochs 1 \\
    --max-train-cases 2 --max-val-cases 1 --val-interval 1
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    SpatialPadd,
    Spacingd,
)


def resolve_media_roots(data_root: Path, media_root: Path | None) -> list[Path]:
    if media_root is not None:
        return [media_root.expanduser().resolve()]
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
        if r in seen or not p.is_dir():
            continue
        seen.add(r)
        out.append(r)
    return out if out else [data_root]


def resolve_pair(image_rel: str, mask_rel: str, roots: Sequence[Path]) -> tuple[Path, Path] | None:
    for root in roots:
        ip, mp = root / image_rel, root / mask_rel
        if ip.is_file() and mp.is_file():
            return ip, mp
    return None


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_data_list(
    rows: list[dict[str, str]],
    split_name: str,
    roots: Sequence[Path],
    max_cases: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("split", "").strip() != split_name:
            continue
        rel_img, rel_msk = row["image_path"], row["mask_path"]
        resolved = resolve_pair(rel_img, rel_msk, roots)
        if resolved is None:
            print(f"Warning: missing files for {row.get('sample_id')}", file=sys.stderr)
            continue
        ip, mp = resolved
        out.append({"image": str(ip), "label": str(mp)})
    if max_cases is not None:
        out = out[: max_cases]
    return out


def pick_device(prefer: str) -> torch.device:
    prefer = prefer.lower().strip()
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    print(f"Device '{prefer}' unavailable; falling back.", file=sys.stderr)
    return pick_device("auto")


def train_transforms(
    patch_size: tuple[int, int, int],
    spacing_mm: tuple[float, float, float] | None,
) -> Compose:
    x: list[Any] = [
        LoadImaged(keys=["image", "label"], reader="ITKReader", ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int16)),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if spacing_mm is not None:
        x.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing_mm,
                mode=("trilinear", "nearest"),
            )
        )
    x.extend(
        [
            CastToTyped(keys=["label"], dtype=torch.int64),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant", constant_values=0),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=1,
                allow_smaller=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),
        ]
    )
    return Compose(x)


def val_transforms(spacing_mm: tuple[float, float, float] | None) -> Compose:
    x: list[Any] = [
        LoadImaged(keys=["image", "label"], reader="ITKReader", ensure_channel_first=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=(torch.float32, torch.int16)),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ]
    if spacing_mm is not None:
        x.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=spacing_mm,
                mode=("trilinear", "nearest"),
            )
        )
    x.extend(
        [
            CastToTyped(keys=["label"], dtype=torch.int64),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
    )
    return Compose(x)


def mean_fg_dice_from_logits(logits: torch.Tensor, label: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Mean Dice over foreground classes 1–4; empty-set rule: both empty -> 1, one empty -> 0."""
    pred = logits.argmax(dim=1)  # (B, D, H, W)
    lab = label.squeeze(1)
    dices: list[torch.Tensor] = []
    for c in range(1, 5):
        p = (pred == c).float()
        g = (lab == c).float()
        inter = (p * g).sum()
        ps, gs = p.sum(), g.sum()
        if ps == 0 and gs == 0:
            dices.append(logits.new_tensor(1.0))
        elif ps == 0 or gs == 0:
            dices.append(logits.new_tensor(0.0))
        else:
            dices.append((2 * inter + eps) / (ps + gs + eps))
    return torch.stack(dices).mean()


@torch.no_grad()
def validate_one(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
) -> float:
    """Mean foreground Dice (classes 1–4) on one full volume."""
    model.eval()
    image = batch["image"].to(device)
    label = batch["label"].to(device)
    logits = sliding_window_inference(
        image,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
    )
    return float(mean_fg_dice_from_logits(logits, label).cpu())


def main() -> None:
    p = argparse.ArgumentParser(description="Train 3D U-Net (MONAI) on split CSV + NRRD.")
    p.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--media-root", type=Path, default=None, help="Folder with External/ Internal/ (default: auto)")
    p.add_argument(
        "--split-csv",
        type=Path,
        default=None,
        help="Default: <data-root>/reports/splits/internal_train_val_external_test.csv",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: <data-root>/runs/segmentation")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96), metavar=("Z", "Y", "X"))
    p.add_argument("--val-interval", type=int, default=5, help="Run full-volume val every N epochs")
    p.add_argument("--sw-batch-size", type=int, default=1, help="Sliding-window micro-batch on GPU")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA recommended)")
    p.add_argument("--max-train-cases", type=int, default=None)
    p.add_argument("--max-val-cases", type=int, default=None)
    p.add_argument(
        "--spacing-mm",
        type=float,
        nargs=3,
        default=None,
        metavar=("SZ", "SY", "SX"),
        help="Optional isotropic-ish resampling in mm (e.g. 1.2 1.2 1.2). Omit to keep native spacing.",
    )
    p.add_argument("--final-test", action="store_true", help="After training, evaluate best ckpt on external test")
    args = p.parse_args()

    data_root = args.data_root.expanduser().resolve()
    split_csv = (
        args.split_csv.expanduser().resolve()
        if args.split_csv
        else data_root / "reports" / "splits" / "internal_train_val_external_test.csv"
    )
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else data_root / "runs" / "segmentation")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not split_csv.is_file():
        print(f"Missing split CSV: {split_csv}", file=sys.stderr)
        sys.exit(1)

    roots = resolve_media_roots(data_root, args.media_root.expanduser().resolve() if args.media_root else None)
    all_rows = read_split_csv(split_csv)
    train_list = build_data_list(all_rows, "train", roots, args.max_train_cases)
    val_list = build_data_list(all_rows, "val", roots, args.max_val_cases)
    if not train_list:
        print("No training cases with files on disk. Check --media-root and paths in split CSV.", file=sys.stderr)
        sys.exit(1)
    if not val_list:
        print("No validation cases; use more internal data or lower val holdout in build_splits.", file=sys.stderr)
        sys.exit(1)

    spacing = tuple(args.spacing_mm) if args.spacing_mm is not None else None
    patch_size = tuple(args.patch_size)
    train_tf = train_transforms(patch_size, spacing)
    val_tf = val_transforms(spacing)

    train_ds = Dataset(data=train_list, transform=train_tf)
    val_ds = Dataset(data=val_list, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    # Val: batch 1 full volumes
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    channels = (16, 32, 64, 128, 256)
    strides = (2, 2, 2, 2)
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=channels,
        strides=strides,
        num_res_units=2,
    ).to(device)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = {
        "data_root": str(data_root),
        "split_csv": str(split_csv),
        "media_roots_tried": [str(r) for r in roots],
        "train_cases": len(train_list),
        "val_cases": len(val_list),
        "patch_size": list(patch_size),
        "spacing_mm": list(spacing) if spacing else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "device": str(device),
        "amp_cuda": use_amp,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_mean_dice = -1.0
    best_path = out_dir / "checkpoint_best.pt"
    last_path = out_dir / "checkpoint_last.pt"
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast("cuda", enabled=True) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                logits = model(images)
                loss = loss_fn(logits, labels)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            running_loss += float(loss.detach().cpu())
            n_steps += 1
        mean_loss = running_loss / max(n_steps, 1)

        val_dice = None
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            dices: list[float] = []
            for vb in val_loader:
                try:
                    d = validate_one(
                        model,
                        vb,
                        device,
                        roi_size=patch_size,
                        sw_batch_size=args.sw_batch_size,
                        overlap=args.overlap,
                    )
                    dices.append(d)
                except Exception as e:
                    print(f"Val case failed: {e}", file=sys.stderr)
            if dices:
                val_dice = float(np.mean(dices))
                if val_dice > best_mean_dice:
                    best_mean_dice = val_dice
                    torch.save({"epoch": epoch, "model": model.state_dict(), "config": config}, best_path)

        row = {"epoch": epoch, "train_loss": mean_loss, "val_mean_fg_dice": val_dice}
        history.append(row)
        dt = time.time() - t0
        vd = f"{val_dice:.4f}" if val_dice is not None else "n/a"
        print(f"epoch {epoch:04d}  loss={mean_loss:.4f}  val_mean_fg_dice={vd}  ({dt:.1f}s)")

        torch.save({"epoch": epoch, "model": model.state_dict(), "config": config}, last_path)

    # Write history CSV
    hist_path = out_dir / "history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_mean_fg_dice"])
        w.writeheader()
        for h in history:
            w.writerow(
                {
                    "epoch": h["epoch"],
                    "train_loss": f"{h['train_loss']:.6f}",
                    "val_mean_fg_dice": ""
                    if h["val_mean_fg_dice"] is None
                    else f"{h['val_mean_fg_dice']:.6f}",
                }
            )

    summary = {"best_val_mean_fg_dice": best_mean_dice, "best_checkpoint": str(best_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {best_path} (best val mean fg Dice ~ {best_mean_dice:.4f})")

    if args.final_test:
        test_list = build_data_list(all_rows, "test", roots, None)
        if not test_list:
            print("No test split rows.", file=sys.stderr)
            return
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_ds = Dataset(data=test_list, transform=val_tf)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        tdices: list[float] = []
        for tb in test_loader:
            try:
                tdices.append(
                    validate_one(
                        model,
                        tb,
                        device,
                        roi_size=patch_size,
                        sw_batch_size=args.sw_batch_size,
                        overlap=args.overlap,
                    )
                )
            except Exception as e:
                print(f"Test case failed: {e}", file=sys.stderr)
        if tdices:
            print(f"External test mean foreground Dice: {float(np.mean(tdices)):.4f} (n={len(tdices)})")
            (out_dir / "test_summary.json").write_text(
                json.dumps({"mean_fg_dice": float(np.mean(tdices)), "n": len(tdices)}, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()

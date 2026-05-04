"""
Shared helpers for MONAI cardiac MRI segmentation (train / eval / test).

Imported by monai_train_segmentation.py, monai_eval_segmentation.py, etc.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from monai.inferers import sliding_window_inference
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
    *,
    with_meta: bool = False,
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
        item: dict[str, Any] = {"image": str(ip), "label": str(mp)}
        if with_meta:
            item["sample_id"] = row.get("sample_id", "")
            item["patient_id"] = row.get("patient_id", "")
            item["site"] = row.get("site", "")
        out.append(item)
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


def build_unet(device: torch.device) -> UNet:
    channels = (16, 32, 64, 128, 256)
    strides = (2, 2, 2, 2)
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=channels,
        strides=strides,
        num_res_units=2,
    ).to(device)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt if isinstance(ckpt, dict) else {}


def per_class_fg_dice_from_logits(
    logits: torch.Tensor, label: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """Dice for classes 1..4 only; shape (4,). Empty-set: both empty -> 1, one empty -> 0."""
    pred = logits.argmax(dim=1)
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
    return torch.stack(dices)


def mean_fg_dice_from_logits(logits: torch.Tensor, label: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    return per_class_fg_dice_from_logits(logits, label, eps).mean()


@torch.no_grad()
def sliding_window_logits(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return logits, label


@torch.no_grad()
def validate_one(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    roi_size: tuple[int, int, int],
    sw_batch_size: int,
    overlap: float,
) -> float:
    logits, label = sliding_window_logits(model, batch, device, roi_size, sw_batch_size, overlap)
    return float(mean_fg_dice_from_logits(logits, label).cpu())


def batch_from_val_dict(val_tf: Compose, entry: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Apply val transform and add batch dimension (B=1) for tensor keys."""
    d = {k: entry[k] for k in ("image", "label")}
    out_t = val_tf(d)
    batch: dict[str, torch.Tensor] = {}
    for k, v in out_t.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
    return batch


def patch_size_from_checkpoint(ckpt: dict[str, Any], override: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if override is not None:
        return override
    cfg = ckpt.get("config") or {}
    ps = cfg.get("patch_size")
    if ps is not None and len(ps) == 3:
        return int(ps[0]), int(ps[1]), int(ps[2])
    return (96, 96, 96)


def spacing_from_checkpoint(ckpt: dict[str, Any], override: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
    if override is not None:
        return override
    cfg = ckpt.get("config") or {}
    s = cfg.get("spacing_mm")
    if s is not None and len(s) == 3:
        return float(s[0]), float(s[1]), float(s[2])
    return None

# `monai_train_segmentation.py` — full technical walkthrough

This document explains **every major block** of `scripts/monai_train_segmentation.py`: data flow, MONAI transforms, the **3D U-Net**, **loss functions (with math)**, **optimization**, **mixed precision**, **sliding-window validation**, and **CLI flags**. Line numbers refer to the version in the repository at the time this doc was written.

---

## Table of contents

1. [Program purpose and high-level flow](#1-program-purpose-and-high-level-flow)
2. [Imports and dependencies](#2-imports-and-dependencies)
3. [Path resolution: `resolve_media_roots` and `resolve_pair`](#3-path-resolution)
4. [CSV I/O: `read_split_csv` and `build_data_list`](#4-csv-io)
5. [Device selection: `pick_device`](#5-device-selection)
6. [Training transforms](#6-training-transforms)
7. [Validation transforms](#7-validation-transforms)
8. [Dice metric on logits: `mean_fg_dice_from_logits`](#8-dice-metric)
9. [Validation pass: `validate_one` and sliding-window inference](#9-validation-pass)
10. [Model: MONAI `UNet`](#10-model-monai-unet)
11. [Loss: `DiceCELoss`](#11-loss-diceceloss)
12. [Optimizer and AMP](#12-optimizer-and-amp)
13. [Training loop, checkpointing, history](#13-training-loop)
14. [Optional final test split](#14-optional-final-test-split)
15. [Command-line reference](#15-command-line-reference)

---

## 1. Program purpose and high-level flow

**Goal:** Train a **volumetric (3D)** semantic segmentation model that maps a single-channel MRI volume to **5 classes** (background + four structures), using **supervised learning** from paired NRRD masks.

**Inputs:**

- A **split CSV** (default `reports/splits/internal_train_val_external_test.csv`) listing `image_path`, `mask_path`, and `split` per row.
- NRRD files on disk, resolved via a **media root** (see [DATASET.md](DATASET.md)).

**Outputs (under `--out-dir`, default `runs/segmentation/`):**

- `config.json` — serialized hyperparameters and path metadata.
- `checkpoint_best.pt` — weights that achieved the **best mean foreground Dice on validation** so far.
- `checkpoint_last.pt` — weights after the latest epoch.
- `history.csv` — per-epoch `train_loss` and `val_mean_fg_dice` (when validation ran).
- `summary.json` — best validation score and path to best checkpoint.
- If `--final-test`: `test_summary.json` — mean foreground Dice on the **external test** split.

**High-level algorithm:**

1. Build lists of `{"image": abs_path, "label": abs_path}` for `train` and `val` splits.
2. Attach **MONAI transform pipelines** (different for train vs val).
3. **Train:** sample **fixed-size 3D patches** with foreground-aware random cropping; minimize **Dice + cross-entropy** loss; optionally use **FP16 autocast** on CUDA.
4. **Validate (every `--val-interval` epochs):** run **sliding-window inference** over each **full** validation volume, compare argmax prediction to ground truth, compute **mean Dice over classes 1–4** with an explicit **empty-set rule**.

---

## 2. Imports and dependencies

```19:49:scripts/monai_train_segmentation.py
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
```

- **`torch`:** Tensors, autograd, device placement, `torch.save` / `torch.load`, AMP (`torch.amp`).
- **`monai.data.Dataset`:** Lightweight dataset: each index applies `transform` to one dict from `data=`.
- **`DataLoader`:** Batches dicts; default PyTorch collate **stacks** `image` and `label` tensors — hence all samples in a batch must have **identical spatial shape** after transforms (we enforce this with padding + fixed crop; see §6).
- **`sliding_window_inference`:** Runs the network on overlapping **ROIs** that tile a large volume and **blends** predictions (here with Gaussian weighting) into one full-volume logits tensor.
- **`DiceCELoss`:** Differentiable surrogate for overlap (Dice) plus multi-class log loss (CE).
- **`UNet`:** Standard encoder–decoder with skip connections for dense prediction.
- **Transforms:** dictionary-style (`*d` suffix) so each step receives `{"image": ..., "label": ...}`.

Install stack: see `requirements-training.txt` in the repo root (`torch`, `monai`, `itk` for `ITKReader`, etc.).

---

## 3. Path resolution

### 3.1 `resolve_media_roots`

```52:72:scripts/monai_train_segmentation.py
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
```

**Meaning:** If `--media-root` is set, only that directory is tried. Otherwise we try a short ordered list so a typical layout (`repo` next to a sibling `data/` folder) works without extra flags.

### 3.2 `resolve_pair`

```74:80:scripts/monai_train_segmentation.py
def resolve_pair(image_rel: str, mask_rel: str, roots: Sequence[Path]) -> tuple[Path, Path] | None:
    for root in roots:
        ip, mp = root / image_rel, root / mask_rel
        if ip.is_file() and mp.is_file():
            return ip, mp
    return None
```

**Meaning:** First root where **both** relative paths exist wins. If none match, the row is skipped (with a warning in `build_data_list`).

---

## 4. CSV I/O

### 4.1 `read_split_csv`

```83:86:scripts/monai_train_segmentation.py
def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
```

### 4.2 `build_data_list`

```88:107:scripts/monai_train_segmentation.py
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
```

**Design choice:** Only `image` and `label` keys are kept so the `DataLoader` never tries to **stack string fields** (e.g. `sample_id`) into tensors.

**Debugging flags:** `--max-train-cases` / `--max-val-cases` truncate lists for quick smoke tests.

---

## 5. Device selection

```110:125:scripts/monai_train_segmentation.py
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
```

**Note:** **Mixed precision (`--amp`)** is only enabled when `device.type == "cuda"` in `main()`; MPS/CPU runs use full-precision forward for the train step.

---

## 6. Training transforms

```128:165:scripts/monai_train_segmentation.py
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
```

### 6.1 `LoadImaged` + `ITKReader`

Reads NRRD (and other ITK-supported formats) into tensor-like objects with spatial metadata. `ensure_channel_first=False` first, then:

### 6.2 `EnsureChannelFirstd`

Reorders to **channel-first** layout `(C, D, H, W)` for PyTorch / MONAI conventions.

### 6.3 `EnsureTyped` and `CastToTyped`

Forces image to **float32** and label to integer type before orientation; label is recast to **int64** before loss (cross-entropy expects class indices).

### 6.4 `Orientationd(..., axcodes="RAS")`

Reorients the volume to a **standard anatomical frame** (RAS: +X right, +Y anterior, +Z superior). Different scanners encode axes differently; this step reduces **orientation-driven domain shift**.

### 6.5 Optional `Spacingd`

If `--spacing-mm SZ SY SX` is passed, resamples both modalities to the given **voxel spacing in mm**:

- Image: **trilinear** interpolation (continuous signal model).
- Label: **nearest** neighbor (class indices must stay discrete).

If omitted, **native voxel spacing** from the file is kept (still subject to orientation).

### 6.6 `NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)`

Per-channel **standardization** over **nonzero voxels** (typical for MRI where background air is meaningless):

\[
x \leftarrow \frac{x - \mu_{\text{nz}}}{\sigma_{\text{nz}} + \epsilon}
\]

(with implementation-specific \(\epsilon\)). This makes intensity scales more comparable across scanners.

### 6.7 `SpatialPadd` + `RandCropByPosNegLabeld`

**Why both:** The DataLoader must **stack** a batch along `B` dimension. Without fixed spatial size, different random crops from different patients would have different `(D,H,W)` and `torch.stack` would fail.

- **`SpatialPadd`** pads each volume so every spatial dimension is **at least** `patch_size[k]` (constant `0` for both image and background class on label).
- **`RandCropByPosNegLabeld`** extracts a patch of exactly `spatial_size=patch_size`. It tries to include **foreground** voxels (`pos` crops) and sometimes **background-only** regions (`neg`) for class balance.

`num_samples=1` → one patch per `__getitem__` call. `allow_smaller=False` is safe after padding.

### 6.8 `RandFlipd`

Independent 10% probability flips along each axis — a mild **data augmentation** preserving labels under reflection.

---

## 7. Validation transforms

```168:189:scripts/monai_train_segmentation.py
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
```

**Difference from training:** **No random crop or flips.** Entire volume is fed to `sliding_window_inference` in `validate_one` so metrics reflect **whole-scan** behavior.

---

## 8. Dice metric

```192:208:scripts/monai_train_segmentation.py
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
```

For each foreground class \(c \in \{1,2,3,4\}\), define binary masks \(P_c = \mathbb{1}[\hat{Y}=c]\) and \(G_c = \mathbb{1}[Y=c]\) from predicted class map \(\hat{Y} = \arg\max_j \text{logits}_j\) and ground truth \(Y\).

**Standard Dice:**

\[
\text{Dice}_c = \frac{2 \sum_i P_{c,i} G_{c,i}}{\sum_i P_{c,i} + \sum_i G_{c,i} + \varepsilon}
\]

**Empty-set rule (per class, per volume):**

- If \(\sum P_c = 0\) **and** \(\sum G_c = 0\): treat as **perfect agreement** → score **1** (class absent in both).
- If exactly one is empty: score **0** (penalize false negative or false positive).
- Else: use the formula above.

The returned scalar is the **unweighted mean** \(\frac{1}{4}\sum_{c=1}^4 \text{Dice}_c\).

---

## 9. Validation pass

```211:232:scripts/monai_train_segmentation.py
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
```

**Sliding-window inference (conceptual):**

1. Tile the volume with windows of shape `roi_size` (here equal to **training patch size**).
2. Adjacent windows **overlap** by fraction `overlap` (default `0.5` → 50% overlap along each axis where applicable).
3. Each window’s logits are accumulated into a full-volume buffer; **`mode="gaussian"`** weights contributions so the **center** of each window counts more than edges, reducing **block-boundary artifacts**.

**Complexity note:** Validation is **much slower** per case than one patch forward pass because it runs the network on **many** windows per volume.

---

## 10. Model: MONAI `UNet`

```318:327:scripts/monai_train_segmentation.py
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
```

- **`spatial_dims=3`:** Volumes, not 2D slices.
- **`in_channels=1`:** Single MRI contrast (scalar image).
- **`out_channels=5`:** One **logit per class** per voxel (not yet a probability).
- **`channels`:** Feature width per encoder level; deeper layers are wider (standard U-Net pattern).
- **`strides`:** Downsample factor **2** between levels (spatial halving per axis each stage).
- **`num_res_units=2`:** Residual-style blocks inside each stage (MONAI’s `UNet` implementation detail for trainability).

**Forward output:** Logits tensor of shape `(B, 5, D, H, W)\). **Softmax** is applied **inside** `DiceCELoss` because we pass `softmax=True`.

---

## 11. Loss: `DiceCELoss`

```329:329:scripts/monai_train_segmentation.py
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True)
```

**Cross-entropy part (multiclass):** For each voxel \(i\) and class \(k\), with predicted probabilities \(p_{i,k} = \text{softmax}(\text{logits}_i)_k\) and one-hot target \(y_{i,k}\):

\[
\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_i \sum_k y_{i,k} \log p_{i,k}
\]

`to_onehot_y=True` means the target tensor holds **class indices** \((B,1,D,H,W)\) and the loss expands to one-hot internally.

**Dice part (differentiable soft Dice):** MONAI implements a **soft** Dice using predicted probabilities and smoothed squares (`squared_pred=True` is a stabilization variant). Intuitively it **rewards overlap** between \(p_{:,c}\) and \(y_{:,c}\) across voxels, helping with **class imbalance** (large background).

The library combines both into a **single scalar** loss per batch (default weighting is internal to MONAI’s implementation for this constructor).

---

## 12. Optimizer and AMP

```330:332:scripts/monai_train_segmentation.py
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
```

**AdamW update (sketch):** Let \(L\) be the loss. With gradients \(g_t = \nabla_\theta L\), AdamW maintains moment estimates and applies **decoupled weight decay** \(\lambda\) (here `weight_decay`):

\[
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t
\]

(\(\eta =\) `--lr`, default `1e-4`; \(\lambda =\) `--weight-decay`, default `1e-5`.)

**AMP path (`--amp` on CUDA):**

```363:374:scripts/monai_train_segmentation.py
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
```

`autocast` runs most matmuls/convolutions in **float16** where safe; loss backward uses **GradScaler** to avoid underflow in small gradients.

---

## 13. Training loop

Pseudo-structure:

```355:407:scripts/monai_train_segmentation.py
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            ...
        mean_loss = running_loss / max(n_steps, 1)

        val_dice = None
        if epoch % args.val_interval == 0 or epoch == args.epochs:
            dices: list[float] = []
            for vb in val_loader:
                try:
                    d = validate_one(...)
                    dices.append(d)
                except Exception as e:
                    print(f"Val case failed: {e}", file=sys.stderr)
            if dices:
                val_dice = float(np.mean(dices))
                if val_dice > best_mean_dice:
                    best_mean_dice = val_dice
                    torch.save({"epoch": epoch, "model": model.state_dict(), "config": config}, best_path)
        ...
        torch.save({"epoch": epoch, "model": model.state_dict(), "config": config}, last_path)
```

- **`train_loss`:** Mean minibatch loss over all training batches in the epoch (not a held-out subset).
- **`best_mean_dice`:** Tracks the **maximum** mean validation Dice seen; ties implicitly keep the earlier checkpoint unless a strict improvement occurs.
- **Checkpoints** store `state_dict` plus a small `config` dict for provenance.

---

## 14. Optional final test split

```429:461:scripts/monai_train_segmentation.py
    if args.final_test:
        test_list = build_data_list(all_rows, "test", roots, None)
        ...
        model.load_state_dict(ckpt["model"])
        test_ds = Dataset(data=test_list, transform=val_tf)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
        ...
```

Uses the **same** preprocessing as validation (`val_tf`) and the **best** checkpoint. Intended for a **single** honest external benchmark after training — **not** for iterative tuning.

---

## 15. Command-line reference

| Flag | Default | Role |
|------|---------|------|
| `--data-root` | repo root | Contains `reports/` |
| `--media-root` | auto list | Root with `External/`, `Internal/` |
| `--split-csv` | `reports/splits/internal_train_val_external_test.csv` | Train/val/test assignments |
| `--out-dir` | `runs/segmentation` | Artifacts |
| `--epochs` | 100 | Full passes over training set |
| `--batch-size` | 2 | Training minibatch size (patches) |
| `--lr` | 1e-4 | AdamW learning rate |
| `--weight-decay` | 1e-5 | AdamW weight decay |
| `--patch-size Z Y X` | 96 96 96 | Patch / sliding-window ROI |
| `--val-interval` | 5 | Validate every N epochs |
| `--sw-batch-size` | 1 | Micro-batch inside sliding window |
| `--overlap` | 0.5 | Sliding window overlap fraction |
| `--num-workers` | 0 | DataLoader workers (ITK often safer at 0 on clusters) |
| `--seed` | 42 | RNG seed |
| `--device` | auto | `auto\|cuda\|mps\|cpu` |
| `--amp` | off | CUDA mixed precision |
| `--max-train-cases` / `--max-val-cases` | None | Subset caps for debugging |
| `--spacing-mm` | off | Optional `Spacingd` resampling |
| `--final-test` | off | Score `test` split at end |

---

## Related scripts (train / eval / test)

Training logic imports **`scripts/monai_segmentation_common.py`** (transforms, U-Net, sliding-window Dice). Line references in earlier sections still describe the same code paths; some definitions now live in that module instead of inline in `monai_train_segmentation.py`.

- **`scripts/monai_eval_segmentation.py`** — load a checkpoint, run val preprocessing + sliding-window inference, write **`eval_<split>_per_case.csv`**, **`eval_<split>_summary.json`**, **`eval_<split>_summary.md`**; optional **`--save-predictions-dir`** for ITK NRRD class maps (preprocessed grid).
- **`scripts/monai_test_segmentation.py`** — always **`split=test`** with a reminder not to tune on test metrics.

Optional later: **bootstrap confidence intervals** on external test Dice for reporting.

This document should be updated if `monai_train_segmentation.py` or `monai_segmentation_common.py` changes materially (loss, transforms, or metric definitions).

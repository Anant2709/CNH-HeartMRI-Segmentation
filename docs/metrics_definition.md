# Metrics definition (Stage D companion)

This document defines exactly how we compute and report segmentation metrics for baseline v1.

---

## 1) Class set

Expected labels:
- `0`: background
- `1`: atrium
- `2`: ventricle
- `3`: CPC
- `4`: arch

Foreground classes are `1..4`.

---

## 2) Dice score per class

For one class `c`, with binary masks:
- `P_c`: predicted voxels of class `c`
- `G_c`: ground-truth voxels of class `c`

Dice:

`Dice(c) = 2 * |P_c ∩ G_c| / (|P_c| + |G_c| + eps)`

- `eps` is a tiny constant to avoid division by zero.

### Empty-set handling rule (must be fixed)

For class `c` in one case:
- if `|P_c| == 0` and `|G_c| == 0`: set Dice(c) = 1.0 (class absent in both).
- if one is empty and the other is not: Dice(c) = 0.0.

This rule must be used consistently across val and test.

---

## 3) Per-case summary metrics

For each case (volume):

- `dice_1`, `dice_2`, `dice_3`, `dice_4`
- `mean_dice_fg = mean(dice_1..dice_4)`

Optional:
- `dice_0` (background; not primary).

---

## 4) Split-level aggregate metrics

For each split (`train` optional, `val`, `test` required):

- `mean(dice_c)` over cases for each foreground class `c in {1..4}`
- `std(dice_c)` over cases
- `mean(mean_dice_fg)` over cases
- median and IQR of `mean_dice_fg` (optional but recommended)

Report separately for:
- internal validation (`split == val`)
- external holdout (`split == test`)

---

## 5) Site-stratified reporting

For `test` rows (external) and `val` rows (internal), report:

- number of cases
- mean foreground Dice
- per-class mean Dice

If future mixed-site CV is used, report per-fold metrics and across-fold mean ± std.

---

## 6) Thresholding / argmax rule

For multiclass outputs (softmax probabilities):
- predicted class per voxel = `argmax` over channels.
- No postprocessing for baseline v1 (unless explicitly added and versioned).

---

## 7) Recommended saved outputs

Per evaluation run:
- case-level metrics CSV (`sample_id`, site, split, dice_1..dice_4, mean_dice_fg)
- aggregate summary JSON/MD
- model checkpoint identifier and config hash

This is required for reproducibility and future comparisons (v1 vs v1.1, etc.).

---

## 8) Why these metrics

- Dice is standard for medical segmentation overlap quality.
- Per-class Dice avoids hiding weak classes behind easy ones.
- Mean foreground Dice is a compact single-number tracker for training/selection.
- Site-stratified reporting reflects your project goal of external robustness.


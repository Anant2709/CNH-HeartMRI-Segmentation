# Dataset manifest summary (Stage B)

- **Source:** `reports/baseline_manifest_v1.csv`
- **Output:** `reports/dataset_manifest.csv`
- **Total rows:** 132

## Paths checked for NRRD existence

- `/Users/juhi2/Desktop/AnantUNI/CNH Part time/Heart MRI Segmentation/data`
- `/Users/juhi2/Desktop/AnantUNI/CNH Part time/Heart MRI Segmentation/CNH-HeartMRI-Segmentation`
- `/Users/juhi2/Desktop/AnantUNI/CNH Part time/Heart MRI Segmentation`

## By site

- `external`: **19**
- `internal`: **113**

## Cropped vs not (is_cropped)

- `False`: **131**
- `True`: **1**

## 4D / multi-frame (is_4d)

- `False`: **132**

## labels_present (unique strings in manifest)

- `[0, 1, 2, 3, 4]`: **132**

## quality_flag

- `ok_automated`: **132**

## Notes

- `patient_id` is a **heuristic** grouping key from the image filename (not necessarily a hospital MRN). Use for split logic until official IDs are provided.
- `pairing_status` reflects that rows came from automated strict v1 filtering (geometry + labels 0–4).
- File existence is checked under `--media-root` if set, else in order: **`../data`**, `./data`, `--data-root`, parent of `--data-root` (manifest paths are like `External/...` under that folder).
- Rows with `quality_flag` other than `ok_automated` need path fixes or `--media-root` before training.

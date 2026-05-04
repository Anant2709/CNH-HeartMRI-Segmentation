# Running on UMIACS Nexus (GPU / scratch)

This runbook matches a **login node** session like `anant04@nexuscsd01` with scratch under **`/fs/nexus-scratch/anant04/`** (adjust user and paths if yours differ). Exact **module names** vary by cluster policy; always run `module avail cuda` (or ask staff) before scripting.

---

## 0. Step-by-step: copy data from laptop to scratch

**A. On your laptop** — open a terminal in the folder that **contains** `External/` and `Internal/` (often `Heart MRI Segmentation/data/`).

**B. Create the target directory on Nexus** (SSH once):

```bash
ssh anant04@nexuscsd.umiacs.umd.edu "mkdir -p /fs/nexus-scratch/anant04/heart-mri-data"
```

**C. Copy with `rsync` (recommended)** — resumable, preserves timestamps, run from the **laptop**:

```bash
# From the parent of the folder that contains External/ and Internal/
rsync -avz --progress "./data/" \
  "anant04@nexuscsd.umiacs.umd.edu:/fs/nexus-scratch/anant04/heart-mri-data/"
```

If your tree is not named `data/`, point the source at the directory whose **children** are `External/` and `Internal/`:

```bash
rsync -avz --progress "/Users/you/path/to/nrrd_root/" \
  "anant04@nexuscsd.umiacs.umd.edu:/fs/nexus-scratch/anant04/heart-mri-data/"
```

**Alternative: `scp` (simple but no resume)** — from laptop, copying one folder:

```bash
scp -r "/Users/you/path/to/data/External" \
  "anant04@nexuscsd.umiacs.umd.edu:/fs/nexus-scratch/anant04/heart-mri-data/"
scp -r "/Users/you/path/to/data/Internal" \
  "anant04@nexuscsd.umiacs.umd.edu:/fs/nexus-scratch/anant04/heart-mri-data/"
```

**D. On Nexus, verify:**

```bash
ls /fs/nexus-scratch/anant04/heart-mri-data/External | head
ls /fs/nexus-scratch/anant04/heart-mri-data/Internal | head
```

**E. Clone repo, venv, train** — see sections 2–4 and **Slurm** below. Training command (interactive GPU session):

```bash
export REPO_ROOT=/fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
export MEDIA_ROOT=/fs/nexus-scratch/anant04/heart-mri-data
cd "$REPO_ROOT"
source .venv/bin/activate
python scripts/monai_train_segmentation.py \
  --data-root "$REPO_ROOT" \
  --media-root "$MEDIA_ROOT" \
  --out-dir /fs/nexus-scratch/anant04/runs/segmentation_01 \
  --device cuda --amp --epochs 100
```

**F. After training — eval (val) and test (external):**

```bash
python scripts/monai_eval_segmentation.py \
  --data-root "$REPO_ROOT" --media-root "$MEDIA_ROOT" \
  --checkpoint /fs/nexus-scratch/anant04/runs/segmentation_01/checkpoint_best.pt \
  --split val --out-dir /fs/nexus-scratch/anant04/reports/eval_val_01 --device cuda

python scripts/monai_test_segmentation.py \
  --data-root "$REPO_ROOT" --media-root "$MEDIA_ROOT" \
  --checkpoint /fs/nexus-scratch/anant04/runs/segmentation_01/checkpoint_best.pt \
  --out-dir /fs/nexus-scratch/anant04/reports/test_external_01 --device cuda
```

Results: `eval_val_*_per_case.csv`, `eval_val_*_summary.md`, and the same pattern for `eval_test_*` from the test script.

**G. Slurm batch jobs (RTX A6000–style templates)** — from `$REPO_ROOT` after setting env vars:

```bash
# Set each variable to a real path (do not run `export REPO_ROOT` with no `=...` — that does not assign values):
export REPO_ROOT=/fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
export MEDIA_ROOT=/fs/nexus-scratch/anant04/heart-mri-data
export RUN_DIR=/fs/nexus-scratch/anant04/runs/segmentation_01
sbatch slurm/train_a6000.slurm
# For eval / test, also: export CKPT=... export OUT_DIR=...
sbatch slurm/eval_a6000.slurm
sbatch slurm/test_a6000.slurm
```

Edit `#SBATCH` lines inside `slurm/*.slurm` for your partition, time limit, and optional A6000 constraint.

**You do not need to repeat the interactive train/eval commands if you prefer Slurm** — the Slurm scripts run the same `python scripts/...` lines. The only hard requirement is **one-time (or occasional) venv + `pip install -r requirements-training.txt`** on scratch, because the job script **sources** `${REPO_ROOT}/.venv` but does not create it. Optional: run a one-minute interactive GPU session first to confirm `torch.cuda.is_available()` before submitting a long `sbatch`.

**Slurm `train-<jobid>.out` looks empty while the job runs:** Python **buffers** stdout when not attached to a TTY, so `print` lines may not appear until the buffer fills or the process exits. The Slurm scripts set **`export PYTHONUNBUFFERED=1`** so logs stream. For a job already submitted without that, check progress via **`ls -la "$RUN_DIR"`** (e.g. growing `history.csv` / checkpoints) or wait for the first epoch to finish.

---

## 1. Recommended layout on scratch

Keep **large NRRD trees** and **training outputs** on scratch; keep **the git repo** there too so checkpoints and `runs/` do not fill your home directory.

Example:

```text
/fs/nexus-scratch/anant04/
  heart-mri-data/          # NRRD root: contains External/, Internal/
  CNH-HeartMRI-Segmentation/   # git clone of this repo
```

---

## 2. Push from your laptop, pull on the cluster

On your **local machine** (inside the repo):

```bash
git status
git add -A
git commit -m "Describe your change in a full sentence."
git push origin main
```

On **Nexus** (scratch):

```bash
cd /fs/nexus-scratch/anant04
git clone <YOUR_GIT_REMOTE_URL> CNH-HeartMRI-Segmentation
# later updates:
cd CNH-HeartMRI-Segmentation
git pull origin main
```

If the repository is **private**, use HTTPS with a personal access token, SSH keys registered on the cluster, or a private fork URL your account can read.

---

## 3. Stage imaging data

Training resolves paths from the split CSV against **`--media-root`**. Copy or rsync your `data` folder so it contains `External/` and `Internal/` next to or under scratch:

```bash
# Example: one-time sync from laptop (run on laptop; replace HOST and paths)
rsync -avz --progress "/path/to/Heart MRI Segmentation/data/" \
  anant04@nexuscsd.umiacs.umd.edu:/fs/nexus-scratch/anant04/heart-mri-data/
```

On the cluster, verify:

```bash
ls /fs/nexus-scratch/anant04/heart-mri-data/External | head
```

---

## 4. Python environment on the cluster

Use a **venv** (or conda) **on scratch** so installs are writable and large.

```bash
cd /fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
module unload cuda cuda/12.1 cuda/12.1.1 cuda/13.1.1 2>/dev/null || true
module load cuda/12.1.1   # use `module avail cuda`; Slurm scripts default to 12.1.1 and honor CUDA_MODULE
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-training.txt
```

If `torch` does not see CUDA after install, reinstall the **CUDA-specific** wheel from [pytorch.org](https://pytorch.org/get-started/locally/) matching the **driver / module** on Nexus.

Quick check:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## 5. Interactive GPU session (debugging)

Many clusters require **`salloc`** or **`interact`** for GPU; login nodes may forbid long jobs. Example **Slurm**-style (names are illustrative — confirm with `cat /etc/slurm.conf` or cluster docs):

```bash
salloc --account=nexus --partition=tron --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=4:00:00
# wait for prompt, then:
cd /fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
module unload cuda cuda/12.1 cuda/12.1.1 cuda/13.1.1 2>/dev/null || true
module load cuda/12.1.1 || true
source .venv/bin/activate
python scripts/monai_train_segmentation.py \
  --data-root . \
  --media-root /fs/nexus-scratch/anant04/heart-mri-data \
  --epochs 5 \
  --val-interval 1 \
  --device cuda \
  --amp \
  --out-dir /fs/nexus-scratch/anant04/runs/segmentation_exp01
```

Use **`--out-dir`** on scratch so checkpoints land on the fast filesystem.

---

## 6. Batch job (long training)

Create `train.slurm` (edit partition, account, modules):

```bash
#!/bin/bash
#SBATCH --job-name=heart-mri-seg
#SBATCH --output=/fs/nexus-scratch/anant04/logs/%x-%j.out
#SBATCH --error=/fs/nexus-scratch/anant04/logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=nexus
#SBATCH --partition=tron
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p /fs/nexus-scratch/anant04/logs
module unload cuda cuda/12.1 cuda/12.1.1 cuda/13.1.1 2>/dev/null || true
module load cuda/12.1.1   # adjust; avoid `module load cuda` alone if another CUDA is already loaded

cd /fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
source .venv/bin/activate

python scripts/monai_train_segmentation.py \
  --data-root . \
  --media-root /fs/nexus-scratch/anant04/heart-mri-data \
  --epochs 100 \
  --val-interval 5 \
  --device cuda \
  --amp \
  --out-dir /fs/nexus-scratch/anant04/runs/segmentation_exp01
```

Submit:

```bash
sbatch train.slurm
```

---

## 7. Pitfalls

- **CUDA module conflicts:** If the log shows `Unable to locate a modulefile for 'cuda/12.1'` then `Conflicting 'cuda' is loaded`, the batch script tried a wrong name, then a generic `cuda` pulled a **different** version than what Slurm already preloaded. Fix: **`module unload …`** then **`module load cuda/12.1.1`** (or whatever `module avail cuda` lists). Repo `slurm/*.slurm` scripts do this; override with `export CUDA_MODULE=cuda/X.Y.Z` before `sbatch`.
- **Slurm partition name:** On **UMIACS Nexus**, the usual GPU partition is **`tron`**, not `gpu`. If you see `invalid partition specified: gpu`, set `#SBATCH --partition=tron` (and often `#SBATCH --account=nexus`) as in the repo `slurm/*.slurm` files. Confirm with `sinfo -o "%P %a" | head` or [UMIACS SLURM job submission](https://wiki.umiacs.umd.edu/umiacs/index.php/SLURM/JobSubmission). Override without editing the file: `sbatch --partition=tron --account=nexus slurm/train_a6000.slurm`.
- **Slurm QoS / CPUs:** If you see `QoS default has a max CPUs per job of 4`, your association’s default QoS caps **`--cpus-per-task`** (and sometimes memory). Lower the script to **4 CPUs** (as in `slurm/train_a6000.slurm`) or ask UMIACS for a GPU QoS/partition that allows more. **`export REPO_ROOT` without `=...`** does not set paths — use full `export REPO_ROOT=/path/...` before `sbatch`.
- **ITK / NRRD reads:** If you see import errors for `ITKReader`, ensure `itk` is installed (`requirements-training.txt` includes it). If **`Orientationd`** fails with **No module named `nibabel`**, run `pip install nibabel` (it is listed in `requirements-training.txt`). If multiprocessing workers crash, keep **`--num-workers 0`** (default in training script).
- **Path mismatch:** If training prints “No training cases”, your **`--media-root`** does not contain the `External/...` paths exactly as in the CSV. Fix root or regenerate manifests with the same layout.
- **Do not tune on test:** Use **`--final-test`** only when reporting a locked configuration; routine development uses **val** only.

---

## 8. After the run

- Copy **`config.json`**, **`history.csv`**, **`summary.json`** (small) into git or lab wiki if you want history versioned; **checkpoints** are large — keep on scratch or use **Git LFS** if your policy allows.
- Update **`docs/progress_log.md`** with the cluster path, commit hash, and key hyperparameters for reproducibility.

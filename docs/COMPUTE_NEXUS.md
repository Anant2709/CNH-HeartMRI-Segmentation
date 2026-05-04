# Running on UMIACS Nexus (GPU / scratch)

This runbook matches a **login node** session like `anant04@nexuscsd01` with scratch under **`/fs/nexus-scratch/anant04/`** (adjust user and paths if yours differ). Exact **module names** vary by cluster policy; always run `module avail cuda` (or ask staff) before scripting.

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
module load cuda/12.1   # example only — pick what `module avail` shows
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
salloc --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=4:00:00
# wait for prompt, then:
cd /fs/nexus-scratch/anant04/CNH-HeartMRI-Segmentation
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
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p /fs/nexus-scratch/anant04/logs
module load cuda/12.1   # adjust

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

- **ITK / NRRD reads:** If you see import errors for `ITKReader`, ensure `itk` is installed (`requirements-training.txt` includes it). If multiprocessing workers crash, keep **`--num-workers 0`** (default in training script).
- **Path mismatch:** If training prints “No training cases”, your **`--media-root`** does not contain the `External/...` paths exactly as in the CSV. Fix root or regenerate manifests with the same layout.
- **Do not tune on test:** Use **`--final-test`** only when reporting a locked configuration; routine development uses **val** only.

---

## 8. After the run

- Copy **`config.json`**, **`history.csv`**, **`summary.json`** (small) into git or lab wiki if you want history versioned; **checkpoints** are large — keep on scratch or use **Git LFS** if your policy allows.
- Update **`docs/progress_log.md`** with the cluster path, commit hash, and key hyperparameters for reproducibility.

# SinSR — Cluster Execution Plan

Complete reference for running SinSR on VSC Tier 2 Antwerp.
All scripts live in `hpc/` inside this repo.
Run all commands from the cluster login node unless stated otherwise. You can also browse files, check job status, and open a terminal through the portal at https://portal.hpc.uantwerpen.be/ without a local SSH client.

## Script inventory

All scripts live in `hpc/` inside this repo. `setup_project.sh` and `clone_repo.sh` also have copies in home directory `~/` on the cluster — they must exist before the repo is cloned. The repo is the source of truth; if you update them, copy the new version to cluster.

| Script | Type | What it does |
|---|---|---|
| `setup_project.sh` | bash | Creates folder tree under `$VSC_DATA` |
| `clone_repo.sh` | bash | Clones the SinSR repo; pulls if it already exists |
| `train_bci.sh` | sbatch | Trains on BCI dataset (Apptainer container) |
| `run_sinsr_bci.sh` | bash | Runs inside the container — called by `train_bci.sh` |
| `train_mist.sh` | sbatch | Trains on MIST stains (Apptainer container) |
| `run_sinsr_mist.sh` | bash | Runs inside the container — called by `train_mist.sh` |
| `infer_bci.sh` | sbatch | Runs inference on BCI test set |
| `infer_mist.sh` | sbatch | Runs inference on all four MIST stains sequentially |

---

# Execution order

## 1. One-time setup

**Step 1.1. Add SSH key and connect**

Add your public key to your VSC account via the VSC account page.
Connect:

```bash
ssh <username>@login.hpc.uantwerpen.be
echo $VSC_DATA
echo $VSC_SCRATCH
```

Expected:
- `$VSC_DATA`    = `/data/antwerpen/<group>/<username>`
- `$VSC_SCRATCH` = `/scratch/antwerpen/<group>/<username>`

**Step 1.2. Prepare and upload SquashFS archives**

The container mounts datasets as read-only SquashFS images for fast I/O. Only the `.sqsh` files are needed on the cluster — do not upload the raw dataset directories.

Pack the datasets locally before uploading:

```bash
mksquashfs /path/to/BCI  BCI.sqsh  -noappend
mksquashfs /path/to/MIST MIST.sqsh -noappend
```

The directory structure inside the archives must be:
```
BCI.sqsh root:
    HE/train/    HE/test/
    IHC/train/   IHC/test/

MIST.sqsh root:
    ER/TrainValAB/{trainA,trainB,valA,valB}
    HER2/TrainValAB/...
    Ki67/TrainValAB/...
    PR/TrainValAB/...
```

Upload the archives to the cluster using a file transfer tool (Cyberduck, FileZilla, WinSCP, scp, rsync):
- Destination: `$VSC_SCRATCH/datasets/BCI.sqsh` and `$VSC_SCRATCH/datasets/MIST.sqsh`

Verify:

```bash
unsquashfs -l $VSC_SCRATCH/datasets/BCI.sqsh  | head -10
unsquashfs -l $VSC_SCRATCH/datasets/MIST.sqsh | head -10
```

**Step 1.3. Create the project folder tree (manual one time setup, no sbatch)**

Upload `hpc/setup_project.sh` from the repo to your home directory on the cluster, then run it:

```bash
bash ~/setup_project.sh
```

**Step 1.4. Clone the repository (manual one time setup, no sbatch)**

Upload `hpc/clone_repo.sh` from the repo to your home directory on the cluster, then run it:

```bash
bash ~/clone_repo.sh
```

The `hpc/` folder is now available at `$VSC_DATA/projects/sinsr/code/SinSR/hpc/`.

**Step 1.5. Download model weights**

Required weights (not included in the repo). Download and place them in the `weights/` folder under the project root on the cluster.

| File | Download |
|---|---|
| `weights/resshift_realsrx4_s15_v1.pth` | https://github.com/wyf0912/SinSR/releases/download/v1.0/resshift_realsrx4_s15_v1.pth |
| `weights/autoencoder_vq_f4.pth` | https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth |

Verify:

```bash
ls $VSC_DATA/projects/sinsr/code/SinSR/weights/
```

Expected: `resshift_realsrx4_s15_v1.pth` and `autoencoder_vq_f4.pth`

**Step 1.6. Build the Apptainer container**

The training scripts run inside a container defined by `sinsr_nvidia.def` in the repo root. Build it locally (requires Apptainer installed), then upload the `.sif` to the cluster:

```bash
# On your local machine
apptainer build sinsr_nvidia.sif sinsr_nvidia.def
```

Upload the `.sif` to the cluster using a file transfer tool (Cyberduck, FileZilla, WinSCP, scp, rsync):
- Destination: `$VSC_SCRATCH/containers/sinsr_nvidia.sif`

Verify on the cluster:

```bash
ls $VSC_SCRATCH/containers/sinsr_nvidia.sif
```

---

### 2. Smoke tests (sbatch)

Run a 1-epoch job for each dataset before committing to full training. This confirms the container, dataset mount, and code all work together.

**Step 2.1. BCI smoke test**

Temporarily set in `configs/virtualstaining_bci.yaml` on the cluster:

```yaml
iterations: 269   # ~1 epoch (3896 images / batch 16)
milestones: [269, 269]
save_freq: 269
val_freq: 269
```

Also set `--time=00:30:00` in `train_bci.sh`. Then submit:

```bash
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/train_bci.sh
```

Pass criteria:
1. Log exits without a Python traceback.
2. Loss values are not NaN.
3. Checkpoint exists under `$VSC_DATA/projects/sinsr/outputs/checkpoints/bci_run/`.
4. GPU log has non-zero utilization entries.

**Step 2.2. MIST smoke test (ER stain only)**

Temporarily set in `configs/virtualstaining_mist_er.yaml` on the cluster:

```yaml
iterations: 65   # ~1 epoch (4153 images / batch 16)
milestones: [65, 65]
save_freq: 65
val_freq: 65
```

Also set `--time=00:30:00` in `train_mist.sh`. Then submit:

```bash
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/train_mist.sh
```

Pass criteria: same as BCI, but check `$VSC_DATA/projects/sinsr/outputs/checkpoints/mist_er_run/`.

After both smoke tests pass, restore all config values and `--time` before full training.

---

### 3. Full training (sbatch)

Restore config values on the cluster before submitting:

**BCI** (`configs/virtualstaining_bci.yaml`):
```yaml
iterations: 24400
milestones: [244, 24400]
save_freq: 24400
val_freq: 244
```

**MIST** (all four stain configs):
```yaml
iterations: 6500
milestones: [65, 6500]
save_freq: 6500
val_freq: 65
```

`save_freq` is set to the total iterations so checkpoints are only written once at the end — this avoids NFS write failures during training. `val_freq` is set to one epoch so the full convergence curve is captured in the log.

Restore `--time=20:00:00` in both `train_bci.sh` and `train_mist.sh`.

Submit BCI and all four MIST stains as separate jobs — they can run in parallel if GPUs are available:

```bash
cd $VSC_DATA/projects/sinsr/code/SinSR

sbatch hpc/train_bci.sh

sbatch --job-name=sinsr_mist_er   --export=ALL,STAIN=ER   hpc/train_mist.sh
sbatch --job-name=sinsr_mist_her2 --export=ALL,STAIN=HER2 hpc/train_mist.sh
sbatch --job-name=sinsr_mist_ki67 --export=ALL,STAIN=Ki67 hpc/train_mist.sh
sbatch --job-name=sinsr_mist_pr   --export=ALL,STAIN=PR   hpc/train_mist.sh
```

---

### 4. Inference (sbatch)

Run after training completes. Scripts automatically pick the most recent training run and its highest-iteration EMA checkpoint.

```bash
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/infer_bci.sh
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/infer_mist.sh
```

Verify output:

```bash
find $VSC_DATA/projects/sinsr/outputs/results/bci_test -name "*.png" | wc -l
find $VSC_DATA/projects/sinsr/outputs/results/mist_er_test -name "*.png" | wc -l
```

---

## Monitoring commands

Job status can also be checked from the VSC portal at https://portal.hpc.uantwerpen.be/ without using the command line.

```bash
# Check all running and queued jobs
squeue -u $USER

# Check GPU node state
sinfo -p ampere_gpu

# Get detailed job info including estimated start time
scontrol show job <jobid>

# Watch a log file live
tail -f $VSC_DATA/projects/sinsr/logs/train_bci.<jobid>.out

# Check GPU utilization during training
tail -5 $VSC_DATA/projects/sinsr/logs/gpu_bci_<jobid>.csv

# Find all saved checkpoints
find $VSC_DATA/projects/sinsr/outputs/checkpoints -name "*.pth" | sort
```

---

## Issues

| Problem | Cause | Fix |
|---|---|---|
| `Disk quota exceeded` when creating symlinks | Scratch inode limit (~48k symlinks exceeded quota) | Do not create symlinks — configs already point directly to source dataset paths |
| `RuntimeError: File ... cannot be opened` when saving checkpoint | Transient NFS write error on `$VSC_DATA` | Set `save_freq` to total iterations so checkpoints are only written once at the end |
| `destination .../datasets/BCI doesn't exist in container` | SquashFS mount point directory missing | The training scripts create it automatically with `mkdir -p` before the apptainer call |
| `RuntimeError: Input type (Half) and bias type (float)` during validation | fp16 bug in the validation code path | Set all three `use_fp16: False` in the config: `model.params`, `autoencoder`, and `train` |

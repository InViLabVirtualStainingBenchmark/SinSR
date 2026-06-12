# SinSR — Cluster Instructions

Complete reference for running SinSR on VSC Tier 2 Antwerp.
All scripts live in `hpc/` inside this repo.
Run all commands from the cluster login node unless stated otherwise. You can also browse files, check job status, and open a terminal through the portal at https://portal.hpc.uantwerpen.be/ without a local SSH client.

## Script inventory

`setup_project.sh` and `clone_repo.sh` also have copies in home directory `~/` on the cluster — they must exist before the repo is cloned. The repo is the source of truth; if you update them, copy the new version to cluster.

| Script | Type | What it does |
|---|---|---|
| `setup_project.sh` | bash | Creates folder tree under `$VSC_DATA` |
| `clone_repo.sh` | bash | Clones the SinSR repo; pulls if it already exists |
| `train_bci.sh` | sbatch | Trains on BCI dataset (Apptainer container) |
| `run_sinsr_bci.sh` | bash | Runs inside the container — called by `train_bci.sh` |
| `train_mist.sh` | sbatch | Trains on MIST stains (Apptainer container) |
| `run_sinsr_mist.sh` | bash | Runs inside the container — called by `train_mist.sh` |
| `infer_bci.sh` | sbatch | Runs inference on BCI test set |
| `run_infer_bci.sh` | bash | Runs inside the container — called by `infer_bci.sh` |
| `infer_mist.sh` | sbatch | Runs inference on one MIST stain per job; STAIN = ER \| HER2 \| Ki67 \| PR |
| `run_infer_mist.sh` | bash | Runs inside the container — called by `infer_mist.sh` |
| `eval_bci.sh` | sbatch | Evaluates BCI predictions against ground truth |
| `eval_mist.sh` | sbatch | Evaluates all four MIST stain predictions against ground truth |

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

where `VSC_SCRATCH=/scratch/antwerpen/212/vsc21211` (per-user scratch).

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
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/train/train_bci.sh
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
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/train/train_mist.sh
```

Pass criteria: same as BCI, but check `$VSC_DATA/projects/sinsr/outputs/checkpoints/mist_er_run/`.

After both smoke tests pass, restore all config values and `--time` before full training.

---

### 3. Full training (sbatch)

After the smoke tests pass, restore all config values in the cluster-side copies
(`configs/virtualstaining_bci.yaml` and all four `configs/virtualstaining_mist_*.yaml`):

```yaml
iterations: 45000
milestones: [5000, 500000]
save_freq: 2440
val_freq: 3000
```

These are the committed values. The smoke test temporarily overrides them; this step
restores them. All four MIST configs use the same values as BCI.

Submit BCI and all four MIST stains as separate jobs — they can run in parallel if GPUs are available:

```bash
cd $VSC_DATA/projects/sinsr/code/SinSR

sbatch hpc/train/train_bci.sh

sbatch --job-name=sinsr_mist_er   --export=ALL,STAIN=ER   hpc/train/train_mist.sh
sbatch --job-name=sinsr_mist_her2 --export=ALL,STAIN=HER2 hpc/train/train_mist.sh
sbatch --job-name=sinsr_mist_ki67 --export=ALL,STAIN=Ki67 hpc/train/train_mist.sh
sbatch --job-name=sinsr_mist_pr   --export=ALL,STAIN=PR   hpc/train/train_mist.sh
```

---

### 4. Inference (sbatch)

Run after training completes. Scripts automatically find `ema_best.pth` (the best validation LPIPS checkpoint) by modification time, with fallback to `ema_model_last.pth` if no best checkpoint exists.

```bash
cd $VSC_DATA/projects/sinsr/code/SinSR

sbatch hpc/infer/infer_bci.sh

sbatch --job-name=sinsr_infer_mist_er   --export=ALL,STAIN=ER   hpc/infer/infer_mist.sh
sbatch --job-name=sinsr_infer_mist_her2 --export=ALL,STAIN=HER2 hpc/infer/infer_mist.sh
sbatch --job-name=sinsr_infer_mist_ki67 --export=ALL,STAIN=Ki67 hpc/infer/infer_mist.sh
sbatch --job-name=sinsr_infer_mist_pr   --export=ALL,STAIN=PR   hpc/infer/infer_mist.sh
```

For output folder naming and `RUN_SUFFIX` details, see `DOCUMENTATION.md` section 11.

Verify output (replace GRP_SCRATCH with the actual path):

```bash
GRP_SCRATCH="/scratch/antwerpen/grp/ap_invilab_td_thesis"
find "$GRP_SCRATCH/diffusion-predictions/sinsr/bci_chop256"      -name "*.png" | wc -l
find "$GRP_SCRATCH/diffusion-predictions/sinsr/mist_er_chop256"  -name "*.png" | wc -l
```

---

### 5. Evaluation (sbatch)

Run after inference completes. The eval scripts use the shared `evaluate_nvidia.sif`
container from the evaluate repo. This container must be built once and its weights
pre-downloaded before any eval job runs.

**One-time evaluate container setup**

Follow the instructions in `evaluate/hpc_jobs/cluster_plan_container.md` (in the
evaluate repo). In brief:

1. Build the container locally:
   ```bash
   cd ~/projects/evaluate/hpc_jobs
   sudo APPTAINER_TMPDIR=$HOME/apptainer_tmp APPTAINER_CACHEDIR=$HOME/apptainer_cache \
       apptainer build evaluate_nvidia.sif evaluate_nvidia.def
   ```

2. Upload to the cluster:
   ```bash
   rsync -avz --progress evaluate_nvidia.sif \
       vsc21211@login.hpc.uantwerpen.be:$VSC_SCRATCH/containers/evaluate_nvidia.sif
   ```

3. Pre-download LPIPS and Cellpose weights on the login node (no internet on compute nodes):
   ```bash
   module purge
   module load calcua/2025a
   apptainer exec --nv $VSC_SCRATCH/containers/evaluate_nvidia.sif python -c "
   import lpips; lpips.LPIPS(net='alex'); lpips.LPIPS(net='vgg')
   from cellpose import models; models.CellposeModel(pretrained_model='cpsam')
   print('Weights cached.')
   "
   ```

**Submit eval jobs**

```bash
cd $VSC_DATA/projects/sinsr/code/SinSR

sbatch hpc/eval/eval_bci.sh
sbatch hpc/eval/eval_mist.sh
```

Results are appended to `$VSC_DATA/benchmark_results.csv`.

---

### 6. Resuming Training

If training is interrupted (time limit, node failure, manual cancellation), resume from the last saved checkpoint. The trainer saves a single overwriting checkpoint `ckpts/model_last.pth` every `save_freq` iterations.

Find the timestamped run directory on the login node:

```bash
ls $VSC_DATA/projects/sinsr/outputs/checkpoints/mist_er_run/
```

Resubmit with `RESUME` pointing to `model_last.pth` in that directory:

```bash
sbatch --job-name=sinsr_mist_er \
  --export=ALL,STAIN=ER,RESUME="$VSC_DATA/projects/sinsr/outputs/checkpoints/mist_er_run/2026-05-04-14-19/ckpts/model_last.pth" \
  hpc/train/train_mist.sh
```

For BCI:

```bash
sbatch --export=ALL,RESUME="$VSC_DATA/projects/sinsr/outputs/checkpoints/bci_run/2026-05-04-14-23/ckpts/model_last.pth" \
  hpc/train/train_bci.sh
```

The `RESUME` path must point to `ckpts/model_last.pth` inside the timestamped run directory — not the EMA checkpoint. Training continues from that iteration and saves into the same timestamped directory.

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
| `RuntimeError: File ... cannot be opened` when saving checkpoint | Transient NFS write error on `$VSC_DATA` | Retry the job — the error is transient. If it recurs consistently, reduce `save_freq` to write checkpoints less often |
| `destination .../datasets/BCI doesn't exist in container` | SquashFS mount point directory missing | The training scripts create it automatically with `mkdir -p` before the apptainer call |
| `RuntimeError: Input type (Half) and bias type (float)` during validation | fp16 bug in the validation code path | Set all three `use_fp16: False` in the config: `model.params`, `autoencoder`, and `train` |
| `AttributeError: 'TrainerDistillDifIR' object has no attribute 'loss_mean'` on resumed run | `loss_mean` was only initialized when `current_iters % log_freq == 1`; resumed runs start mid-cycle so the condition is never met before the first access | Fixed in `trainer.py` — `not hasattr(self, 'loss_mean')` guard added to the initialization condition |
| `FATAL: container creation failed: image driver squashfuse_ll instance exited` in eval job | `evaluate_nvidia.sif` does not include the `squashfuse_ll` FUSE driver, so the `image-src=/` sqsh bind mount fails | Evaluation not yet completed — this issue is unresolved |

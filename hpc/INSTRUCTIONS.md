# SinSR — Cluster Execution Plan

Complete reference for running SinSR on VSC Tier 2 Antwerp.
All scripts live in `hpc/` inside this repo.
Run all commands from the cluster login node unless stated otherwise. You can also browse files, check job status, and open a terminal through the portal at https://portal.hpc.uantwerpen.be/ without a local SSH client.

## Script inventory

All scripts live in `hpc/` inside this repo. `setup_project.sh` and `clone_repo.sh` also have copies in `~/` on the cluster — they must exist before the repo is cloned. The repo is the source of truth; if you update them, copy the new version to `~/`.

| Script | Type | What it does |
|---|---|---|
| `setup_project.sh` | bash | Creates folder tree under `$VSC_DATA` |
| `clone_repo.sh` | bash | Clones the SinSR repo; pulls if it already exists |
| `install_deps.sh` | sbatch | Creates venv, installs dependencies, runs sanity checks |
| `train_bci.sh` | sbatch | Trains on BCI dataset |
| `train_mist.sh` | sbatch | Trains on all four MIST stains sequentially (ER, HER2, Ki67, PR) |

---

# Execution order

## 1. One-time setup

**Step 1.1. Add SSH key and connect**

Add your public key to your VSC account via the VSC account page.
Connect:

```bash
ssh vsc21211@login.hpc.uantwerpen.be
echo $VSC_DATA
echo $VSC_SCRATCH
```

Expected:
- `$VSC_DATA`    = `/data/antwerpen/212/vsc21211`
- `$VSC_SCRATCH` = `/scratch/antwerpen/212/vsc21211`

**Step 1.2. Transfer datasets**

Use a file transfer tool (Cyberduck, FileZilla, WinSCP, scp, rsync) to upload datasets from your local machine to the cluster.

BCI:
- Source: local `BCI/` folder containing `HE/` and `IHC/`
- Destination: `$VSC_SCRATCH/datasets/BCI/`
- Result: `$VSC_SCRATCH/datasets/BCI/HE/{train,test}` and `IHC/{train,test}`

MIST:
- Source: local `MIST/` folder containing `ER/`, `HER2/`, `Ki67/`, `PR/`
- Destination: `$VSC_SCRATCH/datasets/MIST/`
- Result: `$VSC_SCRATCH/datasets/MIST/{ER,HER2,Ki67,PR}/TrainValAB/{trainA,trainB,valA,valB}`

Verify on the cluster:

```bash
ls $VSC_SCRATCH/datasets/BCI/HE/train | wc -l
ls $VSC_SCRATCH/datasets/MIST/ER/TrainValAB/trainA | wc -l
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

---

### 2. Environment install (sbatch)

**Step 2.1. Submit install job**

```bash
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/install_deps.sh
```

**Step 2.2. Monitor**

```bash
squeue -u $USER
```

**Step 2.3. Verify log after job completes**

```bash
cat $VSC_DATA/projects/sinsr/logs/install.<jobid>.out
```

Gate: the log must end with `All checks passed` before submitting training.
If any import check fails, add the missing package to the pip install block in `install_deps.sh` and resubmit.

---

### 3. Confirmation run (sbatch)

Run 1 epoch on BCI to confirm everything works before committing to full training.

**Step 3.1. Submit**

```bash
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/train_bci.sh
```

**Step 3.2. Monitor**

```bash
squeue -u $USER
tail -f $VSC_DATA/projects/sinsr/logs/train_bci.<jobid>.out
```

**Step 3.3. Verify pass criteria**

All four must be true before submitting full runs:

1. Log exits without a Python traceback.
2. Loss values in the log are not NaN.
3. Checkpoint exists:
   ```bash
   find $VSC_DATA/projects/sinsr/outputs/checkpoints/bci_run -name "*.pth" | sort
   ```
4. GPU log has entries with non-zero utilization:
   ```bash
   tail -5 $VSC_DATA/projects/sinsr/logs/gpu_bci_<jobid>.csv
   ```

**Step 3.4. Calibrate timing for full runs**

Check the wall time of the 1-epoch job. Multiply by the number of epochs you plan to run and add a 20% margin. Update `--time` in `train_bci.sh` and `train_mist.sh` accordingly.

The partition maximum is 24 hours. If the full run exceeds this, split into multiple jobs resuming from the latest checkpoint.

---

### 4. Full training (sbatch)

Update `iterations`, `milestones`, `save_freq`, `val_freq`, and `log_freq` in the config files before submitting. See `DOCUMENTATION.md` for the parameter reference.

Only one GPU node is available so BCI and MIST jobs will queue sequentially. You can submit both since the scheduler runs them one after the other automatically.

```bash
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/train_bci.sh
sbatch $VSC_DATA/projects/sinsr/code/SinSR/hpc/train_mist.sh
```

Monitor progress:

```bash
squeue -u $USER

tail -f $VSC_DATA/projects/sinsr/logs/train_bci.<jobid>.out
tail -f $VSC_DATA/projects/sinsr/logs/train_mist.<jobid>.out

find $VSC_DATA/projects/sinsr/outputs/checkpoints -name "*.pth" | sort
```

---

## Key paths on the cluster

| Artifact | Path |
|---|---|
| Repo | `$VSC_DATA/projects/sinsr/code/SinSR/` |
| venv | `$VSC_DATA/venvs/venv_sinsr/` |
| Scripts | `$VSC_DATA/projects/sinsr/code/SinSR/hpc/` |
| Configs | `$VSC_DATA/projects/sinsr/code/SinSR/configs/` |
| Weights | `$VSC_DATA/projects/sinsr/code/SinSR/weights/` |
| SLURM logs | `$VSC_DATA/projects/sinsr/logs/` |
| Checkpoints | `$VSC_DATA/projects/sinsr/outputs/checkpoints/` |
| Results | `$VSC_DATA/projects/sinsr/outputs/results/` |
| BCI dataset | `$VSC_SCRATCH/datasets/BCI/` |
| MIST dataset | `$VSC_SCRATCH/datasets/MIST/` |

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

## Common issues

| Problem | Cause | Fix                                                                                                 |
|---|---|-----------------------------------------------------------------------------------------------------|
| `Disk quota exceeded` when creating symlinks | Scratch inode limit (~48k symlinks exceeded quota) | Do not create symlinks — configs already point directly to source dataset paths                     |
| `ModuleNotFoundError: No module named 'albumentations'` | Package missing from venv | `pip install albumentations` or add it to the pip install block in `install_deps.sh` and resubmit   |
| `ERROR: Model weights not found` at job start | Weights not downloaded | Copy the weights as described above to the project root on the cluster                              |
| `RuntimeError: Input type (Half) and bias type (float)` during validation | fp16 bug in the validation code path | Set all three `use_fp16: False` in the config: `model.params`, `autoencoder`, and `train`           |

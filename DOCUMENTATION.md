# SinSR — Virtual Staining Adaptation

Changes made to the original SinSR repository to support paired HE→IHC virtual staining on BCI and MIST datasets.

---

## 1. Config Files

| File | Dataset | Marker |
|---|---|---|
| `configs/virtualstaining_bci.yaml` | BCI | — |
| `configs/virtualstaining_mist_er.yaml` | MIST | ER |
| `configs/virtualstaining_mist_her2.yaml` | MIST | HER2 |
| `configs/virtualstaining_mist_ki67.yaml` | MIST | Ki67 |
| `configs/virtualstaining_mist_pr.yaml` | MIST | PR |

Key changes from the original `configs/SinSR.yaml`:

| Setting | Original | New | Reason |
|---|---|---|---|
| `diffusion.params.sf` | `4` | `1` | No upscaling; staining is 1:1 |
| `data.train.type` | `realesrgan` | `paired` | Real paired images, no synthetic degradation |
| `data.val.type` | `folder` | `paired` | Consistent with train |
| `data.*.params` | RealESRGAN paths | `he_path` / `ihc_path` per dataset | Dataset-specific paths |
| `data.*.params.val_ratio` | — | `0.1` | 10% of training data held out for validation |
| `data.*.params.split` | — | `train` / `val` | Deterministic split by sorted filename |
| `train.learn_xT` | `True` | `False` | Teacher trained on SR degradations, incompatible with HE→IHC |
| `train.microbatch` | `64` | `16` | OOM fix — see note below |
| `train.save_freq` | `2000` | `2440` | Save every ~10% of iterations for resume safety |
| `train.val_freq` | `2000` | `244` | Validate every ~1% for best checkpoint tracking |
| `train.save_images` | `True` | `False` | Saves cluster storage |

All other training parameters (`lr`, `batch`, `num_workers`, `prefetch_factor`, `ema_rate`, `iterations`, `weight_decay`) are kept at the original SinSR values.

`learn_xT: False` — the teacher was trained on super-resolution degradations, so its noise predictions are incompatible with the HE→IHC domain.

The `degradation` block was removed — virtual staining uses real paired images, no synthetic degradation needed.

### microbatch=16 — OOM fix

The original `microbatch: 64` causes CUDA out of memory on a 40 GB A100. The root cause is that virtual staining with `sf=1` feeds **both** HE and IHC images through the autoencoder at full 256×256 resolution. In the original SinSR SR task, the input LQ was 64×64 (4× smaller area) — this task requires roughly 4× more activation memory per sample. `microbatch: 16` processes 16 samples per forward pass with 4 gradient accumulation steps, keeping the effective batch size at 64.

---

## 2. Train / Val / Test Split

To prevent data leakage, training and validation data are split from the same source folder using a deterministic 90/10 split by sorted filename. The test set is a completely separate folder never seen during training.

| Split | BCI | MIST |
|---|---|---|
| Train (90%) | `HE/train` + `IHC/train` first 90% | `trainA` + `trainB` first 90% |
| Val (10%) | `HE/train` + `IHC/train` last 10% | `trainA` + `trainB` last 10% |
| Test | `HE/test` + `IHC/test` | `valA` + `valB` (all) |

The split is controlled by `val_ratio: 0.1` in the config data sections. Both `train` and `val` config sections point to the same source folder — `PairedStainDataset` handles the split internally by sorted filename order.

For MIST, `valA/valB` are used as the held-out test set. They cannot be further split because valA[i] and valB[i] are paired images of the same tissue.

---

## 3. `datapipe/datasets.py` — `PairedStainDataset`

New dataset class for loading matched HE and IHC image pairs. Registered in `create_dataset()` under type `paired`.

- Filenames in `he_path` and `ihc_path` must match exactly
- Crop coordinates and augmentation are synchronized between HE and IHC
- Augmentation is only applied during training
- `val_ratio` parameter controls deterministic train/val split: last `val_ratio` fraction of sorted files = val, remainder = train

---

## 4. `models/gaussian_diffusion.py` — LQ Conditioning Fix

In the original SR setup (`sf=4`), the LQ pixel image is 64×64 (256÷4), which matches the VQ-VAE latent size. With `sf=1`, the LQ image stays at 256×256 and no longer matches the 64×64 latent, causing a size mismatch in the UNet.

Fix: after encoding `z_y`, replace `model_kwargs['lq']` with `z_y` when sizes differ:

```python
if model_kwargs is not None and 'lq' in model_kwargs and model_kwargs['lq'].shape[2:] != z_y.shape[2:]:
    model_kwargs = {**model_kwargs, 'lq': z_y}
```

Applied in: `training_losses_distill`, `p_sample_loop_progressive`, `ddim_sample_loop_progressive`, `ddim_inverse_loop_progressive`.

---

## 5. `trainer.py` — Checkpoint Saving and IQA Fix

### Checkpoint saving

Only two checkpoint files are kept, both overwritten on each update:

| File | When saved | Purpose |
|---|---|---|
| `ema_ckpts/ema_model_last.pth` | Every `save_freq` iterations | Resume if job is killed |
| `ema_ckpts/ema_best.pth` | When val LPIPS improves | Best model for inference |

No numbered checkpoints accumulate. `ema_best.pth` is selected by val LPIPS (lower = better).

### IQA metric clamp

Diffusion outputs can slightly exceed `[0, 1]` after denormalization, causing CLIP-IQA and MUSIQ to crash. Added `.clamp(0, 1)` before passing to IQA metrics in both validation loops:

```python
iqa_input = (results.detach() * 0.5 + 0.5).clamp(0, 1)
```

---

## 6. Training Data Structure

Images must be at least 256×256. Filenames must match between HE and IHC folders.

**Local:**
```
traindata/
    BCI/
        train/he/    train/ihc/
        val/he/      val/ihc/
```

**Cluster** — datasets are mounted as read-only SquashFS archives. The training scripts mount them automatically; configs point directly to the source paths inside the archive:
```
$VSC_SCRATCH/datasets/
    BCI.sqsh   → mounted at $VSC_SCRATCH/datasets/BCI/
        HE/train/    HE/test/
        IHC/train/   IHC/test/
    MIST.sqsh  → mounted at $VSC_SCRATCH/datasets/MIST/
        ER/TrainValAB/trainA   trainB   valA   valB
        HER2/TrainValAB/...
        Ki67/TrainValAB/...
        PR/TrainValAB/...
```

---

## 7. Model Weights

Required weights (not included in the repo). Download and place them in the `weights/` folder under the project root.

| File | Purpose |
|---|---|
| `weights/resshift_realsrx4_s15_v1.pth` | Teacher model for distillation |
| `weights/autoencoder_vq_f4.pth` | VQ-VAE encoder/decoder |

| File | Download |
|---|---|
| `weights/resshift_realsrx4_s15_v1.pth` | https://github.com/wyf0912/SinSR/releases/download/v1.0/resshift_realsrx4_s15_v1.pth |
| `weights/autoencoder_vq_f4.pth` | https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth |

You can also download by running inference once — weights are fetched automatically from GitHub releases:

```bash
python inference.py --task realsrx4 -i testdata/RealSet65/00003.png -o ./results --scale 4
```

---

## 8. HPC Scripts

Scripts in `hpc/` for running on the VSC cluster. Training uses an Apptainer container (`sinsr_nvidia.def` in the repo root). Evaluation uses a shared container from the evaluate repo (`evaluate_nvidia.sif`). See `hpc/INSTRUCTIONS.md` for the full setup sequence.

| Script | How to run | Purpose |
|---|---|---|
| `setup_project.sh` | `bash setup_project.sh` | Create project folder structure |
| `clone_repo.sh` | `bash clone_repo.sh` | Clone the repository |
| `train_bci.sh` | `sbatch train_bci.sh` | Train on BCI dataset (Apptainer container) |
| `run_sinsr_bci.sh` | called by `train_bci.sh` | Runs training inside the container |
| `train_mist.sh` | `sbatch --job-name=sinsr_mist_er --export=ALL,STAIN=ER train_mist.sh` | Train one MIST stain per job; STAIN = ER \| HER2 \| Ki67 \| PR |
| `run_sinsr_mist.sh` | called by `train_mist.sh` | Runs training inside the container |
| `infer_bci.sh` | `sbatch infer_bci.sh` | Run inference on BCI test set |
| `run_infer_bci.sh` | called by `infer_bci.sh` | Runs inference inside the container; auto-discovers `ema_best.pth` |
| `infer_mist.sh` | `sbatch infer_mist.sh` | Run inference on all four MIST stains |
| `run_infer_mist.sh` | called by `infer_mist.sh` | Runs inference inside the container per stain |
| `eval_bci.sh` | `sbatch eval_bci.sh` | Evaluate BCI predictions using evaluate container |
| `eval_mist.sh` | `sbatch eval_mist.sh` | Evaluate all four MIST stain predictions using evaluate container |

`setup_project.sh` and `clone_repo.sh` must be run manually on the login node. All remaining scripts are submitted as SLURM jobs.

Checkpoints are saved under `$VSC_DATA/projects/sinsr/outputs/checkpoints/`. SLURM and GPU logs are saved under `$VSC_DATA/projects/sinsr/logs/`.

### Evaluation pipeline

Evaluation uses the shared `evaluate_nvidia.sif` container from the evaluate repo. The eval scripts mount the dataset SquashFS directly for GT access — no unsquashing needed. Results are appended to `$VSC_DATA/benchmark_results.csv`.

Submit in order:
```bash
sbatch hpc/infer_bci.sh    # after training completes
sbatch hpc/eval_bci.sh     # after inference completes
```

The eval container must be built once locally from `evaluate/hpc_jobs/evaluate_nvidia.def` and uploaded to `$VSC_SCRATCH/containers/evaluate_nvidia.sif`. LPIPS weights must be pre-downloaded on the login node before the first eval job — see the evaluate repo's `hpc_jobs/cluster_plan_container.md`.

---

## 9. Running

**On cluster** (requires container and SquashFS archives — see `hpc/INSTRUCTIONS.md`):

```bash
# Training
sbatch hpc/train_bci.sh

sbatch --job-name=sinsr_mist_er   --export=ALL,STAIN=ER   hpc/train_mist.sh
sbatch --job-name=sinsr_mist_her2 --export=ALL,STAIN=HER2 hpc/train_mist.sh
sbatch --job-name=sinsr_mist_ki67 --export=ALL,STAIN=Ki67 hpc/train_mist.sh
sbatch --job-name=sinsr_mist_pr   --export=ALL,STAIN=PR   hpc/train_mist.sh

# Inference (after training)
sbatch hpc/infer_bci.sh
sbatch hpc/infer_mist.sh

# Evaluation (after inference)
sbatch hpc/eval_bci.sh
sbatch hpc/eval_mist.sh
```

Each MIST stain runs as a separate job. All four can run simultaneously if GPUs are available.

**Local single GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python main_distill.py --cfg_path configs/virtualstaining_bci.yaml --save_dir ./outputs/bci_run
```
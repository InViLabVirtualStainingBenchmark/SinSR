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

| Setting | Original | New |
|---|---|---|
| `diffusion.params.sf` | `4` | `1` |
| `data.train.type` | `realesrgan` | `paired` |
| `data.val.type` | `folder` | `paired` |
| `data.*.params` | RealESRGAN paths | `he_path` / `ihc_path` per dataset |
| `autoencoder.use_fp16` | `True` | `False` |
| `train.learn_xT` | `True` | `False` |
| `train.batch` | `[64, 8]` | `[16, 16]` |
| `train.microbatch` | `64` | `16` |
| `train.num_workers` | `16` | `8` |
| `train.prefetch_factor` | `4` | `8` |

`autoencoder.use_fp16` is disabled — validation crashed with a mixed-precision error (`Input type (Half) and bias type (float) should be the same`). The exact source was not isolated so all fp16 flags were set to False.

`learn_xT: False` — the teacher was trained on super-resolution degradations, so its noise predictions are incompatible with the HE→IHC domain.

The `degradation` block was removed — virtual staining uses real paired images, no synthetic degradation needed.

---

## 2. `datapipe/datasets.py` — Added `PairedStainDataset`

New dataset class for loading matched HE and IHC image pairs. Registered in `create_dataset()` under type `paired`.

- Filenames in `he_path` and `ihc_path` must match exactly
- Crop coordinates and augmentation are synchronized between HE and IHC
- Augmentation is only applied during training

---

## 3. `models/gaussian_diffusion.py` — LQ Conditioning Fix

In the original SR setup (`sf=4`), the LQ pixel image is 64×64 (256÷4), which matches the VQ-VAE latent size. With `sf=1`, the LQ image stays at 256×256 and no longer matches the 64×64 latent, causing a size mismatch in the UNet.

Fix: after encoding `z_y`, replace `model_kwargs['lq']` with `z_y` when sizes differ:

```python
if model_kwargs is not None and 'lq' in model_kwargs and model_kwargs['lq'].shape[2:] != z_y.shape[2:]:
    model_kwargs = {**model_kwargs, 'lq': z_y}
```

Applied in: `training_losses_distill`, `p_sample_loop_progressive`, `ddim_sample_loop_progressive`, `ddim_inverse_loop_progressive`.

---

## 4. `trainer.py` — IQA Metric Clamp

Diffusion outputs can slightly exceed `[0, 1]` after denormalization, causing CLIP-IQA and MUSIQ to crash. Added `.clamp(0, 1)` before passing to IQA metrics in both validation loops:

```python
iqa_input = (results.detach() * 0.5 + 0.5).clamp(0, 1)
```

---

## 5. Training Data Structure

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

## 6. Model Weights

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

## 7. HPC Scripts

Scripts in `hpc/` for running on the VSC cluster. Training uses an Apptainer container (`sinsr_nvidia.def` in the repo root). See `hpc/INSTRUCTIONS.md` for the full setup sequence including building the container and packing datasets into SquashFS archives.

| Script | How to run | Purpose |
|---|---|---|
| `setup_project.sh` | `bash setup_project.sh` | Create project folder structure |
| `clone_repo.sh` | `bash clone_repo.sh` | Clone the repository |
| `train_bci.sh` | `sbatch train_bci.sh` | Train on BCI dataset (Apptainer container) |
| `run_sinsr_bci.sh` | called by `train_bci.sh` | Runs training inside the container |
| `train_mist.sh` | `sbatch --job-name=sinsr_mist_er --export=ALL,STAIN=ER train_mist.sh` | Train one MIST stain per job; STAIN = ER \| HER2 \| Ki67 \| PR |
| `run_sinsr_mist.sh` | called by `train_mist.sh` | Runs training inside the container |
| `infer_bci.sh` | `sbatch infer_bci.sh` | Run inference on BCI test set |
| `infer_mist.sh` | `sbatch infer_mist.sh` | Run inference on all four MIST stains |
| `eval_bci.sh` | `sbatch eval_bci.sh` | Evaluate BCI predictions |
| `eval_mist.sh` | `sbatch eval_mist.sh` | Evaluate all four MIST stain predictions |

`setup_project.sh` and `clone_repo.sh` must be run manually on the login node. They also have copies in `~/` on the cluster since they are needed before the repo is cloned. The remaining scripts are submitted as SLURM jobs.

Checkpoints are saved under `$VSC_DATA/projects/sinsr/outputs/checkpoints/`. SLURM and GPU logs are saved under `$VSC_DATA/projects/sinsr/logs/`. Log filenames include the job name: `sinsr_bci.<jobid>.out`, `sinsr_mist_er.<jobid>.out`, etc.

---

## 8. Running

**On cluster** (requires container and SquashFS archives. See `hpc/INSTRUCTIONS.md`):

From the project root:
```bash
sbatch hpc/train_bci.sh

sbatch --job-name=sinsr_mist_er   --export=ALL,STAIN=ER   hpc/train_mist.sh
sbatch --job-name=sinsr_mist_her2 --export=ALL,STAIN=HER2 hpc/train_mist.sh
sbatch --job-name=sinsr_mist_ki67 --export=ALL,STAIN=Ki67 hpc/train_mist.sh
sbatch --job-name=sinsr_mist_pr   --export=ALL,STAIN=PR   hpc/train_mist.sh
```

Each MIST stain runs as a separate job (~6h each). All four can run simultaneously if GPUs are available.

**Local single GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python main_distill.py --cfg_path configs/virtualstaining_bci.yaml --save_dir ./outputs/bci_run
```
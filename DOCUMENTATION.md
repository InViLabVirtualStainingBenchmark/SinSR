# SinSR - Virtual Staining Adaptation

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
| `train.iterations` | `500000` | `45000` | Capped by compute time budget on HPC |
| `train.save_freq` | `2000` | `2440` | ~18 saves over full training for resume safety |
| `train.val_freq` | `2000` | `3000` | 15 validation passes over full training for best checkpoint tracking |
| `train.save_images` | `True` | `False` | Saves cluster storage |

All other training parameters (`lr`, `batch`, `num_workers`, `prefetch_factor`, `ema_rate`, `weight_decay`) are kept at the original SinSR values.

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

Three checkpoint files are kept, all overwritten on each update:

| File | When saved | Purpose |
|---|---|---|
| `ckpts/model_last.pth` | Every `save_freq` iterations | Resume anchor — passed to `--resume` |
| `ema_ckpts/ema_model_last.pth` | Every `save_freq` iterations | EMA weights for inference fallback |
| `ema_ckpts/ema_best.pth` | When val LPIPS improves | Best EMA weights for inference |

No numbered checkpoints accumulate. `ema_best.pth` is selected by val LPIPS (lower = better).

### IQA metric clamp

Diffusion outputs can slightly exceed `[0, 1]` after denormalization, causing CLIP-IQA and MUSIQ to crash. Added `.clamp(0, 1)` before passing to IQA metrics in both validation loops:

```python
iqa_input = (results.detach() * 0.5 + 0.5).clamp(0, 1)
```

### Resume crash fix (`loss_mean`)

`loss_mean` was only initialized inside `log_step_train` when `current_iters % log_freq[0] == 1`. On a resumed run `current_iters` picks up mid-cycle, so that condition is never true before the first access, crashing with `AttributeError: 'TrainerDistillDifIR' object has no attribute 'loss_mean'`.

Fix: added `or not hasattr(self, 'loss_mean')` to the initialization guard in both `log_step_train` implementations (`TrainerBase` and `TrainerDistillDifIR`):

```python
if self.current_iters % self.configs.train.log_freq[0] == 1 or not hasattr(self, 'loss_mean'):
```

---

## 6. Training Data Structure

Images must be at least 256×256. Filenames must match between HE and IHC folders.

**Cluster** — datasets are mounted as read-only SquashFS archives. For archive structure, upload paths, and mount verification, see `hpc/INSTRUCTIONS.md` step 1.2.

---

## 7. Model Weights

Required weights (not included in the repo). Download and place them in the `weights/` folder under the project root.

| File | Purpose | Download |
|---|---|---|
| `weights/resshift_realsrx4_s15_v1.pth` | Teacher model for distillation | https://github.com/wyf0912/SinSR/releases/download/v1.0/resshift_realsrx4_s15_v1.pth |
| `weights/autoencoder_vq_f4.pth` | VQ-VAE encoder/decoder | https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth |

---

## 8. HPC Scripts

Scripts in `hpc/` for running on the VSC cluster. See `hpc/INSTRUCTIONS.md` for the full script inventory, setup sequence, and all submission commands.

---

## 9. Resuming Training

See `hpc/INSTRUCTIONS.md` section 6.

---

## 10. Running

**Cluster** — see `hpc/INSTRUCTIONS.md` for all sbatch commands (training, inference, evaluation).

**Local single GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python main_distill.py --cfg_path configs/virtualstaining_bci.yaml --save_dir ./outputs/bci_run
```

---

## 11. Inference

`inference.py` takes a folder of HE images and writes predicted IHC images to an output folder. It always uses a sliding window — there is no full-image mode.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `-i` | required | Input folder of HE images |
| `-o` | required | Output folder for predicted IHC images |
| `-c` / `--config` | required | YAML config path (e.g. `configs/virtualstaining_bci.yaml`) |
| `--ckpt` | required | Checkpoint path (`ema_best.pth` or `ema_model_last.pth`) |
| `--scale` | `4` | Scale factor — must be `1` for virtual staining (1:1 resolution) |
| `--one_step` | flag | Single-step distilled inference; required for the trained student model |
| `--chop_size` | `512` | Sliding window patch size in pixels; choices: `256`, `512` |
| `--seed` | `12345` | Random seed |
| `--ddim` | flag | Use DDIM sampling instead of DDPM |

### Sliding window

| `--chop_size` | Stride | Overlap |
|---|---|---|
| `256` | `224` | 32 px |
| `512` | `448` | 64 px |

### Current configuration

Both `run_infer_bci.sh` and `run_infer_mist.sh` use `--chop_size 256`:

```bash
python3 inference.py \
    -c configs/virtualstaining_bci.yaml \
    --ckpt path/to/ema_best.pth \
    -i <HE_test_folder> \
    -o <output_folder> \
    --scale 1 \
    --one_step \
    --chop_size 256
```

`--scale 1` is required: the original SinSR default is 4× super-resolution; without this the output dimensions would be wrong.
`--one_step` is required: the distilled student model predicts directly in a single denoising step.
`--chop_size 256` overrides the default of 512. To use 512×512 patches instead, change `--chop_size 256` to `--chop_size 512` in `hpc/infer/run_infer_bci.sh` and `hpc/infer/run_infer_mist.sh`.

### Output locations

Inference output folders are written to `$GRP_SCRATCH/diffusion-predictions/sinsr/` where `GRP_SCRATCH=/scratch/antwerpen/grp/ap_invilab_td_thesis`. The folder name is controlled by `RUN_SUFFIX` (default: `chop256`):

| Run | Output folder |
|---|---|
| BCI | `$GRP_SCRATCH/diffusion-predictions/sinsr/bci_chop256/` |
| MIST ER | `$GRP_SCRATCH/diffusion-predictions/sinsr/mist_er_chop256/` |
| MIST HER2 | `$GRP_SCRATCH/diffusion-predictions/sinsr/mist_her2_chop256/` |
| MIST Ki67 | `$GRP_SCRATCH/diffusion-predictions/sinsr/mist_ki67_chop256/` |
| MIST PR | `$GRP_SCRATCH/diffusion-predictions/sinsr/mist_pr_chop256/` |

To run with a different setting and save to a new folder without overwriting the default results, pass `RUN_SUFFIX` at submission time. Use the same value for both inference and eval:

```bash
sbatch --export=ALL,RUN_SUFFIX=chop512           hpc/infer/infer_bci.sh
sbatch --export=ALL,RUN_SUFFIX=chop512           hpc/eval/eval_bci.sh

sbatch --export=ALL,STAIN=ER,RUN_SUFFIX=chop512  hpc/infer/infer_mist.sh
sbatch --export=ALL,RUN_SUFFIX=chop512           hpc/eval/eval_mist.sh
```

---

## 12. Current Status

| Stage | Status | Notes |
|---|---|---|
| Training — BCI | Complete | `ema_best.pth` saved under `checkpoints/bci_run/` |
| Training — MIST ER / HER2 / Ki67 / PR | Complete | `ema_best.pth` saved for each stain |
| Inference — BCI | Complete | `RUN_SUFFIX=chop256` (`--chop_size 256`) |
| Inference — MIST | Complete | `RUN_SUFFIX=chop256` (`--chop_size 256`) for all four stains |
| Inference — chop512 / full-image | Not run | Planned to compare patch visibility against chop256, as done for SR3; not attempted due to time |
| Evaluation — BCI / MIST | Failed | See Issues section in `hpc/INSTRUCTIONS.md` |

The chop256 inference setting was chosen to compare patch-level visibility across different crop sizes (256, 512, full-image), mirroring the SR3 evaluation approach. Only chop256 was completed before time ran out.

---

## 13. Environment

This section is for local development only. On the cluster, the container handles the environment.

- Python 3.9
- PyTorch 2.1.2 + CUDA 12.1

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_frozen.txt
```
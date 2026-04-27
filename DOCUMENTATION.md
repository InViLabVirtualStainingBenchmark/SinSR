# SinSR — Virtual Staining Adaptation

Changes made to the original SinSR repository to support paired HE→IHC virtual staining on BCI and MIST datasets.

---

## 1. Config Files

| File | Dataset | Marker |
|---|---|---|
| `configs/virtualstaining.yaml` | BCI | — |
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
| `train.batch` | `[64, 8]` | `[4, 4]` |
| `train.iterations` | `500000` | `4` (smoke test) |
| `train.log_freq` | `[200, 5000, 1]` | `[2, 4, 1]` |

The `degradation` block was removed — virtual staining uses real paired images, no synthetic degradation needed.

---

## 2. `datapipe/datasets.py` — Added `PairedStainDataset`

New dataset class for loading matched HE and IHC image pairs. Registered in `create_dataset()` under type `paired`.

- Filenames in `he/` and `ihc/` must match exactly
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

```
traindata/
    BCI/
        train/he/   train/ihc/
        val/he/     val/ihc/
    MIST/
        ER/   train/he/  train/ihc/  val/he/  val/ihc/
        HER2/ ...
        Ki67/ ...
        PR/   ...
```

Images must be at least 256×256. Filenames must match between `he/` and `ihc/` folders.

---

## 6. Model Weights

Required weights (not included in the repo):

| File | Purpose |
|---|---|
| `weights/resshift_realsrx4_s15_v1.pth` | Teacher model for distillation |
| `weights/autoencoder_vq_f4.pth` | VQ-VAE encoder/decoder |

Download by running inference once — weights are fetched automatically from GitHub releases:

```bash
python inference.py --task realsrx4 -i testdata/RealSet65/00003.png -o ./results --scale 4
```

---

## Running

Requires CUDA. Use `CUDA_VISIBLE_DEVICES=0` to restrict to a single GPU, which skips the distributed setup.

**Single GPU:**
```bash
# BCI
CUDA_VISIBLE_DEVICES=0 python main_distill.py --cfg_path configs/virtualstaining_bci.yaml --save_dir ./saved_logs/BCI

# MIST — ER
CUDA_VISIBLE_DEVICES=0 python main_distill.py --cfg_path configs/virtualstaining_mist_er.yaml --save_dir ./saved_logs/MIST_ER

# MIST — HER2
CUDA_VISIBLE_DEVICES=0 python main_distill.py --cfg_path configs/virtualstaining_mist_her2.yaml --save_dir ./saved_logs/MIST_HER2

# MIST — Ki67
CUDA_VISIBLE_DEVICES=0 python main_distill.py --cfg_path configs/virtualstaining_mist_ki67.yaml --save_dir ./saved_logs/MIST_Ki67

# MIST — PR
CUDA_VISIBLE_DEVICES=0 python main_distill.py --cfg_path configs/virtualstaining_mist_pr.yaml --save_dir ./saved_logs/MIST_PR
```
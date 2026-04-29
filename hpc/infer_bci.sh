#!/bin/bash
#SBATCH --job-name=sinsr_infer_bci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sinsr/logs/infer_bci.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sinsr/logs/infer_bci.%j.err

# infer_bci.sh — BCI test inference. Submit after train_bci.sh completes.
# Automatically picks the most recent training run and its highest-iteration EMA checkpoint.

set -euo pipefail

REPO_DIR="$VSC_DATA/projects/sinsr/code/SinSR"
CONFIG="$REPO_DIR/configs/virtualstaining_bci.yaml"
HE_TEST="$VSC_SCRATCH/datasets/BCI/HE/test"
OUT_DIR="$VSC_DATA/projects/sinsr/outputs/results/bci_test"
CKPT_BASE="$VSC_DATA/projects/sinsr/outputs/checkpoints/bci_run"

# =========================
# MODULES
# =========================

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load OpenCV/4.8.1-foss-2023a-contrib

source "$VSC_DATA/venvs/venv_sinsr/bin/activate"

# =========================
# PRE-FLIGHT CHECKS
# =========================

echo "=== Environment ==="
which python
python -V
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo ""
echo "=== Checkpoint check ==="
RUN_DIR=$(find "$CKPT_BASE" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: No training run found under $CKPT_BASE"
    deactivate; exit 1
fi
CKPT_PATH=$(find "$RUN_DIR/ema_ckpts" -name "ema_model_*.pth" 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: No EMA checkpoint found in $RUN_DIR/ema_ckpts"
    deactivate; exit 1
fi
echo "  Run dir    : $RUN_DIR"
echo "  Checkpoint : $CKPT_PATH"

echo ""
echo "=== Test dataset check ==="
if [ ! -d "$HE_TEST" ]; then
    echo "ERROR: Missing dataset folder: $HE_TEST"
    deactivate; exit 1
fi
echo "  HE test : $(find "$HE_TEST" -maxdepth 1 -type f | wc -l) images"

mkdir -p "$OUT_DIR"

# =========================
# GPU LOGGING
# =========================

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$VSC_DATA/projects/sinsr/logs/gpu_infer_bci_${SLURM_JOB_ID}.csv" & GPU_LOG_PID=$!

# =========================
# INFERENCE
# =========================

cd "$REPO_DIR"

echo ""
echo "=== Starting BCI inference ==="
echo "  input  : $HE_TEST"
echo "  output : $OUT_DIR"

python inference.py \
    -c "$CONFIG" \
    --ckpt "$CKPT_PATH" \
    -i "$HE_TEST" \
    -o "$OUT_DIR" \
    --scale 1 \
    --one_step

# =========================
# POST-RUN REPORT
# =========================

kill $GPU_LOG_PID 2>/dev/null || true

echo ""
echo "=== Output image count ==="
find "$OUT_DIR" -name "*.png" | wc -l

echo ""
echo "=== GPU log ==="
echo "  $VSC_DATA/projects/sinsr/logs/gpu_infer_bci_${SLURM_JOB_ID}.csv"

deactivate
echo ""
echo "BCI inference complete."

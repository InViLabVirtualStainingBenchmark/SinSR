#!/bin/bash
# Runs inside the Apptainer container for SinSR BCI inference.
# Called by infer_bci.sh via: apptainer exec ... bash run_infer_bci.sh
# Variables exported from the SLURM script: REPO_DIR, LOG_DIR, OUT_DIR, CKPT_BASE, SLURM_JOB_ID

set -euo pipefail

echo "=== Environment ==="
python3 --version
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$LOG_DIR/gpu_infer_bci_${SLURM_JOB_ID}.csv" & GPU_LOG_PID=$!

# =========================================================
# CHECKPOINT DISCOVERY
# =========================================================

echo ""
echo "=== Checkpoint check ==="
CKPT_PATH=$(find "$CKPT_BASE" -name "ema_best.pth" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [ -z "$CKPT_PATH" ]; then
    CKPT_PATH=$(find "$CKPT_BASE" -name "ema_model_last.pth" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
fi
if [ -z "$CKPT_PATH" ]; then
    echo "ERROR: No EMA checkpoint found under $CKPT_BASE"
    kill $GPU_LOG_PID 2>/dev/null || true; exit 1
fi
echo "  Checkpoint : $CKPT_PATH"

# =========================================================
# DATASET CHECK
# =========================================================

GRP_SCRATCH="/scratch/antwerpen/grp/ap_invilab_td_thesis"
HE_TEST="$GRP_SCRATCH/datasets/BCI/HE/test"
CONFIG="$REPO_DIR/configs/virtualstaining_bci.yaml"

echo ""
echo "=== Test dataset check ==="
if [ ! -d "$HE_TEST" ]; then
    echo "ERROR: Missing dataset folder: $HE_TEST"
    kill $GPU_LOG_PID 2>/dev/null || true; exit 1
fi
echo "  HE test : $(find "$HE_TEST" -maxdepth 1 -type f | wc -l) images"

# =========================================================
# INFERENCE
# =========================================================

echo ""
echo "=== Starting BCI inference ==="
echo "  input  : $HE_TEST"
echo "  output : $OUT_DIR"

cd "$REPO_DIR"

python3 inference.py \
    -c "$CONFIG" \
    --ckpt "$CKPT_PATH" \
    -i "$HE_TEST" \
    -o "$OUT_DIR" \
    --scale 1 \
    --one_step

kill $GPU_LOG_PID 2>/dev/null || true

echo ""
echo "=== Output image count ==="
find "$OUT_DIR" -name "*.png" | wc -l
echo "GPU log : $LOG_DIR/gpu_infer_bci_${SLURM_JOB_ID}.csv"

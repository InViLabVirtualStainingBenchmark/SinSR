#!/bin/bash
# Runs inside the Apptainer container for SinSR MIST inference.
# Called by infer_mist.sh via: apptainer exec ... bash run_infer_mist.sh
# Variables exported from the SLURM script: REPO_DIR, LOG_DIR, STAIN, CKPT_BASE, OUT_DIR, SLURM_JOB_ID

set -euo pipefail

echo "=== Environment ==="
python3 --version
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
           --format=csv -l 5 \
    > "$LOG_DIR/gpu_infer_mist_${STAIN}_${SLURM_JOB_ID}.csv" & GPU_LOG_PID=$!

stain_lower=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')
CONFIG="$REPO_DIR/configs/virtualstaining_mist_${stain_lower}.yaml"
HE_TEST="$VSC_SCRATCH/datasets/MIST/$STAIN/TrainValAB/valA"
CKPT_DIR_BASE="$CKPT_BASE/mist_${stain_lower}_run"

cd "$REPO_DIR"

# =========================================================
# CHECKPOINT DISCOVERY
# =========================================================

echo ""
echo "=== Checkpoint check ==="
RUN_DIR=$(find "$CKPT_DIR_BASE" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: No training run found under $CKPT_DIR_BASE"
    kill $GPU_LOG_PID 2>/dev/null || true; exit 1
fi
if [ -f "$RUN_DIR/ema_ckpts/ema_best.pth" ]; then
    CKPT_PATH="$RUN_DIR/ema_ckpts/ema_best.pth"
elif [ -f "$RUN_DIR/ema_ckpts/ema_model_last.pth" ]; then
    CKPT_PATH="$RUN_DIR/ema_ckpts/ema_model_last.pth"
else
    echo "ERROR: No EMA checkpoint found in $RUN_DIR/ema_ckpts"
    kill $GPU_LOG_PID 2>/dev/null || true; exit 1
fi
echo "  Run dir    : $RUN_DIR"
echo "  Checkpoint : $CKPT_PATH"

# =========================================================
# DATASET CHECK
# =========================================================

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
echo "=== Starting $STAIN inference ==="
echo "  input  : $HE_TEST"
echo "  output : $OUT_DIR"

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
      echo "  $STAIN : $(find "$OUT_DIR" -name "*.png" | wc -l) images written."
      echo "GPU log : $LOG_DIR/gpu_infer_mist_${STAIN}_${SLURM_JOB_ID}.csv"
#!/bin/bash
# Runs inside the Apptainer container for a single MIST stain.
# Called by train_mist.sh via: apptainer exec ... bash run_sinsr_mist.sh
# Variables exported from the SLURM script: STAIN, REPO_DIR, CONFIG, SAVE_DIR, LOG_DIR, SLURM_JOB_ID

set -euo pipefail

echo "=== Environment ==="
python3 --version
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# Start GPU logging in background
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 5 > "$LOG_DIR/gpu_mist_${STAIN}_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

echo ""
echo "Starting MIST training — stain: $STAIN"
echo "  config : $CONFIG"
echo "  save   : $SAVE_DIR"

cd "$REPO_DIR"
python3 main_distill.py \
    --cfg_path "$CONFIG" \
    --save_dir "$SAVE_DIR"

kill $GPU_LOG_PID 2>/dev/null || true

echo ""
echo "$STAIN training complete."
echo "Checkpoints : $SAVE_DIR"
echo "GPU log     : $LOG_DIR/gpu_mist_${STAIN}_${SLURM_JOB_ID}.csv"

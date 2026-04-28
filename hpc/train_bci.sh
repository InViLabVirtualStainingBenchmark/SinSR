#!/bin/bash
#SBATCH --job-name=sinsr_bci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sinsr/logs/train_bci.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sinsr/logs/train_bci.%j.err

set -euo pipefail

# =========================================================
# USER SETTINGS
# =========================================================

REPO_DIR="$VSC_DATA/projects/sinsr/code/SinSR"
# Root folder of the SinSR repository

CONFIG="$REPO_DIR/configs/virtualstaining_bci.yaml"
DATA_ROOT="$VSC_SCRATCH/sinsr/bci"
SAVE_DIR="$VSC_DATA/projects/sinsr/outputs/checkpoints/bci_run1"
# Where checkpoints, logs, and sample images will be saved.
# Change the suffix for each new run to avoid overwriting.

# =========================================================
# ENVIRONMENT
# =========================================================

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load OpenCV/4.8.1-foss-2023a-contrib

source "$VSC_DATA/venvs/venv_sinsr/bin/activate"

# =========================================================
# PRE-FLIGHT CHECKS
# =========================================================

echo "Python:"
which python
python -V

echo "Checking repository path..."
if [ ! -f "$REPO_DIR/main_distill.py" ]; then
    echo "ERROR: main_distill.py not found in $REPO_DIR"
    deactivate; exit 1
fi

echo "Checking config..."
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    deactivate; exit 1
fi

echo "Checking model weights..."
if [ ! -f "$REPO_DIR/weights/resshift_realsrx4_s15_v1.pth" ] || [ ! -f "$REPO_DIR/weights/autoencoder_vq_f4.pth" ]; then
    echo "ERROR: Model weights not found in $REPO_DIR/weights/"
    deactivate; exit 1
fi

echo "Checking dataset..."
for split in train val; do
    for domain in input target; do
        if [ ! -d "$DATA_ROOT/$split/$domain" ]; then
            echo "ERROR: Missing dataset folder: $DATA_ROOT/$split/$domain"
            deactivate; exit 1
        fi
    done
done

echo "Training images : $(find "$DATA_ROOT/train/input" -maxdepth 1 \( -type f -o -type l \) | wc -l)"
echo "Validation images: $(find "$DATA_ROOT/val/input" -maxdepth 1 \( -type f -o -type l \) | wc -l)"

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# =========================================================
# TRAINING
# =========================================================

cd "$REPO_DIR"

echo ""
echo "Starting BCI training..."
echo "  config : $CONFIG"
echo "  data   : $DATA_ROOT"
echo "  save   : $SAVE_DIR"

# Start GPU logging in background
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
           --format=csv -l 5 > "$VSC_DATA/projects/sinsr/logs/gpu_bci_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

CUDA_VISIBLE_DEVICES=0 python main_distill.py \
    --cfg_path "$CONFIG" \
    --save_dir "$SAVE_DIR"

# Stop GPU logging
kill $GPU_LOG_PID 2>/dev/null || true

deactivate

echo ""
echo "BCI training complete."
echo "Checkpoints : $SAVE_DIR"
echo "GPU log     : $VSC_DATA/projects/sinsr/logs/gpu_bci_${SLURM_JOB_ID}.csv"
#!/bin/bash
#SBATCH --job-name=sinsr_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sinsr/logs/train.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sinsr/logs/train.%j.err

set -euo pipefail

# =========================================================
# USER SETTINGS
# Change these between runs.
# =========================================================

REPO_DIR="$VSC_DATA/projects/sinsr/code/SinSR"
# Root folder of the SinSR repository

CONFIG="$REPO_DIR/configs/virtualstaining_bci.yaml"
# Config file for this run. Available configs:
#   virtualstaining_bci.yaml
#   virtualstaining_mist_er.yaml
#   virtualstaining_mist_her2.yaml
#   virtualstaining_mist_ki67.yaml
#   virtualstaining_mist_pr.yaml

DATA_ROOT="$VSC_SCRATCH/sinsr/bci"
# Dataset root created by prepare_dataset.sh.
# Used here only for pre-flight checks — the actual paths
# are read from the config. Must match what the config points to:
#   bci       → virtualstaining_bci.yaml
#   mist_ER   → virtualstaining_mist_er.yaml
#   mist_HER2 → virtualstaining_mist_her2.yaml
#   mist_Ki67 → virtualstaining_mist_ki67.yaml
#   mist_PR   → virtualstaining_mist_pr.yaml

SAVE_DIR="$VSC_DATA/projects/sinsr/outputs/checkpoints/bci_run1"
# Where checkpoints, logs, and sample images will be saved.
# Change the suffix for each new run to avoid overwriting.

# =========================================================
# ENVIRONMENT
# =========================================================

module purge
# Start from a clean module environment

module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load OpenCV/4.8.1-foss-2023a-contrib

source "$VSC_DATA/venvs/venv_sinsr/bin/activate"

# =========================================================
# BASIC CHECKS
# =========================================================

echo "Python executable:"
which python

echo "Python version:"
python -V

echo "Checking repository path..."
if [ ! -f "$REPO_DIR/main_distill.py" ]; then
    echo "ERROR: main_distill.py not found in $REPO_DIR"
    deactivate
    exit 1
fi

echo "Checking config..."
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    deactivate
    exit 1
fi

echo "Checking model weights..."
if [ ! -f "$REPO_DIR/weights/resshift_realsrx4_s15_v1.pth" ]; then
    echo "ERROR: Teacher weights not found. Run inference.py once to download them."
    deactivate
    exit 1
fi
if [ ! -f "$REPO_DIR/weights/autoencoder_vq_f4.pth" ]; then
    echo "ERROR: Autoencoder weights not found. Run inference.py once to download them."
    deactivate
    exit 1
fi

echo "Checking dataset..."
for split in train val; do
    for domain in input target; do
        dir="$DATA_ROOT/$split/$domain"
        if [ ! -d "$dir" ]; then
            echo "ERROR: Missing dataset folder: $dir"
            echo "       Run prepare_dataset.sh first."
            deactivate
            exit 1
        fi
    done
done

echo "Training images available:"
find "$DATA_ROOT/train/input" -maxdepth 1 \( -type f -o -type l \) | wc -l

echo "Validation images available:"
find "$DATA_ROOT/val/input" -maxdepth 1 \( -type f -o -type l \) | wc -l

echo "Checking CUDA..."
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# =========================================================
# START TRAINING
# =========================================================

cd "$REPO_DIR"

echo "Starting training..."
echo "  repo    : $REPO_DIR"
echo "  config  : $CONFIG"
echo "  data    : $DATA_ROOT"
echo "  save    : $SAVE_DIR"

# Start GPU logging in background
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total \
    --format=csv -l 10 \
    > "$VSC_DATA/projects/sinsr/logs/gpu_usage_${SLURM_JOB_ID}.csv" &
GPU_LOG_PID=$!

CUDA_VISIBLE_DEVICES=0 python main_distill.py \
    --cfg_path "$CONFIG" \
    --save_dir "$SAVE_DIR"

# Stop GPU logging
kill $GPU_LOG_PID 2>/dev/null || true

deactivate

echo ""
echo "Training complete."
echo "Checkpoints : $SAVE_DIR"
echo "GPU log     : $VSC_DATA/projects/sinsr/logs/gpu_usage_${SLURM_JOB_ID}.csv"

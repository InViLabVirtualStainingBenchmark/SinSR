#!/bin/bash
#SBATCH --job-name=sinsr_mist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sinsr/logs/%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sinsr/logs/%x.%j.err

set -euo pipefail

# =========================================================
# USER SETTINGS
# =========================================================

export REPO_DIR="$VSC_DATA/projects/sinsr/code/SinSR"
export LOG_DIR="$VSC_DATA/projects/sinsr/logs"

CONTAINER="$VSC_SCRATCH/containers/sinsr_nvidia.sif"
RUN_SCRIPT="$REPO_DIR/hpc/run_sinsr_mist.sh"

# Stain to train: ER | HER2 | Ki67 | PR
# Override at submission with: sbatch --export=ALL,STAIN=HER2 train_mist.sh
: "${STAIN:=ER}"

stain_lower=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')

export STAIN
export CONFIG="$REPO_DIR/configs/virtualstaining_mist_${stain_lower}.yaml"
export SAVE_DIR="$VSC_DATA/projects/sinsr/outputs/checkpoints/mist_${stain_lower}_run"

# Set to a checkpoint .pth path to resume, e.g.:
#   export RESUME="$VSC_DATA/projects/sinsr/outputs/checkpoints/mist_er_run/2026-05-04-14-19/ckpts/model_65.pth"
# Leave empty for a fresh run.
export RESUME=""

# =========================================================
# ENVIRONMENT
# =========================================================

module purge
module load calcua/2025a

# =========================================================
# PRE-FLIGHT CHECKS
# =========================================================

echo "=== Container ==="
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi
echo "  $CONTAINER"

echo "=== Checking dataset ==="
if [ ! -f "$VSC_SCRATCH/datasets/MIST.sqsh" ]; then
    echo "ERROR: MIST SquashFS archive not found: $VSC_SCRATCH/datasets/MIST.sqsh"
    exit 1
fi
echo "  MIST.sqsh : $(du -h "$VSC_SCRATCH/datasets/MIST.sqsh" | cut -f1)"

echo "=== Checking weights ==="
if [ ! -f "$REPO_DIR/weights/resshift_realsrx4_s15_v1.pth" ] || \
   [ ! -f "$REPO_DIR/weights/autoencoder_vq_f4.pth" ]; then
    echo "ERROR: Model weights not found in $REPO_DIR/weights/"
    exit 1
fi

echo "=== Stain: $STAIN ==="
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found at $CONFIG"
    exit 1
fi

# =========================================================
# RUN
# =========================================================

mkdir -p "$VSC_SCRATCH/datasets/MIST"

srun apptainer exec --nv \
    -B "$VSC_SCRATCH/datasets/MIST.sqsh:$VSC_SCRATCH/datasets/MIST:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    "$CONTAINER" \
    bash "$RUN_SCRIPT"

echo ""
echo "MIST $STAIN training complete."

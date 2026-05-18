#!/bin/bash
#SBATCH --job-name=sinsr_infer_mist
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

# infer_mist.sh — MIST inference for SinSR. Submit after train_mist.sh completes.
# Override stain at submission: sbatch --export=ALL,STAIN=HER2 infer_mist.sh
# Automatically picks ema_best.pth (or ema_model_last.pth) from the most recent training run.

set -euo pipefail

export REPO_DIR="$VSC_DATA/projects/sinsr/code/SinSR"
export LOG_DIR="$VSC_DATA/projects/sinsr/logs"
export CKPT_BASE="$VSC_DATA/projects/sinsr/outputs/checkpoints"

# Stain to infer: ER | HER2 | Ki67 | PR
: "${STAIN:=ER}"
export STAIN

GRP_SCRATCH="/scratch/antwerpen/grp/ap_invilab_td_thesis"
stain_lower=$(echo "$STAIN" | tr '[:upper:]' '[:lower:]')
export OUT_DIR="$GRP_SCRATCH/predictions/sinsr/mist_${stain_lower}_test"

CONTAINER="$VSC_SCRATCH/containers/sinsr_nvidia.sif"
RUN_SCRIPT="$REPO_DIR/hpc/run_infer_mist.sh"

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
if [ ! -f "$REPO_DIR/weights/autoencoder_vq_f4.pth" ]; then
    echo "ERROR: autoencoder_vq_f4.pth not found in $REPO_DIR/weights/"
    exit 1
fi

echo "=== Stain: $STAIN ==="

# =========================================================
# RUN
# =========================================================

mkdir -p "$VSC_SCRATCH/datasets/MIST"
mkdir -p "$OUT_DIR"

srun apptainer exec --nv \
    -B "$VSC_SCRATCH/datasets/MIST.sqsh:$VSC_SCRATCH/datasets/MIST:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$GRP_SCRATCH:$GRP_SCRATCH" \
    "$CONTAINER" \
    bash "$RUN_SCRIPT"

echo ""
echo "MIST $STAIN inference complete."
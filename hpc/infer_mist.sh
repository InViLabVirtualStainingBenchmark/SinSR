#!/bin/bash
#SBATCH --job-name=sinsr_infer_mist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sinsr/logs/%x.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sinsr/logs/%x.%j.err

# infer_mist.sh — MIST inference for all four stains. Submit after train_mist.sh completes.
# Automatically picks the most recent training run and its highest-iteration EMA checkpoint per stain.

set -euo pipefail

export REPO_DIR="$VSC_DATA/projects/sinsr/code/SinSR"
export LOG_DIR="$VSC_DATA/projects/sinsr/logs"
export CKPT_BASE="$VSC_DATA/projects/sinsr/outputs/checkpoints"

GRP_SCRATCH="/scratch/antwerpen/grp/ap_invilab_td_thesis"
export OUT_BASE="$GRP_SCRATCH/predictions/sinsr"

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

# =========================================================
# RUN
# =========================================================

mkdir -p "$VSC_SCRATCH/datasets/MIST"

srun apptainer exec --nv \
    -B "$VSC_SCRATCH/datasets/MIST.sqsh:$VSC_SCRATCH/datasets/MIST:image-src=/" \
    -B "$VSC_DATA:$VSC_DATA" \
    -B "$GRP_SCRATCH:$GRP_SCRATCH" \
    "$CONTAINER" \
    bash "$RUN_SCRIPT"

echo ""
echo "All MIST stains inference complete."

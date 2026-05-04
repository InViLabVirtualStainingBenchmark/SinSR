#!/bin/bash
#SBATCH --job-name=sinsr_mist
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sinsr/logs/train_mist.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sinsr/logs/train_mist.%j.err

set -euo pipefail

# =========================================================
# USER SETTINGS
# =========================================================

export REPO_DIR="$VSC_DATA/projects/sinsr/code/SinSR"
export LOG_DIR="$VSC_DATA/projects/sinsr/logs"

CONTAINER="$VSC_SCRATCH/containers/sinsr_nvidia.sif"
RUN_SCRIPT="$REPO_DIR/hpc/run_sinsr_mist.sh"

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

# =========================================================
# TRAINING — ALL FOUR MIST STAINS
# =========================================================

for stain in ER HER2 Ki67 PR; do

    stain_lower=$(echo "$stain" | tr '[:upper:]' '[:lower:]')

    export STAIN="$stain"
    export CONFIG="$REPO_DIR/configs/virtualstaining_mist_${stain_lower}.yaml"
    export SAVE_DIR="$VSC_DATA/projects/sinsr/outputs/checkpoints/mist_${stain_lower}_run"

    echo ""
    echo "========================================="
    echo "  Stain: $stain"
    echo "========================================="

    if [ ! -f "$CONFIG" ]; then
        echo "ERROR: Config not found at $CONFIG"
        exit 1
    fi

    srun apptainer exec --nv \
        -B "$VSC_SCRATCH/datasets/MIST.sqsh:$VSC_SCRATCH/datasets/MIST:image-src=/" \
        -B "$VSC_DATA:$VSC_DATA" \
        "$CONTAINER" \
        bash "$RUN_SCRIPT"

done

echo ""
echo "All MIST stains complete."

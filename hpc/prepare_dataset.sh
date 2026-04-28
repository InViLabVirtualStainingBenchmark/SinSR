#!/bin/bash
#SBATCH --job-name=sinsr_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sinsr/logs/data.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sinsr/logs/data.%j.err

set -euo pipefail

# =========================================================
# PATHS
# =========================================================

DATASETS_ROOT="$VSC_SCRATCH/datasets"
# Source datasets:
#   $DATASETS_ROOT/BCI/HE/{train,test}
#   $DATASETS_ROOT/BCI/IHC/{train,test}
#   $DATASETS_ROOT/MIST/{ER,HER2,Ki67,PR}/TrainValAB/{trainA,trainB,valA,valB}

WORK_BASE="$VSC_SCRATCH/sinsr"
# Symlinked datasets will be written here.
# SinSR expects:
#   <dataset>/train/input   ← H&E images
#   <dataset>/train/target  ← IHC images
#   <dataset>/val/input
#   <dataset>/val/target

# =========================================================
# PRE-FLIGHT CHECKS
# =========================================================

echo "Checking source folders..."

for dir in \
    "$DATASETS_ROOT/BCI/HE/train" \
    "$DATASETS_ROOT/BCI/HE/test" \
    "$DATASETS_ROOT/BCI/IHC/train" \
    "$DATASETS_ROOT/BCI/IHC/test"; do
    if [ ! -d "$dir" ]; then
        echo "ERROR: Missing source folder: $dir"
        exit 1
    fi
done

for stain in ER HER2 Ki67 PR; do
    for split in trainA trainB valA valB; do
        dir="$DATASETS_ROOT/MIST/$stain/TrainValAB/$split"
        if [ ! -d "$dir" ]; then
            echo "ERROR: Missing source folder: $dir"
            exit 1
        fi
    done
done

echo "All source folders found."

# =========================================================
# BCI DATASET
# =========================================================
# Source layout:
#   HE/train  ← H&E training images
#   HE/test   ← H&E test images (used as val — no separate val split)
#   IHC/train ← IHC training images
#   IHC/test  ← IHC test images (used as val)
# =========================================================

echo ""
echo "Preparing BCI dataset..."

BCI_SRC="$DATASETS_ROOT/BCI"
BCI_DST="$WORK_BASE/bci"

rm -rf "$BCI_DST"
mkdir -p "$BCI_DST/train/input" "$BCI_DST/train/target"
mkdir -p "$BCI_DST/val/input"   "$BCI_DST/val/target"

echo "  Linking HE/train -> bci/train/input"
count=0
for f in "$BCI_SRC/HE/train"/*; do
    [ -f "$f" ] || continue
    ln -sf "$f" "$BCI_DST/train/input/"
    count=$((count + 1))
    if (( count % 1000 == 0 )); then echo "    linked $count files"; fi
done
if [ "$count" -eq 0 ]; then echo "ERROR: No files linked from HE/train"; exit 1; fi
echo "  total: $count"

echo "  Linking IHC/train -> bci/train/target"
count=0
for f in "$BCI_SRC/IHC/train"/*; do
    [ -f "$f" ] || continue
    ln -sf "$f" "$BCI_DST/train/target/"
    count=$((count + 1))
    if (( count % 1000 == 0 )); then echo "    linked $count files"; fi
done
if [ "$count" -eq 0 ]; then echo "ERROR: No files linked from IHC/train"; exit 1; fi
echo "  total: $count"

echo "  Linking HE/test -> bci/val/input"
count=0
for f in "$BCI_SRC/HE/test"/*; do
    [ -f "$f" ] || continue
    ln -sf "$f" "$BCI_DST/val/input/"
    count=$((count + 1))
    if (( count % 1000 == 0 )); then echo "    linked $count files"; fi
done
if [ "$count" -eq 0 ]; then echo "ERROR: No files linked from HE/test"; exit 1; fi
echo "  total: $count"

echo "  Linking IHC/test -> bci/val/target"
count=0
for f in "$BCI_SRC/IHC/test"/*; do
    [ -f "$f" ] || continue
    ln -sf "$f" "$BCI_DST/val/target/"
    count=$((count + 1))
    if (( count % 1000 == 0 )); then echo "    linked $count files"; fi
done
if [ "$count" -eq 0 ]; then echo "ERROR: No files linked from IHC/test"; exit 1; fi
echo "  total: $count"

echo "BCI done."

# =========================================================
# MIST DATASET
# =========================================================
# Source layout per stain (ER, HER2, Ki67, PR):
#   TrainValAB/trainA ← H&E training images
#   TrainValAB/trainB ← IHC training images
#   TrainValAB/valA   ← H&E validation images
#   TrainValAB/valB   ← IHC validation images
#
# MIST has no test split — only train and val.
# =========================================================

echo ""
echo "Preparing MIST datasets..."

for stain in ER HER2 Ki67 PR; do

    echo ""
    echo "  Stain: $stain"

    SRC="$DATASETS_ROOT/MIST/$stain/TrainValAB"
    DST="$WORK_BASE/mist_${stain}"

    rm -rf "$DST"
    mkdir -p "$DST/train/input" "$DST/train/target"
    mkdir -p "$DST/val/input"   "$DST/val/target"

    echo "    Linking trainA -> mist_$stain/train/input"
    count=0
    for f in "$SRC/trainA"/*; do
        [ -f "$f" ] || continue
        ln -sf "$f" "$DST/train/input/"
        count=$((count + 1))
        if (( count % 1000 == 0 )); then echo "      linked $count files"; fi
    done
    if [ "$count" -eq 0 ]; then echo "ERROR: No files linked from MIST/$stain/trainA"; exit 1; fi
    echo "    total: $count"

    echo "    Linking trainB -> mist_$stain/train/target"
    count=0
    for f in "$SRC/trainB"/*; do
        [ -f "$f" ] || continue
        ln -sf "$f" "$DST/train/target/"
        count=$((count + 1))
        if (( count % 1000 == 0 )); then echo "      linked $count files"; fi
    done
    if [ "$count" -eq 0 ]; then echo "ERROR: No files linked from MIST/$stain/trainB"; exit 1; fi
    echo "    total: $count"

    echo "    Linking valA -> mist_$stain/val/input"
    count=0
    for f in "$SRC/valA"/*; do
        [ -f "$f" ] || continue
        ln -sf "$f" "$DST/val/input/"
        count=$((count + 1))
        if (( count % 1000 == 0 )); then echo "      linked $count files"; fi
    done
    if [ "$count" -eq 0 ]; then echo "ERROR: No files linked from MIST/$stain/valA"; exit 1; fi
    echo "    total: $count"

    echo "    Linking valB -> mist_$stain/val/target"
    count=0
    for f in "$SRC/valB"/*; do
        [ -f "$f" ] || continue
        ln -sf "$f" "$DST/val/target/"
        count=$((count + 1))
        if (( count % 1000 == 0 )); then echo "      linked $count files"; fi
    done
    if [ "$count" -eq 0 ]; then echo "ERROR: No files linked from MIST/$stain/valB"; exit 1; fi
    echo "    total: $count"

    echo "  $stain done."

done

# =========================================================
# VALIDATE PAIR COUNTS
# =========================================================

echo ""
echo "Validating pair counts..."

all_ok=true

for split in train val; do
    n_input=$(find  "$WORK_BASE/bci/$split/input"  -maxdepth 1 \( -type f -o -type l \) | wc -l)
    n_target=$(find "$WORK_BASE/bci/$split/target" -maxdepth 1 \( -type f -o -type l \) | wc -l)
    echo "  bci/$split — input: $n_input  target: $n_target"
    if [ "$n_input" -ne "$n_target" ]; then
        echo "  ERROR: mismatch in bci/$split"
        all_ok=false
    fi
done

for stain in ER HER2 Ki67 PR; do
    for split in train val; do
        n_input=$(find  "$WORK_BASE/mist_${stain}/$split/input"  -maxdepth 1 \( -type f -o -type l \) | wc -l)
        n_target=$(find "$WORK_BASE/mist_${stain}/$split/target" -maxdepth 1 \( -type f -o -type l \) | wc -l)
        echo "  mist_$stain/$split — input: $n_input  target: $n_target"
        if [ "$n_input" -ne "$n_target" ]; then
            echo "  ERROR: mismatch in mist_$stain/$split"
            all_ok=false
        fi
    done
done

if [ "$all_ok" = false ]; then
    echo ""
    echo "Dataset preparation failed — fix mismatches above."
    exit 1
fi

echo ""
echo "All counts match."
echo "Datasets ready under: $WORK_BASE"
echo ""
echo "Done."
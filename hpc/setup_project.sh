#!/bin/bash
# setup_project.sh
# Run once manually on the login node. Do NOT sbatch.
# Usage: bash setup_project.sh

set -euo pipefail

# Base directory (change if needed)
BASE_DIR="$VSC_DATA/projects/sinsr"

echo "Creating project structure at: $BASE_DIR"

# Main folders
mkdir -p "$BASE_DIR"/{code,logs,outputs,jobs}
# code      → cloned repos
# logs      → slurm outputs
# outputs   → model outputs
# jobs      → slurm scripts

# Output folders
mkdir -p "$BASE_DIR"/outputs/{checkpoints,results}
# checkpoints → saved model weights and training samples
# results     → inference outputs

# Ensure venv folder exists
mkdir -p "$VSC_DATA/venvs"

echo "Done. Next: bash clone_repo.sh"

#!/bin/bash
# clone_repo.sh
# Run once manually on the login node. Do NOT sbatch.
# Usage: bash clone_repo.sh

set -euo pipefail

# Base directory
BASE_DIR="$VSC_DATA/projects/sinsr"
REPO_DIR="$BASE_DIR/code/SinSR"

echo "Cloning SinSR into: $REPO_DIR"

# Skip clone if repo already exists
if [ -d "$REPO_DIR/.git" ]; then
    echo "Repo already exists — skipping clone."
    echo "To re-clone: rm -rf $REPO_DIR and rerun this script."
else
    cd "$BASE_DIR/code"
    git clone https://github.com/InViLabVirtualStainingBenchmark/SinSR.git
fi

echo ""
echo "Done. Next: sbatch install_deps.sh"

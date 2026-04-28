#!/bin/bash
#SBATCH --job-name=sinsr_install
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH -A ap_invilab_td_thesis
#SBATCH -p ampere_gpu
#SBATCH --gres=gpu:1
#SBATCH -o /data/antwerpen/212/vsc21211/projects/sinsr/logs/install.%j.out
#SBATCH -e /data/antwerpen/212/vsc21211/projects/sinsr/logs/install.%j.err

set -euo pipefail

# =========================================================
# CONFIG
# =========================================================

BASE_DIR="$VSC_DATA/projects/sinsr"
VENV_DIR="$VSC_DATA/venvs/venv_sinsr"
REPO_DIR="$BASE_DIR/code/SinSR"

# =========================================================
# MODULES
# =========================================================

module purge
module load calcua/2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1
module load OpenCV/4.8.1-foss-2023a-contrib

# Check python version
echo "Python used:"
which python
python -V

# =========================================================
# RECREATE VENV (CLEAN)
# =========================================================

rm -rf "$VENV_DIR"

python -m venv "$VENV_DIR" --system-site-packages

source "$VENV_DIR/bin/activate"

# Verify correct python
echo "Active python:"
which python
python -V

# =========================================================
# INSTALL ONLY EXTRA PACKAGES
# =========================================================

python -m pip install --upgrade pip

python -m pip install \
    einops \
    omegaconf \
    lpips \
    basicsr \
    loguru \
    pyiqa \
    scikit-image \
    tqdm \
    --no-cache-dir \
    --no-build-isolation

# =========================================================
# SANITY CHECKS
# =========================================================

python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import einops; print('einops OK')"
python -c "import omegaconf; print('omegaconf OK')"
python -c "import pyiqa; print('pyiqa OK')"
python -c "import lpips; print('lpips OK')"
python -c "import basicsr; print('basicsr OK')"
python -c "import loguru; print('loguru OK')"
python -c "import cv2; print('cv2:', cv2.__version__)"
python -c "import skimage; print('skimage OK')"
python -c "import tqdm; print('tqdm OK')"

deactivate

echo "Done. Next: sbatch prepare_dataset.sh"

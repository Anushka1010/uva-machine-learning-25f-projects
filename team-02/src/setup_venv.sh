#!/usr/bin/env bash

# Bootstrap the local Python virtual environment and install dependencies used in this repo:
# - CUDA-enabled PyTorch stack
# - Ultralytics YOLO CLI
# - Computer vision / ML utilities for mask handling and RMBG support

set -euo pipefail
IFS=$'\n\t'

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip

# Install CUDA 12.4 wheels; override TORCH_INDEX to switch to CPU or another CUDA version if needed.
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
python -m pip install \
  --extra-index-url "${TORCH_INDEX}" \
  "torch==2.6.0+cu124" \
  "torchvision==0.21.0+cu124" \
  "torchaudio==2.6.0+cu124"

python -m pip install \
  "ultralytics==8.3.236" \
  opencv-python-headless \
  numpy \
  timm \
  huggingface_hub \
  onnxruntime \
  transformers \
  kornia

echo "Virtual environment ready in ${VENV_DIR}"

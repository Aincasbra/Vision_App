#!/bin/bash
set -euo pipefail

# Script lanzador de Calippo Jetson (comentado)
# - Activa el entorno virtual (.venv)
# - Exporta variables de entorno necesarias (CUDA, OpenCV, PyTorch)
# - Ejecuta la aplicación principal (PruebaAravis.py)

# 1) Ir al directorio del proyecto
cd /home/nvidia/Desktop/Calippo_jetson

# 2) Activar entorno virtual de Python
VENV="/home/nvidia/Desktop/Calippo_jetson/gentl/.venv"
# 'source' carga las rutas de binarios y site-packages del venv
source "$VENV/bin/activate"

# 3) Configurar CUDA y librerías compartidas
# - CUDA_HOME apunta al toolkit
# - PATH incluye binarios de CUDA
# - LD_LIBRARY_PATH añade lib64 de CUDA y las libs empaquetadas con OpenCV del venv
export CUDA_HOME="/usr/local/cuda"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$VENV/lib/python3.8/site-packages/cv2/../../lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

# 4) Asegurar que Python consulte el site-packages del venv
export PYTHONPATH="$VENV/lib/python3.8/site-packages:${PYTHONPATH:-}"

# 5) Optimizar ejecución de PyTorch/NumPy/BLAS en Jetson (evitar sobre-subscription)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TORCH_CPP_LOG_LEVEL="ERROR"
# export CUDA_VISIBLE_DEVICES=0  # opcional si quieres fijar GPU

# 6) Plugins de Qt para OpenCV (solo si se usa UI con cv2)
export QT_QPA_PLATFORM_PLUGIN_PATH="$VENV/lib/python3.8/site-packages/cv2/qt/plugins"
export QT_QPA_FONTDIR="$VENV/lib/python3.8/site-packages/cv2/qt/fonts"

# 7) Ejecutar la aplicación con el Python del venv
# 'exec' reemplaza el proceso del shell por el de Python (mejor para systemd)
exec "$VENV/bin/python" /home/nvidia/Desktop/Calippo_jetson/gentl/PruebaAravis.py



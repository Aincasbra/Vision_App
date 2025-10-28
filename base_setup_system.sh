#!/bin/bash
set -euo pipefail

# Base setup para Jetson Orin con JetPack 5.1.1 (R35.3.x)
# Instala CUDA toolkit meta, cuDNN 8.6, TensorRT 8.5.2, OpenCV Python, dev tools

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info(){ echo -e "${BLUE}[INFO]${NC} $1"; }
ok(){ echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err(){ echo -e "${RED}[ERR]${NC} $1"; }

if [[ $EUID -ne 0 ]]; then
  err "Ejecuta como root: sudo $0"; exit 1; fi

info "Actualizando sistema..."
apt update && apt install -y software-properties-common

info "Instalando paquetes base..."
apt install -y build-essential cmake git wget curl unzip pkg-config \
  python3 python3-pip python3-venv python3-dev \
  libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  libgtk-3-dev libcanberra-gtk3-module libgpiod2 gpiod

ok "Paquetes base instalados"

info "Instalando CUDA Toolkit 11.4 meta (si falta)..."
if ! dpkg -l | grep -q '^ii\s\+cuda-toolkit-11-4\s'; then
  apt install -y cuda-toolkit-11-4 || true
else
  ok "CUDA toolkit ya instalado"
fi

info "Instalando cuDNN 8.6 (si falta)..."
if ! dpkg -l | grep -q '^ii\s\+libcudnn8\s'; then
  apt install -y libcudnn8 libcudnn8-dev libcudnn8-samples || true
else
  ok "cuDNN ya instalado"
fi

info "Instalando TensorRT 8.5 (si falta)..."
if ! dpkg -l | grep -q '^ii\s\+tensorrt\s'; then
  apt install -y tensorrt tensorrt-dev libnvinfer8 libnvinfer-dev libnvinfer-plugin8 libnvinfer-plugin-dev python3-libnvinfer-dev || true
else
  ok "TensorRT ya instalado"
fi

info "OpenCV Python (bindings sistema)"
apt install -y python3-opencv || true

info "Configurando variables de entorno CUDA en /etc/ld.so.conf.d/cuda.conf y ~/.bashrc"
if [[ ! -f /etc/ld.so.conf.d/cuda.conf ]]; then
  echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
  ldconfig
fi

BASHRC="/home/nvidia/.bashrc"
if ! grep -q 'CUDA_HOME=/usr/local/cuda' "$BASHRC"; then
  cat >> "$BASHRC" <<'EOT'
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# Optimizaciones PyTorch para Jetson
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
EOT
fi

ok "Entorno CUDA configurado"

info "Habilitando logrotate.timer (persistente)"
systemctl enable --now logrotate.timer || true
ok "Base del sistema lista"

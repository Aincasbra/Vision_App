#!/bin/bash
set -euo pipefail
# Instala PyTorch 2.0.0+nv23.05 y torchvision compatible para JetPack 5.1.1 (CUDA 11.4)

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info(){ echo -e "${BLUE}[INFO]${NC} $1"; }
ok(){ echo -e "${GREEN}[OK]${NC} $1"; }
err(){ echo -e "${RED}[ERR]${NC} $1"; }

if [[ $EUID -eq 0 ]]; then
  err "Ejecuta como usuario nvidia con sudo sólo cuando se pida"; exit 1; fi

VENV_DIR="/home/nvidia/Desktop/Calippo_jetson/gentl/.venv"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

if [[ ! -x "$PY" ]]; then
  info "Creando venv..."
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
$PIP install --upgrade pip

# Rutas de wheels locales (ajusta si las ruedas están en otro sitio)
TORCH_WHL="/home/nvidia/tmp_jp/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl"
TV_SRC="/home/nvidia/tvsrc"  # si tienes torchvision local editable

if [[ -f "$TORCH_WHL" ]]; then
  info "Instalando Torch desde wheel local..."
  $PIP install "$TORCH_WHL"
else
  info "No se encontró wheel local. Intentando instalar desde índice NVIDIA (puede fallar)."
  $PIP install --extra-index-url https://pypi.ngc.nvidia.com torch==2.0.0+nv23.05 || true
fi

# Torchvision: preferir wheel/local src compatible con JP 5.1.1
if [[ -d "$TV_SRC" ]]; then
  info "Instalando torchvision desde fuente local..."
  $PIP install -e "$TV_SRC" || true
else
  info "Instalando torchvision precompilado compatible (puede no estar en PyPI)"
  $PIP install torchvision==0.15.1 --no-build-isolation || true
fi

# Verificación rápida
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
PY

ok "PyTorch/torchvision instalados (si no hubo errores arriba)"

#!/bin/bash
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
msg(){ echo -e "${BLUE}[INFO]${NC} $1"; }
ok(){ echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err(){ echo -e "${RED}[ERR]${NC} $1"; }

require_sudo(){ if [[ $(id -u) -ne 0 ]]; then err "Ejecuta con sudo: sudo $0"; exit 1; fi }

require_sudo

msg "Actualizando índices APT..."
apt update

msg "Instalando utilidades básicas..."
apt install -y build-essential cmake git pkg-config curl wget unzip python3-venv python3-pip libjpeg-dev libpng-dev

msg "Verificando JetPack/CUDA/cuDNN/TensorRT..."

if command -v nvcc >/dev/null 2>&1; then nvcc --version || true; ok "CUDA detectado"; else warn "CUDA no detectado (nvcc). En JetPack debería venir preinstalado."; fi

if ldconfig -p | grep -qi libcudnn; then ok "cuDNN detectado"; else warn "cuDNN no detectado en ldconfig. En JetPack debería venir preinstalado."; fi

if ldconfig -p | grep -qi nvinfer; then ok "TensorRT detectado"; else warn "TensorRT no detectado. En JetPack debería venir preinstalado."; fi

msg "Ajustando variables de entorno del sistema (/etc/profile.d/calippo_env.sh)"
cat > /etc/profile.d/calippo_env.sh << 'EOT'
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
# Rutas comunes Jetson para TensorRT/cuDNN
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH
EOT
chmod 644 /etc/profile.d/calippo_env.sh
ok "Variables de entorno persistentes configuradas"

msg "Habilitando logrotate.timer persistente"
systemctl enable --now logrotate.timer || true
ok "logrotate.timer habilitado"

msg "Creando grupos/permisos de video si fuera necesario"
groupadd -f video
usermod -aG video nvidia || true
ok "Usuario nvidia añadido a video"

ok "Setup de runtime JetPack verificado. Si faltan CUDA/cuDNN/TRT, instala JetPack 5.1.1 completo."

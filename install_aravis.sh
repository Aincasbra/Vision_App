#!/bin/bash
set -euo pipefail
# Instala Aravis (preferible 0.8) para usar con gi.require_version('Aravis','0.8')

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info(){ echo -e "${BLUE}[INFO]${NC} $1"; }
ok(){ echo -e "${GREEN}[OK]${NC} $1"; }
err(){ echo -e "${RED}[ERR]${NC} $1"; }

if [[ $EUID -ne 0 ]]; then
  err "Ejecuta como root: sudo $0"; exit 1; fi

info "Intentando instalar Aravis desde paquetes..."
apt update
apt install -y libaravis-0.8-0 libaravis-0.8-dev gir1.2-aravis-0.8 || true

python3 - <<'PY'
import sys
try:
    import gi
    gi.require_version('Aravis','0.8')
    from gi.repository import Aravis
    print('OK: Aravis 0.8 disponible')
except Exception as e:
    print('NOOK', e)
    sys.exit(2)
PY
status=$?

if [[ $status -ne 0 ]]; then
  info "Compilando Aravis 0.8 desde fuente (fallback)..."
  apt install -y meson ninja-build gobject-introspection libglib2.0-dev libgirepository1.0-dev libxml2-utils \
    libusb-1.0-0-dev libudev-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  WORK=/tmp/build_aravis
  rm -rf "$WORK" && mkdir -p "$WORK"
  cd "$WORK"
  # versión estable 0.8.x
  git clone --depth=1 --branch=0.8 https://github.com/AravisProject/aravis.git
  cd aravis
  meson setup build --prefix=/usr -Dviewer=disabled -Dgst-plugin=disabled
  ninja -C build
  ninja -C build install
  ldconfig
  ok "Aravis instalado desde fuente"
fi

ok "Instalación de Aravis finalizada"

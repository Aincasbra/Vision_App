#!/bin/bash
# -------------------------------------------------------------
# install_aravis.sh
# -------------------------------------------------------------
# Instala Aravis 0.6 (paquetes o compilación como fallback).
# Soporte GenICam/Aravis en la aplicación (gi.require_version).
# Dónde se usa: preparación del entorno de cámaras GenICam.
# -------------------------------------------------------------
set -euo pipefail
# Instala Aravis 0.6 (preferible) para usar con gi.require_version('Aravis','0.6')

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info(){ echo -e "${BLUE}[INFO]${NC} $1"; }
ok(){ echo -e "${GREEN}[OK]${NC} $1"; }
err(){ echo -e "${RED}[ERR]${NC} $1"; }

if [[ $EUID -ne 0 ]]; then
  err "Ejecuta como root: sudo $0"; exit 1; fi

info "Intentando instalar Aravis 0.6 desde paquetes..."
apt update
apt install -y gir1.2-aravis-0.6 libaravis-0.6-0 aravis-tools || true

python3 - <<'PY'
import sys
try:
    import gi
    gi.require_version('Aravis','0.6')
    from gi.repository import Aravis
    print('OK: Aravis 0.6 disponible')
except Exception as e:
    print('NOOK', e)
    sys.exit(2)
PY
status=$?

if [[ $status -ne 0 ]]; then
  info "Compilando Aravis 0.6 desde fuente (fallback)..."
  apt install -y meson ninja-build gobject-introspection libglib2.0-dev libgirepository1.0-dev libxml2-utils \
    libusb-1.0-0-dev libudev-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
  WORK=/tmp/build_aravis
  rm -rf "$WORK" && mkdir -p "$WORK"
  cd "$WORK"
  # versión estable 0.6.x
  git clone --depth=1 --branch=0.6 https://github.com/AravisProject/aravis.git
  cd aravis
  meson setup build --prefix=/usr -Dviewer=disabled -Dgst-plugin=disabled
  ninja -C build
  ninja -C build install
  ldconfig
  ok "Aravis instalado desde fuente"
fi

ok "Instalación de Aravis finalizada"

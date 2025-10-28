#!/bin/bash
# Script maestro para replicar sistema completo
set -e

echo '🚀 Iniciando replicación del sistema...'

# Ejecutar scripts en orden
chmod +x *.sh
./setup_system.sh
./install_packages.sh
./setup_venv.sh
./configure_environment.sh

# Instalar paquetes Python
cd /home/nvidia/Desktop/Calippo_jetson/gentl
source .venv/bin/activate
pip install -r requirements.txt

echo '✅ Replicación completada'

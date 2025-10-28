#!/bin/bash
# Script para crear y configurar entorno virtual
set -e

echo '🐍 Configurando entorno virtual...'

# Crear directorio del proyecto
mkdir -p /home/nvidia/Desktop/Calippo_jetson/gentl
cd /home/nvidia/Desktop/Calippo_jetson/gentl

# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Actualizar pip
pip install --upgrade pip setuptools wheel

echo '✅ Entorno virtual creado'

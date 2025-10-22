#!/bin/bash
# Script para instalar paquetes necesarios
set -e

echo '📦 Instalando paquetes...'

# Paquetes base
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y build-essential cmake git wget curl unzip
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgtk2.0-dev libcanberra-gtk-module libcanberra-gtk3-module

# OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv

# GPIOD
sudo apt install -y gpiod

echo '✅ Paquetes instalados'

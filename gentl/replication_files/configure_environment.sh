#!/bin/bash
# Script para configurar variables de entorno
set -e

echo '🔧 Configurando variables de entorno...'

# Configurar CUDA
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Configurar Python
echo 'export PYTHONPATH=$VIRTUAL_ENV/lib/python3.8/site-packages:$PYTHONPATH' >> ~/.bashrc

# Recargar configuración
source ~/.bashrc

echo '✅ Variables de entorno configuradas'

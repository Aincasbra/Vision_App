#!/bin/bash
# Script para ejecutar Vision App en modo debug (con UI)
# Uso: ./run_debug.sh

set -e

PROJECT_DIR="/home/nvidia/Desktop/Vision_App"
VENV_DIR="$PROJECT_DIR/vision_app/.venv"

# Verificar que estamos en el directorio correcto
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "❌ Error: Ejecuta este script desde $PROJECT_DIR"
    exit 1
fi

# Detener servicio systemd si está corriendo
if systemctl is-active --quiet vision-app.service 2>/dev/null; then
    echo "🛑 Deteniendo servicio systemd..."
    sudo systemctl stop vision-app.service
    sleep 1
fi

# Activar entorno virtual
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "❌ Error: Entorno virtual no encontrado en $VENV_DIR"
    exit 1
fi

echo "🔧 Activando entorno virtual..."
source "$VENV_DIR/bin/activate"

# Configurar PYTHONPATH (solo si no está en el venv)
export PYTHONPATH="$PROJECT_DIR/vision_app:/usr/lib/python3/dist-packages:$PYTHONPATH"

# Variables de entorno para modo debug (opcionales, tienen defaults)
# HEADLESS=0 y AUTO_RUN=0 son los defaults, pero las ponemos explícitamente
export HEADLESS=0
export AUTO_RUN=0

echo "🚀 Iniciando Vision App en modo debug..."
echo "   - UI habilitada (HEADLESS=0)"
echo "   - Auto-RUN deshabilitado (AUTO_RUN=0)"
echo "   - Presiona Ctrl+C para detener"
echo ""

# Ejecutar aplicación
cd "$PROJECT_DIR"
python main.py


#!/bin/bash
# Script para configurar Jetson Orin con JetPack 5.1.1
set -e

echo '🚀 Configurando Jetson Orin...'

# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Configurar Jetson Clocks
sudo jetson_clocks

# Configurar modo de potencia
sudo nvpmodel -m 0

# Configurar fan
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'

echo '✅ Sistema configurado'

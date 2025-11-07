#!/usr/bin/env bash

# Script para configurar permisos GPIO UNA VEZ
# Después podrás usar los GPIOs sin sudo

echo "🔧 CONFIGURANDO PERMISOS GPIO"
echo "Esto se hace UNA VEZ y después no necesitas sudo"
echo ""

# Verificar root
if [[ $EUID -ne 0 ]]; then
    echo "❌ Ejecuta con sudo: sudo ./setup_gpio_permissions.sh"
    exit 1
fi

echo "1. Agregando usuario al grupo gpio..."
usermod -a -G gpio nvidia

echo "2. Configurando permisos del dispositivo GPIO..."
chmod 666 /dev/gpiochip0

echo "3. Creando regla udev para persistencia..."
cat > /etc/udev/rules.d/99-gpio.rules << 'EOF'
# Regla para permitir acceso a GPIO sin sudo
SUBSYSTEM=="gpio", GROUP="gpio", MODE="0666"
KERNEL=="gpiochip*", GROUP="gpio", MODE="0666"
EOF

echo "4. Recargando reglas udev..."
udevadm control --reload-rules
udevadm trigger

echo ""
echo "✅ PERMISOS CONFIGURADOS"
echo ""
echo "Para aplicar los cambios:"
echo "1. Reinicia el sistema, O"
echo "2. Ejecuta: newgrp gpio"
echo ""
echo "Después podrás usar: ./test_do_safe.sh (sin sudo)"

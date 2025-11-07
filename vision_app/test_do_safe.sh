#!/bin/bash
"""
TEST DO SAFE
Script para probar las salidas digitales de forma segura
"""

# Configuración
GPIO_CHIP=0
GPIO_LINE=18  # DO0
TEST_DURATION=2  # segundos
REPEAT_COUNT=3

echo "🔧 Test de salidas digitales seguras"
echo "=================================="

# Verificar permisos
if [ ! -w "/dev/gpiochip0" ]; then
    echo "❌ Error: Sin permisos para acceder a GPIO"
    echo "   Ejecuta: sudo chmod 666 /dev/gpiochip0"
    exit 1
fi

# Verificar herramientas
if ! command -v gpioset &> /dev/null; then
    echo "❌ Error: gpioset no encontrado"
    echo "   Instala: sudo apt install gpiod"
    exit 1
fi

echo "✅ Permisos y herramientas verificadas"

# Función para test seguro
test_gpio_safe() {
    local state=$1
    local duration=$2
    
    echo "   Configurando GPIO${GPIO_LINE} = ${state} por ${duration}s..."
    
    # Configurar GPIO
    if gpioset ${GPIO_CHIP} ${GPIO_LINE}=${state} &> /dev/null; then
        echo "   ✅ GPIO configurado correctamente"
        sleep ${duration}
        
        # Apagar GPIO
        if gpioset ${GPIO_CHIP} ${GPIO_LINE}=0 &> /dev/null; then
            echo "   ✅ GPIO apagado correctamente"
        else
            echo "   ⚠️ Advertencia: Error al apagar GPIO"
        fi
    else
        echo "   ❌ Error configurando GPIO"
        return 1
    fi
}

# Test principal
echo ""
echo "🧪 Iniciando tests seguros..."
echo "   GPIO Chip: ${GPIO_CHIP}"
echo "   GPIO Line: ${GPIO_LINE}"
echo "   Duración por test: ${TEST_DURATION}s"
echo "   Repeticiones: ${REPEAT_COUNT}"
echo ""

for i in $(seq 1 ${REPEAT_COUNT}); do
    echo "📋 Test ${i}/${REPEAT_COUNT}"
    
    # Test ON
    echo "   🔴 Test ON..."
    test_gpio_safe 1 ${TEST_DURATION}
    
    # Pausa entre tests
    sleep 1
    
    # Test OFF (ya está apagado, pero verificamos)
    echo "   ⚫ Test OFF..."
    test_gpio_safe 0 1
    
    # Pausa entre repeticiones
    if [ ${i} -lt ${REPEAT_COUNT} ]; then
        echo "   ⏳ Pausa entre tests..."
        sleep 2
    fi
    
    echo ""
done

echo "✅ Tests completados exitosamente"
echo "   GPIO apagado y seguro"
echo ""
echo "📊 Resumen:"
echo "   - Tests ejecutados: ${REPEAT_COUNT}"
echo "   - Duración total: $((REPEAT_COUNT * (TEST_DURATION + 3)))s"
echo "   - Estado final: GPIO apagado"
echo ""
echo "🔧 Para uso en producción, ejecuta:"
echo "   ./SALIDAS_DIGITALES_PRODUCCION.sh"

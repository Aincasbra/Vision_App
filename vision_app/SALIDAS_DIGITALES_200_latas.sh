#!/usr/bin/env bash
set -euo pipefail

# SIMULADOR SALIDAS DIGITALES (configuración actual)
# - DO1 (clasificación, pin 52): ventana estable de 20ms por lata (1 = buena, 0 = mala)
# - DO0 (DATA_READY, pin 51): pulso de 20ms, se dispara 1ms después de iniciar DO1
# - DO2 (VISION_OK, pin 53): alto permanente durante toda la simulación
# - Ritmo: espera fija de 200ms entre latas (además de la ventana DO1)

echo "🏭 SIMULADOR SALIDAS DIGITALES - CONFIG ACTUAL"
echo "Ventana DO1: 20ms por lata"
echo "DO0 = DATA_READY: pulso 20ms, 1ms después de DO1"
echo "DO2 = VISION_OK: alto permanente"
echo "Espera entre latas: 200ms (pitch total ≈ 220ms)"
#echo "Espera entre latas: 280ms (pitch total ≈ 300ms)"
echo ""

# Verificar permisos GPIO
if [[ ! -r "/dev/gpiochip0" ]]; then
    echo "❌ Permisos GPIO no configurados. Ejecuta: sudo ./setup_gpio_permissions.sh"
    exit 1
fi

# Configuración
CHIP="gpiochip0"
DO0=51  # DATA_READY (pulso presencia de nueva lata)
DO1=52  # Clasificación buena/mala (ventana de decisión)
DO2=53  # VISION_OK (estado general de visión)

TOTAL_LATAS=200         # Total de latas a simular
TIEMPO_POR_LATA_MS=20  # Ventana DO1 por lata (20ms)
PULSO_DO0_MS=20         # Ancho del pulso de DO0 (20ms)
DELAY_DO0_MS=1          # Retardo de DO0 desde inicio DO1 (1ms)
LATAS_MALAS=20          # Total de latas malas (200/10 = 20)
LATA_MALA_CADA=10       # Frecuencia: una lata mala cada 10

# Variables de control
latas_malas_enviadas=0

# Función para activar DO0 (DATA_READY) 1ms después de DO1
activate_do0() {
    sleep 0.001  # 1ms desde el inicio de DO1
    echo "    🔴 I08 (DO0): ACTIVANDO - Nueva lata detectada"
    gpioset -m signal "$CHIP" "$DO0=1" & local pid=$!
    sleep 0.02   # 20ms
    kill "$pid" >/dev/null 2>&1 || true
    gpioset -m exit "$CHIP" "$DO0=0"
    echo "    ⚫ I08 (DO0): DESACTIVANDO - Pulso completado"
}

# Función para activar DO1 (clasificación) durante 20ms por lata
activate_do1() {
    local calidad="$1"
    if [[ "$calidad" == "1" ]]; then
        echo "  🟢 I09 (DO1): ACTIVANDO - Calidad BUENA por 20ms"
    else
        echo "  🔴 I09 (DO1): ACTIVANDO - Calidad MALA por 20ms"
    fi
    gpioset -m signal "$CHIP" "$DO1=$calidad" & local pid=$!
    sleep 0.02    # 20ms (pitch temporal) - MANTIENE EL VALOR
    kill "$pid" >/dev/null 2>&1 || true
    gpioset -m exit "$CHIP" "$DO1=0"
    echo "  ⚫ I09 (DO1): DESACTIVANDO - Siguiente lata"
}

# Función de limpieza
cleanup() {
    echo -e "\n🧹 Parando simulador..."
    gpioset -m exit "$CHIP" "$DO0=0" >/dev/null 2>&1 || true
    gpioset -m exit "$CHIP" "$DO1=0" >/dev/null 2>&1 || true
    gpioset -m exit "$CHIP" "$DO2=0" >/dev/null 2>&1 || true
    echo "✅ Simulador detenido"
}
trap cleanup EXIT

# Activar DO2 (VISION_OK) al principio
echo "🔴 Activando DO2 (VISION_OK) - al inicio del programa..."
gpioset -m signal "$CHIP" "$DO2=1" & 
do2_pid=$!

echo ""
echo "=== INICIANDO SIMULACIÓN ==="
echo "Latas totales: $TOTAL_LATAS"
echo "Ventana DO1: ${TIEMPO_POR_LATA_MS}ms"
echo "Pulso DO0: ${PULSO_DO0_MS}ms (${DELAY_DO0_MS}ms después de inicio DO1)"
echo "Espera entre latas: 200ms (pitch total ≈ 220ms)"
#echo "Espera entre latas: 280ms (pitch total ≈ 300ms)"
echo "Latas malas: $LATAS_MALAS (cada $LATA_MALA_CADA latas)"
echo "Presiona Ctrl+C para detener"
echo ""

# Simulación de 200 latas
for ((lata=1; lata<=TOTAL_LATAS; lata++)); do
    # Determinar si esta lata es mala
    if (( (lata % LATA_MALA_CADA) == 0 )); then
        calidad=0  # Mala
        latas_malas_enviadas=$((latas_malas_enviadas + 1))
        echo "Lata $lata/$TOTAL_LATAS: MALA calidad (mala #$latas_malas_enviadas/$LATAS_MALAS)"
    else
        calidad=1  # Buena
        echo "Lata $lata/$TOTAL_LATAS: BUENA calidad"
    fi
    
    # Activar DO1 (clasificación) por 20ms (ventana de calidad)
    activate_do1 "$calidad" &
    do1_pid=$!
    
    # Activar DO0 (DATA_READY) 1ms después de DO1 - CADA LATA
    activate_do0 &
    do0_pid=$!
    
    # Esperar a que terminen ambos
    wait $do1_pid $do0_pid
    
    echo "  → Lata $lata procesada (ciclo completado)"
    
    # Pausa hasta la siguiente lata (excepto la última)
    if ((lata < TOTAL_LATAS)); then
        echo "  → Esperando siguiente lata..."
        sleep 0.2  # 200ms de espera fija entre latas (20ms DO1 + 200ms = 220ms)
        #sleep 0.28  # 280ms de espera fija entre latas (20ms DO1 + 280ms = 300ms)
    fi
done

# Desactivar DO2 al final del programa
echo "🔴 Desactivando DO2 (VISION_OK) - fin del programa..."
kill "$do2_pid" >/dev/null 2>&1 || true
gpioset -m exit "$CHIP" "$DO2=0"

echo ""
echo "✅ SIMULACIÓN COMPLETADA"
echo "Latas procesadas: $TOTAL_LATAS"
echo "Latas malas: $latas_malas_enviadas (buenas: $((TOTAL_LATAS - latas_malas_enviadas)))"
echo "Tiempo total (incluye espera 200ms): $((((TOTAL_LATAS * TIEMPO_POR_LATA_MS) + ((TOTAL_LATAS - 1) * 200)) / 1000)) segundos"
#echo "Tiempo total (incluye espera 280ms): $((((TOTAL_LATAS * TIEMPO_POR_LATA_MS) + ((TOTAL_LATAS - 1) * 280)) / 1000)) segundos"
echo "Todas las salidas en 0"

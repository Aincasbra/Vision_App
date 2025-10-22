#!/usr/bin/env bash
set -euo pipefail

# SIMULADOR DE PRODUCCIÓN - 200 latas/minuto
# DO0 = Dato nuevo disponible (se activa 10ms después de DO1)
# DO1 = Calidad (buena/mala) - cada 50 pulsos una mala
# DO2 = Visión OK (continuamente activada)

echo "🏭 SIMULADOR DE PRODUCCIÓN - 200 latas/minuto"
echo "DO0 = Dato nuevo disponible"
echo "DO1 = Calidad (buena/mala)"  
echo "DO2 = Visión OK (continuo)"
echo ""

# Verificar permisos GPIO
if [[ ! -r "/dev/gpiochip0" ]]; then
    echo "❌ Permisos GPIO no configurados. Ejecuta: sudo ./setup_gpio_permissions.sh"
    exit 1
fi

# Configuración
CHIP="gpiochip0"
DO0=51  # Dato nuevo disponible
DO1=52  # Calidad (buena/mala)
DO2=53  # Visión OK (continuo)

TOTAL_PULSOS=200        # 200 latas/minuto
PULSOS_MALOS=4          # Cada 50 pulsos una mala (200/50 = 4)
PULSO_MALO_CADA=50      # Cada 50 pulsos
DELAY_DO0_MS=2          # DO0 se activa 2ms después de DO1
PULSE_DURATION_MS=20    # Duración de cada pulso (20ms)
CYCLE_TIME_MS=300       # Tiempo total por ciclo (300ms = 200 pulsos/minuto)

# Variables de control
pulsos_malos_enviados=0
pulso_actual=0

# Función para activar DO0 (dato nuevo) 10ms después de DO1
activate_do0() {
    sleep 0.002  # 2ms
    gpioset -m signal "$CHIP" "$DO0=1" & local pid=$!
    sleep 0.02  # 20ms
    kill "$pid" >/dev/null 2>&1 || true
    gpioset -m exit "$CHIP" "$DO0=0"
}

# Función para activar DO1 (calidad) con duración de 50ms
activate_do1() {
    local calidad="$1"
    gpioset -m signal "$CHIP" "$DO1=$calidad" & local pid=$!
    sleep 0.02  # 20ms
    kill "$pid" >/dev/null 2>&1 || true
    gpioset -m exit "$CHIP" "$DO1=0"
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

# Activar DO2 (visión OK) continuamente
echo "🔴 Activando DO2 (Visión OK) - continuo..."
gpioset -m signal "$CHIP" "$DO2=1" & 
do2_pid=$!

echo ""
echo "=== INICIANDO SIMULACIÓN ==="
echo "Pulsos totales: $TOTAL_PULSOS"
echo "Pulsos malos: $PULSOS_MALOS (cada $PULSO_MALO_CADA pulsos)"
echo "Tiempo por pulso: ${CYCLE_TIME_MS}ms"
echo "Presiona Ctrl+C para detener"
echo ""

# Simulación de 200 pulsos
for ((pulso=1; pulso<=TOTAL_PULSOS; pulso++)); do
    pulso_actual=$pulso
    
    # Determinar si este pulso es malo
    if (( (pulso % PULSO_MALO_CADA) == 0 )); then
        calidad=0  # Mala
        pulsos_malos_enviados=$((pulsos_malos_enviados + 1))
        echo "Pulso $pulso/$TOTAL_PULSOS: MALA calidad (malo #$pulsos_malos_enviados/$PULSOS_MALOS)"
    else
        calidad=1  # Buena
        echo "Pulso $pulso/$TOTAL_PULSOS: BUENA calidad"
    fi
    
    # Activar DO1 (calidad) y DO0 (dato nuevo) en paralelo
    # DO0 se activa en TODOS los pulsos (cada lata es dato nuevo)
    activate_do1 "$calidad" &
    do1_pid=$!
    activate_do0 &
    do0_pid=$!
    
    # Esperar a que terminen ambos
    wait $do1_pid $do0_pid
    
    # Pausa entre pulsos (excepto el último)
    if ((pulso < TOTAL_PULSOS)); then
        sleep 0.3  # 300ms = 200 pulsos/minuto
    fi
done

# Detener DO2
kill "$do2_pid" >/dev/null 2>&1 || true
gpioset -m exit "$CHIP" "$DO2=0"

echo ""
echo "✅ SIMULACIÓN COMPLETADA"
echo "Pulsos enviados: $TOTAL_PULSOS"
echo "Pulsos malos enviados: $pulsos_malos_enviados"
echo "Todas las salidas en 0"

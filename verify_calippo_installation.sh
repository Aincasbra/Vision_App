#!/bin/bash

# =============================================================================
# SCRIPT DE VERIFICACIÓN POST-INSTALACIÓN CALIPPO
# =============================================================================
# Este script verifica que Calippo esté funcionando correctamente después
# de la instalación en el equipo de fábrica
# =============================================================================

set -euo pipefail

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Función para verificar servicio systemd
check_systemd_service() {
    print_status "Verificando servicio systemd..."
    
    # Verificar que el servicio existe
    if systemctl list-unit-files | grep -q "calippo.service"; then
        print_success "✓ Servicio calippo.service existe"
    else
        print_error "✗ Servicio calippo.service no encontrado"
        return 1
    fi
    
    # Verificar que está habilitado
    if systemctl is-enabled calippo.service >/dev/null 2>&1; then
        print_success "✓ Servicio habilitado para autoarranque"
    else
        print_error "✗ Servicio no habilitado para autoarranque"
        return 1
    fi
    
    # Verificar estado actual
    local status=$(systemctl is-active calippo.service 2>/dev/null || echo "inactive")
    if [[ "$status" == "active" ]]; then
        print_success "✓ Servicio actualmente activo"
    else
        print_warning "⚠ Servicio actualmente $status"
    fi
    
    return 0
}

# Función para verificar logs
check_logs() {
    print_status "Verificando sistema de logs..."
    
    # Verificar directorios de logs
    local log_dirs=("system" "digital" "photos" "vision")
    for dir in "${log_dirs[@]}"; do
        if [[ -d "/var/log/calippo/$dir" ]]; then
            print_success "✓ Directorio /var/log/calippo/$dir existe"
        else
            print_error "✗ Directorio /var/log/calippo/$dir no existe"
            return 1
        fi
    done
    
    # Verificar permisos
    local owner=$(stat -c '%U:%G' /var/log/calippo 2>/dev/null || echo "unknown")
    if [[ "$owner" == "nvidia:nvidia" ]]; then
        print_success "✓ Permisos correctos en directorio de logs"
    else
        print_warning "⚠ Permisos incorrectos: $owner (esperado: nvidia:nvidia)"
    fi
    
    # Verificar archivos de log específicos
    local log_files=(
        "/var/log/calippo/system/calippo_jetson.log"
        "/var/log/calippo/system/calippo_jetson_metrics.log"
    )
    
    for file in "${log_files[@]}"; do
        if [[ -f "$file" ]]; then
            local size=$(stat -c '%s' "$file" 2>/dev/null || echo "0")
            if [[ "$size" -gt 0 ]]; then
                print_success "✓ $file existe y tiene contenido ($size bytes)"
            else
                print_warning "⚠ $file existe pero está vacío"
            fi
        else
            print_warning "⚠ $file no existe aún"
        fi
    done
    
    return 0
}

# Función para verificar configuración de rsyslog
check_rsyslog() {
    print_status "Verificando configuración de rsyslog..."
    
    # Verificar archivo de configuración
    if [[ -f "/etc/rsyslog.d/50-calippo.conf" ]]; then
        print_success "✓ Configuración de rsyslog instalada"
    else
        print_error "✗ Configuración de rsyslog no encontrada"
        return 1
    fi
    
    # Verificar que rsyslog está funcionando
    if systemctl is-active rsyslog >/dev/null 2>&1; then
        print_success "✓ Servicio rsyslog activo"
    else
        print_error "✗ Servicio rsyslog no activo"
        return 1
    fi
    
    return 0
}

# Función para verificar configuración de logrotate
check_logrotate() {
    print_status "Verificando configuración de logrotate..."
    
    # Verificar archivo de configuración
    if [[ -f "/etc/logrotate.d/calippo" ]]; then
        print_success "✓ Configuración de logrotate instalada"
    else
        print_error "✗ Configuración de logrotate no encontrada"
        return 1
    fi
    
    # Verificar que logrotate está instalado
    if command -v logrotate >/dev/null 2>&1; then
        print_success "✓ logrotate instalado"
    else
        print_error "✗ logrotate no instalado"
        return 1
    fi
    
    return 0
}

# Función para verificar script launcher
check_launcher() {
    print_status "Verificando script launcher..."
    
    local launcher="/home/nvidia/Desktop/Calippo_jetson/run_calippo.sh"
    
    # Verificar que existe
    if [[ -f "$launcher" ]]; then
        print_success "✓ Script launcher existe"
    else
        print_error "✗ Script launcher no encontrado"
        return 1
    fi
    
    # Verificar permisos de ejecución
    if [[ -x "$launcher" ]]; then
        print_success "✓ Script launcher tiene permisos de ejecución"
    else
        print_error "✗ Script launcher no tiene permisos de ejecución"
        return 1
    fi
    
    return 0
}

# Función para verificar proceso en ejecución
check_running_process() {
    print_status "Verificando proceso en ejecución..."
    
    # Buscar proceso de la app modular
    local process_count=$(ps aux | grep -E "python.*(/home/.*/Calippo_jetson/main\.py|-m +gentl\.app|gentl/app\.py)" | grep -v grep | wc -l)
    
    if [[ "$process_count" -gt 0 ]]; then
        print_success "✓ Proceso Calippo ejecutándose ($process_count proceso(s))"
        
        # Mostrar información del proceso
        echo "   Detalles del proceso:"
        ps aux | grep -E "python.*(/home/.*/Calippo_jetson/main\.py|-m +gentl\.app|gentl/app\.py)" | grep -v grep | while read line; do
            echo "   $line"
        done
    else
        print_warning "⚠ No se encontró proceso Calippo ejecutándose"
    fi
    
    return 0
}

# Función para verificar logs en tiempo real
check_realtime_logs() {
    print_status "Verificando logs en tiempo real..."
    
    local log_file="/var/log/calippo/system/calippo_jetson.log"
    
    if [[ -f "$log_file" ]]; then
        # Verificar si el archivo está siendo escrito (últimos 30 segundos)
        local last_modified=$(stat -c '%Y' "$log_file" 2>/dev/null || echo "0")
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_modified))
        
        if [[ "$time_diff" -lt 30 ]]; then
            print_success "✓ Logs actualizándose (última modificación hace $time_diff segundos)"
        else
            print_warning "⚠ Logs no actualizándose recientemente (última modificación hace $time_diff segundos)"
        fi
        
        # Mostrar últimas líneas
        echo "   Últimas 3 líneas del log:"
        tail -3 "$log_file" 2>/dev/null | while read line; do
            echo "   $line"
        done
    else
        print_warning "⚠ Archivo de log principal no existe"
    fi
    
    return 0
}

# Función para verificar espacio en disco
check_disk_space() {
    print_status "Verificando espacio en disco..."
    
    # Verificar espacio en /var/log
    local log_space=$(df -h /var/log 2>/dev/null | tail -1 | awk '{print $4}')
    if [[ -n "$log_space" ]]; then
        print_success "✓ Espacio disponible en /var/log: $log_space"
    else
        print_warning "⚠ No se pudo verificar espacio en /var/log"
    fi
    
    # Verificar espacio en directorio de trabajo
    local work_space=$(df -h /home/nvidia/Desktop/Calippo_jetson 2>/dev/null | tail -1 | awk '{print $4}')
    if [[ -n "$work_space" ]]; then
        print_success "✓ Espacio disponible en directorio de trabajo: $work_space"
    else
        print_warning "⚠ No se pudo verificar espacio en directorio de trabajo"
    fi
    
    return 0
}

# Función para realizar prueba de reinicio
test_reboot() {
    print_status "¿Desea probar el autoarranque reiniciando el sistema?"
    echo "   Esta prueba reiniciará el equipo y verificará que Calippo arranque automáticamente."
    echo "   ⚠️  ADVERTENCIA: Esto reiniciará el sistema inmediatamente"
    echo ""
    read -p "¿Continuar con la prueba de reinicio? (s/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        print_status "Reiniciando sistema en 10 segundos..."
        print_status "Después del reinicio, ejecute este script nuevamente para verificar"
        sleep 10
        sudo reboot
    else
        print_status "Prueba de reinicio cancelada"
    fi
}

# Función para mostrar resumen
show_summary() {
    echo ""
    echo "============================================================================="
    print_status "📊 RESUMEN DE VERIFICACIÓN"
    echo "============================================================================="
    echo ""
    
    # Estado del servicio
    echo "🔧 Estado del servicio:"
    systemctl status calippo.service --no-pager -l || true
    echo ""
    
    # Logs recientes
    echo "📝 Logs recientes:"
    if [[ -f "/var/log/calippo/system/calippo_jetson.log" ]]; then
        echo "Últimas 5 líneas de calippo_jetson.log:"
        tail -5 /var/log/calippo/system/calippo_jetson.log
    else
        echo "No hay logs de aplicación"
    fi
    echo ""
    
    # Métricas de rendimiento
    if [[ -f "/var/log/calippo/system/calippo_jetson_metrics.log" ]]; then
        echo "📊 Métricas de rendimiento (última entrada):"
        tail -1 /var/log/calippo/system/calippo_jetson_metrics.log
    fi
    echo ""
    
    # Uso de recursos
    echo "💾 Uso de recursos:"
    ps aux | grep -E "python.*(/home/.*/Calippo_jetson/main\.py|-m +gentl\.app|gentl/app\.py)" | grep -v grep || echo "Proceso no encontrado"
    echo ""
    
    # Espacio en disco
    echo "💽 Espacio en disco:"
    df -h /var/log/calippo 2>/dev/null || echo "Directorio de logs no encontrado"
    echo ""
}

# Función principal
main() {
    echo "============================================================================="
    echo "🔍 VERIFICACIÓN POST-INSTALACIÓN CALIPPO"
    echo "============================================================================="
    echo ""
    
    local errors=0
    
    # Ejecutar todas las verificaciones
    check_systemd_service || ((errors++))
    echo ""
    
    check_logs || ((errors++))
    echo ""
    
    check_rsyslog || ((errors++))
    echo ""
    
    check_logrotate || ((errors++))
    echo ""
    
    check_launcher || ((errors++))
    echo ""
    
    check_running_process || ((errors++))
    echo ""
    
    check_realtime_logs || ((errors++))
    echo ""
    
    check_disk_space || ((errors++))
    echo ""
    
    # Mostrar resumen
    show_summary
    
    # Resultado final
    if [[ $errors -eq 0 ]]; then
        echo "============================================================================="
        print_success "🎉 VERIFICACIÓN COMPLETADA - TODO FUNCIONANDO CORRECTAMENTE"
        echo "============================================================================="
        echo ""
        print_status "Comandos útiles:"
        echo "  systemctl status calippo.service"
        echo "  tail -f /var/log/calippo/system/calippo_jetson.log"
        echo "  sudo journalctl -u calippo.service -f"
        echo ""
    else
        echo "============================================================================="
        print_error "❌ VERIFICACIÓN COMPLETADA CON $errors ERROR(ES)"
        echo "============================================================================="
        echo ""
        print_status "Revise los errores anteriores y ejecute el script de instalación nuevamente si es necesario"
        echo ""
    fi
    
    # Ofrecer prueba de reinicio
    test_reboot
}

# Ejecutar función principal
main "$@"

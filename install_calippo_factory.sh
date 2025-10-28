#!/bin/bash

# =============================================================================
# SCRIPT DE INSTALACIÓN CALIPPO PARA FÁBRICA
# =============================================================================
# Este script instala y configura Calippo en un equipo de fábrica (Jetson Orin)
# Incluye: autoarranque, logging, sistema de monitoreo y comprobaciones
# =============================================================================

set -euo pipefail  # Salir si hay errores

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para imprimir mensajes con colores
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

# Función para verificar si el comando existe
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Función para verificar si el usuario es nvidia
check_user() {
    if [[ "$USER" != "nvidia" ]]; then
        print_error "Este script debe ejecutarse como usuario 'nvidia'"
        print_error "Usuario actual: $USER"
        exit 1
    fi
    print_success "Usuario correcto: $USER"
}

# Función para verificar directorio de trabajo
check_working_directory() {
    local expected_dir="/home/nvidia/Desktop/Calippo_jetson"
    if [[ "$PWD" != "$expected_dir" ]]; then
        print_error "Directorio incorrecto. Debe estar en: $expected_dir"
        print_error "Directorio actual: $PWD"
        print_status "Ejecuta: cd $expected_dir"
        exit 1
    fi
    print_success "Directorio correcto: $PWD"
}

# Función para verificar archivos necesarios
check_required_files() {
    local files=(
        "gentl/PruebaAravis.py"
        "gentl/logging_system.py"
        "run_calippo.sh"
    )
    
    print_status "Verificando archivos necesarios..."
    for file in "${files[@]}"; do
        if [[ ! -f "$file" ]]; then
            print_error "Archivo faltante: $file"
            exit 1
        fi
        print_success "✓ $file"
    done
}

# Función para crear directorios de logs
create_log_directories() {
    print_status "Creando directorios de logs..."
    
    sudo mkdir -p /var/log/calippo/{system,digital,photos,vision}
    sudo chown -R nvidia:nvidia /var/log/calippo
    sudo chmod -R 755 /var/log/calippo
    
    print_success "Directorios de logs creados"
}

# Función para instalar dependencias del sistema
install_system_dependencies() {
    print_status "Instalando dependencias del sistema..."
    
    # Actualizar repositorios
    sudo apt update
    
    # Instalar logrotate si no está instalado
    if ! command_exists logrotate; then
        print_status "Instalando logrotate..."
        sudo apt install -y logrotate
    else
        print_success "logrotate ya instalado"
    fi
    
    # Verificar rsyslog
    if ! command_exists rsyslog; then
        print_status "Instalando rsyslog..."
        sudo apt install -y rsyslog
    else
        print_success "rsyslog ya instalado"
    fi
    
    print_success "Dependencias del sistema instaladas"
}

# Función para configurar rsyslog
configure_rsyslog() {
    print_status "Configurando rsyslog..."
    
    # Crear configuración de rsyslog para Calippo
    sudo tee /etc/rsyslog.d/50-calippo.conf > /dev/null << 'EOF'
# Configuración de logs para Calippo Jetson
# LOCAL0 facility para logs de la aplicación

# Todos los mensajes LOCAL0 van al archivo principal
local0.* /var/log/calippo/system/syslog.log

# Errores específicos van a un archivo separado
local0.err /var/log/calippo/system/syslog_errors.log

# Configuración de permisos para archivos de log
$FileOwner nvidia
$FileGroup nvidia
$FileCreateMode 0644
EOF

    # Reiniciar rsyslog
    sudo systemctl restart rsyslog
    
    # Crear archivo de log inicial
    sudo touch /var/log/calippo/system/syslog.log
    sudo chown nvidia:nvidia /var/log/calippo/system/syslog.log
    sudo chmod 644 /var/log/calippo/system/syslog.log
    
    print_success "rsyslog configurado"
}

# Función para configurar logrotate
configure_logrotate() {
    print_status "Configurando logrotate..."
    
    # Crear configuración de logrotate
    sudo tee /etc/logrotate.d/calippo > /dev/null << 'EOF'
# Configuración de rotación de logs para Calippo Jetson
/var/log/calippo/system/*.log
/var/log/calippo/digital/*.log
/var/log/calippo/vision/vision_log.csv
/var/log/calippo/vision/vision_log.jsonl
{
    daily
    rotate 30
    compress
    delaycompress
    dateext
    dateformat -%Y%m%d
    copytruncate
    create 0644 nvidia nvidia
    postrotate
        sudo systemctl restart rsyslog > /dev/null 2>&1 || true
    endscript
}
EOF

    print_success "logrotate configurado"
}

# Función para configurar systemd service
configure_systemd_service() {
    print_status "Configurando servicio systemd..."
    
    # Crear archivo de servicio systemd
    sudo tee /etc/systemd/system/calippo.service > /dev/null << 'EOF'
[Unit]
Description=Calippo Jetson - Vision App (PruebaAravis)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=10

[Service]
Type=notify
User=nvidia
Group=nvidia
WorkingDirectory=/home/nvidia/Desktop/Calippo_jetson
ExecStart=/home/nvidia/Desktop/Calippo_jetson/run_calippo.sh
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1
Environment=HEADLESS=1
Environment=QT_QPA_PLATFORM=offscreen
StandardOutput=journal
StandardError=journal
# Watchdog: la app envía READY=1 y WATCHDOG=1 periódicamente
WatchdogSec=30
NotifyAccess=main
# Límites de recursos
CPUAccounting=true
MemoryAccounting=true
IOAccounting=true
Nice=0
# Reinicios limpios
KillMode=mixed
TimeoutStopSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Recargar configuración de systemd
    sudo systemctl daemon-reload
    
    # Habilitar el servicio para autoarranque
    sudo systemctl enable calippo.service
    
    print_success "Servicio systemd configurado y habilitado"
}

# Función para configurar permisos del script launcher
configure_launcher_permissions() {
    print_status "Configurando permisos del script launcher..."
    
    # Dar permisos de ejecución al script
    chmod +x run_calippo.sh
    
    # Verificar que el script tiene permisos correctos
    if [[ -x "run_calippo.sh" ]]; then
        print_success "Script launcher configurado correctamente"
    else
        print_error "Error configurando permisos del script launcher"
        exit 1
    fi
}

# Función para configurar cron job para logrotate
configure_cron_job() {
    print_status "Configurando cron job para logrotate..."
    
    # Crear cron job para ejecutar logrotate diariamente a las 00:00
    (crontab -l 2>/dev/null; echo "0 0 * * * /usr/sbin/logrotate /etc/logrotate.d/calippo > /dev/null 2>&1") | crontab -
    
    print_success "Cron job configurado"
}

# Función para verificar instalación
verify_installation() {
    print_status "Verificando instalación..."
    
    # Verificar que el servicio está habilitado
    if systemctl is-enabled calippo.service >/dev/null 2>&1; then
        print_success "✓ Servicio habilitado para autoarranque"
    else
        print_error "✗ Servicio no habilitado"
        return 1
    fi
    
    # Verificar directorios de logs
    if [[ -d "/var/log/calippo" ]]; then
        print_success "✓ Directorios de logs creados"
    else
        print_error "✗ Directorios de logs no creados"
        return 1
    fi
    
    # Verificar configuración de rsyslog
    if [[ -f "/etc/rsyslog.d/50-calippo.conf" ]]; then
        print_success "✓ Configuración de rsyslog instalada"
    else
        print_error "✗ Configuración de rsyslog no instalada"
        return 1
    fi
    
    # Verificar configuración de logrotate
    if [[ -f "/etc/logrotate.d/calippo" ]]; then
        print_success "✓ Configuración de logrotate instalada"
    else
        print_error "✗ Configuración de logrotate no instalada"
        return 1
    fi
    
    # Verificar script launcher
    if [[ -x "run_calippo.sh" ]]; then
        print_success "✓ Script launcher configurado"
    else
        print_error "✗ Script launcher no configurado"
        return 1
    fi
    
    print_success "Instalación verificada correctamente"
}

# Función para probar el servicio
test_service() {
    print_status "Probando el servicio..."
    
    # Iniciar el servicio
    sudo systemctl start calippo.service
    
    # Esperar un momento para que inicie
    sleep 5
    
    # Verificar estado
    if systemctl is-active calippo.service >/dev/null 2>&1; then
        print_success "✓ Servicio iniciado correctamente"
    else
        print_error "✗ Error iniciando el servicio"
        print_status "Verificando logs del servicio..."
        sudo journalctl -u calippo.service --no-pager -n 20
        return 1
    fi
    
    # Verificar que los logs se están generando
    sleep 10
    if [[ -f "/var/log/calippo/system/calippo_jetson.log" ]]; then
        print_success "✓ Logs de aplicación generándose"
    else
        print_warning "⚠ Logs de aplicación aún no generados (puede ser normal)"
    fi
    
    print_success "Servicio funcionando correctamente"
}

# Función para mostrar información de estado
show_status() {
    print_status "Estado del sistema Calippo:"
    echo ""
    
    # Estado del servicio
    echo "📊 Estado del servicio:"
    systemctl status calippo.service --no-pager || true
    echo ""
    
    # Logs recientes
    echo "📝 Logs recientes (últimas 5 líneas):"
    if [[ -f "/var/log/calippo/system/calippo_jetson.log" ]]; then
        tail -5 /var/log/calippo/system/calippo_jetson.log
    else
        echo "No hay logs de aplicación aún"
    fi
    echo ""
    
    # Uso de memoria
    echo "💾 Uso de memoria:"
    ps aux | grep -E "(PruebaAravis|python.*PruebaAravis)" | grep -v grep || echo "Proceso no encontrado"
    echo ""
    
    # Espacio en disco
    echo "💽 Espacio en disco para logs:"
    df -h /var/log/calippo 2>/dev/null || echo "Directorio de logs no encontrado"
}

# Función principal
main() {
    echo "============================================================================="
    echo "🚀 INSTALACIÓN CALIPPO PARA FÁBRICA"
    echo "============================================================================="
    echo ""
    
    # Verificaciones iniciales
    check_user
    check_working_directory
    check_required_files
    
    echo ""
    print_status "Iniciando instalación..."
    echo ""
    
    # Instalación paso a paso
    install_system_dependencies
    create_log_directories
    configure_rsyslog
    configure_logrotate
    configure_systemd_service
    configure_launcher_permissions
    configure_cron_job
    
    echo ""
    print_status "Verificando instalación..."
    verify_installation
    
    echo ""
    print_status "Probando servicio..."
    test_service
    
    echo ""
    echo "============================================================================="
    print_success "🎉 INSTALACIÓN COMPLETADA EXITOSAMENTE"
    echo "============================================================================="
    echo ""
    
    show_status
    
    echo ""
    print_status "Comandos útiles para monitoreo:"
    echo "  systemctl status calippo.service          # Estado del servicio"
    echo "  systemctl restart calippo.service         # Reiniciar servicio"
    echo "  tail -f /var/log/calippo/system/calippo_jetson.log  # Ver logs en tiempo real"
    echo "  sudo journalctl -u calippo.service -f    # Ver logs del sistema"
    echo ""
    print_status "El servicio se iniciará automáticamente al reiniciar el equipo"
    echo ""
}

# Ejecutar función principal
main "$@"

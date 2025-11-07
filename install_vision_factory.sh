#!/bin/bash
# -------------------------------------------------------------
# install_vision_factory.sh
# -------------------------------------------------------------
# instala/actualiza la Vision App como servicio systemd.
# prepara directorios de logs, escribe la unidad
#           `vision-app.service` y la habilita/arranca.
# Dónde se usa: despliegue en fábrica (headless).
# -------------------------------------------------------------
set -euo pipefail

# Instalación y configuración de la Vision App (Jetson, systemd + journald)

BLUE='\033[0;34m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info(){ echo -e "${BLUE}[INFO]${NC} $1"; }
ok(){ echo -e "${GREEN}[OK]${NC} $1"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $1"; }
err(){ echo -e "${RED}[ERR]${NC} $1"; }

PROJECT_DIR="/home/nvidia/Desktop/Calippo_jetson"
SERVICE_NAME="vision-app.service"
PYTHON_BIN="$PROJECT_DIR/vision_app/.venv/bin/python"
ENTRYPOINT="$PROJECT_DIR/main.py"
LOG_DIR="/var/log/vision_app"
YOLO_CFG="$PROJECT_DIR/vision_app/config_yolo.yaml"

require(){ command -v "$1" >/dev/null 2>&1 || { err "Falta comando $1"; exit 1; }; }

main(){
  if [[ "${USER:-}" != "nvidia" ]]; then err "Ejecuta como usuario nvidia"; exit 1; fi
  if [[ "$PWD" != "$PROJECT_DIR" ]]; then err "cd $PROJECT_DIR"; exit 1; fi

  info "Creando directorios de logs"
  sudo mkdir -p "$LOG_DIR"/{system,io,images,vision,archive,photosmanual}
  sudo chown -R nvidia:nvidia "$LOG_DIR"

  info "Comprobando intérprete Python"
  if [[ ! -x "$PYTHON_BIN" ]]; then err "No existe $PYTHON_BIN (crea venv)"; exit 1; fi

  info "Escribiendo unidad systemd: $SERVICE_NAME"
  sudo tee "/etc/systemd/system/$SERVICE_NAME" >/dev/null <<EOF
[Unit]
Description=Vision App (headless)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON_BIN $ENTRYPOINT
Environment=HEADLESS=1
Environment=AUTO_RUN=1
Environment=PYTHONPATH=$PROJECT_DIR/vision_app
Environment=CONFIG_YOLO=$YOLO_CFG
Environment=LOG_TO_SYSLOG=0
Environment=LOG_TO_FILE=1
Environment=LOG_DIR=$LOG_DIR
SyslogIdentifier=vision-app
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  info "Recargando systemd"
  sudo systemctl daemon-reload
  sudo systemctl enable "$SERVICE_NAME"
  sudo systemctl restart "$SERVICE_NAME"

  ok "Servicio activo"
  systemctl status --no-pager "$SERVICE_NAME" || true
  echo
  info "Comandos útiles"
  echo "  sudo journalctl -u $SERVICE_NAME -f --no-pager"
  echo "  tail -f $LOG_DIR/vision/vision_log.csv   # tras primeras detecciones"
  echo "  tail -f $LOG_DIR/system/system.log       # si LOG_TO_FILE=1"
}

require systemctl
main "$@"



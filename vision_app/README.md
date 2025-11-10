# 🎯 Vision App (YOLO + Aravis)

Aplicación modular de visión por computador para Jetson (YOLO + cámaras GenICam vía Aravis).

## 📦 Estructura del Proyecto

```
vision_app/
├── app.py                    # Orquestador principal (inicialización y bucle principal)
├── main.py                   # Punto de entrada (usado por systemd)
├── config_yolo.yaml          # Configuración YOLO (modelo, thresholds, clases)
├── core/                     # Módulos centrales
│   ├── settings.py           # Configuración (YAML + env vars) y AppContext
│   ├── logging.py            # Sistema de logging multi-dominio
│   ├── optimizations.py      # Optimizaciones del sistema (CUDA, OpenCV, PyTorch)
│   ├── recording.py          # Grabación de vídeo e imágenes (Recorder, ImagesManager)
│   └── timings.py            # Logging de tiempos de procesamiento
├── model/                    # Modelos ML y servicios de inferencia
│   ├── detection/            # Detección de objetos
│   │   ├── detection_service.py  # Servicio completo (YOLO + tracking + clasificación)
│   │   ├── config.py         # Configuración y carga del modelo YOLO
│   │   └── yolo_wrapper.py   # Wrapper del modelo Ultralytics YOLO
│   ├── classifier/           # Clasificación de imágenes
│   │   └── multiclass.py     # Clasificador multiclase (buenas/malas/defectuosas)
│   └── tracking/             # Tracking de objetos
│       └── simple_tracker.py # Asignación de IDs estables (IoU-based)
├── camera/                   # Backends de cámara
│   ├── device_manager.py     # Interfaz CameraBackend y funciones open_camera/stop_camera
│   ├── selector.py           # Selección automática de backend (GenICam/ONVIF)
│   ├── genicam_aravis_backend.py  # Backend GenICam/Aravis (GigE/USB)
│   └── onvif_rtsp_backend.py # Backend ONVIF/RTSP (cámaras IP)
└── developer_ui/             # Interfaz de depuración (OpenCV)
    ├── app_controller.py      # Controlador de eventos UI
    ├── handlers.py           # Lógica de acciones del panel (RUN/STOP/INFO/CONFIG)
    ├── window.py             # Gestión de ventana principal
    ├── panel.py              # Panel lateral y detección de clics
    ├── overlay.py            # Overlays visuales (detecciones, HUD, gamma)
    ├── compositor.py         # Compositor de frame + panel
    ├── indicators.py         # Indicadores HUD (REC, AWB, AutoCal)
    └── state.py              # Estado compartido de la ventana
```

## 🏗️ Arquitectura

### Diseño Modular

La aplicación está diseñada con separación clara de responsabilidades:

- **Orquestación**: `app.py` coordina inicialización y bucle principal
- **Configuración**: `core/settings.py` centraliza configuración (YAML + env)
- **Modelos ML**: `model/` contiene detección, clasificación y tracking
- **Cámaras**: `camera/` abstrae acceso a diferentes backends
- **UI**: `developer_ui/` proporciona interfaz de depuración opcional

### Flujo de Ejecución

1. **Inicialización** (`app.py`):
   - Carga configuración desde YAML y variables de entorno
   - Aplica optimizaciones del sistema (CUDA, OpenCV, PyTorch)
   - Carga modelos (YOLO y clasificador)
   - Inicializa cámara (auto-detección o backend forzado)
   - Crea `DetectionService` en hilo separado

2. **Bucle Principal** (`app.py`):
   - Captura frames de la cámara
   - Publica frames en cola para `DetectionService`
   - Consume resultados de detección para overlay
   - Procesa eventos de UI (RUN/STOP/CONFIG/INFO)
   - Aplica overlays visuales y muestra frame

3. **Servicio de Detección** (`model/detection/detection_service.py`):
   - Consume frames de la cola
   - Ejecuta inferencia YOLO
   - Aplica tracking (asignación de IDs)
   - Clasifica cada detección
   - Registra eventos en CSV y guarda imágenes

### Módulos Principales

#### `core/settings.py`
- **`load_settings()`**: Fusiona configuración YAML y variables de entorno
- **`Settings`**: Configuración estática (headless, auto_run, yolo, camera, etc.)
- **`AppContext`**: Contexto de ejecución con colas y estado compartido

#### `core/logging.py`
- Sistema de logging multi-dominio:
  - `system` (default): logs del sistema
  - `vision`: telemetría por detección
  - `images`: guardado de imágenes
  - `timings`: tiempos de procesamiento
  - `io`: salidas digitales/PLC
- Handlers configurables:
  - `LOG_TO_SYSLOG=0|1`: salida a journald (default: 0 en systemd)
  - `LOG_TO_FILE=1`: archivos rotativos en `LOG_DIR`
  - `LOG_LEVEL=INFO|DEBUG|WARNING|ERROR`

#### `model/detection/detection_service.py`
- Servicio completo de detección en hilo separado
- Pipeline: YOLO → tracking → clasificación → logging
- Registra `vision_log.csv` con tracking details (track_id, track_age_ms, event, id_switch)
- Gestiona guardado de imágenes (bad/good) vía `core/recording.py`

#### `camera/device_manager.py`
- **`CameraBackend`**: Interfaz común para todos los backends
  - Métodos: `open()`, `start()`, `stop()`, `get_frame()`, `get()`, `set()`, `get_node()`
  - Utilidades estáticas: `safe_get()`, `safe_set()`
- **`open_camera()`**: Abre cámara usando selector automático o backend forzado
- **`stop_camera()`**: Cierra cámara de forma segura

#### `camera/selector.py`
- **`CameraSelector.create()`**: Selecciona backend apropiado
  - Modo "auto": detecta GenICam/Aravis o usa ONVIF si hay URL
  - Modo "aravis"/"genicam": fuerza backend GenICam/Aravis
  - Modo "onvif": fuerza backend ONVIF/RTSP

#### `camera/genicam_aravis_backend.py`
- Implementación GenICam/Aravis para cámaras GigE/USB
- Captura frames BGR con conversión Bayer automática
- Control de parámetros (exposición, ganancia, ROI, formato de píxel)
- Métricas de rendimiento (FPS, ancho de banda, latencia)
- Métodos específicos: `set_roi()`, `restore_full_frame()`

#### `camera/onvif_rtsp_backend.py`
- Implementación ONVIF/RTSP para cámaras IP
- Captura frames BGR desde stream RTSP usando OpenCV
- Soporte para propiedades básicas (Width, Height, FPS)

#### `developer_ui/`
- **`app_controller.py`**: Coordina eventos UI y los traduce a acciones
- **`handlers.py`**: Lógica de acciones (RUN/STOP/INFO/CONFIG/AWB/AUTO_CAL/RECORD_60S)
- **`window.py`**: Gestión de ventana principal (creación, destrucción, renderizado)
- **`panel.py`**: Panel lateral con botones y detección de clics
- **`overlay.py`**: Overlays visuales (detecciones YOLO, HUD, corrección gamma)
- **`compositor.py`**: Combina frame y panel para renderizado final
- **`indicators.py`**: Indicadores HUD (REC, AWB, AutoCal) sincronizados con flags

## ⚙️ Configuración

### Archivo YAML (`config_yolo.yaml`)

```yaml
yolo:
  model_path: "/home/nvidia/Desktop/Vision_App/vision_app/v2_yolov8n_HERMASA_finetune.pt"
  image_size: 416
  confidence_threshold: 0.3
  iou_threshold: 0.45
  classes: [0, 1]  # can, hand
```

### Variables de Entorno

#### Ejecución
- `HEADLESS=1`: No crea UI, auto-enfila `RUN`
- `AUTO_RUN=1`: Auto-enfila `RUN` aunque exista UI
- `CONFIG_YOLO`: Ruta al archivo YAML de configuración

#### Cámara
- `CAMERA_BACKEND=auto|aravis|onvif`: Backend de cámara (default: auto)
- `RTSP_URL=rtsp://user:pass@ip/...`: URL RTSP (para backend ONVIF)

#### Logging
- `LOG_TO_SYSLOG=0|1`: Salida a journald (default: 0 en systemd)
- `LOG_TO_FILE=1`: Archivos rotativos en `LOG_DIR`
- `LOG_DIR=/var/log/vision_app`: Directorio de logs (default: `/var/log/vision_app`)
- `LOG_LEVEL=INFO|DEBUG|WARNING|ERROR`: Nivel de logging (default: INFO)

#### Clasificador
- `CLF_BAD_THRESHOLD=0.87`: Umbral para clasificar como "bad" (default: 0.87)

## 🖥️ Uso

### Ejecución Manual (con UI)

```bash
cd /home/nvidia/Desktop/Vision_App
source vision_app/.venv/bin/activate
export PYTHONPATH=/home/nvidia/Desktop/Vision_App/vision_app
python main.py
```

### Ejecución Headless (systemd)

```bash
sudo systemctl start vision-app.service
sudo systemctl status vision-app.service
sudo journalctl -u vision-app.service -f --no-pager
```

### Ver Logs

```bash
# Logs del sistema (journald)
sudo journalctl -u vision-app.service -f --no-pager

# Logs de archivos (si LOG_TO_FILE=1)
tail -f /var/log/vision_app/system/system.log
tail -f /var/log/vision_app/vision/vision_log.csv
tail -f /var/log/vision_app/images/YYYY-MM-DD/images.csv
```

## 📊 Logging y Trazabilidad

### Estructura de Logs

```
/var/log/vision_app/
├── system/
│   └── system.log          # Logs del sistema (si LOG_TO_FILE=1)
├── vision/
│   └── vision_log.csv     # Por detección: ts, frame_id, num_boxes, classes, avg_conf,
│                           #              track_id, track_age_ms, track_event, id_switch, etc.
├── images/
│   └── YYYY-MM-DD/
│       ├── good_*.jpg      # Imágenes "good" periódicas
│       ├── bad_*.jpg       # Imágenes "bad" por detección
│       └── images.csv      # Registro: ts, tipo, path, reason, avg_conf, class, track_id
├── timings/
│   └── timings.csv         # Tiempos de procesamiento por etapa
└── archive/
    └── YYYY-MM-DD.zip      # Archivos ZIP diarios de imágenes
```

### Formatos de Log

#### `vision_log.csv`
Columnas: `timestamp`, `frame_id`, `num_boxes`, `classes`, `avg_conf`, `proc_ms`, `camera_exposure`, `camera_gain`, `width`, `height`, `yolo_threshold`, `bbox`, `track_id`, `track_age_ms`, `track_event`, `id_switch`, `clasificador`, `clasificador_conf`, `decision`

#### `images.csv`
Columnas: `timestamp`, `tipo`, `path`, `reason`, `avg_conf`, `class`, `track_id`

#### `timings.csv`
Columnas: `timestamp`, `frame_id`, `yolo_ms`, `crop_ms`, `forward_ms`, `classify_ms`, `csv_ms`, `images_ms`, `total_ms`

## 🔧 Optimizaciones

El módulo `core/optimizations.py` aplica optimizaciones automáticas:

- **CUDA**: Configuración de memoria y streams
- **OpenCV**: Optimizaciones de threading y memoria
- **PyTorch**: Configuración de backend y optimizaciones de inferencia
- **Sistema**: CPU governor, memoria, red (Jetson)

## 🎥 Cámara

### Backends Soportados

1. **GenICam/Aravis** (`genicam_aravis_backend.py`):
   - Cámaras GigE/USB compatibles con GenICam
   - Control completo de parámetros (exposición, ganancia, ROI, etc.)
   - Métricas de rendimiento (FPS, ancho de banda, latencia)

2. **ONVIF/RTSP** (`onvif_rtsp_backend.py`):
   - Cámaras IP compatibles con ONVIF
   - Captura desde stream RTSP
   - Propiedades básicas (Width, Height, FPS)

### Selección Automática

El selector (`camera/selector.py`) detecta automáticamente:
- Dispositivos GenICam/Aravis disponibles
- URL RTSP proporcionada
- Backend forzado vía `CAMERA_BACKEND`

## 🧠 Modelos ML

### YOLO (Detección)
- Modelo: Ultralytics YOLOv8
- Backend: PyTorch con CUDA
- Post-procesamiento: NMS y merge de detecciones superpuestas

### Clasificador Multiclase
- Modelo: MobileNetV2 fine-tuned
- Clases: buenas, malas, defectuosas
- Umbral configurable: `CLF_BAD_THRESHOLD`

### Tracking
- Algoritmo: IoU-based tracking
- Asignación de IDs estables
- Eventos: start, update, end
- Detección de ID switching

## 🎨 UI de Desarrollo

La UI de desarrollo (`developer_ui/`) proporciona:

- **Ventana principal**: Frame de cámara + panel lateral
- **Panel de control**: Botones RUN/STOP/INFO/CONFIG/AWB/AUTO_CAL/RECORD_60S
- **Ventanas modales**: INFO (información de cámara), CONFIG (sliders de parámetros)
- **Overlays**: Detecciones YOLO, HUD, indicadores visuales
- **Corrección gamma**: Ajuste visual en tiempo real

## 🚀 Systemd Service

### Unidad: `vision-app.service`

```ini
[Unit]
Description=Vision App (headless)
After=network.target

[Service]
Type=simple
User=nvidia
WorkingDirectory=/home/nvidia/Desktop/Vision_App
ExecStart=/home/nvidia/Desktop/Vision_App/vision_app/.venv/bin/python /home/nvidia/Desktop/Vision_App/main.py
Restart=always
RestartSec=5

Environment=HEADLESS=1
Environment=AUTO_RUN=1
Environment=PYTHONPATH=/home/nvidia/Desktop/Vision_App/vision_app:/usr/lib/python3/dist-packages
Environment=CONFIG_YOLO=/home/nvidia/Desktop/Vision_App/vision_app/config_yolo.yaml
Environment=LOG_TO_SYSLOG=0
Environment=LOG_TO_FILE=1
Environment=LOG_DIR=/var/log/vision_app

[Install]
WantedBy=multi-user.target
```

### Comandos Útiles

```bash
# Iniciar servicio
sudo systemctl start vision-app.service

# Detener servicio
sudo systemctl stop vision-app.service

# Reiniciar servicio
sudo systemctl restart vision-app.service

# Ver estado
sudo systemctl status vision-app.service

# Ver logs en tiempo real
sudo journalctl -u vision-app.service -f --no-pager

# Ver últimos 100 logs
sudo journalctl -u vision-app.service -n 100 --no-pager
```

## 📝 Notas de Desarrollo

### Separación de Responsabilidades

- **`app.py`**: Solo orquestación (inicialización y bucle principal)
- **`core/`**: Funcionalidades centrales (config, logging, optimizaciones)
- **`model/`**: Modelos ML y servicios de inferencia
- **`camera/`**: Abstracción de backends de cámara
- **`developer_ui/`**: Interfaz de depuración (opcional)

### Logging

- Todos los logs pasan por `core/logging.py`
- No usar `print()` directamente (usar `log_info()`, `log_warning()`, etc.)
- Logs de debug usan `log_debug()` (solo aparecen si `LOG_LEVEL=DEBUG`)

### Cámaras

- Interfaz común: `CameraBackend` en `camera/device_manager.py`
- Utilidades genéricas: `CameraBackend.safe_get()`, `CameraBackend.safe_set()`
- Backends específicos: `AravisBackend`, `OnvifRtspBackend`

### Tracking

- Implementado en `model/tracking/simple_tracker.py`
- Integrado en `model/detection/detection_service.py`
- Eventos registrados en `vision_log.csv`

## 🔍 Troubleshooting

### Cámara no detectada
- Verificar que `LOG_TO_SYSLOG=0` (stdout capturado por systemd)
- Revisar logs: `sudo journalctl -u vision-app.service -n 50`
- Verificar permisos de cámara: `ls -l /dev/video*`

### Modelo no carga
- Verificar ruta en `config_yolo.yaml`
- Verificar que el archivo existe: `ls -lh vision_app/v2_yolov8n_HERMASA_finetune.pt`
- Revisar logs de inicialización

### Logs duplicados
- Verificar `LOG_TO_SYSLOG=0` en systemd (stdout ya va a journald)
- Verificar que no hay múltiples handlers en `core/logging.py`

### Width/Height muestran "N/A"
- Verificar que el backend implementa `get_node_value()` correctamente
- Revisar logs de inicialización de cámara

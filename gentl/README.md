# 🎯 gentl – Vision App (YOLO + Aravis)

Este directorio contiene la aplicación modular de visión por computador para Jetson (YOLO + cámaras GenICam vía Aravis).

## 📦 Estructura (arquitectura)
- `app.py`: orquestador. Crea contexto, carga settings, inicializa cámara, modelo e hilos.
- `core/`
  - `settings.py`: configuración central (YAML + env: `HEADLESS`, `AUTO_RUN`, `CONFIG_YOLO`).
  - `logging.py`: logging multi-dominio (`system`/`vision`/`images`/`io`) → journald y ficheros (si `LOG_TO_FILE=1`).
  - `device_manager.py`: apertura de cámara y aplicación de nodos base (PixelFormat, ROI, Exposure/Gain, etc.).
  - `recording.py`: grabación y gestor de imágenes (`ImagesManager`: bad/good, `images.csv`, archivado diario a `archive/`).
- `vision/`
  - `yolo_wrapper.py`: carga del modelo Ultralytics.
  - `yolo_service.py`: hilo de inferencia; publica resultados y registra `vision_log.csv` por detección (exposición, ganancia, resolución, umbral, bbox).
  - `overlay.py`, `ops.py`, `tracking.py`: visualización y utilidades.
- `camera/`
  - `camera_service.py`: `safe_get/safe_set` y helpers ROI.
- `ui/`
  - `window.py`, `panel.py`, `handlers.py`, `app_controller.py`: UI OpenCV (INFO/CONFIG modales y robustas en Jetson).



## Arquitectura modular de `gentl`

### Objetivo
Diseño modular, mantenible y robusto que conserva funcionalidad original (INFO/CONFIG/YOLO/recording) y soporta ejecución headless vía systemd.

### Orquestador
- `app.py`
  - Crea `AppContext` y carga `Settings`.
  - Aplica optimizaciones globales (vía `core/optimizations.py`).
  - Inicializa cámara (`core/device_manager.py`).
  - Carga modelos (`vision/yolo_wrapper.py`) y arranca `YoloService`.
  - Ejecuta el bucle principal: captura, compone UI, aplica overlays y procesa eventos mediante `ui/app_controller.py`.
  - Mantiene estado ligero de UI (gamma/patrón actual) y flags de indicadores (AWB/AutoCal).

### Configuración
- `core/settings.py`
  - `load_settings()`: fusiona YAML (`config_yolo.yaml`) y variables de entorno (`HEADLESS`, `AUTO_RUN`).
  - `Settings`: expone `raw_config`, `headless`, `auto_run`, `yolo`.
- `config_yolo.yaml`
  - `yolo.model_path|model`, `image_size`, `confidence_threshold`, `iou_threshold`, `classes`.
  

### Logging y trazabilidad

`core/logging.py` expone loggers por dominio (misma API info/warning/error/debug):
- `system` (logger por defecto `gentl`): ciclo de vida y estado.
- `vision`: telemetría por lata y métricas de inferencia.
- `images`: guardado de imágenes “bad” y “good” + CSV por día y archivado.
- `io`: salidas digitales/PLC (preparado; activo cuando haya hardware).

Handlers configurables por entorno:
- `LOG_TO_SYSLOG=1` (default): al journal (systemd).
- `LOG_TO_FILE=1`: a ficheros bajo `LOG_DIR` (default `/var/log/calippo`).
- `LOG_DIR=/var/log/calippo`, `LOG_LEVEL=INFO`.

Estructura de `/var/log/calippo/`:
- `system/system.log`: log de sistema (si `LOG_TO_FILE=1`).
- `vision/vision_log.csv`: por detección → `ts,frame_id,num_boxes,classes,avg_conf,proc_ms,camera_exposure,camera_gain,width,height,yolo_threshold,bbox`.
- `images/YYYY-MM-DD/`: JPG “bad” por detección y “good” periódicas + `images.csv` (ts,tipo,path,reason,avg_conf,class,track_id).
- `archive/`: ZIPs diarios de `images/YYYY-MM-DD`.
- `io/` o `digital/` (legacy): eventos de IO.
- `photosmanual/` (opcional): capturas manuales.


### Cámara
- `core/device_manager.py`
  - `DeviceManager.open_camera()`: abre backend (Aravis), aplica setup básico (PixelFormat/Trigger/FPS/Expo/Gain/AWB).
  - `stop_camera()`: paro seguro.
- `camera/camera_service.py`
  - `safe_get/safe_set`: acceso resiliente a nodos GenICam.
  - `set_roi/restore_full_frame`: gestión de ROI y restauración a frame completo.

### Inferencia y visión
- `vision/yolo_wrapper.py`: wrapper del modelo YOLO (CUDA cuando disponible).
- `vision/yolo_service.py`: hilo de inferencia; lee `builtins.latest_frame`, publica resultados en `context.infer_queue`.
- `vision/overlay.py`: dibuja detecciones y HUD; consume `context.infer_queue` con TTL para evitar parpadeos; HUD de cámara (ET/FPS) seguro.
- `vision/image_utils.py`: utilidades (gamma, etc.).
- `vision/ops.py|tracking.py`: operaciones auxiliares y asignación de IDs estables (usadas dentro de overlay/servicios).

### UI
- `ui/window.py`: creación/destrucción de ventana, pintado de frame con panel, pantalla negra inicial.
- `ui/panel.py`: composición del panel lateral y detección de clics (zonas RUN/STOP/CONFIG/INFO, etc.).
- `ui/handlers.py`: lógica de acciones del panel (RUN/STOP/INFO/CONFIG/AWB/AUTO_CAL/RECORD_60S). INFO/CONFIG son modales y robustos.
- `ui/app_controller.py`: capa fina que consume la cola de eventos y delega en `handlers`; gestiona callback de ratón.
- `ui/indicators.py`: overlays visuales (AWB/AutoCal) sincronizados con flags de `App`.

### Optimización y logging
- `core/optimizations.py`: aplica optimizaciones SO/CUDA/OpenCV/PyTorch; devuelve (conf, iou) YOLO desde settings.
- `core/logging.py`: logger multi-dominio (system/vision/images/io) con Syslog y file handlers opcionales.

### Grabación e imágenes
- `core/recording.py`:
  - `Recorder`: grabación de vídeo + overlay de estado.
  - `ImagesManager`: guarda imágenes “bad” por detección y “good” periódicas, mantiene `images.csv` y archiva cada día.

### Concurrencia y colas
- Hilos:
  - `YoloService` (daemon): consume frames y publica resultados.
  - INFO/CONFIG: modales en hilo principal para estabilidad GTK/X en Jetson.
- Colas:
  - `context.evt_queue`: acciones de UI (RUN/STOP/CONFIG/INFO/...).
  - `context.infer_queue`: resultados YOLO para overlay.

### Flags de ejecución
- `HEADLESS=1`: no crea UI, auto-enfila `RUN`.
- `AUTO_RUN=1`: auto-enfila `RUN` aunque exista UI.

### Systemd (headless)
- Unidad: `vision-app.service`.
- `WorkingDirectory`: `/home/nvidia/Desktop/Calippo_jetson`.
- `ExecStart`: `/home/nvidia/Desktop/Calippo_jetson/gentl/.venv/bin/python /home/nvidia/Desktop/Calippo_jetson/main.py`.
- `Environment` típico:
  - `HEADLESS=1`, `AUTO_RUN=1`.
  - `PYTHONPATH=/home/nvidia/Desktop/Calippo_jetson/gentl`.
  - `CONFIG_YOLO=/home/nvidia/Desktop/Calippo_jetson/gentl/config_yolo.yaml`.
  - `LOG_TO_SYSLOG=0|1`, `LOG_TO_FILE=1`, `LOG_DIR=/var/log/calippo`.
- Operación:
  - `systemctl status --no-pager vision-app`
  - `sudo journalctl -u vision-app -f --no-pager`
  - Ficheros (si activos): `tail -f /var/log/calippo/vision/vision_log.csv`, `tail -f /var/log/calippo/system/system.log`.

  
## ⚙️ Configuración
- YAML: `gentl/config_yolo.yaml` (recomendado usar ruta absoluta en `yolo.model|model_path`).
- Env opcionales (servicio/systemd):
  - `HEADLESS=1`, `AUTO_RUN=1`
  - `CONFIG_YOLO=/home/nvidia/Desktop/Calippo_jetson/gentl/config_yolo.yaml`
  - `LOG_TO_SYSLOG=0|1`, `LOG_TO_FILE=1`, `LOG_DIR=/var/log/calippo`

## 🖥️ Uso
### UI (debug local)
```bash
sudo systemctl stop vision-app.service   # liberar cámara
cd /home/nvidia/Desktop/Calippo_jetson/gentl && source .venv/bin/activate
export HEADLESS=0
python /home/nvidia/Desktop/Calippo_jetson/main.py
```
Controles: RUN/STOP, CONFIG/INFO (sliders exposición/ganancia/FPS; cierre limpio con ESC/ENTER/X), Gamma, Bayer, RECORD, AWB Once, AutoCal.

### Headless (producción)
El servicio `vision-app.service` arranca la app en continuo (sin UI).
```bash
sudo systemctl start vision-app.service
sudo systemctl enable vision-app.service
systemctl status --no-pager vision-app
sudo journalctl -u vision-app -f --no-pager
```

## 📝 Logging (dominios y ficheros)
- Dominios (`core/logging.py`): `system` (gentl), `vision`, `images`, `io`.
- Journal (filtrado):
```bash
sudo journalctl -u vision-app --no-pager | grep " gentl:"
sudo journalctl -u vision-app --no-pager | grep " vision:"
sudo journalctl -u vision-app --no-pager | grep " images:"
```
- Ficheros (si `LOG_TO_FILE=1` y `LOG_DIR=/var/log/calippo`):
  - `vision/vision_log.csv`: por detección → `ts,frame_id,num_boxes,classes,avg_conf,proc_ms,camera_exposure,camera_gain,width,height,yolo_threshold,bbox`.
  - `images/YYYY-MM-DD/images.csv` + JPGs bad/good; zip diario en `archive/`.
  - `system/system.log`: estado/arranque.

## 🔧 Ajustes de cámara al inicio
`core/device_manager.py` aplica nodos base: `PixelFormat=BayerBG8`, `Trigger=Off`, `FPS≈15`, `ExposureAuto/Mode=Off/Timed`, `ExposureTime` y `Gain` (valores de arranque editables en código; el log imprime los efectivos). 

## 🧪 Diagnóstico rápido
```bash
python - <<'PY'
import torch, cv2; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('opencv', cv2.__version__)
PY
python -c "import gi; gi.require_version('Aravis','0.6'); from gi.repository import Aravis as A; A.update_device_list(); print('Cam:', A.get_n_devices())"
```

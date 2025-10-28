# 🎯 App YOLO + Aravis (PruebaAravis) – Guía de uso específica

## 📋 Descripción

Sistema de detección de objetos en tiempo real con YOLO y cámaras GenICam (Aravis), optimizado para Jetson Orin. Esta guía se centra en la UI, ajustes y flujos internos de `PruebaAravis.py`.

## 📦 Arranque rápido (recordatorio)
```bash
sudo systemctl stop vision-app.service
export HEADLESS=0
python3 PruebaAravis.py
```

## 🖥️ UI: Paneles y controles
La ventana principal tiene imagen de cámara (izquierda) y panel lateral (derecha).

- RUN/STOP: iniciar/detener captura y detección.
- GRABAR 60s: guarda frames y ensambla vídeo (si hay ffmpeg).
- Gamma: deslizador; aplica LUT en software y en HW si la cámara soporta Gamma.
- Bayer (BG/RG/GR/GB): cambia demosaico y, en reposo, intenta ajustar PixelFormat en cámara.
- YOLO Confidence: umbral de confianza (típico 0.25–0.60).
- YOLO IOU: umbral NMS IOU (típico 0.40–0.50).
- Clasificador:
  - Confianza: umbral para marcar “Mala” en modo CONSERVADOR.
  - Modo: CONSERVADOR/NORMAL.
- INFO: ventana de sólo lectura con parámetros de cámara (PixelFormat, FPS, Exposure, Gain, Gamma, ROI...).
- CONFIG: ventana editable con sliders para Exposición, Ganancia y FPS, y toggles AUTO.
- AWB ONCE: auto-balance de blancos una vez; se desactiva al terminar.
- AUTO CAL: activa AUTO (exposición/ganancia/balance) brevemente y fija los valores resultantes.

Atajos en CONFIG:
- T: TriggerMode On/Off si existe.
- A/G: ExposureAuto/GainAuto.
- ENTER/ESC: aceptar/cancelar.

Indicadores en imagen:
- Overlays YOLO (cajas, IDs estables) y HUD de latencia/FPS.
- Indicador REC con cuenta atrás durante la grabación.

## ⚙️ Ajustes recomendados
- YOLO Confidence: sube para menos falsos positivos; baja para detectar más.
- IOU NMS: alto para suprimir solapes; bajo para permitir más cajas.
- Exposición/Ganancia: balancea blur/ruido (p.ej., ~5ms + 24dB en líneas rápidas).
- Gamma: mejora contraste; aplica en HW si la cámara lo soporta.
- Clasificador: modo CONSERVADOR exige alta confianza para “Mala”.

## 🔄 Flujo interno (resumen)
1) Captura (AravisBackend): configura ROI y obtiene el último frame (latest-frame) con demosaico Bayer→BGR.
2) Preprocesado: LUT de gamma y/o ROI de inferencia.
3) YOLO: detección (Ultralytics), NMS, fusión de solapes; parámetros ajustables (conf, iou, imgsz).
4) Tracking: IDs estables por similitud/IoU; persistencia breve anti-parpadeo.
5) Clasificación por lata: ROI circular, modelo PyTorch; guarda imagen si “Mala”.
6) Logging: por lata (CSV/JSONL), eventos del sistema y snapshots/defects.
7) Headless/Servicio: HEADLESS=1 con watchdog via systemd.

## 🧪 Verificaciones rápidas
```bash
# PyTorch
python3 - <<'PY'
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
PY
# Aravis 0.6
python3 -c "import gi; gi.require_version('Aravis','0.6'); from gi.repository import Aravis; print('Aravis OK')"
# Cámaras
python3 -c "import gi; gi.require_version('Aravis','0.6'); from gi.repository import Aravis; Aravis.update_device_list(); print('Cámaras:', Aravis.get_n_devices())"
```

## 📁 Archivos relevantes
- `PruebaAravis.py`: aplicación principal
- `config_yolo.yaml`: configuración de modelo/umbrales
- `requirements.jetson.txt`: dependencias pip (sin OpenCV)
- `diagnostico_jetpack511.py`: diagnóstico del sistema

## 🐛 Problemas frecuentes
- Sin cámaras: verifica conexión y `Aravis.get_n_devices()`.
- Pocos FPS: baja resolución/YOLO imgsz; ajusta exposición/ganancia.
- Detecciones inestables: sube IOU o Confidence; usa modo CONSERVADOR en clasificador.

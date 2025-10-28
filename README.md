# 🎯 Sistema de Visión Industrial para Jetson (YOLO + Aravis)

Sistema de visión en tiempo real para Jetson Orin con detección YOLO, cámaras GenICam vía Aravis, logging industrial y autoarranque en fábrica.

## 🚀 Características Principales

- **🤖 Detección YOLO v8** con modelo personalizado
- **📹 Soporte Aravis** para cámaras GenICam (USB/GigE)
- **🖥️ Interfaz gráfica** con OpenCV
- **⚙️ Control GPIO** para hardware externo
- **📹 Grabación de video** con ffmpeg
- **📊 Análisis de rendimiento** en tiempo real
- **🚀 Optimizado para Jetson** con CPU optimizado (ARM64)


## 🖥️ Plataforma validada

- JetPack 5.1.1 (L4T R35.3.x) • CUDA 11.4 • cuDNN 8.6 • TensorRT 8.5.2
- Aravis 0.6 (paquetes del sistema)
- PyTorch 2.0.0+nv23.05 (Jetson) • TorchVision 0.15.x
- OpenCV del sistema (python3-opencv 4.2)

## 🎯 Aplicaciones Industriales

### **Sistemas de Detección en Tiempo Real**
- **Control de calidad** en líneas de producción
- **Detección de objetos** en entornos industriales
- **Inspección automatizada** con cámaras GenICam
- **Sistemas de visión** para manufactura

### **Características Técnicas**
- **Detección YOLO v8** optimizada para CPU
- **Soporte universal** para cámaras GenICam
- **Interfaz gráfica** profesional
- **Control GPIO** para hardware externo
- **Grabación de video** integrada
- **Análisis de rendimiento** en tiempo real

## 📦 Instalación (resumen)

Sigue la guía completa: `GUIA_INSTALACION_FABRICA.md`.

Orden recomendado:
```bash
sudo /home/nvidia/Desktop/Calippo_jetson/base_setup_system.sh
/home/nvidia/Desktop/Calippo_jetson/install_pytorch_jetson.sh
sudo /home/nvidia/Desktop/Calippo_jetson/install_aravis.sh   # si faltara Aravis 0.6
/home/nvidia/Desktop/Calippo_jetson/install_calippo_factory.sh
/home/nvidia/Desktop/Calippo_jetson/verify_calippo_installation.sh
```

## 📁 Estructura

```
Calippo_jetson/
├── 🎯 gentl/                    # App principal YOLO + Aravis + logging
│   ├── PruebaAravis.py          # Script principal
│   ├── config_yolo.yaml         # Configuración YOLO
│   ├── requirements.jetson.txt  # Requisitos pip (sin OpenCV)
│   ├── diagnostico_jetpack511.py# Diagnóstico del sistema
│   ├── config_yolo.yaml         # Configuración de modelo YOLO
│   └── README.md                # Descripción de flujos y modelos
└── 📋 README.md                 # Este archivo
```

## 🎮 Uso

### Modo UI (pruebas locales)
```bash
# Asegúrate de detener el servicio para liberar la cámara
sudo systemctl stop vision-app.service

# Lanza con UI (HEADLESS desactivado)
export HEADLESS=0
python /home/nvidia/Desktop/Calippo_jetson/gentl/PruebaAravis.py
```

### Modo continuo (fábrica)
```bash
# Arranca el servicio en headless y déjalo habilitado
sudo systemctl start vision-app.service
sudo systemctl enable vision-app.service

# Verificación que funciona y loggea
systemctl status vision-app.service --no-pager
sudo journalctl -u vision-app.service -f
tail -f /var/log/calippo/system/calippo_jetson.log
tail -f /var/log/calippo/system/calippo_jetson_metrics.log
tail -f /var/log/calippo/vision/vision_log.csv

# Prueba de reinicio (opcional)
sudo reboot
# luego de 2-3 min:
systemctl is-active vision-app.service
```

### **Controles de la interfaz**
- **RUN/STOP**: Iniciar/parar detección
- **REC**: Iniciar/parar grabación
- **Confianza**: Ajustar umbral de detección (0.1-0.9)
- **IOU**: Ajustar umbral de solapamiento (0.1-0.9)
- **Track Buffer**: Ajustar buffer de seguimiento

### **Configuración**
Editar `config_yolo.yaml` para personalizar:
- Modelo YOLO
- Clases de detección
- Parámetros de confianza
- Configuración de seguimiento

## 🔧 Funcionalidades del Sistema

### **🤖 Detección YOLO v8**
- **Modelo personalizado** entrenado para detección específica
- **Clases:** 'can' (lata) y 'hand' (mano)
- **Optimizado para CPU** en Jetson Orin
- **Inferencia en tiempo real** con tracking persistente

### **📹 Cámaras GenICam (Aravis)**
- **Soporte universal** para cámaras GenICam
- **USB y GigE** compatible
- **Configuración automática** de parámetros
- **Control de exposición** y balance de blancos

### **🖥️ Interfaz Gráfica**
- **Vista en tiempo real** con overlays de detección
- **Controles intuitivos** para ajustar parámetros
- **Métricas de rendimiento** en tiempo real
- **Grabación de video** integrada

### **⚙️ Control GPIO**
- **Control de hardware externo** (Jetson GPIO)
- **Señales de control** para sistemas industriales
- **Integración** con sistemas de producción

### **📊 Análisis de Rendimiento**
- **FPS en tiempo real**
- **Latencia de detección**
- **Estadísticas de tracking**
- **Métricas de sistema**


### **Rendimiento lento**
- Verificar que PyTorch esté optimizado
- Reducir resolución de cámara
- Usar modelo YOLO más pequeño

### **Verificación del sistema**
```bash
# Ejecutar diagnóstico completo
python3 diagnostico_completo.py

# Verificar replicación
python3 verificar_replicacion.py
```


## 🔄 Sistema de Autoarranque y Logging

### Autoarranque Industrial
- **Servicio systemd**: `vision-app.service` se ejecuta automáticamente al arrancar el sistema
- **Modo headless**: Sin interfaz gráfica, optimizado para fábrica
- **Watchdog**: Reinicio automático si la aplicación se cuelga
- **Persistencia**: Sobrevive a reinicios y cortes de energía

### Sistema de Logging (4 categorías)
- **System**: Eventos del sistema, métricas de rendimiento, errores críticos
- **Digital**: Salidas digitales, comunicación PLC, señales de control
- **Photos**: Snapshots periódicos, imágenes de defectos detectados
- **Vision**: Logs detallados por lata procesada (CSV/JSONL)

### Niveles de Logging
Cada categoría soporta niveles: `debug`, `info`, `warning`, `error`, `critical`
- **Rotación automática**: Logs se comprimen diariamente
- **Retención**: 30 días de historial
- **Ubicación**: `/var/log/calippo/` organizados por categoría

## 📚 Documentación Adicional

- **`gentl/README.md`**: Flujos y modelos de la aplicación
- **`GUIA_INSTALACION_FABRICA.md`**: Guía completa de instalación paso a paso
- **`SYSTEM_REFERENCE.md`**: Referencia técnica completa (versiones, rutas, comandos)
- **`gentl/diagnostico_jetpack511.py`**: Diagnóstico del sistema (ejecutar para verificar)

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request


# 🎯 Sistema de Visión Industrial para Jetson (YOLO + Aravis)

Sistema de visión en tiempo real para Jetson Orin con detección YOLO, cámaras GenICam vía Aravis, logging industrial y autoarranque en fábrica.

> **📚 Documentación**: 
> - **`vision_app/README.md`**: Documentación técnica detallada (arquitectura, módulos, API)
> - **`GUIA_INSTALACION_FABRICA.md`**: Guía completa de instalación paso a paso

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

### Scripts de Instalación

1. **`install_base_setup_system.sh`** (requiere sudo)
   - Instala CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2
   - Configura dependencias del sistema
   - Habilita variables de entorno

2. **`install_pytorch_jetson.sh`** (usuario normal)
   - Instala PyTorch 2.0.0+nv23.05 y TorchVision en `.venv`
   - Configura entorno virtual del proyecto

3. **`install_aravis.sh`** (requiere sudo, opcional)
   - Instala Aravis 0.6 (paquetes del sistema)
   - Solo necesario si Aravis no está instalado

4. **`install_vision_factory.sh`** (usuario normal)
   - Crea servicio systemd `vision-app.service`
   - Configura directorios de logs con permisos
   - Habilita autoarranque

5. **`verify_vision_installation.sh`** (usuario normal)
   - Verifica instalación completa
   - Comprueba servicio, logs y proceso

### Orden de Ejecución Recomendado

```bash
# 1. Base del sistema (CUDA, cuDNN, TensorRT)
sudo /home/nvidia/Desktop/Vision_App/install_base_setup_system.sh

# 2. PyTorch en venv
/home/nvidia/Desktop/Vision_App/install_pytorch_jetson.sh

# 3. Aravis (solo si falta)
sudo /home/nvidia/Desktop/Vision_App/install_aravis.sh

# 4. Servicio systemd y logs
/home/nvidia/Desktop/Vision_App/install_vision_factory.sh

# 5. Verificación
/home/nvidia/Desktop/Vision_App/verify_vision_installation.sh
```

## 📁 Estructura

```
Vision_App/
├── 🎯 vision_app/                # App principal YOLO + Aravis + logging
│   ├── app.py                   # Orquestador principal
│   ├── main.py                  # Punto de entrada (systemd)
│   ├── config_yolo.yaml         # Configuración YOLO
│   ├── model/                   # Modelos ML (detección, clasificación, tracking)
│   ├── core/                    # Módulos centrales (logging, settings, optimizations, recording)
│   ├── camera/                  # Backends de cámara (GenICam/Aravis, ONVIF/RTSP)
│   └── developer_ui/            # Interfaz de depuración (OpenCV)
├── install_base_setup_system.sh # Setup SO base (CUDA, cuDNN, TensorRT)
├── install_pytorch_jetson.sh    # PyTorch en venv
├── install_aravis.sh            # Instalación Aravis 0.6
├── install_vision_factory.sh    # Servicio systemd + logs
├── run_debug.sh                 # Script para ejecutar en modo debug (con UI)
├── verify_vision_installation.sh # Verificación post-instalación
├── README.md                    # Este archivo (visión general)
├── GUIA_INSTALACION_FABRICA.md  # Guía completa de instalación
└── vision_app/README.md         # Documentación técnica detallada
```

## 🎮 Uso

### Modo Debug (con UI - pruebas locales)
```bash
# Opción 1: Script automático (recomendado)
cd /home/nvidia/Desktop/Vision_App
./run_debug.sh

# Opción 2: Manual
sudo systemctl stop vision-app.service  # Detener servicio para liberar cámara
cd /home/nvidia/Desktop/Vision_App
source vision_app/.venv/bin/activate
python main.py
```

**Nota:** El script `run_debug.sh` detiene automáticamente el servicio systemd, activa el entorno virtual y ejecuta la aplicación con UI habilitada.

### Modo Continuo (fábrica - headless)
```bash
# Iniciar el servicio (se auto-arranca al encender el equipo)
sudo systemctl start vision-app.service

# Verificar estado
systemctl status vision-app.service

# El servicio ya está habilitado para auto-arranque (se configuró con install_vision_factory.sh)

# Verificación que funciona y loggea
systemctl status --no-pager vision-app
sudo journalctl -u vision-app -f --no-pager

# Logs por dominio (journal)
sudo journalctl -u vision-app --no-pager | grep " vision_app:"
sudo journalctl -u vision-app --no-pager | grep " vision:"
sudo journalctl -u vision-app --no-pager | grep " images:"
sudo journalctl -u vision-app --no-pager | grep " io:"

# Logs en ficheros (si LOG_TO_FILE=1)
tail -f /var/log/vision_app/system/system.log
tail -f /var/log/vision_app/vision/vision_log.csv
tail -f /var/log/vision_app/images/$(date +%F)/images.csv
tail -f /var/log/vision_app/timings/timings_log.csv

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


### **Troubleshooting**

**Rendimiento lento:**
- Verificar que PyTorch esté optimizado
- Reducir resolución de cámara
- Usar modelo YOLO más pequeño

**Cámara no detectada:**
- Verificar permisos: `ls -l /dev/video*`
- Revisar logs: `sudo journalctl -u vision-app.service -n 50`
- Verificar que el servicio esté detenido si se usa UI: `sudo systemctl stop vision-app.service`

**Más información:** Consulta `vision_app/README.md` para troubleshooting detallado y arquitectura completa.


## 🔄 Sistema de Autoarranque y Logging

### Autoarranque Industrial
- **Servicio systemd**: `vision-app.service` se ejecuta automáticamente al arrancar el sistema
- **Modo headless**: Sin interfaz gráfica, optimizado para fábrica
- **Watchdog**: Reinicio automático si la aplicación se cuelga
- **Persistencia**: Sobrevive a reinicios y cortes de energía

### Sistema de Logging (5 categorías)
- **system**: estado/arranque de la app
- **vision**: eventos de visión por lata (además de `vision_log.csv`)
- **images**: guardado de imágenes (CSV diario + JPGs)
- **io**: I/O/PLC (cuando exista hardware)
- **timings**: latencias por etapa (complementa `timings_log.csv`)

### Niveles de Logging
Cada categoría soporta niveles: `debug`, `info`, `warning`, `error`, `critical`
- **Rotación automática**: Logs se comprimen diariamente
- **Retención**: 30 días de historial
- **Ubicación**: `/var/log/vision_app/` organizados por categoría

## 📚 Documentación Adicional

- **`vision_app/README.md`**: Documentación técnica detallada de la aplicación (arquitectura, módulos, API)
- **`GUIA_INSTALACION_FABRICA.md`**: Guía completa de instalación paso a paso para fábrica

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request


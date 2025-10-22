# 🎯 Sistema de Detección YOLO + Aravis para Jetson

Sistema completo de **detección de objetos en tiempo real** usando **YOLO v8** y cámaras **GenICam (Aravis)** optimizado para **Jetson Orin**. Incluye interfaz gráfica, control GPIO, grabación de video y análisis de rendimiento.

## 🚀 Características Principales

- **🤖 Detección YOLO v8** con modelo personalizado
- **📹 Soporte Aravis** para cámaras GenICam (USB/GigE)
- **🖥️ Interfaz gráfica** con OpenCV
- **⚙️ Control GPIO** para hardware externo
- **📹 Grabación de video** con ffmpeg
- **📊 Análisis de rendimiento** en tiempo real
- **🚀 Optimizado para Jetson** con CPU optimizado (ARM64)


## 🖥️ Requisitos del Sistema

### **Hardware**
- **Jetson Orin** (ARM64) o PC compatible
- **Ubuntu 22.04 LTS** (recomendado)
- **Cámara GenICam** (USB o GigE)
- **Memoria:** Mínimo 8GB RAM
- **Almacenamiento:** 20GB libres

### **Software**
- **Python 3.10+**
- **PyTorch 2.0.1+** (CPU optimizado)
- **OpenCV 4.12.0+**
- **Aravis 0.8+**
- **Ultralytics YOLO 8.3.207+**

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

## 📦 Instalación

### **1. Clonar el repositorio**
```bash
git clone <repository-url>
cd Calippo_jetson/gentl
```

### **2. Instalación automática (Recomendado)**
```bash
chmod +x install_aravis_yolo.sh
./install_aravis_yolo.sh
```

### **3. Instalación manual**
```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias del sistema
sudo apt update
sudo apt install -y python3-pip libaravis-dev python3-gi python3-gi-cairo gir1.2-aravis-0.8

# Instalar PyTorch para ARM64 (CPU optimizado)
python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# Instalar dependencias de Python
python3 -m pip install -r requirements.txt
```

## 📁 Estructura del Proyecto

```
Calippo_jetson/
├── 🎯 gentl/                    # Sistema principal de detección YOLO + Aravis
│   ├── prueba.py                # Script principal del sistema
│   ├── config_yolo.yaml         # Configuración YOLO
│   ├── requirements.txt         # Dependencias con versiones específicas
│   ├── install_aravis_yolo.sh   # Instalación automática
│   ├── diagnostico_completo.py  # Diagnóstico completo del sistema
│   ├── verificar_replicacion.py # Verificación de replicación
│   ├── v2_yolov8n_HERMASA_finetune.pt # Modelo YOLO personalizado
│   ├── README.md                # Documentación del sistema
│   ├── INSTALACION_COMPLETA.md  # Guía detallada de instalación
│   ├── RESUMEN_VERSIONES.md     # Resumen de versiones
│   ├── REPLICACION_COMPLETA.md  # Guía de replicación
│   └── vista_gentl_yolo.py      # Código de referencia (opcional)
├── 📹 stapi/                    # Sistema anterior (StApi) - DEPRECADO
└── 📋 README.md                 # Este archivo principal
```

## 🎮 Uso del Sistema

### **Ejecutar el sistema principal**
```bash
# Activar entorno virtual
source .venv/bin/activate

# Ejecutar sistema principal
python3 prueba.py
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

## 📦 Dependencias del Sistema

### **Dependencias Principales**
```
# Core dependencies
numpy==1.24.3
opencv-python-headless==4.12.0.88
torch==2.0.1
torchvision==0.15.2
torchaudio==2.0.2
ultralytics==8.3.207

# Camera and hardware
PyGObject==3.42.1
Jetson.GPIO==2.1.7

# Utilities
PyYAML==5.4.1
psutil==7.1.0
pillow==11.0.0
matplotlib==3.5.1
scipy==1.8.0
pandas==1.3.5

# Optional: ONNX Runtime (para futuras optimizaciones)
onnxruntime==1.23.1
```

### **Dependencias del Sistema (Ubuntu)**
```bash
sudo apt install -y \
    python3-pip \
    libaravis-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-aravis-0.8 \
    build-essential \
    cmake \
    pkg-config
```

## 🐛 Solución de Problemas

### **Error: "No cameras found (Aravis)"**
- **Causa**: No hay cámara conectada
- **Solución**: Conectar cámara GenICam USB o GigE

### **Error: "ModuleNotFoundError: torch"**
- **Causa**: PyTorch no instalado
- **Solución**: Ejecutar instalación automática
```bash
./install_aravis_yolo.sh
```

### **Error: "Numpy is not available"**
- **Causa**: Incompatibilidad de NumPy
- **Solución**: Reinstalar NumPy compatible
```bash
python3 -m pip uninstall numpy
python3 -m pip install numpy==1.24.3
```

### **Error: "libcudnn.so.8 not found" (Jetson)**
- **Causa**: Incompatibilidad de CuDNN
- **Solución**: Crear enlace simbólico
```bash
sudo ln -s /usr/lib/aarch64-linux-gnu/libcudnn.so.9.3.0 /usr/lib/aarch64-linux-gnu/libcudnn.so.8
```

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


## 📊 Rendimiento

### **Especificaciones de prueba**
- **Jetson Orin**: 7.44GB RAM
- **Cámara**: USB3 GenICam 1920x1080@30fps
- **Modelo**: YOLOv8n personalizado
- **FPS**: 15-20 fps en detección (CPU)
- **Latencia**: <100ms

### **Optimizaciones aplicadas**
- PyTorch optimizado para CPU
- OpenCV optimizado
- Pipeline asíncrono
- Buffer de seguimiento eficiente
- Memoria gestionada

## 📚 Documentación Adicional

- **`gentl/README.md`**: Documentación detallada del sistema
- **`gentl/INSTALACION_COMPLETA.md`**: Guía detallada de instalación
- **`gentl/RESUMEN_VERSIONES.md`**: Resumen de versiones instaladas
- **`gentl/REPLICACION_COMPLETA.md`**: Guía de replicación del sistema
- **Comentarios en código**: Explicaciones detalladas de cada función

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 🔄 Historial de Versiones

### **v2.1.0 (Actual)**
- Migración de GenTL a Aravis
- Optimización para CPU en Jetson Orin
- YOLO v8 actualizado
- Interfaz mejorada
- Diagnóstico completo del sistema
- PyTorch optimizado para ARM64

### **v1.0.0 (Anterior)**
- Implementación inicial con GenTL
- YOLO v5
- Soporte básico Jetson

---

**Desarrollado para Jetson Orin con Aravis y YOLO v8** 🚀

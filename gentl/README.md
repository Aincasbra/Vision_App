# 🎯 Sistema de Detección YOLO + Aravis para Jetson

## 📋 Descripción

Sistema completo de detección de objetos en tiempo real usando YOLO v8 y cámaras GenICam (Aravis) optimizado para Jetson Orin. Incluye interfaz gráfica, control GPIO, grabación de video y análisis de rendimiento.

## 🚀 Características

- **Detección YOLO v8** con modelo personalizado
- **Soporte Aravis** para cámaras GenICam (USB/GigE)
- **Interfaz gráfica** con OpenCV
- **Control GPIO** para hardware externo
- **Grabación de video** con ffmpeg
- **Análisis de rendimiento** en tiempo real
- **Optimizado para CPU** en Jetson Orin (ARM64)

## 🖥️ Requisitos del Sistema

### Hardware
- **Jetson Orin** (ARM64)
- **Ubuntu 22.04 LTS**
- **Cámara GenICam** (USB o GigE)
- **Memoria:** Mínimo 8GB RAM
- **Almacenamiento:** 20GB libres

### Software
- **Python 3.10+**
- **PyTorch 2.0.1+** (CPU optimizado)
- **OpenCV 4.12.0+**
- **Aravis 0.8+**
- **Ultralytics YOLO 8.3.207+**

## 📦 Instalación

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd Calippo_jetson/gentl
```

### 2. Instalación automática (Recomendado)
```bash
chmod +x install_aravis_yolo.sh
./install_aravis_yolo.sh
```

### 3. Instalación manual
```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias del sistema
sudo apt update
sudo apt install -y python3-pip libaravis-dev python3-gi python3-gi-cairo gir1.2-aravis-0.8

# Instalar PyTorch para ARM64 (CPU optimizado)
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar dependencias de Python
python3 -m pip install -r requirements_aravis_arm64.txt
```

## 🎮 Uso

### Ejecutar el sistema
```bash
# Activar entorno virtual
source .venv/bin/activate

# Ejecutar sistema principal
python3 prueba.py
```

### Controles de la interfaz
- **RUN/STOP**: Iniciar/parar detección
- **REC**: Iniciar/parar grabación
- **Confianza**: Ajustar umbral de detección
- **IOU**: Ajustar umbral de solapamiento
- **Track Buffer**: Ajustar buffer de seguimiento

### Configuración
Editar `config_yolo.yaml` para personalizar:
- Modelo YOLO
- Clases de detección
- Parámetros de confianza
- Configuración de seguimiento

## 📁 Estructura del Proyecto

```
gentl/
├── prueba.py                    # Script principal
├── vista_gentl_yolo.py         # Referencia de implementación
├── config_yolo.yaml            # Configuración YOLO
├── requirements.txt            # Dependencias con versiones específicas
├── install_aravis_yolo.sh      # Instalación automática
├── diagnostico_completo.py     # Diagnóstico completo del sistema
├── verificar_replicacion.py    # Verificación de replicación
├── README.md                   # Documentación principal
├── INSTALACION_COMPLETA.md     # Guía detallada de instalación
├── RESUMEN_VERSIONES.md        # Resumen de versiones
├── REPLICACION_COMPLETA.md     # Guía de replicación
├── v2_yolov8n_HERMASA_finetune.pt # Modelo YOLO entrenado
└── diagnostico_resultados.json # Resultados del diagnóstico (JSON)
```

## 🔧 Configuración Avanzada

### Modelo YOLO Personalizado
1. Entrenar modelo con `ultralytics`
2. Guardar como `.pt`
3. Actualizar `config_yolo.yaml`
4. Reiniciar sistema

### Cámaras GenICam
- **USB**: Conectar y ejecutar
- **GigE**: Configurar IP estática
- **Múltiples**: Cambiar `index` en `AravisBackend`

### Optimización de Rendimiento
- Ajustar resolución de cámara
- Modificar tamaño de modelo YOLO
- Optimizar para CPU (ARM64)
- Ajustar parámetros de seguimiento

## 🐛 Solución de Problemas

### Error: "No cameras found (Aravis)"
- **Causa**: No hay cámara conectada
- **Solución**: Conectar cámara GenICam

### Error: "ModuleNotFoundError: torch"
- **Causa**: PyTorch no instalado
- **Solución**: Ejecutar instalación automática

### Error: "Numpy is not available"
- **Causa**: Incompatibilidad de NumPy
- **Solución**: Reinstalar NumPy compatible: `pip install numpy==1.24.3`

### Rendimiento lento
- Verificar que PyTorch esté optimizado
- Reducir resolución de cámara
- Usar modelo YOLO más pequeño

## 📊 Diagnóstico del Sistema

### Ejecutar diagnóstico completo
```bash
python3 diagnostico_completo.py
```

El diagnóstico genera dos archivos:
- `diagnostico_resultados.json` - Datos completos en formato JSON
- `RESUMEN_VERSIONES.md` - Resumen legible de versiones y estado

### Verificar que el sistema esté listo para replicación
```bash
python3 verificar_replicacion.py
```

### Verificar componentes individuales
```bash
# Verificar PyTorch
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Verificar Aravis
python3 -c "import gi; gi.require_version('Aravis', '0.8'); from gi.repository import Aravis; print('Aravis:', Aravis.get_version())"

# Verificar cámaras
python3 -c "import gi; gi.require_version('Aravis', '0.8'); from gi.repository import Aravis; Aravis.update_device_list(); print('Cámaras:', Aravis.get_n_devices())"
```

## 📈 Rendimiento

### Especificaciones de prueba
- **Jetson Orin**: 7.44GB RAM
- **Cámara**: USB3 GenICam 1920x1080@30fps
- **Modelo**: YOLOv8n personalizado
- **FPS**: 15-20 fps en detección (CPU)
- **Latencia**: <100ms

### Optimizaciones aplicadas
- PyTorch optimizado para CPU
- OpenCV optimizado
- Pipeline asíncrono
- Buffer de seguimiento eficiente
- Memoria gestionada

## 🤝 Contribución

1. Fork del repositorio
2. Crear rama de feature
3. Commit de cambios
4. Push a la rama
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 📞 Soporte

Para soporte técnico o preguntas:
- Crear issue en GitHub
- Revisar `ESTADO_ACTUAL.md`
- Ejecutar `diagnostico.py`

## 🔄 Historial de Versiones

### v2.1.0 (Actual)
- Migración de GenTL a Aravis
- Optimización para CPU en Jetson Orin
- YOLO v8 actualizado
- Interfaz mejorada
- Diagnóstico completo del sistema
- PyTorch optimizado para ARM64

### v1.0.0 (Anterior)
- Implementación inicial con GenTL
- YOLO v5
- Soporte básico Jetson

---

**Desarrollado para Jetson Orin con Aravis y YOLO v8** 🚀

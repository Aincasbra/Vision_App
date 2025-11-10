# 🚀 GUÍA DE INSTALACIÓN VISION APP PARA FÁBRICAAP

## 📋 INSTRUCCIONES PASO A PASO

### PASO 0: Preparar el sistema Jetson (CUDA/cuDNN/TensorRT, PyTorch y Aravis)

1. Base del sistema (CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2, deps, entorno):
   ```bash
   sudo /home/nvidia/Desktop/Vision_App/install_base_setup_system.sh
   ```

2. PyTorch/TorchVision en la venv del proyecto (usa wheel local si lo tienes):
   ```bash
   /home/nvidia/Desktop/Vision_App/install_pytorch_jetson.sh
   # Si tienes el wheel local de torch:
   # pip install /home/nvidia/tmp_jp/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
   # Y para torchvision: pip install /home/nvidia/tvsrc  (o el wheel compatible)
   ```

3. Aravis 0.6 (paquetes del sistema):
   ```bash
   sudo apt install -y gir1.2-aravis-0.6 libaravis-0.6-0 aravis-tools
   ```

### **PASO 1: PREPARACIÓN DEL EQUIPO DE FÁBRICA**

1. **Conectar al equipo de fábrica (Jetson Orin NX/AGX)**
   ```bash
   # SSH o acceso directo
   ssh nvidia@<IP_DEL_EQUIPO>
   ```

2. **Verificar que estás en el directorio correcto**
   ```bash
   cd /home/nvidia/Desktop/Vision_App
   pwd  # Debe mostrar: /home/nvidia/Desktop/Vision_App
   ```

3. **Verificar que tienes los archivos necesarios**
   ```bash
   ls -la vision_app/app.py
   ls -la install_vision_factory.sh
   ls -la verify_vision_installation.sh
   ```

### **PASO 2: EJECUTAR INSTALACIÓN AUTOMÁTICA**

1. **Ejecutar el script de instalación**
   ```bash
   ./install_vision_factory.sh
   ```

2. **El script hará automáticamente:**
   - ✅ Verificar usuario y directorio
   - ✅ Crear directorios de logs con permisos correctos
   - ✅ Crear servicio systemd `vision-app.service` para autoarranque
   - ✅ Configurar permisos del script launcher
   - ✅ Verificar toda la instalación
   - ✅ Probar el servicio

### **PASO 3: VERIFICAR INSTALACIÓN**

1. **Ejecutar script de verificación**
   ```bash
   ./verify_vision_installation.sh
   ```

2. **El script verificará:**
   - ✅ Servicio systemd configurado y habilitado
   - ✅ Directorios de logs creados con permisos correctos
   - ✅ Proceso Vision App ejecutándose
   - ✅ Logs actualizándose en tiempo real
   - ✅ Espacio en disco suficiente

### **PASO 4: PRUEBA DE AUTOARRANQUE**

1. **Reiniciar el equipo para probar autoarranque**
   ```bash
   sudo reboot
   ```

2. **Después del reinicio, verificar que funciona**
   ```bash
   # Esperar 2-3 minutos para que arranque completamente
   systemctl status vision-app.service
   ```

3. **Verificar logs**
   ```bash
   sudo journalctl -u vision-app -f --no-pager
   ```

## 🔧 COMANDOS ÚTILES PARA MONITOREO

### **Estado del Servicio**
```bash
systemctl status vision-app.service          # Estado general
systemctl is-active vision-app.service       # Solo si está activo
systemctl is-enabled vision-app.service      # Solo si está habilitado
```

### **Control del Servicio**
```bash
sudo systemctl start vision-app.service      # Iniciar servicio
sudo systemctl stop vision-app.service       # Detener servicio
sudo systemctl restart vision-app.service    # Reiniciar servicio
sudo systemctl reload vision-app.service     # Recargar configuración
```

### **Logs en Tiempo Real**
```bash
sudo journalctl -u vision-app -f --no-pager
# Filtrado por dominios
sudo journalctl -u vision-app --no-pager | grep " vision_app:"
sudo journalctl -u vision-app --no-pager | grep " vision:"
sudo journalctl -u vision-app --no-pager | grep " images:"
sudo journalctl -u vision-app --no-pager | grep " io:"
# Ficheros si LOG_TO_FILE=1
tail -f /var/log/vision_app/system/system.log
tail -f /var/log/vision_app/vision/vision_log.csv
tail -f /var/log/vision_app/images/$(date +%F)/images.csv
tail -f /var/log/vision_app/timings/timings_log.csv
```

### **Logs Históricos**
```bash
sudo journalctl -u vision-app --since "1 hour ago"     # Última hora
sudo journalctl -u vision-app --since "2025-01-01"     # Desde fecha específica
sudo journalctl -u vision-app -n 100                    # Últimas 100 líneas
```

### **Verificar Proceso**
```bash
PID=$(systemctl show -p MainPID --value vision-app); ps -fp "$PID"
top -b -n1 -p "$PID"
```

### **Espacio en Disco**
```bash
df -h /var/log/vision_app                  # Espacio usado por logs
du -sh /var/log/vision_app/*               # Tamaño por directorio
```

## 📁 ESTRUCTURA DE LOGS GENERADOS

```
/var/log/vision_app/
├── system/
│   └── system.log                    # Logs de sistema (si LOG_TO_FILE=1)
├── io/
│   └── io.log                        # Logs IO (cuando haya hardware)
├── vision/
│   └── vision_log.csv                # Por detección (CSV)
├── timings/
│   └── timings_log.csv               # Latencias por etapa (CSV)
├── images/
│   └── YYYY-MM-DD/
│       ├── images.csv                # CSV de imágenes guardadas
│       └── *.jpg                     # bad/good
└── archive/                          # ZIPs diarios de imágenes
```

## ⚠️ SOLUCIÓN DE PROBLEMAS

### **Servicio no arranca**
```bash
sudo journalctl -u vision-app.service -n 50    # Ver últimos errores
sudo systemctl daemon-reload                   # Recargar configuración
sudo systemctl restart vision-app.service      # Reiniciar servicio
```

### **Logs no se generan**
```bash
ls -la /var/log/vision_app/                    # Verificar permisos
systemctl show vision-app -p Environment    # Ver variables LOG_*
```

### **Proceso no ejecutándose**
```bash
PID=$(systemctl show -p MainPID --value vision-app); ps -fp "$PID"
sudo systemctl start vision-app.service
```

### **Espacio en disco lleno**
```bash
df -h                                       # Verificar espacio
sudo du -sh /var/log/vision_app/*             # Ver tamaño de logs
```

## 🎯 VERIFICACIÓN FINAL

Después de la instalación, debe cumplirse:

1. ✅ **Autoarranque**: El servicio inicia automáticamente al reiniciar
2. ✅ **Ejecución continua**: El proceso corre sin intervención
3. ✅ **Logs activos**: Se generan logs en tiempo real
4. ✅ **Reinicio automático**: Si falla, se reinicia automáticamente
5. ✅ **Rotación de logs**: (si se configura a futuro)
6. ✅ **Modo headless**: Funciona sin interfaz gráfica

## 🧭 RESUMEN DE SCRIPTS Y CUÁNDO USARLOS

- **install_base_setup_system.sh** (root): instala/asegura CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2, OpenCV del sistema, dependencias y variables de entorno; habilita `logrotate.timer`.
  - Uso:
    ```bash
    sudo /home/nvidia/Desktop/Vision_App/install_base_setup_system.sh
    ```

- **install_pytorch_jetson.sh** (usuario normal): instala PyTorch 2.0.0+nv23.05 y torchvision compatibles en la `.venv` del proyecto.
  - Uso:
    ```bash
    /home/nvidia/Desktop/Vision_App/install_pytorch_jetson.sh
    ```

- **install_aravis.sh** (root): instala Aravis 0.6 por paquetes.
  - Uso:
    ```bash
    sudo /home/nvidia/Desktop/Vision_App/install_aravis.sh
    ```

- **install_vision_factory.sh** (usuario normal): configura autoarranque (`systemd`), directorios/permisos de logs, activa `LOG_*` y prueba el servicio.
  - Uso:
    ```bash
    /home/nvidia/Desktop/Vision_App/install_vision_factory.sh
    ```

- **verify_vision_installation.sh** (usuario normal): verificaciones post-instalación (servicio, logs, espacio, proceso en ejecución).
  - Uso:
    ```bash
    /home/nvidia/Desktop/Vision_App/verify_vision_installation.sh
    ```

- **run_vision_app.sh** (no ejecutar manualmente en producción): lanzador local para debug.

### Orden recomendado (equipo de fábrica, JetPack 5.1.1 limpio)
1. CUDA/cuDNN/TensorRT y deps del SO:
   ```bash
   sudo /home/nvidia/Desktop/Vision_App/install_base_setup_system.sh
   ```
2. PyTorch en la `.venv` del proyecto:
   ```bash
   /home/nvidia/Desktop/Vision_App/install_pytorch_jetson.sh
   ```
3. Aravis 0.6 (paquetes del sistema):
   ```bash
   sudo apt install -y gir1.2-aravis-0.6 libaravis-0.6-0 aravis-tools
   ```
4. Autoarranque + logs:
   ```bash
   /home/nvidia/Desktop/Vision_App/install_vision_factory.sh
   /home/nvidia/Desktop/Vision_App/verify_vision_installation.sh
   ```

## 📎 Anexo: Referencia técnica (plataforma y versiones)

### Plataforma validada
- JetPack: 5.1.1 (L4T R35.3.x)
- Kernel: 5.10.104-tegra (aarch64)
- Ubuntu: 20.04 LTS
- Python: 3.8.x

### NVIDIA stack
- CUDA Toolkit: 11.4 (`/usr/local/cuda`)
- cuDNN: 8.6 (`libcudnn8{,-dev}`)
- TensorRT: 8.5 (`tensorrt{,-dev}`, `libnvinfer8`)

### Librerías de la app
- PyTorch: 2.0.0+nv23.05 (Jetson wheel)
- TorchVision: 0.15.x compatible con la anterior
- OpenCV (sistema): 4.2 (`python3-opencv`)
- Aravis: 0.6 (paquetes `gir1.2-aravis-0.6 libaravis-0.6-0 aravis-tools`)

### Dependencias (apt)
- build-essential cmake git wget curl unzip pkg-config
- python3 python3-pip python3-venv python3-dev
- libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev
- libgtk-3-dev libcanberra-gtk3-module
- gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good
- python3-opencv ffmpeg

### Entorno (sugerido en `~/.bashrc`)
- `CUDA_HOME=/usr/local/cuda`
- `PATH=$CUDA_HOME/bin:$PATH`
- `LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`
- Jetson perf (opcional): `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`

### Variables del servicio
- `HEADLESS=1`, `AUTO_RUN=1`
- `PYTHONPATH=/home/nvidia/Desktop/Vision_App/vision_app`
- `CONFIG_YOLO=/home/nvidia/Desktop/Vision_App/vision_app/config_yolo.yaml`
- `LOG_TO_SYSLOG=0|1`, `LOG_TO_FILE=1`, `LOG_DIR=/var/log/vision_app`

### Chequeos rápidos
```bash
nvcc --version
ldconfig -p | grep libcudnn
cd /home/nvidia/Desktop/Vision_App/vision_app && source .venv/bin/activate
python - <<'PY'
import torch, cv2
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('opencv', cv2.__version__)
PY
python -c "import gi; gi.require_version('Aravis','0.6'); from gi.repository import Aravis as A; A.update_device_list(); print('Cámaras:', A.get_n_devices())"
systemctl status vision-app.service --no-pager
```

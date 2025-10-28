## Sistema de visión Calippo Jetson - Referencia Técnica Completa

### 1) Plataforma y versiones validadas
- JetPack: 5.1.1 (L4T R35.3.x)
- Kernel: 5.10.104-tegra (aarch64)
- Ubuntu: 20.04.5/6 LTS (Focal)
- Python: 3.8.10
- CUDA Toolkit: 11.4 (/usr/local/cuda)
- cuDNN: 8.6.0 (libcudnn8, libcudnn8-dev)
- TensorRT: 8.5.2.2 (tensorrt, tensorrt-dev, libnvinfer8)
- PyTorch: 2.0.0+nv23.05 (Jetson wheel)
- TorchVision: 0.15.x compatible (wheel/local source)
- OpenCV (sistema): 4.2.0 (python3-opencv apt)
- Aravis: 0.6.0-3 (paquetes del sistema)

### 2) Estructura del proyecto
- Código app: `gentl/PruebaAravis.py`, `gentl/logging_system.py`
- Lanzador: `run_calippo.sh`
- Servicio systemd: `/etc/systemd/system/calippo.service` (opcionalmente renombrado a `vision-app.service`)
- Instaladores y utilidades:
  - `base_setup_system.sh`: CUDA/cuDNN/TensorRT/Deps/Entorno
  - `install_pytorch_jetson.sh`: Torch + TorchVision en `.venv`
  - `install_aravis.sh`: Aravis 0.8 (paquetes o build)
  - `install_calippo_factory.sh`: autoarranque, rsyslog, logrotate, permisos
  - `verify_calippo_installation.sh`: verificación integral
  - `GUIA_INSTALACION_FABRICA.md`: guía paso a paso

### 3) Rutas y variables clave
- Proyecto: `/home/nvidia/Desktop/Calippo_jetson`
- App: `/home/nvidia/Desktop/Calippo_jetson/gentl/PruebaAravis.py`
- venv: `/home/nvidia/Desktop/Calippo_jetson/gentl/.venv`
- Logs: `/var/log/calippo/{system,digital,photos,vision}`
- Syslog (rsyslog): `/etc/rsyslog.d/50-calippo.conf`
- Logrotate: `/etc/logrotate.d/calippo`
- Entorno (añadido a ~/.bashrc por instalación base):
  - `CUDA_HOME=/usr/local/cuda`
  - `PATH=$CUDA_HOME/bin:$PATH`
  - `LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`
  - PyTorch Jetson: `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`
  - Headless (en servicio): `HEADLESS=1`, `QT_QPA_PLATFORM=offscreen`

### 4) Dependencias del sistema (apt)
- Base: `build-essential cmake git wget curl unzip pkg-config`
- Python: `python3 python3-pip python3-venv python3-dev`
- Multimedia/Imagen: `libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev`
- GUI/GTK: `libgtk-3-dev libcanberra-gtk3-module`
- GStreamer: `gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good`
- GPIO: `libgpiod2 gpiod`
- OpenCV Python: `python3-opencv`
- NVIDIA stack (JetPack): `cuda-toolkit-11-4`, `libcudnn8{,-dev,-samples}`, `tensorrt{,-dev}`, `libnvinfer8{,-dev}`

### 5) Entorno Python (venv)
- Ruta: `gentl/.venv`
- Versión Torch validada: `2.0.0+nv23.05` (wheel Jetson)
- TorchVision: `0.15.1` aprox. (compatible con torch nv23.05)
- Paquetes relevantes: `ultralytics==8.3.219`, `onnxruntime==1.12.1`, `psutil`, `PyGObject` (si se usa gi), etc.

### 6) Aravis 0.6 (cámara)
- Preferente por paquetes: `gir1.2-aravis-0.6 libaravis-0.6-0 aravis-tools`
- Fallback desde fuente (meson+ninja) si no hay paquetes
- En el código: `gi.require_version('Aravis','0.6')`

### 7) Logging de producción
- Módulo: `gentl/logging_system.py`
- Categorías:
  - system: eventos y métricas → `/var/log/calippo/system/calippo_jetson*.log`
  - digital: I/O/PLC → `/var/log/calippo/digital/*.log`
  - photos: snapshots/defects → `/var/log/calippo/photos/...`
  - vision: por-lata CSV/JSONL → `/var/log/calippo/vision/vision_log.{csv,jsonl}`
- rsyslog (LOCAL0) → `/etc/rsyslog.d/50-calippo.conf`
- logrotate diario (gzip, 30 días) → `/etc/logrotate.d/calippo`
- `logrotate.timer` habilitado (persistente)

### 8) Servicio systemd (autoarranque)
- Nombre recomendado: `vision-app.service` (antes `calippo.service`)
- Unit típico (resumen):
  - `Type=notify`, `WatchdogSec=30`, `NotifyAccess=main`
  - `User=nvidia`, `WorkingDirectory=/home/nvidia/Desktop/Calippo_jetson`
  - `ExecStart=/home/nvidia/Desktop/Calippo_jetson/run_calippo.sh`
  - `Restart=always`, `RestartSec=5`
  - `Environment=HEADLESS=1`, `Environment=QT_QPA_PLATFORM=offscreen`
  - `StandardOutput=journal`, `StandardError=journal`

### 9) Operación (modos)
- UI local (pruebas):
  ```bash
  sudo systemctl stop vision-app.service  # liberar cámara
  export HEADLESS=0
  python /home/nvidia/Desktop/Calippo_jetson/gentl/PruebaAravis.py
  ```
- Headless continuo (fábrica):
  ```bash
  sudo systemctl enable --now vision-app.service
  systemctl status vision-app.service --no-pager
  ```

### 10) Verificación rápida
```bash
nvcc --version
ldconfig -p | grep libcudnn
python -c "import torch,cv2; print(torch.__version__, torch.cuda.is_available()); print(cv2.__version__)"
python -c "import gi; gi.require_version('Aravis','0.6'); from gi.repository import Aravis; print('Aravis OK')"
systemctl status vision-app.service --no-pager
tail -f /var/log/calippo/system/calippo_jetson.log
```

### 11) Renombrado del servicio (opcional)
```bash
sudo cp /etc/systemd/system/calippo.service /etc/systemd/system/vision-app.service
sudo sed -i 's/^Description=.*/Description=Vision App (PruebaAravis)/' /etc/systemd/system/vision-app.service
sudo systemctl daemon-reload
sudo systemctl enable --now vision-app.service
sudo systemctl disable calippo.service
```

### 12) Rutas de logs (si se quisiera renombrar)
- Por defecto: `/var/log/calippo/...`
- Para usar `/var/log/vision-app`:
  - Crear directorios y permisos
  - Reemplazar rutas en `/etc/rsyslog.d/50-calippo.conf` y `/etc/logrotate.d/calippo`
  - Reiniciar `rsyslog` y el servicio de la app

### 13) Seguridad (sudo sin contraseña opcional)
```bash
echo "nvidia ALL=(ALL) NOPASSWD: /usr/bin/apt, /usr/bin/systemctl, /usr/bin/ldconfig, /usr/bin/tee, /usr/bin/touch, /bin/mkdir, /bin/chown, /bin/chmod" | sudo tee /etc/sudoers.d/calippo-nopasswd
sudo chmod 440 /etc/sudoers.d/calippo-nopasswd
```

### 14) Contenedores (Jetson)
- Ver Dockerfile y docker-compose incluidos en este repositorio para ejecución con `--runtime=nvidia`, `network_mode: host`, `privileged: true` y volúmenes de logs.


# 🚀 GUÍA DE INSTALACIÓN CALIPPO PARA FÁBRICA

## 📋 INSTRUCCIONES PASO A PASO

### PASO 0: Preparar el sistema Jetson (CUDA/cuDNN/TensorRT, PyTorch y Aravis)

1. Base del sistema (CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2, deps, entorno):
   ```bash
   sudo /home/nvidia/Desktop/Calippo_jetson/base_setup_system.sh
   ```

2. PyTorch/TorchVision en la venv del proyecto (usa wheel local si lo tienes):
   ```bash
   /home/nvidia/Desktop/Calippo_jetson/install_pytorch_jetson.sh
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
   cd /home/nvidia/Desktop/Calippo_jetson
   pwd  # Debe mostrar: /home/nvidia/Desktop/Calippo_jetson
   ```

3. **Verificar que tienes los archivos necesarios**
   ```bash
   ls -la gentl/PruebaAravis.py
   ls -la gentl/logging_system.py
   ls -la run_calippo.sh
   ls -la install_calippo_factory.sh
   ls -la verify_calippo_installation.sh
   ```

### **PASO 2: EJECUTAR INSTALACIÓN AUTOMÁTICA**

1. **Ejecutar el script de instalación**
   ```bash
   ./install_calippo_factory.sh
   ```

2. **El script hará automáticamente:**
   - ✅ Verificar usuario y directorio
   - ✅ Instalar dependencias del sistema (logrotate, rsyslog)
   - ✅ Crear directorios de logs con permisos correctos
   - ✅ Configurar rsyslog para logs de aplicación
   - ✅ Configurar logrotate para rotación diaria
   - ✅ Crear servicio systemd para autoarranque
   - ✅ Configurar permisos del script launcher
   - ✅ Configurar cron job para logrotate
   - ✅ Verificar toda la instalación
   - ✅ Probar el servicio

### **PASO 3: VERIFICAR INSTALACIÓN**

1. **Ejecutar script de verificación**
   ```bash
   ./verify_calippo_installation.sh
   ```

2. **El script verificará:**
   - ✅ Servicio systemd configurado y habilitado
   - ✅ Directorios de logs creados con permisos correctos
   - ✅ Configuración de rsyslog funcionando
   - ✅ Configuración de logrotate instalada
   - ✅ Script launcher con permisos correctos
   - ✅ Proceso Calippo ejecutándose
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
   systemctl status calippo.service
   ```

3. **Verificar logs**
   ```bash
   tail -f /var/log/calippo/system/calippo_jetson.log
   ```

## 🔧 COMANDOS ÚTILES PARA MONITOREO

### **Estado del Servicio**
```bash
systemctl status calippo.service          # Estado general
systemctl is-active calippo.service       # Solo si está activo
systemctl is-enabled calippo.service      # Solo si está habilitado
```

### **Control del Servicio**
```bash
sudo systemctl start calippo.service      # Iniciar servicio
sudo systemctl stop calippo.service       # Detener servicio
sudo systemctl restart calippo.service    # Reiniciar servicio
sudo systemctl reload calippo.service     # Recargar configuración
```

### **Logs en Tiempo Real**
```bash
tail -f /var/log/calippo/system/calippo_jetson.log           # Logs principales
tail -f /var/log/calippo/system/calippo_jetson_metrics.log   # Métricas
tail -f /var/log/calippo/vision/vision_log.csv              # Logs de visión
sudo journalctl -u calippo.service -f                        # Logs del sistema
```

### **Logs Históricos**
```bash
sudo journalctl -u calippo.service --since "1 hour ago"     # Última hora
sudo journalctl -u calippo.service --since "2025-01-01"     # Desde fecha específica
sudo journalctl -u calippo.service -n 100                    # Últimas 100 líneas
```

### **Verificar Proceso**
```bash
ps aux | grep PruebaAravis              # Proceso ejecutándose
top -p $(pgrep -f PruebaAravis)         # Uso de recursos
```

### **Espacio en Disco**
```bash
df -h /var/log/calippo                  # Espacio usado por logs
du -sh /var/log/calippo/*               # Tamaño por directorio
```

## 📁 ESTRUCTURA DE LOGS GENERADOS

```
/var/log/calippo/
├── system/
│   ├── calippo_jetson.log              # Logs principales de aplicación
│   ├── calippo_jetson_metrics.log      # Métricas de rendimiento
│   ├── syslog.log                       # Logs del sistema (rsyslog)
│   └── syslog_errors.log               # Solo errores del sistema
├── digital/
│   └── digital_io.log                  # Logs de salidas digitales/PLC
├── photos/
│   ├── snapshots/                      # Fotos periódicas
│   └── defects/                        # Fotos de defectos
└── vision/
    ├── vision_log.csv                  # Logs detallados por lata (CSV)
    └── vision_log.jsonl                # Logs detallados por lata (JSON)
```

## 🔄 ROTACIÓN AUTOMÁTICA DE LOGS

- **Frecuencia**: Diaria a las 00:00
- **Retención**: 30 días
- **Compresión**: Automática con gzip
- **Formato**: `archivo-YYYYMMDD.log.gz`

## ⚠️ SOLUCIÓN DE PROBLEMAS

### **Servicio no arranca**
```bash
sudo journalctl -u calippo.service -n 50    # Ver últimos errores
sudo systemctl daemon-reload                # Recargar configuración
sudo systemctl restart calippo.service      # Reiniciar servicio
```

### **Logs no se generan**
```bash
sudo systemctl status rsyslog               # Verificar rsyslog
sudo systemctl restart rsyslog              # Reiniciar rsyslog
ls -la /var/log/calippo/                    # Verificar permisos
```

### **Proceso no ejecutándose**
```bash
ps aux | grep PruebaAravis                  # Buscar proceso
sudo systemctl start calippo.service        # Iniciar servicio
./run_calippo.sh                           # Ejecutar manualmente para debug
```

### **Espacio en disco lleno**
```bash
df -h                                       # Verificar espacio
sudo du -sh /var/log/calippo/*             # Ver tamaño de logs
sudo logrotate -f /etc/logrotate.d/calippo # Forzar rotación
```

## 🎯 VERIFICACIÓN FINAL

Después de la instalación, debe cumplirse:

1. ✅ **Autoarranque**: El servicio inicia automáticamente al reiniciar
2. ✅ **Ejecución continua**: El proceso corre sin intervención
3. ✅ **Logs activos**: Se generan logs en tiempo real
4. ✅ **Reinicio automático**: Si falla, se reinicia automáticamente
5. ✅ **Rotación de logs**: Los logs se comprimen diariamente
6. ✅ **Modo headless**: Funciona sin interfaz gráfica

## 📞 SOPORTE

Si encuentras problemas:

1. **Ejecutar verificación completa**:
   ```bash
   ./verify_calippo_installation.sh
   ```

2. **Revisar logs del sistema**:
   ```bash
   sudo journalctl -u calippo.service --no-pager
   ```

3. **Reinstalar si es necesario**:
   ```bash
   sudo systemctl stop calippo.service
   sudo systemctl disable calippo.service
   ./install_calippo_factory.sh
   ```

---

**¡El sistema está listo para funcionar en fábrica de forma completamente autónoma!** 🎉

---

## 🧭 RESUMEN DE SCRIPTS Y CUÁNDO USARLOS

- **base_setup_system.sh** (root): instala/asegura CUDA 11.4, cuDNN 8.6, TensorRT 8.5.2, OpenCV del sistema, dependencias y variables de entorno; habilita `logrotate.timer`.
  - Uso:
    ```bash
    sudo /home/nvidia/Desktop/Calippo_jetson/base_setup_system.sh
    ```

- **install_pytorch_jetson.sh** (usuario normal): instala PyTorch 2.0.0+nv23.05 y torchvision compatibles en la `.venv` del proyecto. Usa wheel local si existe en `/home/nvidia/tmp_jp/`.
  - Uso:
    ```bash
    /home/nvidia/Desktop/Calippo_jetson/install_pytorch_jetson.sh
    ```

- **install_aravis.sh** (root): intenta instalar Aravis 0.8 por paquetes; si no están, compila e instala desde fuente.
  - Uso:
    ```bash
    sudo /home/nvidia/Desktop/Calippo_jetson/install_aravis.sh
    ```

- **install_calippo_factory.sh** (usuario normal): configura autoarranque (`systemd`), `rsyslog`, `logrotate`, directorios/permisos de logs, cron, y prueba el servicio.
  - Uso:
    ```bash
    /home/nvidia/Desktop/Calippo_jetson/install_calippo_factory.sh
    ```

- **verify_calippo_installation.sh** (usuario normal): verificaciones post-instalación (servicio, logs, espacio, proceso en ejecución) y prueba opcional de reinicio.
  - Uso:
    ```bash
    /home/nvidia/Desktop/Calippo_jetson/verify_calippo_installation.sh
    ```

- **run_calippo.sh** (no ejecutar manualmente en producción): lanzador que usa el servicio `systemd`.

### Orden recomendado (equipo de fábrica, JetPack 5.1.1 limpio)
1. CUDA/cuDNN/TensorRT y deps del SO:
   ```bash
   sudo /home/nvidia/Desktop/Calippo_jetson/base_setup_system.sh
   ```
2. PyTorch en la `.venv` del proyecto:
   ```bash
   /home/nvidia/Desktop/Calippo_jetson/install_pytorch_jetson.sh
   ```
3. Aravis 0.6 (paquetes del sistema):
   ```bash
   sudo apt install -y gir1.2-aravis-0.6 libaravis-0.6-0 aravis-tools
   ```
4. Autoarranque + logs:
   ```bash
   /home/nvidia/Desktop/Calippo_jetson/install_calippo_factory.sh
   /home/nvidia/Desktop/Calippo_jetson/verify_calippo_installation.sh
   sudo reboot
   ```

### ¿Hace falta contraseña (sudo)?
### Verificación final

```bash
# NVIDIA / CUDA
nvcc --version
ldconfig -p | grep libcudnn

# PyTorch / NumPy / OpenCV
cd /home/nvidia/Desktop/Calippo_jetson/gentl && source .venv/bin/activate
python - <<'PY'
import numpy as np, torch, cv2
print('numpy', np.__version__)
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('opencv', cv2.__version__)
PY

# Aravis 0.6
python -c "import gi; gi.require_version('Aravis','0.6'); from gi.repository import Aravis; print('Aravis 0.6 OK')"
dpkg -l | grep aravis

# Servicio
systemctl status vision-app.service --no-pager || systemctl status calippo.service --no-pager

# Logs
ls -la /var/log/calippo/system
tail -n 50 /var/log/calippo/system/calippo_jetson.log
```
- Sí, para scripts que modifican el sistema: `base_setup_system.sh`, `install_aravis.sh`, y algunas operaciones internas de `install_calippo_factory.sh`.
- El resto se ejecutan como usuario normal.

Opcional: habilitar sudo sin contraseña para comandos concretos (recomendado solo en equipos de producción cerrados):
```bash
echo "nvidia ALL=(ALL) NOPASSWD: /usr/bin/apt, /usr/bin/systemctl, /usr/bin/ldconfig, /usr/bin/tee, /usr/bin/touch, /bin/mkdir, /bin/chown, /bin/chmod" | sudo tee /etc/sudoers.d/calippo-nopasswd
sudo chmod 440 /etc/sudoers.d/calippo-nopasswd
```

### ¿Sobra algún script?
- No. Cada uno cubre una fase diferente: base del SO, ML (PyTorch), cámara (Aravis), y despliegue (autoarranque/logs/verificación).
- Si el equipo ya trae CUDA/cuDNN/TensorRT correctos, puedes saltarte `base_setup_system.sh`.

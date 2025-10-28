# Replicación de Sistema Jetson Orin JetPack 5.1.1

## Archivos incluidos:
- `requirements.txt`: Paquetes Python exactos
- `setup_system.sh`: Configuración del sistema
- `install_packages.sh`: Instalación de paquetes
- `setup_venv.sh`: Creación del entorno virtual
- `configure_environment.sh`: Variables de entorno
- `replicate_system.sh`: Script maestro
- `system_config.json`: Configuración del sistema

## Instrucciones:
1. Copiar todos los archivos al nuevo Jetson
2. Ejecutar: `chmod +x *.sh`
3. Ejecutar: `./replicate_system.sh`
4. Reiniciar el sistema

## Verificación:
```bash
cd /home/nvidia/Desktop/Calippo_jetson/gentl
source .venv/bin/activate
python3 diagnostico_jetpack511.py
```

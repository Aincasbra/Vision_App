"""
Camera module
-------------
- Módulo de backends de cámara: interfaz común, selección automática, implementaciones y gestión.
- Estructura:
  * `device_manager.py`: interfaz `CameraBackend` (genérica) y funciones `open_camera()`/`stop_camera()`
  * `selector.py`: detección automática y selección del backend apropiado
  * `genicam_aravis_backend.py`: implementación GenICam/Aravis (heredando de `CameraBackend`)
  * `onvif_rtsp_backend.py`: implementación ONVIF/RTSP (heredando de `CameraBackend`)
"""
from .genicam_aravis_backend import AravisBackend
from .onvif_rtsp_backend import OnvifRtspBackend
from .selector import CameraSelector
from .device_manager import CameraBackend, open_camera, stop_camera

__all__ = [
    'AravisBackend',
    'OnvifRtspBackend',
    'CameraBackend',
    'CameraSelector',
    'open_camera',
    'stop_camera',
]


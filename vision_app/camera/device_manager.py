"""
CameraBackend
------------------------------------------------------
- Define la interfaz común `CameraBackend` y proporciona utilidades genéricas para cualquier cámara.
- Funcionalidades principales:
  * `CameraBackend`: clase base abstracta que define el contrato mínimo para backends
    - Métodos de instancia: `open()`, `start()`, `stop()`, `get_frame()`, `get()`, `set()`, `get_node()`
    - Métodos estáticos: `safe_get()`, `safe_set()` para operaciones genéricas
  * Funciones de orquestación:
    - `open_camera()`: abre la cámara usando `camera/selector.py` (auto-detección o backend forzado)
    - `stop_camera()`: cierra la cámara de forma segura
- Nota: La configuración inicial (PixelFormat, ExposureTime, Gain, etc.) se aplica en el método
  `open()` de cada backend específico (AravisBackend, OnvifRtspBackend), no aquí.
- Usa módulos de `camera/`:
  * `camera/selector.py`: para seleccionar y crear el backend apropiado (GenICam/Aravis u ONVIF/RTSP)
- Llamado desde:
  * `camera/genicam_aravis_backend.py`: `AravisBackend` hereda de `CameraBackend`
  * `camera/onvif_rtsp_backend.py`: `OnvifRtspBackend` hereda de `CameraBackend`
  * `vision_app/app.py`: usa `open_camera()` durante la inicialización
  * `developer_ui/handlers.py`: usa `CameraBackend.safe_get/safe_set` para operaciones desde la UI
  * `model/detection/detection_service.py`: usa `CameraBackend.safe_get` para leer propiedades de cámara
"""
from __future__ import annotations

from typing import Optional, Any, Tuple
import os

from core.logging import log_info, log_error, log_warning
from .selector import CameraSelector


class CameraBackend:
    """Interfaz mínima común para backends de cámara.
    
    Define el contrato que deben implementar todos los backends de cámara
    (GenICam/Aravis, ONVIF/RTSP, etc.).
    """

    def __init__(self, index: int = 0, bayer_code: Optional[int] = None, **kwargs: Any) -> None:
        self.index = index
        self.bayer_code = bayer_code

    def open(self) -> "CameraBackend":
        raise NotImplementedError

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_frame(self, timeout_ms: int = 100) -> Optional[Tuple[Any, float, float]]:
        """Devuelve (img_bgr, ts_unix, latency_ms) o None."""
        raise NotImplementedError

    def get(self, name: str, default: Any = None) -> Any:
        """Implementación genérica que detecta automáticamente el método del backend.
        
        Intenta usar `get_node_value()` (GenICam) o `get()` (otros), con fallback genérico.
        Los backends pueden sobrescribir este método si necesitan comportamiento específico.
        """
        # Intentar métodos específicos del backend
        if hasattr(self, "get_node_value"):
            try:
                return self.get_node_value(name, default)  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback: usar CameraBackend.safe_get como wrapper genérico
        return CameraBackend.safe_get(self, name, default)

    def set(self, name: str, value: Any) -> None:
        """Implementación genérica que detecta automáticamente el método del backend.
        
        Intenta usar `set_node_value()` (GenICam) o `set()` (otros), con fallback genérico.
        Los backends pueden sobrescribir este método si necesitan comportamiento específico.
        """
        # Intentar métodos específicos del backend
        if hasattr(self, "set_node_value"):
            try:
                self.set_node_value(name, value)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        # Fallback: usar CameraBackend.safe_set como wrapper genérico
        CameraBackend.safe_set(self, name, value)

    def get_node(self, name: str) -> Any:
        """Acceso a nodos específicos del protocolo.
        
        Por defecto retorna None. Los backends deben implementar este método
        si su protocolo soporta nodos (ej: GenICam).
        """
        return None

    @staticmethod
    def safe_get(camera: Any, name: str, default=None):
        """Lee propiedades/nodos de la cámara con manejo de errores (método estático genérico).
        
        Soporta `get_node_value()` (GenICam) y acceso directo según el backend.
        NO llama a `camera.get()` para evitar recursión.
        """
        try:
            # Prioridad 1: get_node_value() (GenICam/Aravis)
            if hasattr(camera, "get_node_value"):
                return camera.get_node_value(name, default)
            # Prioridad 2: acceso directo a propiedades (ONVIF/RTSP u otros)
            # No llamamos a camera.get() para evitar recursión con CameraBackend.get()
        except Exception:
            pass
        return default

    @staticmethod
    def safe_set(camera: Any, name: str, value) -> bool:
        """Escribe propiedades/nodos con validación y logging de errores (método estático genérico).
        
        Soporta `set_node_value()` (GenICam) y acceso directo según el backend.
        NO llama a `camera.set()` para evitar recursión.
        """
        try:
            # Prioridad 1: set_node_value() (GenICam/Aravis)
            if hasattr(camera, "set_node_value"):
                return bool(camera.set_node_value(name, value))
            # Prioridad 2: acceso directo a propiedades (ONVIF/RTSP u otros)
            # No llamamos a camera.set() para evitar recursión con CameraBackend.set()
        except Exception as e:
            log_warning(f"⚠️ CameraBackend.safe_set {name}: {e}")
        return False


# Funciones de orquestación (compatibilidad con código existente)
def open_camera(backend_cls=None, bayer_code=None, index: int = 0):
    """Abre la cámara usando selector (auto-detección o backend forzado).
    
    Args:
        backend_cls: Clase de backend para fallback (legacy, normalmente None)
        bayer_code: Código Bayer para conversión
        index: Índice de la cámara
    
    Returns:
        Instancia del backend abierto o None si falla
    """
    import cv2  # aseguramos disponibilidad cuando se llame
    try:
        # Intentar mediante factory (auto/backend desde settings/env)
        from core.settings import load_settings
        try:
            backend_pref = getattr(load_settings(), 'camera', {}).get('backend', 'auto')
        except Exception:
            backend_pref = 'auto'
        camera = CameraSelector.create(backend=backend_pref, index=index, bayer_code=bayer_code)
        if camera is None and backend_cls is not None:
            # Fallback al constructor legado si está disponible (compatibilidad)
            try:
                camera = backend_cls(index=index, bayer_code=bayer_code).open()
            except Exception:
                pass
        log_info("🔑 Cámara abierta")
        # Nota: La configuración inicial (PixelFormat, ExposureTime, Gain, etc.) se aplica
        # en el método open() de cada backend específico (AravisBackend, OnvifRtspBackend)
        return camera
    except Exception as e:
        log_error(f"❌ Error abriendo cámara: {e}")
        return None


def stop_camera(camera: Optional[CameraBackend]) -> None:
    """Cierra la cámara de forma segura."""
    try:
        if camera:
            camera.stop()
    except Exception:
        pass


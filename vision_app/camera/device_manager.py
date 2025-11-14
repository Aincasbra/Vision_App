"""
CameraBackend y gestión de configuración de cámara
------------------------------------------------------
- Define la interfaz común `CameraBackend` y proporciona utilidades genéricas para cualquier cámara.
- Funcionalidades principales:
  * `CameraBackend`: clase base abstracta que define el contrato mínimo para backends
    - Métodos de instancia: `open()`, `start()`, `stop()`, `get_frame()`, `get()`, `set()`, `get_node()`
    - Métodos estáticos: `safe_get()`, `safe_set()` para operaciones genéricas
  * Funciones de orquestación:
    - `open_camera()`: abre la cámara usando `camera/selector.py` (auto-detección o backend forzado)
    - `stop_camera()`: cierra la cámara de forma segura
  * Gestión de configuración:
    - `load_camera_config()`: carga configuración desde config_camera.yaml
    - Soporta múltiples cámaras: cada cámara puede tener su propia zona de trabajo
- Nota: La configuración inicial (PixelFormat, ExposureTime, Gain, etc.) se aplica en el método
  `open()` de cada backend específico (AravisBackend, OnvifRtspBackend), no aquí.
- Usa módulos de `camera/`:
  * `camera/selector.py`: para seleccionar y crear el backend apropiado (GenICam/Aravis u ONVIF/RTSP)
- Llamado desde:
  * `camera/genicam_aravis_backend.py`: `AravisBackend` hereda de `CameraBackend`
  * `camera/onvif_rtsp_backend.py`: `OnvifRtspBackend` hereda de `CameraBackend`
  * `vision_app/app.py`: usa `open_camera()` durante la inicialización, pasa config a DetectionService
  * `developer_ui/handlers.py`: usa `CameraBackend.safe_get/safe_set` para operaciones desde la UI
  * `model/detection/detection_service.py`: recibe configuración desde app.py (no carga directamente)
"""
from __future__ import annotations

from typing import Optional, Any, Tuple, Dict
import os
import yaml

from core.logging import log_info, log_error, log_warning
from .selector import CameraSelector


def load_camera_config(config_path: str = "config_camera.yaml") -> Dict[str, Any]:
    """
    Carga la configuración de cámara desde archivo YAML.
    
    Esta función centraliza la carga de configuración para que cualquier backend
    (GenICam, ONVIF, etc.) pueda acceder a ella. La configuración incluye:
    - ROI de la cámara
    - Zona de trabajo (work_zone) para validación de botes
    - Tamaños de bote esperados (bottle_sizes)
    
    Soporta múltiples cámaras: cada cámara puede tener su propia configuración
    si se especifica en el YAML usando el índice de la cámara.
    
    Args:
        config_path: Ruta al archivo de configuración (por defecto: "config_camera.yaml")
                    También puede especificarse vía variable de entorno CONFIG_CAMERA
    
    Returns:
        Dict con toda la configuración del YAML, o {} si no se encuentra
    
    Ejemplo de estructura en config_camera.yaml:
        # Configuración global (aplica a todas las cámaras)
        work_zone:
          center_x: null
          center_y: null
          radius: 50
        
        # Configuración por cámara (opcional)
        cameras:
          0:  # Cámara índice 0
            work_zone:
              center_x: 812
              radius: 50
          1:  # Cámara índice 1
            work_zone:
              center_x: 1000
              radius: 60
    """
    # Permitir sobreescribir vía variable de entorno
    env_config_path = os.environ.get("CONFIG_CAMERA")
    if env_config_path:
        config_path = env_config_path
    
    # Si es una ruta absoluta, intentar directamente
    if os.path.isabs(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            log_info(f"✅ Configuración de cámara cargada desde {config_path}", logger_name="system")
            return config or {}
        except FileNotFoundError:
            pass
        except Exception as e:
            log_warning(f"⚠️ Error cargando {config_path}: {e}", logger_name="system")
            return {}
    
    # Buscar en múltiples ubicaciones
    search_paths = [
        config_path,  # Ruta especificada (relativa)
        os.path.join("vision_app", "config_camera.yaml"),  # vision_app/config_camera.yaml
        os.path.join(os.path.dirname(__file__), "..", "config_camera.yaml"),  # Relativo a este módulo
    ]
    
    for path in search_paths:
        try:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                with open(abs_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                log_info(f"✅ Configuración de cámara cargada desde {abs_path}", logger_name="system")
                return config or {}
        except Exception as e:
            continue
    
    # Si no se encuentra en ninguna ubicación
    log_warning(f"⚠️ Archivo config_camera.yaml no encontrado en: {', '.join(search_paths)}, usando configuración por defecto", logger_name="system")
    return {}


def get_camera_config(camera_index: int = 0, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Obtiene la configuración específica de una cámara.
    
    Si existe configuración específica para el índice de la cámara, la retorna.
    Si no, retorna la configuración global.
    
    Args:
        camera_index: Índice de la cámara (0, 1, 2, ...)
        config: Configuración completa cargada (si None, la carga automáticamente)
    
    Returns:
        Dict con configuración específica de la cámara (work_zone, bottle_sizes, etc.)
    """
    if config is None:
        config = load_camera_config()
    
    # Si hay configuración específica para esta cámara, usarla
    cameras_config = config.get("cameras", {})
    if camera_index in cameras_config:
        camera_specific = cameras_config[camera_index].copy()
        # Combinar con configuración global (la específica tiene prioridad)
        result = config.copy()
        result.update(camera_specific)
        # Asegurar que work_zone y bottle_sizes estén presentes
        if "work_zone" not in result:
            result["work_zone"] = config.get("work_zone", {})
        if "bottle_sizes" not in result:
            result["bottle_sizes"] = config.get("bottle_sizes", {})
        return result
    
    # Usar configuración global
    return config


class CameraBackend:
    """Interfaz mínima común para backends de cámara.
    
    Define el contrato que deben implementar todos los backends de cámara
    (GenICam/Aravis, ONVIF/RTSP, etc.).
    """

    def __init__(self, index: int = 0, bayer_code: Optional[int] = None, **kwargs: Any) -> None:
        self.index = index
        self.bayer_code = bayer_code
        # Configuración de esta cámara (se carga cuando se abre)
        self.config: Optional[Dict[str, Any]] = None

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
def log_camera_parameters(camera: Any) -> None:
    """Registra los parámetros principales de la cámara en los logs.
    
    Args:
        camera: Instancia de cámara (CameraBackend o compatible)
    """
    try:
        log_info("📷 Parámetros de cámara:")
        log_info(f"   - PixelFormat: {CameraBackend.safe_get(camera, 'PixelFormat', 'N/A')}")
        log_info(f"   - WidthMax: {CameraBackend.safe_get(camera, 'WidthMax', 'N/A')}")
        log_info(f"   - HeightMax: {CameraBackend.safe_get(camera, 'HeightMax', 'N/A')}")
        log_info(f"   - Width (ROI actual): {CameraBackend.safe_get(camera, 'Width', 'N/A')}")
        log_info(f"   - Height (ROI actual): {CameraBackend.safe_get(camera, 'Height', 'N/A')}")
        
        # Log IP si está disponible (solo GenICam/Aravis)
        try:
            if hasattr(camera, 'get_node') and camera.get_node("GevCurrentIPAddress"):
                ip_node = camera.get_node("GevCurrentIPAddress")
                if ip_node:
                    ip_int = int(ip_node.value)
                    ip_str = ".".join(str((ip_int >> (8*i)) & 0xff) for i in [3,2,1,0])
                    log_info(f"📡 IP cámara (GenICam): {ip_str}")
        except Exception:
            pass
    except Exception:
        pass


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
        
        if camera is not None:
            log_info("🔑 Cámara abierta")
            # Registrar parámetros de la cámara
            log_camera_parameters(camera)
        
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


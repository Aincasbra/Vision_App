from __future__ import annotations

"""
Backend ONVIF/RTSP
------------------
- Implementación del backend de cámara para cámaras IP usando protocolo RTSP (ONVIF).
- Funcionalidades principales:
  * Captura de frames BGR desde stream RTSP usando OpenCV VideoCapture
  * Soporte para propiedades básicas (Width, Height, FPS) mediante mapeo a propiedades OpenCV
  * Requiere URL RTSP para inicialización (`rtsp_url` o `url` en kwargs)
- Implementa la interfaz `CameraBackend` para mantener compatibilidad con la API unificada.
- Nota: Control ONVIF completo (PTZ, parámetros avanzados) se puede añadir en el futuro.
- Llamado desde:
  * `camera/selector.py`: crea instancias de `OnvifRtspBackend` cuando se proporciona URL RTSP
    o cuando se fuerza el modo "onvif"
  * `camera/device_manager.py`: función `open_camera()` crea y configura el backend
"""

from typing import Any, Optional, Tuple
import time
import cv2
import numpy as np

from .device_manager import CameraBackend
from core.logging import log_info, log_warning


class OnvifRtspBackend(CameraBackend):
    """Backend ONVIF sobre RTSP (stub inicial).

    Implementa captura RTSP con OpenCV; control ONVIF (PTZ, params) se podrá añadir.
    """

    def __init__(self, index: int = 0, bayer_code: int | None = None, rtsp_url: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(index=index, bayer_code=bayer_code)
        self.rtsp_url = rtsp_url or kwargs.get("url") or ""
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> "OnvifRtspBackend":
        if not self.rtsp_url:
            raise RuntimeError("ONVIF/RTSP: falta URL (rtsp_url)")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir RTSP: {self.rtsp_url}")
        log_info(f"📡 ONVIF/RTSP abierto: {self.rtsp_url}")
        return self

    def stop(self) -> None:
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    def get_frame(self, timeout_ms: int = 100) -> Optional[Tuple[np.ndarray, float, float]]:
        if not self.cap:
            return None
        t0 = time.time()
        ok, frame = self.cap.read()
        if not ok:
            time.sleep(max(0.0, timeout_ms/1000.0))
            return None
        ts = time.time()
        lat_ms = (ts - t0) * 1000.0
        return frame, ts, lat_ms

    def get(self, name: str, default: Any = None) -> Any:
        """Sobrescribe get() para mapear propiedades a OpenCV (específico de ONVIF/RTSP)."""
        if not self.cap:
            return default
        prop_map = {
            "Width": cv2.CAP_PROP_FRAME_WIDTH,
            "Height": cv2.CAP_PROP_FRAME_HEIGHT,
            "FPS": cv2.CAP_PROP_FPS,
        }
        if name in prop_map:
            val = self.cap.get(prop_map[name])
            return val if val else default
        return default

    def set(self, name: str, value: Any) -> None:
        """Sobrescribe set() para mapear propiedades a OpenCV (específico de ONVIF/RTSP)."""
        if not self.cap:
            return
        prop_map = {
            "FPS": cv2.CAP_PROP_FPS,
        }
        if name in prop_map:
            ok = self.cap.set(prop_map[name], value)
            if not ok:
                log_warning(f"⚠️ No se pudo setear {name} en ONVIF/RTSP")

    # get_node() no se sobrescribe, usa la implementación por defecto de CameraBackend (retorna None)



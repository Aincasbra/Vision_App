from __future__ import annotations

"""
Backend ONVIF/RTSP
------------------
- Implementación de `CameraBackend` que captura vídeo vía RTSP
  (cámaras ONVIF) usando OpenCV.
- Habilita cámaras IP para la app con la misma API que GenICam.
- Instanciado por `camera/selector.py` (modo auto o forzado)
  y retornado a `core/device_manager.py`.
"""

from typing import Any, Optional, Tuple
import time
import cv2
import numpy as np

from camera.interface import CameraBackend
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
        if not self.cap:
            return
        prop_map = {
            "FPS": cv2.CAP_PROP_FPS,
        }
        if name in prop_map:
            ok = self.cap.set(prop_map[name], value)
            if not ok:
                log_warning(f"⚠️ No se pudo setear {name} en ONVIF/RTSP")

    def get_node(self, name: str) -> Any:
        return None



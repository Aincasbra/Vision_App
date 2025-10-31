from __future__ import annotations

"""
Interfaz de cámara (CameraBackend)
---------------------------------
- Cotrato mínimo que deben implementar los backends de cámara.
- Homogeniza el acceso a cámaras de distintos protocolos
  (GenICam/Aravis, ONVIF/RTSP, etc.).
- `core/device_manager.py` la consume indirectamente a través
  de `camera/selector.py` para instanciar el backend apropiado.
"""

from typing import Any, Optional, Tuple


class CameraBackend:
    """Interfaz mínima común para backends de cámara."""

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
        raise NotImplementedError

    def set(self, name: str, value: Any) -> None:
        raise NotImplementedError

    def get_node(self, name: str) -> Any:
        raise NotImplementedError



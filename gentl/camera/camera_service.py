"""
CameraService: utilidades seguras para operar la cámara (get/set/ROI) sin
duplicar lógica en UI. No cambia comportamiento; solo centraliza helpers.
"""
from __future__ import annotations

"""
CameraService (helpers de cámara)
---------------------------------
- Utilidades de acceso seguro (`safe_get/safe_set`) y ROI sobre una
  cámara ya instanciada (cualquier backend).
- Centralizar validaciones y logs al leer/escribir nodos.
- Desde `core/device_manager.py` y `ui/handlers.py`.
"""

from typing import Any, Tuple

from core.logging import log_warning


class CameraService:
    def __init__(self, camera: Any) -> None:
        self.camera = camera

    def safe_get(self, name: str, default=None):
        try:
            cam = self.camera
            if hasattr(cam, "get_node_value"):
                return cam.get_node_value(name, default)
            if hasattr(cam, "get"):
                return cam.get(name, default)  # type: ignore[attr-defined]
        except Exception:
            pass
        return default

    def safe_set(self, name: str, value) -> bool:
        try:
            cam = self.camera
            if hasattr(cam, "set_node_value"):
                return bool(cam.set_node_value(name, value))
            if hasattr(cam, "set"):
                return bool(cam.set(name, value))  # type: ignore[attr-defined]
        except Exception as e:
            log_warning(f"⚠️ CameraService.safe_set {name}: {e}")
        return False

    def set_roi(self, x: int, y: int, w: int, h: int) -> bool:
        ok = True
        ok &= self.safe_set("OffsetX", int(x))
        ok &= self.safe_set("OffsetY", int(y))
        ok &= self.safe_set("Width", int(w))
        ok &= self.safe_set("Height", int(h))
        return ok

    def restore_full_frame(self) -> Tuple[int, int]:
        """Restaura ancho/alto máximos; devuelve (w,h) efectivos."""
        try:
            h_max = int(self.safe_get("HeightMax", self.safe_get("Height", 1240)))
            w_max = int(self.safe_get("WidthMax", self.safe_get("Width", 1624)))
            self.set_roi(0, 0, w_max, h_max)
            return w_max, h_max
        except Exception as e:
            log_warning(f"⚠️ restore_full_frame: {e}")
            return int(self.safe_get("Width", 1624)), int(self.safe_get("Height", 1240))



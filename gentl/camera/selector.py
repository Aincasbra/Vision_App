from __future__ import annotations

"""
Selector de cámara (CameraSelector)
----------------------------------
- Componente que detecta y construye el backend de cámara correcto.
- Encapsula la lógica de elección (GenICam/Aravis u ONVIF/RTSP).
- `core/device_manager.py` → `CameraSelector.create(...)`.
"""

from core.logging import log_info, log_warning


def _has_aravis_devices() -> int:
    try:
        import gi
        gi.require_version('Aravis', '0.6')
        from gi.repository import Aravis
        Aravis.update_device_list()
        return int(Aravis.get_n_devices())
    except Exception:
        return 0


class CameraSelector:
    """Selecciona y construye el backend de cámara adecuado."""

    @staticmethod
    def create(backend: str = "auto", index: int = 0, **kwargs):
        backend = (backend or "auto").lower()
        if backend in ("aravis", "genicam"):
            from camera.genicam_aravis_backend import AravisBackend
            log_info("🎛 Backend seleccionado: GenICam/Aravis")
            return AravisBackend(index=index, **kwargs).open()
        if backend == "onvif":
            from camera.onvif_rtsp_backend import OnvifRtspBackend
            log_info("🎛 Backend seleccionado: ONVIF/RTSP")
            return OnvifRtspBackend(index=index, **kwargs).open()

        # auto
        n = _has_aravis_devices()
        if n > 0:
            from camera.genicam_aravis_backend import AravisBackend
            log_info(f"🔍 Detectado GenICam/Aravis: {n} dispositivo(s)")
            return AravisBackend(index=index, **kwargs).open()

        if 'rtsp_url' in kwargs or 'url' in kwargs:
            from camera.onvif_rtsp_backend import OnvifRtspBackend
            log_info("🔍 Usando ONVIF/RTSP por URL proporcionada")
            return OnvifRtspBackend(index=index, **kwargs).open()

        log_warning("⚠️ No se detectó backend de cámara compatible")
        return None



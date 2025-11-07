from __future__ import annotations

"""
Selector de cámara (CameraSelector)
-----------------------------------
- Componente que detecta y construye el backend de cámara apropiado.
- Funcionalidades principales:
  * `CameraSelector.create()`: método estático que selecciona y crea el backend:
    - Modo "auto": detecta automáticamente dispositivos Aravis, o usa ONVIF si hay URL
    - Modo "aravis"/"genicam": fuerza uso de backend GenICam/Aravis
    - Modo "onvif": fuerza uso de backend ONVIF/RTSP
  * `_has_aravis_devices()`: detecta si hay dispositivos GenICam/Aravis disponibles
- Llamado desde:
  * `camera/device_manager.py`: función `open_camera()` llama a `CameraSelector.create()` para obtener
    el backend de cámara según configuración o detección automática
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
            from .genicam_aravis_backend import AravisBackend
            log_info("🎛 Backend seleccionado: GenICam/Aravis")
            return AravisBackend(index=index, **kwargs).open()
        if backend == "onvif":
            from .onvif_rtsp_backend import OnvifRtspBackend
            log_info("🎛 Backend seleccionado: ONVIF/RTSP")
            return OnvifRtspBackend(index=index, **kwargs).open()

        # auto
        n = _has_aravis_devices()
        if n > 0:
            from .genicam_aravis_backend import AravisBackend
            log_info(f"🔍 Detectado GenICam/Aravis: {n} dispositivo(s)")
            return AravisBackend(index=index, **kwargs).open()

        if 'rtsp_url' in kwargs or 'url' in kwargs:
            from .onvif_rtsp_backend import OnvifRtspBackend
            log_info("🔍 Usando ONVIF/RTSP por URL proporcionada")
            return OnvifRtspBackend(index=index, **kwargs).open()

        log_warning("⚠️ No se detectó backend de cámara compatible")
        return None



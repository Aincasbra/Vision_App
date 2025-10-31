"""
DeviceManager
-------------
- Orquestador de dispositivo de captura. Encapsula apertura/cierre de cámara y setup básico.
- Abre la cámara a través de `camera/selector.py`,
  aplica configuración base con `camera/camera_service.py` y devuelve
  una instancia lista para usar.
- Desde `gentl/app.py` durante la inicialización.
"""
from __future__ import annotations

from typing import Optional
import os

from core.logging import log_info, log_warning, log_error
from camera.camera_service import CameraService
from camera.selector import CameraSelector


class DeviceManager:
    def __init__(self, backend_cls, bayer_code) -> None:
        self._backend_cls = backend_cls  # legacy compat (no usado si factory)
        self._bayer_code = bayer_code
        self.camera = None

    def open_camera(self, index: int = 0):
        import cv2  # aseguramos disponibilidad cuando se llame
        try:
            # Intentar mediante factory (auto/backend desde settings/env)
            from core.settings import load_settings
            try:
                backend_pref = getattr(load_settings(), 'camera', {}).get('backend', 'auto')
            except Exception:
                backend_pref = 'auto'
            self.camera = CameraSelector.create(backend=backend_pref, index=index, bayer_code=self._bayer_code)
            if self.camera is None:
                # Fallback al constructor legado si está disponible
                self.camera = self._backend_cls(index=index, bayer_code=self._bayer_code).open()
            log_info("🔑 Cámara abierta con privilegio: Control")
            self._apply_basic_nodes()
            return self.camera
        except Exception as e:
            log_error(f"❌ Error abriendo cámara: {e}")
            self.camera = None
            return None

    def _apply_basic_nodes(self):
        if not self.camera:
            return
        try:
            svc = CameraService(self.camera)
            svc.safe_set('PixelFormat', 'BayerBG8')
            svc.safe_set('TriggerMode', 'Off')
            svc.safe_set('AcquisitionFrameRate', 15.0)
            svc.safe_set('ExposureAuto', 'Off')
            svc.safe_set('ExposureMode', 'Timed')
            svc.safe_set('ExposureTime', 9000.0)
            svc.safe_set('Gain', 50.0)
            log_info("📸 Solicitado ExposureTime=9000.0 µs, Gain=50.0 dB")
            svc.safe_set('BalanceWhiteAuto', 'Off')
            # Reportar valores efectivos
            try:
                exp_val = svc.safe_get('ExposureTime', None)
                if exp_val is not None:
                    log_info(f"📸 ExposureTime aplicado: {float(exp_val):.1f} µs")
                else:
                    log_warning("⚠️ No se pudo leer ExposureTime tras aplicar")
            except Exception:
                pass
            try:
                gain_val = svc.safe_get('Gain', None)
                if gain_val is not None:
                    log_info(f"📸 Gain aplicado: {float(gain_val):.1f} dB")
                else:
                    log_warning("⚠️ No se pudo leer Gain tras aplicar")
            except Exception:
                pass
        except Exception as e:
            log_warning(f"⚠️ No se aplicó configuración extendida de cámara: {e}")

    def stop_camera(self):
        try:
            if self.camera:
                self.camera.stop()
        except Exception:
            pass



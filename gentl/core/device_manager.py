"""
DeviceManager: encapsula apertura/cierre de cámara y setup básico.
Mantiene API mínima para que `App` quede como orquestador.
"""
from __future__ import annotations

from typing import Optional
import os

from core.logging import log_info, log_warning, log_error
from camera.camera_service import CameraService


class DeviceManager:
    def __init__(self, backend_cls, bayer_code) -> None:
        self._backend_cls = backend_cls
        self._bayer_code = bayer_code
        self.camera = None

    def open_camera(self, index: int = 0):
        import cv2  # aseguramos disponibilidad cuando se llame
        try:
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



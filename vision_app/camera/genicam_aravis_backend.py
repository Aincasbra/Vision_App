"""
Backend GenICam/Aravis
----------------------
- Implementación del backend de cámara usando Aravis (GenICam) para cámaras GigE/USB.
- Funcionalidades principales:
  * Captura de frames BGR desde cámaras GenICam con conversión Bayer automática
  * Control de parámetros de cámara (exposición, ganancia, ROI, formato de píxel, etc.)
  * Gestión de buffers y stream con métricas de rendimiento (FPS, ancho de banda)
  * Soporte para múltiples dispositivos (selección por índice)
  * Acceso a nodos GenICam mediante `get_node_value()` y `set_node_value()`
  * `set_roi()`: configura región de interés (ROI) usando nodos GenICam
  * `restore_full_frame()`: restaura el frame completo a dimensiones máximas
- Implementa la interfaz `CameraBackend` heredando de ella.
- Llamado desde:
  * `camera/selector.py`: crea instancias de `AravisBackend` cuando se detectan dispositivos
    GenICam o cuando se fuerza el modo "aravis"/"genicam"
  * `camera/device_manager.py`: función `open_camera()` crea y configura el backend
  * `developer_ui/handlers.py`: usa `set_roi()` y `restore_full_frame()` para operaciones de ROI
"""

import time
import numpy as np
import cv2
import gi
gi.require_version("Aravis", "0.6")
from gi.repository import Aravis, GLib

from typing import Any
from .device_manager import CameraBackend
from core.logging import log_info, log_warning, log_debug


class AravisBackend(CameraBackend):
    """Wrapper mínimo para usar Aravis como fuente de frames BGR."""
    
    def __init__(self, index=0, n_buffers=6, bayer_code=cv2.COLOR_BayerBG2BGR, **kwargs):
        super().__init__(index=index, bayer_code=bayer_code, **kwargs)
        self.n_buffers = n_buffers
        self.camera = None
        self.dev = None
        self.stream = None
        self.payload = 0
        self.width = 0
        self.height = 0
        self.pixfmt = "Unknown"
        self.started = False
        # Métricas de red / stream
        self._stat_last_t = time.time()
        self._stat_frames_acc = 0
        self._stat_bytes_acc = 0

    def _try_pop(self, poll_us=20000):
        try:
            return self.stream.try_pop_buffer()
        except TypeError:
            try:
                return self.stream.try_pop_buffer(poll_us)
            except:
                return None

    def get_node_value(self, name, default=None):
        try:
            if name == "PixelFormat":
                return self.pixfmt
            if name == "DeviceVendorName":
                return self.dev.get_string_feature_value("DeviceVendorName")
            if name == "DeviceModelName":
                return self.dev.get_string_feature_value("DeviceModelName")
            if name == "DeviceSerialNumber":
                return self.dev.get_string_feature_value("DeviceSerialNumber")
            if name == "AcquisitionMode":
                return self.dev.get_string_feature_value("AcquisitionMode")
            if name == "ExposureTime":
                return self.dev.get_float_feature_value("ExposureTime")
            if name == "Gain":
                return self.dev.get_float_feature_value("Gain")
            if name == "AcquisitionFrameRate":
                return self.dev.get_float_feature_value("AcquisitionFrameRate")
            if name == "AcquisitionFrameRateEnable":
                return self.dev.get_boolean_feature_value("AcquisitionFrameRateEnable")
            if name == "TriggerMode":
                return self.dev.get_string_feature_value("TriggerMode")
            if name in ("BalanceWhiteAuto","WhiteBalanceAuto"):
                return self.dev.get_string_feature_value(name)
            if name == "Gamma":
                return self.dev.get_float_feature_value("Gamma")
            if name == "GammaEnable":
                return self.dev.get_boolean_feature_value("GammaEnable")
            if name == "Width":
                return int(self.dev.get_integer_feature_value("Width"))
            if name == "Height":
                return int(self.dev.get_integer_feature_value("Height"))
            if name == "WidthMax":
                return int(self.dev.get_integer_feature_value("WidthMax"))
            if name == "HeightMax":
                return int(self.dev.get_integer_feature_value("HeightMax"))
        except Exception:
            pass
        return default

    def set_node_value(self, name, value):
        try:
            if name == "PixelFormat":
                try:
                    if not self.started:
                        self.dev.set_string_feature_value("PixelFormat", str(value))
                    else:
                        raise RuntimeError("acquisition running")
                    self.pixfmt = str(value)
                    return True
                except Exception as e:
                    self.pixfmt = str(value)
                    print(f"[Aravis set PixelFormat] diferido hasta próximo START ({e})")
                    return True
            if name == "ExposureTime":
                self.dev.set_float_feature_value("ExposureTime", float(value))
                return True
            if name == "Gain":
                self.dev.set_float_feature_value("Gain", float(value))
                return True
            if name == "AcquisitionFrameRate":
                self.dev.set_float_feature_value("AcquisitionFrameRate", float(value))
                return True
            if name == "TriggerMode":
                self.dev.set_string_feature_value("TriggerMode", str(value))
                return True
            if name in ("BalanceWhiteAuto","WhiteBalanceAuto"):
                self.dev.set_string_feature_value(name, str(value))
                return True
            if name in ("ExposureAuto","GainAuto"):
                self.dev.set_string_feature_value(name, str(value)); return True
            if name in ("Gamma","GammaEnable"):
                if name == "Gamma": self.dev.set_float_feature_value("Gamma", float(value)); return True
                if name == "GammaEnable": self.dev.set_boolean_feature_value("GammaEnable", bool(value)); return True
        except Exception as e:
            print(f"[Aravis set {name}] {e}")
            return False
        return False

    def open(self):
        Aravis.update_device_list()
        n = Aravis.get_n_devices()
        if n <= 0:
            raise RuntimeError("No cameras found (Aravis)")
        if self.index >= n:
            raise RuntimeError(f"Index fuera de rango 0..{n-1}")
        cam_id = Aravis.get_device_id(self.index)
        self.camera = Aravis.Camera.new(cam_id)
        if self.camera is None:
            raise RuntimeError("Aravis.Camera.new() devolvió None")
        self.dev = self.camera.get_device()
        if self.dev is None:
            raise RuntimeError("camera.get_device() devolvió None")
        try:
            try:
                _ = self.dev.get_string_feature_value("GevCCP")
                have_gevccp = True
            except Exception:
                have_gevccp = False
            if have_gevccp:
                try:
                    self.dev.set_string_feature_value("GevCCP", "Control")
                except Exception:
                    for v in ("ExclusiveAccess", "Exclusive", "ControlWithSwitchover", "OpenAccess"):
                        try:
                            self.dev.set_string_feature_value("GevCCP", v)
                            break
                        except Exception:
                            pass
                try:
                    GvcpPrivilege = getattr(Aravis, "GvcpPrivilege", None)
                    if GvcpPrivilege is not None:
                        self.dev.set_control_channel_privilege(GvcpPrivilege.CONTROL)
                except Exception:
                    pass
                try:
                    self.dev.set_integer_feature_value("GevHeartbeatTimeout", 2000)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Aravis] Aviso al fijar privilegio: {e}")
        
        # Configuraciones específicas del backend (no duplicadas en CameraBackend)
        try:
            try:
                self.dev.set_integer_feature_value("GevSCPSPacketSize", 9000)
                print("[GigE] GevSCPSPacketSize = 9000 (jumbo frames)")
            except Exception:
                try:
                    self.dev.set_integer_feature_value("GevSCPSPacketSize", 8192)
                    print("[GigE] GevSCPSPacketSize = 8192")
                except Exception:
                    print("[GigE] No se pudo configurar GevSCPSPacketSize")
        except Exception:
            pass
        
        # Configuración básica específica de GenICam/Aravis
        try:
            try:
                self.dev.set_string_feature_value("PixelFormat", "BayerBG8")
                self.pixfmt = "BayerBG8"
                self.bayer_code = cv2.COLOR_BayerBG2BGR
            except Exception:
                try:
                    self.dev.set_string_feature_value("PixelFormat", "BayerRG8")
                    self.pixfmt = "BayerRG8"
                    self.bayer_code = cv2.COLOR_BayerRG2BGR
                except Exception:
                    self.pixfmt = "Unknown"
        except Exception:
            pass
        
        try:
            try:
                self.dev.set_string_feature_value("TriggerMode", "Off")
            except Exception:
                pass
            for enabler in ("AcquisitionFrameRateEnable",):
                try:
                    self.dev.set_integer_feature_value(enabler, 1)
                except Exception:
                    try:
                        self.dev.set_boolean_feature_value(enabler, True)
                    except Exception:
                        pass
            try:
                self.dev.set_float_feature_value("AcquisitionFrameRate", 15.0)
            except Exception:
                try:
                    self.dev.set_integer_feature_value("AcquisitionFrameRate", 15)
                except Exception:
                    pass
        except Exception:
            pass
        
        try:
            self.dev.set_string_feature_value("ExposureAuto", "Off")
            self.dev.set_string_feature_value("ExposureMode", "Timed")
            self.dev.set_float_feature_value("ExposureTime", 9000.0)
            self.dev.set_float_feature_value("Gain", 50.0)
            self.dev.set_string_feature_value("BalanceWhiteAuto", "Off")
            log_info("Configuración básica aplicada: ExposureTime=9000.0 µs, Gain=50.0 dB, FPS=15.0", logger_name="system")
        except Exception as e:
            log_warning(f"Aviso al aplicar configuración básica: {e}", logger_name="system")
        
        # Configuraciones específicas del backend (GigE)
        try:
            self.dev.set_integer_feature_value("GevSCPD", 2000)
            log_debug("GevSCPD = 2000 ns", logger_name="system")
        except Exception:
            pass
        self.stream = self.camera.create_stream(None, None)
        if self.stream is None:
            raise RuntimeError("create_stream() failed")
        try:
            def _align(v, m=8):
                return int(v) // m * m
            def _even2(v):
                return (int(v) // 2) * 2
            h_max = int(self.dev.get_integer_feature_value("HeightMax"))
            w_max = int(self.dev.get_integer_feature_value("WidthMax"))
            y1 = _even2(_align(h_max * 0.30, 8))
            y2 = _even2(_align(h_max * 0.82, 8))
            new_h = _even2(max(_align(y2 - y1, 8), 64))
            new_w = _align(w_max, 8)
            self.dev.set_integer_feature_value("OffsetX", 0)
            self.dev.set_integer_feature_value("OffsetY", int(y1))
            self.dev.set_integer_feature_value("Width",   int(new_w))
            self.dev.set_integer_feature_value("Height",  int(new_h))
            try:
                import builtins
                builtins.offsetY = int(y1)
            except Exception:
                pass
            log_debug(f"ROI/open Región aplicada: X=0 Y={int(y1)} W={int(new_w)} H={int(new_h)}", logger_name="system")
        except Exception as e:
            log_warning(f"ROI/open No se pudo aplicar ROI en open: {e}", logger_name="system")
        self.payload = self.camera.get_payload()
        if not isinstance(self.payload, int) or self.payload <= 0:
            w = int(self.dev.get_integer_feature_value("Width"))
            h = int(self.dev.get_integer_feature_value("Height"))
            self.payload = int(w * h)
        for _ in range(self.n_buffers):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(self.payload))
        try:
            self.width = int(self.dev.get_integer_feature_value("Width"))
            self.height = int(self.dev.get_integer_feature_value("Height"))
        except Exception:
            self.width, self.height = 0, 0
        # pixfmt ya se configuró arriba, solo verificamos si no se pudo leer
        if self.pixfmt == "Unknown":
            try:
                self.pixfmt = self.dev.get_string_feature_value("PixelFormat")
            except Exception:
                pass
        return self

    def start(self):
        if not self.started:
            try:
                def align(v, m=8):
                    return int(v) // m * m
                def even2(v):
                    return (int(v) // 2) * 2
                h_max = int(self.dev.get_integer_feature_value("HeightMax"))
                w_max = int(self.dev.get_integer_feature_value("WidthMax"))
                y1 = even2(align(h_max * 0.38, 8))
                y2 = even2(align(h_max * 0.78, 8))
                new_h = even2(max(align(y2 - y1, 8), 64))
                new_w = align(w_max, 8)
                self.dev.set_integer_feature_value("OffsetX", 0)
                self.dev.set_integer_feature_value("OffsetY", int(y1))
                self.dev.set_integer_feature_value("Width",   int(new_w))
                self.dev.set_integer_feature_value("Height",  int(new_h))
                try:
                    if self.started:
                        self.camera.stop_acquisition()
                        self.started = False
                except Exception:
                    pass
                try:
                    self.stream.flush()
                except Exception:
                    pass
                try:
                    while True:
                        b = self.stream.pop_buffer(0)
                        if b is None:
                            break
                except Exception:
                    pass
                try:
                    self.width  = int(self.dev.get_integer_feature_value("Width"))
                    self.height = int(self.dev.get_integer_feature_value("Height"))
                    try:
                        self.payload = int(self.camera.get_payload())
                    except Exception:
                        self.payload = int(self.width * self.height)
                    for _ in range(self.n_buffers):
                        self.stream.push_buffer(Aravis.Buffer.new_allocate(self.payload))
                except Exception:
                    pass
                try:
                    import builtins
                    builtins.offsetY = int(y1)
                except Exception:
                    pass
                log_debug(f"ROI/backend Región aplicada: X=0 Y={int(y1)} W={int(new_w)} H={int(new_h)}", logger_name="system")
            except Exception as e:
                log_warning(f"ROI/backend No se pudo aplicar ROI antes de start: {e}", logger_name="system")
            try:
                self.dev.set_string_feature_value("PixelFormat", "BayerBG8")
                self.pixfmt = "BayerBG8"
                self.bayer_code = cv2.COLOR_BayerBG2BGR
                log_debug("Forzando PixelFormat BayerBG8 (coherente con cámara)", logger_name="system")
            except Exception as e:
                log_warning(f"Aviso: no pude fijar PixelFormat: {e}", logger_name="system")
            self.camera.start_acquisition()
            self.started = True
            try:
                eff = self.dev.get_string_feature_value("PixelFormat")
            except Exception:
                eff = self.pixfmt
            log_info(f"START PixelFormat efectivo: {eff}", logger_name="system")

    def stop(self):
        if self.started:
            self.camera.stop_acquisition()
            self.started = False

    def close(self):
        try:
            self.stop()
        except:
            ...
        self.stream = None
        self.dev = None
        self.camera = None

    def _gvsp_bytes_per_buffer(self, payload_size, pkt_size=8192):
        overhead_per_pkt = 42
        n_pkts = (payload_size + pkt_size - 1) // pkt_size
        return payload_size + n_pkts * overhead_per_pkt

    def _log_stream_stats(self, every_sec=1.0):
        now = time.time()
        dt = now - self._stat_last_t
        if dt < every_sec:
            return
        fps = self._stat_frames_acc / dt if dt > 0 else 0.0
        mbps = (self._stat_bytes_acc * 8) / dt / 1e6
        resent = missing = completed = failures = underruns = -1
        try:
            resent = self.stream.get_n_resent_packets()
            missing = self.stream.get_n_missing_packets()
            completed = self.stream.get_n_completed_buffers()
            failures = self.stream.get_n_failures()
            underruns = self.stream.get_n_underruns()
        except Exception:
            try:
                stats = getattr(self.stream, 'get_statistics', None)
                if callable(stats):
                    _ = stats()
            except Exception:
                pass
        log_debug(f"NET FPS={fps:5.1f}  Throughput={mbps:6.1f} Mb/s  resent={resent}  missing={missing}  completed={completed}  fail={failures}  underrun={underruns}", logger_name="system")
        self._stat_last_t = now
        self._stat_frames_acc = 0
        self._stat_bytes_acc = 0

    def get_frame(self, timeout_ms=1000):
        if not self.started:
            return None
        latest = None
        drained_count = 0
        while drained_count < 10:
            try:
                buf = self.stream.try_pop_buffer()
                if buf is None:
                    break
                if latest is not None:
                    try:
                        self.stream.push_buffer(latest)
                    except Exception:
                        pass
                latest = buf
                drained_count += 1
            except Exception:
                break
        if latest is None:
            waited = 0
            poll_us = 20000
            max_wait = timeout_ms * 1000
            while waited < max_wait:
                try:
                    latest = self.stream.try_pop_buffer(poll_us)
                    if latest is not None:
                        break
                except Exception:
                    pass
                waited += poll_us
        if latest is None:
            return None
        buf = latest
        ts_cam_ns = None
        try:
            ts_cam_ns = buf.get_timestamp()
            if not hasattr(self, '_first_cam_timestamp'):
                self._first_cam_timestamp = ts_cam_ns
                self._first_sys_timestamp = time.time_ns()
        except Exception:
            pass
        data = buf.get_data()
        try:
            st = buf.get_status()
            if st != 0:
                try:
                    self.stream.push_buffer(buf)
                except Exception:
                    pass
                return None
        except Exception:
            pass
        if data is None:
            return None
        npbuf = np.frombuffer(bytes(data), dtype=np.uint8)
        try:
            bw = int(buf.get_image_width())
            bh = int(buf.get_image_height())
        except Exception:
            bw, bh = int(self.width), int(self.height)
        try:
            bstride = int(buf.get_image_stride())
        except Exception:
            bstride = bw
        total_needed = bstride * bh
        if npbuf.size < total_needed:
            try:
                self.stream.push_buffer(buf)
            except Exception:
                pass
            return None
        raw_strided = npbuf[:total_needed].reshape((bh, bstride))
        raw = raw_strided[:, :bw]
        self.width, self.height = bw, bh
        pxf = (self.pixfmt or "").upper()
        if pxf in ("MONO8",):
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif pxf in ("BAYERBG8","BAYER_BG8","BAYERBG8","BAYERBG8","BAYERBG8"):
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
        elif pxf in ("BAYERRG8","BAYER_RG8"):
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
        elif pxf in ("BAYERGB8","BAYER_GB8"):
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerGB2BGR)
        elif pxf in ("BAYERGR8","BAYER_GR8"):
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerGR2BGR)
        else:
            bgr = cv2.cvtColor(raw, self.bayer_code)
        self.stream.push_buffer(buf)
        try:
            bpp = 1
            payload_size = int(self.width * self.height * bpp)
            self._stat_frames_acc += 1
            self._stat_bytes_acc += self._gvsp_bytes_per_buffer(payload_size, pkt_size=8192)
            self._log_stream_stats(every_sec=1.0)
        except Exception:
            pass
        ts_now_ns = time.time_ns()
        if ts_cam_ns is not None and hasattr(self, '_first_cam_timestamp'):
            cam_diff_ns = ts_cam_ns - self._first_cam_timestamp
            sys_diff_ns = ts_now_ns - self._first_sys_timestamp
            lat_ms = (sys_diff_ns - cam_diff_ns) / 1e6
            if hasattr(self, '_frame_count'):
                self._frame_count += 1
            else:
                self._frame_count = 1
            if self._frame_count % 30 == 0:
                log_debug(f"LAT End2end={lat_ms:.1f}ms (cámara→CPU)", logger_name="system")
        else:
            lat_ms = None
        return bgr, time.time(), lat_ms

    def get(self, name: str, default: Any = None) -> Any:
        """Sobrescribe get() para usar get_node_value() específico de GenICam."""
        return self.get_node_value(name, default)

    def set(self, name: str, value: Any) -> None:
        """Sobrescribe set() para usar set_node_value() específico de GenICam."""
        self.set_node_value(name, value)

    def get_node(self, name: str) -> Any:
        """Implementa get_node() para acceso directo a nodos GenICam."""
        try:
            return self.dev.get_feature(name) if self.dev else None
        except Exception:
            return None

    def set_roi(self, x: int, y: int, w: int, h: int) -> bool:
        """Configura región de interés (ROI) usando nodos GenICam."""
        ok = True
        try:
            ok &= bool(self.set_node_value("OffsetX", int(x)))
            ok &= bool(self.set_node_value("OffsetY", int(y)))
            ok &= bool(self.set_node_value("Width", int(w)))
            ok &= bool(self.set_node_value("Height", int(h)))
        except Exception:
            ok = False
        return ok

    def restore_full_frame(self):
        """Restaura ancho/alto máximos; devuelve (w,h) efectivos."""
        try:
            h_max = int(self.get_node_value("HeightMax", self.get_node_value("Height", 1240)))
            w_max = int(self.get_node_value("WidthMax", self.get_node_value("Width", 1624)))
            self.set_roi(0, 0, w_max, h_max)
            return w_max, h_max
        except Exception as e:
            print(f"[Aravis] Error en restore_full_frame: {e}")
            return int(self.get_node_value("Width", 1624)), int(self.get_node_value("Height", 1240))



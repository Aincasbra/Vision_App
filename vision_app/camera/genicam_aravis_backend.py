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

from typing import Any, Tuple, Optional
from .device_manager import CameraBackend, get_camera_config
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
    
    @staticmethod
    def _align_dimension(v: int, m: int = 8) -> int:
        """Alinea una dimensión a múltiplo de m (típicamente 8 para alineación de píxeles)."""
        return int(v) // m * m
    
    @staticmethod
    def _even_dimension(v: int) -> int:
        """Asegura que una dimensión sea par (requerido para algunas cámaras)."""
        return (int(v) // 2) * 2
    
    @staticmethod
    def _eval_dimension_expression(value: Any, max_dim: int) -> Optional[int]:
        """
        Evalúa expresiones relativas a dimensiones máximas de la cámara.
        
        Soporta expresiones como:
        - "WidthMax/2" o "max/2" → calcula max_dim / 2
        - "HeightMax" o "max" → usa max_dim directamente
        - Valores numéricos → se devuelven directamente
        
        Args:
            value: Valor del config (puede ser int, float, str con expresión, o None)
            max_dim: Dimensión máxima de la cámara (WidthMax o HeightMax)
        
        Returns:
            Valor calculado en píxeles, o None si no se puede evaluar
        """
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            # Normalizar: convertir a minúsculas y quitar espacios
            expr = value.strip().lower()
            
            # Reemplazar referencias comunes
            expr = expr.replace("widthmax", str(max_dim))
            expr = expr.replace("heightmax", str(max_dim))
            expr = expr.replace("max", str(max_dim))
            
            # Evaluar expresión matemática simple (solo operaciones básicas)
            try:
                # Permitir expresiones como "max/2", "WidthMax/2", etc.
                result = eval(expr, {"__builtins__": {}}, {})
                return int(result)
            except Exception:
                # Si falla, intentar convertir directamente
                try:
                    return int(value)
                except Exception:
                    return None
        return None
    
    def _get_roi_config(self, h_max: int, w_max: int) -> Tuple[int, int, int, int]:
        """
        Calcula el ROI basado en la configuración de la cámara.
        
        Usa self.config que ya fue cargado en open(). Si no está disponible,
        intenta cargarlo automáticamente.
        
        Args:
            h_max: Altura máxima de la cámara
            w_max: Ancho máximo de la cámara
            
        Returns:
            Tupla (offset_x, offset_y, width, height) en píxeles
        """
        try:
            # Usar self.config si está disponible (ya cargado en open())
            if self.config is None:
                self.config = get_camera_config(self.index)
            roi_cfg = self.config.get("roi", {}) if self.config else {}
            mode = roi_cfg.get("mode", "full")
            
            if mode == "full":
                # Frame completo
                return (0, 0, self._align_dimension(w_max, 8), self._even_dimension(max(self._align_dimension(h_max, 8), 64)))
            
            elif mode == "custom":
                # ROI personalizado
                # Valores por defecto para offsets
                offset_x = int(roi_cfg.get("offset_x", 0))
                offset_y = int(roi_cfg.get("offset_y", 0))
                
                # Obtener valores de width y height del config
                # Si están especificados y son válidos, usar valores absolutos (con evaluación de expresiones)
                # Si no están especificados o son None/null, usar porcentajes del config
                width_val = roi_cfg.get("width")
                height_val = roi_cfg.get("height")
                
                # Solo usar valores absolutos si ambos están especificados y son válidos
                if width_val is not None and height_val is not None:
                    width = self._eval_dimension_expression(width_val, w_max)
                    height = self._eval_dimension_expression(height_val, h_max)
                    
                    # Si ambos se evaluaron correctamente, usar valores absolutos
                    if width is not None and height is not None:
                        result = (
                            self._align_dimension(offset_x, 8),
                            self._even_dimension(self._align_dimension(offset_y, 8)),
                            self._align_dimension(width, 8),
                            self._even_dimension(max(self._align_dimension(height, 8), 64))
                        )
                        return result
                    # Si alguno falló al evaluar, continuar con porcentajes
                
                # Usar porcentajes del config (valores por defecto vienen del config, no hardcodeados)
                offset_y_percent = roi_cfg.get("offset_y_percent", 0.0)
                height_percent = roi_cfg.get("height_percent", 1.0)
                offset_x_percent = roi_cfg.get("offset_x_percent", 0.0)
                width_percent = roi_cfg.get("width_percent", 1.0)
                
                y1 = self._even_dimension(self._align_dimension(h_max * offset_y_percent, 8))
                y2 = self._even_dimension(self._align_dimension(h_max * (offset_y_percent + height_percent), 8))
                new_h = self._even_dimension(max(self._align_dimension(y2 - y1, 8), 64))
                
                x1 = self._align_dimension(w_max * offset_x_percent, 8)
                x2 = self._align_dimension(w_max * (offset_x_percent + width_percent), 8)
                new_w = self._align_dimension(x2 - x1, 8)
                
                return (x1, y1, new_w, new_h)
            else:
                # Modo desconocido, usar frame completo
                log_warning(f"⚠️ Modo ROI desconocido: {mode}, usando frame completo", logger_name="system")
                return (0, 0, self._align_dimension(w_max, 8), self._even_dimension(max(self._align_dimension(h_max, 8), 64)))
        except Exception as e:
            log_warning(f"⚠️ Error leyendo configuración ROI: {e}, usando frame completo", logger_name="system")
            # Fallback a frame completo
            return (0, 0, self._align_dimension(w_max, 8), self._even_dimension(max(self._align_dimension(h_max, 8), 64)))

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
        # Cargar configuración de cámara ANTES de aplicar parámetros
        # Esto permite que los valores de exposición/FPS se lean desde config_camera.yaml
        try:
            self.config = get_camera_config(self.index)
        except Exception as e:
            log_warning(f"⚠️ No se pudo cargar configuración de cámara {self.index}: {e}", logger_name="system")
            self.config = {}
        
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
        
        # Aplicar configuración de parámetros de cámara desde config_camera.yaml
        # La fuente de verdad es config_camera.yaml (sección camera_params)
        # Si no existe configuración, usar valores por defecto como fallback de seguridad
        try:
            camera_params = self.config.get("camera_params", {}) if self.config else {}
            # Leer desde config_camera.yaml (valores por defecto solo como fallback si no existe YAML)
            exposure_time_us = float(camera_params.get("exposure_time_us", 9000.0))
            gain_db = float(camera_params.get("gain_db", 50.0))
            fps = float(camera_params.get("fps", 15.0))
            exposure_auto = str(camera_params.get("exposure_auto", "Off"))
            gain_auto = str(camera_params.get("gain_auto", "Off"))
        except Exception:
            # Fallback de seguridad: solo si hay error al leer configuración
            # NOTA: Estos valores deberían estar en config_camera.yaml, estos son solo emergencia
            log_warning("⚠️ No se pudo leer camera_params desde config_camera.yaml, usando valores por defecto", logger_name="system")
            exposure_time_us = 9000.0
            gain_db = 50.0
            fps = 15.0
            exposure_auto = "Off"
            gain_auto = "Off"
        
        # Configurar TriggerMode
        try:
            self.dev.set_string_feature_value("TriggerMode", "Off")
        except Exception:
            pass
        
        # Configurar FPS
        try:
            for enabler in ("AcquisitionFrameRateEnable",):
                try:
                    self.dev.set_integer_feature_value(enabler, 1)
                except Exception:
                    try:
                        self.dev.set_boolean_feature_value(enabler, True)
                    except Exception:
                        pass
            try:
                self.dev.set_float_feature_value("AcquisitionFrameRate", fps)
            except Exception:
                try:
                    self.dev.set_integer_feature_value("AcquisitionFrameRate", int(fps))
                except Exception:
                    pass
        except Exception:
            pass
        
        # Configurar exposición y ganancia
        try:
            self.dev.set_string_feature_value("ExposureAuto", exposure_auto)
            if exposure_auto == "Off":
                self.dev.set_string_feature_value("ExposureMode", "Timed")
                self.dev.set_float_feature_value("ExposureTime", exposure_time_us)
            
            self.dev.set_string_feature_value("GainAuto", gain_auto)
            if gain_auto == "Off":
                self.dev.set_float_feature_value("Gain", gain_db)
            
            self.dev.set_string_feature_value("BalanceWhiteAuto", "Off")
            log_info(f"✅ Configuración de cámara aplicada: ExposureTime={exposure_time_us} µs, Gain={gain_db} dB, FPS={fps} (desde config_camera.yaml)", logger_name="system")
        except Exception as e:
            log_warning(f"⚠️ Aviso al aplicar configuración de cámara: {e}", logger_name="system")
        
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
            # Obtener resolución máxima de la cámara (valores reales sin configurar)
            h_max = int(self.dev.get_integer_feature_value("HeightMax"))
            w_max = int(self.dev.get_integer_feature_value("WidthMax"))
            log_info(f"📷 Resolución máxima de la cámara: WidthMax={w_max}px, HeightMax={h_max}px", logger_name="system")
            
            # Obtener ROI desde configuración
            offset_x, offset_y, new_w, new_h = self._get_roi_config(h_max, w_max)
            log_info(f"📷 ROI configurado (desde config_camera.yaml): Width={new_w}px, Height={new_h}px, OffsetX={offset_x}px, OffsetY={offset_y}px", logger_name="system")
            
            self.dev.set_integer_feature_value("OffsetX", offset_x)
            self.dev.set_integer_feature_value("OffsetY", offset_y)
            self.dev.set_integer_feature_value("Width",   new_w)
            self.dev.set_integer_feature_value("Height",  new_h)
            try:
                import builtins
                builtins.offsetY = offset_y
            except Exception:
                pass
            log_debug(f"ROI/open Región aplicada: X={offset_x} Y={offset_y} W={new_w} H={new_h}", logger_name="system")
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
        # La configuración ya se cargó al inicio de open() para poder aplicar parámetros
        # (exposición, FPS, work_zone, bottle_sizes, etc.) desde config_camera.yaml
        return self

    def start(self):
        if not self.started:
            try:
                h_max = int(self.dev.get_integer_feature_value("HeightMax"))
                w_max = int(self.dev.get_integer_feature_value("WidthMax"))
                # Obtener ROI desde configuración
                offset_x, offset_y, new_w, new_h = self._get_roi_config(h_max, w_max)
                self.dev.set_integer_feature_value("OffsetX", offset_x)
                self.dev.set_integer_feature_value("OffsetY", offset_y)
                self.dev.set_integer_feature_value("Width",   new_w)
                self.dev.set_integer_feature_value("Height",  new_h)
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
                    builtins.offsetY = offset_y
                except Exception:
                    pass
                log_debug(f"ROI/backend Región aplicada: X={offset_x} Y={offset_y} W={new_w} H={new_h}", logger_name="system")
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



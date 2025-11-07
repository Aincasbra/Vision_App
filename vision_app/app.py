"""
App (orquestador principal)
----------------------------
- Orquesta la inicialización y ejecución del bucle principal de la aplicación.
- Llama a funciones de módulos especializados.
- Flujo: inicializa dispositivo → aplica optimizaciones → carga modelos → 
  inicializa cámara → ejecuta bucle principal.
- Se apoya en módulos modulares:
  * `core/optimizations`: optimizaciones genéricas del sistema
  * `model/detection/config`: carga del modelo YOLO y configuración (incluye lectura desde settings)
  * `model/classifier`: carga del clasificador (load_classifier desde multiclass.py)
  * `camera/device_manager`: gestión de dispositivos (cámara)
  * `model/detection/detection_service`: servicio completo de detección (YOLO + tracking + clasificación)
  * `developer_ui/*`: interfaz de depuración (ventana local)
  * `core/recording`: grabación de vídeo/imágenes
- Se invoca desde `main.py`.
"""
from dataclasses import dataclass
from typing import Optional
import os
import threading
import queue
import time
import builtins
import numpy as np
import cv2

from core.settings import load_settings
from core.logging import get_logger, log_info, log_warning, log_error
from core.settings import AppContext
from core.optimizations import apply_all as apply_all_optimizations
from camera.device_manager import CameraBackend, open_camera
from developer_ui.overlay import apply_yolo_overlay, apply_gamma_from_state
from developer_ui.app_controller import AppController
from developer_ui.window import create_main_window, show_frame_with_panel, show_black_with_panel, destroy_window
from core.recording import Recorder
from model.detection import DetectionService, load_yolo_model, load_yolo_config, YOLOPyTorchCUDA
from model.classifier import load_classifier


@dataclass
class App:
    """Aplicación principal."""
    context: AppContext
    camera: Optional[CameraBackend] = None
    yolo_model: Optional[YOLOPyTorchCUDA] = None
    detection_service: Optional[DetectionService] = None
    running: bool = False
    # Estado de grabación
    recording_active: bool = False
    recording_end_time: float = 0.0
    recording_frame_count: int = 0
    recording_frames_dir: Optional[str] = None
    recording_out_path: Optional[str] = None
    last_rec_log_second: int = -1
    # Estado de UI y controles
    yolo_running: bool = False
    gamma_actual: float = 0.8
    patron_actual: str = "BG"
    awb_indicator_active: bool = False
    awb_indicator_time: float = 0.0
    auto_cal_indicator_active: bool = False
    auto_cal_indicator_time: float = 0.0
    
    def __post_init__(self):
        self.logger = get_logger("system")
        self.context.logger = self.logger
        settings = load_settings()
        self.context.settings = settings
        # Compat: mantener config dict para módulos antiguos
        self.context.config = settings.raw_config
        self.context.evt_queue = queue.Queue()
        # Asegurar colas usadas por hilos (UI/YOLO)
        try:
            if getattr(self.context, 'infer_queue', None) is None:
                self.context.infer_queue = queue.Queue()
        except Exception:
            self.context.infer_queue = queue.Queue()
        # Servicio de grabación
        self.recorder = Recorder(out_dir=os.path.join(os.path.dirname(__file__), "Videos_YOLO"))
    
    def initialize(self) -> bool:
        """Inicializa componentes de la aplicación."""
        try:
            log_info("🚀 Inicializando aplicación Vision App...")
            
            # Inicializar dispositivo unificado
            self._initialize_device()
            
            # Aplicar optimizaciones
            self._apply_optimizations()
            
            # Cargar modelos
            self._load_models()
            
            # Inicializar cámara
            self._initialize_camera()
            
            log_info("✅ Aplicación inicializada correctamente")
            return True
            
        except Exception as e:
            log_error(f"❌ Error inicializando aplicación: {e}")
            return False
    
    def start(self) -> bool:
        """Inicia la aplicación."""
        try:
            log_info("▶️ Iniciando aplicación...")
            self.running = True
            
            # Iniciar hilos de procesamiento
            self._start_threads()
            
            # Iniciar bucle principal
            self._run_main_loop()
            
            return True
            
        except Exception as e:
            log_error(f"❌ Error iniciando aplicación: {e}")
            return False
    
    def stop(self):
        """Detiene la aplicación."""
        log_info("⏹️ Deteniendo aplicación...")
        self.running = False
        # Detener servicio de detección si está activo
        try:
            if self.detection_service is not None:
                self.detection_service.stop()
        except Exception:
            pass
        
        # Detener cámara
        if self.camera:
            self.camera.stop()
        
        # Limpiar recursos
        self._cleanup()
        
        log_info("✅ Aplicación detenida")
    
    def _initialize_device(self):
        """Inicializa el dispositivo unificado."""
        import torch
        
        if self.context.device is None:
            self.context.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            log_info(f"🔧 Dispositivo unificado inicializado: {self.context.device}")
            if torch.cuda.is_available():
                log_info(f"   - GPU: {torch.cuda.get_device_name(0)}")
                log_info(f"   - Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        return self.context.device
    
    def _apply_optimizations(self):
        """Aplica optimizaciones del sistema."""
        apply_all_optimizations(self.context)
        
        # Leer y guardar configuración YOLO para uso posterior
        yolo_conf, yolo_iou = load_yolo_config(self.context)
        self.context.yolo_conf = yolo_conf
        self.context.yolo_iou = yolo_iou
        log_info(f"✅ Configuración YOLO: Conf={yolo_conf}, IOU={yolo_iou}")
    
    def _load_models(self):
        """Carga modelos YOLO y clasificador."""
        # Cargar modelo YOLO (toda la lógica está en model/detection/config.py)
        self.yolo_model = load_yolo_model(self.context)
        
        # Cargar clasificador (toda la lógica está en model/classifier/multiclass.py)
        load_classifier()
    
    def _initialize_camera(self):
        """Inicializa la cámara."""
        try:
            # Abrir cámara usando función genérica (auto-detección o backend configurado)
            self.camera = open_camera(backend_cls=None, bayer_code=cv2.COLOR_BayerBG2BGR, index=0)
            
            # Dump inicial de parámetros (si existe utilidad)
            try:
                log_info("📷 Parámetros de cámara:")
                log_info(f"   - PixelFormat: {CameraBackend.safe_get(self.camera, 'PixelFormat', 'N/A')}")
                log_info(f"   - Width: {CameraBackend.safe_get(self.camera, 'Width', 'N/A')}")
                log_info(f"   - Height: {CameraBackend.safe_get(self.camera, 'Height', 'N/A')}")
            except Exception:
                pass
            
            # Configurar código Bayer en contexto si se usa aguas abajo
            self.context.config["cv_code_bayer"] = cv2.COLOR_BayerBG2BGR
            
            # Log IP si está disponible (solo GenICam/Aravis)
            try:
                if hasattr(self.camera, 'get_node') and self.camera.get_node("GevCurrentIPAddress"):
                    ip_node = self.camera.get_node("GevCurrentIPAddress")
                    if ip_node:
                        ip_int = int(ip_node.value)
                        ip_str = ".".join(str((ip_int >> (8*i)) & 0xff) for i in [3,2,1,0])
                        log_info(f"📡 IP cámara (GenICam): {ip_str}")
            except Exception:
                pass
            
            log_info("📷 Cámara inicializada correctamente")
        except Exception as e:
            log_error(f"❌ Error inicializando cámara: {e}")
            self.camera = None
    
    def _start_threads(self):
        """Inicia hilos de procesamiento."""
        try:
            # Iniciar servicio de inferencia YOLO (sustituye hilo interno)
            # Usar conf_threshold de optimizaciones si está disponible, sino DetectionService usará su default
            yolo_conf = getattr(self.context, 'yolo_conf', None)
            # process_every: DetectionService tiene default de 1, no es necesario pasarlo
            self.detection_service = DetectionService(
                yolo_model=self.yolo_model,
                infer_queue=self.context.infer_queue,
                conf_threshold=yolo_conf,  # None = usa default de DetectionService (0.4)
                process_every=1,  # Valor fijo, podría venir de settings en el futuro
                camera=self.camera,
            )
            self.detection_service.start()
        except Exception as e:
            log_error(f"❌ No se pudo iniciar DetectionService: {e}")
        
        # Aquí podría iniciarse hilo de captura si es necesario en el futuro
        
    def _run_main_loop(self):
        """Ejecuta el bucle principal."""
        # Flags desde settings centralizadas
        try:
            headless = bool(self.context.settings.headless)
        except Exception:
            headless = False
        try:
            auto_run = bool(self.context.settings.auto_run)
        except Exception:
            auto_run = False
        win_name = "Vision App"
        
        # Variables del bucle principal
        f = 0
        t0 = time.time()
        acquisition_running = False
        controller = AppController()
        
        # Crear UI si no es headless
        if not headless:
            try:
                # Interfaz inicial (tamaño original de cámara)
                try:
                    h_max = int(CameraBackend.safe_get(self.camera, 'HeightMax', 1240))
                    w_max = int(CameraBackend.safe_get(self.camera, 'WidthMax', 1624))
                except Exception:
                    h_max, w_max = 1240, 1624  # Valores por defecto
                
                # Crear ventana principal usando el módulo dedicado
                create_main_window(w_max + 350, h_max)
                
                # Configurar callback del ratón en la ventana correcta
                try:
                    # Registrar controlador de ratón desacoplado
                    cv2.setMouseCallback(win_name, lambda e, x, y, f, p=None: controller.handle_mouse_click(e, x, y, f, self))
                except Exception:
                    pass
                
                # Mostrar interfaz inicial con pantalla negra
                show_black_with_panel(w_max, h_max)
            except Exception as e:
                log_warning(f"⚠️ No se pudo crear ventana UI: {e}. Forzando HEADLESS.")
                headless = True
        
        # Auto-ejecutar RUN si headless o AUTO_RUN
        if headless or auto_run:
            try:
                self.context.evt_queue.put("RUN")
            except Exception:
                pass
        
        log_info("🏃 Entrando en bucle principal de la aplicación")
        try:
            while self.running:
                # Procesar eventos de UI del panel (controlador dedicado)
                resp = controller.process_pending(self)
                if resp:
                    if "acquisition_running" in resp:
                        acquisition_running = bool(resp["acquisition_running"])
                        log_info(f"📷 Estado de adquisición: {acquisition_running}")
                    if resp.get("record_start"):
                        try:
                            self.recorder.start(seconds=int(resp["record_start"]))
                            self.recording_active = True
                            log_info(f"🎬 Grabación iniciada: {resp['record_start']}s")
                        except Exception as e:
                            log_warning(f"⚠️ Error iniciando grabación: {e}")
                            self.recording_active = False
                
                # Bucle principal de captura y visualización
                if acquisition_running:
                    try:
                        # Obtener frame de la cámara
                        fb = self.camera.get_frame(timeout_ms=120)
                        if fb is None:
                            time.sleep(0.002)
                            continue
                        
                        img, ts_cap, lat_ms = fb
                        
                        # Demosaico si procede
                        try:
                            pxf = (self.camera.pixfmt or "").upper()
                            if pxf and 'BAYER' in pxf or pxf in ("MONO8",):
                                img_bgr = cv2.cvtColor(img, self.camera.bayer_code)
                            else:
                                img_bgr = img
                        except Exception:
                            img_bgr = img
                        
                        # Aplicar gamma desde utilidades
                        img_bgr = apply_gamma_from_state(img_bgr, self.gamma_actual)
                        
                        # Actualizar dimensiones UI para detección de clics
                        try:
                            h, w = img_bgr.shape[:2]
                            builtins.current_img_w = w
                            builtins.current_img_h = h
                            builtins.panel_offset_x = w
                        except Exception:
                            pass
                        
                        # Publicar frame para hilo YOLO
                        try:
                            builtins.latest_frame = img_bgr.copy()
                            fid = int(time.time()*1000) & 0x7FFFFFFF
                            builtins.latest_fid = fid
                        except Exception:
                            pass
                        
                        # Aplicar indicadores visuales desde módulos especializados
                        img_bgr = self.recorder.draw_recording_overlay(img_bgr)
                        from developer_ui.indicators import draw_all_indicators
                        img_bgr = draw_all_indicators(img_bgr, self)
                        
                        # Aplicar overlay YOLO (modular)
                        try:
                            out = apply_yolo_overlay(
                                img_bgr,
                                self.context,
                                builtins.latest_fid,
                                self.camera,
                            )
                        except Exception:
                            out = img_bgr
                        
                        # Mostrar en ventana (compositor modular)
                        if not headless:
                            try:
                                # Mostrar frame con panel usando el módulo dedicado
                                show_frame_with_panel(out, camera=self.camera, acquisition_running=acquisition_running, 
                                                   gamma_actual=self.gamma_actual, patron_actual=self.patron_actual, yolo_stats=None)
                            except Exception as e:
                                log_warning(f"⚠️ Error mostrando imagen: {e}")
                        
                        f += 1
                        
                    except Exception as e:
                        log_warning(f"⚠️ Error en captura: {e}")
                        time.sleep(0.005)
                else:
                    # Mostrar pantalla negra cuando está parado
                    if not headless:
                        try:
                            # Mostrar pantalla negra con panel usando el módulo dedicado
                            show_black_with_panel(w_max, h_max)
                        except Exception as e:
                            log_warning(f"⚠️ Error mostrando pantalla negra: {e}")
                    time.sleep(0.01)
                
                # Manejo de teclado
                if not headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                else:
                    time.sleep(0.01)

        except Exception as e:
            log_error(f"❌ Error en bucle principal: {e}")
        finally:
            try:
                if not headless:
                    destroy_window()
            except Exception:
                pass
            log_info("🏁 Saliendo del bucle principal")
    
    def _cleanup(self):
        """Limpia recursos."""
        # Implementar limpieza
        pass


def main():
    """Punto de entrada principal."""
    # Crear contexto de aplicación
    context = AppContext()
    
    # Crear aplicación
    app = App(context)
    
    # Inicializar
    if not app.initialize():
        log_error("❌ Falló la inicialización de la aplicación")
        return 1
    
    try:
        # Iniciar aplicación
        app.start()
    except KeyboardInterrupt:
        log_info("🛑 Interrupción por usuario")
    except Exception as e:
        log_error(f"❌ Error en ejecución: {e}")
    finally:
        # Detener aplicación
        app.stop()
    
    return 0


if __name__ == "__main__":
    exit(main())
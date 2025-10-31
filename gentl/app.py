from dataclasses import dataclass
from typing import Optional
import threading
import queue
import time
import numpy as np

from core.settings import load_settings
from core.logging import get_logger, log_info, log_warning, log_error
from core.context import AppContext
from core.optimizations import apply_all as apply_all_optimizations
from camera.aravis_backend import AravisBackend
from core.device_manager import DeviceManager
from vision.yolo_wrapper import YOLOPyTorchCUDA
from vision.classifier import clf_load
from vision.overlay import apply_yolo_overlay
# Limpieza: imports de ops/tracking/panel no usados aquí
from ui.handlers import handle_action
from ui.app_controller import AppController
from ui.window import create_main_window, show_frame_with_panel, show_black_with_panel, destroy_window
from vision.image_utils import apply_gamma_from_state
from core.recording import Recorder
from vision.yolo_service import YoloService

# Constantes de configuración YOLO
PROCESS_EVERY = 1
MISS_HOLD = 10
RESULT_TTL_S = 0.05
YOLO_CONF = 0.1  # Se puede sobrescribir por YAML
YOLO_IOU = 0.5
YOLO_WEIGHTS = "yolov8n.pt"  # Se sobrescribe por YAML
YOLO_IMGSZ = 640
# Ruta del clasificador multiclase (ajustada al archivo existente en el proyecto)
CLF_MODEL_PATH = "/home/nvidia/Desktop/Calippo_jetson/gentl/clasificador_multiclase_torch.pt"

# Estado global para tracking
bbox_ema = {}
histories = {}
last_seen = {}
last_conf = {}
frame_count = 0


@dataclass
class App:
    """Aplicación principal con ciclo de vida claro."""
    context: AppContext
    camera: Optional[AravisBackend] = None
    yolo_model: Optional[YOLOPyTorchCUDA] = None
    yolo_service: Optional[YoloService] = None
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
        self.logger = get_logger("calippo")
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
        import os
        self.recorder = Recorder(out_dir=os.path.join(os.path.dirname(__file__), "Videos_YOLO"))
    
    def initialize(self) -> bool:
        """Inicializa componentes de la aplicación."""
        try:
            log_info("🚀 Inicializando aplicación Calippo...")
            
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
        # Detener servicio YOLO si está activo
        try:
            if self.yolo_service is not None:
                self.yolo_service.stop()
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
        # Variables globales que necesitamos configurar
        global PROCESS_EVERY, MISS_HOLD, RESULT_TTL_S, YOLO_CONF, YOLO_IOU
        global bbox_ema, histories, last_seen, last_conf

        # Aplicar optimizaciones centralizadas y recoger umbrales
        YOLO_CONF, YOLO_IOU = apply_all_optimizations(self.context)

        # Parámetros de tracking de alta velocidad
        MISS_HOLD = 4
        RESULT_TTL_S = 0.05

        log_info(f"✅ Optimizaciones aplicadas:")
        log_info(f"   - Procesa 1 de cada {PROCESS_EVERY} frames")
        log_info(f"   - Hold: {MISS_HOLD} frames")
        log_info(f"   - TTL: {RESULT_TTL_S}s")
        log_info(f"   - YOLO Conf: {YOLO_CONF}, IOU: {YOLO_IOU}")

        # Reset de estructuras de tracking
        bbox_ema.clear()
        histories.clear()
        last_seen.clear()
        last_conf.clear()
        log_info("✅ Sistema de tracking optimizado para alta velocidad")
    
    def _load_models(self):
        """Carga modelos YOLO y clasificador."""
        import torch
        
        # Cargar modelo YOLO
        if torch.cuda.is_available():
            log_info(f"🚀 HILO YOLO usando CUDA: {torch.cuda.get_device_name(0)}")
            log_info(f"🚀 Memoria GPU antes de YOLO: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        else:
            log_warning("⚠️ HILO YOLO usando CPU (sin CUDA)")
        
        try:
            # Aplicar configuración YAML para modelo/imagen
            try:
                yolo_cfg = dict(getattr(self.context, 'settings', None).yolo)
            except Exception:
                yolo_cfg = {}
            weights_path = str(yolo_cfg.get('model_path') or yolo_cfg.get('model') or YOLO_WEIGHTS)
            imgsz_cfg = yolo_cfg.get('image_size', YOLO_IMGSZ)
            if isinstance(imgsz_cfg, (list, tuple)) and len(imgsz_cfg) > 0:
                imgsz_val = int(imgsz_cfg[0])
            else:
                imgsz_val = int(imgsz_cfg)

            # Crear modelo PyTorch optimizado
            self.yolo_model = YOLOPyTorchCUDA(weights_path, imgsz_val)
            log_info(f"✅ Modelo PyTorch cargado: {weights_path} (imgsz={imgsz_val})")
            
            # Aplicar optimizaciones específicas del modelo
            if hasattr(self.yolo_model, 'model') and hasattr(self.yolo_model.model, 'eval'):
                self.yolo_model.model.eval()  # Modo evaluación para mejor rendimiento
                log_info("✅ Modelo configurado en modo evaluación")
            
            # Configurar clases del modelo desde YAML si existen
            try:
                class_list = list(yolo_cfg.get('classes', []))
            except Exception:
                class_list = []
            if class_list:
                MODEL_NAMES = {i: name for i, name in enumerate(class_list)}
                KEEP_IDX = set(range(len(class_list)))
                log_info(f"✅ Clases del modelo (YAML): {MODEL_NAMES}")
            else:
                MODEL_NAMES = {0: 'can', 1: 'hand'}
                KEEP_IDX = {0, 1}
                log_info(f"ℹ️ Clases por defecto: {MODEL_NAMES}")
            
        except Exception as e:
            log_error(f"❌ Error cargando modelo YOLO: {e}")
            self.yolo_model = None
        
        # Cargar clasificador
        try:
            log_info("🔄 Cargando clasificador...")
            if not clf_load(CLF_MODEL_PATH):
                log_warning("❌ No se pudo cargar el clasificador, desactivando clasificación")
                # Desactivar clasificación si falla
                self.classifier = None
            else:
                log_info("✅ Clasificador listo")
        except Exception as e:
            log_error(f"❌ Error cargando clasificador: {e}")
            self.classifier = None
    
    def _initialize_camera(self):
        """Inicializa la cámara."""
        import cv2
        import os
        
        try:
            # Usar DeviceManager para apertura y setup
            dm = DeviceManager(AravisBackend, cv2.COLOR_BayerBG2BGR)
            self.camera = dm.open_camera(index=0)
            
            # Dump inicial de parámetros (si existe utilidad)
            try:
                log_info("📷 Parámetros de cámara:")
                log_info(f"   - PixelFormat: {self.camera.get('PixelFormat', 'N/A')}")
                log_info(f"   - Width: {self.camera.get('Width', 'N/A')}")
                log_info(f"   - Height: {self.camera.get('Height', 'N/A')}")
            except Exception:
                pass
            
            # Configurar código Bayer en contexto si se usa aguas abajo
            self.context.config["cv_code_bayer"] = cv2.COLOR_BayerBG2BGR
            
            # Log IP si está disponible
            try:
                if "GevCurrentIPAddress" in self.camera:
                    ip_int = int(self.camera.get_node("GevCurrentIPAddress").value)
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
            self.yolo_service = YoloService(
                yolo_model=self.yolo_model,
                infer_queue=self.context.infer_queue,
                conf_threshold=YOLO_CONF,
                process_every=PROCESS_EVERY,
                camera=self.camera,
            )
            self.yolo_service.start()
        except Exception as e:
            log_error(f"❌ No se pudo iniciar YoloService: {e}")
        
        # Aquí podría iniciarse hilo de captura si es necesario en el futuro
        
    def _yolo_inference_thread(self):
        """Sustituido por YoloService. Método legado no utilizado."""
        log_info("ℹ️ _yolo_inference_thread está obsoleto: use YoloService")
        time.sleep(0.01)
        return
        
    def _run_main_loop(self):
        """Ejecuta el bucle principal."""
        import time
        import cv2
        import builtins
        import numpy as np
        
        # Flags desde settings centralizadas
        try:
            headless = bool(self.context.settings.headless)
        except Exception:
            headless = False
        try:
            auto_run = bool(self.context.settings.auto_run)
        except Exception:
            auto_run = False
        win_name = "Calippo"
        
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
                    h_max = int(self.camera.get('HeightMax', 1240))
                    w_max = int(self.camera.get('WidthMax', 1624))
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
                        from ui.indicators import draw_all_indicators
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
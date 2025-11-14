"""
App (orquestador principal)
----------------------------
- Responsabilidad: ORQUESTAR la inicialización y ejecución del bucle principal.
- NO extrae ni procesa configuración: solo pasa `context` a los módulos especializados.
- Los módulos especializados leen directamente desde `context.settings` cuando necesitan config.

Flujo de inicialización:
  1. Carga configuración: `load_settings()` → `context.settings`
  2. Inicializa dispositivo: detecta CUDA/CPU
  3. Aplica optimizaciones: optimizaciones genéricas del sistema
  4. Carga modelos: `DetectionService` carga YOLO automáticamente, `load_classifier()` carga clasificador
     - Estos módulos leen directamente desde `context.settings`
  5. Inicializa cámara: abre cámara y carga su configuración desde `config_camera.yaml`
  6. Inicia hilos: crea `DetectionService(context)` que lee config desde `context.settings`
  7. Ejecuta bucle principal: captura frames y muestra UI

Módulos especializados (cada uno lee su propia config):
  * `core/settings`: carga YAML y crea Settings
  * `core/optimizations`: optimizaciones genéricas del sistema
  * `model/detection/detection_service`: carga modelo YOLO automáticamente (lee desde `context.settings.yolo`)
  * `model/classifier/multiclass`: carga clasificador (lee desde `context.settings.classifier`)
  * `camera/device_manager`: gestión de dispositivos (cámara)
  * `model/detection/detection_service`: servicio de detección (lee desde `context.settings`)
  * `developer_ui/*`: interfaz de depuración (ventana local)
  * `core/recording`: grabación de vídeo/imágenes

Se invoca desde `main.py`.
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
from core.timings import TimingsLogger
from camera.device_manager import CameraBackend, open_camera
from developer_ui.overlay import apply_yolo_overlay, apply_gamma_from_state
from developer_ui.app_controller import AppController
from developer_ui.window import create_main_window, show_frame_with_panel, show_black_with_panel, destroy_window
from core.recording import Recorder
from model.detection import DetectionService, YOLOPyTorchCUDA
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
        """Inicializa el contexto de la aplicación.
        
        Carga configuración desde YAML y la guarda en context.settings.
        Los módulos especializados leerán directamente desde context.settings.
        """
        self.logger = get_logger("system")
        self.context.logger = self.logger
        # Cargar configuración desde YAML (config_model.yaml)
        settings = load_settings()
        self.context.settings = settings  # Accesible por todos los módulos
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
        
        # TimingsLogger para mediciones de inicialización y pipeline
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        self.timings_logger = TimingsLogger(log_dir=log_dir, enable_stats=True, report_interval=50)
    
    def initialize(self) -> bool:
        """Inicializa componentes de la aplicación."""
        try:
            log_info("🚀 Inicializando aplicación Vision App...")
            
            # Inicializar dispositivo unificado
            self.timings_logger.start('init_device')
            self._initialize_device()
            self.timings_logger.end('init_device')
            
            # Aplicar optimizaciones
            self.timings_logger.start('init_optimizations')
            self._apply_optimizations()
            self.timings_logger.end('init_optimizations')
            
            # Cargar modelos
            self.timings_logger.start('init_load_models')
            self._load_models()
            self.timings_logger.end('init_load_models')
            
            # Inicializar cámara
            self.timings_logger.start('init_camera')
            self._initialize_camera()
            self.timings_logger.end('init_camera')
            
            log_info("✅ Aplicación inicializada correctamente")
            
            # Imprimir reporte de inicialización
            self.timings_logger.print_report()
            
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
        
        # Generar reporte final de timing
        if hasattr(self, 'timings_logger') and self.timings_logger:
            log_info("📊 Generando reporte final de timing...", logger_name="timings")
            self.timings_logger.print_report()
            self.timings_logger.save_report()
        
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
    
    def _load_models(self):
        """Carga modelos YOLO y clasificador.
        
        Los módulos especializados leen directamente desde config_model.yaml:
        - DetectionService carga el modelo YOLO automáticamente leyendo directamente desde config_model.yaml
        - load_classifier() lee directamente desde config_model.yaml
        """
        # Cargar clasificador
        # load_classifier() lee directamente desde config_model.yaml (sin pasar config)
        self.timings_logger.start('init_load_classifier')
        load_classifier(classifier_config=None)  # None = leer directamente desde config_model.yaml
        self.timings_logger.end('init_load_classifier')
        
        # NOTA: El modelo YOLO se carga automáticamente en DetectionService.__init__()
        # leyendo directamente desde config_model.yaml, no es necesario cargarlo aquí
    
    def _initialize_camera(self):
        """Inicializa la cámara."""
        try:
            # Abrir cámara usando función genérica (auto-detección o backend configurado)
            # El logging de parámetros se hace automáticamente en device_manager.open_camera()
            self.camera = open_camera(backend_cls=None, bayer_code=cv2.COLOR_BayerBG2BGR, index=0)
            
            # Configurar código Bayer en contexto si se usa aguas abajo
            self.context.config["cv_code_bayer"] = cv2.COLOR_BayerBG2BGR
            
            if self.camera is not None:
                log_info("📷 Cámara inicializada correctamente")
        except Exception as e:
            log_error(f"❌ Error inicializando cámara: {e}")
            self.camera = None
    
    def _start_threads(self):
        """Inicia hilos de procesamiento.
        
        DetectionService lee directamente desde context.settings:
        - context.settings.yolo.confidence_threshold
        - context.settings.classifier.bad_threshold
        - context.settings.classifier.classes
        
        app.py NO extrae estos valores, solo pasa el context completo.
        """
        try:
            # Obtener configuración de cámara (work_zone, bottle_sizes)
            # La configuración se carga automáticamente cuando se abre la cámara
            camera_config = None
            if self.camera is not None and hasattr(self.camera, 'config'):
                camera_config = self.camera.config
            
            # Pasar timings_logger al context para que DetectionService lo use
            self.context.timings_logger = self.timings_logger
            
            # Iniciar servicio de inferencia YOLO
            # DetectionService carga el modelo YOLO automáticamente leyendo directamente desde config_model.yaml
            self.timings_logger.start('init_detection_service')
            self.detection_service = DetectionService(
                infer_queue=self.context.infer_queue,
                context=self.context,  # Solo para colas y logger, NO para configuración
                yolo_model=None,  # None = cargar automáticamente leyendo desde config_model.yaml
                process_every=1,  # Valor fijo, podría venir de settings en el futuro
                camera=self.camera,
                camera_config=camera_config,  # Configuración de cámara (work_zone, bottle_sizes)
            )
            # Guardar referencia al modelo cargado para uso en app.py si es necesario
            self.yolo_model = self.detection_service.yolo_model
            self.detection_service.start()
            self.timings_logger.end('init_detection_service')
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
        # Variables para tamaño de ventana (se actualizarán con el tamaño real del frame)
        w_display, h_display = 1624, 1240  # Valores por defecto iniciales
        if not headless:
            try:
                self.timings_logger.start('init_create_ui')
                from developer_ui.window import get_window_size_from_camera
                # Calcular tamaño de ventana basado en ROI de la cámara
                w_display, h_display = get_window_size_from_camera(self.camera)
                
                # Crear ventana principal (el módulo calcula el tamaño automáticamente desde la cámara)
                create_main_window(camera=self.camera)
                self.timings_logger.end('init_create_ui')
                
                # Configurar callback del ratón en la ventana correcta
                try:
                    # Registrar controlador de ratón desacoplado
                    cv2.setMouseCallback(win_name, lambda e, x, y, f, p=None: controller.handle_mouse_click(e, x, y, f, self))
                except Exception:
                    pass
                
                # Mostrar interfaz inicial con pantalla negra (mostrar log solo al inicio)
                show_black_with_panel(w_display, h_display, log_once=True)
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
                        # NOTA: El ROI de la cámara NO puede cambiar durante la ejecución por seguridad.
                        # La ventana se redimensiona solo al inicio según el ROI configurado.
                        try:
                            h, w = img_bgr.shape[:2]
                            builtins.current_img_w = w
                            builtins.current_img_h = h
                            builtins.panel_offset_x = w
                            # Actualizar w_display y h_display solo la primera vez (para referencia)
                            if w_display == 1624 and h_display == 1240:  # Valores por defecto iniciales
                                w_display, h_display = w, h
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
                                                   gamma_actual=self.gamma_actual, patron_actual=self.patron_actual, yolo_stats=None, context=self.context)
                            except Exception as e:
                                log_warning(f"⚠️ Error mostrando imagen: {e}")
                        
                        f += 1
                        
                    except Exception as e:
                        log_warning(f"⚠️ Error en captura: {e}")
                        time.sleep(0.005)
                else:
                    # Mostrar pantalla negra cuando está parado (sin log, ya se mostró al inicio)
                    if not headless:
                        try:
                            # Mostrar pantalla negra con panel usando el tamaño del ROI (no el máximo)
                            # log_once=True para no mostrar el log repetidamente (ya se mostró al inicio)
                            show_black_with_panel(w_display, h_display, log_once=True)
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
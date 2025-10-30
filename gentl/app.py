from dataclasses import dataclass
from typing import Optional
import threading
import queue
import time
import numpy as np

from core.config import load_yaml_config
from core.logging import get_logger, log_info, log_warning, log_error
from core.context import AppContext
from core.device import apply_system_optimizations
from camera.aravis_backend import AravisBackend
from vision.yolo_wrapper import YOLOPyTorchCUDA
from vision.classifier import clf_load
from vision.overlay import apply_yolo_overlay
from vision.ops import merge_overlapping_detections
from vision.tracking import assign_stable_ids
from ui.panel import detectar_clic_panel_control, actualizar_panel_control
from ui.handlers import handle_action
from ui.window import create_main_window, show_frame_with_panel, show_black_with_panel, destroy_window
from vision.image_utils import apply_gamma_from_state
from core.recording import Recorder

# Constantes de configuración YOLO
PROCESS_EVERY = 1
MISS_HOLD = 10
RESULT_TTL_S = 0.05
YOLO_CONF = 0.1  # Se puede sobrescribir por YAML
YOLO_IOU = 0.5
YOLO_WEIGHTS = "yolov8n.pt"  # Se sobrescribe por YAML
YOLO_IMGSZ = 640
CLF_MODEL_PATH = "models/classifier.pth"

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
        self.context.config = load_yaml_config()
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
        import torch
        import cv2
        
        # Variables globales que necesitamos configurar
        global PROCESS_EVERY, MISS_HOLD, RESULT_TTL_S, YOLO_CONF, YOLO_IOU
        global bbox_ema, histories, last_seen, last_conf
        
        # Detectar CUDA
        torch_cuda_available = torch.cuda.is_available()
        
        # Aplicar optimizaciones del sistema usando el módulo dedicado
        apply_system_optimizations()
        
        # Optimizaciones para alta velocidad
        log_info("🚀 APLICANDO OPTIMIZACIONES PARA ALTA VELOCIDAD (300 latas/min)")
        
        MISS_HOLD = 4  # Mantener detecciones 4 frames sin verlas
        RESULT_TTL_S = 0.05  # TTL más corto para objetos rápidos
        # Cargar umbrales desde YAML si existen, si no usar defaults optimizados
        try:
            yolo_cfg = (self.context.config or {}).get('yolo', {})
        except Exception:
            yolo_cfg = {}
        YOLO_CONF = float(yolo_cfg.get('confidence_threshold', 0.4))
        YOLO_IOU = float(yolo_cfg.get('iou_threshold', 0.7))
        
        log_info(f"✅ Optimizaciones aplicadas:")
        log_info(f"   - Procesa 1 de cada {PROCESS_EVERY} frames")
        log_info(f"   - Hold: {MISS_HOLD} frames")
        log_info(f"   - TTL: {RESULT_TTL_S}s")
        log_info(f"   - YOLO Conf: {YOLO_CONF}, IOU: {YOLO_IOU}")
        
        # Optimizaciones CUDA
        if torch_cuda_available:
            log_info("🚀 HABILITANDO OPTIMIZACIONES CUDA UNIFICADAS")
            
            # Configurar PyTorch para máximo rendimiento en Jetson
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
            # Optimizaciones específicas para Jetson
            torch.set_num_threads(1)  # Un solo hilo para CUDA
            torch.set_num_interop_threads(1)
            
            # Configurar memoria GPU
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)  # Usar 80% de GPU
            
            # Configurar OpenCV para usar todos los cores disponibles
            cv2.setUseOptimized(True)
            cv2.setNumThreads(0)
            
            log_info("✅ Optimizaciones CUDA unificadas habilitadas")
            log_info(f"   - PyTorch CUDA: {torch.cuda.is_available()}")
            log_info(f"   - Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        else:
            log_warning("⚠️ CUDA no disponible, usando optimizaciones CPU")
        
        # Configurar tracking para alta velocidad
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
                yolo_cfg = (self.context.config or {}).get('yolo', {})
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
            # Crear backend Aravis y abrir cámara
            self.camera = AravisBackend(index=0, bayer_code=cv2.COLOR_BayerBG2BGR).open()
            log_info("🔑 Cámara abierta con privilegio: Control")
            
            # Dump inicial de parámetros (si existe utilidad)
            # Debug: mostrar parámetros de cámara
            try:
                log_info("📷 Parámetros de cámara:")
                log_info(f"   - PixelFormat: {self.camera.get('PixelFormat', 'N/A')}")
                log_info(f"   - Width: {self.camera.get('Width', 'N/A')}")
                log_info(f"   - Height: {self.camera.get('Height', 'N/A')}")
            except Exception:
                pass
            
            # Configuración básica de nodos si no está en solo lectura
            try:
                # Usar métodos directos del backend Aravis
                self.camera.set('PixelFormat', 'BayerBG8')
                self.camera.set('TriggerMode', 'Off')
                self.camera.set('AcquisitionFrameRate', 15.0)
                self.camera.set('ExposureAuto', 'Off')
                self.camera.set('ExposureMode', 'Timed')
                self.camera.set('ExposureTime', 4000.0)
                self.camera.set('Gain', 28.0)
                self.camera.set('BalanceWhiteAuto', 'Off')
            except Exception as e:
                log_warning(f"⚠️ No se aplicó configuración extendida de cámara: {e}")
            
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
            # Hilo YOLO reactivado - ahora lee desde builtins.latest_frame
            t = threading.Thread(target=self._yolo_inference_thread, daemon=True)
            t.start()
            log_info("🧵 Hilo de inferencia YOLO modular iniciado")
        except Exception as e:
            log_error(f"❌ No se pudo iniciar hilo YOLO: {e}")
        
        # Aquí podría iniciarse hilo de captura si es necesario en el futuro
        
    def _yolo_inference_thread(self):
        """Hilo de inferencia YOLO modular."""
        import builtins
        global frame_count
        
        log_info("🧵 Hilo de inferencia YOLO modular iniciado")
        
        while True:
            try:
                # Leer frame más reciente desde builtins
                if not hasattr(builtins, 'latest_frame') or builtins.latest_frame is None:
                    time.sleep(0.01)
                    continue
                
                # Obtener frame actual
                img_bgr = builtins.latest_frame.copy()
                frame_id = getattr(builtins, 'latest_fid', 0)
                
                # Procesar solo cada PROCESS_EVERY frame
                if frame_count % PROCESS_EVERY != 0:
                    frame_count += 1
                    continue
                
                # Inferencia YOLO
                if self.yolo_model is not None:
                    try:
                        # Detecciones raw
                        results = self.yolo_model.predict(img_bgr, conf_threshold=YOLO_CONF)
                        
                        # Debug: verificar resultados
                        xyxy = np.array([])
                        confs = np.array([])
                        clss = np.array([])
                        
                        if results and len(results) > 0:
                            raw_boxes, raw_confs, raw_clss = [], [], []

                            # Caso ONNX/Runtime: lista de dicts [{'bbox':[x1,y1,x2,y2],'confidence':..,'class_id':..}, ...]
                            if isinstance(results[0], dict):
                                dets = results
                                for det in dets:
                                    bbox = det.get('bbox') or det.get('xyxy') or det.get('box')
                                    if bbox is None or len(bbox) < 4:
                                        continue
                                    raw_boxes.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                                    raw_confs.append(float(det.get('confidence', 0.0)))
                                    raw_clss.append(int(det.get('class_id', -1)))
                                raw_boxes = np.asarray(raw_boxes, dtype=np.float32)
                                raw_confs = np.asarray(raw_confs, dtype=np.float32)
                                raw_clss = np.asarray(raw_clss, dtype=np.int32)

                            # Caso Ultralytics: objeto con .boxes
                            else:
                                result = results[0]
                                if hasattr(result, 'boxes') and result.boxes is not None:
                                    raw_boxes = result.boxes.xyxy.cpu().numpy()
                                    raw_confs = result.boxes.conf.cpu().numpy()
                                    try:
                                        raw_clss = result.boxes.cls.cpu().numpy()
                                    except Exception:
                                        raw_clss = np.full((len(raw_boxes),), -1, dtype=np.int32)
                                else:
                                    raw_boxes = np.empty((0,4), dtype=np.float32)
                                    raw_confs = np.empty((0,), dtype=np.float32)
                                    raw_clss = np.empty((0,), dtype=np.int32)
                            
                            log_info(f"🔍 YOLO raw: {len(raw_boxes)} detecciones, confs: {raw_confs[:3] if len(raw_confs) > 0 else 'ninguna'}")
                            
                            # Filtrar clases permitidas (0, 1, 2)
                            allowed_classes = {0, 1, 2}
                            mask = np.isin(raw_clss, list(allowed_classes))
                            xyxy = raw_boxes[mask]
                            confs = raw_confs[mask]
                            clss = raw_clss[mask]
                            
                            log_info(f"🎯 YOLO filtrado: {len(xyxy)} detecciones válidas")
                        else:
                            log_info("❌ YOLO: Sin resultados")
                            
                        # Fusionar detecciones superpuestas
                        if len(xyxy) > 0:
                            xyxy, confs, clss = merge_overlapping_detections(xyxy, confs, clss, iou_threshold=0.7)
                            
                            # Asignar IDs estables
                            if len(xyxy) > 0:
                                local_ids = assign_stable_ids(xyxy, confs, clss, frame_count)
                                
                                # Preparar resultado para la cola de inferencia
                                infer_result = {
                                    'frame_id': frame_id,
                                    'xyxy': xyxy.tolist(),
                                    'tids': local_ids,
                                    'lids': local_ids,
                                    'proc_ms': 0.0,  # TODO: medir tiempo real
                                    'ts': time.time()
                                }
                                
                                # Enviar a cola de inferencia
                                try:
                                    self.context.infer_queue.put_nowait(infer_result)
                                except queue.Full:
                                    pass  # Descartar si está llena
                                
                                log_info(f"🎯 YOLO: {len(xyxy)} detecciones, IDs: {local_ids}")
                            else:
                                log_info("❌ Sin detecciones válidas")
                        else:
                            log_info("❌ Sin resultados YOLO")
                            
                    except Exception as e:
                        log_warning(f"⚠️ Error en inferencia YOLO: {e}")
                
                frame_count += 1
                
            except Exception as e:
                log_warning(f"⚠️ Error en hilo YOLO: {e}")
                time.sleep(0.01)
        
        log_info("🔄 Hilo de inferencia YOLO modular detenido")
        
    def _handle_mouse_click(self, event, x, y, flags, param):
        """Maneja clics del ratón en la ventana principal."""
        import cv2
        import builtins
        
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                # Debug de clics 
                panel_offset_x = getattr(builtins, 'panel_offset_x', 0)
                current_img_h = getattr(builtins, 'current_img_h', 720)
                log_info(f"[click] x={x}, y={y}, panel_x={x - panel_offset_x}, panel_y={y}")
                
                # Detectar clic en panel usando dimensiones actuales
                panel_h_effective = max(current_img_h, 900)
                accion = detectar_clic_panel_control(x, y, panel_offset_x, panel_h_effective)
                log_info(f"🔍 Detección de clic: panel_offset_x={panel_offset_x}, panel_h_effective={panel_h_effective}, accion={accion}")
                
                if accion:
                    log_info(f"📤 Enviando acción a cola: {accion}")
                    self.context.evt_queue.put(accion)
                    
                    # Log de acciones
                    if accion == "RUN":
                        log_info("🚀 Botón RUN presionado - Iniciando cámara...")
                    elif accion == "STOP":
                        log_info("⏹️ Botón STOP presionado - Pausando cámara...")
                    elif accion == "AWB_ONCE":
                        log_info("🔄 AWB Once presionado...")
                    elif accion == "AUTO_CAL":
                        log_info("🎨 Auto Calibración presionado...")
                    elif accion == "CONFIG":
                        log_info("⚙️ Configuración presionado...")
                    elif accion == "EXIT":
                        log_info("🚪 Botón SALIR presionado - Cerrando aplicación...")
                        self.running = False
                    elif accion.startswith("GAMMA_"):
                        gamma_val = float(accion.split("_")[1])
                        self.gamma_actual = gamma_val
                        log_info(f"📊 Gamma ajustado a: {gamma_val}")
                    elif accion.startswith("BAYER_"):
                        bayer_code = accion.split("_")[1]
                        log_info(f"🎨 Patrón Bayer cambiado a: {bayer_code}")
                        
            except Exception as e:
                log_warning(f"⚠️ Error procesando clic: {e}")
        
    def _run_main_loop(self):
        """Ejecuta el bucle principal."""
        import os
        import time
        import cv2
        import builtins
        import numpy as np
        
        headless = os.environ.get("HEADLESS", "0") == "1"
        win_name = "Calippo"
        
        # Variables del bucle principal
        f = 0
        t0 = time.time()
        acquisition_running = False
        
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
                    cv2.setMouseCallback(win_name, self._handle_mouse_click)
                except Exception:
                    pass
                
                # Mostrar interfaz inicial con pantalla negra
                show_black_with_panel(w_max, h_max)
            except Exception as e:
                log_warning(f"⚠️ No se pudo crear ventana UI: {e}. Forzando HEADLESS.")
                headless = True
        
        # En headless, auto-ejecutar RUN
        if headless:
            try:
                self.context.evt_queue.put("RUN")
            except Exception:
                pass
        
        log_info("🏃 Entrando en bucle principal de la aplicación")
        try:
            while self.running:
                # Procesar eventos de UI del panel
                accion = None
                try:
                    accion = self.context.evt_queue.get_nowait()
                except Exception:
                    accion = None
                
                if accion:
                    log_info(f"🎯 Procesando acción: {accion}")
                    try:
                        resp = handle_action(self, accion)
                        log_info(f"✅ Respuesta de acción: {resp}")
                        
                        if "acquisition_running" in resp:
                            acquisition_running = bool(resp["acquisition_running"])
                            log_info(f"📷 Estado de adquisición: {acquisition_running}")
                        if resp.get("record_start"):
                            # Iniciar grabación N segundos
                            try:
                                self.recorder.start(seconds=int(resp["record_start"]))
                                self.recording_active = True
                                log_info(f"🎬 Grabación iniciada: {resp['record_start']}s")
                            except Exception as e:
                                log_warning(f"⚠️ Error iniciando grabación: {e}")
                                self.recording_active = False
                    except Exception as e:
                        log_warning(f"⚠️ Error procesando acción {accion}: {e}")
                    # EXIT se maneja en handler RUN/STOP; aquí solo seguimos
                    accion = None
                
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
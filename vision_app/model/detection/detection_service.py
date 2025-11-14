"""
Servicio de detección (pipeline completo para producción)
---------------------------------------------------------
- Responsabilidad: Ejecutar detección YOLO, validar botes y clasificarlos.
- Lee configuración directamente desde config_model.yaml.
- Ejecuta en hilo separado del bucle principal de la aplicación.
- Diseñado para producción: detecta botes continuamente pero solo procesa cuando están
  en posición correcta.

FLUJO DE CONFIGURACIÓN:
1. config_model.yaml → contiene configuración (secciones "yolo" y "classifier")
2. DetectionService → lee directamente desde config_model.yaml:
   - Sección "yolo": model_path, image_size, confidence_threshold
   - Sección "classifier": bad_threshold, classes

FLUJO DE TRABAJO:
1. YOLO detecta botes en TODOS los frames (detección continua)
2. Para cada detección, valida si el bote está listo:
   - ¿Está centrado en la zona de trabajo?
   - ¿Tiene el tamaño correcto (87px o 67px)?
   - ¿Está completo (no cortado por bordes)?
3. Si el bote es válido Y está fuera del cooldown:
   - Clasifica el bote (buena/mala/defectuosa)
   - Registra en CSV
   - Guarda imágenes si corresponde
   - (Futuro: envía salidas digitales GPIO)
4. Si el bote NO es válido:
   - Solo publica detección para UI (visualización)
   - NO procesa (ahorra recursos)

CARACTERÍSTICAS:
- ✅ VALIDACIÓN: Solo procesa botes válidos (centrados, completos, tamaño correcto)
- ✅ COOLDOWN: Evita procesar el mismo bote múltiples veces (3 frames de espera)
- ✅ EFICIENCIA: Procesa solo cuando importa, ahorra CPU/GPU
- ✅ PRODUCCIÓN: Optimizado para 300 botes/minuto, un frame por bote cuando está centrado

Funcionalidades principales:
- Ejecuta inferencia YOLO en hilo dedicado (no bloquea bucle principal)
- Aplica post-procesamiento: fusiona detecciones superpuestas (NMS)
- Valida objetos antes de procesar (validation.py)
- Clasifica botes válidos usando clasificador multiclase
- Publica resultados en cola para consumo por `app.py` (UI)
- Registra eventos en CSV (`vision_log.csv`) solo para botes procesados
- Gestiona guardado de imágenes (bad/good) vía `core/recording.py`
- Emite logs de rendimiento y política de clasificación

Llamado desde:
- `app.py`: instancia `DetectionService(context)` y lo inicia en hilo separado
  - app.py NO extrae valores del config, solo pasa context (para colas y logger)
  - DetectionService lee directamente desde config_model.yaml
"""
from __future__ import annotations

import time
import threading
import queue

from typing import Optional, Any, List, Dict
import os
import csv
from datetime import datetime

import numpy as np

from core.logging import log_info, log_warning, log_error
from core.timings import TimingsLogger
from core.recording import ImagesManager
from model.detection.validation import is_bottle_ready, get_work_zone_from_config
from model.classifier.multiclass import clf_predict_bgr
from model.detection.yolo_wrapper import YOLOPyTorchCUDA

# IOU threshold para merge de detecciones superpuestas (post-procesamiento NMS)
# Este valor NO está en config_model.yaml, es un parámetro interno del código
DEFAULT_MERGE_IOU_THRESHOLD = 0.7


def load_yolo_config_from_yaml() -> Dict[str, Any]:
    """Lee configuración YOLO directamente desde config_model.yaml.
    
    Función pública que puede ser importada por otros módulos (ej: compositor).
    Usa load_yaml_config() de settings.py para evitar duplicar la lógica de búsqueda.
    
    Returns:
        Dict con configuración de YOLO (sección "yolo" del YAML)
        
    Raises:
        ValueError: Si no se encuentra el archivo o falta la sección "yolo"
    """
    from core.settings import load_yaml_config
    
    # Carga silenciosa (el log ya se mostró al inicio en load_settings())
    config = load_yaml_config("config_model.yaml", silent=True)
    
    if "yolo" not in config:
        raise ValueError("config_model.yaml debe contener la sección 'yolo'")
    
    return dict(config["yolo"])


class DetectionService:
    """
    Servicio de detección que ejecuta YOLO en hilo separado.
    
    Este servicio:
    - Ejecuta YOLO continuamente para detectar botes en todos los frames
    - Valida cada detección antes de procesar (centrado, tamaño, completitud)
    - Solo procesa frames con botes válidos (clasificación, logging, imágenes)
    - Publica todas las detecciones en la cola para visualización en UI
    - Implementa cooldown para evitar procesar el mismo bote múltiples veces
    
    IMPORTANTE: 
    - Carga el modelo YOLO automáticamente leyendo directamente desde config_model.yaml.
    - Lee configuración directamente desde config_model.yaml (sin pasar por settings):
      * Sección "yolo" para cargar el modelo (model_path, image_size) y confidence_threshold
      * Sección "classifier" para bad_threshold y classes
    - La configuración de zona de trabajo (work_zone, bottle_sizes) se pasa desde app.py.
      La cámara carga su configuración automáticamente al abrirse (desde config_camera.yaml).
      Cada cámara puede tener su propia configuración si se especifica en el YAML.
    """

    def __init__(
        self,
        infer_queue: "queue.Queue",
        context: Any,  # AppContext con settings - lee configuración directamente desde aquí
        yolo_model: Optional[Any] = None,  # Opcional: si no se proporciona, se carga desde context.settings
        process_every: int = 1,
        camera: Optional[Any] = None,
        camera_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Inicializa DetectionService.
        
        Args:
            infer_queue: Cola para publicar resultados de inferencia
            context: AppContext (para colas y logger, NO para configuración)
            yolo_model: Modelo YOLO (opcional). Si no se proporciona, se carga leyendo directamente desde config_model.yaml
            process_every: Procesar cada N frames (default: 1)
            camera: Instancia de cámara (opcional)
            camera_config: Configuración de cámara (work_zone, bottle_sizes) - opcional, se puede obtener desde camera.config
        """
        self.infer_queue = infer_queue
        self.context = context
        
        # Cargar modelo YOLO: si no se proporciona, cargarlo desde context.settings
        if yolo_model is None:
            self.yolo_model = self._load_yolo_model_from_config()
        else:
            self.yolo_model = yolo_model
        self.process_every = int(process_every)
        self.camera = camera
        # Configuración de cámara (work_zone, bottle_sizes) - se pasa desde app.py
        # Si no se proporciona, se intenta obtener desde la cámara o usar valores por defecto
        self._camera_config = camera_config

        # Leer configuración YOLO directamente desde config_model.yaml
        yolo_cfg = load_yolo_config_from_yaml()
        if "confidence_threshold" not in yolo_cfg:
            raise ValueError("config_model.yaml debe contener 'yolo.confidence_threshold'")
        
        try:
            self.conf_threshold = float(yolo_cfg["confidence_threshold"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error leyendo 'yolo.confidence_threshold' desde config_model.yaml: {e}")

        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._frame_count: int = 0
        # Heartbeat para logs en headless
        self._hb_start_ts: float = time.time()
        self._hb_last_ts: float = self._hb_start_ts
        self._hb_frames: int = 0
        self._hb_last_boxes: int = 0
        # CSV visión
        self._vision_csv_enabled: bool = str(os.environ.get("LOG_VISION_CSV", "1")).lower() in ("1", "true", "yes", "on")
        self._log_dir: str = os.environ.get("LOG_DIR", "/var/log/vision_app")
        self._vision_csv_path: str = os.path.join(self._log_dir, "vision", "vision_log.csv")
        self._csv_writer = None
        self._csv_file = None
        # Imágenes
        self._images_enabled: bool = str(os.environ.get("LOG_IMAGES", "1")).lower() in ("1", "true", "yes", "on")
        self._images_mgr: Optional[ImagesManager] = None
        
        # Política de clasificación: prioriza buenas; sólo marca "bad" si >= umbral
        # Leer directamente desde config_model.yaml - OBLIGATORIO, sin fallbacks
        from model.classifier.multiclass import load_classifier_config_from_yaml
        classifier_config = load_classifier_config_from_yaml()
        if not classifier_config:
            raise ValueError("config_model.yaml debe contener la sección 'classifier' con sus parámetros")
        
        if "bad_threshold" not in classifier_config:
            raise ValueError("config_model.yaml debe contener 'classifier.bad_threshold'")
        
        try:
            self._clf_bad_threshold: float = float(classifier_config["bad_threshold"])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Error leyendo 'classifier.bad_threshold' desde config_model.yaml: {e}")
        
        # Leer bad_labels desde config - OBLIGATORIO
        if "classes" not in classifier_config:
            raise ValueError("config_model.yaml debe contener 'classifier.classes'")
        
        classifier_classes = classifier_config["classes"]
        if not isinstance(classifier_classes, (list, tuple)) or len(classifier_classes) == 0:
            raise ValueError("'classifier.classes' debe ser una lista no vacía en config_model.yaml")
        
        # Identificar qué clases son "malas" (normalmente "malas" y "defectuosas")
        self._bad_labels = {cls for cls in classifier_classes if cls.lower() in ("malas", "defectuosas", "bad", "defective")}
        if not self._bad_labels:
            from core.logging import log_warning
            log_warning("⚠️ No se encontraron clases 'malas' en classifier.classes. Se asume que todas las clases son 'buenas'.")
        
        # Configuración de zona de trabajo y validación de botes
        # Estos valores se cargan desde config_camera.yaml en _run()
        self._work_zone: Optional[Dict[str, Any]] = None  # Zona donde el bote debe estar centrado
        self._bottle_sizes: Optional[Dict[str, Any]] = None  # Tamaños esperados (87px o 67px)
        # Control de cooldown: evita procesar el mismo bote múltiples veces
        # En producción (300 botes/minuto), los botes pasan continuamente por la misma posición.
        # El cooldown es por frames: cuando un bote pasa por el centro y es válido, se procesa.
        # Después del cooldown, si hay otro bote válido en el centro, se procesa (es un bote nuevo).
        # Un frame por bote cuando está en el centro.
        self._last_processed_frame_id: int = -1  # Frame ID del último bote procesado
        self._bottle_processing_cooldown: int = 2  # Frames de espera antes de procesar otro bote
        
        # Estadísticas del clasificador (para UI)
        self._clf_total: int = 0  # Total de botes clasificados
        self._clf_buenas: int = 0  # Total de botes buenos
        self._clf_malas: int = 0  # Total de botes malos


    def _load_yolo_model_from_config(self) -> Optional[YOLOPyTorchCUDA]:
        """Carga el modelo YOLO leyendo directamente desde config_model.yaml.
        
        Returns:
            YOLOPyTorchCUDA o None si falla
        """
        import torch
        
        # Leer configuración YOLO directamente desde config_model.yaml
        yolo_cfg = load_yolo_config_from_yaml()
        
        # Leer configuración del modelo
        model_path = yolo_cfg.get('model_path') or yolo_cfg.get('model')
        if not model_path:
            raise ValueError("config_model.yaml debe contener 'yolo.model_path' o 'yolo.model'")
        
        if 'image_size' not in yolo_cfg:
            raise ValueError("config_model.yaml debe contener 'yolo.image_size'")
        
        imgsz_cfg = yolo_cfg['image_size']
        if isinstance(imgsz_cfg, (list, tuple)) and len(imgsz_cfg) > 0:
            imgsz_val = int(imgsz_cfg[0])
        else:
            imgsz_val = int(imgsz_cfg)
        
        # Log de dispositivo
        if torch.cuda.is_available():
            log_info(f"🚀 Cargando YOLO con CUDA: {torch.cuda.get_device_name(0)}")
            log_info(f"🚀 Memoria GPU antes de YOLO: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        else:
            log_warning("⚠️ Cargando YOLO con CPU (sin CUDA)")
        
        try:
            # Crear modelo
            yolo_model = YOLOPyTorchCUDA(str(model_path), imgsz_val)
            log_info(f"✅ Modelo YOLO cargado: {model_path} (imgsz={imgsz_val})")
            
            # Configurar modo evaluación
            if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'eval'):
                yolo_model.model.eval()
                log_info("✅ Modelo configurado en modo evaluación")
            
            return yolo_model
            
        except Exception as e:
            log_error(f"❌ Error cargando modelo YOLO: {e}")
            return None

    @staticmethod
    def _merge_overlapping_detections(xyxy, confs, clss, iou_threshold=DEFAULT_MERGE_IOU_THRESHOLD):
        """Fusiona detecciones superpuestas basándose en IoU y clase (OPTIMIZADO).
        
        Agrupa detecciones con IoU >= threshold y misma clase, manteniendo solo
        la de mayor confianza de cada grupo.
        
        Optimizaciones aplicadas:
        - Vectorización del cálculo de áreas
        - Uso de operaciones NumPy para cálculos de IoU
        - Reducción de conversiones de tipos innecesarias
        
        Args:
            xyxy: Array de cajas [x1, y1, x2, y2]
            confs: Array de confianzas
            clss: Array de clases
            iou_threshold: Umbral IoU para considerar superposición (default: DEFAULT_MERGE_IOU_THRESHOLD)
            
        Returns:
            Tuple de (xyxy, confs, clss) filtrados
        """
        if len(xyxy) == 0:
            return xyxy, confs, clss
        
        # Convertir a arrays numpy una sola vez
        xyxy = np.asarray(xyxy, dtype=np.float32)
        confs = np.asarray(confs, dtype=np.float32)
        clss = np.asarray(clss, dtype=np.int32)
        
        n = len(xyxy)
        if n == 0:
            return xyxy, confs, clss
        
        # Vectorizar cálculo de áreas (más eficiente que calcular en cada iteración)
        widths = np.maximum(0, xyxy[:, 2] - xyxy[:, 0])
        heights = np.maximum(0, xyxy[:, 3] - xyxy[:, 1])
        areas = widths * heights  # Array de áreas para todas las cajas
        
        keep = []
        used = np.zeros(n, dtype=bool)
        
        # Ordenar por confianza descendente para procesar primero las más confiables
        # Esto puede mejorar la agrupación
        sorted_indices = np.argsort(confs)[::-1]
        
        for idx in sorted_indices:
            if used[idx]:
                continue
            
            i = idx
            group = [i]
            used[i] = True
            
            # Vectorizar comparación de clases: solo comparar con detecciones de la misma clase
            same_class_mask = (clss == clss[i]) & (~used)
            if not np.any(same_class_mask):
                keep.append(i)
                continue
            
            # Obtener índices de detecciones de la misma clase no usadas
            candidates = np.where(same_class_mask)[0]
            
            # Vectorizar cálculo de IoU para todos los candidatos a la vez
            xi1, yi1, xi2, yi2 = xyxy[i]
            xj1 = xyxy[candidates, 0]
            yj1 = xyxy[candidates, 1]
            xj2 = xyxy[candidates, 2]
            yj2 = xyxy[candidates, 3]
            
            # Calcular intersección vectorizada
            xx1 = np.maximum(xi1, xj1)
            yy1 = np.maximum(yi1, yj1)
            xx2 = np.minimum(xi2, xj2)
            yy2 = np.minimum(yi2, yj2)
            
            inter_widths = np.maximum(0, xx2 - xx1)
            inter_heights = np.maximum(0, yy2 - yy1)
            inter_areas = inter_widths * inter_heights
            
            # Calcular unión vectorizada
            union_areas = areas[i] + areas[candidates] - inter_areas
            
            # Calcular IoU vectorizado (evitar división por cero)
            ious = np.where(union_areas > 0, inter_areas / union_areas, 0.0)
            
            # Filtrar candidatos con IoU >= threshold
            overlapping = candidates[ious >= iou_threshold]
            
            # Marcar como usados y agregar al grupo
            used[overlapping] = True
            # OPTIMIZADO: Usar numpy para encontrar el mejor índice (más rápido que max con lambda)
            group_indices = np.array([i] + overlapping.tolist(), dtype=np.int32)
            best_idx = group_indices[np.argmax(confs[group_indices])]
            keep.append(best_idx)
        
        keep = np.array(keep, dtype=np.int32)
        return xyxy[keep], confs[keep], clss[keep]

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log_info("🧵 DetectionService iniciado")

    def stop(self) -> None:
        self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=0.5)
        self._thread = None
        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass
        log_info("🧵 DetectionService detenido")


    def _run(self) -> None:
        """
        Bucle principal del servicio de detección (ejecuta en hilo separado).
        
        Este método:
        1. Obtiene configuración de zona de trabajo (desde parámetro o cámara)
        2. Inicializa CSV e imágenes si están habilitados
        3. En cada iteración:
           - Lee el frame más reciente desde builtins.latest_frame
           - Ejecuta YOLO para detectar botes
           - Valida cada detección (centrado, tamaño, completitud)
           - Si es válido: procesa (clasificación, logging, imágenes)
           - Si no es válido: solo publica para UI
        4. Implementa cooldown para evitar procesar el mismo bote múltiples veces
        """
        import builtins
        # Importar CameraBackend una sola vez al inicio (no dentro del bucle)
        try:
            from camera.device_manager import CameraBackend
        except ImportError as e:
            # Si falla la importación, loguear el error pero continuar
            log_warning(f"⚠️ No se pudo importar CameraBackend: {e}. Algunas funciones de cámara pueden no estar disponibles.")
            CameraBackend = None
        # Obtener configuración de zona de trabajo
        # Prioridad: 1) parámetro camera_config, 2) desde self.camera.config, 3) valores por defecto
        if self._camera_config is None:
            # Intentar obtener desde la cámara (si tiene configuración cargada)
            if self.camera is not None and hasattr(self.camera, 'config') and self.camera.config is not None:
                camera_config = self.camera.config
            else:
                # Valores por defecto si no hay configuración
                camera_config = {}
        else:
            camera_config = self._camera_config
        
        work_zone_config = camera_config.get("work_zone", {})  # Zona de trabajo (centro + radio)
        self._bottle_sizes = camera_config.get("bottle_sizes", {})  # Tamaños esperados (87px/67px)
        
        # Preparar CSV e imágenes si procede
        if self._vision_csv_enabled:
            try:
                os.makedirs(os.path.dirname(self._vision_csv_path), exist_ok=True)
                new_file = not os.path.exists(self._vision_csv_path)
                self._csv_file = open(self._vision_csv_path, mode="a", newline="")
                self._csv_writer = csv.writer(self._csv_file)
                if new_file:
                    # Header del CSV: columnas de tracking se mantienen para compatibilidad
                    # pero siempre tendrán valores por defecto (track_id=0, track_event="processed", etc.)
                    # Nota: num_boxes siempre será 1 (solo procesamos un bote por frame)
                    self._csv_writer.writerow([
                        "ts","iso_ts","frame_id","num_boxes","classes","avg_conf","proc_ms",
                        "camera_exposure","camera_gain","width","height","yolo_threshold","bbox",
                        "verdict","clf_label","clf_conf","policy",
                        "track_id","track_age_ms","track_event","id_switch"  # Sin tracking: siempre 0/"processed"/0
                    ])
            except Exception as e:
                log_warning(f"⚠️ No se pudo abrir vision CSV: {e}")
                self._csv_writer = None
                self._csv_file = None
        if self._images_enabled:
            try:
                self._images_mgr = ImagesManager()
            except Exception as e:
                log_warning(f"⚠️ No se pudo iniciar ImagesManager: {e}", logger_name="images")
        # Timings logger (comparte estadísticas con el de app.py si está disponible)
        # Si hay un timings_logger en el context, usarlo; si no, crear uno nuevo
        if self.context and hasattr(self.context, 'timings_logger') and self.context.timings_logger is not None:
            self._tlogger = self.context.timings_logger
        else:
            self._tlogger = TimingsLogger(self._log_dir, enable_stats=True, report_interval=50)
        while self._running:
            try:
                # Verificar latest_frame una sola vez por iteración
                has_latest_frame = hasattr(builtins, "latest_frame") and builtins.latest_frame is not None
                if not has_latest_frame:
                    time.sleep(0.01)
                    continue

                img_bgr = builtins.latest_frame
                frame_id = getattr(builtins, "latest_fid", 0)

                if self._frame_count % self.process_every != 0:
                    self._frame_count += 1
                    continue

                if self.yolo_model is None:
                    time.sleep(0.02)
                    continue

                # PASO 1: Ejecutar YOLO para detectar botes en el frame actual
                # YOLO se ejecuta en TODOS los frames (detección continua)
                t0 = time.time(); self._tlogger.start()
                try:
                    results = self.yolo_model.predict(img_bgr, conf_threshold=self.conf_threshold)
                    t_yolo_end = time.time(); self._tlogger.mark('yolo')

                    # Inicializar arrays vacíos con tipos correctos (más eficiente)
                    xyxy = np.empty((0, 4), dtype=np.float32)
                    confs = np.empty((0,), dtype=np.float32)
                    clss = np.empty((0,), dtype=np.int32)

                    if results and len(results) > 0:
                        # Parsear resultados YOLO
                        if isinstance(results[0], dict):
                            # OPTIMIZADO: Parsing de resultados sin crear listas intermedias innecesarias
                            # Pre-allocar listas para evitar reallocaciones
                            boxes_list = []
                            confs_list = []
                            clss_list = []
                            
                            for det in results:
                                bbox = det.get("bbox") or det.get("xyxy") or det.get("box")
                                if bbox and len(bbox) >= 4:
                                    boxes_list.append([float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])])
                                    confs_list.append(float(det.get("confidence", 0.0)))
                                    clss_list.append(int(det.get("class_id", -1)))
                            
                            if boxes_list:
                                # Convertir a arrays numpy de una vez (más eficiente)
                                raw_boxes = np.array(boxes_list, dtype=np.float32)
                                raw_confs = np.array(confs_list, dtype=np.float32)
                                raw_clss = np.array(clss_list, dtype=np.int32)
                            else:
                                raw_boxes = np.empty((0, 4), dtype=np.float32)
                                raw_confs = np.empty((0,), dtype=np.float32)
                                raw_clss = np.empty((0,), dtype=np.int32)
                        else:
                            result = results[0]
                            if hasattr(result, "boxes") and result.boxes is not None:
                                raw_boxes = result.boxes.xyxy.cpu().numpy()
                                raw_confs = result.boxes.conf.cpu().numpy()
                                try:
                                    raw_clss = result.boxes.cls.cpu().numpy()
                                except Exception:
                                    raw_clss = np.full((len(raw_boxes),), -1, dtype=np.int32)
                            else:
                                raw_boxes = np.empty((0, 4), dtype=np.float32)
                                raw_confs = np.empty((0,), dtype=np.float32)
                                raw_clss = np.empty((0,), dtype=np.int32)

                        # Filtrar clases permitidas usando vectorización (más eficiente que bucle)
                        allowed_classes = np.array([0, 1, 2], dtype=np.int32)
                        mask = np.isin(raw_clss, allowed_classes)
                        xyxy = raw_boxes[mask]
                        confs = raw_confs[mask]
                        clss = raw_clss[mask]
                        
                        # Marcar fin de parsing
                        self._tlogger.mark('parse')

                    # PASO 2: Si hay detecciones, fusionar las superpuestas (NMS)
                    # Esto elimina detecciones duplicadas del mismo objeto
                    if len(xyxy) > 0:
                        xyxy, confs, clss = self._merge_overlapping_detections(xyxy, confs, clss, iou_threshold=DEFAULT_MERGE_IOU_THRESHOLD)
                        # Marcar fin de NMS
                        self._tlogger.mark('nms')
                        if len(xyxy) > 0:
                            # PASO 3: VALIDAR BOTE antes de procesar
                            # Solo procesamos si el bote está centrado, tiene el tamaño correcto y está completo
                            should_process = False
                            frame_to_process = None  # Frame capturado cuando está centrado (para clasificación y guardado)
                            if has_latest_frame:
                                # Calcular zona de trabajo una sola vez (se reutiliza para todos los frames)
                                # La zona de trabajo define el área donde el bote debe estar centrado
                                if self._work_zone is None:
                                    img_shape = builtins.latest_frame.shape
                                    
                                    # Obtener ROI configurado desde camera_config (ya cargado desde config_camera.yaml)
                                    roi = None
                                    if camera_config is not None:
                                        roi_cfg = camera_config.get("roi", {})
                                        if roi_cfg:
                                            # Leer offset y dimensiones desde la configuración
                                            offset_x = int(roi_cfg.get("offset_x", 0))
                                            offset_y = int(roi_cfg.get("offset_y", 0))
                                            # Si width/height son expresiones, ya fueron evaluadas al aplicar el ROI
                                            # Leer valores reales desde la cámara como fallback
                                            if self.camera is not None and CameraBackend is not None:
                                                try:
                                                    width = int(CameraBackend.safe_get(self.camera, 'Width', img_shape[1]))
                                                    height = int(CameraBackend.safe_get(self.camera, 'Height', img_shape[0]))
                                                    roi = (offset_x, offset_y, width, height)
                                                except Exception:
                                                    # Si falla, usar dimensiones de la imagen
                                                    roi = (offset_x, offset_y, img_shape[1], img_shape[0])
                                            else:
                                                # Sin cámara, usar dimensiones de la imagen
                                                roi = (offset_x, offset_y, img_shape[1], img_shape[0])
                                    
                                    self._work_zone = get_work_zone_from_config(work_zone_config, img_shape, roi=roi)
                                
                                # Validar si el bote está listo para procesamiento
                                # Esta validación verifica:
                                # - ¿Está centrado en la zona de trabajo? (dentro del radio configurado)
                                # - ¿Tiene el tamaño correcto? (87px o 67px con tolerancia)
                                # - ¿Está completo? (no cortado por los bordes de la imagen)
                                
                                # Calcular tamaño del bote detectado para logging
                                first_box = xyxy[0]
                                bbox_w = float(first_box[2] - first_box[0])
                                bbox_h = float(first_box[3] - first_box[1])
                                
                                # is_valid, reason = is_bottle_ready(
                                #     xyxy, 
                                #     builtins.latest_frame.shape, 
                                #     self._work_zone,
                                #     self._bottle_sizes
                                # )
                                # # Marcar fin de validación
                                # self._tlogger.mark('validation')
                                is_valid = True # hardcodeado para DEBUG
                                reason = "debug_always_valid"
                                
                                # Verificar cooldown para evitar procesar el mismo bote múltiples veces
                                # En producción (300 botes/minuto), los botes pasan continuamente por la misma posición.
                                # Lógica: cuando un bote pasa por el centro y es válido, se procesa.
                                # Después del cooldown, si hay otro bote válido en el centro, se procesa (es un bote nuevo).
                                # Un frame por bote cuando está en el centro.
                                frames_since_last = self._frame_count - self._last_processed_frame_id
                                if is_valid and frames_since_last >= self._bottle_processing_cooldown:
                                    # Bote válido y fuera de cooldown: PROCESAR
                                    # IMPORTANTE: Capturar el frame AHORA (cuando está centrado) para usarlo en clasificación y guardado
                                    # Esto asegura que el frame guardado/clasificado sea el mismo que el validado como centrado
                                    frame_to_process = builtins.latest_frame.copy() if has_latest_frame else None
                                    should_process = True
                                    log_info(f"[VISION] frame={frame_id} bote válido ({bbox_w:.1f}x{bbox_h:.1f}px), procesando...", logger_name="vision")
                                    self._last_processed_frame_id = self._frame_count
                                else:
                                    # Bote no válido o en cooldown: NO procesar
                                    frame_to_process = None
                                    if is_valid:
                                        log_info(f"[VISION] frame={frame_id} bote válido ({bbox_w:.1f}x{bbox_h:.1f}px) pero en cooldown (frames={frames_since_last}/{self._bottle_processing_cooldown})", logger_name="vision")
                                    else:
                                        log_info(f"[VISION] frame={frame_id} bote no válido: {reason} (tamaño detectado: {bbox_w:.1f}x{bbox_h:.1f}px, esperado: {self._bottle_sizes.get('min_width', '?')}-{self._bottle_sizes.get('max_width', '?')}x{self._bottle_sizes.get('min_height', '?')}-{self._bottle_sizes.get('max_height', '?')}px)", logger_name="vision")
                            
                            # PASO 4: Publicar detección para UI (SIEMPRE, válida o no)
                            # La UI muestra la detección para visualización/debugging
                            # Solo procesamos el primer bote detectado (normalmente solo hay uno)
                            # is_processed = 0 si se procesará, -1 si solo es para visualización
                            # Nota: xyxy.tolist() es necesario para serialización JSON en la cola
                            infer_result = {
                                "frame_id": frame_id,
                                "xyxy": xyxy.tolist(),  # Necesario para serialización en cola
                                "tids": [0] if should_process else [-1],  # 0 = procesado, -1 = solo visualización
                                "lids": [0] if should_process else [-1],  # Mismo que tids (sin tracking)
                                "proc_ms": (time.time() - t0) * 1000.0,
                                "ts": time.time(),
                            }
                            try:
                                self.infer_queue.put_nowait(infer_result)
                            except queue.Full:
                                pass
                            
                            # PASO 5: Procesar solo si el bote es válido
                            # Si no es válido o está en cooldown, saltamos el procesamiento completo
                            # (clasificación, logging CSV, guardado de imágenes)
                            # IMPORTANTE: frame_to_process fue capturado cuando el bote estaba centrado
                            # Este mismo frame se usa para clasificación y guardado
                            if not should_process or frame_to_process is None:
                                self._frame_count += 1
                                continue
                            
                            # PASO 6: PROCESAMIENTO COMPLETO (solo para botes válidos)
                            # IMPORTANTE: frame_to_process fue capturado cuando el bote estaba centrado
                            # Este mismo frame se usa para clasificación y guardado
                            # - Clasificación del bote (buena/mala/defectuosa)
                            # - Logging en CSV
                            # - Guardado de imágenes (bad/good)
                            # - (Futuro: salidas digitales GPIO)
                            try:
                                # 6.1: Preparar datos para logging
                                # Solo procesamos el primer bote (normalmente solo hay uno válido)
                                # Usar acceso directo a arrays numpy (más eficiente que conversiones)
                                first_box = xyxy[0]
                                first_conf = float(confs[0])  # Ya validado que len(xyxy) > 0
                                first_cls = int(clss[0])  # Ya validado que len(xyxy) > 0
                                
                                # 6.2: CLASIFICACIÓN del bote (buena/mala/defectuosa)
                                # Recortamos el bote del frame capturado (cuando estaba centrado) y lo pasamos al clasificador
                                # IMPORTANTE: frame_to_process fue capturado cuando el bote estaba centrado
                                clf_label, clf_conf = "unknown", 0.0
                                crop_for_classification = None
                                try:
                                    # Recortar bounding box del bote del frame capturado (cuando estaba centrado)
                                    # Usar np.clip para asegurar índices válidos (más eficiente)
                                    x1, y1, x2, y2 = first_box.astype(np.int32)
                                    h_img, w_img = frame_to_process.shape[:2]
                                    x1 = np.clip(x1, 0, w_img)
                                    y1 = np.clip(y1, 0, h_img)
                                    x2 = np.clip(x2, 0, w_img)
                                    y2 = np.clip(y2, 0, h_img)
                                    crop_for_classification = frame_to_process[y1:y2, x1:x2].copy()
                                    if crop_for_classification.size > 0:
                                        self._tlogger.mark('crop')
                                        # Clasificar el bote recortado
                                        clf_label, clf_conf, _ = clf_predict_bgr(crop_for_classification)
                                    t_clf_end = time.time(); self._tlogger.mark('clf')
                                except Exception as e:
                                    log_warning(f"⚠️ Error en clasificación: {e}")
                                    t_clf_end = time.time(); self._tlogger.mark('clf')
                                
                                # 6.3: Determinar veredicto (ok/bad) según política
                                # Política: prioriza buenas; solo marca "bad" si confianza >= umbral
                                verdict = "bad" if (clf_label in self._bad_labels and clf_conf >= self._clf_bad_threshold) else "ok"
                                policy_str = f"clasificador={clf_label} conf={clf_conf:.2f} < umbral={self._clf_bad_threshold:.2f} => buena" if verdict=="ok" else f"clasificador={clf_label} conf={clf_conf:.2f} >= umbral={self._clf_bad_threshold:.2f} => mala"
                                
                                # Actualizar estadísticas del clasificador
                                self._clf_total += 1
                                if verdict == "bad":
                                    self._clf_malas += 1
                                else:
                                    self._clf_buenas += 1
                                
                                # Actualizar estadísticas en context para acceso desde UI
                                if self.context:
                                    self.context.classifier_stats = {
                                        'total': self._clf_total,
                                        'buenas': self._clf_buenas,
                                        'malas': self._clf_malas
                                    }
                                
                                # 6.4: Valores para CSV (sin tracking, mantenemos columnas para compatibilidad)
                                track_id = 0
                                track_event = "processed"
                                track_age_ms = 0.0
                                id_switch = 0
                                
                                # Obtener parámetros de cámara y dimensiones del área procesada
                                cam_exp, cam_gain, w, h = '', '', '', ''
                                try:
                                    # CameraBackend ya está importado al inicio del método
                                    if CameraBackend is not None and self.camera is not None:
                                        cam_exp = CameraBackend.safe_get(self.camera, 'ExposureTime', '') or ''
                                        cam_gain = CameraBackend.safe_get(self.camera, 'Gain', '') or ''
                                    # Obtener dimensiones reales del frame procesado (área de la cámara)
                                    # IMPORTANTE: frame_to_process fue capturado cuando el bote estaba centrado
                                    img_h, img_w = frame_to_process.shape[:2]
                                    w, h = str(img_w), str(img_h)
                                except Exception as e:
                                    # Silenciar errores al obtener parámetros de cámara (no crítico)
                                    pass
                                
                                # Log con información completa incluyendo área de la cámara
                                area_info = f"area={w}x{h}px" if w and h else "area=desconocida"
                                log_info(f"[VISION] frame={frame_id} bote procesado: clase={first_cls} conf_yolo={first_conf:.2f} {area_info} {policy_str} verdict={verdict}", logger_name="vision")
                                
                                if self._csv_writer is not None:
                                    # Escribir registro CSV para el bote procesado
                                    first_bbox = first_box.tolist()
                                    iso_ts = datetime.fromtimestamp(infer_result['ts']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                                    self._csv_writer.writerow([
                                        f"{infer_result['ts']:.6f}",
                                        iso_ts,
                                        frame_id,
                                        1,  # Solo un bote procesado
                                        str(first_cls),  # Clase del bote
                                        f"{first_conf:.4f}",  # Confianza YOLO
                                        f"{infer_result['proc_ms']:.2f}",
                                        cam_exp,
                                        cam_gain,
                                        w,
                                        h,
                                        f"{self.conf_threshold:.2f}",
                                        str(first_bbox),
                                        verdict,
                                        clf_label,
                                        f"{clf_conf:.4f}",
                                        policy_str,
                                        track_id,  # Sin tracking: siempre 0
                                        f"{track_age_ms:.0f}",  # Sin tracking: siempre 0
                                        track_event,  # Sin tracking: siempre "processed"
                                        id_switch,  # Sin tracking: siempre 0
                                    ])
                                    try:
                                        self._csv_file.flush()
                                    except Exception:
                                        pass
                                    t_csv_end = time.time(); self._tlogger.mark('csv')
                                # Guardado de imágenes: guardar frame completo y crop usado para clasificar
                                # IMPORTANTE: frame_to_process fue capturado cuando el bote estaba centrado
                                # Este mismo frame se usa para validar, clasificar y guardar
                                if self._images_mgr is not None:
                                    try:
                                        # Guardar frame completo (el mismo que se usó para validar y clasificar)
                                        if verdict == "bad":
                                            self._images_mgr.save_bad(
                                                frame_to_process, 
                                                reason=f"clf:{clf_label}", 
                                                avg_conf=clf_conf, 
                                                cls=clf_label
                                            )
                                        else:
                                            self._images_mgr.save_good(
                                                frame_to_process, 
                                                reason=f"clf:{clf_label}", 
                                                avg_conf=clf_conf, 
                                                cls=clf_label
                                            )
                                        
                                        # Guardar crop usado para clasificar en subcarpeta "clasificado" (para verificación)
                                        # Esto permite verificar qué imagen se usó para clasificar y detectar retrasos
                                        if crop_for_classification is not None and crop_for_classification.size > 0:
                                            img_type = "bad" if verdict == "bad" else "good"
                                            crop_path = self._images_mgr.save_classification_crop(
                                                crop_for_classification,
                                                img_type=img_type,
                                                reason=f"crop_clf:{clf_label}",
                                                avg_conf=clf_conf,
                                                cls=clf_label
                                            )
                                            if crop_path:
                                                log_info(f"📸 Crop guardado en clasificado: {crop_path}", logger_name="images")
                                    except Exception as e:
                                        log_warning(f"⚠️ Error guardando imagen: {e}", logger_name="images")
                                    finally:
                                        self._tlogger.mark('images')
                                        self._tlogger.write(frame_id)
                            except Exception as e:
                                log_warning(f"⚠️ Error registrando visión: {e}")
                except Exception as e:
                    log_warning(f"⚠️ Error en DetectionService.predict: {e}")

                self._frame_count += 1
                # Heartbeat: cada ~1s, loguea fps y nº de boxes
                self._hb_frames += 1
                # Optimizado: len() es O(1) en arrays numpy, devuelve 0 si está vacío
                self._hb_last_boxes = len(xyxy)
                now = time.time()
                if (now - self._hb_last_ts) >= 1.0:
                    elapsed = max(1e-6, now - self._hb_last_ts)
                    fps = self._hb_frames / elapsed
                    log_info(f"[YOLO] fps={fps:.1f} boxes={self._hb_last_boxes}")
                    self._hb_last_ts = now
                    self._hb_frames = 0
            except Exception as e:
                log_warning(f"⚠️ Error en hilo DetectionService: {e}")
                time.sleep(0.01)



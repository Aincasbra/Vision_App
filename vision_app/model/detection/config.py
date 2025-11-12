"""
Configuración y carga de YOLO
------------------------------
- Define valores por defecto para YOLO (conf, iou, weights, image_size).
- Lee configuración desde settings/YAML y carga el modelo YOLO.
- Proporciona función de post-procesamiento para fusionar detecciones superpuestas.
- Funciones principales:
  * `load_yolo_config()`: lee conf_threshold e iou_threshold desde settings
  * `get_yolo_model_config()`: lee configuración del modelo (path, image_size, classes)
  * `load_yolo_model()`: carga el modelo YOLO con configuración desde settings
  * `merge_overlapping_detections()`: fusiona detecciones superpuestas usando IoU
- Llamado desde:
  * `app.py`: llama a `load_yolo_config()` y `load_yolo_model()` para inicializar YOLO
  * `model/detection/detection_service.py`: llama a `merge_overlapping_detections()` y usa
    `DEFAULT_MERGE_IOU_THRESHOLD` para post-procesamiento
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
from core.logging import log_info, log_warning, log_error
from model.detection.yolo_wrapper import YOLOPyTorchCUDA

# Valores por defecto para YOLO
DEFAULT_CONF_THRESHOLD = 0.4
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_WEIGHTS = "v2_yolov8n_HERMASA_finetune.pt"
DEFAULT_IMAGE_SIZE = 640
DEFAULT_PROCESS_EVERY = 1

# IOU threshold para merge de detecciones superpuestas
DEFAULT_MERGE_IOU_THRESHOLD = 0.7


def load_yolo_config(context: Any) -> Tuple[float, float]:
    """Lee configuración YOLO desde settings y devuelve (conf_threshold, iou_threshold).
    
    Args:
        context: AppContext con settings
        
    Returns:
        Tuple[float, float]: (confidence_threshold, iou_threshold)
    """
    try:
        yolo_cfg: Dict[str, Any] = dict(getattr(context, "settings", None).yolo)
    except Exception:
        yolo_cfg = {}
    
    conf_threshold = float(yolo_cfg.get("confidence_threshold", DEFAULT_CONF_THRESHOLD))
    iou_threshold = float(yolo_cfg.get("iou_threshold", DEFAULT_IOU_THRESHOLD))
    
    return conf_threshold, iou_threshold


def get_yolo_model_config(context: Any) -> Dict[str, Any]:
    """Lee configuración del modelo YOLO desde settings.
    
    Args:
        context: AppContext con settings
        
    Returns:
        Dict con 'model_path', 'image_size', 'classes'
    """
    try:
        yolo_cfg: Dict[str, Any] = dict(getattr(context, "settings", None).yolo)
    except Exception:
        yolo_cfg = {}
    
    return {
        "model_path": str(yolo_cfg.get('model_path') or yolo_cfg.get('model') or DEFAULT_WEIGHTS),
        "image_size": yolo_cfg.get('image_size', DEFAULT_IMAGE_SIZE),
        "classes": yolo_cfg.get('classes', []),
    }


def load_yolo_model(context: Any) -> Optional[YOLOPyTorchCUDA]:
    """Carga el modelo YOLO con configuración desde settings.
    
    Args:
        context: AppContext con settings
        
    Returns:
        YOLOPyTorchCUDA o None si falla
    """
    import torch
    
    # Log de dispositivo
    if torch.cuda.is_available():
        log_info(f"🚀 Cargando YOLO con CUDA: {torch.cuda.get_device_name(0)}")
        log_info(f"🚀 Memoria GPU antes de YOLO: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    else:
        log_warning("⚠️ Cargando YOLO con CPU (sin CUDA)")
    
    try:
        # Leer configuración del modelo desde settings
        model_cfg = get_yolo_model_config(context)
        weights_path = model_cfg["model_path"]
        imgsz_cfg = model_cfg["image_size"]
        
        # Procesar image_size
        if isinstance(imgsz_cfg, (list, tuple)) and len(imgsz_cfg) > 0:
            imgsz_val = int(imgsz_cfg[0])
        else:
            imgsz_val = int(imgsz_cfg)

        # Crear modelo
        yolo_model = YOLOPyTorchCUDA(weights_path, imgsz_val)
        log_info(f"✅ Modelo YOLO cargado: {weights_path} (imgsz={imgsz_val})")
        
        # Configurar modo evaluación
        if hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'eval'):
            yolo_model.model.eval()
            log_info("✅ Modelo configurado en modo evaluación")
        
        # Configurar clases desde YAML si existen
        class_list = list(model_cfg.get('classes', []))
        if class_list:
            MODEL_NAMES = {i: name for i, name in enumerate(class_list)}
            KEEP_IDX = set(range(len(class_list)))
            log_info(f"✅ Clases del modelo (YAML): {MODEL_NAMES}")
        else:
            MODEL_NAMES = {0: 'can', 1: 'hand'}
            KEEP_IDX = {0, 1}
            log_info(f"ℹ️ Clases por defecto: {MODEL_NAMES}")
        
        return yolo_model
        
    except Exception as e:
        log_error(f"❌ Error cargando modelo YOLO: {e}")
        return None


def merge_overlapping_detections(xyxy, confs, clss, iou_threshold=DEFAULT_MERGE_IOU_THRESHOLD):
    """Fusiona detecciones superpuestas basándose en IoU y clase.
    
    Agrupa detecciones con IoU >= threshold y misma clase, manteniendo solo
    la de mayor confianza de cada grupo.
    
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
    xyxy = np.asarray(xyxy)
    confs = np.asarray(confs)
    clss = np.asarray(clss)
    keep = []
    used = np.zeros(len(xyxy), dtype=bool)
    for i in range(len(xyxy)):
        if used[i]:
            continue
        xi1, yi1, xi2, yi2 = xyxy[i]
        ai = max(0, (xi2 - xi1)) * max(0, (yi2 - yi1))
        group = [i]
        used[i] = True
        for j in range(i + 1, len(xyxy)):
            if used[j]:
                continue
            xj1, yj1, xj2, yj2 = xyxy[j]
            aj = max(0, (xj2 - xj1)) * max(0, (yj2 - yj1))
            xx1 = max(xi1, xj1)
            yy1 = max(yi1, yj1)
            xx2 = min(xi2, xj2)
            yy2 = min(yi2, yj2)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = ai + aj - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold and clss[i] == clss[j]:
                group.append(j)
                used[j] = True
        best_idx = max(group, key=lambda k: confs[k])
        keep.append(best_idx)
    keep = np.array(keep, dtype=int)
    return xyxy[keep], confs[keep], clss[keep]


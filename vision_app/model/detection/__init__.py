"""
Detection module
----------------
- Módulo de detección YOLO: servicio de inferencia, configuración, carga y wrapper
"""
from model.detection.detection_service import DetectionService
from model.detection.yolo_wrapper import YOLOPyTorchCUDA
from model.detection.config import (
    load_yolo_config,
    get_yolo_model_config,
    load_yolo_model,
    merge_overlapping_detections,
    DEFAULT_MERGE_IOU_THRESHOLD,
)

__all__ = [
    'DetectionService',
    'YOLOPyTorchCUDA',
    'load_yolo_config',
    'get_yolo_model_config',
    'load_yolo_model',
    'merge_overlapping_detections',
    'DEFAULT_MERGE_IOU_THRESHOLD',
]

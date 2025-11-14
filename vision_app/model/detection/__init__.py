"""
Detection module
----------------
- Módulo de detección YOLO: servicio de inferencia y wrapper
"""
from model.detection.detection_service import DetectionService
from model.detection.yolo_wrapper import YOLOPyTorchCUDA

__all__ = [
    'DetectionService',
    'YOLOPyTorchCUDA',
]

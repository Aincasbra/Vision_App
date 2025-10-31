"""
Optimizations
-------------
- Aplica optimizaciones de CUDA/OpenCV/PyTorch y devuelve
  umbrales de YOLO desde settings.
- Centralizar tuning de rendimiento.
- Se llama desde `gentl/app.py` en la inicialización.
"""
"""
Optimizations: centraliza ajustes de rendimiento (CUDA/OpenCV/PyTorch)
para ser invocados por `App._apply_optimizations`.
"""
from __future__ import annotations

from typing import Tuple, Dict, Any

from core.device import apply_system_optimizations
from core.logging import log_info, log_warning


def apply_all(context) -> Tuple[float, float]:
    """Aplica optimizaciones del sistema y devuelve (conf, iou) para YOLO.

    Lee umbrales desde `context.settings.yolo` si está disponible.
    """
    import torch
    import cv2

    # Optimizaciones de SO
    apply_system_optimizations()

    # Umbrales YOLO desde settings (con defaults seguros)
    try:
        yolo_cfg: Dict[str, Any] = dict(getattr(context, "settings", None).yolo)
    except Exception:
        yolo_cfg = {}
    yolo_conf = float(yolo_cfg.get("confidence_threshold", 0.4))
    yolo_iou = float(yolo_cfg.get("iou_threshold", 0.7))

    # Log base
    log_info("🚀 APLICANDO OPTIMIZACIONES PARA ALTA VELOCIDAD (300 latas/min)")

    # Optimizaciones CUDA
    if torch.cuda.is_available():
        log_info("🚀 HABILITANDO OPTIMIZACIONES CUDA UNIFICADAS")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.empty_cache()
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)
        except Exception:
            pass
        cv2.setUseOptimized(True)
        cv2.setNumThreads(0)
        log_info("✅ Optimizaciones CUDA unificadas habilitadas")
    else:
        log_warning("⚠️ CUDA no disponible, usando optimizaciones CPU")

    return yolo_conf, yolo_iou



"""
Overlay de visión y procesamiento de imagen (UI desarrollador)
---------------------------------------------------------------
- Procesamiento visual de imágenes antes de mostrar en la UI de depuración.
- Funciones principales:
  * `apply_gamma_from_state()`: aplica corrección gamma a la imagen usando LUT
  * `apply_yolo_overlay()`: dibuja cajas, textos y HUD sobre la imagen con resultados YOLO
- Llamado desde:
  * `app.py`: llama a `apply_gamma_from_state()` antes de mostrar el frame y a
    `apply_yolo_overlay()` para dibujar las detecciones sobre la imagen
"""
import time
import builtins
from typing import Optional

import cv2
import numpy as np

from core.logging import log_warning


# Estado interno para mantener el último resultado y evitar parpadeos
_last_infer = None
_last_draw_ts = 0.0


def apply_gamma_from_state(img_bgr, gamma_value=1.0):
    """Aplica corrección gamma a la imagen usando LUT.
    
    Args:
        img_bgr: Imagen BGR de entrada
        gamma_value: Valor de gamma (1.0 = sin corrección)
        
    Returns:
        Imagen BGR con corrección gamma aplicada
    """
    try:
        if abs(gamma_value - 1.0) <= 1e-3:
            return img_bgr
        
        # Crear LUT para gamma
        lut = np.array([((i / 255.0) ** (1.0 / gamma_value)) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img_bgr, lut)
    except Exception:
        return img_bgr


def _safe_get_camera_attr(camera, name: str, default_value="N/A"):
    try:
        if camera is None:
            return default_value
        # Intentar API tipo diccionario/propiedades del backend
        if hasattr(camera, "get"):
            return camera.get(name)  # type: ignore[attr-defined]
        # Intentar atributos directos
        if hasattr(camera, name):
            return getattr(camera, name)
    except Exception:
        pass
    return default_value


def apply_yolo_overlay(
    image_bgr: np.ndarray,
    context,  # AppContext con infer_queue
    frame_num: Optional[int] = None,
    camera=None,
    ttl_seconds: float = 0.05,
    show_perf_hud: bool = True,
):
    """
    Aplica overlays de YOLO al frame actual usando la última inferencia disponible.
    Mantiene el último resultado hasta ttl_seconds para evitar parpadeos.

    Args:
        image_bgr: Frame BGR actual
        context: AppContext con infer_queue (Queue de resultados de YOLO)
        frame_num: ID/frame number actual (para emparejar con inferencias)
        camera: Backend de cámara (opcional) para HUD (ET/FPS)
        ttl_seconds: Tiempo de vida del último resultado para evitar parpadeo
        show_perf_hud: Mostrar HUD de latencia/FPS/boxes
    """
    global _last_infer, _last_draw_ts

    out = image_bgr

    # Consumir resultados más recientes de la cola de inferencia
    try:
        while True:
            infer = context.infer_queue.get_nowait()  # type: ignore[attr-defined]
            _last_infer = infer
            _last_draw_ts = time.time()
    except Exception:
        pass  # Cola vacía

    # Seleccionar inferencia a dibujar según frame_num/TTL
    infer = None
    if _last_infer is not None:
        lf_fid = _last_infer.get("frame_id")
        if frame_num is not None and lf_fid == frame_num:
            infer = _last_infer
        elif (
            isinstance(lf_fid, int)
            and isinstance(frame_num, int)
            and 0 <= (frame_num - lf_fid) <= 1
        ):
            infer = _last_infer
        elif (time.time() - _last_draw_ts) < ttl_seconds:
            infer = _last_infer

    if infer is None:
        # No hay datos para dibujar aún
        try:
            cv2.putText(
                out,
                "YOLO esperando...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass
        return out

    # Extraer datos de la inferencia
    xyxy = infer.get("xyxy") or []
    tids = infer.get("tids")
    lids = infer.get("lids")
    proc_ms = float(infer.get("proc_ms", 0.0))
    ts_cap = float(infer.get("ts", time.time()))

    # HUD básico de rendimiento
    if show_perf_hud:
        try:
            cap_ms = (time.time() - ts_cap) * 1000.0
            cv2.putText(
                out,
                f"LAT:{cap_ms:.0f}ms YOLO:{proc_ms:.0f}ms",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                f"YOLO ON - boxes:{len(xyxy)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            # Info de cámara si disponible
            exposure_us = _safe_get_camera_attr(camera, "ExposureTime", "N/A")
            camera_fps = _safe_get_camera_attr(camera, "AcquisitionFrameRate", "N/A")
            cv2.putText(
                out,
                f"ET:{exposure_us}us FPS:{camera_fps}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
                cv2.LINE_AA,
            )
        except Exception:
            pass

    # ROI y línea de decisión (si existe en builtins)
    try:
        rx, ry, rw, rh = getattr(builtins, "roi", (0, 0, out.shape[1], out.shape[0]))
        decision_line_y = int(0.7 * rh)
        cv2.line(out, (rx, ry + decision_line_y), (rx + rw, ry + decision_line_y), (0, 255, 255), 1)
    except Exception:
        pass

    if len(xyxy) == 0:
        return out

    # Normalizar arrays
    xyxy = np.asarray(xyxy)
    if tids is None or len(tids) != len(xyxy):
        tids = np.full((len(xyxy),), -1, dtype=int)
    if lids is None or len(lids) != len(xyxy):
        lids = np.arange(len(xyxy), dtype=int) + 1

    # Dibujo de cajas e IDs
    try:
        for i, (box, rid, lid) in enumerate(zip(xyxy, tids, lids)):
            x1, y1, x2, y2 = map(int, box)
            key_id = int(lid) if (lid is not None and int(lid) >= 0) else int(rid)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                out,
                f"ID:{key_id}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
                cv2.LINE_AA,
            )
    except Exception as e:
        log_warning(f"⚠️ Error dibujando overlay YOLO: {e}")

    return out

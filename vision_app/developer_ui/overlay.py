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

    # NOTA: Línea de decisión legacy eliminada - ya no se usa para validación.
    # La validación ahora se hace con la línea central de work_zone (ver más abajo).
    
    # Dibujar línea central de zona de trabajo (work_zone) desde configuración de cámara
    # Esta es la línea central (horizontal o vertical) donde el bote debe estar centrado
    # IMPORTANTE: La línea mostrada aquí es la MISMA que se usa para validar botes
    # en detection_service.py, garantizando que lo que ves en la UI coincide con
    # la configuración real que se está usando para procesar
    try:
        if camera is not None and hasattr(camera, 'config') and camera.config is not None:
            from model.detection.validation import get_work_zone_from_config
            work_zone_config = camera.config.get("work_zone", {})
            if work_zone_config:
                h, w = out.shape[:2]
                
                # Obtener ROI configurado desde camera.config (ya cargado desde config_camera.yaml)
                roi = None
                try:
                    roi_cfg = camera.config.get("roi", {}) if camera.config else {}
                    if roi_cfg:
                        offset_x = int(roi_cfg.get("offset_x", 0))
                        offset_y = int(roi_cfg.get("offset_y", 0))
                        # Leer valores reales desde la cámara (ya evaluados al aplicar el ROI)
                        width = int(CameraBackend.safe_get(camera, 'Width', w))
                        height = int(CameraBackend.safe_get(camera, 'Height', h))
                        roi = (offset_x, offset_y, width, height)
                except Exception:
                    # Si falla, usar None (se calculará desde img_shape)
                    roi = None
                
                work_zone = get_work_zone_from_config(work_zone_config, (h, w), roi=roi)
                if work_zone:
                    # Obtener eje y tolerancia
                    axis = work_zone.get("axis", "vertical").lower()
                    tolerance = work_zone.get("tolerance", 30)
                    
                    # Obtener posición de la línea central (ya calculada en get_work_zone_from_config)
                    center_position = work_zone.get("center_position")
                    
                    if axis == "vertical":
                        # Línea horizontal: cinta se mueve horizontalmente, validar Y
                        # Usar posición configurada (offset_y + height/2) o mitad por defecto si no se especificó
                        if center_position is None:
                            center_y = h / 2.0
                        else:
                            center_y = float(center_position)
                        
                        # Dibujar línea central horizontal (verde, más gruesa para mejor visibilidad)
                        cv2.line(out, (0, int(center_y)), (w, int(center_y)), (0, 255, 0), 2)
                        
                        # Dibujar zona de tolerancia (líneas horizontales arriba y abajo de la central)
                        if tolerance > 0:
                            # Línea superior (tolerancia)
                            cv2.line(out, (0, int(center_y - tolerance)), (w, int(center_y - tolerance)), (0, 255, 255), 1)
                            # Línea inferior (tolerancia)
                            cv2.line(out, (0, int(center_y + tolerance)), (w, int(center_y + tolerance)), (0, 255, 255), 1)
                        
                        # Texto indicativo en el lado izquierdo
                        cv2.putText(
                            out,
                            f"CENTER LINE (Y={int(center_y)}, tol: {tolerance}px)",
                            (10, int(center_y - tolerance - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
                    else:
                        # Línea vertical: cinta se mueve verticalmente, validar X
                        # Usar posición configurada (offset_x + width/2) o mitad por defecto si no se especificó
                        if center_position is None:
                            center_x = w / 2.0
                        else:
                            center_x = float(center_position)
                        
                        # Dibujar línea central vertical (verde, más gruesa para mejor visibilidad)
                        cv2.line(out, (int(center_x), 0), (int(center_x), h), (0, 255, 0), 2)
                        
                        # Dibujar zona de tolerancia (líneas verticales a ambos lados de la central)
                        if tolerance > 0:
                            # Línea izquierda (tolerancia)
                            cv2.line(out, (int(center_x - tolerance), 0), (int(center_x - tolerance), h), (0, 255, 255), 1)
                            # Línea derecha (tolerancia)
                            cv2.line(out, (int(center_x + tolerance), 0), (int(center_x + tolerance), h), (0, 255, 255), 1)
                        
                        # Texto indicativo en la parte superior
                        cv2.putText(
                            out,
                            f"CENTER LINE (X={int(center_x)}, tol: {tolerance}px)",
                            (int(center_x - 100), 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                            cv2.LINE_AA,
                        )
    except Exception as e:
        # Silenciar errores de visualización (no crítico)
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
    # Color de la caja indica si el bote está dentro de la zona de trabajo:
    # - Verde: dentro de la zona (tids/lids = 0, se procesará)
    # - Amarillo: fuera de la zona o en cooldown (tids/lids = -1, solo visualización)
    try:
        for i, (box, rid, lid) in enumerate(zip(xyxy, tids, lids)):
            x1, y1, x2, y2 = map(int, box)
            key_id = int(lid) if (lid is not None and int(lid) >= 0) else int(rid)
            
            # Determinar color según si está dentro de la zona de trabajo
            # tids/lids = 0 significa que se procesará (dentro de zona y válido)
            # tids/lids = -1 significa que NO se procesará (fuera de zona o en cooldown)
            is_in_work_zone = (rid is not None and int(rid) >= 0) or (lid is not None and int(lid) >= 0)
            box_color = (0, 255, 0) if is_in_work_zone else (0, 255, 255)  # Verde o Amarillo
            text_color = (0, 255, 0) if is_in_work_zone else (0, 255, 255)
            status_text = "OK" if is_in_work_zone else "OUT"
            
            # Dibujar bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)
            
            # Dibujar centro del bounding box (cruz) para verificar centrado
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            # Cruz más grande para mejor visibilidad
            cross_size = 8
            cv2.line(out, (cx - cross_size, cy), (cx + cross_size, cy), box_color, 2)
            cv2.line(out, (cx, cy - cross_size), (cx, cy + cross_size), box_color, 2)
            
            # Texto con información del bote
            cv2.putText(
                out,
                f"ID:{key_id} {status_text}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
                cv2.LINE_AA,
            )
    except Exception as e:
        log_warning(f"⚠️ Error dibujando overlay YOLO: {e}")

    return out

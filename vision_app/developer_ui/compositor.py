"""
Compositor de UI de desarrollador
---------------------------------
- Combina frame de cámara y panel de control en una sola imagen.
- Apoya a `developer_ui/window.py` y `app.py` para mostrar la interfaz.
"""
import numpy as np
import cv2
from developer_ui.panel import crear_panel_control_stviewer, actualizar_panel_control


def compose_with_panel(image_bgr, cam=None, acquisition_running=False, gamma_actual=0.8, patron_actual="BG", yolo_stats=None):
    """Compone la vista principal con el panel lateral actualizado, aplicando padding vertical mínimo.
    No modifica el ancho original de la imagen de cámara.
    """
    # Obtener panel base
    panel_base = crear_panel_control_stviewer(image_bgr.shape[1], image_bgr.shape[0])
    
    # Actualizar panel con información en tiempo real
    panel = actualizar_panel_control(
        panel_base, 
        metricas={}, 
        estado_camara=acquisition_running, 
        img_width=image_bgr.shape[1], 
        img_height=image_bgr.shape[0],
        gamma_actual=gamma_actual,
        patron_actual=patron_actual,
        yolo_stats=yolo_stats,
        cam=cam
    )
    
    MIN_UI_H = 720
    h_img, w_img = image_bgr.shape[:2]
    h_pan, w_pan = panel.shape[:2]
    canvas_h = max(h_img, h_pan, MIN_UI_H)
    # padding imagen
    if h_img < canvas_h:
        pad = np.zeros((canvas_h - h_img, w_img, 3), dtype=image_bgr.dtype)
        img_padded = np.vstack([image_bgr, pad])
    else:
        img_padded = image_bgr
    # padding panel
    if h_pan < canvas_h:
        padp = np.zeros((canvas_h - h_pan, w_pan, 3), dtype=panel.dtype)
        panel_padded = np.vstack([panel, padp])
    else:
        panel_padded = panel
    return np.hstack([img_padded, panel_padded])

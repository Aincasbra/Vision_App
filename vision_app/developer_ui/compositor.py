"""
Compositor de UI de desarrollador
---------------------------------
- Combina frame de cámara y panel de control en una sola imagen.
- Apoya a `developer_ui/window.py` y `app.py` para mostrar la interfaz.
"""
import numpy as np
import cv2
from developer_ui.panel import crear_panel_control_stviewer, actualizar_panel_control


def compose_with_panel(image_bgr, cam=None, acquisition_running=False, gamma_actual=0.8, patron_actual="BG", yolo_stats=None, context=None):
    """Compone la vista principal con el panel lateral actualizado, aplicando padding vertical mínimo.
    No modifica el ancho original de la imagen de cámara.
    """
    # Obtener panel base
    h_img, w_img = image_bgr.shape[:2]
    panel_base = crear_panel_control_stviewer(w_img, h_img)
    
    # Obtener valores de confianza importando desde los módulos correspondientes
    from model.detection.detection_service import load_yolo_config_from_yaml
    from model.classifier.multiclass import load_classifier_config_from_yaml
    
    yolo_cfg = load_yolo_config_from_yaml()
    classifier_cfg = load_classifier_config_from_yaml()
    
    # Extraer umbrales de confianza
    if "confidence_threshold" not in yolo_cfg:
        raise ValueError("config_model.yaml debe contener 'yolo.confidence_threshold'")
    if "bad_threshold" not in classifier_cfg:
        raise ValueError("config_model.yaml debe contener 'classifier.bad_threshold'")
    
    try:
        yolo_conf_threshold = float(yolo_cfg["confidence_threshold"])
        classifier_bad_threshold = float(classifier_cfg["bad_threshold"])  # Usar bad_threshold en lugar de confidence_threshold
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error leyendo umbrales desde config_model.yaml: {e}")
    
    # Obtener estadísticas del clasificador desde context
    classifier_stats = None
    if context and hasattr(context, 'classifier_stats'):
        classifier_stats = context.classifier_stats
    
    # Actualizar panel con información en tiempo real
    panel = actualizar_panel_control(
        panel_base, 
        metricas={}, 
        estado_camara=acquisition_running, 
        img_width=w_img, 
        img_height=h_img,
        gamma_actual=gamma_actual,
        patron_actual=patron_actual,
        yolo_stats=yolo_stats,
        cam=cam,
        context=context,  # Pasar context para detectar dispositivo
        yolo_conf_threshold=yolo_conf_threshold,  # Confianza YOLO desde config
        classifier_conf_threshold=classifier_bad_threshold,  # bad_threshold del clasificador desde config
        classifier_stats=classifier_stats  # Estadísticas del clasificador (buenas/malas)
    )
    
    MIN_UI_H = 720
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
    
    # Combinar horizontalmente: imagen + panel
    result = np.hstack([img_padded, panel_padded])
    return result

import cv2
import time
from core.logging import log_warning

def draw_awb_indicator(img_bgr, app=None):
    """Dibuja el indicador de AWB si está activo."""
    try:
        if app and getattr(app, 'awb_indicator_active', False) and time.time() < getattr(app, 'awb_indicator_time', 0):
            # Overlay verde para AWB
            overlay = img_bgr.copy()
            cv2.rectangle(overlay, (10, 70), (180, 110), (0, 255, 0), -1)
            img_bgr = cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0)
            cv2.putText(img_bgr, "AWB CALIBRANDO...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    except Exception as e:
        log_warning(f"⚠️ Error dibujando indicador AWB: {e}")
    
    return img_bgr

def draw_auto_cal_indicator(img_bgr, app=None):
    """Dibuja el indicador de Auto Cal si está activo."""
    try:
        if app and getattr(app, 'auto_cal_indicator_active', False) and time.time() < getattr(app, 'auto_cal_indicator_time', 0):
            # Overlay azul para Auto Cal
            overlay = img_bgr.copy()
            cv2.rectangle(overlay, (10, 70), (200, 110), (255, 0, 0), -1)
            img_bgr = cv2.addWeighted(overlay, 0.3, img_bgr, 0.7, 0)
            cv2.putText(img_bgr, "AUTO CAL...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    except Exception as e:
        log_warning(f"⚠️ Error dibujando indicador Auto Cal: {e}")
    
    return img_bgr

def draw_all_indicators(img_bgr, app=None):
    """Aplica todos los indicadores visuales a la imagen."""
    img_bgr = draw_awb_indicator(img_bgr, app)
    img_bgr = draw_auto_cal_indicator(img_bgr, app)
    return img_bgr

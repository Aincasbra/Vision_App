import cv2
import numpy as np


def apply_gamma_from_state(img_bgr, gamma_value=1.0):
    """Aplica gamma usando el valor proporcionado."""
    try:
        if abs(gamma_value - 1.0) <= 1e-3:
            return img_bgr
        
        # Crear LUT para gamma
        lut = np.array([((i / 255.0) ** (1.0 / gamma_value)) * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(img_bgr, lut)
    except Exception:
        return img_bgr

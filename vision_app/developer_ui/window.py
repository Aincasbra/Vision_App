"""
Ventana de la UI de desarrollador
---------------------------------
- Crea la ventana principal y dibuja frames con el panel lateral.
- Proporciona utilidades para mostrar negro con panel y destruir ventana.
- Usado por `app.py` para la visualización cuando HEADLESS=0.
"""
"""
Módulo para gestión de ventana y presentaciones de la UI de depuración.
"""
import cv2
import numpy as np
from developer_ui.compositor import compose_with_panel
from developer_ui.state import set_ui_dimensions
from core.logging import log_info, log_warning


def get_win_name():
    """
    Obtiene el nombre de la ventana principal.
    """
    return "Vision App"


def create_main_window(width, height):
    """
    Crea la ventana principal de la aplicación.
    
    Args:
        width: Ancho de la ventana
        height: Alto de la ventana
    """
    try:
        win_name = get_win_name()
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow(win_name, width, height)
        log_info(f"🖥️ Ventana principal creada: {width}x{height}")
    except Exception as e:
        log_warning(f"⚠️ Error creando ventana: {e}")


def show_frame_with_panel(image_bgr, camera=None, acquisition_running=False, gamma_actual=0.8, patron_actual="BG", yolo_stats=None):
    """
    Muestra un frame con el panel de control compuesto.
    
    Args:
        image_bgr: Imagen BGR a mostrar
        camera: Objeto cámara para actualizaciones en tiempo real
        acquisition_running: Estado de adquisición de la cámara
        gamma_actual: Valor actual de gamma
        patron_actual: Patrón Bayer actual
        yolo_stats: Estadísticas de YOLO
    """
    try:
        win_name = get_win_name()
        # Actualizar dimensiones de UI
        set_ui_dimensions(image_bgr.shape[1], image_bgr.shape[0])
        
        # Componer con panel
        out = compose_with_panel(image_bgr, cam=camera, acquisition_running=acquisition_running, 
                                gamma_actual=gamma_actual, patron_actual=patron_actual, yolo_stats=yolo_stats)
        
        # Mostrar frame
        cv2.imshow(win_name, out)
        cv2.waitKey(1)
        
    except Exception as e:
        log_warning(f"⚠️ Error mostrando frame: {e}")


def show_black_with_panel(width, height):
    """
    Muestra una pantalla negra con el panel de control.
    
    Args:
        width: Ancho de la imagen negra
        height: Alto de la imagen negra
    """
    try:
        win_name = get_win_name()
        # Crear imagen negra del tamaño original de la cámara
        black_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Actualizar dimensiones de UI
        set_ui_dimensions(width, height)
        
        # Componer con panel
        out = compose_with_panel(black_img, cam=None)
        
        # Mostrar pantalla negra
        cv2.imshow(win_name, out)
        cv2.waitKey(1)
        
        log_info(f"🖤 Pantalla negra mostrada: {width}x{height}")
        
    except Exception as e:
        log_warning(f"⚠️ Error mostrando pantalla negra: {e}")


def destroy_window():
    """
    Destruye la ventana principal.
    """
    try:
        cv2.destroyAllWindows()
        log_info("🗑️ Ventana destruida")
    except Exception as e:
        log_warning(f"⚠️ Error destruyendo ventana: {e}")

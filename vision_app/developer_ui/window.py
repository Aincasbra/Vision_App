"""
Ventana de la UI de desarrollador
---------------------------------
- Crea la ventana principal y dibuja frames con el panel lateral.
- Proporciona utilidades para mostrar negro con panel y destruir ventana.
- Calcula el tamaño de la ventana basado en el ROI configurado de la cámara.
- Usado por `app.py` para la visualización cuando HEADLESS=0.
"""
import cv2
import numpy as np
from typing import Tuple
from developer_ui.compositor import compose_with_panel
from developer_ui.state import set_ui_dimensions
from core.logging import log_info, log_warning
from camera.device_manager import CameraBackend


def get_win_name():
    """
    Obtiene el nombre de la ventana principal.
    """
    return "Vision App"


def get_window_size_from_camera(camera) -> Tuple[int, int]:
    """
    Calcula el tamaño de la ventana basado en el ROI configurado de la cámara.
    
    Esta función lee la configuración del ROI desde la cámara y calcula el tamaño
    de ventana apropiado. El ROI se configura una vez al inicio y NO puede cambiar
    durante la ejecución por seguridad. Para cambiar el ROI, es necesario parar
    y reiniciar la aplicación.
    
    Si la cámara tiene ROI configurado, usa ese tamaño. Si no, usa el tamaño máximo.
    
    Args:
        camera: Instancia de CameraBackend (puede ser None)
    
    Returns:
        Tupla (width, height) en píxeles para el área de imagen (sin panel)
    """
    try:
        if camera is None:
            return 1624, 1240  # Valores por defecto
        
        h_max = int(CameraBackend.safe_get(camera, 'HeightMax', 1240))
        w_max = int(CameraBackend.safe_get(camera, 'WidthMax', 1624))
        
        # Obtener ROI configurado (si el backend lo soporta)
        if hasattr(camera, '_get_roi_config'):
            try:
                offset_x, offset_y, roi_w, roi_h = camera._get_roi_config(h_max, w_max)
                # Usar tamaño del ROI para la ventana
                return roi_w, roi_h
            except Exception:
                return w_max, h_max
        else:
            # Fallback: usar tamaño máximo
            return w_max, h_max
    except Exception:
        return 1624, 1240  # Valores por defecto


def create_main_window(camera=None, width=None, height=None):
    """
    Crea la ventana principal de la aplicación.
    
    Si no se especifican width/height, calcula el tamaño basado en el ROI de la cámara.
    El ROI se configura una vez al inicio y NO puede cambiar durante la ejecución.
    Para cambiar el ROI, es necesario parar y reiniciar la aplicación.
    
    Args:
        camera: Instancia de CameraBackend (opcional, para calcular tamaño automáticamente)
        width: Ancho de la ventana (opcional, se calcula desde camera si no se especifica)
        height: Alto de la ventana (opcional, se calcula desde camera si no se especifica)
    """
    try:
        # Calcular tamaño si no se especifica
        if width is None or height is None:
            img_w, img_h = get_window_size_from_camera(camera)
            width = width if width is not None else img_w
            height = height if height is not None else img_h
        
        win_name = get_win_name()
        # Usar WINDOW_NORMAL para permitir redimensionamiento manual y control del tamaño
        # +350 para el panel lateral (ancho fijo del panel)
        panel_width = 350
        total_width = width + panel_width
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, total_width, height)
        log_info(f"🖥️ Ventana principal creada: {total_width}x{height} (imagen: {width}x{height}, panel: {panel_width}px)")
    except Exception as e:
        log_warning(f"⚠️ Error creando ventana: {e}")


def resize_window_if_needed(camera, current_width, current_height):
    """
    DEPRECADO: El ROI de la cámara NO puede cambiar durante la ejecución por seguridad.
    
    Esta función se mantiene por compatibilidad, pero no debería usarse.
    El ROI se configura una vez al inicio y no cambia durante la ejecución.
    Para cambiar el ROI, es necesario parar y reiniciar la aplicación.
    
    Args:
        camera: Instancia de CameraBackend
        current_width: Ancho actual del frame
        current_height: Altura actual del frame
    
    Returns:
        Tupla (current_width, current_height) sin cambios
    """
    # No redimensionar: el ROI es fijo durante la ejecución por seguridad
    return current_width, current_height


def show_frame_with_panel(image_bgr, camera=None, acquisition_running=False, gamma_actual=0.8, patron_actual="BG", yolo_stats=None, context=None):
    """
    Muestra un frame con el panel de control compuesto.
    
    Args:
        image_bgr: Imagen BGR a mostrar
        camera: Objeto cámara para actualizaciones en tiempo real
        acquisition_running: Estado de adquisición de la cámara
        gamma_actual: Valor actual de gamma
        patron_actual: Patrón Bayer actual
        yolo_stats: Estadísticas de YOLO
        context: AppContext para detectar dispositivo (GPU/CPU)
    """
    try:
        win_name = get_win_name()
        # Actualizar dimensiones de UI
        set_ui_dimensions(image_bgr.shape[1], image_bgr.shape[0])
        
        # Componer con panel
        out = compose_with_panel(image_bgr, cam=camera, acquisition_running=acquisition_running, 
                                gamma_actual=gamma_actual, patron_actual=patron_actual, yolo_stats=yolo_stats, context=context)
        
        # Mostrar frame
        cv2.imshow(win_name, out)
        cv2.waitKey(1)
        
    except Exception as e:
        log_warning(f"⚠️ Error mostrando frame: {e}")


# Variable global para rastrear si ya se mostró el log de pantalla negra
_black_screen_logged = False

def show_black_with_panel(width, height, log_once: bool = True):
    """
    Muestra una pantalla negra con el panel de control.
    
    Args:
        width: Ancho de la imagen negra
        height: Alto de la imagen negra
        log_once: Si es True, solo muestra el log la primera vez (al inicio)
    """
    global _black_screen_logged
    
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
        
        # Solo mostrar log la primera vez (al inicio) si log_once=True
        if log_once and not _black_screen_logged:
            log_info(f"🖤 Pantalla negra mostrada: {width}x{height}")
            _black_screen_logged = True
        elif not log_once:
            # Si log_once=False, siempre mostrar (comportamiento anterior)
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

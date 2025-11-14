"""
Validación de detecciones
--------------------------
Módulo de validación que verifica si un objeto detectado por YOLO está listo
para procesamiento completo (clasificación, logging, salidas digitales).

Funcionalidades:
- Valida que el objeto esté centrado en la zona de trabajo configurada
  (el objeto debe estar en la posición correcta para inspección)
- Valida que el objeto tenga el tamaño correcto 
  (asegura que estamos procesando objetos del tamaño esperado, no objetos pequeños/grandes)
- Valida que el objeto esté completo (no cortado por los bordes de la imagen)
  (un objeto cortado no puede ser clasificado correctamente porque falta información)

En producción, solo queremos procesar objetos cuando están en la posición correcta
y completamente visibles. Esto evita:
- Procesar objetos parcialmente visibles (menor calidad de clasificación)
- Procesar objetos fuera de la zona de trabajo (no están en posición de inspección)
- Desperdiciar recursos procesando frames que no aportan valor
- Nota: El cooldown para evitar procesar múltiples veces el mismo objeto se maneja
  en detection_service.py, no aquí.

Configuración:
- La configuración (work_zone, bottle_sizes) se carga desde config_camera.yaml
- Se pasa a este módulo desde detection_service.py (que la recibe de app.py)
- Cada cámara puede tener su propia configuración si se especifica en el YAML

Llamado desde:
- `model/detection/detection_service.py`: valida cada detección antes de procesar
"""
from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np
from core.logging import log_warning, log_error


def is_bottle_ready(
    xyxy: np.ndarray,
    img_shape: Tuple[int, int],
    work_zone: Dict[str, Any],
    bottle_sizes: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """
    Valida si un objeto detectado está listo para procesamiento completo.
    
    Realiza tres validaciones en orden:
    1. Tamaño: verifica que el objeto tenga dimensiones dentro del rango esperado
       (configurable en bottle_sizes)
    2. Completitud: verifica que el objeto no esté cortado por los bordes de la imagen
       (necesario para clasificación de calidad)
    3. Centrado: verifica que el centro del objeto esté cerca de la línea central del ROI.
       - Si axis="vertical": línea horizontal (cinta se mueve horizontalmente, valida Y)
       - Si axis="horizontal": línea vertical (cinta se mueve verticalmente, valida X)
       Cuando el centro del bbox cruza esa línea (con tolerancia), el objeto está centrado.
    
    Args:
        xyxy: Array numpy de shape [N, 4] con bounding boxes [x1, y1, x2, y2].
              Solo se valida la primera detección (xyxy[0]).
        img_shape: Tupla (height, width) de la imagen completa (ROI configurado).
        work_zone: Dict con configuración de zona de trabajo (OBLIGATORIO, se lee desde config_camera.yaml):
            - axis: Eje de validación ("vertical" o "horizontal", OBLIGATORIO)
            - tolerance: Tolerancia en píxeles desde la línea central (OBLIGATORIO)
            - radius: (legacy, se usa como tolerance si tolerance no está definido)
            - center_position: Posición de la línea central (opcional, se calcula automáticamente si es None)
            Si falta configuración, se lanza ValueError con logging de error.
        bottle_sizes: Dict con tamaños esperados del objeto (OBLIGATORIO, se leen desde config_camera.yaml):
            - min_width: Ancho mínimo en píxeles 
            - max_width: Ancho máximo en píxeles 
            - min_height: Altura mínima en píxeles
            - max_height: Altura máxima en píxeles
            - margin: Margen de borde para validar completitud 
            Si falta alguna clave o bottle_sizes es None, se lanza ValueError con logging de error.
    
    Returns:
        Tuple[bool, str]:
            - bool: True si el objeto está listo para procesamiento, False en caso contrario
            - str: Razón del rechazo si es False, o "ok" si es válido
    
    """
    if len(xyxy) == 0:
        return False, "sin_detecciones"
    
    # Obtener primera detección (máximo un objeto por imagen en producción)
    bbox = xyxy[0]
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    
    # Calcular dimensiones y centros del objeto
    w = x2 - x1           # Ancho del objeto
    h = y2 - y1           # Altura del objeto
    cx = (x1 + x2) * 0.5  # Centro X del objeto 
    cy = (y1 + y2) * 0.5  # Centro Y del objeto
    
    # Validar y extraer valores de configuración desde config_camera.yaml (sección bottle_sizes)
    # Si falta configuración, hacer logging y lanzar excepción 
    if bottle_sizes is None:
        log_error("❌ bottle_sizes es None - falta configuración en config_camera.yaml", logger_name="vision")
        raise ValueError("bottle_sizes no puede ser None. Debe estar configurado en config_camera.yaml")
    
    if not isinstance(bottle_sizes, dict):
        log_error(f"❌ bottle_sizes debe ser un dict, recibido: {type(bottle_sizes)}", logger_name="vision")
        raise ValueError(f"bottle_sizes debe ser un dict, recibido: {type(bottle_sizes)}")
    
    # Validar que todas las claves requeridas estén presentes (OPTIMIZADO: usar set para O(1) lookup)
    required_keys = {"min_width", "max_width", "min_height", "max_height", "margin"}
    missing_keys = required_keys - set(bottle_sizes.keys())
    
    if missing_keys:
        missing_keys_list = sorted(missing_keys)  # Ordenar para mensaje consistente
        log_error(f"❌ Faltan claves requeridas en bottle_sizes: {missing_keys_list}. Revisa config_camera.yaml", logger_name="vision")
        raise ValueError(f"Faltan claves requeridas en bottle_sizes: {missing_keys_list}. Configura en config_camera.yaml")
    
    # Leer valores desde config_camera.yaml (bottle_sizes)
    try:
        min_w = float(bottle_sizes["min_width"])
        max_w = float(bottle_sizes["max_width"])
        min_h = float(bottle_sizes["min_height"])
        max_h = float(bottle_sizes["max_height"])
        margin = float(bottle_sizes["margin"])
    except (ValueError, TypeError) as e:
        log_error(f"❌ Error al convertir valores de bottle_sizes a float: {e}. Valores: {bottle_sizes}", logger_name="vision")
        raise ValueError(f"Error al convertir valores de bottle_sizes a float: {e}") from e
    
    # Validar que los valores sean razonables
    if min_w >= max_w:
        log_error(f"❌ min_width ({min_w}) debe ser menor que max_width ({max_w})", logger_name="vision")
        raise ValueError(f"min_width ({min_w}) debe ser menor que max_width ({max_w})")
    
    if min_h >= max_h:
        log_error(f"❌ min_height ({min_h}) debe ser menor que max_height ({max_h})", logger_name="vision")
        raise ValueError(f"min_height ({min_h}) debe ser menor que max_height ({max_h})")
    
    if margin < 0:
        log_error(f"❌ margin ({margin}) no puede ser negativo", logger_name="vision")
        raise ValueError(f"margin ({margin}) no puede ser negativo")
    
    # VALIDACIÓN 1: Tamaño del objeto 
    if w < min_w or w > max_w:
        return False, f"ancho_invalido_{w:.1f}px"
    if h < min_h or h > max_h:
        return False, f"altura_invalida_{h:.1f}px"
    
    # VALIDACIÓN 2: Completitud del objeto 
    # OPTIMIZACIÓN: Pre-calcular límites de imagen una sola vez
    # IMPORTANTE: El margin se aplica según la dirección de movimiento de la cinta:
    # - Si axis="vertical" (cinta horizontal): solo validar bordes verticales (arriba/abajo)
    #   El bote puede tocar los bordes laterales mientras pasa
    # - Si axis="horizontal" (cinta vertical): solo validar bordes laterales (izquierda/derecha)
    #   El bote puede tocar los bordes verticales mientras pasa
    img_h, img_w = float(img_shape[0]), float(img_shape[1])
    
    # Obtener axis de work_zone para aplicar margin inteligentemente
    axis = work_zone.get("axis", "vertical").lower() if work_zone else "vertical"
    
    if axis == "vertical":
        # Cinta horizontal: solo validar bordes verticales (arriba/abajo)
        # Permitir que toque bordes laterales (izquierda/derecha)
        max_y = img_h - margin
        if y1 < margin or y2 > max_y:
            return False, f"objeto_cortado_margen_vertical_{margin}px; (x1, y1, x2, y2) = {(x1, y1, x2, y2)}"
    else:
        # Cinta vertical: solo validar bordes laterales (izquierda/derecha)
        # Permitir que toque bordes verticales (arriba/abajo)
        max_x = img_w - margin
        if x1 < margin or x2 > max_x:
            return False, f"objeto_cortado_margen_horizontal_{margin}px; (x1, y1, x2, y2) = {(x1, y1, x2, y2)}"
    
    # VALIDACIÓN 3: Centrado en zona de trabajo
    # Validar que work_zone esté configurado correctamente
    if work_zone is None:
        log_error("❌ work_zone es None - falta configuración en config_camera.yaml", logger_name="vision")
        raise ValueError("work_zone no puede ser None. Debe estar configurado en config_camera.yaml")
    
    if not isinstance(work_zone, dict):
        log_error(f"❌ work_zone debe ser un dict, recibido: {type(work_zone)}", logger_name="vision")
        raise ValueError(f"work_zone debe ser un dict, recibido: {type(work_zone)}")
    
    # Extraer valores de work_zone (tolerance puede venir de "tolerance" o "radius" para compatibilidad)
    axis = work_zone.get("axis", "vertical")
    if axis is None:
        log_error("❌ work_zone.axis no está configurado en config_camera.yaml", logger_name="vision")
        raise ValueError("work_zone.axis debe estar configurado en config_camera.yaml")
    axis = str(axis).lower()
    
    if axis not in ("vertical", "horizontal"):
        log_error(f"❌ work_zone.axis debe ser 'vertical' o 'horizontal', recibido: {axis}", logger_name="vision")
        raise ValueError(f"work_zone.axis debe ser 'vertical' o 'horizontal', recibido: {axis}")
    
    # tolerance puede venir de "tolerance" o "radius" (legacy)
    tolerance = work_zone.get("tolerance") or work_zone.get("radius")
    if tolerance is None:
        log_error("❌ work_zone.tolerance (o work_zone.radius) no está configurado en config_camera.yaml", logger_name="vision")
        raise ValueError("work_zone.tolerance (o work_zone.radius) debe estar configurado en config_camera.yaml")
    
    try:
        tolerance = float(tolerance)
        if tolerance < 0:
            log_error(f"❌ work_zone.tolerance ({tolerance}) no puede ser negativo", logger_name="vision")
            raise ValueError(f"work_zone.tolerance ({tolerance}) no puede ser negativo")
    except (ValueError, TypeError) as e:
        log_error(f"❌ Error al convertir work_zone.tolerance a float: {e}. Valor: {work_zone.get('tolerance') or work_zone.get('radius')}", logger_name="vision")
        raise ValueError(f"Error al convertir work_zone.tolerance a float: {e}") from e
    
    center_position = work_zone.get("center_position")  # Puede ser None (se calcula automáticamente)
    
    if axis == "vertical":
        # Línea horizontal: validar centro vertical del objeto (Y)
        roi_center_y = float(center_position) if center_position is not None else img_h * 0.5
        dist_from_center = abs(cy - roi_center_y)
        
        if dist_from_center > tolerance:
            return False, f"no_centrado_Y_dist_{dist_from_center:.1f}px_tolerancia_{tolerance}px"
    else:
        # Línea vertical: validar centro horizontal del objeto (X)
        roi_center_x = float(center_position) if center_position is not None else img_w * 0.5
        dist_from_center = abs(cx - roi_center_x)
        
        if dist_from_center > tolerance:
            return False, f"no_centrado_X_dist_{dist_from_center:.1f}px_tolerancia_{tolerance}px"
    
    # Todas las validaciones pasaron
    return True, "ok"


def get_work_zone_from_config(
    config: Dict[str, Any], 
    img_shape: Tuple[int, int],
    roi: Optional[Tuple[int, int, int, int]] = None
) -> Dict[str, Any]:
    """
    Procesa la configuración de zona de trabajo desde config_camera.yaml.
    
    Convierte la configuración del archivo YAML a un formato normalizado que puede usar
    is_bottle_ready(). La validación simplificada usa una línea central (horizontal o vertical)
    y verifica si el centro del objeto está cerca de esa línea.
    
    La línea central se calcula basándose en el ROI configurado:
    - Si axis="vertical": línea horizontal en offset_y + height/2
    - Si axis="horizontal": línea vertical en offset_x + width/2
    
    La zona de trabajo se configura en config_camera.yaml bajo la sección "work_zone".
    
    Args:
        config: Dict con configuración de zona de trabajo desde YAML:
            - axis: Eje de validación ("vertical" o "horizontal", default: "vertical")
                   "vertical" = línea horizontal (cinta se mueve horizontalmente, valida Y)
                   "horizontal" = línea vertical (cinta se mueve verticalmente, valida X)
            - tolerance: Tolerancia en píxeles desde la línea central (default: 30px)
            - radius: (legacy, se usa como tolerance si tolerance no está definido)
        img_shape: Tupla (height, width) de la imagen (ROI configurado)
        roi: Tupla opcional (offset_x, offset_y, width, height) del ROI configurado.
             Si se proporciona, se usa para calcular la línea central.
             Si es None, se calcula desde img_shape asumiendo offset=0.
    
    Returns:
        Dict normalizado con:
            - axis: Eje de validación (str)
            - center_position: Posición de la línea central en píxeles (float)
            - tolerance: Tolerancia en píxeles desde la línea central (float)
    
    Ejemplo de configuración en config_camera.yaml:
        work_zone:
          axis: "vertical"      # Línea horizontal (cinta se mueve horizontalmente)
          tolerance: 30         # Tolerancia de 30px desde el centro de la altura del ROI
    """
    img_h, img_w = img_shape[:2]
    
    # Obtener eje de validación (default: "vertical" = línea horizontal)
    axis = config.get("axis", "vertical").lower()
    if axis not in ("vertical", "horizontal"):
        axis = "vertical"  # Fallback a vertical si el valor no es válido
    
    # Calcular posición de la línea central basándose en el ROI configurado
    if roi is not None:
        # ROI proporcionado: usar offset + dimension/2
        offset_x, offset_y, roi_width, roi_height = roi
        if axis == "vertical":
            # Línea horizontal: offset_y + height/2
            center_position = float(offset_y) + float(roi_height) / 2.0
        else:
            # Línea vertical: offset_x + width/2
            center_position = float(offset_x) + float(roi_width) / 2.0
    else:
        # Sin ROI: calcular desde img_shape asumiendo offset=0
        if axis == "vertical":
            # Línea horizontal: mitad de la altura
            center_position = img_h / 2.0
        else:
            # Línea vertical: mitad del ancho
            center_position = img_w / 2.0
    
    # Obtener tolerancia (nuevo parámetro) o usar radius como fallback (legacy)
    tolerance = config.get("tolerance")
    if tolerance is None:
        # Si no hay tolerance, usar radius como fallback (compatibilidad con configs antiguas)
        tolerance = config.get("radius", 30)
    
    return {
        "axis": axis,
        "center_position": float(center_position),
        "tolerance": float(tolerance),
    }


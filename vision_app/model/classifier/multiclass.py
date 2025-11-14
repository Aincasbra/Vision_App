"""
Clasificador multiclase (wrapper)
---------------------------------
- Responsabilidad: Cargar modelo de clasificación y realizar predicciones sobre recortes BGR.
- Lee configuración directamente desde config_model.yaml.

FLUJO DE CONFIGURACIÓN:
1. config_model.yaml → contiene configuración del clasificador (sección "classifier")
2. load_classifier() → lee directamente desde config_model.yaml (sin pasar por settings)
   - Lee model_path y classes desde la sección "classifier"
   - Actualiza CLASS_NAMES global desde config (NO usa valores hardcodeados)

Funciones principales:
  * `clf_load()`: carga el modelo desde archivo (ruta desde config o búsqueda automática)
  * `clf_predict_bgr()`: realiza predicción sobre un recorte BGR
  * `load_classifier()`: función de conveniencia con logs para cargar el clasificador
    - Si classifier_config es None, lee directamente desde config_model.yaml
    - Si se proporciona classifier_config, lo usa directamente

Llamado desde:
  * `app.py`: llama a `load_classifier()` sin parámetros (lee directamente desde config_model.yaml)
  * `model/detection/detection_service.py`: llama a `clf_predict_bgr()` para clasificar detecciones
"""
from typing import Tuple, List
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


# NOTA: CLF_MODEL_PATH y CLASS_NAMES ya NO se usan como defaults.
# Ahora TODO debe venir de config_model.yaml (sección 'classifier').
# Si no se proporciona config, se lanza ValueError.
# Estos valores solo se mantienen para compatibilidad temporal (búsqueda de archivos).
# IMPORTANTE: CLASS_NAMES se actualiza automáticamente desde config_model.yaml cuando se carga el clasificador.
CLF_MODEL_PATH = "clasificador_multiclase_torch.pt"  # DEPRECATED: usar classifier.model_path en config_model.yaml
CLASS_NAMES = ['buenas', 'malas', 'defectuosas']  # DEPRECATED: se actualiza desde config_model.yaml (sección 'classifier.classes')
_MODEL = None
_DEVICE = None
# Transform pre-computado para reutilización (optimización: evitar crear en cada predicción)
_TRANSFORM = None


def _device():
    """Obtiene el dispositivo CUDA/CPU (singleton)."""
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEVICE


def _get_transform():
    """Obtiene el transform de preprocesamiento (singleton, optimización).
    
    El transform se crea una sola vez y se reutiliza en todas las predicciones.
    """
    global _TRANSFORM
    if _TRANSFORM is None:
        _TRANSFORM = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return _TRANSFORM


def _create_custom_mobilenet_v2(num_classes: int):
    """Crea una instancia de CustomMobileNetV2 (helper para evitar duplicación).
    
    Args:
        num_classes: Número de clases de salida
        
    Returns:
        Instancia de CustomMobileNetV2
    """
    from torchvision.models import mobilenet_v2
    import torch.nn as nn
    
    class CustomMobileNetV2(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            mobilenet = mobilenet_v2(pretrained=False)
            self.features = mobilenet.features
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes)
            )
        def forward(self, x):
            x = self.features(x)
            x = x.mean([2, 3])  # Global average pooling
            x = self.classifier(x)
            return x
    
    return CustomMobileNetV2(num_classes=num_classes)


def clf_load(model_path: str = "") -> bool:
    """Carga el modelo de clasificación desde un archivo (OPTIMIZADO).
    
    Args:
        model_path: Ruta al archivo del modelo (OBLIGATORIO, debe venir de config_model.yaml).
                    Puede ser ruta absoluta o relativa. Si es relativa, se busca en múltiples ubicaciones.
        
    Returns:
        bool: True si se cargó correctamente, False en caso contrario.
        
    Note:
        En producción, model_path debe venir siempre de config_model.yaml (classifier.model_path).
        Si es una ruta relativa, se busca en:
        1. Ruta especificada (relativa al directorio actual)
        2. Relativo a model/classifier/
        3. Relativo a vision_app/
    """
    global _MODEL, _DEVICE
    
    # Si model_path está vacío, usar ruta por defecto (solo para compatibilidad)
    if not model_path:
        model_path = CLF_MODEL_PATH
    
    # Si la ruta no es absoluta, buscar en múltiples ubicaciones
    if not os.path.isabs(model_path):
        # Buscar en diferentes ubicaciones posibles
        search_paths = [
            model_path,  # Primero la ruta especificada (relativa al directorio actual)
            os.path.join(os.path.dirname(__file__), model_path),  # Relativo a model/classifier/
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), model_path),  # Relativo a vision_app/
        ]
        
        # Buscar la primera ruta que exista
        found_path = None
        for path in search_paths:
            if os.path.exists(path):
                found_path = path
                break
        
        if found_path:
            model_path = found_path
        # Si no se encuentra, usar la ruta original (puede fallar, pero al menos se intenta)
    
    try:
        dev = _device()
        checkpoint = torch.load(model_path, map_location=dev)
        
        # Optimizado: usar función helper para evitar duplicación de código
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            _MODEL = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Intentar reconstruir una arquitectura compatible (MobileNetV2 + clasificador)
            try:
                _MODEL = _create_custom_mobilenet_v2(num_classes=len(CLASS_NAMES))
                _MODEL.load_state_dict(checkpoint['state_dict'])
            except Exception as e:
                from core.logging import log_error
                log_error(f"Error cargando state_dict del clasificador: {e}")
                _MODEL = None
        elif isinstance(checkpoint, dict) and 'features.0.0.weight' in checkpoint:
            # Formato alternativo: state_dict directo
            try:
                _MODEL = _create_custom_mobilenet_v2(num_classes=len(CLASS_NAMES))
                _MODEL.load_state_dict(checkpoint)
            except Exception as e:
                from core.logging import log_error
                log_error(f"Error cargando state_dict directo del clasificador: {e}")
                _MODEL = None
        else:
            _MODEL = checkpoint
        
        if _MODEL is not None:
            _MODEL.eval()
            _MODEL.to(dev)
            
            # OPTIMIZACIÓN: Usar half precision (FP16) para acelerar en GPU
            if dev.type == 'cuda' and torch.cuda.is_available():
                try:
                    _MODEL = _MODEL.half()  # Convertir a FP16 para mayor velocidad
                    from core.logging import log_info
                    log_info("✅ Clasificador optimizado con FP16 (half precision)")
                except Exception as e:
                    from core.logging import log_warning
                    log_warning(f"⚠️ No se pudo convertir a FP16: {e}, usando FP32")
            
            # OPTIMIZACIÓN: Precalentar el modelo (warmup) para evitar el primer forward lento
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224).to(dev)
                    if dev.type == 'cuda' and hasattr(_MODEL, 'half') and next(_MODEL.parameters()).dtype == torch.float16:
                        dummy_input = dummy_input.half()
                    _MODEL(dummy_input)  # Warmup
                    if dev.type == 'cuda':
                        torch.cuda.synchronize()  # Sincronizar para asegurar que el warmup termine
                from core.logging import log_info
                log_info("✅ Clasificador precalentado (warmup completado)")
            except Exception as e:
                from core.logging import log_warning
                log_warning(f"⚠️ No se pudo precalentar el modelo: {e}")
                
        return _MODEL is not None
    except FileNotFoundError:
        from core.logging import log_error
        log_error(f"Archivo del modelo no encontrado: {model_path}")
        _MODEL = None
        return False
    except Exception as e:
        from core.logging import log_error
        log_error(f"Error cargando modelo del clasificador desde {model_path}: {e}")
        import traceback
        log_error(f"Traceback: {traceback.format_exc()}")
        _MODEL = None
        return False


def clf_predict_bgr(bgr_roi: np.ndarray) -> Tuple[str, float, List[float]]:
    """Realiza predicción sobre un recorte BGR (OPTIMIZADO).
    
    Optimizaciones aplicadas:
    - Reutilización del transform (singleton)
    - Conversión BGR2RGB optimizada
    - Cálculo de probabilidades base optimizado
    - Uso de operaciones NumPy vectorizadas
    
    Args:
        bgr_roi: Array numpy BGR del recorte a clasificar
        
    Returns:
        Tuple[str, float, List[float]]: (etiqueta, confianza, probabilidades de todas las clases)
    """
    global CLASS_NAMES
    
    if _MODEL is None:
        # Optimizado: calcular probabilidad base una sola vez
        n = len(CLASS_NAMES)
        base_prob = 1.0 / n if n > 0 else 0.0
        base = [base_prob] * n
        return "unknown", 0.0, base
    
    try:
        # Optimizado: reutilizar transform (singleton, evita crear en cada predicción)
        transform = _get_transform()
        
        # Conversión BGR2RGB (OpenCV es eficiente, pero validar entrada)
        if bgr_roi.size == 0:
            n = len(CLASS_NAMES)
            base_prob = 1.0 / n if n > 0 else 0.0
            return "unknown", 0.0, [base_prob] * n
        
        rgb_img = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
        
        # Aplicar transform y mover a dispositivo
        dev = _device()
        tensor_img = transform(rgb_img).unsqueeze(0).to(dev)
        
        # OPTIMIZACIÓN: Convertir a half precision si el modelo está en FP16
        if dev.type == 'cuda' and hasattr(_MODEL, 'half') and next(_MODEL.parameters()).dtype == torch.float16:
            tensor_img = tensor_img.half()
        
        # Inferencia con torch.no_grad() (ya optimizado)
        with torch.no_grad():
            outputs = _MODEL(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            
            # OPTIMIZACIÓN: Usar torch.argmax en GPU (más rápido que mover a CPU primero)
            # Solo mover a CPU lo mínimo necesario
            idx = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0, idx].item())
            
            # Mover a CPU solo para obtener la lista completa (si es necesario)
            # Convertir a float32 antes de numpy para compatibilidad
            prob_np = probabilities.float().cpu().numpy().squeeze()
        
        # Convertir a lista solo al final (necesario para el return type)
        # NOTA: No filtramos por confidence_threshold aquí. El filtrado se hace en detection_service.py
        # usando bad_threshold para determinar si es "bad" o "ok".
        return CLASS_NAMES[idx], confidence, prob_np.tolist()
    except Exception:
        # Optimizado: calcular probabilidad base una sola vez
        n = len(CLASS_NAMES)
        base_prob = 1.0 / n if n > 0 else 0.0
        base = [base_prob] * n
        return "unknown", 0.0, base


def load_classifier(classifier_config: dict = None) -> bool:
    """Carga el clasificador leyendo directamente desde config_model.yaml si no se proporciona config.
    
    IMPORTANTE: Esta función actualiza CLASS_NAMES desde config_model.yaml.
    NO se usan valores hardcodeados, todo viene del config.
    
    Args:
        classifier_config: Configuración del clasificador (opcional). Si es None, lee directamente desde config_model.yaml.
    
    Returns:
        bool: True si se cargó correctamente, False en caso contrario.
        
    Raises:
        ValueError: Si falta 'model_path' o 'classes' en el config.
    """
    from core.logging import log_info, log_warning, log_error
    
    # Si no se proporciona config, leer directamente desde config_model.yaml
    if classifier_config is None:
        classifier_config = load_classifier_config_from_yaml()
    
    # Validar que classifier_config existe y tiene los campos requeridos
    if not classifier_config:
        raise ValueError("config_model.yaml debe contener la sección 'classifier'")
    
    if "model_path" not in classifier_config:
        raise ValueError("config_model.yaml debe contener 'classifier.model_path'")
    
    if "classes" not in classifier_config:
        raise ValueError("config_model.yaml debe contener 'classifier.classes'")
    
    try:
        log_info("🔄 Cargando clasificador...")
        
        # Obtener ruta del modelo desde config - OBLIGATORIO
        model_path = classifier_config["model_path"]
        if not model_path or not isinstance(model_path, str):
            raise ValueError(f"'classifier.model_path' debe ser una cadena no vacía. Valor recibido: {model_path}")
        
        log_info(f"   Usando modelo desde config: {model_path}")
        
        # Actualizar CLASS_NAMES global desde config (para uso en clf_predict_bgr)
        global CLASS_NAMES, _TRANSFORM
        classes_from_config = classifier_config["classes"]
        if not isinstance(classes_from_config, (list, tuple)) or len(classes_from_config) == 0:
            raise ValueError(f"'classifier.classes' debe ser una lista no vacía. Valor recibido: {classes_from_config}")
        CLASS_NAMES = list(classes_from_config)
        # Resetear transform si cambian las clases (por si acaso, aunque normalmente no cambia)
        _TRANSFORM = None
        log_info(f"   Clases del clasificador (desde config): {CLASS_NAMES}")
        
        if clf_load(model_path):
            log_info("✅ Clasificador listo")
            return True
        else:
            # Mostrar información de depuración sobre dónde se buscó
            search_paths = [
                model_path,  # Primero la ruta del config
                os.path.join(os.path.dirname(__file__), model_path),  # Relativo a este módulo
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), model_path),  # Relativo a vision_app/
            ]
            log_warning(f"❌ No se pudo cargar el clasificador, desactivando clasificación")
            log_warning(f"   Buscado en: {search_paths}")
            return False
    except ValueError as e:
        # Re-lanzar ValueError para que se propague
        raise
    except Exception as e:
        log_error(f"❌ Error cargando clasificador: {e}")
        import traceback
        log_error(f"   Traceback: {traceback.format_exc()}")
        return False


def load_classifier_config_from_yaml() -> dict:
    """Lee configuración del clasificador directamente desde config_model.yaml.
    
    Función pública que puede ser importada por otros módulos (ej: detection_service, compositor).
    Usa load_yaml_config() de settings.py para evitar duplicar la lógica de búsqueda.
    
    Returns:
        Dict con configuración del clasificador (sección "classifier" del YAML)
        
    Raises:
        ValueError: Si no se encuentra el archivo o falta la sección "classifier"
    """
    from core.settings import load_yaml_config
    
    # Carga silenciosa (el log ya se mostró al inicio en load_settings())
    config = load_yaml_config("config_model.yaml", silent=True)
    
    if "classifier" not in config:
        raise ValueError("config_model.yaml debe contener la sección 'classifier'")
    
    return dict(config["classifier"])


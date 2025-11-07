"""
Clasificador multiclase (wrapper)
---------------------------------
- Carga del modelo de clasificación y predicción sobre recortes BGR.
- Funciones principales:
  * `clf_load()`: carga el modelo desde archivo (búsqueda automática de rutas)
  * `clf_predict_bgr()`: realiza predicción sobre un recorte BGR
  * `load_classifier()`: función de conveniencia con logs para cargar el clasificador
- Llamado desde:
  * `app.py`: llama a `load_classifier()` para inicializar el clasificador
  * `model/detection/detection_service.py`: llama a `clf_predict_bgr()` para clasificar detecciones
"""
from typing import Tuple, List
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


CLF_MODEL_PATH = "clasificador_multiclase_torch.pt"
CLASS_NAMES = ['buenas', 'malas', 'defectuosas']
_MODEL = None
_DEVICE = None


def _device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _DEVICE


def clf_load(model_path: str = "") -> bool:
    """Carga el modelo de clasificación desde un archivo.
    
    Args:
        model_path: Ruta al archivo del modelo. Si está vacío, usa CLF_MODEL_PATH.
        
    Returns:
        bool: True si se cargó correctamente, False en caso contrario.
    """
    global _MODEL, _DEVICE
    # Si no se proporciona ruta, intentar buscar en diferentes ubicaciones
    if not model_path:
        import os
        # Buscar en diferentes ubicaciones posibles
        search_paths = [
            os.path.join(os.path.dirname(__file__), "clasificador_multiclase_torch.pt"),  # model/classifier/
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "clasificador_multiclase_torch.pt"),  # vision_app/
            CLF_MODEL_PATH,  # Ruta relativa (directorio actual)
        ]
        for path in search_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            # Si no se encuentra en ninguna ubicación, usar la ruta por defecto
            model_path = CLF_MODEL_PATH
    try:
        dev = _device()
        checkpoint = torch.load(model_path, map_location=dev)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            _MODEL = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Intentar reconstruir una arquitectura compatible (MobileNetV2 + clasificador)
            try:
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
                        x = x.mean([2, 3])
                        x = self.classifier(x)
                        return x
                _MODEL = CustomMobileNetV2(num_classes=len(CLASS_NAMES))
                _MODEL.load_state_dict(checkpoint['state_dict'])
            except Exception:
                _MODEL = None
        elif isinstance(checkpoint, dict) and 'features.0.0.weight' in checkpoint:
            try:
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
                        x = x.mean([2, 3])
                        x = self.classifier(x)
                        return x
                _MODEL = CustomMobileNetV2(num_classes=len(CLASS_NAMES))
                _MODEL.load_state_dict(checkpoint)
            except Exception:
                _MODEL = None
        else:
            _MODEL = checkpoint
        if _MODEL is not None:
            _MODEL.eval()
            _MODEL.to(dev)
        return _MODEL is not None
    except Exception:
        _MODEL = None
        return False


def clf_predict_bgr(bgr_roi: np.ndarray) -> Tuple[str, float, List[float]]:
    if _MODEL is None:
        n = len(CLASS_NAMES)
        base = [1.0/n]*n
        return "unknown", 0.0, base
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        rgb_img = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
        tensor_img = transform(rgb_img).unsqueeze(0).to(_device())
        with torch.no_grad():
            outputs = _MODEL(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            prob_np = probabilities.cpu().numpy().squeeze()
        idx = int(np.argmax(prob_np))
        confidence = float(prob_np[idx])
        return CLASS_NAMES[idx], confidence, prob_np.tolist()
    except Exception:
        n = len(CLASS_NAMES)
        base = [1.0/n]*n
        return "unknown", 0.0, base


def load_classifier() -> bool:
    """Función de conveniencia para cargar el clasificador con logs.
    
    Returns:
        bool: True si se cargó correctamente, False en caso contrario.
    """
    from core.logging import log_info, log_warning, log_error
    import os
    try:
        log_info("🔄 Cargando clasificador...")
        if clf_load(""):  # clf_load ahora busca automáticamente
            log_info("✅ Clasificador listo")
            return True
        else:
            # Mostrar información de depuración sobre dónde se buscó
            search_paths = [
                os.path.join(os.path.dirname(__file__), "clasificador_multiclase_torch.pt"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "clasificador_multiclase_torch.pt"),
                CLF_MODEL_PATH,
            ]
            log_warning(f"❌ No se pudo cargar el clasificador, desactivando clasificación")
            log_warning(f"   Buscado en: {search_paths}")
            return False
    except Exception as e:
        log_error(f"❌ Error cargando clasificador: {e}")
        import traceback
        log_error(f"   Traceback: {traceback.format_exc()}")
        return False


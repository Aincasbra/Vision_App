"""
Clasificador multiclase (wrapper)
---------------------------------
- Carga del modelo de clasificación y predicción sobre recortes BGR.
- Usado por `vision/yolo_service.py`.
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
    global _MODEL, _DEVICE
    model_path = model_path if model_path else CLF_MODEL_PATH
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



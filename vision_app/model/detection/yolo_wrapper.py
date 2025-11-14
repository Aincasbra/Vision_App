"""
YOLO Wrapper (encapsulación de Ultralytics YOLO)
-------------------------------------------------
- Wrapper que encapsula la carga y uso del modelo Ultralytics YOLO.
- Funcionalidades principales:
  * Carga modelos YOLO (.pt o .engine) con optimizaciones CUDA
  * Configura modelo en modo evaluación con optimizaciones PyTorch (cudnn.benchmark)
  * Proporciona método `predict()` que ejecuta inferencia y devuelve detecciones
  * Soporta modelos PyTorch y TensorRT (.engine)
  * Gestiona dispositivo CUDA/CPU automáticamente
- Llamado desde:
  * `model/detection/detection_service.py`: crea instancia de `YOLOPyTorchCUDA` y usa `predict()` para ejecutar inferencias YOLO
"""
from typing import List, Dict
import os
import numpy as np
import torch


class YOLOPyTorchCUDA:
    """YOLO usando PyTorch optimizado; soporta .pt y .engine (Ultralytics)."""
    
    def __init__(self, model_path: str, input_shape: int = 640):
        self.model_path = model_path
        self.input_shape = input_shape
        self.session = None
        self.input_name = None
        self.output_names = None
        self.class_names = ['can', 'hand', 'bottle']
        self.setup_pytorch_cuda()
    
    def setup_pytorch_cuda(self):
        global UNIFIED_DEVICE
        try:
            if 'UNIFIED_DEVICE' not in globals() or UNIFIED_DEVICE is None:
                UNIFIED_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = UNIFIED_DEVICE
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning)
            os.environ['YOLO_VERBOSE'] = 'False'
            from ultralytics import YOLO
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            self.model = YOLO(self.model_path)
            if hasattr(self.model, 'model'):
                self.model.model.eval()
                for param in self.model.model.parameters():
                    param.requires_grad = False
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            if not self.model_path.endswith('.engine'):
                self.model.to(self.device)
                try:
                    self.model.model.eval()
                except Exception:
                    pass
            self.is_engine = str(self.model_path).lower().endswith('.engine')
            if self.is_engine:
                self.run_predict = lambda img, args: self.model.predict(
                    source=img,
                    device=0,
                    imgsz=args.get('imgsz', 416),
                    conf=args.get('conf', 0.3),
                    iou=args.get('iou', 0.45),
                    agnostic_nms=True,
                    max_det=args.get('max_det', 10),
                    verbose=False,
                    stream=False,
                    half=True
                )
            else:
                self.run_predict = lambda img, args: self.model.predict(
                    source=img,
                    device=self.device,
                    **args
                )
            def dummy_train(*args, **kwargs):
                return None
            def dummy_val(*args, **kwargs):
                return None
            self.model.train = dummy_train
            self.model.val = dummy_val
            if hasattr(self.model, 'overrides'):
                self.model.overrides = {}
                self.model.overrides['mode'] = 'predict'
                self.model.overrides['task'] = 'detect'
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                self.model.model.eval()
                return True
            except Exception as e2:
                print(f"❌ Error final: {e2}")
                return False

    def predict(self, image: np.ndarray, conf_threshold: float = 0.5, yolo_args: Dict = None) -> List[Dict]:
        try:
            with torch.no_grad():
                results = self.model(image, conf=conf_threshold, verbose=False)
                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id
                            })
                return detections
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return []


YOLO_AVAILABLE = True



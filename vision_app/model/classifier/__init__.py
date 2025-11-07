"""
Classifier module
-----------------
- Expone funciones para cargar el clasificador multiclase y realizar predicciones
- `multiclass.py` contiene la implementación real (modelo, carga, inferencia)
"""
from model.classifier.multiclass import clf_load, clf_predict_bgr, CLF_MODEL_PATH, CLASS_NAMES, load_classifier

__all__ = ['clf_load', 'clf_predict_bgr', 'CLF_MODEL_PATH', 'CLASS_NAMES', 'load_classifier']

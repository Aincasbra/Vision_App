# 🎥 YOLOCAMARA - GenTL + YOLO + Tracking + Clasificación de Latas # # Nueva versión que integra: 
# - GenTL/Harvester para cámara industrial (interfaz original) 
# - YOLO para detección de objetos (basado en track_speed.py) 
# - ByteTrack para tracking persistente # - SimpleTunaDetector para clasificación de latas 
# - Interfaz profesional tipo StViewer (mantiene funcionalidades originales) 
# Importaciones necesarias 
import os # Manejo de variables de entorno 
import time # Para manejo de tiempo y retrasos 
import threading # Para hilos (threads) 
import queue # Para colas de comunicación entre hilos 
import cv2 # OpenCV para procesamiento de imágenes 
import numpy as np # Numpy para operaciones numéricas
import math # Matemáticas básicas 
import csv # Salida opcional de métricas 
import yaml # Para carga de configuración YAML 
import tempfile # Para manejo de archivos temporales 
import pathlib # Para manejo de rutas 
import textwrap # Para ajuste de texto 
from collections import deque # Para colas dobles (deque) 
from typing import List, Dict, Tuple # Para type hints
import torch # Framework principal para YOLO y clasificador
import torchvision.transforms as transforms
import torch.nn.functional as F
import subprocess # Para ejecutar comandos del sistema 
import shutil # Para operaciones de alto nivel en archivos y directorios 
from time import perf_counter
import sys

# --- Utilidad: fusionar detecciones muy solapadas para evitar duplicados ---
def merge_overlapping_detections(xyxy, confs, clss, iou_threshold=0.7):
    """Fusiona detecciones que se solapan fuertemente manteniendo la de mayor confianza."""
    if len(xyxy) == 0:
        return xyxy, confs, clss

    xyxy = np.asarray(xyxy)
    confs = np.asarray(confs)
    clss = np.asarray(clss)

    keep = []
    used = np.zeros(len(xyxy), dtype=bool)
    for i in range(len(xyxy)):
        if used[i]:
            continue
        xi1, yi1, xi2, yi2 = xyxy[i]
        ai = max(0, (xi2 - xi1)) * max(0, (yi2 - yi1))
        group = [i]
        used[i] = True
        for j in range(i + 1, len(xyxy)):
            if used[j]:
                continue
            xj1, yj1, xj2, yj2 = xyxy[j]
            aj = max(0, (xj2 - xj1)) * max(0, (yj2 - yj1))
            xx1 = max(xi1, xj1)
            yy1 = max(yi1, yj1)
            xx2 = min(xi2, xj2)
            yy2 = min(yi2, yj2)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = ai + aj - inter
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold and clss[i] == clss[j]:
                group.append(j)
                used[j] = True
        best_idx = max(group, key=lambda k: confs[k])
        keep.append(best_idx)

    keep = np.array(keep, dtype=int)
    return xyxy[keep], confs[keep], clss[keep]

def calculate_object_similarity(box1, box2, class1, class2):
    """Calcula similitud entre dos objetos usando múltiples criterios"""
    # 1. IoU (Intersection over Union)
    x1, y1, x2, y2 = box1
    px1, py1, px2, py2 = box2
    
    inter_w = max(0, min(x2, px2) - max(x1, px1))
    inter_h = max(0, min(y2, py2) - max(y1, py1))
    inter = inter_w * inter_h
    
    area1 = max(0, x2-x1) * max(0, y2-y1)
    area2 = max(0, px2-px1) * max(0, py2-py1)
    union = area1 + area2 - inter
    iou = inter / union if union > 0 else 0.0
    
    # 2. Distancia entre centros
    cx1, cy1 = (x1+x2)/2, (y1+y2)/2
    cx2, cy2 = (px1+px2)/2, (py1+py2)/2
    center_dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
    
    # 3. Diferencia de tamaño
    size1 = np.sqrt(area1)
    size2 = np.sqrt(area2)
    size_ratio = min(size1, size2) / max(size1, size2) if max(size1, size2) > 0 else 0
    
    # 4. Misma clase
    class_match = 1.0 if class1 == class2 else 0.0
    
    # Puntuación combinada (pesos ajustables)
    similarity = (
        0.4 * iou +  # IoU es muy importante
        0.2 * max(0, 1 - center_dist/100) +  # Distancia normalizada
        0.2 * size_ratio +  # Tamaño similar
        0.2 * class_match  # Misma clase
    )
    
    return similarity, iou, center_dist

def assign_stable_ids(xyxy, confs, clss, current_frame):
    """Asigna IDs estables basados en matching inteligente con historial"""
    global object_history, next_stable_id, frame_counter
    
    frame_counter = current_frame
    stable_ids = []
    
    if len(xyxy) == 0:
        return stable_ids
    
    # Limpiar historial de objetos muy antiguos (más de 10 frames)
    to_remove = []
    for obj_id, data in object_history.items():
        if current_frame - data['last_frame'] > 10:
            to_remove.append(obj_id)
    for obj_id in to_remove:
        del object_history[obj_id]
    
    # Límite de objetos activos simultáneos para evitar explosión de IDs
    MAX_OBJECTS = 12

    # Para cada detección actual, buscar el mejor match en el historial
    used_historical_ids = set()
    
    for i, (box, conf, cls) in enumerate(zip(xyxy, confs, clss)):
        best_match_id = None
        best_similarity = 0.0
        best_iou = 0.0
        
        # Buscar en objetos históricos
        for obj_id, data in object_history.items():
            if obj_id in used_historical_ids:
                continue
                
            hist_box = data['last_box']
            hist_cls = data.get('class', cls)
            
            similarity, iou, center_dist = calculate_object_similarity(
                box, hist_box, cls, hist_cls
            )
            
            # Criterios de matching
            if (similarity > best_similarity and 
                iou >= 0.3 and  # IoU mínimo (mantener)
                center_dist <= 100 and  # Distancia máxima algo más permisiva
                similarity >= 0.45):  # Similitud mínima más baja para adoptar más rápido
                
                best_match_id = obj_id
                best_similarity = similarity
                best_iou = iou
        
        # Asignar ID
        if best_match_id is not None:
            # Usar ID existente
            stable_ids.append(best_match_id)
            used_historical_ids.add(best_match_id)
            
            # Actualizar historial
            object_history[best_match_id].update({
                'last_box': box.copy(),
                'last_frame': current_frame,
                'stable_count': object_history[best_match_id]['stable_count'] + 1,
                'class': cls
            })
        else:
            # Sin match: intentar adoptar el ID más cercano si está muy cerca
            adopt_id = None
            min_dist = 1e9
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for obj_id, data in object_history.items():
                px1, py1, px2, py2 = data['last_box']
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                d = np.hypot(cx - pcx, cy - pcy)
                if d < min_dist:
                    min_dist = d
                    adopt_id = obj_id
            # Si está muy cerca de un objeto existente, heredar su ID
            if adopt_id is not None and min_dist <= 80:
                stable_ids.append(adopt_id)
                used_historical_ids.add(adopt_id)
                object_history[adopt_id].update({
                    'last_box': box.copy(),
                    'last_frame': current_frame,
                    'stable_count': object_history[adopt_id]['stable_count'] + 1,
                    'class': cls
                })
            # Si no, sólo crear nuevo ID si la confianza es alta y no superamos el máximo
            elif float(confs[i]) >= 0.5 and len(object_history) < MAX_OBJECTS:
                new_id = next_stable_id
                next_stable_id += 1
                stable_ids.append(new_id)
                object_history[new_id] = {
                    'last_box': box.copy(),
                    'last_frame': current_frame,
                    'stable_count': 1,
                    'class': cls
                }
            else:
                # Rechazar esta detección (ID temporal -1) para no contaminar con IDs nuevos
                stable_ids.append(-1)
    
    return stable_ids

# ===== SISTEMA DE PROFILING AVANZADO =====
class PerformanceProfiler:
    """Sistema de profiling para medir tiempos de cada componente"""
    def __init__(self, name="Profiler", enable=True):
        self.name = name
        self.enable = enable
        self.times = {}
        self.start_time = None
        self.current_operation = None
        self.frame_count = 0
        self.avg_times = {}
        self.max_times = {}
        self.min_times = {}
        
    def start(self, operation):
        """Inicia medición de una operación"""
        if not self.enable:
            return
        self.current_operation = operation
        self.start_time = perf_counter()
        
    def mark(self, operation):
        """Marca el fin de una operación y el inicio de la siguiente"""
        if not self.enable or self.start_time is None:
            return
            
        if self.current_operation:
            elapsed = perf_counter() - self.start_time
            if self.current_operation not in self.times:
                self.times[self.current_operation] = []
            self.times[self.current_operation].append(elapsed)
            
            # Actualizar estadísticas
            if self.current_operation not in self.avg_times:
                self.avg_times[self.current_operation] = 0
                self.max_times[self.current_operation] = 0
                self.min_times[self.current_operation] = float('inf')
                
            # Promedio móvil
            self.avg_times[self.current_operation] = 0.9 * self.avg_times[self.current_operation] + 0.1 * elapsed
            self.max_times[self.current_operation] = max(self.max_times[self.current_operation], elapsed)
            self.min_times[self.current_operation] = min(self.min_times[self.current_operation], elapsed)
        
        self.current_operation = operation
        self.start_time = perf_counter()
        
    def end(self):
        """Termina la medición actual"""
        if not self.enable or self.start_time is None or not self.current_operation:
            return
            
        elapsed = perf_counter() - self.start_time
        if self.current_operation not in self.times:
            self.times[self.current_operation] = []
        self.times[self.current_operation].append(elapsed)
        
        # Actualizar estadísticas
        if self.current_operation not in self.avg_times:
            self.avg_times[self.current_operation] = 0
            self.max_times[self.current_operation] = 0
            self.min_times[self.current_operation] = float('inf')
            
        self.avg_times[self.current_operation] = 0.9 * self.avg_times[self.current_operation] + 0.1 * elapsed
        self.max_times[self.current_operation] = max(self.max_times[self.current_operation], elapsed)
        self.min_times[self.current_operation] = min(self.min_times[self.current_operation], elapsed)
        
        self.current_operation = None
        self.start_time = None
        
    def get_report(self, show_details=True):
        """Genera reporte de rendimiento"""
        if not self.enable or not self.avg_times:
            return "Profiling deshabilitado o sin datos"
            
        report = f"\n=== {self.name} - REPORTE DE RENDIMIENTO ===\n"
        
        # Ordenar por tiempo promedio descendente
        sorted_ops = sorted(self.avg_times.items(), key=lambda x: x[1], reverse=True)
        
        total_time = sum(self.avg_times.values())
        
        for operation, avg_time in sorted_ops:
            percentage = (avg_time / total_time) * 100 if total_time > 0 else 0
            max_time = self.max_times.get(operation, 0)
            min_time = self.min_times.get(operation, float('inf'))
            min_time = min_time if min_time != float('inf') else 0
            
            report += f"{operation:20s}: {avg_time*1000:6.2f}ms avg | {max_time*1000:6.2f}ms max | {min_time*1000:6.2f}ms min | {percentage:5.1f}%\n"
        
        report += f"{'TOTAL':20s}: {total_time*1000:6.2f}ms\n"
        
        if show_details and self.frame_count > 0:
            fps = self.frame_count / total_time if total_time > 0 else 0
            report += f"FPS estimado: {fps:.1f}\n"
            
        return report
        
    def reset(self):
        """Reinicia las estadísticas"""
        self.times.clear()
        self.avg_times.clear()
        self.max_times.clear()
        self.min_times.clear()
        self.frame_count = 0

# Instancias globales de profiling
yolo_profiler = PerformanceProfiler("YOLO", enable=True)
ui_profiler = PerformanceProfiler("UI", enable=True)
cv2_profiler = PerformanceProfiler("OpenCV", enable=True)
tracking_profiler = PerformanceProfiler("Tracking", enable=True)
clf_profiler = PerformanceProfiler("Clasificador", enable=True)

# ====== CONFIGURACIÓN CLASIFICADOR UNIFICADO ======
CLF_MODEL_PATH = "clasificador_multiclase_torch.pt"  # PyTorch model
CLF_MODEL = None
CLF_DEVICE = None  # Se inicializa igual que YOLO_DEVICE
CLASS_NAMES = ['buenas', 'malas', 'defectuosas']  # 3 clases según el modelo
CLASSIFY_ENABLED = True

# Dispositivo unificado para todos los modelos
UNIFIED_DEVICE = None

def initialize_unified_device():
    """Inicializa el dispositivo unificado para todos los modelos"""
    global UNIFIED_DEVICE
    if UNIFIED_DEVICE is None:
        UNIFIED_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Dispositivo unificado inicializado: {UNIFIED_DEVICE}")
        if torch.cuda.is_available():
            print(f"   - GPU: {torch.cuda.get_device_name(0)}")
            print(f"   - Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    return UNIFIED_DEVICE

# Parámetros configurables del clasificador
CLF_CONF_THRESHOLD = 0.8  # Confianza mínima para clasificar como "Mala" (conservador)
CLF_MODE_CONSERVATIVE = True  # Modo conservador: preferir "Buena" por defecto

# Estado por track para clasificación
track_votes = {}  # tid -> {"best_area":0, "best_roi":None, "sum_prob":np.zeros(3), "n":0}
results_summary = {}  # tid -> texto final (Buena / Mala(Mancha) / Mala(Empaque))

# ===== FUNCIONES DEL CLASIFICADOR =====
def clf_load(model_path=""):
    """Carga el clasificador PyTorch unificado"""
    global CLF_MODEL, CLF_DEVICE, UNIFIED_DEVICE
    model_path = model_path if model_path else CLF_MODEL_PATH
    try:
        # Usar dispositivo unificado (mismo que YOLO)
        if UNIFIED_DEVICE is None:
            UNIFIED_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        CLF_DEVICE = UNIFIED_DEVICE
        print(f"🔧 Clasificador usando dispositivo unificado: {CLF_DEVICE}")
        
        # Cargar modelo PyTorch (puede ser estado del modelo o modelo completo)
        checkpoint = torch.load(model_path, map_location=CLF_DEVICE)
        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Es un checkpoint con estado del modelo
            CLF_MODEL = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # Es un checkpoint con state_dict
            print("⚠️ Modelo con state_dict - necesitamos la arquitectura")
            CLF_MODEL = None
        elif isinstance(checkpoint, dict) and 'features.0.0.weight' in checkpoint:
            # Es un state_dict directo con arquitectura personalizada
            print("🔍 Detectado state_dict personalizado - creando arquitectura...")
            try:
                # Crear modelo personalizado basado en MobileNetV2 pero con clasificador diferente
                from torchvision.models import mobilenet_v2
                import torch.nn as nn
                
                class CustomMobileNetV2(nn.Module):
                    def __init__(self, num_classes=3):
                        super().__init__()
                        # Usar features de MobileNetV2
                        mobilenet = mobilenet_v2(pretrained=False)
                        self.features = mobilenet.features
                        
                        # Clasificador personalizado (basado en el state_dict)
                        self.classifier = nn.Sequential(
                            nn.Dropout(0.2),
                            nn.Linear(1280, 128),  # Primera capa
                            nn.ReLU(),
                            nn.Dropout(0.2),
                            nn.Linear(128, num_classes)  # Capa final
                        )
                    
                    def forward(self, x):
                        x = self.features(x)
                        x = x.mean([2, 3])  # Global average pooling
                        x = self.classifier(x)
                        return x
                
                CLF_MODEL = CustomMobileNetV2(num_classes=3)
                CLF_MODEL.load_state_dict(checkpoint)
                print("✅ Arquitectura personalizada creada y pesos cargados")
            except Exception as e:
                print(f"❌ Error creando arquitectura personalizada: {e}")
                CLF_MODEL = None
        else:
            # Es el modelo completo
            CLF_MODEL = checkpoint
            
        if CLF_MODEL is not None:
            CLF_MODEL.eval()
            CLF_MODEL.to(CLF_DEVICE)
        
        print(f"✅ Clasificador PyTorch cargado: {model_path}")
        print(f"   Dispositivo: {CLF_DEVICE}")
        print(f"   Clases: {CLASS_NAMES}")
        return True
    except Exception as e:
        print(f"❌ No pude cargar el modelo PyTorch: {e}")
        CLF_MODEL = None
        return False

def clf_predict_bgr(bgr_roi):
    """Predice clase con PyTorch unificado."""
    if CLF_MODEL is None:
        n = len(CLASS_NAMES)
        base = [1.0/n]*n
        return "unknown", 0.0, base

    clf_profiler.start("clf_predict")
    
    try:
        
        # Preprocesado estándar para clasificación
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convertir BGR a RGB y aplicar transformaciones
        rgb_img = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)
        tensor_img = transform(rgb_img).unsqueeze(0).to(CLF_DEVICE)
        
        # Inferencia
        with torch.no_grad():
            outputs = CLF_MODEL(tensor_img)
            probabilities = torch.softmax(outputs, dim=1)
            prob_np = probabilities.cpu().numpy().squeeze()
            
        idx = int(np.argmax(prob_np))
        confidence = float(prob_np[idx])
        
        clf_profiler.end()
        return CLASS_NAMES[idx], confidence, prob_np.tolist()
        
    except Exception as e:
        print(f"❌ Error en clasificación PyTorch: {e}")
        clf_profiler.end()
        n = len(CLASS_NAMES)
        base = [1.0/n]*n
        return "unknown", 0.0, base

def circle_crop(img, cx, cy, r):
    """Recorta una región circular de la imagen"""
    h, w = img.shape[:2]
    y1, y2 = max(0, cy-r), min(h, cy+r)
    x1, x2 = max(0, cx-r), min(w, cx+r)
    roi = img[y1:y2, x1:x2]
    
    # Crear máscara circular
    mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
    cv2.circle(mask, (cx-x1, cy-y1), r, 255, -1)
    
    # Aplicar máscara
    result = roi.copy()
    result[mask == 0] = 0
    return result

def update_track_sample(tid, bgr, box):
    """Actualiza la muestra del track para clasificación"""
    x1, y1, x2, y2 = map(int, box)
    area = (x2 - x1) * (y2 - y1)
    st = track_votes.setdefault(tid, {"best_area": 0, "best_roi": None, "sum_prob": np.zeros(len(CLASS_NAMES)), "n": 0, "frames_seen": 0, "last_box": None, "last_frame": 0})
    
    # Incrementar contador de frames vistos
    st["frames_seen"] += 1
    
    # Guardar posición del último bounding box
    st["last_box"] = (x1, y1, x2, y2)
    st["last_frame"] = st.get("last_frame", 0) + 1
    
    # Guarda mejor ROI por área (más permisivo)
    if area > st["best_area"] * 0.8:  # Aceptar ROI si es al menos 80% del mejor
        w, h = x2 - x1, y2 - y1
        r = int(0.5 * min(w, h))
        cx, cy = x1 + w // 2, y1 + h // 2
        st["best_roi"] = circle_crop(bgr, cx, cy, r)
        st["best_area"] = area

def is_track_near_exit(tid, image_width, threshold_px=100):
    """Verifica si un track está cerca del borde derecho (zona de salida)"""
    st = track_votes.get(tid)
    if not st or not st.get("last_box"):
        return False
    
    x1, y1, x2, y2 = st["last_box"]
    # Verificar si el centro del bounding box está cerca del borde derecho
    center_x = (x1 + x2) / 2
    return center_x >= (image_width - threshold_px)

def finalize_track_class(tid, image_width=None):
    """Clasifica el track final y devuelve resultado"""
    st = track_votes.get(tid)
    if not st or CLF_SESS is None or st["best_roi"] is None:
        return None

    # Criterio de salida: 1 frame sin ver + posición cerca del borde derecho
    frames_since_last = st.get("last_frame", 0)
    near_exit = image_width and is_track_near_exit(tid, image_width)
    
    # Solo clasificar si:
    # 1. Hemos visto al menos 2 frames Y
    # 2. (Ha pasado 1 frame sin ver Y está cerca del borde) O hemos visto suficientes frames
    should_classify = (st.get("frames_seen", 0) >= 2 and 
                      ((frames_since_last >= 1 and near_exit) or st.get("frames_seen", 0) >= 5))
    
    if not should_classify:
        del track_votes[tid]
        return None

    clase, conf, prob = clf_predict_bgr(st["best_roi"])
    del track_votes[tid]

    # Aplicar umbral de confianza y modo conservador
    if CLF_MODE_CONSERVATIVE:
        # Con 2 clases (buenas/malas): sólo aceptar "mala" si conf >= umbral
        if isinstance(clase, str) and clase.lower().startswith("mala") and conf < CLF_CONF_THRESHOLD:
            clase = "Buena"
            conf = 1.0 - conf

    # Traducción al formato final (2 clases: buenas/malas)
    if isinstance(clase, str) and clase.lower().startswith("buena"):
        verdict = "Buena"
    elif isinstance(clase, str) and clase.lower().startswith("mala"):
        verdict = "Mala"
    else:
        verdict = f"Desconocida ({clase})"

    results_summary[tid] = verdict
    print(f"[INFO] Lata ID={tid} → {verdict} (conf={conf:.2f})")
    return clase, conf, prob

# ===== CONTROLES CÁMARA: GET/SET EXPOSURE y GAIN =====
def get_camera_params(device):
    params = {}
    try:
        params['ExposureTime'] = device.get_float_feature_value("ExposureTime")
    except Exception:
        try:
            params['ExposureTime'] = device.get_integer_feature_value("ExposureTime")
        except Exception:
            params['ExposureTime'] = None
    try:
        params['Gain'] = device.get_float_feature_value("Gain")
    except Exception:
        try:
            params['Gain'] = device.get_integer_feature_value("Gain")
        except Exception:
            params['Gain'] = None
    return params

def set_camera_param(device, name, value):
    try:
        if name == "ExposureTime":
            try:
                device.set_float_feature_value("ExposureTime", float(value))
            except Exception:
                device.set_integer_feature_value("ExposureTime", int(value))
        elif name == "Gain":
            try:
                device.set_float_feature_value("Gain", float(value))
            except Exception:
                device.set_integer_feature_value("Gain", int(value))
        print(f"[CAM] {name} set to {value}")
    except Exception as e:
        print(f"[ERROR] Cannot set {name}: {e}")

# ===== OPTIMIZACIONES DE RENDIMIENTO =====
def optimize_for_high_speed():
    """Aplica optimizaciones específicas para alta velocidad (300 latas/min)"""
    global PROCESS_EVERY, MISS_HOLD, RESULT_TTL_S, YOLO_CONF, YOLO_IOU
    
    print("🚀 APLICANDO OPTIMIZACIONES PARA ALTA VELOCIDAD (300 latas/min)")
    
    # Reducir procesamiento de frames (más agresivo)
    # PROCESS_EVERY = 2  # Comentado para usar valor de configuración principal
    
    # Reducir tiempo de hold para detecciones
    MISS_HOLD = 4 # Mantener detecciones 5 frames sin verlas para reducir parpadeo
    
    # Reducir TTL de resultados
    RESULT_TTL_S = 0.05  # TTL más corto para objetos rápidos  # 200ms TTL para resultados para mayor estabilidad
    
    # Ajustar parámetros YOLO para velocidad
    YOLO_CONF = 0.4  # Confianza algo más alta para evitar duplicados
    YOLO_IOU = 0.7   # IOU más alto para suprimir cajas solapadas
    
    print(f"✅ Optimizaciones aplicadas:")
    print(f"   - Procesa 1 de cada {PROCESS_EVERY} frames")
    print(f"   - Hold: {MISS_HOLD} frames")
    print(f"   - TTL: {RESULT_TTL_S}s")
    print(f"   - YOLO Conf: {YOLO_CONF}, IOU: {YOLO_IOU}")

def enable_cuda_optimizations():
    """Habilita optimizaciones CUDA unificadas con PyTorch"""
    global TORCH_CUDA_AVAILABLE
    
    if TORCH_CUDA_AVAILABLE:
        print("🚀 HABILITANDO OPTIMIZACIONES CUDA UNIFICADAS")
        
        # Configurar PyTorch para máximo rendimiento en Jetson
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        
        # Optimizaciones específicas para Jetson
        torch.set_num_threads(1)  # Un solo hilo para CUDA
        torch.set_num_interop_threads(1)
        
        # Configurar memoria GPU
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)  # Usar 80% de GPU
        
        # Configurar OpenCV para usar todos los cores disponibles
        cv2.setUseOptimized(True)
        cv2.setNumThreads(0)
        
        print("✅ Optimizaciones CUDA unificadas habilitadas")
        print(f"   - PyTorch CUDA: {torch.cuda.is_available()}")
        print(f"   - Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    else:
        print("⚠️ CUDA no disponible, usando optimizaciones CPU")

def setup_high_performance_tracking():
    """Configura el sistema de tracking para alta velocidad"""
    global bbox_ema, histories, last_seen, last_conf
    
    # Limpiar historiales existentes
    bbox_ema.clear()
    histories.clear()
    last_seen.clear()
    last_conf.clear()
    
    print("✅ Sistema de tracking optimizado para alta velocidad")

# Configuración CUDA para ONNX Runtime y OpenCV
# Comentado para JetPack 5.1.1 (CUDA 11.4)
# os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'
# os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.6/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# ===== VERIFICACIÓN CUDA =====
def check_opencv_cuda():
    """Verifica si OpenCV tiene soporte CUDA"""
    try:
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            print(f"✅ OpenCV con soporte CUDA: {cuda_devices} dispositivos")
            return True
        else:
            print("⚠️ OpenCV sin soporte CUDA")
            return False
    except:
        print("⚠️ OpenCV sin soporte CUDA")
        return False

def check_torch_cuda():
    """Verifica si PyTorch tiene soporte CUDA"""
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ PyTorch con soporte CUDA: {torch.cuda.device_count()} dispositivos")
            print(f"   Dispositivo: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ PyTorch sin soporte CUDA")
        return cuda_available
    except:
        print("⚠️ PyTorch no disponible")
        return False

# Verificar CUDA al inicio (unificado)
TORCH_CUDA_AVAILABLE = check_torch_cuda()
OPENCV_CUDA_AVAILABLE = check_opencv_cuda()  # Mantener para compatibilidad con OpenCV

# ===== FUNCIONES DE PROCESAMIENTO CON CUDA =====
def process_image_with_pytorch(image, operation="resize", target_size=(640, 640)):
    """Procesa imagen usando PyTorch tensors con CUDA"""
    try:
        # Convertir imagen a tensor PyTorch
        if len(image.shape) == 3:
            # HWC -> CHW
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        else:
            tensor = torch.from_numpy(image).float() / 255.0
        
        # Mover a GPU si está disponible
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tensor = tensor.to(device)
        
        if operation == "resize":
            # Redimensionar usando interpolación
            tensor = tensor.unsqueeze(0)  # Añadir batch dimension
            resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
            result = resized.squeeze(0)  # Quitar batch dimension
        elif operation == "cvt_color":
            # Convertir BGR a RGB
            if tensor.shape[0] == 3:  # CHW format
                result = tensor[[2, 1, 0], :, :]  # BGR -> RGB
            else:
                result = tensor
        else:
            result = tensor
        
        # Convertir de vuelta a numpy
        if len(result.shape) == 3:
            result = result.permute(1, 2, 0)  # CHW -> HWC
        result = (result * 255).byte().cpu().numpy()
        
        return result
        
    except Exception as e:
        print(f"⚠️ Error en procesamiento PyTorch: {e}, usando OpenCV")
        # Fallback a OpenCV
        if operation == "resize":
            return cv2.resize(image, target_size)
        elif operation == "cvt_color":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
# ===== ARAVIS BACKEND (sustituye al productor GenTL) =====
import gi
gi.require_version("Aravis", "0.6")
from gi.repository import Aravis, GLib

class AravisBackend:
    """Wrapper mínimo para usar Aravis como fuente de frames BGR."""
    
    def __init__(self, index=0, n_buffers=8, bayer_code=cv2.COLOR_BayerBG2BGR):
        self.index = index
        self.n_buffers = n_buffers
        self.bayer_code = bayer_code
        self.camera = None
        self.dev = None
        self.stream = None
        self.payload = 0
        self.width = 0
        self.height = 0
        self.pixfmt = "Unknown"
        self.started = False
        # Métricas de red / stream
        self._stat_last_t = time.time()
        self._stat_frames_acc = 0
        self._stat_bytes_acc = 0

    # --- util pop portable (algunos builds aceptan timeout_us, otros no)
    def _try_pop(self, poll_us=20000):
        try:
            return self.stream.try_pop_buffer(poll_us)  # con timeout
        except TypeError:
            buf = self.stream.try_pop_buffer()  # sin timeout
            if buf is None:
                GLib.usleep(poll_us)
            return buf

    # --- "nodos" de cámara (get/set) compatibles con tus helpers safe_get/safe_set
    def get_node_value(self, name, default=None):
        try:
            # nombres típicos
            if name == "PixelFormat":
                return self.pixfmt
            if name == "DeviceVendorName":
                return self.dev.get_string_feature_value("DeviceVendorName")
            if name == "DeviceModelName":
                return self.dev.get_string_feature_value("DeviceModelName")
            if name == "DeviceSerialNumber":
                return self.dev.get_string_feature_value("DeviceSerialNumber")
            if name == "AcquisitionMode":
                return self.dev.get_string_feature_value("AcquisitionMode")
            if name == "ExposureTime":
                return self.dev.get_float_feature_value("ExposureTime")
            if name == "Gain":
                return self.dev.get_float_feature_value("Gain")
            if name == "AcquisitionFrameRate":  # Aravis usa "AcquisitionFrameRate"
                return self.dev.get_float_feature_value("AcquisitionFrameRate")
            if name == "AcquisitionFrameRateEnable":
                return self.dev.get_boolean_feature_value("AcquisitionFrameRateEnable")
            if name == "TriggerMode":
                return self.dev.get_string_feature_value("TriggerMode")
            if name in ("BalanceWhiteAuto","WhiteBalanceAuto"):
                return self.dev.get_string_feature_value(name)
            if name == "Gamma":
                return self.dev.get_float_feature_value("Gamma")
            if name == "GammaEnable":
                return self.dev.get_boolean_feature_value("GammaEnable")
        except Exception:
            pass
        return default

    def set_node_value(self, name, value):
        try:
            if name == "PixelFormat":
                # intenta escribir al dispositivo si el nodo existe
                try:
                    if not self.started:
                        self.dev.set_string_feature_value("PixelFormat", str(value))
                    else:
                        raise RuntimeError("acquisition running")
                    self.pixfmt = str(value)
                    return True
                except Exception as e:
                    # no se pudo (probablemente en adquisición): guarda preferencia
                    self.pixfmt = str(value)
                    print(f"[Aravis set PixelFormat] diferido hasta próximo START ({e})")
                    return True
            if name == "ExposureTime":
                self.dev.set_float_feature_value("ExposureTime", float(value))
                return True
            if name == "Gain":
                self.dev.set_float_feature_value("Gain", float(value))
                return True
            if name == "AcquisitionFrameRate":
                self.dev.set_float_feature_value("AcquisitionFrameRate", float(value))
                return True
            if name == "TriggerMode":
                self.dev.set_string_feature_value("TriggerMode", str(value))
                return True
            if name in ("BalanceWhiteAuto","WhiteBalanceAuto"):
                self.dev.set_string_feature_value(name, str(value))
                return True
            if name in ("ExposureAuto","GainAuto"):
                self.dev.set_string_feature_value(name, str(value)); return True
            if name in ("Gamma","GammaEnable"):
                # muchas cámaras tienen estos nodos:
                if name == "Gamma": self.dev.set_float_feature_value("Gamma", float(value)); return True
                if name == "GammaEnable": self.dev.set_boolean_feature_value("GammaEnable", bool(value)); return True

        except Exception as e:
            print(f"[Aravis set {name}] {e}")
            return False
        return False

    def open(self):
        # 1) Descubrir cámaras y crear self.camera / self.dev
        Aravis.update_device_list()
        n = Aravis.get_n_devices()
        if n <= 0:
            raise RuntimeError("No cameras found (Aravis)")
        if self.index >= n:
            raise RuntimeError(f"Index fuera de rango 0..{n-1}")
        
        cam_id = Aravis.get_device_id(self.index)
        self.camera = Aravis.Camera.new(cam_id)
        if self.camera is None:
            raise RuntimeError("Aravis.Camera.new() devolvió None")
        
        self.dev = self.camera.get_device()
        if self.dev is None:
            raise RuntimeError("camera.get_device() devolvió None")

        # 2) (Opcional) tomar privilegio de control si es GigE y el nodo existe
        try:
            # No todas las cámaras (USB3) tienen GevCCP. Intentar sólo si existe.
            try:
                _ = self.dev.get_string_feature_value("GevCCP")
                have_gevccp = True
            except Exception:
                have_gevccp = False

            if have_gevccp:
                try:
                    self.dev.set_string_feature_value("GevCCP", "Control")
                except Exception:
                    # valores alternativos según fabricante
                    for v in ("ExclusiveAccess", "Exclusive", "ControlWithSwitchover", "OpenAccess"):
                        try:
                            self.dev.set_string_feature_value("GevCCP", v)
                            break
                        except Exception:
                            pass

                # Intento adicional con la API de privilegios si está disponible
                try:
                    GvcpPrivilege = getattr(Aravis, "GvcpPrivilege", None)
                    if GvcpPrivilege is not None:
                        self.dev.set_control_channel_privilege(GvcpPrivilege.CONTROL)
                except Exception:
                    pass

                # (opcional) reducir heartbeat para liberar rápido si el proceso muere
                try:
                    self.dev.set_integer_feature_value("GevHeartbeatTimeout", 2000)
                except Exception:
                    pass
        except Exception as e:
            print(f"[Aravis] Aviso al fijar privilegio: {e}")

        # 3) Fijar PixelFormat y parámetros de red antes de crear el stream
        try:
            # Intenta BayerBG8 por defecto (coherente con conversión OpenCV)
            self.dev.set_string_feature_value("PixelFormat", "BayerBG8")
            self.pixfmt = "BayerBG8"
            self.bayer_code = cv2.COLOR_BayerBG2BGR
        except Exception:
            try:
                # Fallback consistente a RG8
                self.dev.set_string_feature_value("PixelFormat", "BayerRG8")
                self.pixfmt = "BayerRG8"
                self.bayer_code = cv2.COLOR_BayerRG2BGR
            except Exception:
                pass
        # Ajustes GigE recomendados
        try:
            self.dev.set_integer_feature_value("GevSCPSPacketSize", 1500)
        except Exception:
            pass

        # Intentar fijar FPS 54 y espaciado de paquetes si los nodos existen
        try:
            try:
                self.dev.set_string_feature_value("TriggerMode", "Off")
            except Exception:
                pass
            for enabler in ("AcquisitionFrameRateEnable",):
                try:
                    self.dev.set_integer_feature_value(enabler, 1)
                except Exception:
                    try:
                        self.dev.set_boolean_feature_value(enabler, True)
                    except Exception:
                        pass
            try:
                self.dev.set_float_feature_value("AcquisitionFrameRate", 54.0)
            except Exception:
                try:
                    self.dev.set_integer_feature_value("AcquisitionFrameRate", 54)
                except Exception:
                    pass
            try:
                self.dev.set_integer_feature_value("GevSCPD", 4000)
            except Exception:
                pass
        except Exception:
            pass

        # 4) Crear stream y preparar buffers
        self.stream = self.camera.create_stream(None, None)
        if self.stream is None:
            raise RuntimeError("create_stream() failed")

        # === APLICAR ROI ANTES DE RESERVAR BUFFERS ===
        try:
            def _align(v, m=8):
                return int(v) // m * m
            def _even2(v):
                return (int(v) // 2) * 2
            h_max = int(self.dev.get_integer_feature_value("HeightMax"))
            w_max = int(self.dev.get_integer_feature_value("WidthMax"))
            # Banda por defecto: 30%..82% (ajustable luego desde RUN si se desea)
            y1 = _even2(_align(h_max * 0.30, 8))
            y2 = _even2(_align(h_max * 0.82, 8))
            new_h = _even2(max(_align(y2 - y1, 8), 64))
            new_w = _align(w_max, 8)
            self.dev.set_integer_feature_value("OffsetX", 0)
            self.dev.set_integer_feature_value("OffsetY", int(y1))
            self.dev.set_integer_feature_value("Width",   int(new_w))
            self.dev.set_integer_feature_value("Height",  int(new_h))
            try:
                import builtins
                builtins.offsetY = int(y1)
            except Exception:
                pass
            print(f"[ROI/open] Región aplicada: X=0 Y={int(y1)} W={int(new_w)} H={int(new_h)}")
        except Exception as e:
            print(f"[ROI/open] No se pudo aplicar ROI en open: {e}")

        # Releer payload tras ROI y reservar buffers del tamaño correcto
        self.payload = self.camera.get_payload()
        if not isinstance(self.payload, int) or self.payload <= 0:
            # fallback si payload no disponible
            w = int(self.dev.get_integer_feature_value("Width"))
            h = int(self.dev.get_integer_feature_value("Height"))
            self.payload = int(w * h)

        for _ in range(self.n_buffers):
            self.stream.push_buffer(Aravis.Buffer.new_allocate(self.payload))

        # 4) leer dimensiones / formato (best-effort)
        try:
            self.width = int(self.dev.get_integer_feature_value("Width"))
            self.height = int(self.dev.get_integer_feature_value("Height"))
        except Exception:
            self.width, self.height = 0, 0

        try:
            self.pixfmt = self.dev.get_string_feature_value("PixelFormat")
        except Exception:
            self.pixfmt = "Unknown"

        return self

    def start(self):
        if not self.started:
            # Aplicar ROI por GenICam ANTES de allocar buffers e iniciar
            try:
                def align(v, m=8):
                    return int(v) // m * m
                def even2(v):
                    return (int(v) // 2) * 2
                # Si el usuario ya fijó un ROI (globals), respetarlo; si no, proponer banda media
                h_max = int(self.dev.get_integer_feature_value("HeightMax"))
                w_max = int(self.dev.get_integer_feature_value("WidthMax"))
                y1 = even2(align(h_max * 0.38, 8))  # subir un poco la banda
                y2 = even2(align(h_max * 0.78, 8))
                new_h = even2(max(align(y2 - y1, 8), 64))
                new_w = align(w_max, 8)
                # Aplicar SIEMPRE el ROI antes de iniciar
                self.dev.set_integer_feature_value("OffsetX", 0)
                self.dev.set_integer_feature_value("OffsetY", int(y1))
                self.dev.set_integer_feature_value("Width",   int(new_w))
                self.dev.set_integer_feature_value("Height",  int(new_h))
                try:
                    # Si por cualquier razón ya estábamos arrancados, parar de forma segura
                    if self.started:
                        self.camera.stop_acquisition()
                        self.started = False
                except Exception:
                    pass
                # Vaciar stream de buffers antiguos y reservar con el nuevo payload
                try:
                    self.stream.flush()
                except Exception:
                    pass
                try:
                    while True:
                        b = self.stream.pop_buffer(0)
                        if b is None:
                            break
                except Exception:
                    pass
                try:
                    # Actualizar dimensiones y payload reales
                    self.width  = int(self.dev.get_integer_feature_value("Width"))
                    self.height = int(self.dev.get_integer_feature_value("Height"))
                    # Preferir payload reportado por cámara si existe
                    try:
                        self.payload = int(self.camera.get_payload())
                    except Exception:
                        self.payload = int(self.width * self.height)
                    from gi.repository import Aravis
                    for _ in range(self.n_buffers):
                        self.stream.push_buffer(Aravis.Buffer.new_allocate(self.payload))
                except Exception:
                    pass
                try:
                    import builtins
                    builtins.offsetY = int(y1)
                except Exception:
                    pass
                print(f"[ROI/backend] Región aplicada: X=0 Y={int(y1)} W={int(new_w)} H={int(new_h)}")
            except Exception as e:
                print(f"[ROI/backend] No se pudo aplicar ROI antes de start: {e}")
            # Reintentar aplicar PixelFormat recordado antes de iniciar
            try:
                # Forzar explícitamente BayerBG8 para coherencia con demosaico
                self.dev.set_string_feature_value("PixelFormat", "BayerBG8")
                self.pixfmt = "BayerBG8"
                self.bayer_code = cv2.COLOR_BayerBG2BGR
                print("[Aravis] Forzando PixelFormat BayerBG8 (coherente con cámara)")
            except Exception as e:
                print(f"[Aravis] Aviso: no pude fijar PixelFormat: {e}")
            self.camera.start_acquisition()
            self.started = True
            # Log del PixelFormat efectivo
            try:
                eff = self.dev.get_string_feature_value("PixelFormat")
            except Exception:
                eff = self.pixfmt
            print(f"[Aravis START] PixelFormat efectivo: {eff}")

    def stop(self):
        if self.started:
            self.camera.stop_acquisition()
            self.started = False

    def close(self):
        try:
            self.stop()
        except:
            ...
        self.stream = None
        self.dev = None
        self.camera = None

    # --- util: estimar bytes GVSP (payload + overhead UDP/IP/Eth aprox) ---
    def _gvsp_bytes_per_buffer(self, payload_size, pkt_size=8192):
        overhead_per_pkt = 42  # UDP+IP+Eth aprox
        n_pkts = (payload_size + pkt_size - 1) // pkt_size
        return payload_size + n_pkts * overhead_per_pkt

    # --- logging de estadísticas del stream (cada ~1s) ---
    def _log_stream_stats(self, every_sec=1.0):
        now = time.time()
        dt = now - self._stat_last_t
        if dt < every_sec:
            return
        fps = self._stat_frames_acc / dt if dt > 0 else 0.0
        mbps = (self._stat_bytes_acc * 8) / dt / 1e6
        resent = missing = completed = failures = underruns = -1
        try:
            resent = self.stream.get_n_resent_packets()
            missing = self.stream.get_n_missing_packets()
            completed = self.stream.get_n_completed_buffers()
            failures = self.stream.get_n_failures()
            underruns = self.stream.get_n_underruns()
        except Exception:
            try:
                stats = getattr(self.stream, 'get_statistics', None)
                if callable(stats):
                    _ = stats()
            except Exception:
                pass
        print(f"[NET] FPS={fps:5.1f}  Throughput={mbps:6.1f} Mb/s  resent={resent}  missing={missing}  completed={completed}  fail={failures}  underrun={underruns}")
        self._stat_last_t = now
        self._stat_frames_acc = 0
        self._stat_bytes_acc = 0

    # --- frame BGR listo para YOLO (devuelve (bgr, ts))
    def get_frame(self, timeout_ms=1000):
        if not self.started:
            return None

        waited = 0
        poll_us = 20000
        buf = None
        while waited < timeout_ms*1000:
            buf = self._try_pop(poll_us=poll_us)
            if buf is not None:
                break
            waited += poll_us

        if buf is None:
            return None

        data = buf.get_data()
        # Descartar buffers con estado no exitoso (incompletos)
        try:
            st = buf.get_status()
            if st != 0:
                try:
                    self.stream.push_buffer(buf)
                except Exception:
                    pass
                return None
        except Exception:
            pass
        # No devolver el buffer al stream hasta haber copiado (evita sobrescrituras visuales)
        # En Jetson la copia es barata a este tamaño
        if data is None:
            return None

        npbuf = np.frombuffer(bytes(data), dtype=np.uint8)
        # Usar dimensiones reales del buffer si están disponibles (evita "bandas" por stride)
        try:
            bw = int(buf.get_image_width())
            bh = int(buf.get_image_height())
        except Exception:
            bw, bh = int(self.width), int(self.height)
        try:
            bstride = int(buf.get_image_stride())
        except Exception:
            bstride = bw
        total_needed = bstride * bh
        if npbuf.size < total_needed:
            try:
                self.stream.push_buffer(buf)
            except Exception:
                pass
            return None
        raw_strided = npbuf[:total_needed].reshape((bh, bstride))
        raw = raw_strided[:, :bw]
        # Actualizar cache de dimensiones al tamaño real
        self.width, self.height = bw, bh

        # demosaic según formato conocido (mantenemos tu LUT gamma después)
        pxf = (self.pixfmt or "").upper()
        if pxf in ("MONO8",):
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        elif pxf in ("BAYERBG8","BAYER_BG8","BAYERBG8","BAYERBG8","BAYERBG8"):
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)
        elif pxf in ("BAYERRG8","BAYER_RG8"):
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerRG2BGR)
        elif pxf in ("BAYERGB8","BAYER_GB8"):
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerGB2BGR)
        elif pxf in ("BAYERGR8","BAYER_GR8"):
            bgr = cv2.cvtColor(raw, cv2.COLOR_BayerGR2BGR)
        else:
            # fallback a tu código Bayer elegido en la UI
            # Si no hay PixelFormat claro, usa el seleccionado en UI
            bgr = cv2.cvtColor(raw, self.bayer_code)

        # Devolver ahora el buffer al pool
        self.stream.push_buffer(buf)

        # --- Métricas de stream: FPS y throughput estimado ---
        try:
            bpp = 1  # BayerRG8/BayerBG8 ~ 1 byte/px
            payload_size = int(self.width * self.height * bpp)
            self._stat_frames_acc += 1
            self._stat_bytes_acc += self._gvsp_bytes_per_buffer(payload_size, pkt_size=8192)
            self._log_stream_stats(every_sec=1.0)
        except Exception:
            pass

        return bgr, time.time()


class Prof:
    """Profiler de un frame/hito. Usa perf_counter() para alta resolución."""
    __slots__ = ("t0","marks","enabled")
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.t0 = perf_counter()
        self.marks = [("start", self.t0)]

    def mark(self, label):
        if not self.enabled:
            return
        self.marks.append((label, perf_counter()))

    def report(self, prefix="PROF", print_fn=print, ret=False):
        if not self.enabled:
            return {}
        items = []
        last_t = self.marks[0][1]
        for lbl, t in self.marks[1:]:
            items.append((lbl, (t-last_t)*1000.0))  # ms desde marca anterior
            last_t = t
        total_ms = (self.marks[-1][1] - self.marks[0][1]) * 1000.0
        line = f"{prefix} " + " | ".join(f"{k}:{v:.1f}ms" for k,v in items) + f" || total:{total_ms:.1f}ms"
        print_fn(line)
        out = {k:v for k,v in items}; out["_total_ms"]=total_ms
        return out if ret else out


class EMA:
    """Media exponencial para suavizar métricas."""
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.v = None

    def push(self, x):
        self.v = x if self.v is None else (self.alpha*x+(1-self.alpha)*self.v)
        return self.v


# ====== DO0 (NRU-52S) ======
GPIO_DO0 = 446  # DO0

def do0_init():
    # export + direction=out + arranca en 0 (abierto)
    try:
        open(f"/sys/class/gpio/gpio{GPIO_DO0}/direction").close()
    except FileNotFoundError:
        with open("/sys/class/gpio/export","w") as f:
            f.write(str(GPIO_DO0))
        with open(f"/sys/class/gpio/gpio{GPIO_DO0}/direction","w") as f:
            f.write("out")
        with open(f"/sys/class/gpio/gpio{GPIO_DO0}/value","w") as f:
            f.write("0")

def do0_set(v: int):
    with open(f"/sys/class/gpio/gpio{GPIO_DO0}/value","w") as f:
        f.write("1" if v else "0")  # 1 = hunde a GND (ON), 0 = abierto (OFF)

def do0_pulse(ms: int=50, repeat: int=1, gap_ms: int=50):
    for _ in range(repeat):
        do0_set(1)
        time.sleep(ms/1000.0)
        do0_set(0)
        time.sleep(gap_ms/1000.0)

def _console_listener():
    """
    Escribe en la consola:
    - 'on' -> pulso 50 ms en DO0
    - 'on 200' -> pulso 200 ms
    - 'on 50 x3' -> 3 pulsos de 50 ms (opcional)
    Cualquier otra cosa: ignora.
    """
    print("⌨️ Consola lista: escribe 'on' o 'on <ms>' o 'on <ms> x<rep>' para pulso DO0.")
    while not stop_event.is_set():
        try:
            line = input().strip().lower()
        except EOFError:
            break
        except Exception:
            continue
        if not line:
            continue
        if not line.startswith("on"):
            continue

        # parseo simple: on [ms] [xN]
        parts = line.split()
        ms = 50
        rep = 1
        if len(parts) >= 2:
            try:
                ms = int(parts[1])
            except ValueError:
                pass
        if len(parts) >= 3 and parts[2].startswith("x"):
            try:
                rep = int(parts[2][1:])
            except ValueError:
                pass

        try:
            do0_pulse(ms=ms, repeat=rep, gap_ms=max(30, ms))
            print(f"⚡ Pulso DO0: {ms} ms x{rep}")
        except Exception as e:
            print(f"❌ Error pulsando DO0: {e}")


# global toggles
PROFILE_UI = True  # tiempos del hilo UI/captura
PROFILE_YOLO = True  # tiempos del hilo YOLO
PROFILE_CSV = False  # escribir CSV con métricas (cambia a True si lo quieres)

# CSV (opcional)
_csv_fp = None
_csv_wr = None

def _csv_open(path):
    global _csv_fp, _csv_wr
    _csv_fp = open(path, "w", newline="")
    _csv_wr = csv.writer(_csv_fp)
    _csv_wr.writerow(["ts","where","fetch","demosaic","gamma","queue","overlay","compose","display","ui_total","yolo_dequeue","yolo_track","yolo_pack","yolo_total"])

def _csv_close():
    global _csv_fp
    try:
        if _csv_fp:
            _csv_fp.close()
    finally:
        _csv_fp = None

def _csv_row(ts, where, ui=None, yolo=None):
    if not PROFILE_CSV or _csv_wr is None:
        return
    # ui / yolo son dict con claves de etapas
    def g(d,k):
        return (None if d is None else float(f"{d.get(k,0):.3f}"))
    _csv_wr.writerow([
        f"{ts:.6f}", where,
        g(ui,"fetch"), g(ui,"demosaic"), g(ui,"gamma"), g(ui,"queue"), g(ui,"overlay"), g(ui,"compose"), g(ui,"display"), g(ui,"_total_ms"),
        g(yolo,"dequeue"), g(yolo,"track"), g(yolo,"pack"), g(yolo,"_total_ms")
    ])

# <<< PERF
try:
    # Excepción específica de GenTL para manejar estados ocupados del módulo
    from genicam.gentl import BusyException as GenTLBusyException
except Exception:
    class GenTLBusyException(Exception):
        pass


# Función calculate_iou
# --------------------------------------------------
# Esta función calcula el Intersection over Union (IoU) entre dos bounding boxes.
# Cada caja se representa como una lista o tupla de cuatro valores: [x1, y1, x2, y2], donde
# (x1, y1) es la esquina superior izquierda y (x2, y2) es la esquina inferior derecha.
# El IoU es una medida de la superposición entre las dos cajas y se calcula como:
# IoU = área de intersección / área de unión
# donde el área de unión es la suma de los dos áreas menos el área de intersección.
# --------------------------------------------------
def calculate_iou(box1, box2):
    """Calcula el Intersection over Union (IoU) entre dos bounding boxes.
    
    Parámetros:
        box1 (list/tuple): Coordenadas [x1, y1, x2, y2] de la primera caja.
        box2 (list/tuple): Coordenadas [x1, y1, x2, y2] de la segunda caja.
    
    Retorna:
        float: Valor de IoU entre 0.0 y 1.0.
    """
    # Desempaquetar coordenadas de la primera caja
    x1_min, y1_min, x1_max, y1_max = box1  # Asigna las coordenadas mínimas y máximas de box1
    
    # Desempaquetar coordenadas de la segunda caja
    x2_min, y2_min, x2_max, y2_max = box2  # Asigna las coordenadas mínimas y máximas de box2
    
    # Calcular los límites de la intersección entre las dos cajas
    inter_x_min = max(x1_min, x2_min)  # El límite izquierdo es el mayor entre los dos x mínimos
    inter_y_min = max(y1_min, y2_min)  # El límite superior es el mayor entre los dos y mínimos
    inter_x_max = min(x1_max, x2_max)  # El límite derecho es el menor entre los dos x máximos
    inter_y_max = min(y1_max, y2_max)  # El límite inferior es el menor entre los dos y máximos
    
    # Verificar si las cajas se intersectan (si no, no hay área de intersección)
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        # No existe intersección válida, retornamos 0.0
        return 0.0
    
    # Calcular el área de intersección
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calcular el área de la primera caja
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    
    # Calcular el área de la segunda caja
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calcular el área de unión
    # Se suma el área de ambas cajas y se resta el área de intersección para no contarla dos veces
    union_area = area1 + area2 - inter_area
    
    # Retornar el IoU, que es la fracción del área de intersección sobre el área de unión
    return inter_area / union_area


def find_best_match_for_ghost_id(ghost_box, last_boxes, last_tids, iou_threshold=0.3):
    """Encuentra el mejor match para un ID fantasma basado en IoU"""
    if len(last_boxes) == 0:
        return None
    
    best_iou = 0.0
    best_rid = None
    
    for last_box, last_rid in zip(last_boxes, last_tids):
        if last_rid < 0:  # Saltar otros IDs fantasma
            continue
        iou = calculate_iou(ghost_box, last_box)
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_rid = last_rid
    
    return best_rid if best_iou >= iou_threshold else None


# ===== YOLO PYTORCH OPTIMIZADO =====
class YOLOPyTorchCUDA:
    """YOLO usando PyTorch optimizado para CPU en Jetson Orin"""
    
    def __init__(self, model_path: str, input_shape: int = 640):
        self.model_path = model_path
        self.input_shape = input_shape
        self.session = None
        self.input_name = None
        self.output_names = None
        self.class_names = ['can', 'hand', 'bottle']  # Clases del modelo (0=can, 1=hand, 2=bottle)
        
        # Configurar PyTorch optimizado
        self.setup_pytorch_cuda()

# ===== YOLO PYTORCH OPTIMIZADO =====
class YOLOPyTorchCUDA:
    """YOLO usando PyTorch optimizado para CPU en Jetson Orin"""
    
    def __init__(self, model_path: str, input_shape: int = 640):
        self.model_path = model_path
        self.input_shape = input_shape
        self.session = None
        self.input_name = None
        self.output_names = None
        self.class_names = ['can', 'hand', 'bottle']  # Clases del modelo (0=can, 1=hand, 2=bottle)
        
        # Configurar PyTorch optimizado
        self.setup_pytorch_cuda()
    
    def setup_pytorch_cuda(self):
        """Configura PyTorch optimizado para CUDA/CPU unificado"""
        global UNIFIED_DEVICE
        try:
            # Usar dispositivo unificado
            if UNIFIED_DEVICE is None:
                UNIFIED_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device = UNIFIED_DEVICE
            
            if torch.cuda.is_available():
                print(f"✅ Usando CUDA unificado para inferencia YOLO: {torch.cuda.get_device_name(0)}")
            else:
                print("✅ Usando CPU optimizado para inferencia YOLO")
            
            # Suprimir warnings de Ultralytics
            import warnings
            import os
            warnings.filterwarnings('ignore', category=UserWarning)
            os.environ['YOLO_VERBOSE'] = 'False'
            
            # Cargar modelo con Ultralytics - SOLO INFERENCIA
            from ultralytics import YOLO
            
            # Verificar que el archivo del modelo existe
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")
            
            # Cargar modelo con configuración específica para evitar entrenamiento
            if self.model_path.endswith('.engine'):
                print("🚀 Cargando modelo TensorRT para máxima velocidad...")
            else:
                print("🤖 Cargando modelo PyTorch...")
            
            self.model = YOLO(self.model_path)
            
            # Optimizaciones específicas para Jetson
            if hasattr(self.model, 'model'):
                # Configurar modelo para inferencia optimizada
                self.model.model.eval()
                
                # Optimizaciones de memoria
                for param in self.model.model.parameters():
                    param.requires_grad = False
                
                # Configurar para inferencia optimizada
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            # Para TensorRT/ONNX no llamar a .to() ni .model.eval()
            if not self.model_path.endswith('.engine'):
                self.model.to(self.device)
                try:
                    self.model.model.eval()
                except Exception:
                    pass

            # Preparar función de inferencia unificada
            self.is_engine = str(self.model_path).lower().endswith('.engine')
            if self.is_engine:
                # Para TensorRT, usar predict con parámetros explícitos
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
            
            # Interceptar y deshabilitar entrenamiento
            original_train = getattr(self.model, 'train', None)
            original_val = getattr(self.model, 'val', None)
            
            def dummy_train(*args, **kwargs):
                print("⚠️ Entrenamiento interceptado - ignorando")
                return None
                
            def dummy_val(*args, **kwargs):
                print("⚠️ Validación interceptada - ignorando")
                return None
            
            # Reemplazar métodos de entrenamiento
            self.model.train = dummy_train
            self.model.val = dummy_val
            
            # Configurar para solo inferencia
            if hasattr(self.model, 'overrides'):
                self.model.overrides = {}
                self.model.overrides['mode'] = 'predict'
                self.model.overrides['task'] = 'detect'
            
            print(f"✅ Modelo Ultralytics cargado: {self.model_path}")
            print(f"✅ Dispositivo: {self.device}")
            print(f"✅ Modo: Solo inferencia")
            
            # === VERIFICACIÓN CUDA EN MODELO YOLO ===
            if torch.cuda.is_available():
                print(f"🚀 YOLO usando CUDA: {torch.cuda.get_device_name(0)}")
                print(f"🚀 Memoria GPU disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                print(f"🚀 Memoria GPU usada: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            else:
                print("⚠️ YOLO usando CPU (sin CUDA)")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            print("🔄 Ignorando error de dataset y continuando...")
            
            # Intentar cargar de todas formas, ignorando errores de dataset
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                self.model.model.eval()
                
                print(f"✅ Modelo cargado ignorando errores: {self.model_path}")
                print(f"✅ Dispositivo: {self.device}")
                return True
                
            except Exception as e2:
                print(f"❌ Error final: {e2}")
                return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesa la imagen para YOLO usando PyTorch tensors"""
        try:
            # Redimensionar usando PyTorch con tamaño optimizado
            resized = process_image_with_pytorch(
                image, 
                operation="resize", 
                target_size=(YOLO_IMGSZ, YOLO_IMGSZ)
            )
            
            # Convertir a RGB usando PyTorch
            rgb = process_image_with_pytorch(
                resized, 
                operation="cvt_color"
            )
            
            # Normalizar
            normalized = rgb.astype(np.float32) / 255.0
            
            # Transponer para CHW
            transposed = np.transpose(normalized, (2, 0, 1))
            
            # Agregar dimensión de batch
            batched = np.expand_dims(transposed, axis=0)
            
            return batched
            
        except Exception as e:
            print(f"❌ Error preprocesando imagen: {e}")
            return None
    
    def postprocess_output(self, outputs: List[np.ndarray], conf_threshold: float = 0.5) -> List[Dict]:
        """Postprocesa las salidas de YOLO"""
        try:
            detections = []
            
            # Procesar cada salida
            for output in outputs:
                # Obtener dimensiones
                batch_size, num_detections, num_attributes = output.shape
                
                # Procesar cada detección
                for i in range(num_detections):
                    # Obtener confianza
                    confidence = output[0, i, 4]
                    
                    if confidence > conf_threshold:
                        # Obtener coordenadas
                        x_center = output[0, i, 0]
                        y_center = output[0, i, 1]
                        width = output[0, i, 2]
                        height = output[0, i, 3]
                        
                        # Obtener clase
                        class_scores = output[0, i, 5:]
                        class_id = np.argmax(class_scores)
                        class_confidence = class_scores[class_id]
                        
                        # Calcular coordenadas de bounding box
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': int(class_id),
                            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown',
                            'class_confidence': float(class_confidence)
                        })
            
            return detections
            
        except Exception as e:
            print(f"❌ Error postprocesando salidas: {e}")
            return []
    
    def predict(self, image: np.ndarray, conf_threshold: float = 0.5, yolo_args: Dict = None) -> List[Dict]:
        """Realiza predicción con Ultralytics PyTorch optimizado"""
        try:
            # Ejecutar inferencia con PyTorch
            with torch.no_grad():
                results = self.model(image, conf=conf_threshold, verbose=False)
                
                detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Obtener coordenadas
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'
                            })
                
                return detections
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return []

# Importar YOLO una sola vez al inicio (COMENTADO: Reemplazado por ONNX Runtime)
# try:
#     from ultralytics import YOLO
#     YOLO_AVAILABLE = True
# except ImportError:
#     YOLO_AVAILABLE = False
#     print("⚠️ YOLO no disponible. Instala: pip install ultralytics")

# Mantener compatibilidad
YOLO_AVAILABLE = True  # ONNX Runtime siempre está disponible


# ============ CONFIG ============
# --- Parche PyTorch 2.6: permitir clase del checkpoint de Ultralytics --- (COMENTADO: No necesario con ONNX)
# try:
#     import torch.serialization as _ts
#     import ultralytics.nn.tasks as _ut
#     if hasattr(_ts, "add_safe_globals"):
#         _ts.add_safe_globals([_ut.DetectionModel])
# except Exception as _e:
#     print(f"[YOLO safe_globals] no aplicado: {_e}")


def load_yaml_config(config_path="config_yolo.yaml"):
    """Carga configuración desde archivo YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuración cargada desde {config_path}")
        return config
    except FileNotFoundError:
        print(f"⚠️ Archivo {config_path} no encontrado, usando configuración por defecto")
        return {}
    except Exception as e:
        print(f"❌ Error cargando {config_path}: {e}")
        return {}


# Cargar configuración YAML
yaml_config = load_yaml_config()

# Variables globales para detección de clases
MODEL_NAMES = {}
KEEP_IDX = set()
KEEP_ALL = False

# Sistema de tracking persistente
object_history = {}  # {id: {'last_box': box, 'last_frame': frame, 'stable_count': count}}
next_stable_id = 1
frame_counter = 0

# Utilidades para tracker YAML temporal
_tracker_yaml_path = None

def build_tracker_yaml_from_dict(cfg: dict) -> str:
    """Escribe un YAML temporal compatible con Ultralytics y devuelve su ruta."""
    global _tracker_yaml_path
    base = {
        "tracker_type": "bytetrack",  # o "botsort" si luego cambias
        "track_high_thresh": 0.5,
        "track_low_thresh": 0.2,
        "new_track_thresh": 0.35,
        "match_thresh": 0.8,
        "track_buffer": 150,
        "fuse_score": True,
        "mot20": False,
    }
    if cfg:
        base.update(cfg)
    
    if _tracker_yaml_path is None:
        fd, path = tempfile.mkstemp(prefix="tracker_", suffix=".yaml")
        os.close(fd)
        _tracker_yaml_path = path
    
    with open(_tracker_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, sort_keys=False)
    
    print(f"🔧 YAML tracker temporal creado: {_tracker_yaml_path}")
    print(f" Configuración: {base}")
    return _tracker_yaml_path


def cleanup_tracker_yaml():
    """Limpia el archivo YAML temporal al cerrar el programa."""
    global _tracker_yaml_path
    if _tracker_yaml_path and os.path.exists(_tracker_yaml_path):
        try:
            os.unlink(_tracker_yaml_path)
            print(f"🧹 Archivo temporal limpiado: {_tracker_yaml_path}")
        except Exception as e:
            print(f"⚠️ Error limpiando archivo temporal: {e}")
        _tracker_yaml_path = None


# Mostrar configuración cargada
if yaml_config:
    print("📋 Configuración YAML cargada:")
    if 'yolo' in yaml_config:
        yolo_cfg = yaml_config['yolo']
        print(f" Modelo: {yolo_cfg.get('model_path', 'N/A')}")
        print(f" Confianza: {yolo_cfg.get('confidence_threshold', 'N/A')}")
        print(f" IOU: {yolo_cfg.get('iou_threshold', 'N/A')}")
        print(f" Clases: {yolo_cfg.get('classes', 'N/A')}")
    if 'tracker' in yaml_config:
        tracker_cfg = yaml_config['tracker']
        print(f" Track Buffer: {tracker_cfg.get('track_buffer', 'N/A')}")
        print(f" Trajectory Retention: {tracker_cfg.get('trajectory_retention', 'N/A')}")
else:
    print("⚠️ Usando configuración por defecto (YAML no disponible)")


# --- nombres de ventanas (ASCII, sin acentos) ---
WIN_MAIN = "YOLO + GenTL - StViewer Style"
WIN_INFO = "INFO_VIEW"
WIN_CONFIG = "CONFIG_EDIT"
config_open = False
info_open = False

# --- GenTL / CTI ---
CTI = os.environ.get("OMRON_CTI", r"C:\Program Files\Common Files\OMRON_SENTECH\GenTL\v1_5\StGenTL_MD_VC141_v1_5.cti")
DEVICE_INDEX = 0  # si tienes 1 cámara, 0

# --- Cámara / Pixel ---
PIXEL_FORMAT = "BayerBG8"  # STC-MCS202POE por defecto en color BayerBG8 (ajusta según tu cámara)
BAYER_CODE = cv2.COLOR_BayerBG2BGR  # cambia a RG/GR/GB si corresponde a tu sensor real

# --- Inferencia / Tracking (basado en track_speed.py) ---
# Cargar configuración desde YAML o usar valores por defecto
yolo_config = yaml_config.get('yolo', {})
YOLO_WEIGHTS = os.path.join(os.path.dirname(__file__), yolo_config.get('model_path', "v2_yolov8n_HERMASA_finetune.pt"))
YOLO_CONF = yolo_config.get('confidence_threshold', 0.40)  # Usar valor del YAML
YOLO_IOU = yolo_config.get('iou_threshold', 0.25)  # Usar valor del YAML
TRACKER_CFG = "bytetrack.yaml"  # el mismo que usas en track_speed.py

# Cargar configuración del tracker desde YAML
tracker_config = yaml_config.get('tracker', {})
TRACK_BUFFER = tracker_config.get('track_buffer', 15)  # track_buffer desde YAML
TRAJECTORY_RET = tracker_config.get('trajectory_retention', 50)  # trajectory_ret desde YAML
PIX_PER_M = 1000.0  # píxeles por metro (ajusta según tu setup)
YOLO_CLASSES = yolo_config.get('classes', ["can", "hand"])  # clases a detectar desde YAML

# --- Clasificación de lata ---
RUN_CLASSIFIER_EVERY_N = 2  # frecuencia por track-id (cada N apariciones)
CLASSIFY_MAX_PER_FRAME = 2  # tope de latas a clasificar por frame para no saturar

# --- Overlay / Panel ---
DRAW_TRAILS = True
HISTORY_LEN = 30
MAX_QUEUE = 3  # cola ultra pequeña para mínima latencia
DOWNSCALE = 1.0  # fijo para UI; reescala solo en hilo YOLO si se desea
SKIP_FRAMES = 1  # procesar todos los frames para mejor tracking
YOLO_TARGET_FPS = 10.0  # FPS objetivo para YOLO (sincronizar con cámara)
YOLO_MAX_PROCESS_TIME = 0.11  # 120ms máximo por frame (8.3 FPS)
PROCESS_EVERY = 1  # Procesar cada frame para mejor tracking durante ajustes
MISS_HOLD = 2  # aguanta 2 frames sin detección (para objetos rápidos)
YOLO_IMGSZ = 224  # Tamaño reducido para máxima velocidad en Jetson

# ROI opcional (y1:y2, x1:x2) para reducir área de inferencia
# Deja en None para desactivar recorte
ROI_Y1 = None
ROI_Y2 = None
ROI_X1 = None
ROI_X2 = None

# ============ Estado global ============
cap_queue = queue.Queue(maxsize=MAX_QUEUE)  # frames crudos desde cámara
infer_queue = queue.Queue(maxsize=MAX_QUEUE)  # resultados detección/tracking para UI/Clasificador
evt_queue = queue.Queue()

stop_event = threading.Event()  # Para detener el pipeline

# Per-track buffers (como en track_speed.py)
histories = {}  # track_id -> [(x,y), ...]. track_id -> deque con (t, cx, cy, w, h) para predicción
last_seen = {}  # track_id -> contador de frames (para RUN_CLASSIFIER_EVERY_N). track_id -> frames_sin_ver

# Suavizado EMA para bounding boxes
bbox_ema = {}  # track_id -> np.array([x1,y1,x2,y2], float)
last_conf = {}  # track_id -> última_confianza (para histéresis)

# Variables para optimización de detección
last_boxes = None  # Últimas cajas detectadas
last_ids = None  # Últimos IDs detectados
miss_cnt = 0  # Contador de frames sin detección
ema_state = None  # Estado para suavizado EMA

# Persistencia de resultados del hilo YOLO para overlay estable
last_infer = None  # Último paquete de inferencia recibido
last_draw_ts = 0.0  # Momento del último paquete recibido
RESULT_TTL_S = 1  # Persistencia más larga para anti-parpadeo.  # sube el hold para evitar parpadeo si YOLO va justo

# Dimensiones actuales de la imagen (para detección de clics) y offset de panel
current_img_w = 1280
current_img_h = 720
# Offset horizontal donde empieza el panel de control en la ventana
panel_offset_x = 1280   # mismo ancho que usas en la portada inicial
PANEL_WIDTH = 350
 
# Persistencia simple para dibujar cajas directamente en la UI
color_map = {}
lock_boxes = threading.Lock()
last_pos = {}  # track_id -> (x, y, t) para velocidad
velocities_sum = {}  # track_id -> suma de velocidades
velocities_count = {}  # track_id -> contador de velocidades

# Clases externas
detector = None  # SimpleTunaDetector (opcional)

# Variables de estado de la interfaz (como en vista_gentl.py original)
button_clicked = None
acquisition_running = False
config_visible = False
gamma_actual = 0.8
patron_actual = "BG"
exposure_auto = True
gain_auto = True
exposure_auto_temp = True
gain_auto_temp = True
exposure_auto_original = True
gain_auto_original = True

# Variables para YOLO
yolo_model = None
yolo_thread = None
yolo_running = False

# LUT de gamma precalculado para mejor rendimiento
gamma_lut = None
last_gamma = None


# ============ Utilidades ============
def draw_rec_indicator(img, seconds_left):
    try:
        # Fondo semitransparente para mejor visibilidad
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (170, 50), (0, 0, 0), -1)
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # Indicador
        cv2.circle(img, (28, 30), 10, (0, 0, 255), -1)
        cv2.putText(img, f"REC {max(0, int(seconds_left)):02d}s", (50, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    except Exception:
        pass


def update_gamma_lut(gamma_value):
    """Actualiza el LUT de gamma solo cuando cambia el valor"""
    global gamma_lut, last_gamma
    if last_gamma is None or abs(gamma_value - last_gamma) > 1e-3:
        gamma_lut = np.array([(i/255.0)**(1.0/gamma_value)*255 for i in range(256)]).astype('uint8')
        last_gamma = gamma_value
    return gamma_lut


def smooth_bbox(track_id, bbox):
    """Suavizado adaptativo: si se mueve poco, suaviza mucho; si se mueve rápido, suaviza poco."""
    bb = np.array(bbox, dtype=np.float32)
    prev = bbox_ema.get(track_id)
    if prev is None:
        bbox_ema[track_id] = bb
        return bb

    # Calcular velocidad de movimiento (optimizado para alta velocidad)
    speed = np.hypot(*(bb[0:2]-prev[0:2])) + np.hypot(*(bb[2:4]-prev[2:4]))
    
    # Alpha adaptativo optimizado para estabilidad: más suavizado para reducir parpadeo
    if speed < 5:  # Movimiento muy lento
        alpha = 0.1  # Suavizado muy fuerte para estabilidad
    elif speed < 15:  # Movimiento lento
        alpha = 0.2  # Suavizado fuerte
    elif speed < 30:  # Movimiento medio
        alpha = 0.3  # Suavizado medio
    else:  # Movimiento rápido
        alpha = 0.5  # Suavizado ligero pero estable
    
    bbox_ema[track_id] = alpha * prev + (1.0 - alpha) * bb
    return bbox_ema[track_id]


def node_limits(n):
    lo = getattr(n, "min", None)
    hi = getattr(n, "max", None)
    try:
        lo = float(lo) if lo is not None else None
        hi = float(hi) if hi is not None else None
    except:
        pass
    return lo, hi


def safe_get(nm, name, default=None):
    # Soporte para AravisBackend
    if hasattr(nm, 'get_node_value'):
        return nm.get_node_value(name, default)
    
    # Ruta antigua (Harvester)
    try:
        n = getattr(nm, name)
        if n is None:
            return default
        if hasattr(n, "is_readable") and callable(n.is_readable) and not n.is_readable():
            return default
        return n.value
    except Exception:
        return default


def safe_set(nm, name, value):
    # Soporte para AravisBackend
    if hasattr(nm, 'set_node_value'):
        return nm.set_node_value(name, value)
    
    # Ruta antigua (Harvester)
    try:
        n = getattr(nm, name)
        if n is None:
            return False
        if hasattr(n, "is_writable") and callable(n.is_writable) and not n.is_writable():
            return False
        n.value = value
        return True
    except Exception:
        return False


# destruye ventana de forma segura
def safe_destroy(win):
    try:
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 0:
            cv2.destroyWindow(win)
    except:
        pass


def demosaic_bayer8(raw):
    """Convierte imagen Bayer 8-bit a BGR"""
    if raw is None:
        return None
    return cv2.cvtColor(raw, BAYER_CODE)


def circle_crop(img, cx, cy, r, pad=6):
    """Recorta ROI circular alrededor de una detección"""
    h, w = img.shape[:2]
    r = int(r)
    cx, cy = int(cx), int(cy)
    x1 = max(0, cx - (r+pad))
    x2 = min(w, cx + (r+pad))
    y1 = max(0, cy - (r+pad))
    y2 = min(h, cy + (r+pad))
    crop = img[y1:y2, x1:x2].copy()
    return crop


def calculate_velocity(raw_tid, xc, yc, frame_t):
    """Calcula velocidad como en track_speed.py"""
    if raw_tid in last_pos:
        x0, y0, _ = last_pos[raw_tid]
        dist_m = math.hypot(xc-x0, yc-y0) / PIX_PER_M
        v_ms = (dist_m / frame_t)
        velocities_sum[raw_tid] = velocities_sum.get(raw_tid, 0.) + v_ms
        velocities_count[raw_tid] = velocities_count.get(raw_tid, 0) + 1
        last_pos[raw_tid] = (xc, yc, frame_t)
    return velocities_sum.get(raw_tid, 0.) / max(1, velocities_count.get(raw_tid, 1))


def predict_next(tid, dt=1.0):
    """Predicción entre frames perdidos (E) - extrapolación con velocidad"""
    H = histories.get(tid)
    if not H or len(H) < 2:
        return None
    
    # Obtener últimas dos posiciones
    (_, x1, y1, w1, h1), (_, x0, y0, w0, h0) = H[-1], H[-2]
    
    # Calcular velocidad
    vx, vy = (x1 - x0), (y1 - y0)
    
    # Extrapolar posición
    return np.array([x1 + vx * dt, y1 + vy * dt, w1, h1], np.float32)


_ema_state = {}

def smooth_ema(boxes, alpha=0.6):
    """
    boxes: list[(x1,y1,x2,y2,tid)] o ndarray Nx5. Devuelve lista suavizada por ID.
    """
    global _ema_state
    if boxes is None:
        return []
    if isinstance(boxes, np.ndarray):
        try:
            boxes = boxes.tolist()
        except Exception:
            return []
    if not boxes:
        return []
    out = []
    for item in boxes:
        if len(item) >= 5:
            x1, y1, x2, y2, tid = item[:5]
        else:
            x1, y1, x2, y2 = item[:4]
            tid = -1
        v = np.array([x1, y1, x2, y2], dtype=float)
        prev = _ema_state.get(tid)
        s = v if prev is None else (alpha * v + (1.0 - alpha) * prev)
        if tid >= 0:
            _ema_state[tid] = s
        xi, yi, xj, yj = s.astype(int).tolist()
        out.append((xi, yi, xj, yj, tid))
    return out


def classify_one_can(bgr, box, label="can"):
    """Hook de clasificación por ROI (buena/mala) con tu SimpleTunaDetector"""
    x1, y1, x2, y2 = map(int, box)
    w = x2 - x1
    h = y2 - y1
    # radio aproximado de la lata dentro de la caja
    r = int(0.5 * min(w, h))
    cx, cy = x1 + w//2, y1 + h//2
    roi = circle_crop(bgr, cx, cy, r)
    verdict, extra = "unknown", {}
    
    if detector is not None:
        try:
            # Preferimos API en memoria si la tienes; fallback a archivo temporal
            tmp = "_tmp_can.jpg"
            cv2.imwrite(tmp, roi)
            result = detector.analyze_image(tmp, save_debug=True)  # genera overlays/metrics en defect_debug/
            os.remove(tmp)
            verdict = "OK" if result.get("status","ok") == "ok" and not result.get("has_defects", False) else "BAD"
            extra = result
        except Exception as e:
            verdict = f"error:{e}"
    
    return verdict, extra


# ============ Funciones de Interfaz (copiadas del original) ============
def get(nm, name, default=None, verbose=True):
    """Función mejorada para obtener valores de nodos GenICam"""
    try:
        n = getattr(nm, name)
        if n is None:
            if verbose:
                print(f"❌ Nodo '{name}' no encontrado")
            return default
        # ← llamar como método
        if hasattr(n, "is_readable") and callable(n.is_readable) and not n.is_readable():
            if verbose:
                print(f"❌ Nodo '{name}' no es legible")
            return default
        value = n.value
        if verbose:
            print(f"📖 '{name}' = {value}")
        return value
    except Exception as e:
        if verbose:
            print(f"❌ Error leyendo '{name}': {e}")
        return default


def set_(nm, name, value):
    """Función mejorada para establecer valores en nodos GenICam"""
    try:
        # Obtener el nodo
        n = getattr(nm, name)
        if n is None:
            print(f"❌ Nodo '{name}' no encontrado")
            return False
        # ← llamar como método
        if hasattr(n, "is_writable") and callable(n.is_writable) and not n.is_writable():
            print(f"❌ Nodo '{name}' no es escribible")
            return False
        # Verificar si el valor es válido
        try:
            # Intentar establecer el valor
            n.value = value
            print(f"✅ '{name}' = {value} (establecido correctamente)")
            return True
        except Exception as e:
            print(f"❌ Error estableciendo '{name}' = {value}: {e}")
            return False
    except Exception as e:
        print(f"❌ Error accediendo al nodo '{name}': {e}")
        return False


def crear_panel_control_stviewer(img_width, img_height):
    """Crea el panel de control estilo StViewer (copiado del original)"""
    # Asegurar altura mínima para dibujar todos los controles (incluyendo clasificador + tracks activos)
    min_h = 900
    ph = max(int(img_height), min_h)
    panel = np.full((ph, 350, 3), (40, 40, 40), dtype=np.uint8)
    
    # Título con gradiente
    cv2.rectangle(panel, (0, 0), (350, 60), (64, 64, 64), -1)
    cv2.putText(panel, "YOLO + GenTL", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Estado de la cámara
    y_offset = 80
    cv2.rectangle(panel, (20, y_offset+10), (330, y_offset+40), (64, 64, 64), -1)
    cv2.rectangle(panel, (20, y_offset+10), (330, y_offset+40), (255, 0, 0), 2)
    cv2.putText(panel, "DETENIDA", (30, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Botones RUN/STOP
    y_offset = 150
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (0, 150, 0), -1)
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "RUN", (60, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (0, 0, 150), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "STOP", (200, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Botón GRABAR (debajo de RUN/STOP)
    y_offset = 190
    cv2.rectangle(panel, (20, y_offset+10), (310, y_offset+40), (0, 0, 255), -1)
    cv2.rectangle(panel, (20, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "GRABAR 60s", (90, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Mini-checklist de configuración optimizada
    y_offset = 200
    cv2.putText(panel, "CONFIG OPTIMIZADA:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.putText(panel, "12 FPS | 5ms Expo | 24dB Gain", (20, y_offset+20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(panel, "Conf: 0.30 | IOU: 0.45 | Track: 90", (20, y_offset+35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    # Botones AWB/AUTO CAL
    y_offset = 220
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (100, 100, 0), -1)
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "AWB ONCE", (30, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (0, 100, 100), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "AUTO CAL", (180, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Slider Gamma
    y_offset = 280
    cv2.putText(panel, "Gamma:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(panel, (20, y_offset+10), (310, y_offset+35), (64, 64, 64), -1)
    cv2.rectangle(panel, (20, y_offset+10), (310, y_offset+35), (128, 128, 128), 2)
    
    # Patrones Bayer
    y_offset = 340
    cv2.putText(panel, "Bayer:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    patrones = [("BG", 20), ("RG", 90), ("GR", 160), ("GB", 230)]
    for patron, x in patrones:
        cv2.rectangle(panel, (x, y_offset+10), (x+60, y_offset+35), (80, 80, 80), -1)
        cv2.rectangle(panel, (x, y_offset+10), (x+60, y_offset+35), (255, 255, 255), 2)
        cv2.putText(panel, patron, (x+20, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Métricas YOLO
    y_offset = 400
    cv2.putText(panel, "YOLO Stats:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # Mostrar información de dispositivos
    device_info = "PyTorch: CPU | OpenCV: CPU"
    cv2.putText(panel, f"Devices: {device_info}", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(panel, "FPS: 0.0", (20, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Tracks: 0", (20, y_offset+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Detections: 0", (20, y_offset+85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Controles del Clasificador
    y_offset = 500
    cv2.putText(panel, "CLASIFICADOR:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
    
    # Barra de confianza para clasificación
    cv2.putText(panel, f"Confianza: {CLF_CONF_THRESHOLD:.1f}", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(panel, (20, y_offset+30), (310, y_offset+50), (64, 64, 64), -1)
    cv2.rectangle(panel, (20, y_offset+30), (310, y_offset+50), (128, 128, 128), 2)
    # Barra de progreso de confianza
    conf_width = int((CLF_CONF_THRESHOLD - 0.5) / 0.5 * 290)  # 0.5-1.0 -> 0-290px
    conf_width = max(0, min(290, conf_width))
    cv2.rectangle(panel, (20, y_offset+30), (20+conf_width, y_offset+50), (255, 165, 0), -1)
    
    # Botones de ajuste de confianza
    cv2.rectangle(panel, (20, y_offset+55), (80, y_offset+75), (100, 100, 100), -1)
    cv2.rectangle(panel, (20, y_offset+55), (80, y_offset+75), (255, 255, 255), 2)
    cv2.putText(panel, "-0.1", (25, y_offset+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.rectangle(panel, (90, y_offset+55), (150, y_offset+75), (100, 100, 100), -1)
    cv2.rectangle(panel, (90, y_offset+55), (150, y_offset+75), (255, 255, 255), 2)
    cv2.putText(panel, "+0.1", (95, y_offset+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Modo conservador
    mode_color = (0, 255, 0) if CLF_MODE_CONSERVATIVE else (255, 0, 0)
    cv2.rectangle(panel, (160, y_offset+55), (310, y_offset+75), mode_color, -1)
    cv2.rectangle(panel, (160, y_offset+55), (310, y_offset+75), (255, 255, 255), 2)
    mode_text = "CONSERVADOR" if CLF_MODE_CONSERVATIVE else "NORMAL"
    cv2.putText(panel, mode_text, (170, y_offset+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Estadísticas del clasificador
    y_offset = 580
    cv2.putText(panel, "Clasificadas: 0", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel, "Buenas: 0", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(panel, "Malas: 0", (20, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Botón Config
    y_offset = 640
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (100, 100, 100), -1)
    cv2.rectangle(panel, (20, y_offset+10), (160, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "CONFIG", (30, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Botón Info (derecha)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (0, 100, 150), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "INFO", (200, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Botón Salir (derecha)
    y_offset = 710
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (100, 0, 0), -1)
    cv2.rectangle(panel, (170, y_offset+10), (310, y_offset+40), (255, 255, 255), 2)
    cv2.putText(panel, "SALIR", (200, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return panel


def detectar_clic_panel_control(x, y, panel_offset_x, img_height):
    """Detecta clics en el panel de control lateral usando offset explícito"""
    panel_x_click = x - panel_offset_x
    panel_y_click = y
    
    # Debug: Log de todos los clics en el panel
    print(f" Clic en panel: x={panel_x_click}, y={panel_y_click}")
    
    if panel_x_click < 0 or panel_x_click >= PANEL_WIDTH or panel_y_click < 0 or panel_y_click >= img_height:
        return None
    
    # Botón RUN
    if 20 <= panel_x_click <= 160 and 160 <= panel_y_click <= 195:
        return "RUN"
    # Botón STOP
    elif 170 <= panel_x_click <= 310 and 160 <= panel_y_click <= 195:
        return "STOP"
    # Botón GRABAR 60s (debajo de RUN/STOP)
    elif 20 <= panel_x_click <= 310 and 200 <= panel_y_click <= 240:
        print(" Botón GRABAR 60s detectado!")
        return "RECORD_60S"
    # Botón AWB ONCE
    elif 20 <= panel_x_click <= 160 and 230 <= panel_y_click <= 260:
        return "AWB_ONCE"
    # Botón AUTO CAL
    elif 170 <= panel_x_click <= 310 and 230 <= panel_y_click <= 260:
        return "AUTO_CAL"
    # Slider Gamma - área más amplia para facilitar el clic
    elif 20 <= panel_x_click <= 310 and 290 <= panel_y_click <= 325:
        # Calcular valor de gamma basado en la posición X (rango más amplio 0.3–1.5)
        gamma_value = 0.3 + (panel_x_click - 20) * (1.5 - 0.3) / 290
        # Redondear a 2 decimales para evitar valores muy específicos
        gamma_value = round(gamma_value, 2)
        gamma_value = max(0.3, min(1.5, gamma_value))  # Limitar entre 0.3 y 1.5
        return f"GAMMA_{gamma_value:.2f}"
    # Patrones Bayer - área más amplia para facilitar el clic
    elif 20 <= panel_x_click <= 80 and 350 <= panel_y_click <= 385:
        return "BAYER_BG"
    elif 90 <= panel_x_click <= 150 and 350 <= panel_y_click <= 385:
        return "BAYER_RG"
    elif 160 <= panel_x_click <= 220 and 350 <= panel_y_click <= 385:
        return "BAYER_GR"
    elif 230 <= panel_x_click <= 290 and 350 <= panel_y_click <= 385:
        return "BAYER_GB"
    # Controles del Clasificador
    # Botón -0.1 confianza
    elif 20 <= panel_x_click <= 80 and 555 <= panel_y_click <= 575:
        return "CLF_CONF_DOWN"
    # Botón +0.1 confianza
    elif 90 <= panel_x_click <= 150 and 555 <= panel_y_click <= 575:
        return "CLF_CONF_UP"
    # Botón modo conservador/normal
    elif 160 <= panel_x_click <= 310 and 555 <= panel_y_click <= 575:
        return "CLF_MODE_TOGGLE"
    
    # Botón CONFIG
    elif 20 <= panel_x_click <= 160 and 650 <= panel_y_click <= 680:
        return "CONFIG"
    # Botón INFO (derecha)
    elif 170 <= panel_x_click <= 310 and 650 <= panel_y_click <= 680:
        return "INFO"
    # Botón SALIR (derecha)
    elif 170 <= panel_x_click <= 310 and 720 <= panel_y_click <= 750:
        return "EXIT"
    
    # Controles del Clasificador
    # Botón -0.1 confianza
    elif 20 <= panel_x_click <= 80 and 555 <= panel_y_click <= 575:
        return "CLF_CONF_DOWN"
    # Botón +0.1 confianza
    elif 90 <= panel_x_click <= 150 and 555 <= panel_y_click <= 575:
        return "CLF_CONF_UP"
    # Botón modo conservador/normal
    elif 160 <= panel_x_click <= 310 and 555 <= panel_y_click <= 575:
        return "CLF_MODE_TOGGLE"
    
    return None


def actualizar_panel_control(panel, metricas, estado_camara, img_width, img_height, gamma_actual=0.8, patron_actual="BG", yolo_stats=None, cam=None):
    """Actualiza el panel de control con información en tiempo real"""
    # Crear una copia del panel base
    panel_actualizado = panel.copy()
    
    # Actualizar estado de la cámara
    y_offset = 80
    color_estado = (0, 255, 0) if estado_camara else (255, 0, 0)
    texto_estado = "GRABANDO" if estado_camara else "DETENIDA"
    cv2.rectangle(panel_actualizado, (20, y_offset+10), (330, y_offset+40), (64, 64, 64), -1)
    cv2.rectangle(panel_actualizado, (20, y_offset+10), (330, y_offset+40), color_estado, 2)
    cv2.putText(panel_actualizado, texto_estado, (30, y_offset+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_estado, 2)
    
    # Actualizar métricas YOLO (solo tracks y detecciones, sin FPS)
    y_offset = 400
    if yolo_stats:
        cv2.putText(panel_actualizado, f"Tracks: {yolo_stats.get('tracks', 0)}", (20, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, f"Detections: {yolo_stats.get('detections', 0)}", (20, y_offset+65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel_actualizado, f"Confidence: {YOLO_CONF:.2f}", (20, y_offset+85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mostrar estadísticas de procesamiento adaptativo
        if 'skip_ratio' in yolo_stats:
            skip_pct = yolo_stats['skip_ratio'] * 100
            cv2.putText(panel_actualizado, f"Skip: {skip_pct:.0f}%", (20, y_offset+105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Actualizar indicador de gamma
    y_offset = 280
    # Mapear 0.3–1.5 al slider de 20–310
    gamma_pos = int(20 + (gamma_actual - 0.3) * 290 / (1.5 - 0.3))
    cv2.rectangle(panel_actualizado, (20, y_offset+10), (310, y_offset+35), (64, 64, 64), -1)
    cv2.rectangle(panel_actualizado, (20, y_offset+10), (310, y_offset+35), (128, 128, 128), 2)
    cv2.rectangle(panel_actualizado, (gamma_pos-8, y_offset+5), (gamma_pos+8, y_offset+40), (0, 255, 255), -1)
    cv2.putText(panel_actualizado, f"{gamma_actual:.2f}", (gamma_pos-15, y_offset+55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Actualizar patrón Bayer seleccionado
    y_offset = 340
    patrones = [("BG", 20), ("RG", 90), ("GR", 160), ("GB", 230)]
    for i, (patron, x) in enumerate(patrones):
        if patron == patron_actual:
            color = (150, 150, 150)  # Seleccionado
            borde = (0, 255, 255)  # Borde cyan
        else:
            color = (80, 80, 80)  # No seleccionado
            borde = (255, 255, 255)  # Borde blanco
        cv2.rectangle(panel_actualizado, (x, y_offset+10), (x+60, y_offset+35), color, -1)
        cv2.rectangle(panel_actualizado, (x, y_offset+10), (x+60, y_offset+35), borde, 2)
        cv2.putText(panel_actualizado, patron, (x+20, y_offset+28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    status = "ON" if yolo_running else "OFF"
    cv2.putText(panel_actualizado, f"YOLO: {status}", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0) if yolo_running else (0,0,255), 1)
    
    # Actualizar controles del clasificador
    y_offset = 500
    # Barra de confianza actualizada
    cv2.putText(panel_actualizado, f"Confianza: {CLF_CONF_THRESHOLD:.1f}", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.rectangle(panel_actualizado, (20, y_offset+30), (310, y_offset+50), (64, 64, 64), -1)
    cv2.rectangle(panel_actualizado, (20, y_offset+30), (310, y_offset+50), (128, 128, 128), 2)
    # Barra de progreso de confianza actualizada
    conf_width = int((CLF_CONF_THRESHOLD - 0.5) / 0.5 * 290)  # 0.5-1.0 -> 0-290px
    conf_width = max(0, min(290, conf_width))
    cv2.rectangle(panel_actualizado, (20, y_offset+30), (20+conf_width, y_offset+50), (255, 165, 0), -1)
    
    # Modo conservador actualizado
    mode_color = (0, 255, 0) if CLF_MODE_CONSERVATIVE else (255, 0, 0)
    cv2.rectangle(panel_actualizado, (160, y_offset+55), (310, y_offset+75), mode_color, -1)
    cv2.rectangle(panel_actualizado, (160, y_offset+55), (310, y_offset+75), (255, 255, 255), 2)
    mode_text = "CONSERVADOR" if CLF_MODE_CONSERVATIVE else "NORMAL"
    cv2.putText(panel_actualizado, mode_text, (170, y_offset+70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Estadísticas del clasificador en tiempo real
    y_offset = 580
    total_clasificadas = len(results_summary)
    buenas = sum(1 for v in results_summary.values() if v == "Buena")
    malas = total_clasificadas - buenas
    
    cv2.putText(panel_actualizado, f"Clasificadas: {total_clasificadas}", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(panel_actualizado, f"Buenas: {buenas}", (20, y_offset+25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(panel_actualizado, f"Malas: {malas}", (20, y_offset+45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Mostrar clasificaciones en tiempo real de tracks activos
    y_offset = 630
    cv2.putText(panel_actualizado, "Tracks Activos:", (20, y_offset+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    track_count = 0
    for tid, track_data in track_votes.items():
        if track_data["best_roi"] is not None and track_count < 3:  # Máximo 3 tracks
            try:
                class_name, confidence, probs = clf_predict_bgr(track_data["best_roi"])
                is_good = isinstance(class_name, str) and class_name.lower().startswith("buena")
                color = (0, 255, 0) if is_good else (255, 0, 0)
                cv2.putText(panel_actualizado, f"T{tid}: {class_name} ({confidence:.2f})", 
                           (20, y_offset+25+track_count*20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                track_count += 1
            except Exception:
                pass
    
    # Mostrar valores de cámara para verificar cambios (usar safe_get)
    if cam is not None:
        y0 = 400  # debajo de YOLO Stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        try:
            et = safe_get(cam, 'ExposureTime', 'N/A')
            gn = safe_get(cam, 'Gain', 'N/A')
            fr = safe_get(cam, 'AcquisitionFrameRate', 'N/A')
            cv2.putText(panel_actualizado, f"ET(us): {et}", (20, y0+130), font, 0.5, (255,255,255), 1)
            cv2.putText(panel_actualizado, f"Gain(dB): {gn}", (20, y0+150), font, 0.5, (255,255,255), 1)
            cv2.putText(panel_actualizado, f"FPS: {fr}", (20, y0+170), font, 0.5, (255,255,255), 1)
        except:
            pass
    
    return panel_actualizado


def handle_mouse_click(event, x, y, flags, param):
    """Maneja los clics del mouse en toda la interfaz"""
    global button_clicked, gamma_actual, patron_actual, current_img_w, current_img_h, cv_code_bayer, cam, panel_offset_x, CLF_CONF_THRESHOLD, CLF_MODE_CONSERVATIVE
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Debug de clics
        print(f"[click] x={x}, y={y}, panel_x={x - panel_offset_x}, panel_y={y}")
        
        # Detectar clic en el panel de control usando dimensiones actuales
        # Usar altura efectiva del panel (considera padding mínimo 900 para clasificador + tracks)
        panel_h_effective = max(current_img_h, 900)
        accion = detectar_clic_panel_control(x, y, panel_offset_x, panel_h_effective)
        if accion:
            evt_queue.put(accion)
            
            if accion == "RUN":
                print("🚀 Botón RUN presionado - Iniciando cámara...")
            elif accion == "STOP":
                print("⏹️ Botón STOP presionado - Pausando cámara...")
            elif accion == "AWB_ONCE":
                print("🔄 AWB Once presionado...")
            elif accion == "AUTO_CAL":
                print("🎨 Auto Calibración presionado...")
            elif accion == "CONFIG":
                print("⚙️ Configuración presionado...")
            elif accion == "EXIT":
                print("🚪 Botón SALIR presionado - Cerrando aplicación...")
                try:
                    stop_event.set()
                except Exception:
                    pass
                try:
                    yolo_running = False
                    if 'yolo_thread' in globals() and yolo_thread and yolo_thread.is_alive():
                        yolo_thread.join(timeout=1.0)
                except Exception:
                    pass
                try:
                    backend.stop()
                except Exception:
                    pass
                try:
                    backend.close()
                except Exception:
                    pass
                try:
                    cleanup_tracker_yaml()
                except Exception:
                    pass
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                return
            elif accion.startswith("GAMMA_"):
                gamma_val = float(accion.split("_")[1])
                gamma_actual = gamma_val
                print(f"📊 Gamma ajustado a: {gamma_val}")
            elif accion == "CLF_CONF_DOWN":
                CLF_CONF_THRESHOLD = max(0.5, CLF_CONF_THRESHOLD - 0.1)
                print(f"🔧 Confianza clasificador: {CLF_CONF_THRESHOLD:.1f}")
            elif accion == "CLF_CONF_UP":
                CLF_CONF_THRESHOLD = min(1.0, CLF_CONF_THRESHOLD + 0.1)
                print(f"🔧 Confianza clasificador: {CLF_CONF_THRESHOLD:.1f}")
            elif accion == "CLF_MODE_TOGGLE":
                CLF_MODE_CONSERVATIVE = not CLF_MODE_CONSERVATIVE
                mode_text = "CONSERVADOR" if CLF_MODE_CONSERVATIVE else "NORMAL"
                print(f"🔧 Modo clasificador: {mode_text}")
            # El manejo de BAYER_ se realiza solo en el bucle principal
            
            button_clicked = None


def show_info_window(cam):
    cv2.namedWindow(WIN_INFO, cv2.WINDOW_AUTOSIZE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Log en consola al abrir INFO
    try:
        vendor = safe_get(cam, 'DeviceVendorName', 'N/A')
        model = safe_get(cam, 'DeviceModelName', 'N/A')
        serial = safe_get(cam, 'DeviceSerialNumber', 'N/A')
        pxf   = safe_get(cam, 'PixelFormat', 'N/A')
        trig  = safe_get(cam, 'TriggerMode', 'N/A')
        acqm  = safe_get(cam, 'AcquisitionMode', 'N/A')
        fr    = safe_get(cam, 'AcquisitionFrameRate', 'N/A')
        et    = safe_get(cam, 'ExposureTime', 'N/A')
        gn    = safe_get(cam, 'Gain', 'N/A')
        gm    = safe_get(cam, 'Gamma', 'N/A')
        bwa   = safe_get(cam, 'BalanceWhiteAuto', safe_get(cam,'WhiteBalanceAuto','N/A'))
        # ROI/Resolución actual reportada por GenICam
        try:
            rx = safe_get(cam, 'OffsetX', 0)
            ry = safe_get(cam, 'OffsetY', 0)
            rw = safe_get(cam, 'Width', 'N/A')
            rh = safe_get(cam, 'Height', 'N/A')
        except Exception:
            rx, ry, rw, rh = 'N/A', 'N/A', 'N/A', 'N/A'
        print(f"[INFO] {vendor} {model} S/N:{serial} PF:{pxf} TRG:{trig} ACQ:{acqm} FPS:{fr} ET(us):{et} Gain(dB):{gn} Gamma:{gm} AWB:{bwa}")
        print(f"[INFO] ROI GenICam: OffsetX:{rx} OffsetY:{ry} Width:{rw} Height:{rh}")
    except Exception:
        pass

    while True:
        # Componer un canvas negro
        img = np.zeros((480, 720, 3), dtype=np.uint8)
        lines = [
            "INFORMACIÓN (solo lectura)",
            f"Vendor: {safe_get(cam, 'DeviceVendorName', 'N/A')}",
            f"Model: {safe_get(cam, 'DeviceModelName', 'N/A')}",
            f"Serial: {safe_get(cam, 'DeviceSerialNumber', 'N/A')}",
            f"PixelFormat: {safe_get(cam, 'PixelFormat', 'N/A')}",
            f"TriggerMode: {safe_get(cam, 'TriggerMode', 'N/A')}",
            f"AcquisitionMode: {safe_get(cam, 'AcquisitionMode', 'N/A')}",
            f"Acq FrameRate: {safe_get(cam, 'AcquisitionFrameRate', 'N/A')}",
            f"ExposureTime: {safe_get(cam, 'ExposureTime', 'N/A')}",
            f"Gain: {safe_get(cam, 'Gain', 'N/A')}",
            f"Gamma: {safe_get(cam, 'Gamma', 'N/A')}",
            f"BalanceWhiteAuto: {safe_get(cam, 'BalanceWhiteAuto', safe_get(cam,'WhiteBalanceAuto','N/A'))}",
            f"ROI: X:{safe_get(cam, 'OffsetX', 'N/A')} Y:{safe_get(cam, 'OffsetY', 'N/A')} W:{safe_get(cam, 'Width', 'N/A')} H:{safe_get(cam, 'Height', 'N/A')}",
            f"Frame (OpenCV): {current_img_w}x{current_img_h}",
            "",
            "ESC o ❌ para cerrar"
        ]
        
        y = 40
        cv2.putText(img, "INFORMACIÓN (solo lectura)", (20, y), font, 0.9, (0,255,0), 2); y += 30
        for ln in lines[1:]:
            cv2.putText(img, ln, (20, y), font, 0.6, (255,255,255), 2); y += 24
        
        cv2.imshow(WIN_INFO, img)
        
        # Salir si ESC o si el usuario cierra con la X:
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break
        if cv2.getWindowProperty(WIN_INFO, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    safe_destroy(WIN_INFO)


def show_config_window(cam):
    import numpy as np, time, cv2
    global YOLO_CONF
    cv2.namedWindow(WIN_CONFIG, cv2.WINDOW_AUTOSIZE)
    
    # auto-modos OFF al entrar
    for auto in ("ExposureAuto", "GainAuto"):
        safe_set(cam, auto, "Off")

    def n_limits(node):
        try:
            lo = float(getattr(node, "min", None)) if getattr(node, "min", None) is not None else None
            hi = float(getattr(node, "max", None)) if getattr(node, "max", None) is not None else None
            return lo, hi
        except:
            return None, None

    def g(name, default=None):
        return safe_get(cam, name, default)

    def s(name, val):
        return safe_set(cam, name, val)

    def clamp_node(name, val):
        n = getattr(cam, name, None)
        if n is None:
            return val
        lo, hi = n_limits(n)
        if (lo is not None) and (hi is not None) and (lo > hi):
            lo, hi = hi, lo  # por si el SDK devuelve invertido
        if lo is not None:
            val = max(lo, val)
        if hi is not None:
            val = min(hi, val)
        return val

    # 1) desactivar auto-modos al entrar y asegurar ExposureMode=Timed si existe
    for auto in ("ExposureAuto", "GainAuto"):
        if hasattr(cam, auto):
            try:
                setattr(getattr(cam, auto), "value", "Off")
            except:
                pass
    safe_set(cam, 'ExposureMode', 'Timed')

    # 2) valores y rangos - CONFIGURACIÓN OPTIMIZADA PARA CPU
    # Exposición: 3000-6000 μs para reducir motion blur
    et_lo, et_hi = 3000.0, 6000.0
    if hasattr(cam, "ExposureTime"):
        lo, hi = n_limits(cam.ExposureTime)
        if lo is not None:
            et_lo = max(et_lo, lo)
        if hi is not None:
            et_hi = min(et_hi, hi)
    et_val = np.clip(float(g("ExposureTime", 5000.0)), et_lo, et_hi)  # 5ms por defecto

    # Ganancia: 20-30 dB para compensar exposición baja
    g_lo, g_hi = 20.0, 30.0
    if hasattr(cam, "Gain"):
        lo, hi = n_limits(cam.Gain)
        if lo is not None:
            g_lo = max(g_lo, lo)
        if hi is not None:
            g_hi = min(g_hi, hi)
    g_val = np.clip(float(g("Gain", 24.0)), g_lo, g_hi)  # 24 dB por defecto

    # FrameRate: 8-10 FPS para sincronizar con YOLO (crítico para estabilidad)
    fr_lo, fr_hi = 8.0, 10.0
    fr_val = 8.0  # 8 FPS por defecto para sincronizar con YOLO
    fr_supported = hasattr(cam, "AcquisitionFrameRate")
    if fr_supported:
        lo, hi = n_limits(cam.AcquisitionFrameRate)
        cur = g("AcquisitionFrameRate", 1/12.0)
        if hi is not None and hi <= 1.0:  # nodo en segundos → UI en fps
            fr_val = np.clip(1.0/max(cur, 1e-6), fr_lo, fr_hi)
        else:
            fr_val = np.clip(cur, fr_lo, fr_hi)

    # habilitar enable antes de tocar fps (CRÍTICO para sincronización)
    if hasattr(cam, "AcquisitionFrameRateEnable"):
        try:
            cam.AcquisitionFrameRateEnable.value = True
            print("✅ AcquisitionFrameRateEnable = On (por defecto)")
        except:
            pass

    # 3) trackbars en la MISMA ventana
    H, W = 360, 720
    cv2.imshow(WIN_CONFIG, np.zeros((H, W, 3), dtype=np.uint8))
    SCALE = 1000

    def mk_trackbar_float(label, init_v, lo, hi, on_change):
        def _cb(pos):
            v = lo + (hi - lo) * (pos / SCALE) if hi > lo else lo
            on_change(v)
        cv2.createTrackbar(label, WIN_CONFIG, 0, SCALE, _cb)
        pos0 = int(round(((init_v - lo) / (hi - lo)) * SCALE)) if hi > lo else 0
        pos0 = max(0, min(SCALE, pos0))
        cv2.setTrackbarPos(label, WIN_CONFIG, pos0)

    _last_cb_t = {"expo":0.0, "gain":0.0, "fps":0.0}
    _th_ms = 100  # throttle 100 ms
    def on_expo(v):
        now = time.time()*1000.0
        if now - _last_cb_t["expo"] < _th_ms:
            return
        _last_cb_t["expo"] = now
        v = np.clip(v, et_lo, et_hi)
        s("ExposureTime", float(v))
        print(f"ExposureTime -> {v:.1f} us")

    def on_gain(v):
        now = time.time()*1000.0
        if now - _last_cb_t["gain"] < _th_ms:
            return
        _last_cb_t["gain"] = now
        v = np.clip(v, g_lo, g_hi)
        s("Gain", float(v))
        print(f"Gain -> {v:.2f} dB")

    def on_fps(v_fps):
        now = time.time()*1000.0
        if now - _last_cb_t["fps"] < _th_ms:
            return
        _last_cb_t["fps"] = now
        v_fps = np.clip(v_fps, fr_lo, fr_hi)
        if hasattr(cam, "AcquisitionFrameRateEnable"):
            try:
                cam.AcquisitionFrameRateEnable.value = True
            except:
                pass
        if hi is not None and hi <= 1.0:
            target = 1.0 / v_fps  # convertir a periodo
        else:
            target = v_fps
        s("AcquisitionFrameRate", float(target))
        print(f"FPS -> {v_fps:.1f}")

    mk_trackbar_float("ExposureTime (us)", et_val, et_lo, et_hi, on_expo)
    mk_trackbar_float("Gain (dB)", g_val, g_lo, g_hi, on_gain)
    if fr_supported:
        mk_trackbar_float("FrameRate (fps)", fr_val, fr_lo, fr_hi, on_fps)

    # Slider para confianza YOLO (rango optimizado)
    def on_yolo_conf(v):
        global YOLO_CONF
        YOLO_CONF = np.clip(v, 0.20, 0.60)  # Rango optimizado para estabilidad
        print(f"YOLO Confidence -> {YOLO_CONF:.2f}")
    mk_trackbar_float("YOLO Confidence", YOLO_CONF, 0.25, 0.35, on_yolo_conf)

    # Slider para IOU NMS
    def on_yolo_iou(v):
        global YOLO_IOU
        YOLO_IOU = np.clip(v, 0.4, 0.5)  # Rango optimizado
        print(f"YOLO IOU -> {YOLO_IOU:.2f}")
    mk_trackbar_float("YOLO IOU", YOLO_IOU, 0.4, 0.5, on_yolo_iou)

    # Guardar valores originales para permitir Cancelar (ESC)
    original_et = float(g("ExposureTime", et_val)) if g("ExposureTime", None) is not None else et_val
    original_gn = float(g("Gain", g_val)) if g("Gain", None) is not None else g_val
    original_fr = float(g("AcquisitionFrameRate", fr_val)) if fr_supported and g("AcquisitionFrameRate", None) is not None else fr_val
    original_yolo_conf = float(YOLO_CONF)

    print("[CONFIG] T=Trigger On/Off | A=ExposureAuto toggle | G=GainAuto toggle | ENTER=Aceptar | ESC=Cancelar")
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    accepted = False
    while True:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        y = 30
        cv2.putText(canvas, "CONFIGURACIÓN (editable)", (12, y), font, 0.9, (0,255,0), 2); y += 30
        cv2.putText(canvas, f"PixelFormat: {g('PixelFormat','N/A')}", (12, y), font, 0.6, (255,255,255), 2); y += 22
        cv2.putText(canvas, f"TriggerMode: {g('TriggerMode','N/A')} (T)", (12, y), font, 0.6, (255,255,255), 2); y += 22
        cv2.putText(canvas, f"ExposureAuto: {g('ExposureAuto','N/A')} (A)", (12, y), font, 0.6, (255,255,255), 2); y += 22
        cv2.putText(canvas, f"GainAuto: {g('GainAuto','N/A')} (G)", (12, y), font, 0.6, (255,255,255), 2); y += 22
        if hasattr(cam, "AcquisitionFrameRateEnable"):
            cv2.putText(canvas, f"FR Enable: {g('AcquisitionFrameRateEnable','N/A')}", (12, y), font, 0.6, (255,255,255), 2); y += 22
        cv2.putText(canvas, "ENTER=Aceptar  |  ESC=Cancelar y restaurar", (12, H-10), font, 0.6, (200,200,200), 1)
        
        # Mostrar valores actuales junto a sliders (incluye originales)
        try:
            cur_et = g('ExposureTime','N/A'); cur_gn = g('Gain','N/A'); cur_fr = g('AcquisitionFrameRate','N/A') if fr_supported else 'N/A'
            cv2.putText(canvas, f"ET(us): {cur_et} (orig: {original_et:.1f})", (12, 250), font, 0.5, (200,200,0), 1)
            cv2.putText(canvas, f"Gain(dB): {cur_gn} (orig: {original_gn:.2f})", (12, 270), font, 0.5, (200,200,0), 1)
            if fr_supported:
                cv2.putText(canvas, f"FPS: {cur_fr} (orig: {original_fr:.1f})", (12, 290), font, 0.5, (200,200,0), 1)
            cv2.putText(canvas, f"YOLO_CONF: {YOLO_CONF:.2f} (orig: {original_yolo_conf:.2f})", (12, 310), font, 0.5, (200,200,0), 1)
        except Exception:
            pass

        cv2.imshow(WIN_CONFIG, canvas)
        
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # ESC = cancelar
            accepted = False
            break
        if k in (13, 10):  # ENTER = aceptar
            accepted = True
            break
        if k in (ord('t'), ord('T')):
            cur = str(g("TriggerMode","Off")).lower()
            s("TriggerMode", "Off" if cur=="on" else "On")
        if k in (ord('a'), ord('A')):
            cur = str(g("ExposureAuto","Off")).lower()
            s("ExposureAuto", "Off" if cur in ("continuous","once") else "Continuous")
        if k in (ord('g'), ord('G')):
            cur = str(g("GainAuto","Off")).lower()
            s("GainAuto", "Off" if cur in ("continuous","once") else "Continuous")
        if cv2.getWindowProperty(WIN_CONFIG, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    safe_destroy(WIN_CONFIG)

    # Restaurar si el usuario canceló
    if not accepted:
        try:
            s("ExposureTime", float(original_et))
            s("Gain", float(original_gn))
            if fr_supported:
                s("AcquisitionFrameRate", float(original_fr))
            # Revertir YOLO_CONF
            YOLO_CONF = float(original_yolo_conf)
            print(f"↩️ Cambios cancelados. ET={original_et}us, Gain={original_gn}dB, FPS={original_fr}, YOLO_CONF={YOLO_CONF}")
        except Exception as e:
            print(f"⚠️ Error revirtiendo valores CONFIG: {e}")


def _assemble_video_async(frames_dir, out_path):
    def _run():
        try:
            cmd = ['ffmpeg','-y','-framerate','30','-i', os.path.join(frames_dir,'frame_%06d.jpg'), '-c:v','libx264','-pix_fmt','yuv420p', out_path]
            subprocess.run(cmd, check=True)
            print(f"🎬 Video creado (async): {out_path}")
        except Exception as e:
            print(f"❌ FFmpeg async falló: {e}")
    threading.Thread(target=_run, daemon=True).start()


# ============ Hilos YOLO ============
def yolo_inference_thread():
    """Hilo de inferencia YOLO + tracking usando PyTorch optimizado para CPU"""
    global next_local, yolo_running, last_boxes, last_ids, miss_cnt
    
    print("🤖 Iniciando hilo de inferencia YOLO con PyTorch...")
    
    # Inicializar profiler para YOLO
    yolo_profiler.reset()
    
    # Inicializar tracker ByteTrack
    try:
        from ultralytics.trackers import BYTETracker
        from argparse import Namespace
        
        # Crear argumentos para ByteTrack
        args = Namespace(
            track_thresh=YOLO_CONF,
            track_buffer=TRACK_BUFFER,
            match_thresh=0.8,
            mot20=False
        )
        
        tracker = BYTETracker(args, frame_rate=30)
        print("✅ ByteTrack inicializado correctamente")
    except Exception as e:
        print(f"❌ Error inicializando ByteTrack: {e}")
        tracker = None
    
    # Verificar si PyTorch está disponible
    try:
        import torch
        print("✅ PyTorch disponible")
    except ImportError:
        print("❌ PyTorch no disponible")
        return
    
    try:
        # Crear modelo PyTorch optimizado para CPU
        model = YOLOPyTorchCUDA(YOLO_WEIGHTS, YOLO_IMGSZ)
        print(f"✅ Modelo PyTorch cargado: {YOLO_WEIGHTS}")
        
        # === VERIFICACIÓN CUDA EN HILO YOLO ===
        if torch.cuda.is_available():
            print(f"🚀 HILO YOLO usando CUDA: {torch.cuda.get_device_name(0)}")
            print(f"🚀 Memoria GPU antes de YOLO: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        else:
            print("⚠️ HILO YOLO usando CPU (sin CUDA)")
        
        # Aplicar optimizaciones específicas del modelo
        if hasattr(model, 'model') and hasattr(model.model, 'eval'):
            model.model.eval()  # Modo evaluación para mejor rendimiento
            print("✅ Modelo configurado en modo evaluación")
        
        # Configurar clases del modelo
        global MODEL_NAMES, KEEP_IDX, KEEP_ALL
        MODEL_NAMES = {0: 'can', 1: 'hand', 2: 'can'}  # Clases del modelo (clase 2 también es lata)
        KEEP_IDX = {0, 1, 2}  # Permitir todas las clases detectadas
        KEEP_ALL = True
        print(f"✅ Clases del modelo: {MODEL_NAMES}")
        print(f"✅ Clases permitidas: {KEEP_IDX}")
        print(f"✅ Filtro de clases: {'TODAS' if KEEP_ALL else 'FILTRADO'}")
        
        # Configurar argumentos YOLO optimizados para máxima velocidad
        YOLO_ARGS = dict(
            imgsz=max(320, YOLO_IMGSZ),  # subir a 416 si lo pones en YAML
            conf=YOLO_CONF,  # Usar confianza del YAML
            iou=YOLO_IOU,    # Usar IOU del YAML
            agnostic_nms=True,
            verbose=False,
            max_det=10,
            half=True,   # Habilitar FP16 para mayor velocidad en Jetson
            device=0 if torch.cuda.is_available() else "cpu",
            save=False,      # No guardar resultados
            save_txt=False,  # No guardar texto
            save_conf=False, # No guardar confianza
            save_crop=False, # No guardar crops
            show=False,      # No mostrar ventanas
            plots=False      # No generar plots
        )
        
        print(f"🚀 Argumentos YOLO optimizados: {YOLO_ARGS}")
        
        # Crear UNA VEZ el YAML del tracker (ajustes más tolerantes para FPS bajo)
        tracker_cfg = dict(
            tracker_type="bytetrack",
            track_buffer=150,        # Más memoria para mantener IDs estables
            match_thresh=0.8,        # Matching algo menos estricto
            track_high_thresh=0.5,   # Mayor confianza para actualizar track
            track_low_thresh=0.20,   # No aceptar tracks muy débiles
            new_track_thresh=0.35,   # Entradas algo más fáciles
            fuse_score=True,
            mot20=False
        )
        tracker_yaml = build_tracker_yaml_from_dict(tracker_cfg)
        print(f"🎯 Tracker YAML creado una vez: {tracker_yaml}")
            
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    frame_count = 0
    frame = 0  # Contador de frame para tracking
    yolo_running = True
    
    # Inicialización segura del estado de detecciones mantenidas
    if 'last_boxes' not in globals() or last_boxes is None:
        last_boxes = None
    if 'last_ids' not in globals() or last_ids is None:
        last_ids = None
    if 'miss_cnt' not in globals() or not isinstance(miss_cnt, int):
        miss_cnt = 0
    
    # Variables para procesamiento adaptativo
    last_yolo_time = time.time()
    yolo_fps_history = deque(maxlen=10)  # Historial de FPS de YOLO
    frames_skipped = 0
    total_frames = 0
    
    # Persistencia de tracker (ByteTrack) al estilo de track_speed.py
    while yolo_running and not stop_event.is_set():
        yolo_profiler.start("frame_start")
        
        try:
            import builtins
            bgr = getattr(builtins, 'latest_frame', None)
            fid = getattr(builtins, 'latest_fid', None)
            if bgr is None:
                if frame_count % 100 == 0:  # Debug cada 100 frames
                    print(f"[YOLO] Esperando frame... latest_frame={bgr is not None}")
                yolo_profiler.end()
                time.sleep(0.002)
                continue
            ts = time.time()
            yolo_profiler.mark("dequeue_done")
        except Exception:
            yolo_profiler.end()
            continue

        
        frame_count += 1
        frame += 1  # Incrementar contador de frame para tracking
        
        # PROCESAMIENTO OPTIMIZADO: saltar frames y mantener última caja
        total_frames += 1
        
        # Procesar cada frame (simplificado para funcionar como ayer)
        # if frame_count % PROCESS_EVERY != 0:
        #     yolo_profiler.mark("frame_skipped")
        #     print(f"[YOLO] Saltando frame {frame_count} (PROCESS_EVERY={PROCESS_EVERY})")
        #     # No se procesó este frame: usar última detección si está disponible
        #     if last_boxes is not None and miss_cnt <= MISS_HOLD:
        # Código de hold comentado para simplificar
        # El sistema ahora procesa cada frame directamente
        
        # PROCESAR FRAME CON YOLO (solo cada PROCESS_EVERY frames)
        # Comentado para reducir logs y mejorar rendimiento
        # print(f"🤖 PROCESANDO FRAME {frame_count} CON YOLO...")
        yolo_profiler.mark("yolo_inference_start")
        t1 = time.time()
        
        # Usar ONNX Runtime para inferencia
        try:
            # Realizar predicción con ONNX
            yolo_profiler.mark("model_predict_start")
            # Aplicar ROI si está configurado para reducir latencia
            roi = bgr
            off_x = 0; off_y = 0
            if ROI_Y1 is not None and ROI_Y2 is not None and ROI_X1 is not None and ROI_X2 is not None:
                off_y = int(ROI_Y1); off_x = int(ROI_X1)
                roi = bgr[int(ROI_Y1):int(ROI_Y2), int(ROI_X1):int(ROI_X2)]
            print(f"[YOLO] Procesando frame {frame_count}, ROI shape: {roi.shape}")
            print(f"[YOLO] Usando confianza: {YOLO_CONF}")
            print(f"[YOLO] PROCESS_EVERY: {PROCESS_EVERY}, frame_count % PROCESS_EVERY = {frame_count % PROCESS_EVERY}")
            
            # === TIMING DETALLADO DE YOLO ===
            yolo_start_time = time.time()
            
            # === VERIFICACIÓN CUDA DURANTE INFERENCIA ===
            if frame_count % 100 == 0:  # Cada 100 frames
                if torch.cuda.is_available():
                    print(f"🚀 CUDA activo: {torch.cuda.get_device_name(0)}")
                    print(f"🚀 Memoria GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
                else:
                    print("⚠️ Usando CPU para inferencia")
            
            try:
                # === TIMING DE PREDICCIÓN YOLO ===
                predict_start = time.time()
                detections = model.predict(roi, YOLO_CONF, yolo_args=YOLO_ARGS)
                predict_time = (time.time() - predict_start) * 1000
                
                print(f"[YOLO] Raw detections: {len(detections)}")
                print(f"[YOLO] ⏱️ Tiempo predicción: {predict_time:.2f}ms")
                
                if len(detections) > 0:
                    print(f"[YOLO] ✅ DETECCIÓN ENCONTRADA!")
                    print(f"[YOLO] Primera detección: {detections[0]}")
                    # Debug adicional para ver todas las detecciones
                    for i, det in enumerate(detections):
                        print(f"[YOLO] Detección {i}: conf={det.get('confidence', 'N/A')}, class={det.get('class_id', 'N/A')}")
                else:
                    print(f"[YOLO] ❌ Sin detecciones en este frame")
            except Exception as e:
                print(f"❌ YOLO predict lanzó excepción: {e}")
                detections = []
            
            print(f"[YOLO] dets={len(detections)} conf>={YOLO_CONF}")
            yolo_profiler.mark("model_predict_done")
            
            # === TIMING DE POSTPROCESAMIENTO ===
            postprocess_start = time.time()
            yolo_profiler.mark("postprocess_start")
            
            if detections:
                xyxy = []
                confs = []
                clss = []
                
                for det in detections:
                    bbox = det['bbox']
                    # Remapear si se usó ROI
                    x1, y1, x2, y2 = bbox[0] + off_x, bbox[1] + off_y, bbox[2] + off_x, bbox[3] + off_y
                    xyxy.append([x1, y1, y2 if False else x2, y2])
                    confs.append(det['confidence'])
                    clss.append(det['class_id'])
                
                xyxy = np.array(xyxy, dtype=np.float32)
                confs = np.array(confs, dtype=np.float32)
                clss = np.array(clss, dtype=np.int32)
                
                # Aplicar NMS (cv2.dnn.NMSBoxes espera [x,y,w,h])
                yolo_profiler.mark("nms_start")
                if len(xyxy) > 0:
                    rects_xywh = []
                    for (x1, y1, x2, y2) in xyxy.tolist():
                        rects_xywh.append([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)])
                    keep_indices = cv2.dnn.NMSBoxes(
                        rects_xywh,
                        confs.tolist(),
                        YOLO_CONF,
                        YOLO_IOU
                    )
                    if len(keep_indices) > 0:
                        keep = np.asarray(keep_indices).reshape(-1)
                        xyxy = xyxy[keep]
                        confs = confs[keep]
                        clss = clss[keep]
                        # Limitar al máximo de latas simultáneas permitidas
                        MAX_SIMULTANEOUS = 4
                        if len(xyxy) > MAX_SIMULTANEOUS:
                            # Elegir las de mayor confianza
                            topk = np.argsort(-confs)[:MAX_SIMULTANEOUS]
                            xyxy = xyxy[topk]
                            confs = confs[topk]
                            clss = clss[topk]
                    else:
                        xyxy = np.empty((0, 4), np.float32)
                        confs = np.empty((0,), np.float32)
                        clss = np.empty((0,), np.int32)
            else:
                xyxy = np.array([])
                confs = np.array([])
                clss = np.array([])
            
            # === TIMING DE POSTPROCESAMIENTO ===
            postprocess_time = (time.time() - postprocess_start) * 1000
            print(f"[POSTPROCESS] ⏱️ Tiempo postprocesamiento: {postprocess_time:.2f}ms")
            
            # === DETECCIÓN BÁSICA ===
            # Asignar IDs simples
            tids = np.arange(len(xyxy), dtype=int) + 1 if len(xyxy) > 0 else np.array([])
            
            # Crear resultado real con tracking
            class TrackingResult:
                def __init__(self, boxes, confs, cls, ids):
                    self.boxes = TrackingBoxes(boxes, confs, cls, ids)
            
            class TrackingBoxes:
                def __init__(self, boxes, confs, cls, ids):
                    self.xyxy = boxes
                    self.conf = confs
                    self.cls = cls
                    self.id = ids  # IDs reales del tracking
            
            res = [TrackingResult(xyxy, confs, clss, tids)]
            
        except Exception as e:
            print(f"❌ YOLO ONNX falló: {e}")
            continue

        
        # COMENTADO: Verificación de dispositivo PyTorch ya no necesaria con ONNX
        # try:
        #     _once
        # except NameError:
        #     _once = True
        #     try:
        #         # si hay tensores en GPU, Ultralytics suele mover a CPU al empaquetar;
        #         # pero el plan de ejecución debe estar en CUDA si device=0.
        #         print("✅ track() ejecutado con device:", device)
        #     except Exception:
        #         pass


        yolo_profiler.mark("yolo_inference_done")
        print(f"🤖 YOLO COMPLETADO en {time.time() - t1:.3f}s")
        t2 = time.time()
        
        # Reporte de profiling cada 30 frames
        if frame_count % 30 == 0:
            print(yolo_profiler.get_report())
        
        # Calcular FPS de YOLO y actualizar historial
        yolo_process_time = t2 - t1
        if yolo_process_time > 0:
            yolo_fps = 1.0 / yolo_process_time
        yolo_fps_history.append(yolo_fps)
        
        # Log de rendimiento cada 30 frames
        if frame_count % 30 == 0:
            avg_yolo_fps = sum(yolo_fps_history) / len(yolo_fps_history) if yolo_fps_history else 0
            print(f"🤖 YOLO Stats: {yolo_process_time*1000:.1f}ms, {avg_yolo_fps:.1f} FPS avg, {frames_skipped}/{total_frames} frames skipped")
        
        # # Extrae predicciones (ya en coordenas del frame original)
        # xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0,4))
        # confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        # clss = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else np.zeros((0,), dtype=int)
        
        # try:
        #     tids = res.boxes.id.cpu().numpy().astype(int)
        # except:
        #     tids = np.full((xyxy.shape[0],), -1, dtype=int)
        # Asignar IDs simples secuenciales en lugar de -1
        tids = np.arange(len(xyxy), dtype=int) + 1 if len(xyxy) > 0 else np.array([], dtype=int)


        # DEBUG: Mostrar detecciones brutas
        if len(xyxy) > 0:
            print(f"🔍 DETECCIONES: {len(xyxy)} boxes, confs: {confs[:3]}, clss: {clss[:3]}")
            if KEEP_ALL:
                print(f"🔍 CLASES PERMITIDAS: {KEEP_IDX}, clases detectadas: {set(clss)}")
        
        # FILTRO SIMPLE TEMPORAL PARA DEBUG
        if len(xyxy) > 0:
            # Filtrar por confianza mínima más alta para evitar duplicados
            # Filtro mínimo acorde al YAML (más permisivo)
            try:
                conf_thresh = float(YOLO_CONF) if 'YOLO_CONF' in globals() else 0.20
            except (ValueError, TypeError):
                conf_thresh = 0.20
            keep = confs >= max(0.20, conf_thresh)
            xyxy = xyxy[keep]
            confs = confs[keep]
            clss = clss[keep]
            tids = tids[keep]
            # Fusionar detecciones muy solapadas (misma clase)
            xyxy, confs, clss = merge_overlapping_detections(xyxy, confs, clss, iou_threshold=0.7)
            print(f"🔍 FILTRO+MERGE: {len(xyxy)} detecciones tras confianza+fusión")
        
        # ✅ Sigue adelante con todo; si no hay ID aún, dibujamos sin ID estable
        has_id = tids >= 0
        # No filtramos por ID aquí; ya filtras por confianza, clase y área más abajo
        
        # === TIMING DE TRACKING ===
        tracking_start = time.time()
        
        # ACTUALIZAR ESTADO DE DETECCIÓN
        if len(xyxy) > 0:
            # Nueva detección: actualizar estado
            last_boxes, last_ids = xyxy.copy(), tids.copy()
            miss_cnt = 0
            print(f"✅ DETECCIÓN: {len(xyxy)} latas encontradas!")
        else:
            # Sin detección: incrementar contador
            miss_cnt += 1
        
        # Usar sistema de tracking inteligente para IDs estables
        local_ids = assign_stable_ids(xyxy, confs, clss, frame_count)
        local_ids = np.array(local_ids, dtype=int)
        # Sin tracker real: usar SIEMPRE el ID estable como tids de dibujo
        tids = local_ids.copy()
        
        tracking_time = (time.time() - tracking_start) * 1000
        print(f"[TRACKING] ⏱️ Tiempo tracking: {tracking_time:.2f}ms")
        # Publicar fid en builtins para priorización en UI
        try:
            import builtins
            builtins.last_fid = int(fid)
        except Exception:
            pass
        
        if len(local_ids) > 0:
            print(f"🆔 IDs asignados: {local_ids.tolist()}")
        
        # PREDICCIÓN ENTRE FRAMES PERDIDOS (E)
        # Si un track falta 1-3 frames, extrapolar con velocidad
        current_tids = set(tids.tolist())
        for rid in list(histories.keys()):
            if rid not in current_tids:
                last_seen[rid] = last_seen.get(rid, 0) + 1
                
                # Si falta 1-2 frames, intentar predicción (optimizado para objetos rápidos)
                if 1 <= last_seen[rid] <= 2:
                    predicted = predict_next(rid, last_seen[rid])
                    if predicted is not None:
                        # Agregar predicción a la lista de detecciones
                        x1, y1, w, h = predicted
                        x2, y2 = x1 + w, y1 + h
                        pred_box = np.array([x1, y1, x2, y2])
                        
                        # Agregar a las detecciones actuales
                        if len(xyxy) == 0:
                            xyxy = pred_box.reshape(1, -1)
                        else:
                            xyxy = np.vstack([xyxy, pred_box.reshape(1, -1)])
                        confs = np.append(confs, 0.5)  # Confianza media para predicción
                        clss = np.append(clss, 0)  # Clase por defecto
                        tids = np.append(tids, rid)  # ID original
                        
                        print(f"🔮 Predicción para ID {rid} (frame {last_seen[rid]})")
                
                # Limpiar si lleva más de 6 frames sin verlo
                if last_seen[rid] > 6:
                    bbox_ema.pop(rid, None)
                    histories.pop(rid, None)
                    last_seen.pop(rid, None)
                    last_conf.pop(rid, None)
            else:
                last_seen[rid] = 0  # Resetear contador si se ve
        
        # Pack a la cola de UI/Clasificador
        proc_ms = (t2 - t1) * 1000.0 if t1 is not None and t2 is not None else 0.0
        
        if PROFILE_YOLO: 
            yolo_profiler.mark("pack_done")
            yolo_rep = yolo_profiler.get_report()
            print(f"[PROFILER] ⏱️ Reporte YOLO: {yolo_rep}")
        else: 
            yolo_rep = {}
        
        tids = local_ids.copy()
        
        # === TIMING TOTAL DEL FRAME ===
        total_yolo_time = (time.time() - yolo_start_time) * 1000
        print(f"[YOLO_TOTAL] ⏱️ Tiempo total frame: {total_yolo_time:.2f}ms")
        
        # Exponer cajas/IDs para dibujo directo en UI (plan B)
        try:
            if len(xyxy) > 0:
                boxes_tmp = [
                    (int(x1), int(y1), int(x2), int(y2), int(tid))
                    for (x1, y1, x2, y2), tid in zip(xyxy.tolist(), tids.tolist())
                ]
                with lock_boxes:
                    globals()['last_boxes'] = boxes_tmp
                print(f"[YOLO_PACK] fid={fid} boxes={len(boxes_tmp)} IDs={tids.tolist()}")
            else:
                # Solo limpiar si no hay detecciones por varios frames
                with lock_boxes:
                    if 'last_boxes' in globals() and len(globals()['last_boxes']) > 0:
                        # Mantener las últimas cajas por un tiempo
                        pass
                print(f"[YOLO_PACK] fid={fid} boxes=0 (manteniendo anteriores)")
        except Exception as e:
            print(f"[YOLO_PACK] Error guardando boxes: {e}")
            with lock_boxes:
                globals()['last_boxes'] = []
        infer = dict(
            bgr=bgr,
            ts=ts,
            frame_id=fid,
            proc_ms=((time.time()-t1)*1000.0),
            xyxy=xyxy,
            tids=tids,
            lids=local_ids,
            confs=confs,
            clss=clss,
            yolo_fps=1.0/yolo_process_time if yolo_process_time > 0 else 0,
            frames_skipped=frames_skipped,
            total_frames=total_frames,
            _yolo_prof=yolo_rep
        )
        
        try:
            infer_queue.put_nowait(infer)  # sin timeout para mejor rendimiento
        except queue.Full:
            # Si la cola está llena, limpiar y volver a intentar
            try:
                infer_queue.get_nowait()  # Limpiar cola
                infer_queue.put_nowait(infer)
            except queue.Empty:
                pass
    
    print("🔄 Hilo de inferencia YOLO detenido")


def apply_yolo_overlay(img, yolo_stats=None, frame_num=None, camera=None):
    """Aplica overlays de YOLO a la imagen sin cambiar el frame actual.
    Mantiene el último resultado hasta RESULT_TTL_S para evitar parpadeo.
    """
    global last_infer, last_draw_ts
    if yolo_stats is None:
        return img
    
    # Iniciar profiling del overlay
    cv2_profiler.start("yolo_overlay")
    
    # Consumir todo lo disponible para quedarse con el más reciente
    try:
        while True:
            infer = infer_queue.get_nowait()
            last_infer = infer
            last_draw_ts = time.time()
    except queue.Empty:
        pass  # No hay datos nuevos
    
    # Usar el último resultado si corresponde al mismo frame; tolerar ±1 o TTL reciente
    infer = None
    decision = "none"
    lf_fid = None
    if last_infer:
        lf_fid = last_infer.get("frame_id")
        if lf_fid == frame_num:
            infer = last_infer
            decision = "exact"
        elif isinstance(lf_fid, int) and isinstance(frame_num, int) and 0 <= (frame_num - lf_fid) <= 1:
            infer = last_infer
            decision = "+/-1"
        elif (time.time() - last_draw_ts) < RESULT_TTL_S:
            infer = last_infer
            decision = "ttl"
    # Debug del emparejamiento
    try:
        print(f"[OVERLAY] frame_num={frame_num} last_fid={lf_fid} decision={decision} boxes={(len(last_infer['xyxy']) if last_infer else 'n/a')}")
    except Exception:
        pass
    if infer is None:
        # Sin overlay disponible todavía: mostrar indicador mínimo
        cv2.putText(img, "YOLO esperando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        return img
    
    # Dibujar SIEMPRE sobre el frame actual (no sustituir)
    xyxy = infer["xyxy"]; tids = infer["tids"]; lids = infer["lids"]
    proc_ms = infer.get("proc_ms", 0.0)
    out = img
    
    # >>> PERF HUD (mostrar siempre para debug):
    now = time.time()
    cap_ms = (now - infer.get("ts", now)) * 1000.0  # ts ya llega del hilo UI (cap)
    yolo_ms = infer.get("proc_ms", 0.0)  # ms de YOLO (aprox)
    cv2.putText(out, f"LAT:{cap_ms:.0f}ms YOLO:{yolo_ms:.0f}ms", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
    # Mostrar contador de boxes siempre, incluso si es 0
    box_count = len(xyxy) if xyxy is not None else 0
    cv2.putText(out, f"YOLO ON - boxes:{box_count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2, cv2.LINE_AA)
    # Logs de cámara con safe_get si está disponible
    try:
        exposure_us = safe_get(camera, 'ExposureTime', 'N/A') if camera else 'N/A'
        camera_fps  = safe_get(camera, 'AcquisitionFrameRate', 'N/A') if camera else 'N/A'
        cv2.putText(out, f"ET:{exposure_us}us FPS:{camera_fps}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2, cv2.LINE_AA)
    except Exception:
        pass
    # <<< PERF HUD
    
    # Nota: evitar prints por frame para no ralentizar la UI
    # Definir ROI y línea de decisión (relativa a ROI)
    try:
        import builtins
        rx, ry, rw, rh = getattr(builtins, 'roi', (0, 0, out.shape[1], out.shape[0]))
        decision_line_y = int(0.7 * rh)
        print(f"[DEBUG] ROI={(rx,ry,rw,rh)} decision_line_y={decision_line_y}")
    except Exception:
        rx, ry, rw, rh = (0, 0, out.shape[1], out.shape[0])
        decision_line_y = int(0.7 * rh)
    
    # Verificar que tenemos datos válidos
    # permitir dibujar sin IDs (se inventan IDs temporales si faltan)
    if len(xyxy) == 0:
        return img
    
        # Garantiza tamaños compatibles
    if tids is None or len(tids) != len(xyxy):
        tids = np.full((len(xyxy),), -1, dtype=int)
    if lids is None or len(lids) != len(xyxy):
        lids = np.arange(len(xyxy), dtype=int) + 1
    
    # Dibujo + trayectorias (como en track_speed.py)
    for i, (box, rid, lid) in enumerate(zip(xyxy, tids, lids)):
        # Preferir ID estable (lid) si existe; fallback al rid
        key_id = int(lid) if (lid is not None and int(lid) >= 0) else int(rid)
        # Aplicar suavizado temporal al bounding box usando ID estable
        
        if key_id < 0:
            x1, y1, x2, y2 = map(int, box)
        else:
            smoothed_box = smooth_bbox(key_id, box)
            if len(smoothed_box) != 4 or np.any(~np.isfinite(smoothed_box)):
                continue
            x1, y1, x2, y2 = map(int, smoothed_box)
        
        # Asegurar que las coordenadas sean enteros válidos
        h_img, w_img = out.shape[:2]
        x1 = max(0, min(x1, w_img - 1)); x2 = max(0, min(x2, w_img - 1))
        y1 = max(0, min(y1, h_img - 1)); y2 = max(0, min(y2, h_img - 1))
            
        # Verificar que el bounding box sea válido
        if x2 <= x1 or y2 <= y1:
            continue
        
        # color estable por ID (usar key_id)
        color_seed = key_id if key_id >= 0 else -(i+1)
        rng = np.random.RandomState(abs(color_seed) % (2**31))
        color = tuple(int(c) for c in rng.randint(0,255,3))
        
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"ID{key_id}", (x1, max(15,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Actualizar muestra para clasificación (solo para latas)
        if key_id >= 0:
            update_track_sample(key_id, img, (x1, y1, x2, y2))
            
            # Clasificación en tiempo real (cada 3 frames para mejor respuesta)
            if frame_num is None or (isinstance(frame_num, int) and frame_num % 3 == 0):
                if key_id in track_votes and track_votes[key_id]["best_roi"] is not None:
                    # Solo clasificar si hemos visto al menos 2 frames
                    frames_seen = track_votes[key_id].get("frames_seen", 0)
                    if frames_seen >= 2:
                        try:
                            class_name, confidence, probs = clf_predict_bgr(track_votes[key_id]["best_roi"])
                            print(f"🔍 Track {key_id}: {class_name} ({confidence:.2f}) - Probs: {[f'{p:.2f}' for p in probs]}")
                        except Exception as e:
                            print(f"⚠️ Error clasificando track {key_id}: {e}")
                    else:
                        # Debug: track con ROI pero no suficientes frames
                        print(f"📍 Track {key_id} tiene ROI pero solo {frames_seen} frames vistos")
            
            # Clasificación temprana si está cerca del borde derecho (zona de salida)
            if key_id in track_votes and track_votes[key_id]["best_roi"] is not None:
                image_width = getattr(builtins, 'current_img_w', None)
                if image_width:
                    near_exit = is_track_near_exit(key_id, image_width)
                    if near_exit:
                        frames_seen = track_votes[key_id].get("frames_seen", 0)
                        if frames_seen >= 2:
                            try:
                                class_name, confidence, probs = clf_predict_bgr(track_votes[key_id]["best_roi"])
                                print(f"🚪 Track {key_id} cerca de salida: {class_name} ({confidence:.2f}) - Probs: {[f'{p:.2f}' for p in probs]}")
                            except Exception as e:
                                print(f"⚠️ Error clasificando track {key_id} cerca de salida: {e}")
                        else:
                            # Log para debug: track cerca del borde pero aún no tiene suficientes frames
                            print(f"📍 Track {key_id} cerca de salida pero solo {frames_seen} frames vistos")
                    else:
                        # Debug: mostrar posición del track
                        if key_id in track_votes and track_votes[key_id].get("last_box"):
                            x1, y1, x2, y2 = track_votes[key_id]["last_box"]
                            center_x = (x1 + x2) / 2
                            print(f"📍 Track {key_id} en posición {center_x:.0f}/{image_width} (umbral: {image_width-100})")
        
        # Diagnóstico y lógica de decisión respecto a línea (coords ROI)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        cy_roi = cy - ry
        try:
            print(f"[DEBUG] T{key_id} cx={cx:.1f} cy={cy:.1f} cy_roi={cy_roi:.1f} line={decision_line_y} decision={decision}")
        except Exception:
            pass

        # Lanzar clasificación también en TTL si está estable (>=2 frames)
        if decision in ("ttl",) and key_id in track_votes:
            if track_votes[key_id].get("frames_seen", 0) >= 2 and track_votes[key_id].get("best_roi") is not None:
                try:
                    cname, cprob, probs = clf_predict_bgr(track_votes[key_id]["best_roi"])
                    print(f"🧾 Lata ID={key_id}: {cname} ({cprob:.2f}) [ttl]")
                except Exception as e:
                    print(f"⚠️ Error clasificando en ttl T{key_id}: {e}")

        # trayectoria + historial para predicción
        if key_id >= 0 and DRAW_TRAILS:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            bw, bh = (x2 - x1), (y2 - y1)
            if key_id  not in histories:
                histories[key_id ] = deque(maxlen=6)
            histories[key_id ].append((time.time(), cx, cy, bw, bh))
            
            # Dibujar trayectoria
            pts = [(int(p[1]), int(p[2])) for p in histories[key_id]]
            for j in range(1, len(pts)):
                cv2.line(out, pts[j-1], pts[j], color, 2)
        
            # velocidad
            v_ms = calculate_velocity(key_id, cx, cy, 1.0/12.0)
            cv2.putText(out, f"{v_ms:.1f}m/s", (x1, y2+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Limpiar historial de bounding boxes para tracks perdidos
    current_tracks = set(lids if (lids is not None and len(lids) == len(xyxy)) else tids)

    
    for track_id in list(bbox_ema.keys()):
        if track_id not in current_tracks:
            del bbox_ema[track_id]
            # Finalizar clasificación del track perdido con ancho de imagen
            image_width = getattr(builtins, 'current_img_w', None)
            finalize_track_class(track_id, image_width)
    # No limpiar histories aquí, se usa para predicción
    

    # Actualizar estadísticas
    if proc_ms is not None and proc_ms > 0:
        yolo_stats['fps'] = 1000.0 / proc_ms
    else:
        yolo_stats['fps'] = 0.0
    yolo_stats['tracks'] = len(set(tids))  # Contar tracks únicos
    yolo_stats['detections'] = len(xyxy)  # Contar detecciones actuales
    
    # Mostrar estadísticas de procesamiento adaptativo
    if 'frames_skipped' in infer:
        yolo_stats['frames_skipped'] = infer['frames_skipped']
        yolo_stats['total_frames'] = infer['total_frames']
        yolo_stats['skip_ratio'] = infer['frames_skipped'] / max(1, infer['total_frames'])
    
    # Debug: mostrar información cada 30 frames
    if len(xyxy) > 0:
        # Obtener tiempo de exposición actual
        try:
            if camera is not None:
                et = getattr(getattr(camera, "ExposureTime", None), "value", None)
                exposure_us = f"{et:.0f}" if et is not None else "N/A"
            else:
                exposure_us = "N/A"
        except:
            exposure_us = "N/A"
        
        # Obtener FPS de cámara actual
        try:
            if camera is not None:
                fr = getattr(getattr(camera, "AcquisitionFrameRate", None), "value", None)
                camera_fps = f"{fr:.1f}" if fr is not None else "N/A"
            else:
                camera_fps = "N/A"
        except:
            camera_fps = "N/A"
        
        skip_info = f", Skip: {yolo_stats.get('skip_ratio', 0)*100:.0f}%" if 'skip_ratio' in yolo_stats else ""
        hold_info = " [HOLD]" if infer.get('is_hold', False) else ""
        print(f"🔍 Frame {frame_num if frame_num is not None else '?'}: {len(xyxy)} detecciones, {len(set(tids))} tracks | Cámara: {camera_fps} FPS, Expo: {exposure_us}μs, Gain: {getattr(getattr(camera, 'Gain', None), 'value', 'N/A')}dB | YOLO: {proc_ms:.1f}ms{skip_info}{hold_info}")
    
    # El indicador REC ahora también se añade desde el bucle principal para que sea continuo
    cv2_profiler.end()
    return out


# ============ Main ============
def main():
    # Variables globales que ajustamos en la UI
    global panel_offset_x, current_img_w, current_img_h
    # --- Chequeo de dependencias .so del CTI ---
    import subprocess
    
    # Inicializar dispositivo unificado
    initialize_unified_device()
    
    # Aplicar optimizaciones para alta velocidad
    optimize_for_high_speed()
    enable_cuda_optimizations()
    setup_high_performance_tracking()
    
    # ahora sí, crea el stream
    # --- 1) Asegurar variables de entorno GenICam/Omron (ANTES de usar CTI) ---
    global cam, cv_code_bayer, recording_active, recording_end_time, video_writer
    global button_clicked, acquisition_running, config_visible, gamma_actual, patron_actual
    global exposure_auto, gain_auto, yolo_model, yolo_thread, yolo_running
    
    # Usa Aravis en lugar de Harvester
    import cv2  # Asegurar que cv2 esté disponible localmente
    import os   # Asegurar que os esté disponible localmente
    backend = AravisBackend(index=0, bayer_code=cv2.COLOR_BayerBG2BGR).open()
    cam_like = backend  # objeto que responde a get/set mediante helpers
    
    STAPI_ROOT = os.environ.get("STAPI_ROOT_PATH", "/opt/sentech")
    GENICAM_DIR = os.environ.get("GENICAM_GENTL64_PATH", f"{STAPI_ROOT}/lib")
    os.environ["STAPI_ROOT_PATH"] = STAPI_ROOT
    os.environ["GENICAM_GENTL64_PATH"] = GENICAM_DIR
    os.environ["LD_LIBRARY_PATH"] = f"{STAPI_ROOT}/lib:{STAPI_ROOT}/lib/GenICam:" + os.environ.get("LD_LIBRARY_PATH","")
    CTI_PATH = os.environ.get("OMRON_CTI", f"{STAPI_ROOT}/lib/libstgentl.cti")
    
    # Bloque de filtrado de dispositivo actualizado con información de depuración
    target_ip = "172.20.2.151"
    selected = None
    
    # === Selección de cámara por serial o MAC (no por IP) ===
    serial_objetivo = "25G5136"  # visto en tu print(info)
    mac_objetivo = "d4:7c:44:31:89:03"  # visto en tu print(info)
    
    # Sustituir por:
    #backend.start()
    acquisition_running = False  # como estado inicial (tu UI arranca en STOP)
    cam = cam_like  # para que show_config_window/show_info_window usen backend
    privilege = "Control"  # ficticio para mantener el mismo flujo de logs
    print(f"🔑 Cámara abierta con privilegio: {privilege}")
    
    # ...cuando vayas a setear parámetros:
    if privilege == "ReadOnly":
        print("ℹ️ Dispositivo en ReadOnly: omitiendo escrituras de nodos.")
    else:
        # aquí sí puedes tocar nodos (TriggerMode, Exposure, etc.)
        pass
    
    # 🔑 node_map de la cámara (esto te faltaba cuando te dio NameError)
    # cam = ia.remote_device.node_map
    
    # (opcional) log de IP si el TL lo expone
    try:
        if "GevCurrentIPAddress" in cam:
            ip_int = int(cam.get_node("GevCurrentIPAddress").value)
            ip_str = ".".join(str((ip_int >> (8*i)) & 0xff) for i in [3,2,1,0])
            print(f"📡 IP cámara (GenICam): {ip_str}")
    except Exception:
        pass
    
    # Si abriste en ReadOnly, NO intentes escribir nodos
    if privilege == "ReadOnly":
        print("ℹ️ Cámara abierta en ReadOnly → omito la escritura de nodos (PixelFormat, FPS, etc.).")
    else:
        print("⚙️ Config cámara:")
        safe_set(cam, 'PixelFormat', PIXEL_FORMAT)
        safe_set(cam, 'TriggerMode', 'Off')
        safe_set(cam, 'AcquisitionFrameRate', 15.0)  # Aumentado para mejor rendimiento
        safe_set(cam, 'ExposureTime', 4000.0)
        safe_set(cam, 'Gain', 28.0)
        safe_set(cam, 'BalanceWhiteAuto', 'Off')
    
    # Configurar código Bayer
    cv_code_bayer = cv2.COLOR_BayerBG2BGR
    
    # CREAR VENTANA PRINCIPAL
    window_width = 1280 + 350  # Inicial (se ajustará con el primer frame)
    window_height = 900  # Altura mínima para el panel con clasificador + tracks activos
    cv2.namedWindow(WIN_MAIN, cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow(WIN_MAIN, window_width, window_height)
    cv2.setMouseCallback(WIN_MAIN, handle_mouse_click)
    
    # CREAR PANEL DE CONTROL
    panel_control = crear_panel_control_stviewer(1280, 720)
    
    # MOSTRAR INTERFAZ INICIAL
    # Crear imagen negra para el área de la cámara (misma altura que el panel)
    panel_height = panel_control.shape[0]  # Usar la altura real del panel
    camera_area = np.full((panel_height, 1280, 3), (32, 32, 32), dtype=np.uint8)
    # Muy importante: que el offset del panel obedezca a lo que realmente dibujas
    current_img_w, current_img_h = 1280, panel_height
    panel_offset_x = current_img_w
    globals()['panel_offset_x'] = int(current_img_w)
    
    # Título en el área de la cámara
    cv2.putText(camera_area, "YOLO + GenTL", (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(camera_area, "Detección + Tracking en Tiempo Real", (350, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.putText(camera_area, "Haz clic en RUN para iniciar", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Combinar área de cámara + panel de control
    interface_completa = np.hstack([camera_area, panel_control])
    cv2.imshow(WIN_MAIN, interface_completa)
    cv2.resizeWindow(WIN_MAIN, interface_completa.shape[1], interface_completa.shape[0])
    
    print("🧪 INTERFAZ PROFESIONAL INICIADA")
    
    # === VERIFICACIÓN CUDA/TORCH EN CADA EJECUCIÓN ===
    print("\n🔍 VERIFICACIÓN DE HARDWARE:")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
            print(f"✅ Memoria CUDA: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"✅ CUDA versión: {torch.version.cuda}")
        else:
            print("❌ CUDA no disponible - usando CPU")
    except Exception as e:
        print(f"❌ Error verificando PyTorch: {e}")
    
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        print(f"✅ OpenCV CUDA: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    except Exception as e:
        print(f"❌ Error verificando OpenCV: {e}")
    
    try:
        import tensorrt
        print(f"✅ TensorRT disponible: {tensorrt.__version__}")
    except ImportError:
        print("⚠️ TensorRT no disponible (usando PyTorch)")
    except Exception as e:
        print(f"❌ Error verificando TensorRT: {e}")
    
    # === DETECCIÓN DE RDP/ESCRITORIO REMOTO ===
    print("\n🔍 DETECCIÓN DE ENTORNO:")
    try:
        import os
        if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
            print("✅ SSH detectado - usando X11 forwarding")
        elif 'DISPLAY' in os.environ:
            display = os.environ['DISPLAY']
            if display.startswith(':') and '.' in display:
                print(f"✅ X11 local detectado - DISPLAY={display}")
            else:
                print(f"⚠️ X11 remoto detectado - DISPLAY={display} (posible RDP)")
        else:
            print("❌ Sin DISPLAY - posible problema de X11")
        
        # Detectar si estamos en RDP
        if os.path.exists('/proc/version'):
            with open('/proc/version', 'r') as f:
                version = f.read().lower()
                if 'microsoft' in version or 'wsl' in version:
                    print("⚠️ WSL/Windows detectado - posible RDP")
                else:
                    print("✅ Linux nativo detectado")
    except Exception as e:
        print(f"❌ Error detectando entorno: {e}")
    
    print("🚀 OPTIMIZACIONES CRÍTICAS APLICADAS:")
    print(" ✅ Sincronización: 8 FPS por defecto para sincronizar con YOLO")
    print(" ✅ Motion blur reducido: 5ms exposición, 24dB ganancia")
    print(" ✅ Tracker robusto: histéresis + predicción 1-2 frames (OBJETOS RÁPIDOS)")
    print(" ✅ Parámetros optimizados: Conf 0.30, IOU 0.45, Track 90")
    print(" ✅ Procesamiento adaptativo: salta frames si YOLO va lento")
    print(" ✅ Optimización crítica: procesa cada frame + mantiene última caja")
    print(" ⚡ OPTIMIZADO PARA OBJETOS RÁPIDOS: TTL=50ms, Hold=2 frames, Predicción=1-2 frames")
    print("\n🎯 CONTROLES DE PROFILING:")
    print("   - Presiona 'P' para mostrar reporte de rendimiento")
    print("   - Presiona 'R' para resetear estadísticas")
    print("   - Reporte automático cada 60 frames")
    print("\n⚡ OPTIMIZADO PARA 300 LATAS/MINUTO:")
    print("   - Procesamiento cada 3 frames")
    print("   - TTL reducido a 100ms")
    print("   - Hold de 2 frames")
    print("   - Parámetros YOLO optimizados")
    print("💡 Haz clic en RUN para iniciar la cámara y YOLO")
    print("💡 Usa el panel lateral para controlar la cámara")
    print("💡 Presiona 'q' para salir")
    
    t0 = None
    f = 0
    yolo_stats = {'fps': 0, 'tracks': 0, 'detections': 0}
    
    # Estado de grabación
    recording_active = False
    recording_end_time = 0.0
    video_writer = None
    recording_frame_count = 0
    recording_frames_dir = None
    recording_out_path = None
    last_rec_log_second = -1
    
    # --- DO0: salida digital para desvío ---
    try:
        do0_init()
        threading.Thread(target=_console_listener, daemon=True).start()
        print("✅ DO0 inicializado (GPIO 446). Usa 'on' en la consola para disparar pulso.")
    except Exception as e:
        print(f"⚠️ No pude inicializar DO0: {e} (¿permiso root?)")
    try:

        while True:
            # Procesar botones
            # Dentro del while True:
            try:
                accion = evt_queue.get_nowait()
            except queue.Empty:
                accion = None

            if accion:
                button_clicked = accion
            if button_clicked == "RUN" and not acquisition_running:
                print("🚀 Iniciando adquisición y YOLO...")
                # aplicar PixelFormat pendiente antes de start
                pf = globals().pop('_pending_pixelformat', None)
                if pf:
                    safe_set(cam, 'PixelFormat', pf)
                # Aplicar ROI por altura en la cámara (una vez) antes de iniciar
                try:
                    # Leer tamaño actual vía GenICam
                    def align(v, m=8):
                        return int(v) // m * m
                    h_max = int(safe_get(cam, 'HeightMax', safe_get(cam, 'Height', 1240)))
                    w_max = int(safe_get(cam, 'WidthMax', safe_get(cam, 'Width', 1624)))
                    # Banda más alta y amplia (ajustable): ~30% a ~82%
                    y1 = align(h_max * 0.50, 8)
                    y2 = align(h_max * 0.9, 8)
                    new_h = max(align(y2 - y1, 8), 64)
                    new_w = align(w_max, 8)
                    # Setear ROI: ancho completo, solo recorte en altura
                    safe_set(cam, 'OffsetX', 0)
                    safe_set(cam, 'OffsetY', y1)
                    safe_set(cam, 'Width', new_w)
                    safe_set(cam, 'Height', new_h)
                    import builtins
                    builtins.offsetY = int(y1)
                    print(f"[ROI] Región aplicada: X=0 Y={int(y1)} W={int(new_w)} H={int(new_h)}")
                except Exception as e:
                    print(f"[ROI] No se pudo aplicar ROI: {e}")
                backend.start()
                acquisition_running = True
                print("🔄 Backend iniciado, acquisition_running = True")
                # Registrar ROI global unificada para overlay/YOLO/decisión
                try:
                    import builtins
                    builtins.roi = (0, int(y1), int(new_w), int(new_h))
                except Exception:
                    pass
                
                # Cargar clasificador antes de iniciar YOLO
                print("🔄 Llegando a cargar clasificador...")
                if not clf_load(CLF_MODEL_PATH):
                    print("❌ No se pudo cargar el clasificador ONNX, desactivando clasificación")
                    try:
                        global CLASSIFY_ENABLED, CLF_SESS
                        CLASSIFY_ENABLED = False
                        CLF_SESS = None
                    except Exception:
                        pass
                else:
                    print("✅ Clasificador ONNX listo")
                
                # Iniciar hilo YOLO
                yolo_running = True  # Establecer antes de iniciar el hilo
                yolo_thread = threading.Thread(target=yolo_inference_thread, daemon=True)
                yolo_thread.start()
                
                button_clicked = None
                t0 = time.time()  # Inicializar t0 correctamente
                f = 0
                print("✅ Cámara y YOLO iniciados - Grabando en continuo")
                
            elif button_clicked == "STOP" and acquisition_running:
                print("⏹️ Deteniendo adquisición y YOLO...")
                
                # Finalizar grabación parcial si está activa
                if recording_active:
                    print("⏹️ Finalizando grabación parcial antes de parar...")
                    recording_active = False
                    try:
                        if recording_frames_dir and recording_out_path:
                            _assemble_video_async(recording_frames_dir, recording_out_path)
                            print(f"🎬 Ensamblando video en background: {recording_out_path}")
                            print(f"🎬 Video parcial creado: {recording_out_path}")
                        else:
                            print("⚠️ No hay rutas de grabación inicializadas para compilar parcial")
                    except FileNotFoundError:
                        print("⚠️ ffmpeg no encontrado (parcial). Dejo JPGs guardados.")
                    except subprocess.CalledProcessError as e:
                        print(f"❌ FFmpeg falló (parcial): {e}")
                
                backend.stop()
                acquisition_running = False
                yolo_running = False
                
                # Vaciar colas
                with cap_queue.mutex:
                    cap_queue.queue.clear()
                with infer_queue.mutex:
                    infer_queue.queue.clear()
                
                button_clicked = None
                print("✅ Cámara y YOLO detenidos")
                
                # Cerrar writer legacy si existiera
                if video_writer is not None:
                    try:
                        video_writer.release()
                    except:
                        pass
                    video_writer = None
                    
            elif button_clicked == "STOP" and not acquisition_running:
                print("⚠️ STOP presionado pero la cámara no está corriendo")
                button_clicked = None
                
            elif button_clicked == "RECORD_60S":
                # Cerrar grabación anterior si existía
                if recording_active:
                    recording_active = False
                    print("⏹️ Grabación anterior cerrada. Iniciando nueva grabación de 60s…")
                
                # Crear carpeta de salida (evitar OneDrive)
                try:
                    out_dir = os.path.join(os.path.dirname(__file__), "Videos_YOLO")
                    os.makedirs(out_dir, exist_ok=True)
                except Exception:
                    out_dir = r"C:\\Videos_YOLO"
                    os.makedirs(out_dir, exist_ok=True)
                
                # Generar nombre único
                base_ts = time.strftime("%Y%m%d_%H%M%S")
                ms = int((time.time() * 1000) % 1000)
                base_name = f"rec_{base_ts}_{ms:03d}"
                out_path_mp4 = os.path.join(out_dir, f"{base_name}.mp4")
                idx = 1
                while os.path.exists(out_path_mp4):
                    out_path_mp4 = os.path.join(out_dir, f"{base_name}_{idx:02d}.mp4")
                    idx += 1
                
                # Configurar captura por frames PNG
                recording_active = True
                recording_end_time = time.time() + 60.0
                recording_frame_count = 0
                recording_out_path = out_path_mp4
                recording_frames_dir = os.path.join(out_dir, f"frames_{base_name}")
                os.makedirs(recording_frames_dir, exist_ok=True)
                
                print(f"🎥 Grabando 60s en: {out_path_mp4} (captura de pantalla)")
                print(f"📁 Frames temporales en: {recording_frames_dir}")
                button_clicked = None
                
            elif button_clicked == "AWB_ONCE" and acquisition_running:
                print("🔄 AWB Once ejecutándose...")
                def _awb_once():
                    try:
                        did = (safe_set(cam, 'BalanceWhiteAuto', 'Once') or safe_set(cam, 'WhiteBalanceAuto', 'Once'))
                        if did:
                            time.sleep(0.6)
                            safe_set(cam, 'BalanceWhiteAuto', 'Off'); safe_set(cam, 'WhiteBalanceAuto', 'Off')
                            print("✅ AWB Once completado")
                        else:
                            print("⚠️ No se encontró nodo AWB (BalanceWhiteAuto/WhiteBalanceAuto)")
                    except Exception as e:
                        print(f"❌ Error en AWB: {e}")
                threading.Thread(target=_awb_once, daemon=True).start()
                button_clicked = None
                
            elif button_clicked == "AUTO_CAL" and acquisition_running:
                print("🎨 Auto Calibración (Expo/Gain) ...")
                def _auto_cal():
                    try:
                        safe_set(cam, 'ExposureAuto', 'Continuous')
                        safe_set(cam, 'GainAuto', 'Continuous')
                        time.sleep(0.5)
                        safe_set(cam, 'ExposureAuto', 'Off')
                        safe_set(cam, 'GainAuto', 'Off')
                        et_final = safe_get(cam, 'ExposureTime', 'N/A')
                        gn_final = safe_get(cam, 'Gain', 'N/A')
                        print(f"✅ Auto calibración completada | ExposureTime: {et_final} us | Gain: {gn_final} dB")
                    except Exception as e:
                        print(f"❌ Error en Auto Cal: {e}")
                threading.Thread(target=_auto_cal, daemon=True).start()
                button_clicked = None
                
            elif button_clicked == "CONFIG":
                global config_open
                if not config_open:
                    config_open = True
                    try:
                        show_config_window(cam)
                    finally:
                        config_open = False
                button_clicked = None
                
            elif button_clicked == "INFO":
                global info_open
                if not info_open:
                    info_open = True
                    try:
                        show_info_window(cam)
                    finally:
                        info_open = False
                button_clicked = None
                
            elif button_clicked and button_clicked.startswith("GAMMA_"):
                gamma_val = float(button_clicked.split("_")[1])
                gamma_actual = gamma_val
                print(f"📊 Aplicando Gamma: {gamma_val}")
                try:
                    safe_set(cam, 'GammaEnable', True)
                    safe_set(cam, 'Gamma', float(gamma_val))
                    print("✅ Gamma aplicado (si el nodo existe)")
                except Exception as e:
                    print(f"❌ Error aplicando Gamma: {e}")
                button_clicked = None
                
            elif button_clicked and button_clicked.startswith("BAYER_"):
                patron = button_clicked.split("_")[1]  # BG/RG/GR/GB
                formato_bayer = f"Bayer{patron}8"
                print(f"🎨 Cambiando Bayer a {patron} → {formato_bayer}")
                try:
                    # Actualiza demosaico OpenCV SIEMPRE
                    if patron == "BG":
                        cv_code = cv2.COLOR_BayerBG2BGR
                    elif patron == "RG":
                        cv_code = cv2.COLOR_BayerRG2BGR
                    elif patron == "GR":
                        cv_code = cv2.COLOR_BayerGR2BGR
                    elif patron == "GB":
                        cv_code = cv2.COLOR_BayerGB2BGR
                    else:
                        cv_code = cv2.COLOR_BayerBG2BGR
                    cv_code_bayer = cv_code
                    try:
                        backend.bayer_code = cv_code  # asegurar demosaico en fuente
                    except Exception:
                        pass
                    patron_actual = patron

                    # Si NO estamos capturando, intentar aplicar en la cámara
                    if not acquisition_running:
                        ok = bool(safe_set(cam, 'PixelFormat', formato_bayer))
                        if ok:
                            print(f"✅ PixelFormat (HW) = {safe_get(cam, 'PixelFormat', formato_bayer)}")
                        else:
                            print("⚠️ No se pudo escribir PixelFormat en reposo (se mantiene solo demosaico SW)")
                    else:
                        print("ℹ️ Capturando: programo PixelFormat(HW) para el próximo START")
                        globals()['_pending_pixelformat'] = formato_bayer
                except Exception as e:
                    print(f"❌ Error cambiando Bayer: {e}")
                button_clicked = None
                
                
            elif button_clicked == "EXIT":
                print("🚪 Botón SALIR presionado - Cerrando aplicación...")
                try:
                    if acquisition_running:
                        backend.stop()
                        acquisition_running = False
                except:
                    pass
                
                # cerrar ventanas auxiliares si estaban abiertas
                safe_destroy(WIN_CONFIG)
                safe_destroy(WIN_INFO)
                
                # cerrar ventana principal y salir
                cv2.destroyAllWindows()
                break
                
            elif button_clicked and button_clicked not in ["RUN", "STOP", "EXIT", "AWB_ONCE", "AUTO_CAL", "CONFIG", "INFO"] and not button_clicked.startswith("GAMMA_") and not button_clicked.startswith("BAYER_"):
                print(f"⚠️ Botón no reconocido: {button_clicked}")
                button_clicked = None
            
            # === TIMING DE INTERFAZ ===
            ui_start_time = time.time()
            
            # Mostrar interfaz según estado
            if acquisition_running:
                ui_profiler.start("ui_frame_start")
                
                # Asegurar que t0 esté inicializado
                if t0 is None:
                    t0 = time.time()
                    f = 0
                
                try:
                    # Capturar frame (API nueva o antigua) - timeout reducido para mejor responsividad
                    # fetch = getattr(ia, 'fetch', None) or ia.fetch_buffer
                    try:
                        # === TIMING DE CAPTURA DE FRAME ===
                        capture_start = time.time()
                        ui_profiler.mark("frame_capture_start")
                        fb = backend.get_frame(timeout_ms=120)
                        if fb is None:
                            # sin frame esta vez; sigue el loop
                            ui_profiler.end()
                            continue
                        img, ts_cap = fb
                        capture_time = (time.time() - capture_start) * 1000
                        print(f"[CAPTURE] ⏱️ Tiempo captura: {capture_time:.2f}ms")
                        ui_profiler.mark("frame_capture_done")
                    except GenTLBusyException:
                        # El módulo está ocupado; reintentar en el siguiente ciclo sin cerrar adquisición
                        ui_profiler.end()
                        time.sleep(0.005)
                        continue
                    
                    # === TIMING DE PROCESAMIENTO DE IMAGEN ===
                    process_start = time.time()
                    
                    # Aplicar gamma en software para ver efecto inmediato (optimizado)
                    if abs(gamma_actual - 1.0) > 1e-3:
                        cv2_profiler.start("gamma_lut")
                        lut = update_gamma_lut(gamma_actual)
                        img = cv2.LUT(img, lut)
                        cv2_profiler.end()
                    ui_profiler.mark("gamma_done")
                    
                    process_time = (time.time() - process_start) * 1000
                    print(f"[PROCESS] ⏱️ Tiempo procesamiento imagen: {process_time:.2f}ms")
                    
                    # Política latest-frame: publicar el último frame disponible (sin cola)
                    if yolo_running:
                        try:
                            import builtins
                            # Asegurar BGR (demosaico si viene Bayer)
                            try:
                                pxf = (backend.pixfmt or "").upper()
                                if pxf and 'BAYER' in pxf or pxf in ("MONO8",):
                                    img_bgr = demosaic_bayer8(img)
                                else:
                                    img_bgr = img
                            except Exception:
                                img_bgr = img
                            builtins.latest_frame = img_bgr.copy()
                            if 'frame_seq' not in globals():
                                globals()['frame_seq'] = 0
                            globals()['frame_seq'] += 1
                            globals()['last_enqueued_fid'] = globals()['frame_seq']
                            builtins.latest_fid = globals()['last_enqueued_fid']
                        except Exception:
                            pass
                    ui_profiler.mark("queue_done")
                    
                    # Inicializar tamaño UI a partir del PRIMER frame real
                    if not globals().get('ui_size_initialized', False):
                        h, w = img.shape[:2]
                        current_img_w, current_img_h = w, h
                        panel_offset_x = current_img_w  # <-- clave: alinear offset con el nuevo ancho
                        globals()['panel_offset_x'] = int(current_img_w)
                        try:
                            # Re-crear panel con ese alto (sin reescalar imagen)
                            panel_control = crear_panel_control_stviewer(current_img_w, current_img_h)
                            # Ajustar ventana a (imagen + panel)
                            cv2.resizeWindow(WIN_MAIN, current_img_w + PANEL_WIDTH, current_img_h)
                            globals()['ui_size_initialized'] = True
                            print(f"[UI] Tamaño inicializado desde cámara: {current_img_w}x{current_img_h}")
                        except Exception as e:
                            print(f"⚠️ No se pudo ajustar ventana/panel al primer frame: {e}")
                            globals()['ui_size_initialized'] = True

                    # Aplicar YOLO si está activo
                    if yolo_running:
                        fid = globals().get('last_enqueued_fid', None)
                        img = apply_yolo_overlay(img, yolo_stats, fid, cam)
                    ui_profiler.mark("overlay_done")

                    # Dibujo directo (plan B) priorizando paquete nuevo sobre HOLD
                    try:
                        import builtins
                        hold_boxes = []  # opcional: conservar últimas dibujadas si deseas congelar
                        with lock_boxes:
                            boxes_new = list(globals().get('last_boxes', []))
                        fid_new = getattr(builtins, 'last_fid', None)
                        fid_prev = getattr(builtins, 'ui_last_fid', None)
                        # Simplificado: siempre usar nuevas detecciones
                        use_hold = False
                        boxes = boxes_new
                        if fid_new is not None:
                            builtins.ui_last_fid = fid_new
                        print(f"[UI_DRAW] boxes={len(boxes)} use_hold={use_hold} last_boxes={len(globals().get('last_boxes', []))}")
                        if len(boxes) > 0:
                            print(f"[UI_DRAW] ✅ DIBUJANDO {len(boxes)} CAJAS CON IDs: {[box[4] for box in boxes]}")
                        else:
                            print(f"[UI_DRAW] ❌ Sin cajas para dibujar")
                        for (x1, y1, x2, y2, tid) in boxes:
                            if tid not in color_map:
                                color_map[tid] = tuple(int(c) for c in np.random.randint(0, 255, 3))
                            color = color_map[tid]
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, f"ID {tid}", (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception:
                        pass
                    
                    # Trabajar 1:1 con el tamaño de la cámara: no reescalar aquí
                    ui_profiler.mark("no_resize_camera")
                    
                    # Dibujar REC + cuenta atrás y guardar frame si está grabando (sobre img)
                    if recording_active:
                        try:
                            secs_left = max(0, int(recording_end_time - time.time()))
                            
                            # Log cada segundo (sin saturar)
                            if secs_left != last_rec_log_second:
                                print(f"⏺️ REC {secs_left}s restantes | frames: {recording_frame_count}")
                                last_rec_log_second = secs_left
                            
                            overlay = img.copy()
                            cv2.rectangle(overlay, (10, 10), (210, 60), (0, 0, 255), -1)
                            img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
                            cv2.circle(img, (35, 35), 8, (0, 0, 255), -1)
                            cv2.putText(img, f"REC {secs_left:02d}s", (55, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
                        except Exception as e:
                            print(f"⚠️ Error dibujando REC: {e}")
                        
                        try:
                            if not recording_frames_dir or not recording_out_path:
                                print("⚠️ recording_paths no inicializados; cancelando grabación.")
                                recording_active = False
                            else:
                                # Guardar más rápido y robusto (JPG) evitando problemas de rutas en Windows
                                frame_path = os.path.join(recording_frames_dir, f"frame_{recording_frame_count:06d}.jpg")
                                ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                                if ok:
                                    try:
                                        enc.tofile(frame_path)
                                    except Exception as e:
                                        print(f"❌ Error escribiendo JPG: {e} -> {frame_path}")
                                        ok = False
                                if not ok:
                                    print(f"⚠️ Error al guardar frame: {frame_path}")
                                recording_frame_count += 1
                        except Exception as e:
                            print(f"❌ Error guardando frame: {e}")
                        
                        if time.time() >= recording_end_time:
                            recording_active = False
                            print(f"✅ Grabación completada: {recording_frame_count} frames capturados")
                            try:
                                cmd = [
                                    'ffmpeg',
                                    '-y',
                                    '-framerate', '30',
                                    '-i', os.path.join(recording_frames_dir, 'frame_%06d.jpg'),
                                    '-c:v', 'libx264',
                                    '-pix_fmt', 'yuv420p',
                                    recording_out_path
                                ]
                                print(f"🎯 Ejecutando FFmpeg: {' '.join(cmd)}")
                                subprocess.run(cmd, check=True, capture_output=True)
                                print(f"🎬 Video creado: {recording_out_path}")
                                
                                # No borrar frames: el usuario quiere conservarlos siempre
                                try:
                                    os.listdir(recording_frames_dir)
                                    print("📦 Frames temporales conservados")
                                except Exception as e:
                                    print(f"⚠️ No se pudo acceder a carpeta de frames: {e}")
                            except FileNotFoundError:
                                print("⚠️ ffmpeg no encontrado. Instálalo o añade a PATH. Dejo PNGs guardados.")
                            except subprocess.CalledProcessError as e:
                                print(f"❌ FFmpeg falló: {e}")
                                print(e.stdout.decode(errors='ignore'))
                                print(e.stderr.decode(errors='ignore'))
                            except Exception as e:
                                print(f"❌ Error creando video: {e}")
                    
                    # Actualizar panel de control
                    # Actualizar dimensiones actuales a las reales de la cámara
                    current_img_w = img.shape[1]
                    current_img_h = img.shape[0]
                    
                    # Panel de control optimizado (altura mínima para que se vean todos los botones)
                    panel_h = max(current_img_h, 720)
                    panel_actualizado = actualizar_panel_control(
                        panel_control, None, acquisition_running, current_img_w, panel_h, gamma_actual, patron_actual, yolo_stats, cam
                    )
                    
                    # Combinar imagen + panel (no reescalar img) asegurando altura suficiente para ver todos los botones
                    ui_profiler.mark("compose_start")
                    try:
                        MIN_UI_H = 720  # altura mínima para panel completo
                        h_img, w_img = img.shape[:2]
                        h_pan, w_pan = panel_actualizado.shape[:2]
                        canvas_h = max(h_img, h_pan, MIN_UI_H)
                        # Padding imagen si es más baja que canvas_h
                        if h_img < canvas_h:
                            pad = np.zeros((canvas_h - h_img, w_img, 3), dtype=img.dtype)
                            img_padded = np.vstack([img, pad])
                        else:
                            img_padded = img
                        # Padding panel si es más bajo que canvas_h
                        if h_pan < canvas_h:
                            padp = np.zeros((canvas_h - h_pan, w_pan, 3), dtype=panel_actualizado.dtype)
                            panel_padded = np.vstack([panel_actualizado, padp])
                        else:
                            panel_padded = panel_actualizado
                        interface_completa = np.hstack([img_padded, panel_padded])
                    except Exception:
                        interface_completa = np.hstack([img, panel_actualizado])
                    ui_profiler.mark("compose_done")
                    # === TIMING DE DISPLAY (RDP/REMOTO) ===
                    display_start = time.time()
                    cv2_profiler.start("imshow")
                    cv2.imshow(WIN_MAIN, interface_completa)
                    cv2_profiler.end()
                    display_time = (time.time() - display_start) * 1000
                    print(f"[DISPLAY] ⏱️ Tiempo display: {display_time:.2f}ms")
                    ui_profiler.mark("display_done")
                    
                    # Finalizar profiling del frame
                    ui_profiler.end()
                    
                    # Reporte de profiling cada 60 frames
                    if f % 60 == 0:
                        print("\n" + "="*60)
                        print("📊 REPORTE DE RENDIMIENTO COMPLETO")
                        print("="*60)
                        print(ui_profiler.get_report())
                        print(cv2_profiler.get_report())
                        print(yolo_profiler.get_report())
                        print(clf_profiler.get_report())
                        print("="*60)
                    
                    # >>> PERF AVG (comentado temporalmente para evitar errores)
                    # if not hasattr(main, "_avg_ui"):
                    #     main._avg_ui = {k:EMA(0.2) for k in ["fetch","demosaic","gamma","queue","overlay","compose","display","_total_ms"]}
                    # if f % 60 == 0:
                    #     av = {k: main._avg_ui[k].push(ui_rep.get(k,0)) for k in main._avg_ui}
                    #     print("AVG UI(60): " + " | ".join(f"{k}:{av[k]:.1f}ms" for k in ["fetch","demosaic","overlay","compose","display","_total_ms"]))
                    # <<< PERF AVG
                    
                    # manda también al CSV si lo activas (comentado temporalmente)
                    # _csv_row(time.time(), "ui", ui=ui_rep, yolo=last_infer.get("_yolo_prof") if 'last_infer' in globals() and last_infer else None)
                    # <<< PERF UI
                    
                    f += 1
                    if f % 200 == 0:  # menos logs para mejor rendimiento
                        if t0 is not None:
                            elapsed = time.time() - t0
                            fps = f / elapsed if elapsed > 0 else 0
                            print(f"📹 Cámara: {f} frames, {fps:.1f} FPS")
                        else:
                            # Si t0 es None, reinicializarlo
                            t0 = time.time()
                            f = 0
                            
                except Exception as e:
                    # 1) Si fue timeout, reintenta sin matar la UI
                    if "Timeout" in str(type(e)) or "Timeout" in str(e):
                        # reintento suave (no paramos adquisición)
                        print("⏳ Timeout de fetch: reintento suave…")
                        time.sleep(0.005)
                        continue
                    
                    # 2) Si es otro error de E/S, intenta reiniciar la adquisición una vez
                    print(f"⚠️ Error de captura, rearmando adquisición: {e}")
                    try:
                        if acquisition_running:
                            backend.stop()
                            backend.start()
                            print("🔄 Adquisición rearmada; continúo.")
                            continue
                    except Exception as e2:
                        print(f"❌ No se pudo rearmar: {e2}. Salgo del bucle de captura.")
                        acquisition_running = False
                        yolo_running = False
                        t0 = None; f = 0
                        break
            else:
                # Mostrar interfaz de inicio o estado detenido
                # Asegurar que ambas mitades tengan la MISMA altura antes de hstack
                try:
                    cam_h, cam_w = int(current_img_h), int(current_img_w)
                    camera_area = np.full((cam_h, cam_w, 3), (32, 32, 32), dtype=np.uint8)
                    if panel_control is None or panel_control.shape[0] != cam_h:
                        panel_control = crear_panel_control_stviewer(cam_w, cam_h)
                except Exception:
                    # Fallback a tamaño inicial si aún no se inicializó
                    camera_area = np.full((720, 1280, 3), (32, 32, 32), dtype=np.uint8)
                    if panel_control is None or panel_control.shape[0] != 720:
                        panel_control = crear_panel_control_stviewer(1280, 720)
                interface_completa = np.hstack([camera_area, panel_control])
                # === TIMING DE DISPLAY SIN CÁMARA ===
                display_start = time.time()
                cv2.imshow(WIN_MAIN, interface_completa)
                display_time = (time.time() - display_start) * 1000
                print(f"[DISPLAY] ⏱️ Tiempo display (sin cámara): {display_time:.2f}ms")
            
            # === TIMING TOTAL DE INTERFAZ ===
            total_ui_time = (time.time() - ui_start_time) * 1000
            print(f"[UI_TOTAL] ⏱️ Tiempo total interfaz: {total_ui_time:.2f}ms")
            
            # Manejar teclas - waitKey más corto para mejor responsividad
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('p') or key == ord('P'):
                # Mostrar reporte de rendimiento con tecla 'P'
                print("\n" + "="*60)
                print("📊 REPORTE DE RENDIMIENTO (TECLA P)")
                print("="*60)
                print(ui_profiler.get_report())
                print(cv2_profiler.get_report())
                print(yolo_profiler.get_report())
                print("="*60)
            elif key == ord('r') or key == ord('R'):
                # Resetear estadísticas con tecla 'R'
                ui_profiler.reset()
                cv2_profiler.reset()
                yolo_profiler.reset()
                print("🔄 Estadísticas de rendimiento reseteadas")
    
    except queue.Empty:
        pass

        
    # Limpiar recursos
    if acquisition_running:
        backend.stop()
        acquisition_running = False
        yolo_running = False
        stop_event.set()
        if yolo_thread:
            yolo_thread.join(timeout=1.0)
    
    # Cerrar writer si quedó abierto
    try:
        if video_writer is not None:
            video_writer.release()
        
        # Limpieza final si no se llegó a escribir
        try:
            if recording_frame_count < 5 and recording_out_path and os.path.exists(recording_out_path):
                os.remove(recording_out_path)
                print(f"🗑️ Archivo sin datos eliminado al salir: {recording_out_path}")
        except:
            pass
    except:
        pass
    
    backend.stop()
    acquisition_running = False
    backend = None
    cv2.destroyAllWindows()
    
    # Mostrar resumen final de clasificaciones
    print("\n" + "="*60)
    print("📋 RESUMEN FINAL DE CLASIFICACIÓN")
    print("="*60)
    if results_summary:
        for i, (tid, verdict) in enumerate(results_summary.items(), start=1):
            print(f"Lata {i} (ID={tid}): {verdict}")
    else:
        print("No se clasificaron latas en esta sesión")
    print("="*60)
    
    print("✅ Recursos liberados")
    
    if PROFILE_CSV:
        _csv_close()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Limpiar archivo temporal al cerrar
        cleanup_tracker_yaml()

def extract_roi_rgb(frame_bgr, x1, y1, x2, y2, size=(224,224)):
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(w-1, int(x1)))
    y1 = max(0, min(h-1, int(y1)))
    x2 = max(0, min(w,   int(x2)))
    y2 = max(0, min(h,   int(y2)))
    if x2 <= x1+1 or y2 <= y1+1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, size)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop
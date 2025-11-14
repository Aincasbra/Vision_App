"""
Settings y Context (configuración y contexto de aplicación)
------------------------------------------------------------
- Responsabilidad: Cargar configuración desde YAML (`config_model.yaml`) y variables de
  entorno (HEADLESS, AUTO_RUN, CONFIG_MODEL, LOG_*...).
- NO extrae ni procesa valores específicos: solo carga el YAML y estructura los datos.
- Los módulos especializados (model/detection, model/classifier) leen directamente
  desde `config_model.yaml` cuando necesitan valores específicos.

Funciones principales:
  * `load_settings()`: carga configuración desde YAML y variables de entorno
  * `load_yaml_config()`: helper para cargar archivo YAML con búsqueda en múltiples ubicaciones

Clases principales:
  * `Settings`: configuración estática de la aplicación (YAML + env vars)
    - `yolo`: diccionario con sección "yolo" del YAML
    - `classifier`: diccionario con sección "classifier" del YAML
  * `AppContext`: contexto de ejecución con colas y estado compartido entre módulos
    - `settings`: instancia de Settings (accesible por todos los módulos)
    - `infer_queue`, `evt_queue`: colas para comunicación entre hilos

Flujo de configuración:
  1. `app.py` llama a `load_settings()` al inicializar
  2. `load_settings()` carga `config_model.yaml` y crea `Settings`
  3. `Settings` se guarda en `AppContext.settings`
  4. Módulos especializados leen directamente desde `config_model.yaml`:
     - `model/detection/detection_service.py` lee sección "yolo"
     - `model/classifier/multiclass.py` lee sección "classifier"

Se llama desde `vision_app/app.py` en el arranque.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os
import yaml
import queue

# Variable global para rastrear si ya se mostró el log de configuración cargada
_config_loaded_logged = False


@dataclass
class Settings:
    raw_config: Dict[str, Any] = field(default_factory=dict)

    # Flags de ejecución
    headless: bool = False
    auto_run: bool = False

    # Sección YOLO
    yolo: Dict[str, Any] = field(default_factory=dict)
    
    # Sección Classifier
    classifier: Dict[str, Any] = field(default_factory=dict)


def load_yaml_config(config_path: str = "config_model.yaml", silent: bool = False):
    """Carga configuración desde archivo YAML con fallback a defaults vacíos.
    
    Busca el archivo en múltiples ubicaciones:
    1. Ruta especificada (o variable de entorno CONFIG_MODEL)
    2. vision_app/config_model.yaml (relativo al directorio actual)
    3. config_model.yaml en el directorio actual
    
    Esta función es usada por load_settings() y también puede ser llamada directamente
    por otros módulos que necesiten leer config_model.yaml.
    
    Args:
        config_path: Ruta al archivo de configuración
        silent: Si es True, no imprime logs (útil para llamadas repetidas)
    
    Returns:
        Dict con la configuración cargada o dict vacío si no se encuentra
    """
    global _config_loaded_logged
    
    # Si es una ruta absoluta, intentar directamente
    if os.path.isabs(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            # Solo mostrar log la primera vez (al inicio)
            if not silent and not _config_loaded_logged:
                print(f"✅ Configuración cargada desde {config_path}")
                _config_loaded_logged = True
            return config or {}
        except FileNotFoundError:
            pass
        except Exception as e:
            if not silent:
                print(f"❌ Error cargando {config_path}: {e}")
            return {}
    
    # Buscar en múltiples ubicaciones
    search_paths = [
        config_path,  # Ruta especificada (relativa)
        os.path.join("vision_app", "config_model.yaml"),  # vision_app/config_model.yaml
        os.path.join(os.path.dirname(__file__), "..", "config_model.yaml"),  # Relativo a este módulo
    ]
    
    # Si hay variable de entorno, usarla primero
    env_config_path = os.environ.get("CONFIG_MODEL")
    if env_config_path:
        search_paths.insert(0, env_config_path)
    
    for path in search_paths:
        try:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                with open(abs_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                # Solo mostrar log la primera vez (al inicio)
                if not silent and not _config_loaded_logged:
                    print(f"✅ Configuración cargada desde {abs_path}")
                    _config_loaded_logged = True
                return config or {}
        except Exception as e:
            continue
    
    # Si no se encuentra en ninguna ubicación
    if not silent:
        print(f"⚠️ Archivo config_model.yaml no encontrado en: {', '.join(search_paths)}, usando configuración por defecto")
    return {}


def _to_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def load_settings(config_path: str = "config_model.yaml") -> Settings:
    """Carga configuración desde YAML y variables de entorno.
    
    Esta función se llama al inicio de la aplicación y muestra el log de configuración cargada.
    """
    # Permite sobreescribir la ruta del YAML vía variable de entorno CONFIG_MODEL
    env_config_path = os.environ.get("CONFIG_MODEL")
    effective_path = env_config_path if env_config_path else config_path
    # Primera carga: mostrar log (silent=False por defecto)
    cfg = load_yaml_config(effective_path, silent=False)

    # YOLO
    yolo_cfg = dict(cfg.get("yolo", {})) if isinstance(cfg, dict) else {}
    
    # Classifier
    classifier_cfg = dict(cfg.get("classifier", {})) if isinstance(cfg, dict) else {}

    # Variables de entorno
    headless = _to_bool(os.environ.get("HEADLESS"), False)
    auto_run = _to_bool(os.environ.get("AUTO_RUN"), False)

    return Settings(
        raw_config=cfg or {},
        headless=headless,
        auto_run=auto_run,
        yolo=yolo_cfg,
        classifier=classifier_cfg,
    )


@dataclass
class AppContext:
    """Contexto de aplicación para compartir dependencias de forma explícita.
    
    Almacena estado compartido en tiempo de ejecución:
    - settings: instancia de Settings (configuración desde YAML) - ACCESIBLE POR TODOS LOS MÓDULOS
    - config: configuración raw (compatibilidad con código antiguo)
    - logger: logger de la aplicación
    - device: dispositivo CUDA/CPU
    - colas: cap_queue, infer_queue, evt_queue para comunicación entre módulos
    
    IMPORTANTE: Los módulos especializados (DetectionService, multiclass, etc.)
    leen directamente desde `config_model.yaml` usando funciones helper.
    app.py NO extrae valores del config, solo pasa el context completo.
    """
    config: Dict[str, Any] = field(default_factory=dict)
    logger: Any = None
    device: Optional[str] = None
    cap_queue: Optional[queue.Queue] = None
    infer_queue: Optional[queue.Queue] = None
    evt_queue: Optional[queue.Queue] = None
    classifier_stats: Dict[str, int] = field(default_factory=lambda: {'total': 0, 'buenas': 0, 'malas': 0})  # Estadísticas del clasificador para UI



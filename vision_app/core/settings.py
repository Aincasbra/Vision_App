"""
Settings y Context (configuración y contexto de aplicación)
------------------------------------------------------------
- Carga parámetros desde YAML (`config_yolo.yaml`) y variables de
  entorno (HEADLESS, AUTO_RUN, CONFIG_YOLO, LOG_*...).
- Gestiona configuración estática (`Settings`) y contexto de ejecución (`AppContext`).
- Funciones principales:
  * `load_settings()`: carga configuración desde YAML y variables de entorno
  * `load_yaml_config()`: helper para cargar archivo YAML con fallback a defaults
- Clases principales:
  * `Settings`: configuración estática de la aplicación (YAML + env vars)
  * `AppContext`: contexto de ejecución con colas y estado compartido entre módulos
- Se llama desde `vision_app/app.py` en el arranque.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os
import yaml
import queue


@dataclass
class Settings:
    raw_config: Dict[str, Any] = field(default_factory=dict)

    # Flags de ejecución
    headless: bool = False
    auto_run: bool = False

    # Sección YOLO
    yolo: Dict[str, Any] = field(default_factory=dict)


def load_yaml_config(config_path: str = "config_yolo.yaml"):
    """Carga configuración desde archivo YAML con fallback a defaults vacíos.
    
    Busca el archivo en múltiples ubicaciones:
    1. Ruta especificada (o variable de entorno CONFIG_YOLO)
    2. vision_app/config_yolo.yaml (relativo al directorio actual)
    3. config_yolo.yaml en el directorio actual
    """
    # Si es una ruta absoluta, intentar directamente
    if os.path.isabs(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuración cargada desde {config_path}")
            return config or {}
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"❌ Error cargando {config_path}: {e}")
            return {}
    
    # Buscar en múltiples ubicaciones
    search_paths = [
        config_path,  # Ruta especificada (relativa)
        os.path.join("vision_app", "config_yolo.yaml"),  # vision_app/config_yolo.yaml
        os.path.join(os.path.dirname(__file__), "..", "config_yolo.yaml"),  # Relativo a este módulo
    ]
    
    for path in search_paths:
        try:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                with open(abs_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"✅ Configuración cargada desde {abs_path}")
                return config or {}
        except Exception as e:
            continue
    
    # Si no se encuentra en ninguna ubicación
    print(f"⚠️ Archivo config_yolo.yaml no encontrado en: {', '.join(search_paths)}, usando configuración por defecto")
    return {}


def _to_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


def load_settings(config_path: str = "config_yolo.yaml") -> Settings:
    # Permite sobreescribir la ruta del YAML vía variable de entorno CONFIG_YOLO
    env_config_path = os.environ.get("CONFIG_YOLO")
    effective_path = env_config_path if env_config_path else config_path
    cfg = load_yaml_config(effective_path)

    # YOLO
    yolo_cfg = dict(cfg.get("yolo", {})) if isinstance(cfg, dict) else {}

    # Variables de entorno
    headless = _to_bool(os.environ.get("HEADLESS"), False)
    auto_run = _to_bool(os.environ.get("AUTO_RUN"), False)

    return Settings(
        raw_config=cfg or {},
        headless=headless,
        auto_run=auto_run,
        yolo=yolo_cfg,
    )


@dataclass
class AppContext:
    """Contexto de aplicación para compartir dependencias de forma explícita.
    
    Almacena estado compartido en tiempo de ejecución:
    - config: configuración raw
    - logger: logger de la aplicación
    - device: dispositivo CUDA/CPU
    - colas: cap_queue, infer_queue, evt_queue para comunicación entre módulos
    """
    config: Dict[str, Any] = field(default_factory=dict)
    logger: Any = None
    device: Optional[str] = None
    cap_queue: Optional[queue.Queue] = None
    infer_queue: Optional[queue.Queue] = None
    evt_queue: Optional[queue.Queue] = None



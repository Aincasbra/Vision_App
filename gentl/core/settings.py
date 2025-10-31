"""
Settings (configuración centralizada)
------------------------------------
- Carga parámetros desde YAML (`config_yolo.yaml`) y variables de
  entorno (HEADLESS, AUTO_RUN, CONFIG_YOLO, LOG_*...).
- Exponer un objeto de settings consistente al resto de
  módulos.
- Se llama desde `gentl/app.py` en el arranque.
"""
"""
Centralización de configuración: YAML + variables de entorno.
Mantiene compatibilidad exponiendo también el dict original en `raw_config`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
import os

from core.config import load_yaml_config


@dataclass
class Settings:
    raw_config: Dict[str, Any] = field(default_factory=dict)

    # Flags de ejecución
    headless: bool = False
    auto_run: bool = False

    # Sección YOLO
    yolo: Dict[str, Any] = field(default_factory=dict)


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



"""
AppContext
----------
- Contenedor de estado compartido (settings, colas, device, logger).
- Pasa contexto entre módulos sin acoplamiento fuerte.
- Se usa instanciado en `gentl/app.py` y consumido por UI/YOLO.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import queue


@dataclass
class AppContext:
    """Contexto de aplicación para compartir dependencias de forma explícita."""
    config: Dict[str, Any] = field(default_factory=dict)
    logger: Any = None
    device: Optional[str] = None
    cap_queue: Optional[queue.Queue] = None
    infer_queue: Optional[queue.Queue] = None
    evt_queue: Optional[queue.Queue] = None



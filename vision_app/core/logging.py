"""
Logging multi-dominio
---------------------
- Sistema de logging con múltiples categorías para trazabilidad granular.
- Funcionalidades principales:
  * `get_logger()`: obtiene logger por categoría (system, vision, images, io, timings)
  * Soporte para múltiples handlers:
    - SysLogHandler: salida a journald/syslog (activado con LOG_TO_SYSLOG=1)
    - RotatingFileHandler: archivos rotativos por categoría (activado con LOG_TO_FILE=1)
  * Configuración por variables de entorno:
    - LOG_TO_SYSLOG: activa logging a syslog
    - LOG_TO_FILE: activa logging a archivos
    - LOG_DIR: directorio para archivos de log (default: /var/log/vision_app)
    - LOG_LEVEL: nivel de logging (default: INFO)
- Trazabilidad en headless y en desarrollo, sin duplicados.
- Llamado desde:
  * `vision_app/app.py`: inicializa loggers al arrancar la aplicación
  * Todos los módulos: usan `get_logger()` o funciones helper (`log_info`, `log_warning`, etc.)
"""
import logging
import os
from logging.handlers import SysLogHandler, RotatingFileHandler
from typing import Optional


_INITIALIZED = {}


def _bool_env(var: str, default: bool = False) -> bool:
    v = os.environ.get(var)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Obtiene un logger por categoría, con Syslog y archivo opcionales.

    Categorías esperadas: 'system', 'io', 'images', 'vision', 'timings'.
    Config por entorno:
      - LOG_TO_SYSLOG=1 activa SysLogHandler(/dev/log)
      - LOG_TO_FILE=1 activa RotatingFileHandler en LOG_DIR (por categoría)
      - LOG_DIR=/var/log/vision_app (por defecto)
      - LOG_LEVEL=INFO (por defecto)
    """
    logger = logging.getLogger(name)
    
    # Si ya está inicializado, retornar sin agregar handlers
    if _INITIALIZED.get(name):
        logger.propagate = False
        return logger

    # Nivel por defecto
    level_name = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    try:
        logger.setLevel(getattr(logging, level_name, logging.INFO))
    except Exception:
        logger.setLevel(logging.INFO)

    # Evitar duplicados: limpiar handlers anteriores y no propagar al root
    logger.handlers.clear()
    logger.propagate = False

    # Formato
    line_fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

    # Solo stdout (systemd/journald captura stdout automáticamente)
    # No agregar SysLogHandler si LOG_TO_SYSLOG=0 o si systemd ya captura stdout
    # En modo systemd, stdout ya va a journald, así que SysLogHandler duplicaría
    if not _bool_env("LOG_TO_SYSLOG", False):
        # Solo stdout (capturado por systemd/journald)
        sh = logging.StreamHandler()
        sh.setFormatter(line_fmt)
        logger.addHandler(sh)
    else:
        # Si LOG_TO_SYSLOG=1 explícitamente, usar syslog directamente
        try:
            sl = SysLogHandler(address="/dev/log")
            sl.setFormatter(line_fmt)
            logger.addHandler(sl)
        except Exception:
            # Fallback a stdout si syslog falla
            sh = logging.StreamHandler()
            sh.setFormatter(line_fmt)
            logger.addHandler(sh)

    # Archivo opcional por categoría
    if _bool_env("LOG_TO_FILE", False):
        base_dir = os.environ.get("LOG_DIR", "/var/log/vision_app")
        sub_dir = os.path.join(base_dir, name)
        _ensure_dir(sub_dir)
        file_path = os.path.join(sub_dir, f"{name}.log")
        try:
            fh = RotatingFileHandler(file_path, maxBytes=5 * 1024 * 1024, backupCount=5)
            fh.setFormatter(line_fmt)
            logger.addHandler(fh)
        except Exception:
            pass

    _INITIALIZED[name] = True
    return logger


def log_info(message: str, logger_name: str = "system"):
    """Log de nivel INFO."""
    logger = get_logger(logger_name)
    logger.info(message)


def log_warning(message: str, logger_name: str = "system"):
    """Log de nivel WARNING."""
    logger = get_logger(logger_name)
    logger.warning(message)


def log_error(message: str, logger_name: str = "system"):
    """Log de nivel ERROR."""
    logger = get_logger(logger_name)
    logger.error(message)


def log_debug(message: str, logger_name: str = "system"):
    """Log de nivel DEBUG."""
    logger = get_logger(logger_name)
    logger.debug(message)

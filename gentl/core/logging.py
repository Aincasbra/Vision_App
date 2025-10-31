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

    Categorías esperadas: 'system', 'io', 'images', 'vision', 'gentl'.
    Config por entorno:
      - LOG_TO_SYSLOG=1 activa SysLogHandler(/dev/log)
      - LOG_TO_FILE=1 activa RotatingFileHandler en LOG_DIR (por categoría)
      - LOG_DIR=/var/log/calippo (por defecto)
      - LOG_LEVEL=INFO (por defecto)
    """
    logger = logging.getLogger(name)
    if _INITIALIZED.get(name):
        # Asegurar que no propaga al root para evitar duplicados
        logger.propagate = False
        return logger

    # Nivel por defecto
    level_name = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    try:
        logger.setLevel(getattr(logging, level_name, logging.INFO))
    except Exception:
        logger.setLevel(logging.INFO)

    # Evitar duplicados: limpiar handlers anteriores y no propagar al root
    try:
        logger.handlers.clear()
    except Exception:
        logger.handlers = []
    logger.propagate = False

    # Formatos
    line_fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')

    # Siempre stdout (capturado por systemd/journald)
    sh = logging.StreamHandler()
    sh.setFormatter(line_fmt)
    logger.addHandler(sh)

    # Syslog opcional
    if _bool_env("LOG_TO_SYSLOG", True):
        try:
            sl = SysLogHandler(address="/dev/log")
            sl.setFormatter(line_fmt)
            logger.addHandler(sl)
        except Exception:
            pass

    # Archivo opcional por categoría
    if _bool_env("LOG_TO_FILE", False):
        base_dir = os.environ.get("LOG_DIR", "/var/log/calippo")
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


def log_info(message: str, logger_name: str = "gentl"):
    """Log de nivel INFO."""
    logger = get_logger(logger_name)
    logger.info(message)


def log_warning(message: str, logger_name: str = "gentl"):
    """Log de nivel WARNING."""
    logger = get_logger(logger_name)
    logger.warning(message)


def log_error(message: str, logger_name: str = "gentl"):
    """Log de nivel ERROR."""
    logger = get_logger(logger_name)
    logger.error(message)


def log_debug(message: str, logger_name: str = "gentl"):
    """Log de nivel DEBUG."""
    logger = get_logger(logger_name)
    logger.debug(message)

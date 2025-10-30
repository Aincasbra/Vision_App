import logging


def get_logger(name: str) -> logging.Logger:
    """Obtiene un logger configurado para el módulo dado."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
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

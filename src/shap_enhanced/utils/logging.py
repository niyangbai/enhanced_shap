"""Logging utility for SHAP Enhanced."""

import logging

def get_logger(name: str) -> logging.Logger:
    """Initialize and return a logger with standard formatting.

    :param str name: Logger name.
    :return logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

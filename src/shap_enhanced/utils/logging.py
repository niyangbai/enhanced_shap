import logging
from typing import Optional

def get_logger(
    name: str,
    log_to_console: bool = True,
    log_to_file: Optional[str] = None,
    level: int = logging.DEBUG
) -> logging.Logger:
    """
    Initialize and return a logger with configurable handlers.

    :param str name: Logger name.
    :param bool log_to_console: If True, log to console.
    :param Optional[str] log_to_file: If set, log to this file path.
    :param int level: Logging level.
    :return logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Remove all handlers to avoid duplicate logs
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    if log_to_file:
        file_handler = logging.FileHandler(log_to_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if not log_to_console and not log_to_file:
        logger.addHandler(logging.NullHandler())
    return logger

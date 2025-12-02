# src/utils/logger.py
"""
Logger Module
Logging utilities
"""
import logging
import sys


def setup_logger(name: str = "uit_ds310", level: str = "INFO"):
    """
    Setup logger for the project.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

if __name__ == "__main__":
    log = setup_logger(level="DEBUG")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")
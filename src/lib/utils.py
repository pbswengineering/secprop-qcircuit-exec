import logging


def create_logger(name: str, log_level: int, file: str) -> logging.Logger:
    """Create a console and file logger."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        while logger.handlers:
            handler = logger.handlers[0]
            handler.close()
            logger.removeHandler(handler)
    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(file)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
    logger.addHandler(sh)
    return logger

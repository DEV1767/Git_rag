import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "git_rag"
LOG_LEVEL = os.getenv("GIT_RAG_LOG_LEVEL", "DEBUG").upper()
LOG_FILE = (
    Path(__file__).resolve().parent.parent
    / "logs"
    / "git_rag.log"
)


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the application logger configured to write detailed records."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, LOG_LEVEL, None)
    if not isinstance(level, int):
        raise ValueError(
            "GIT_RAG_LOG_LEVEL must be a valid logging level, "
            f"got {LOG_LEVEL!r}"
        )

    logger.setLevel(level)
    logger.propagate = False

    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
            "process=%(process)d | thread=%(threadName)s | "
            "%(pathname)s:%(lineno)d | %(funcName)s() | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)

    return logger


logger = get_logger()

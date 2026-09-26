"""Minimal project logging configuration."""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with one project-owned stream handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not any(getattr(handler, "_thesis_fedrec", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        handler._thesis_fedrec = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    logger.propagate = False
    return logger

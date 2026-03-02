"""
Centralized logging configuration for Video Search AI.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")
"""

import logging
import os
from datetime import datetime

LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create a unique log file per run (or use a fixed name)
LOG_FILE = os.path.join(LOG_DIR, "video_search_ai.log")

_configured = False


def setup_logging():
    global _configured
    if _configured:
        return
    _configured = True

    root_logger = logging.getLogger("video_search_ai")
    root_logger.setLevel(logging.DEBUG)

    # ── File handler (detailed) ───────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_fmt)

    # ── Console handler (concise) ─────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "%(levelname)-8s | %(name)s | %(message)s"
    )
    console_handler.setFormatter(console_fmt)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'video_search_ai' namespace."""
    setup_logging()
    return logging.getLogger(f"video_search_ai.{name}")

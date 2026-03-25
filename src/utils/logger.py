"""Logging utility for the LLM-Judge PKI study."""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "llm_judge_pki",
    level: str = "INFO",
    log_dir: str = "logs",
    log_file: str = "experiment.log",
) -> logging.Logger:
    """Configure and return a logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

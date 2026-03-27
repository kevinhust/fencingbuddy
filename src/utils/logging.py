"""
FencerAI Logging Configuration
==============================
Version: 1.0 | Last Updated: 2026-03-27

Loguru-based logging with temporal annotations and JSON output option.
"""

from __future__ import annotations

import sys
from enum import Enum
from loguru import logger as _logger
from typing import Optional, Union


# =============================================================================
# Log Levels
# =============================================================================

class LogLevel(Enum):
    """Standard log levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# =============================================================================
# Logger Configuration
# =============================================================================

_LOGGER_CONFIGURED = False


def configure_logging(
    level: str = "INFO",
    sink: Optional[Union[str, sys.stdout, sys.stderr]] = None,
    json_output: bool = False,
    rotation: Optional[str] = None,
    retention: Optional[str] = None,
) -> None:
    """
    Configure the loguru logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        sink: Output sink (file path, stdout, stderr, or StringIO)
        json_output: Enable JSON structured output
        rotation: Log rotation setting (e.g., "100 MB")
        retention: Log retention setting (e.g., "7 days")
    """
    global _LOGGER_CONFIGURED

    # Remove default handler
    _logger.remove()

    # Determine sink
    if sink is None:
        sink = sys.stderr

    # Format string with temporal annotations
    if json_output:
        format_string = "{level}: {message}"
    else:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "{message}"
        )

    # Build handler options
    options = {
        "sink": sink,
        "level": level.upper(),
        "format": format_string,
        "colorize": not json_output,
    }

    if rotation:
        options["rotation"] = rotation
    if retention:
        options["retention"] = retention

    _logger.add(**options)
    _LOGGER_CONFIGURED = True


def setup_logger() -> "loguru.Logger":
    """
    Set up and return the FencerAI logger.

    Returns:
        Configured loguru logger instance
    """
    if not _LOGGER_CONFIGURED:
        configure_logging(level="INFO")
    return _logger


# Initialize logger with default configuration
setup_logger()

# Export logger as module-level name for convenient access
logger = _logger

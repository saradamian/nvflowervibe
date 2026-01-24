"""
Logging utilities for SFL.

Provides consistent, configurable logging across the entire project.
Supports multiple output formats: rich (colorful), simple, and JSON.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from sfl.types import LoggingConfig

# Module-level logger cache
_loggers: dict = {}
_initialized: bool = False


class SimpleFormatter(logging.Formatter):
    """Simple log formatter with level-based formatting."""
    
    FORMATS = {
        logging.DEBUG: "DEBUG: %(name)s - %(message)s",
        logging.INFO: "INFO: %(name)s - %(message)s",
        logging.WARNING: "WARNING: %(name)s - %(message)s",
        logging.ERROR: "ERROR: %(name)s - %(message)s",
        logging.CRITICAL: "CRITICAL: %(name)s - %(message)s",
    }
    
    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.FORMATS[logging.INFO])
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime
        
        log_dict = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_dict)


def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """Initialize logging with the specified configuration.
    
    Args:
        config: Logging configuration. Uses defaults if not provided.
    
    Example:
        >>> from sfl.types import LoggingConfig
        >>> setup_logging(LoggingConfig(level="DEBUG", format="simple"))
    """
    global _initialized
    
    if config is None:
        config = LoggingConfig()
    
    # Get root logger for SFL
    root_logger = logging.getLogger("sfl")
    root_logger.setLevel(getattr(logging, config.level.upper(), logging.INFO))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(root_logger.level)
    
    # Set formatter based on config
    if config.format == "json":
        console_handler.setFormatter(JSONFormatter())
    elif config.format == "simple":
        console_handler.setFormatter(SimpleFormatter())
    else:
        # Rich format (default)
        try:
            from rich.logging import RichHandler
            console_handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
            )
        except ImportError:
            # Fall back to simple if rich not available
            console_handler.setFormatter(SimpleFormatter())
    
    root_logger.addHandler(console_handler)
    
    # Add file handler if configured
    if config.file_enabled:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(root_logger.level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        root_logger.addHandler(file_handler)
    
    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the specified module.
    
    Args:
        name: Logger name (typically __name__ of the calling module).
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting federated learning")
    """
    global _initialized, _loggers
    
    # Ensure logging is initialized
    if not _initialized:
        setup_logging()
    
    # Prefix with sfl if not already
    if not name.startswith("sfl"):
        name = f"sfl.{name}"
    
    # Return cached logger if available
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    _loggers[name] = logger
    
    return logger

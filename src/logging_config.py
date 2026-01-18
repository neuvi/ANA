"""Logging configuration for ANA.

Provides centralized logging setup with Rich formatting for better debugging.
"""

import logging
import sys
from pathlib import Path
from typing import Literal

from rich.logging import RichHandler


def setup_logging(
    level: str | int = "INFO",
    log_file: Path | str | None = None,
    rich_tracebacks: bool = True,
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        rich_tracebacks: Whether to use Rich for tracebacks
        
    Returns:
        Configured root logger
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Create handlers
    handlers: list[logging.Handler] = []
    
    # Rich console handler
    console_handler = RichHandler(
        level=level,
        rich_tracebacks=rich_tracebacks,
        tracebacks_show_locals=True,
        show_time=True,
        show_path=True,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
            )
        )
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
        force=True,  # Override any existing configuration
    )
    
    # Get ANA logger
    logger = logging.getLogger("ana")
    logger.setLevel(level)
    
    return logger


def get_logger(name: str = "ana") -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name (will be prefixed with 'ana.')
        
    Returns:
        Logger instance
    """
    if not name.startswith("ana"):
        name = f"ana.{name}"
    return logging.getLogger(name)


# Pre-configured loggers for different modules
class ModuleLoggers:
    """Pre-configured loggers for ANA modules."""
    
    @property
    def agent(self) -> logging.Logger:
        return get_logger("agent")
    
    @property
    def graph(self) -> logging.Logger:
        return get_logger("graph")
    
    @property
    def link_analyzer(self) -> logging.Logger:
        return get_logger("link_analyzer")
    
    @property
    def embedding(self) -> logging.Logger:
        return get_logger("embedding")
    
    @property
    def vault(self) -> logging.Logger:
        return get_logger("vault")
    
    @property
    def cli(self) -> logging.Logger:
        return get_logger("cli")
    
    @property
    def api(self) -> logging.Logger:
        return get_logger("api")


# Singleton instance
loggers = ModuleLoggers()

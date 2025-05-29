"""Logging configuration for the Leo RAG Assistant."""
import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from leo_assist.utils.settings import Settings

class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward loguru."""
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def configure_logging(settings: Settings):
    """Configure logging for the application."""
    log_level = "DEBUG" if settings.debug else "INFO"
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    
    # Configure loguru
    logger.remove()  # Remove default handler
    
    # Add console handler
    logger.add(
        sys.stderr,
        level=log_level,
        format=log_format,
        colorize=True,
        backtrace=True,
        diagnose=settings.debug
    )
    
    # Add file handler if in production
    if not settings.debug:
        log_file = settings.data_dir / "leo_rag_assistant.log"
        logger.add(
            log_file,
            rotation="100 MB",
            retention="30 days",
            level=log_level,
            format=log_format,
            backtrace=True,
            diagnose=False
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Disable noisy loggers
    for logger_name in ["urllib3", "chromadb", "fsevents"]:
        logging.getLogger(logger_name).setLevel("WARNING")
    
    logger.info(f"Logging configured with level: {log_level}")
    return logger

# Initialize logger
settings = Settings()
logger = configure_logging(settings)

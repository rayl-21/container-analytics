"""
Logging configuration for the Container Analytics application.

Uses loguru for structured logging with file rotation, different log levels
for development/production, and standardized log formatting across the application.

Example:
    >>> from utils.logging_config import setup_logging, get_logger
    >>> 
    >>> # Setup logging (call once at application startup)
    >>> setup_logging()
    >>> 
    >>> # Get logger for specific module
    >>> logger = get_logger("modules.detection.yolo_detector")
    >>> logger.info("Starting object detection", model="yolov8x", confidence=0.5)
    >>> 
    >>> # Log with structured data
    >>> logger.error("Detection failed", 
    >>>               image_path="image.jpg", 
    >>>               error="Model not loaded",
    >>>               duration_ms=1234)
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    # Fallback to standard logging if loguru is not available
    import logging
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False

try:
    from .config import settings
except ImportError:
    # Basic fallback settings if config is not available
    class BasicSettings:
        log_level = "INFO"
        log_file = "app.log"
        log_max_bytes = 10485760
        log_backup_count = 5
        debug_mode = False
        def is_development(self):
            return False
    settings = BasicSettings()


# Global logger configuration state
_logging_configured = False


class InterceptHandler:
    """Intercept standard logging records and route to loguru."""
    
    def emit(self, record):
        """Emit a logging record to loguru."""
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == __file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    enable_json_logging: bool = False,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
) -> None:
    """
    Configure loguru logging for the entire application.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (defaults to settings.log_file)
        enable_file_logging: Enable logging to file
        enable_console_logging: Enable logging to console
        enable_json_logging: Use JSON format for structured logging
        max_bytes: Maximum log file size in bytes (defaults to settings.log_max_bytes)
        backup_count: Number of backup files to keep (defaults to settings.log_backup_count)
    """
    global _logging_configured
    
    if _logging_configured:
        return
    
    # Use settings defaults if not provided
    log_level = log_level or settings.log_level
    log_file = log_file or settings.log_file
    max_bytes = max_bytes or settings.log_max_bytes
    backup_count = backup_count or settings.log_backup_count
    
    # Remove default handler
    logger.remove()
    
    # Determine format based on environment
    if settings.is_development() or not enable_json_logging:
        # Human-readable format for development
        console_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message} | "
            "{extra}"
        )
    else:
        # JSON format for production
        console_format = _json_format
        file_format = _json_format
    
    # Configure console logging
    if enable_console_logging:
        logger.add(
            sys.stderr,
            format=console_format,
            level=log_level,
            colorize=True,
            backtrace=settings.debug_mode,
            diagnose=settings.debug_mode,
            enqueue=True,  # Thread-safe logging
        )
    
    # Configure file logging
    if enable_file_logging and log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            str(log_path),
            format=file_format,
            level=log_level,
            rotation=max_bytes,
            retention=backup_count,
            compression="gz",  # Compress rotated logs
            backtrace=settings.debug_mode,
            diagnose=settings.debug_mode,
            enqueue=True,
        )
    
    # Configure error file logging (separate file for errors)
    if enable_file_logging and log_file:
        error_log_path = log_path.parent / f"{log_path.stem}_errors.log"
        logger.add(
            str(error_log_path),
            format=file_format,
            level="ERROR",
            rotation=max_bytes,
            retention=backup_count,
            compression="gz",
            backtrace=True,
            diagnose=True,
            enqueue=True,
        )
    
    # Intercept standard library logging
    import logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Suppress specific noisy loggers
    _suppress_noisy_loggers()
    
    # Log initial setup message
    logger.info(
        "Logging configured",
        level=log_level,
        file_logging=enable_file_logging,
        console_logging=enable_console_logging,
        json_format=enable_json_logging,
        log_file=str(log_file) if log_file else None,
    )
    
    _logging_configured = True


def _json_format(record: Dict[str, Any]) -> str:
    """Format log record as JSON."""
    import json
    from datetime import datetime
    
    # Build structured log entry
    log_entry = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    
    # Add extra fields
    if record.get("extra"):
        log_entry.update(record["extra"])
    
    # Add exception info if present
    if record.get("exception"):
        log_entry["exception"] = record["exception"]
    
    return json.dumps(log_entry, default=str) + "\n"


def _suppress_noisy_loggers():
    """Suppress logs from noisy third-party libraries."""
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3.connectionpool",
        "selenium.webdriver.remote.remote_connection",
        "PIL.PngImagePlugin",
        "matplotlib",
        "asyncio",
    ]
    
    import logging
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None):
    """
    Get a logger instance for the specified module/component.
    
    Args:
        name: Logger name (typically __name__ or module path)
        
    Returns:
        Configured logger instance
    """
    if LOGURU_AVAILABLE and hasattr(logger, 'bind'):
        if name:
            return logger.bind(name=name)
        return logger
    else:
        # Fallback to standard logging
        import logging
        if name:
            return logging.getLogger(name)
        return logging.getLogger(__name__)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log a function call with parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger.debug(
        "Function called",
        function=func_name,
        parameters=kwargs,
    )


def log_performance(func_name: str, duration_ms: float, **kwargs) -> None:
    """
    Log performance metrics for a function.
    
    Args:
        func_name: Name of the function
        duration_ms: Execution time in milliseconds
        **kwargs: Additional performance metrics
    """
    logger.info(
        "Performance metric",
        function=func_name,
        duration_ms=duration_ms,
        **kwargs,
    )


def log_detection_result(
    image_path: str,
    detections_count: int,
    confidence_scores: list,
    processing_time_ms: float,
    model_name: str = "yolov8x",
) -> None:
    """
    Log object detection results with structured data.
    
    Args:
        image_path: Path to processed image
        detections_count: Number of objects detected
        confidence_scores: List of confidence scores
        processing_time_ms: Processing time in milliseconds
        model_name: Name of the detection model used
    """
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
    
    logger.info(
        "Object detection completed",
        image_path=image_path,
        model=model_name,
        detections_count=detections_count,
        avg_confidence=round(avg_confidence, 3),
        max_confidence=round(max(confidence_scores), 3) if confidence_scores else 0,
        processing_time_ms=round(processing_time_ms, 2),
    )


def log_download_result(
    url: str,
    success: bool,
    file_path: Optional[str] = None,
    file_size_bytes: Optional[int] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log image download results.
    
    Args:
        url: URL that was downloaded
        success: Whether download was successful
        file_path: Path where file was saved (if successful)
        file_size_bytes: Size of downloaded file
        duration_ms: Download duration in milliseconds
        error: Error message if download failed
    """
    log_data = {
        "operation": "image_download",
        "url": url,
        "success": success,
    }
    
    if file_path:
        log_data["file_path"] = file_path
    if file_size_bytes:
        log_data["file_size_bytes"] = file_size_bytes
    if duration_ms:
        log_data["duration_ms"] = round(duration_ms, 2)
    if error:
        log_data["error"] = error
    
    if success:
        logger.info("Image download successful", **log_data)
    else:
        logger.error("Image download failed", **log_data)


def log_database_operation(
    operation: str,
    table: str,
    success: bool,
    rows_affected: Optional[int] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Log database operations.
    
    Args:
        operation: Type of database operation (INSERT, UPDATE, DELETE, SELECT)
        table: Database table name
        success: Whether operation was successful
        rows_affected: Number of rows affected
        duration_ms: Operation duration in milliseconds
        error: Error message if operation failed
    """
    log_data = {
        "operation": f"db_{operation.lower()}",
        "table": table,
        "success": success,
    }
    
    if rows_affected is not None:
        log_data["rows_affected"] = rows_affected
    if duration_ms:
        log_data["duration_ms"] = round(duration_ms, 2)
    if error:
        log_data["error"] = error
    
    if success:
        logger.info("Database operation completed", **log_data)
    else:
        logger.error("Database operation failed", **log_data)


def log_cache_operation(
    operation: str,
    key: str,
    hit: Optional[bool] = None,
    ttl_seconds: Optional[int] = None,
    size_bytes: Optional[int] = None,
) -> None:
    """
    Log cache operations.
    
    Args:
        operation: Cache operation (GET, SET, DELETE, INVALIDATE)
        key: Cache key
        hit: Whether it was a cache hit (for GET operations)
        ttl_seconds: TTL for cached item (for SET operations)
        size_bytes: Size of cached data
    """
    log_data = {
        "operation": f"cache_{operation.lower()}",
        "key": key,
    }
    
    if hit is not None:
        log_data["hit"] = hit
    if ttl_seconds:
        log_data["ttl_seconds"] = ttl_seconds
    if size_bytes:
        log_data["size_bytes"] = size_bytes
    
    logger.debug("Cache operation", **log_data)


# Convenience function aliases
info = logger.info
debug = logger.debug
warning = logger.warning
error = logger.error
critical = logger.critical
exception = logger.exception
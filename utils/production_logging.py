#!/usr/bin/env python3
"""
Enhanced logging configuration for Container Analytics with centralized logging,
rotation, and alert capabilities.
"""

import os
import sys
import json
import logging
import smtplib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from loguru import logger
import structlog


class ProductionLoggingConfig:
    """Production-ready logging configuration with multiple handlers and alerting."""
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: Path = Path("/logs"),
                 enable_alerts: bool = True,
                 alert_email: Optional[str] = None):
        self.log_level = log_level.upper()
        self.log_dir = log_dir
        self.enable_alerts = enable_alerts
        self.alert_email = alert_email
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure structured logging
        self._setup_structured_logging()
        
        # Configure loguru
        self._setup_loguru()
        
        # Setup alert handler if enabled
        if self.enable_alerts and self.alert_email:
            self._setup_alert_handler()
    
    def _setup_structured_logging(self):
        """Configure structured logging with processors."""
        
        def add_service_context(logger, method_name, event_dict):
            """Add service context to log entries."""
            event_dict["service"] = "container-analytics"
            event_dict["environment"] = os.getenv("ENVIRONMENT", "development")
            event_dict["instance_id"] = os.getenv("HOSTNAME", "local")
            return event_dict
        
        def add_correlation_id(logger, method_name, event_dict):
            """Add correlation ID for request tracing."""
            correlation_id = getattr(logger._context, 'correlation_id', None)
            if correlation_id:
                event_dict["correlation_id"] = correlation_id
            return event_dict
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                add_service_context,
                add_correlation_id,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _setup_loguru(self):
        """Configure loguru with multiple handlers."""
        
        # Remove default handler
        logger.remove()
        
        # Console handler with colored output for development
        if os.getenv("ENVIRONMENT") != "production":
            logger.add(
                sys.stderr,
                level=self.log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True
            )
        
        # JSON handler for production (machine readable)
        logger.add(
            self.log_dir / "app.jsonl",
            level=self.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {extra} | {message}",
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            serialize=True,  # Output as JSON
            enqueue=True,    # Thread-safe
            backtrace=True,
            diagnose=True
        )
        
        # Error-only handler for quick error analysis
        logger.add(
            self.log_dir / "errors.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            rotation="50 MB",
            retention="60 days",
            compression="gz",
            enqueue=True,
            backtrace=True,
            diagnose=True,
            filter=lambda record: record["level"].name in ["ERROR", "CRITICAL"]
        )
        
        # Performance metrics handler
        logger.add(
            self.log_dir / "performance.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {extra} | {message}",
            rotation="100 MB",
            retention="7 days",
            compression="gz",
            enqueue=True,
            filter=lambda record: "performance" in record.get("extra", {})
        )
        
        # Audit trail handler
        logger.add(
            self.log_dir / "audit.log",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra} | {message}",
            rotation="100 MB",
            retention="365 days",  # Keep audit logs longer
            compression="gz",
            enqueue=True,
            filter=lambda record: "audit" in record.get("extra", {})
        )
    
    def _setup_alert_handler(self):
        """Setup email alerts for critical errors."""
        
        class EmailAlertHandler(logging.Handler):
            def __init__(self, alert_config):
                super().__init__()
                self.alert_config = alert_config
                self.setLevel(logging.CRITICAL)
            
            def emit(self, record):
                try:
                    self.send_alert(record)
                except Exception as e:
                    logger.error(f"Failed to send alert email: {e}")
            
            def send_alert(self, record):
                """Send alert email for critical errors."""
                smtp_server = os.getenv("SMTP_SERVER")
                smtp_port = int(os.getenv("SMTP_PORT", "587"))
                smtp_username = os.getenv("SMTP_USERNAME")
                smtp_password = os.getenv("SMTP_PASSWORD")
                
                if not all([smtp_server, smtp_username, smtp_password]):
                    logger.warning("SMTP configuration missing, skipping email alert")
                    return
                
                msg = MimeMultipart()
                msg['From'] = smtp_username
                msg['To'] = self.alert_config["alert_email"]
                msg['Subject'] = f"Container Analytics Alert: {record.levelname}"
                
                body = f"""
Alert Details:
- Time: {datetime.fromtimestamp(record.created).isoformat()}
- Level: {record.levelname}
- Logger: {record.name}
- Location: {record.pathname}:{record.lineno}
- Function: {record.funcName}
- Message: {record.getMessage()}

Exception Details:
{record.exc_text if record.exc_text else 'No exception info'}

Service Information:
- Environment: {os.getenv('ENVIRONMENT', 'unknown')}
- Instance: {os.getenv('HOSTNAME', 'unknown')}
- Service: container-analytics
                """
                
                msg.attach(MimeText(body, 'plain'))
                
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(smtp_username, smtp_password)
                    server.send_message(msg)
        
        # Add email alert handler to root logger
        alert_config = {"alert_email": self.alert_email}
        email_handler = EmailAlertHandler(alert_config)
        logging.getLogger().addHandler(email_handler)
    
    def get_logger(self, name: str = __name__):
        """Get a configured logger instance."""
        return logger.bind(logger_name=name)
    
    def log_performance_metric(self, operation: str, duration: float, **kwargs):
        """Log a performance metric."""
        metric_data = {
            "performance": True,
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        logger.info("Performance metric recorded", extra=metric_data)
    
    def log_audit_event(self, action: str, user: str = "system", **kwargs):
        """Log an audit event."""
        audit_data = {
            "audit": True,
            "action": action,
            "user": user,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        logger.info(f"Audit: {action}", extra=audit_data)
    
    def log_health_check(self, service: str, status: str, **kwargs):
        """Log a health check result."""
        health_data = {
            "health_check": True,
            "service": service,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        logger.info(f"Health check: {service} - {status}", extra=health_data)


class ContextualLogger:
    """Logger with contextual information for request tracing."""
    
    def __init__(self, base_logger, **context):
        self.base_logger = base_logger
        self.context = context
    
    def bind(self, **new_context):
        """Create a new logger with additional context."""
        combined_context = {**self.context, **new_context}
        return ContextualLogger(self.base_logger, **combined_context)
    
    def info(self, message, **kwargs):
        self.base_logger.info(message, extra={**self.context, **kwargs})
    
    def warning(self, message, **kwargs):
        self.base_logger.warning(message, extra={**self.context, **kwargs})
    
    def error(self, message, **kwargs):
        self.base_logger.error(message, extra={**self.context, **kwargs})
    
    def critical(self, message, **kwargs):
        self.base_logger.critical(message, extra={**self.context, **kwargs})
    
    def debug(self, message, **kwargs):
        self.base_logger.debug(message, extra={**self.context, **kwargs})


# Global logging configuration
_logging_config = None

def setup_logging(log_level: str = None, 
                 log_dir: str = None,
                 enable_alerts: bool = None,
                 alert_email: str = None) -> ProductionLoggingConfig:
    """Setup application logging configuration."""
    global _logging_config
    
    # Use environment variables as defaults
    log_level = log_level or os.getenv("LOG_LEVEL", "INFO")
    log_dir = Path(log_dir or os.getenv("LOGS_DIR", "/logs"))
    enable_alerts = enable_alerts if enable_alerts is not None else os.getenv("ENABLE_ALERTS", "true").lower() == "true"
    alert_email = alert_email or os.getenv("ALERT_EMAIL")
    
    _logging_config = ProductionLoggingConfig(
        log_level=log_level,
        log_dir=log_dir,
        enable_alerts=enable_alerts,
        alert_email=alert_email
    )
    
    return _logging_config

def get_logger(name: str = __name__, **context) -> ContextualLogger:
    """Get a contextual logger instance."""
    if _logging_config is None:
        setup_logging()
    
    base_logger = _logging_config.get_logger(name)
    return ContextualLogger(base_logger, **context)

def log_performance(operation: str, duration: float, **kwargs):
    """Log a performance metric."""
    if _logging_config:
        _logging_config.log_performance_metric(operation, duration, **kwargs)

def log_audit(action: str, user: str = "system", **kwargs):
    """Log an audit event."""
    if _logging_config:
        _logging_config.log_audit_event(action, user, **kwargs)

def log_health_check(service: str, status: str, **kwargs):
    """Log a health check result."""
    if _logging_config:
        _logging_config.log_health_check(service, status, **kwargs)


if __name__ == "__main__":
    # Example usage
    setup_logging(log_level="DEBUG", log_dir=Path("./test_logs"))
    
    logger = get_logger("test_module", component="test")
    
    logger.info("Application started")
    logger.warning("This is a warning", extra_field="extra_value")
    
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.error("An error occurred", exc_info=True)
    
    # Log performance metric
    log_performance("image_download", 1.25, image_count=5, stream="in_gate")
    
    # Log audit event
    log_audit("user_login", user="admin", ip_address="192.168.1.100")
    
    # Log health check
    log_health_check("scheduler", "healthy", queue_size=0)
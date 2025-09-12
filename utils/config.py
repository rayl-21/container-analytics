"""
Configuration management for the Container Analytics application.

Uses Pydantic Settings for type-safe configuration loading from environment variables
and .env files. Provides validation, defaults, and centralized access to all settings.

Example:
    >>> from utils.config import settings
    >>> 
    >>> # Access database URL
    >>> db_url = settings.database_url
    >>> 
    >>> # Check if caching is enabled
    >>> if settings.enable_caching:
    >>>     cache_ttl = settings.cache_ttl_seconds
"""

import os
from pathlib import Path
from typing import List, Optional, Union
try:
    from pydantic import field_validator, Field
    from pydantic_settings import BaseSettings
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for when pydantic is not installed
    PYDANTIC_AVAILABLE = False
    class BaseSettings:
        """Fallback BaseSettings when pydantic is not available."""
        def __init__(self):
            pass
    
    def Field(**kwargs):
        """Fallback Field when pydantic is not available."""
        return kwargs.get('default', None)
    
    def field_validator(*args, **kwargs):
        """Fallback field_validator when pydantic is not available."""
        def decorator(func):
            return func
        return decorator


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env files."""
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///data/database.db",
        description="Database connection URL"
    )
    database_echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy query logging"
    )
    
    # Dray Dog Authentication
    draydog_username: str = Field(
        default="",
        description="Dray Dog login username"
    )
    draydog_password: str = Field(
        default="",
        description="Dray Dog login password"
    )
    draydog_base_url: str = Field(
        default="https://draydog.com",
        description="Dray Dog base URL"
    )
    
    # Image Download Configuration
    download_interval_minutes: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Interval between image downloads in minutes"
    )
    max_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts for failed downloads"
    )
    retry_delay_seconds: int = Field(
        default=5,
        ge=1,
        le=300,
        description="Delay between retry attempts in seconds"
    )
    image_storage_path: Path = Field(
        default=Path("data/images"),
        description="Local path for storing downloaded images"
    )
    max_storage_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Maximum days to retain images"
    )
    
    # YOLO Model Configuration
    yolo_model_path: Path = Field(
        default=Path("data/models/yolov12x.pt"),
        description="Path to YOLO model weights"
    )
    detection_confidence_threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.95,
        description="Minimum confidence threshold for detections"
    )
    tracking_enabled: bool = Field(
        default=True,
        description="Enable object tracking across frames"
    )
    tracking_max_disappeared: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Max frames an object can disappear before removal"
    )
    
    # OCR Configuration
    ocr_engine: str = Field(
        default="easyocr",
        pattern="^(easyocr|tesseract)$",
        description="OCR engine to use (easyocr or tesseract)"
    )
    tesseract_cmd: Optional[Path] = Field(
        default="/usr/bin/tesseract",
        description="Path to tesseract executable"
    )
    
    # Streamlit Configuration
    streamlit_port: int = Field(
        default=8501,
        ge=1024,
        le=65535,
        description="Streamlit server port"
    )
    streamlit_host: str = Field(
        default="localhost",
        description="Streamlit server host"
    )
    auto_refresh_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Auto refresh interval for live data in seconds"
    )
    
    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Application log level"
    )
    log_file: Path = Field(
        default=Path("data/logs/app.log"),
        description="Path to log file"
    )
    log_max_bytes: int = Field(
        default=10485760,  # 10MB
        ge=1024,
        le=104857600,  # 100MB
        description="Maximum log file size in bytes"
    )
    log_backup_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of backup log files to keep"
    )
    
    # Performance Settings
    enable_caching: bool = Field(
        default=True,
        description="Enable application caching"
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Default cache TTL in seconds"
    )
    max_concurrent_downloads: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent image downloads"
    )
    image_resize_max_width: int = Field(
        default=1920,
        ge=640,
        le=4096,
        description="Maximum image width for processing"
    )
    image_resize_max_height: int = Field(
        default=1080,
        ge=480,
        le=2160,
        description="Maximum image height for processing"
    )
    
    # Alert Configuration
    enable_alerts: bool = Field(
        default=True,
        description="Enable alert notifications"
    )
    alert_email_enabled: bool = Field(
        default=False,
        description="Enable email alerts"
    )
    alert_email_smtp_host: str = Field(
        default="smtp.gmail.com",
        description="SMTP server hostname"
    )
    alert_email_smtp_port: int = Field(
        default=587,
        ge=25,
        le=65535,
        description="SMTP server port"
    )
    alert_email_username: Optional[str] = Field(
        default=None,
        description="SMTP authentication username"
    )
    alert_email_password: Optional[str] = Field(
        default=None,
        description="SMTP authentication password"
    )
    alert_email_to: Optional[str] = Field(
        default=None,
        description="Alert recipient email address"
    )
    
    # Analytics Configuration
    dwell_time_threshold_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # 1 week
        description="Threshold for container dwell time alerts in hours"
    )
    throughput_calculation_window_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Time window for throughput calculations in hours"
    )
    anomaly_detection_sensitivity: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Anomaly detection sensitivity (0.1-1.0)"
    )
    
    # Redis Configuration
    redis_host: str = Field(
        default="localhost",
        description="Redis server hostname"
    )
    redis_port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis server port"
    )
    redis_db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number"
    )
    redis_password: Optional[str] = Field(
        default=None,
        description="Redis authentication password"
    )
    
    # Development Settings
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    enable_profiling: bool = Field(
        default=False,
        description="Enable performance profiling"
    )
    mock_camera_data: bool = Field(
        default=False,
        description="Use mock data instead of real camera feeds"
    )
    
    # Selenium Configuration
    selenium_driver: str = Field(
        default="chrome",
        pattern="^(chrome|firefox|safari|edge)$",
        description="Selenium WebDriver to use"
    )
    selenium_headless: bool = Field(
        default=True,
        description="Run Selenium in headless mode"
    )
    selenium_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Selenium operation timeout in seconds"
    )
    selenium_implicit_wait_seconds: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Selenium implicit wait timeout in seconds"
    )
    selenium_page_load_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Selenium page load timeout in seconds"
    )
    
    # Detection Classes
    container_classes: List[str] = Field(
        default=["truck", "container", "trailer", "chassis"],
        description="Object classes to detect as containers"
    )
    vehicle_classes: List[str] = Field(
        default=["car", "truck", "bus", "motorcycle"],
        description="Object classes to detect as vehicles"
    )
    
    # API Configuration
    api_key: Optional[str] = Field(
        default=None,
        description="API key for external integrations"
    )
    api_rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="API rate limit per minute"
    )
    
    # Backup Configuration
    backup_enabled: bool = Field(
        default=False,
        description="Enable automatic backups"
    )
    backup_interval_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Backup interval in hours"
    )
    backup_retention_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Number of days to retain backups"
    )
    backup_destination: Path = Field(
        default=Path("data/backups"),
        description="Backup destination directory"
    )
    
    @field_validator("image_storage_path", "yolo_model_path", "log_file", "backup_destination")
    @classmethod
    def ensure_path_exists(cls, v: Path) -> Path:
        """Ensure directory paths exist, create if necessary."""
        if isinstance(v, str):
            v = Path(v)
        
        # Create parent directories if they don't exist
        if v.suffix:  # It's a file path
            v.parent.mkdir(parents=True, exist_ok=True)
        else:  # It's a directory path
            v.mkdir(parents=True, exist_ok=True)
        
        return v
    
    @field_validator("container_classes", "vehicle_classes", mode='before')
    @classmethod
    def parse_comma_separated_list(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse comma-separated strings into lists."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v
    
    @field_validator("draydog_username", "draydog_password")
    @classmethod
    def validate_required_for_production(cls, v: str, info) -> str:
        """Validate that required credentials are provided in production."""
        debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        if not debug_mode and not v:
            raise ValueError(f"{info.field_name} is required in production mode")
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        extra = "ignore"  # Ignore extra fields not defined in model
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
    
    def get_database_path(self) -> Optional[Path]:
        """Get database file path if using SQLite."""
        if self.database_url.startswith("sqlite"):
            # Extract path from sqlite:///path/to/db
            return Path(self.database_url.replace("sqlite:///", ""))
        return None
    
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.debug_mode or self.log_level == "DEBUG"
    
    def validate_required_paths(self) -> List[str]:
        """Validate that all required file paths exist."""
        errors = []
        
        # Check YOLO model exists
        if not self.yolo_model_path.exists():
            errors.append(f"YOLO model not found at {self.yolo_model_path}")
        
        # Check tesseract if using tesseract OCR
        if self.ocr_engine == "tesseract" and self.tesseract_cmd:
            if not self.tesseract_cmd.exists():
                errors.append(f"Tesseract not found at {self.tesseract_cmd}")
        
        return errors


def load_settings() -> Settings:
    """Load and validate application settings."""
    try:
        if not PYDANTIC_AVAILABLE:
            # Return a basic settings object with minimal functionality
            settings = _create_basic_settings()
            return settings
        
        settings = Settings()
        
        # Validate required paths in production
        if not settings.is_development():
            errors = settings.validate_required_paths()
            if errors:
                raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return settings
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise


def _create_basic_settings():
    """Create basic settings when pydantic is not available."""
    import os
    from pathlib import Path
    
    class BasicSettings:
        def __init__(self):
            # Basic default values from .env or environment
            self.database_url = os.getenv("DATABASE_URL", "sqlite:///data/database.db")
            self.database_echo = os.getenv("DATABASE_ECHO", "false").lower() == "true"
            self.log_level = os.getenv("LOG_LEVEL", "INFO")
            self.log_file = Path(os.getenv("LOG_FILE", "data/logs/app.log"))
            self.log_max_bytes = int(os.getenv("LOG_MAX_BYTES", "10485760"))
            self.log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
            self.enable_caching = os.getenv("ENABLE_CACHING", "true").lower() == "true"
            self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "300"))
            self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
            self.redis_host = os.getenv("REDIS_HOST", "localhost")
            self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
            self.redis_db = int(os.getenv("REDIS_DB", "0"))
            self.redis_password = os.getenv("REDIS_PASSWORD")
        
        def is_development(self) -> bool:
            return self.debug_mode or self.log_level == "DEBUG"
        
        def get_redis_url(self) -> str:
            if self.redis_password:
                return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
            return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"
        
        def validate_required_paths(self) -> list:
            return []  # Skip validation in basic mode
    
    return BasicSettings()


# Global settings instance
settings = load_settings()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    return get_project_root() / "data"


def get_logs_dir() -> Path:
    """Get the logs directory."""
    logs_dir = get_data_dir() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir
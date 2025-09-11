"""
Utility modules for the Container Analytics application.

This package provides common utilities including:
- Configuration management
- Logging setup and configuration  
- Caching utilities with Redis integration
- Common helper functions

Example:
    >>> from utils.config import settings
    >>> from utils.logging_config import setup_logging
    >>> from utils.cache import cache_manager
    
    >>> # Setup logging
    >>> setup_logging()
    
    >>> # Access configuration
    >>> db_url = settings.database_url
    
    >>> # Use caching
    >>> @cache_manager.cached(ttl=300)
    >>> def expensive_operation():
    >>>     return some_result
"""

from .config import settings
from .logging_config import setup_logging, get_logger
from .cache import cache_manager, cached, invalidate_cache

__all__ = [
    "settings",
    "setup_logging", 
    "get_logger",
    "cache_manager",
    "cached",
    "invalidate_cache",
]

__version__ = "1.0.0"
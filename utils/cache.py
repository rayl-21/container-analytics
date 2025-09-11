"""
Caching utilities for the Container Analytics application.

Provides memory caching with optional Redis backend, TTL management, and 
function decorators for easy caching of expensive operations.

Example:
    >>> from utils.cache import cache_manager, cached, invalidate_cache
    >>> 
    >>> # Cache a function result
    >>> @cached(ttl=300, key_prefix="detection")
    >>> def detect_objects(image_path: str):
    >>>     return expensive_detection_operation(image_path)
    >>> 
    >>> # Manual cache operations
    >>> cache_manager.set("key", "value", ttl=300)
    >>> value = cache_manager.get("key")
    >>> 
    >>> # Invalidate specific cache entries
    >>> invalidate_cache("detection:*")
"""

import hashlib
import json
import pickle
import time
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Optional, Union, Dict, List, Callable, Pattern
from pathlib import Path
import re

try:
    from .config import settings
except ImportError:
    # Fallback settings if config module is not available
    class FallbackSettings:
        enable_caching = True
        cache_ttl_seconds = 300
        redis_host = "localhost"
        redis_port = 6379
        redis_db = 0
        redis_password = None
    settings = FallbackSettings()

try:
    from .logging_config import get_logger, log_cache_operation
    logger = get_logger(__name__)
except ImportError:
    # Fallback to standard logging
    import logging
    logger = logging.getLogger(__name__)
    
    def log_cache_operation(operation, key, **kwargs):
        logger.debug(f"Cache {operation}: {key}")


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache implementation with TTL support."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize memory cache.
        
        Args:
            max_size: Maximum number of cache entries
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry has expired."""
        if entry["expires_at"] is None:
            return False
        return time.time() > entry["expires_at"]
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry["expires_at"] and current_time > entry["expires_at"]
        ]
        
        for key in expired_keys:
            del self._cache[key]
    
    def _evict_oldest(self) -> None:
        """Evict oldest entries to maintain max_size."""
        if len(self._cache) >= self._max_size:
            # Remove 20% of oldest entries
            entries_to_remove = max(1, len(self._cache) // 5)
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: x[1]["created_at"]
            )
            
            for key, _ in sorted_entries[:entries_to_remove]:
                del self._cache[key]
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self._cache:
            self._misses += 1
            log_cache_operation("GET", key, hit=False)
            return None
        
        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            self._misses += 1
            log_cache_operation("GET", key, hit=False)
            return None
        
        # Update access time for LRU
        entry["accessed_at"] = time.time()
        self._hits += 1
        log_cache_operation("GET", key, hit=True)
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        try:
            self._cleanup_expired()
            self._evict_oldest()
            
            current_time = time.time()
            expires_at = current_time + ttl if ttl else None
            
            self._cache[key] = {
                "value": value,
                "created_at": current_time,
                "accessed_at": current_time,
                "expires_at": expires_at,
            }
            
            self._sets += 1
            
            # Calculate approximate size
            size_bytes = len(pickle.dumps(value)) if hasattr(value, '__dict__') else len(str(value))
            log_cache_operation("SET", key, ttl_seconds=ttl, size_bytes=size_bytes)
            
            return True
        except Exception as e:
            logger.error("Failed to set cache entry", key=key, error=str(e))
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            self._deletes += 1
            log_cache_operation("DELETE", key)
            return True
        return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if key not in self._cache:
            return False
        
        entry = self._cache[key]
        if self._is_expired(entry):
            del self._cache[key]
            return False
        
        return True
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        self._cache.clear()
        log_cache_operation("CLEAR", "all")
        return True
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        if pattern == "*":
            return list(self._cache.keys())
        
        # Convert glob pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".")
        regex = re.compile(regex_pattern)
        
        return [key for key in self._cache.keys() if regex.match(key)]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0
        
        return {
            "backend": "memory",
            "entries": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 3),
            "sets": self._sets,
            "deletes": self._deletes,
        }


class RedisCache(CacheBackend):
    """Redis cache implementation."""
    
    def __init__(self, redis_client=None):
        """
        Initialize Redis cache.
        
        Args:
            redis_client: Redis client instance (will create if None)
        """
        if redis_client:
            self._redis = redis_client
        else:
            self._redis = self._create_redis_client()
        
        self._hits = 0
        self._misses = 0
        self._sets = 0
        self._deletes = 0
    
    def _create_redis_client(self):
        """Create Redis client from settings."""
        try:
            import redis
            
            return redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                password=settings.redis_password,
                decode_responses=False,  # We'll handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
        except ImportError:
            logger.warning("Redis not available, falling back to memory cache")
            return None
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            return None
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for Redis storage."""
        return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from Redis."""
        return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._redis:
            return None
        
        try:
            data = self._redis.get(key)
            if data is None:
                self._misses += 1
                log_cache_operation("GET", key, hit=False)
                return None
            
            value = self._deserialize(data)
            self._hits += 1
            log_cache_operation("GET", key, hit=True)
            return value
            
        except Exception as e:
            logger.error("Redis get failed", key=key, error=str(e))
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        if not self._redis:
            return False
        
        try:
            serialized = self._serialize(value)
            
            if ttl:
                result = self._redis.setex(key, ttl, serialized)
            else:
                result = self._redis.set(key, serialized)
            
            if result:
                self._sets += 1
                log_cache_operation("SET", key, ttl_seconds=ttl, size_bytes=len(serialized))
            
            return bool(result)
            
        except Exception as e:
            logger.error("Redis set failed", key=key, error=str(e))
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self._redis:
            return False
        
        try:
            result = self._redis.delete(key)
            if result:
                self._deletes += 1
                log_cache_operation("DELETE", key)
            return result > 0
            
        except Exception as e:
            logger.error("Redis delete failed", key=key, error=str(e))
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._redis:
            return False
        
        try:
            return bool(self._redis.exists(key))
        except Exception as e:
            logger.error("Redis exists failed", key=key, error=str(e))
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        if not self._redis:
            return False
        
        try:
            self._redis.flushdb()
            log_cache_operation("CLEAR", "all")
            return True
        except Exception as e:
            logger.error("Redis clear failed", error=str(e))
            return False
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        if not self._redis:
            return []
        
        try:
            keys = self._redis.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error("Redis keys failed", pattern=pattern, error=str(e))
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._redis:
            return {"backend": "redis", "available": False}
        
        try:
            info = self._redis.info("memory")
            db_info = self._redis.info("keyspace")
            
            # Extract database info
            db_keys = 0
            if f"db{settings.redis_db}" in db_info:
                db_stats = db_info[f"db{settings.redis_db}"]
                db_keys = db_stats.get("keys", 0)
            
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests) if total_requests > 0 else 0
            
            return {
                "backend": "redis",
                "available": True,
                "entries": db_keys,
                "memory_used_bytes": info.get("used_memory", 0),
                "memory_peak_bytes": info.get("used_memory_peak", 0),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
                "sets": self._sets,
                "deletes": self._deletes,
            }
        except Exception as e:
            logger.error("Redis stats failed", error=str(e))
            return {"backend": "redis", "available": False, "error": str(e)}


class CacheManager:
    """High-level cache manager with fallback support."""
    
    def __init__(self, primary_backend: CacheBackend, fallback_backend: Optional[CacheBackend] = None):
        """
        Initialize cache manager.
        
        Args:
            primary_backend: Primary cache backend (e.g., Redis)
            fallback_backend: Fallback backend (e.g., Memory) if primary fails
        """
        self._primary = primary_backend
        self._fallback = fallback_backend
        self._use_fallback = False
    
    def _get_backend(self) -> CacheBackend:
        """Get the active cache backend."""
        if self._use_fallback and self._fallback:
            return self._fallback
        return self._primary
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        backend = self._get_backend()
        result = backend.get(key)
        
        # Try fallback if primary fails
        if result is None and not self._use_fallback and self._fallback:
            result = self._fallback.get(key)
        
        return result
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        backend = self._get_backend()
        success = backend.set(key, value, ttl)
        
        # Set in fallback as well for redundancy
        if success and not self._use_fallback and self._fallback:
            self._fallback.set(key, value, ttl)
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        backend = self._get_backend()
        success = backend.delete(key)
        
        # Delete from fallback as well
        if self._fallback:
            self._fallback.delete(key)
        
        return success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        backend = self._get_backend()
        return backend.exists(key)
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        backend = self._get_backend()
        success = backend.clear()
        
        # Clear fallback as well
        if self._fallback:
            self._fallback.clear()
        
        return success
    
    def keys(self, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern."""
        backend = self._get_backend()
        return backend.keys(pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        primary_stats = self._primary.get_stats()
        stats = {
            "primary": primary_stats,
            "active_backend": primary_stats.get("backend", "unknown"),
            "fallback_active": self._use_fallback,
        }
        
        if self._fallback:
            stats["fallback"] = self._fallback.get_stats()
        
        return stats
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache entries matching pattern.
        
        Args:
            pattern: Glob pattern to match keys
            
        Returns:
            Number of keys invalidated
        """
        keys = self.keys(pattern)
        count = 0
        
        for key in keys:
            if self.delete(key):
                count += 1
        
        log_cache_operation("INVALIDATE", pattern)
        logger.info("Cache invalidation completed", pattern=pattern, keys_removed=count)
        
        return count


def _create_cache_manager() -> CacheManager:
    """Create and configure the global cache manager."""
    # Create fallback memory cache
    memory_cache = MemoryCache(max_size=1000)
    
    # Try to create Redis cache if enabled
    redis_cache = None
    if settings.enable_caching:
        try:
            redis_cache = RedisCache()
            if redis_cache._redis:
                logger.info("Redis cache initialized successfully")
            else:
                logger.warning("Redis unavailable, using memory cache only")
        except Exception as e:
            logger.warning("Failed to initialize Redis cache", error=str(e))
    
    # Use Redis as primary, memory as fallback
    if redis_cache and redis_cache._redis:
        return CacheManager(primary_backend=redis_cache, fallback_backend=memory_cache)
    else:
        return CacheManager(primary_backend=memory_cache)


def _generate_cache_key(func: Callable, args: tuple, kwargs: dict, key_prefix: str = "") -> str:
    """Generate cache key from function and arguments."""
    # Create a string representation of args and kwargs
    key_parts = [func.__module__, func.__name__]
    
    if key_prefix:
        key_parts.insert(0, key_prefix)
    
    # Add args
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For complex objects, use their string representation
            key_parts.append(str(arg))
        else:
            key_parts.append(str(arg))
    
    # Add kwargs
    for key, value in sorted(kwargs.items()):
        key_parts.append(f"{key}={value}")
    
    # Create hash of the key to ensure consistent length
    key_string = ":".join(key_parts)
    key_hash = hashlib.md5(key_string.encode()).hexdigest()
    
    # Include readable part and hash
    readable_part = ":".join(key_parts[:3])  # First 3 parts for readability
    return f"{readable_part}:{key_hash}"


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    key_func: Optional[Callable] = None,
    condition: Optional[Callable] = None,
) -> Callable:
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds (defaults to settings.cache_ttl_seconds)
        key_prefix: Prefix for cache keys
        key_func: Custom function to generate cache key
        condition: Function to determine if result should be cached
        
    Example:
        >>> @cached(ttl=300, key_prefix="detection")
        >>> def detect_objects(image_path: str):
        >>>     return expensive_operation(image_path)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not settings.enable_caching:
                return func(*args, **kwargs)
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = _generate_cache_key(func, args, kwargs, key_prefix)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Cache result if condition is met
            should_cache = True
            if condition:
                should_cache = condition(result)
            
            if should_cache and result is not None:
                cache_ttl = ttl or settings.cache_ttl_seconds
                cache_manager.set(cache_key, result, cache_ttl)
            
            # Log performance
            from .logging_config import log_performance
            log_performance(func.__name__, execution_time, cached=False)
            
            return result
        
        # Add cache control methods to the wrapped function
        wrapper.cache_clear = lambda: cache_manager.invalidate_pattern(f"{key_prefix}*")
        wrapper.cache_info = lambda: cache_manager.get_stats()
        
        return wrapper
    return decorator


def invalidate_cache(pattern: str) -> int:
    """
    Invalidate cache entries matching the given pattern.
    
    Args:
        pattern: Glob pattern to match cache keys
        
    Returns:
        Number of cache entries invalidated
    """
    return cache_manager.invalidate_pattern(pattern)


def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics."""
    return cache_manager.get_stats()


def clear_all_cache() -> bool:
    """Clear all cache entries."""
    return cache_manager.clear()


# Global cache manager instance
cache_manager = _create_cache_manager()

# Initialize cache stats logging
logger.info("Cache manager initialized", stats=cache_manager.get_stats())
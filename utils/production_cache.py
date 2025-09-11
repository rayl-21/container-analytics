#!/usr/bin/env python3
"""
Advanced caching utilities for Container Analytics with Redis integration,
connection pooling, and cache optimization strategies.
"""

import os
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List, Callable, Union
from pathlib import Path
from functools import wraps

import redis
from diskcache import Cache as DiskCache


class RedisCache:
    """Redis-based cache with connection pooling and failover to disk cache."""
    
    def __init__(self, 
                 redis_url: str = None,
                 default_ttl: int = 3600,
                 fallback_cache_dir: Path = Path("/data/cache")):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.default_ttl = default_ttl
        self.fallback_cache_dir = fallback_cache_dir
        
        # Initialize Redis connection pool
        self.redis_client = None
        self._init_redis()
        
        # Initialize disk cache as fallback
        self.fallback_cache_dir.mkdir(parents=True, exist_ok=True)
        self.disk_cache = DiskCache(str(self.fallback_cache_dir))
        
    def _init_redis(self):
        """Initialize Redis connection with pool."""
        try:
            # Parse Redis URL and create connection pool
            if self.redis_url.startswith('redis://'):
                self.redis_client = redis.from_url(
                    self.redis_url,
                    connection_pool_kwargs={
                        'max_connections': 20,
                        'retry_on_timeout': True,
                        'health_check_interval': 30
                    },
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                print(f"Redis cache initialized: {self.redis_url}")
            else:
                raise ValueError("Invalid Redis URL format")
                
        except Exception as e:
            print(f"Redis initialization failed: {e}")
            print("Falling back to disk cache only")
            self.redis_client = None
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Use JSON for simple types
            if isinstance(value, (str, int, float, bool, list, dict)):
                return json.dumps(value).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(value)
        except Exception:
            # Fallback to string representation
            return str(value).encode('utf-8')
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            try:
                # Try pickle
                return pickle.loads(data)
            except Exception:
                # Return as bytes if all else fails
                return data
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with TTL."""
        ttl = ttl or self.default_ttl
        serialized_value = self._serialize_value(value)
        
        # Try Redis first
        if self.redis_client:
            try:
                return self.redis_client.setex(key, ttl, serialized_value)
            except Exception as e:
                print(f"Redis set failed for key {key}: {e}")
        
        # Fallback to disk cache
        try:
            expire_time = datetime.utcnow() + timedelta(seconds=ttl)
            return self.disk_cache.set(key, serialized_value, expire=expire_time.timestamp())
        except Exception as e:
            print(f"Disk cache set failed for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data is not None:
                    return self._deserialize_value(data)
            except Exception as e:
                print(f"Redis get failed for key {key}: {e}")
        
        # Try disk cache
        try:
            data = self.disk_cache.get(key)
            if data is not None:
                return self._deserialize_value(data)
        except Exception as e:
            print(f"Disk cache get failed for key {key}: {e}")
        
        return None
    
    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        success = True
        
        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                print(f"Redis delete failed for key {key}: {e}")
                success = False
        
        # Delete from disk cache
        try:
            self.disk_cache.delete(key)
        except Exception as e:
            print(f"Disk cache delete failed for key {key}: {e}")
            success = False
        
        return success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        # Check Redis first
        if self.redis_client:
            try:
                if self.redis_client.exists(key):
                    return True
            except Exception:
                pass
        
        # Check disk cache
        try:
            return key in self.disk_cache
        except Exception:
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries."""
        success = True
        
        # Clear Redis
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                print(f"Redis clear failed: {e}")
                success = False
        
        # Clear disk cache
        try:
            self.disk_cache.clear()
        except Exception as e:
            print(f"Disk cache clear failed: {e}")
            success = False
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {}
        
        # Redis stats
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats['redis'] = {
                    'connected': True,
                    'used_memory': redis_info.get('used_memory_human'),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'connected_clients': redis_info.get('connected_clients', 0)
                }
            except Exception:
                stats['redis'] = {'connected': False}
        
        # Disk cache stats
        try:
            stats['disk_cache'] = {
                'size': len(self.disk_cache),
                'volume': self.disk_cache.volume(),
                'directory': str(self.fallback_cache_dir)
            }
        except Exception:
            stats['disk_cache'] = {'error': 'Unable to get disk cache stats'}
        
        return stats


# Global cache instance
_cache_instance = None

def get_cache() -> RedisCache:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


def cache_key_generator(*args, **kwargs) -> str:
    """Generate a cache key from function arguments."""
    # Create a deterministic string from arguments
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For objects, use class name and key attributes
            key_parts.append(f"{arg.__class__.__name__}:{id(arg)}")
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments (sorted for consistency)
    for k, v in sorted(kwargs.items()):
        key_parts.append(f"{k}:{v}")
    
    # Create hash of the key for consistent length
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode('utf-8')).hexdigest()


def cached(ttl: int = 3600, key_prefix: str = ""):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = f"{key_prefix}:{func.__name__}:{cache_key_generator(*args, **kwargs)}"
            
            # Try to get from cache
            cache = get_cache()
            cached_result = cache.get(key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: get_cache().delete(f"{key_prefix}:{func.__name__}")
        wrapper.cache_info = lambda: get_cache().get_stats()
        
        return wrapper
    return decorator


class ImageCache:
    """Specialized cache for image processing results."""
    
    def __init__(self, cache: RedisCache = None):
        self.cache = cache or get_cache()
    
    def cache_detection_result(self, image_path: str, detections: List[Dict], ttl: int = 7200):
        """Cache YOLO detection results for an image."""
        key = f"detection:{hashlib.md5(image_path.encode()).hexdigest()}"
        result = {
            'image_path': image_path,
            'detections': detections,
            'timestamp': datetime.utcnow().isoformat(),
            'detection_count': len(detections)
        }
        return self.cache.set(key, result, ttl)
    
    def get_detection_result(self, image_path: str) -> Optional[Dict]:
        """Get cached detection results for an image."""
        key = f"detection:{hashlib.md5(image_path.encode()).hexdigest()}"
        return self.cache.get(key)
    
    def cache_processed_image(self, image_path: str, processed_data: bytes, ttl: int = 3600):
        """Cache processed image data."""
        key = f"processed:{hashlib.md5(image_path.encode()).hexdigest()}"
        return self.cache.set(key, processed_data, ttl)
    
    def get_processed_image(self, image_path: str) -> Optional[bytes]:
        """Get cached processed image data."""
        key = f"processed:{hashlib.md5(image_path.encode()).hexdigest()}"
        return self.cache.get(key)
    
    def invalidate_image_cache(self, image_path: str):
        """Invalidate all cache entries for an image."""
        image_hash = hashlib.md5(image_path.encode()).hexdigest()
        keys_to_delete = [
            f"detection:{image_hash}",
            f"processed:{image_hash}"
        ]
        
        for key in keys_to_delete:
            self.cache.delete(key)


class MetricsCache:
    """Cache for analytics metrics and aggregations."""
    
    def __init__(self, cache: RedisCache = None):
        self.cache = cache or get_cache()
    
    @cached(ttl=300, key_prefix="metrics")
    def get_hourly_stats(self, hour: datetime) -> Dict[str, Any]:
        """Get cached hourly statistics (5 minute TTL)."""
        # This would be implemented to calculate hourly stats
        # The decorator will handle caching automatically
        pass
    
    @cached(ttl=900, key_prefix="metrics")
    def get_daily_summary(self, date: datetime) -> Dict[str, Any]:
        """Get cached daily summary (15 minute TTL)."""
        # This would be implemented to calculate daily summary
        # The decorator will handle caching automatically
        pass
    
    def cache_kpi(self, kpi_name: str, value: float, ttl: int = 300):
        """Cache a KPI value."""
        key = f"kpi:{kpi_name}"
        data = {
            'value': value,
            'timestamp': datetime.utcnow().isoformat(),
            'kpi_name': kpi_name
        }
        return self.cache.set(key, data, ttl)
    
    def get_kpi(self, kpi_name: str) -> Optional[Dict]:
        """Get cached KPI value."""
        key = f"kpi:{kpi_name}"
        return self.cache.get(key)


# Convenience functions
def cache_image_detections(image_path: str, detections: List[Dict], ttl: int = 7200):
    """Cache detection results for an image."""
    image_cache = ImageCache()
    return image_cache.cache_detection_result(image_path, detections, ttl)

def get_cached_detections(image_path: str) -> Optional[Dict]:
    """Get cached detection results for an image."""
    image_cache = ImageCache()
    return image_cache.get_detection_result(image_path)

def cache_metric(kpi_name: str, value: float, ttl: int = 300):
    """Cache a metric value."""
    metrics_cache = MetricsCache()
    return metrics_cache.cache_kpi(kpi_name, value, ttl)

def get_cached_metric(kpi_name: str) -> Optional[Dict]:
    """Get cached metric value."""
    metrics_cache = MetricsCache()
    return metrics_cache.get_kpi(kpi_name)


if __name__ == "__main__":
    # Example usage
    cache = RedisCache()
    
    # Test basic operations
    cache.set("test_key", {"data": "test_value"}, ttl=60)
    result = cache.get("test_key")
    print(f"Cache test result: {result}")
    
    # Test decorator
    @cached(ttl=300, key_prefix="example")
    def expensive_calculation(x: int, y: int) -> int:
        import time
        time.sleep(1)  # Simulate expensive operation
        return x * y + 42
    
    # First call - will execute function
    start_time = datetime.now()
    result1 = expensive_calculation(10, 20)
    duration1 = (datetime.now() - start_time).total_seconds()
    
    # Second call - will use cache
    start_time = datetime.now()
    result2 = expensive_calculation(10, 20)
    duration2 = (datetime.now() - start_time).total_seconds()
    
    print(f"First call: {result1} in {duration1:.3f}s")
    print(f"Second call: {result2} in {duration2:.3f}s")
    print(f"Cache working: {duration2 < duration1}")
    
    # Print cache stats
    print("Cache stats:", cache.get_stats())
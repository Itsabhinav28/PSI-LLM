"""
Performance Optimization & Caching for RAG Pipeline

Implements caching strategies, batching, and performance optimizations
to improve response times and efficiency.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    data: Any
    timestamp: datetime
    ttl: int  # Time to live in seconds
    access_count: int = 0


class CacheManager:
    """Manages caching for improved performance."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """Initialize cache manager."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_times: Dict[str, datetime] = {}
        
        logger.info(f"Cache manager initialized with max_size={max_size}, ttl={default_ttl}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        # Convert arguments to a consistent string representation
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve an item from cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if expired
        if datetime.now() > entry.timestamp + timedelta(seconds=entry.ttl):
            self.delete(key)
            return None
        
        # Update access metadata
        entry.access_count += 1
        self.access_times[key] = datetime.now()
        
        logger.debug(f"Cache hit for key: {key}")
        return entry.data
    
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Store an item in cache."""
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        ttl = ttl or self.default_ttl
        entry = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl=ttl
        )
        
        self.cache[key] = entry
        self.access_times[key] = datetime.now()
        
        logger.debug(f"Cached item with key: {key}, ttl: {ttl}s")
    
    def delete(self, key: str) -> bool:
        """Delete an item from cache."""
        if key in self.cache:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            logger.debug(f"Deleted cache entry: {key}")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Cache cleared")
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        if not self.access_times:
            return
        
        # Find the least recently used key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self.delete(lru_key)
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        expired_count = sum(
            1 for entry in self.cache.values()
            if now > entry.timestamp + timedelta(seconds=entry.ttl)
        )
        
        return {
            'total_entries': len(self.cache),
            'max_size': self.max_size,
            'expired_entries': expired_count,
            'utilization': len(self.cache) / self.max_size,
            'oldest_entry': min(self.access_times.values()) if self.access_times else None,
            'newest_entry': max(self.access_times.values()) if self.access_times else None
        }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now > entry.timestamp + timedelta(seconds=entry.ttl)
        ]
        
        for key in expired_keys:
            self.delete(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)


class BatchProcessor:
    """Handles batch processing for improved efficiency."""
    
    def __init__(self, batch_size: int = 100, max_wait_time: float = 1.0):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items: List[Any] = []
        self.last_process_time = time.time()
        
        logger.info(f"Batch processor initialized: batch_size={batch_size}, max_wait={max_wait_time}s")
    
    def add_item(self, item: Any) -> bool:
        """Add item to batch and return True if batch should be processed."""
        self.pending_items.append(item)
        
        # Check if batch should be processed
        should_process = (
            len(self.pending_items) >= self.batch_size or
            time.time() - self.last_process_time >= self.max_wait_time
        )
        
        if should_process:
            self.last_process_time = time.time()
        
        return should_process
    
    def get_batch(self) -> List[Any]:
        """Get current batch and clear pending items."""
        batch = self.pending_items.copy()
        self.pending_items.clear()
        return batch
    
    def get_batch_size(self) -> int:
        """Get current batch size."""
        return len(self.pending_items)


class PerformanceMonitor:
    """Monitors and tracks performance metrics."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, List[float]] = {
            'query_time': [],
            'embedding_time': [],
            'retrieval_time': [],
            'generation_time': []
        }
        self.start_times: Dict[str, float] = {}
        
        logger.info("Performance monitor initialized")
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            logger.warning(f"Timer not started for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        if operation in self.metrics:
            self.metrics[operation].append(duration)
        
        logger.debug(f"Operation '{operation}' took {duration:.3f}s")
        return duration
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return 0.0
        
        return sum(self.metrics[operation]) / len(self.metrics[operation])
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    'count': len(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times)
                }
            else:
                summary[operation] = {
                    'count': 0,
                    'average': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'total': 0.0
                }
        
        return summary
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        for operation in self.metrics:
            self.metrics[operation].clear()
        self.start_times.clear()
        logger.info("Performance metrics reset")


class QueryOptimizer:
    """Optimizes queries for better performance."""
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize query optimizer."""
        self.cache_manager = cache_manager
        self.query_cache = {}
        
        logger.info("Query optimizer initialized")
    
    def optimize_query(self, query: str) -> str:
        """Optimize query for better retrieval."""
        # Basic query optimization
        optimized = query.strip().lower()
        
        # Remove excessive whitespace
        optimized = ' '.join(optimized.split())
        
        # Cache the optimization
        cache_key = f"query_opt_{hash(query)}"
        self.cache_manager.set(cache_key, optimized, ttl=1800)  # 30 minutes
        
        return optimized
    
    def get_cached_optimization(self, query: str) -> Optional[str]:
        """Get cached query optimization."""
        cache_key = f"query_opt_{hash(query)}"
        return self.cache_manager.get(cache_key)
    
    def batch_optimize_queries(self, queries: List[str]) -> List[str]:
        """Optimize multiple queries in batch."""
        optimized_queries = []
        
        for query in queries:
            # Check cache first
            cached = self.get_cached_optimization(query)
            if cached:
                optimized_queries.append(cached)
            else:
                optimized = self.optimize_query(query)
                optimized_queries.append(optimized)
        
        return optimized_queries


# Global instances
cache_manager = CacheManager()
performance_monitor = PerformanceMonitor()
query_optimizer = QueryOptimizer(cache_manager)

# Decorator for caching function results
def cached(ttl: int = 3600):
    """Decorator to cache function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator

# Decorator for performance monitoring
def monitored(operation: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            performance_monitor.start_timer(operation)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance_monitor.end_timer(operation)
        return wrapper
    return decorator

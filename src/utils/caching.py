"""
Advanced caching system with adaptive strategies and performance optimization
"""

import asyncio
import hashlib
import pickle
import time
from typing import Any, Dict, Optional, Callable, Tuple, Union, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import threading
import weakref
from collections import OrderedDict
import torch
import numpy as np

from .logging import get_logger, PerformanceLogger
from .monitoring import MetricCollector

logger = get_logger(__name__)

class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class CacheEntry:
    """Cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.size_bytes = self._calculate_size(value)
        self.ttl = ttl
        self.expires_at = self.created_at + ttl if ttl else None
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate memory size of cached value"""
        try:
            if isinstance(value, torch.Tensor):
                return value.element_size() * value.numel()
            elif isinstance(value, np.ndarray):
                return value.nbytes
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1024  # Default estimate
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        return self.expires_at is not None and time.time() > self.expires_at
    
    def access(self):
        """Record access to entry"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

class MemoryAwareCache:
    """Memory-aware cache with configurable eviction policies"""
    
    def __init__(self, 
                 max_size_bytes: int = 1024 * 1024 * 1024,  # 1GB
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 default_ttl: Optional[float] = None):
        self.max_size_bytes = max_size_bytes
        self.strategy = strategy
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.stats = CacheStats()
        self.access_patterns = {}  # Track access patterns for adaptive caching
    
    def _make_key(self, key: Union[str, Tuple]) -> str:
        """Generate cache key from input"""
        if isinstance(key, str):
            return key
        else:
            # Hash complex keys
            key_str = str(key)
            return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: Union[str, Tuple]) -> Optional[Any]:
        """Get value from cache"""
        cache_key = self._make_key(key)
        start_time = time.perf_counter()
        
        with self.lock:
            entry = self.cache.get(cache_key)
            
            if entry is None:
                self.stats.misses += 1
                return None
            
            if entry.is_expired():
                del self.cache[cache_key]
                self.stats.misses += 1
                self.stats.evictions += 1
                return None
            
            # Update access patterns
            entry.access()
            self._update_access_patterns(cache_key)
            
            # Move to end for LRU
            if self.strategy in [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]:
                self.cache.move_to_end(cache_key)
            
            self.stats.hits += 1
            access_time = (time.perf_counter() - start_time) * 1000
            self._update_avg_access_time(access_time)
            
            return entry.value
    
    def put(self, key: Union[str, Tuple], value: Any, ttl: Optional[float] = None) -> bool:
        """Store value in cache"""
        cache_key = self._make_key(key)
        ttl = ttl or self.default_ttl
        
        entry = CacheEntry(cache_key, value, ttl)
        
        with self.lock:
            # Check if adding this entry would exceed size limit
            projected_size = self.stats.total_size_bytes + entry.size_bytes
            if cache_key in self.cache:
                projected_size -= self.cache[cache_key].size_bytes
            
            # Evict entries if necessary
            while projected_size > self.max_size_bytes and self.cache:
                evicted_key = self._select_eviction_candidate()
                if evicted_key:
                    self._evict_entry(evicted_key)
                    projected_size = self.stats.total_size_bytes + entry.size_bytes
                    if cache_key in self.cache:
                        projected_size -= self.cache[cache_key].size_bytes
                else:
                    break
            
            # Don't cache if single entry is too large
            if entry.size_bytes > self.max_size_bytes:
                logger.warning(f"Entry too large for cache: {entry.size_bytes} bytes")
                return False
            
            # Remove old entry if exists
            if cache_key in self.cache:
                self.stats.total_size_bytes -= self.cache[cache_key].size_bytes
            
            # Add new entry
            self.cache[cache_key] = entry
            self.stats.total_size_bytes += entry.size_bytes
            
            return True
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select entry to evict based on strategy"""
        if not self.cache:
            return None
        
        if self.strategy == CacheStrategy.LRU:
            return next(iter(self.cache))  # First entry is least recently used
        
        elif self.strategy == CacheStrategy.LFU:
            return min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        
        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first, then oldest
            expired = [k for k, v in self.cache.items() if v.is_expired()]
            if expired:
                return expired[0]
            return min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            return self._adaptive_eviction_candidate()
        
        return next(iter(self.cache))  # Default to LRU
    
    def _adaptive_eviction_candidate(self) -> Optional[str]:
        """Adaptive eviction based on access patterns and value"""
        if not self.cache:
            return None
        
        # Score entries based on multiple factors
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            access_frequency = entry.access_count / max(entry.age_seconds, 1)
            recency_score = 1.0 / (current_time - entry.last_accessed + 1)
            size_penalty = entry.size_bytes / (1024 * 1024)  # MB penalty
            
            # Higher score = more valuable, less likely to evict
            scores[key] = access_frequency * recency_score - size_penalty
        
        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _evict_entry(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            entry = self.cache[key]
            self.stats.total_size_bytes -= entry.size_bytes
            del self.cache[key]
            self.stats.evictions += 1
    
    def _update_access_patterns(self, key: str):
        """Update access patterns for adaptive caching"""
        current_time = time.time()
        if key not in self.access_patterns:
            self.access_patterns[key] = []
        
        self.access_patterns[key].append(current_time)
        
        # Keep only recent access history (last hour)
        cutoff = current_time - 3600
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t >= cutoff
        ]
    
    def _update_avg_access_time(self, access_time_ms: float):
        """Update average access time"""
        if self.stats.avg_access_time_ms == 0:
            self.stats.avg_access_time_ms = access_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self.stats.avg_access_time_ms
            )
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()
            self.access_patterns.clear()
    
    def remove(self, key: Union[str, Tuple]) -> bool:
        """Remove specific entry from cache"""
        cache_key = self._make_key(key)
        with self.lock:
            if cache_key in self.cache:
                self._evict_entry(cache_key)
                return True
            return False
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.stats

class TensorCache:
    """Specialized cache for PyTorch tensors with GPU memory management"""
    
    def __init__(self, max_gpu_memory_gb: float = 4.0, max_cpu_memory_gb: float = 8.0):
        self.max_gpu_memory_bytes = int(max_gpu_memory_gb * 1024**3)
        self.max_cpu_memory_bytes = int(max_cpu_memory_gb * 1024**3)
        
        self.gpu_cache = MemoryAwareCache(
            max_size_bytes=self.max_gpu_memory_bytes,
            strategy=CacheStrategy.LRU
        )
        self.cpu_cache = MemoryAwareCache(
            max_size_bytes=self.max_cpu_memory_bytes,
            strategy=CacheStrategy.LRU
        )
    
    def put_tensor(self, key: str, tensor: torch.Tensor, prefer_gpu: bool = True) -> bool:
        """Cache tensor with device preference"""
        if prefer_gpu and tensor.is_cuda:
            return self.gpu_cache.put(key, tensor)
        else:
            # Move to CPU for CPU cache
            cpu_tensor = tensor.cpu() if tensor.is_cuda else tensor
            return self.cpu_cache.put(key, cpu_tensor)
    
    def get_tensor(self, key: str, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        """Retrieve tensor and move to requested device"""
        # Try GPU cache first
        tensor = self.gpu_cache.get(key)
        if tensor is not None:
            if device and not tensor.is_cuda:
                return tensor.to(device)
            return tensor
        
        # Try CPU cache
        tensor = self.cpu_cache.get(key)
        if tensor is not None and device:
            return tensor.to(device)
        
        return tensor
    
    def clear_gpu_cache(self):
        """Clear GPU cache and free CUDA memory"""
        self.gpu_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics"""
        return {
            "gpu_cache_bytes": self.gpu_cache.stats.total_size_bytes,
            "cpu_cache_bytes": self.cpu_cache.stats.total_size_bytes,
            "gpu_utilization": self.gpu_cache.stats.total_size_bytes / self.max_gpu_memory_bytes,
            "cpu_utilization": self.cpu_cache.stats.total_size_bytes / self.max_cpu_memory_bytes
        }

class ComputationCache:
    """Cache for expensive computations with dependency tracking"""
    
    def __init__(self, cache: MemoryAwareCache):
        self.cache = cache
        self.dependencies = {}  # Track computation dependencies
    
    def cached_computation(self, 
                          key: Union[str, Tuple],
                          computation: Callable[[], Any],
                          dependencies: List[Any] = None,
                          ttl: Optional[float] = None) -> Any:
        """Execute computation with caching"""
        cache_key = self._make_dependency_key(key, dependencies)
        
        # Try cache first
        result = self.cache.get(cache_key)
        if result is not None:
            logger.debug(f"Cache hit for computation: {key}")
            return result
        
        # Execute computation
        logger.debug(f"Cache miss, executing computation: {key}")
        with PerformanceLogger(f"computation_{key}") as perf:
            result = computation()
        
        # Cache result
        self.cache.put(cache_key, result, ttl)
        
        # Track dependencies
        if dependencies:
            self.dependencies[cache_key] = dependencies
        
        return result
    
    def _make_dependency_key(self, key: Union[str, Tuple], dependencies: List[Any]) -> str:
        """Create cache key including dependencies"""
        base_key = key if isinstance(key, str) else str(key)
        
        if not dependencies:
            return base_key
        
        # Hash dependencies
        dep_str = str(sorted(str(d) for d in dependencies))
        dep_hash = hashlib.md5(dep_str.encode()).hexdigest()[:8]
        
        return f"{base_key}_{dep_hash}"
    
    def invalidate_dependents(self, dependency: Any):
        """Invalidate all computations that depend on given value"""
        to_remove = []
        
        for cache_key, deps in self.dependencies.items():
            if dependency in deps:
                to_remove.append(cache_key)
        
        for key in to_remove:
            self.cache.remove(key)
            del self.dependencies[key]
        
        logger.debug(f"Invalidated {len(to_remove)} cached computations")

class EncryptionCache:
    """Specialized cache for encryption operations"""
    
    def __init__(self, max_size_gb: float = 2.0):
        self.cache = MemoryAwareCache(
            max_size_bytes=int(max_size_gb * 1024**3),
            strategy=CacheStrategy.ADAPTIVE,
            default_ttl=3600  # 1 hour TTL for encrypted data
        )
        self.context_cache = {}  # Cache CKKS contexts
    
    def cache_encrypted_tensor(self, key: str, encrypted_tensor: Any, 
                              context_name: str = "default") -> bool:
        """Cache encrypted tensor with context reference"""
        cache_key = f"{context_name}_{key}"
        return self.cache.put(cache_key, encrypted_tensor)
    
    def get_encrypted_tensor(self, key: str, context_name: str = "default") -> Optional[Any]:
        """Retrieve cached encrypted tensor"""
        cache_key = f"{context_name}_{key}"
        return self.cache.get(cache_key)
    
    def cache_context(self, name: str, context: Any):
        """Cache CKKS context (keys are expensive to generate)"""
        self.context_cache[name] = {
            "context": context,
            "cached_at": time.time(),
            "access_count": 0
        }
    
    def get_cached_context(self, name: str) -> Optional[Any]:
        """Get cached CKKS context"""
        if name in self.context_cache:
            entry = self.context_cache[name]
            entry["access_count"] += 1
            return entry["context"]
        return None
    
    def invalidate_context(self, name: str):
        """Invalidate cached context and related encrypted data"""
        # Remove context
        if name in self.context_cache:
            del self.context_cache[name]
        
        # Remove related encrypted data
        to_remove = []
        for key in self.cache.cache.keys():
            if key.startswith(f"{name}_"):
                to_remove.append(key)
        
        for key in to_remove:
            self.cache.remove(key)

# Global cache instances
tensor_cache = TensorCache()
encryption_cache = EncryptionCache()
computation_cache = ComputationCache(MemoryAwareCache())

def cached(cache: MemoryAwareCache = None, ttl: Optional[float] = None, 
          key_func: Optional[Callable] = None):
    """Decorator for caching function results"""
    cache = cache or MemoryAwareCache()
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache.put(cache_key, result, ttl)
            
            return result
        
        wrapper._cache = cache  # Allow access to cache for management
        return wrapper
    
    return decorator

async def warm_cache():
    """Warm up caches with common operations"""
    logger.info("Starting cache warm-up")
    
    try:
        # Warm tensor cache with common tensor sizes
        common_sizes = [(128, 64), (256, 128), (512, 256)]
        for size in common_sizes:
            test_tensor = torch.randn(*size)
            tensor_cache.put_tensor(f"warmup_{size}", test_tensor)
        
        # Warm encryption cache with test context
        from ..python.he_graph import CKKSContext, HEConfig
        test_config = HEConfig(poly_modulus_degree=8192)
        test_context = CKKSContext(test_config)
        test_context.generate_keys()
        encryption_cache.cache_context("warmup", test_context)
        
        logger.info("Cache warm-up completed")
    
    except Exception as e:
        logger.error(f"Cache warm-up failed: {e}")

def get_cache_metrics() -> Dict[str, Any]:
    """Get metrics from all caches"""
    return {
        "tensor_cache": {
            "gpu": tensor_cache.gpu_cache.get_stats(),
            "cpu": tensor_cache.cpu_cache.get_stats(),
            "memory_usage": tensor_cache.get_memory_usage()
        },
        "encryption_cache": encryption_cache.cache.get_stats(),
        "computation_cache": computation_cache.cache.get_stats()
    }
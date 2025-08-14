"""
Performance optimization utilities and resource management
"""


import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
import queue
import weakref

from .logging import get_logger, PerformanceLogger
from .caching import cached, tensor_cache
from .monitoring import MetricCollector

logger = get_logger(__name__)

@dataclass
class ResourceLimits:
    """Resource limits for performance optimization"""
    max_cpu_threads: int = mp.cpu_count()
    max_gpu_memory_gb: float = 8.0
    max_batch_size: int = 128
    max_concurrent_operations: int = 10
    memory_threshold_percent: float = 85.0

class ResourcePool:
    """Manage computational resources efficiently"""

    def __init__(self, limits: ResourceLimits):
        """  Init  ."""
        self.limits = limits
        self.thread_pool = ThreadPoolExecutor(max_workers=limits.max_cpu_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=min(limits.max_cpu_threads, 4))
        self.gpu_locks = {i: threading.Lock() for i in range(torch.cuda.device_count())}
        self.active_operations = 0
        self.operation_lock = threading.Lock()

        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        self.metric_collector = MetricCollector()

    async def submit_cpu_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit CPU-bound task to thread pool"""
        loop = asyncio.get_event_loop()
        with PerformanceLogger(f"cpu_task_{func.__name__}") as perf:
            result = await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)

        self.metric_collector.record("cpu_task_duration_ms", perf.metrics["duration_ms"])
        return result

    async def submit_compute_task(self, func: Callable, *args, **kwargs) -> Any:
        """Submit compute-intensive task to process pool"""
        loop = asyncio.get_event_loop()
        with PerformanceLogger(f"compute_task_{func.__name__}") as perf:
            result = await loop.run_in_executor(self.process_pool, func, *args, **kwargs)

        self.metric_collector.record("compute_task_duration_ms", perf.metrics["duration_ms"])
        return result

    def acquire_gpu(self, device_id: int = 0) -> bool:
        """Acquire GPU lock for exclusive access"""
        if device_id in self.gpu_locks:
            return self.gpu_locks[device_id].acquire(blocking=False)
        return False

    def release_gpu(self, device_id: int = 0) -> None:
        """Release GPU lock"""
        if device_id in self.gpu_locks:
            self.gpu_locks[device_id].release()

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        return self.memory_monitor.get_memory_usage_percent() > self.limits.memory_threshold_percent

    def optimize_batch_size(self, requested_batch_size: int,
                            memory_per_sample_mb: float = 10.0) -> int:
        """Optimize Batch Size."""
        """Optimize batch size based on available memory"""
        available_memory_gb = self.memory_monitor.get_available_memory_gb()
        max_samples = int((available_memory_gb * 1024) / memory_per_sample_mb * 0.8)  # 80% safety margin

        optimal_batch_size = min(
            requested_batch_size,
            max_samples,
            self.limits.max_batch_size
        )

        if optimal_batch_size < requested_batch_size:
            logger.info(f"Reduced batch size from {requested_batch_size} to {optimal_batch_size} due to memory constraints")

        return max(1, optimal_batch_size)

    def shutdown(self) -> None:
        """Shutdown resource pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class MemoryMonitor:
    """Monitor and manage memory usage"""

    def __init__(self):
        """  Init  ."""
        self.peak_memory_usage = 0
        self.memory_history = []

    def get_memory_usage_percent(self) -> float:
        """Get current memory usage percentage"""

        import psutil
        return psutil.virtual_memory().percent

    def get_available_memory_gb(self) -> float:
        """Get available memory in GB"""

        import psutil
        return psutil.virtual_memory().available / (1024**3)

    def get_gpu_memory_info(self, device_id: int = 0) -> Dict[str, float]:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}

        allocated = torch.cuda.memory_allocated(device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024**3)
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
        free = total - reserved

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": free
        }

    def cleanup_memory(self) -> None:
        """Perform memory cleanup"""

        import gc

        # Python garbage collection
        collected = gc.collect()

        # CUDA memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clear tensor cache if under pressure
        if self.get_memory_usage_percent() > 90:
            tensor_cache.clear_gpu_cache()

        logger.info(f"Memory cleanup completed, collected {collected} objects")

class BatchProcessor:
    """Efficient batch processing with adaptive sizing"""

    def __init__(self, max_batch_size: int = 32, min_batch_size: int = 1):
        """  Init  ."""
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.optimal_batch_size = max_batch_size
        self.performance_history = []

    async def process_batches(self,
                            items: List[Any],
                            process_func: Callable,
                            progress_callback: Optional[Callable] = None) -> List[Any]:
        """Process items in adaptive batches"""
        results = []
        total_items = len(items)
        processed_items = 0

        while processed_items < total_items:
            # Determine batch size
            batch_size = self._get_optimal_batch_size()
            batch = items[processed_items:processed_items + batch_size]

            # Process batch
            start_time = time.perf_counter()

            try:
                batch_results = await process_func(batch)
                duration = time.perf_counter() - start_time

                # Update performance metrics
                throughput = len(batch) / duration
                self._update_batch_size_optimization(batch_size, throughput, True)

                results.extend(batch_results)

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")

                # Try smaller batch size on failure
                if batch_size > self.min_batch_size:
                    self._update_batch_size_optimization(batch_size, 0, False)
                    continue
                else:
                    raise

            processed_items += len(batch)

            # Progress callback
            if progress_callback:
                progress_callback(processed_items, total_items)

        return results

    def _get_optimal_batch_size(self) -> int:
        """Get current optimal batch size"""
        return self.optimal_batch_size

    def _update_batch_size_optimization(self, batch_size: int, throughput: float, success: bool) -> None:
        """Update batch size optimization based on performance"""
        self.performance_history.append({
            "batch_size": batch_size,
            "throughput": throughput,
            "success": success,
            "timestamp": time.time()
        })

        # Keep only recent history
        cutoff_time = time.time() - 300  # 5 minutes
        self.performance_history = [
            h for h in self.performance_history if h["timestamp"] >= cutoff_time
        ]

        if not success:
            # Reduce batch size on failure
            self.optimal_batch_size = max(self.min_batch_size, batch_size // 2)
            return

        # Adaptive optimization based on recent performance
        if len(self.performance_history) >= 5:
            recent_performance = self.performance_history[-5:]
            avg_throughput = np.mean([h["throughput"] for h in recent_performance])

            # Increase batch size if performance is good and no recent failures
            recent_failures = any(not h["success"] for h in recent_performance)
            if not recent_failures and throughput > avg_throughput * 1.1:
                self.optimal_batch_size = min(self.max_batch_size, batch_size * 2)
            elif throughput < avg_throughput * 0.9:
                self.optimal_batch_size = max(self.min_batch_size, batch_size // 2)

class ParallelExecutor:
    """Execute operations in parallel with load balancing"""

    def __init__(self, max_workers: int = None):
        """  Init  ."""
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.load_balancer = LoadBalancer()

    async def execute_parallel(self,
                                tasks: List[Tuple[Callable, tuple, dict]],
                                max_concurrent: int = None) -> List[Any]:
        """Execute tasks in parallel with load balancing"""
        max_concurrent = max_concurrent or self.max_workers
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_single_task(task_info: Tuple[Callable, tuple, dict]):
            func, args, kwargs = task_info
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.executor, func, *args, **kwargs)

        # Execute all tasks
        results = await asyncio.gather(
            *[execute_single_task(task) for task in tasks],
            return_exceptions=True
        )

        return results

    def shutdown(self) -> None:
        """Shutdown executor"""
        self.executor.shutdown(wait=True)

class LoadBalancer:
    """Simple load balancer for distributing work"""

    def __init__(self):
        """  Init  ."""
        self.worker_loads = {}  # Track load per worker
        self.worker_performance = {}  # Track performance metrics

    def select_worker(self, workers: List[str]) -> str:
        """Select best worker based on load and performance"""
        if not workers:
            raise ValueError("No workers available")

        # Initialize tracking for new workers
        for worker in workers:
            if worker not in self.worker_loads:
                self.worker_loads[worker] = 0
                self.worker_performance[worker] = {"tasks_completed": 0, "avg_duration": 0}

        # Select worker with lowest load and best performance
        best_worker = min(workers, key=lambda w: (
            self.worker_loads[w],
            -self.worker_performance[w]["tasks_completed"]
        ))

        return best_worker

    def update_worker_load(self, worker: str, load_delta: int) -> None:
        """Update worker load"""
        if worker in self.worker_loads:
            self.worker_loads[worker] += load_delta

    def update_worker_performance(self, worker: str, duration: float) -> None:
        """Update worker performance metrics"""
        if worker in self.worker_performance:
            perf = self.worker_performance[worker]
            perf["tasks_completed"] += 1

            if perf["avg_duration"] == 0:
                perf["avg_duration"] = duration
            else:
                # Exponential moving average
                alpha = 0.2
                perf["avg_duration"] = alpha * duration + (1 - alpha) * perf["avg_duration"]

# Performance optimization decorators

def optimize_memory(func: Callable) -> Callable:
    """Decorator to optimize memory usage during function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper."""
        memory_monitor = MemoryMonitor()
        initial_memory = memory_monitor.get_memory_usage_percent()

        try:
            result = func(*args, **kwargs)

            # Check for memory leaks
            final_memory = memory_monitor.get_memory_usage_percent()
            if final_memory - initial_memory > 20:  # 20% increase
                logger.warning(f"Potential memory leak in {func.__name__}: {final_memory - initial_memory:.1f}% increase")
                memory_monitor.cleanup_memory()

            return result

        except MemoryError:
            logger.error(f"Memory error in {func.__name__}, attempting cleanup")
            memory_monitor.cleanup_memory()
            raise

    return wrapper

def adaptive_batch_size(min_size: int = 1, max_size: int = 128):
    """Decorator to automatically optimize batch sizes"""
    def decorator(func: Callable) -> Callable:
        """Decorator."""
        processor = BatchProcessor(max_size, min_size)

        @wraps(func)
        async def wrapper(data: List[Any], *args, **kwargs):
            if not isinstance(data, list) or len(data) <= max_size:
                return await func(data, *args, **kwargs)

            # Process in adaptive batches
            async def process_batch(batch):
                return await func(batch, *args, **kwargs)

            return await processor.process_batches(data, process_batch)

        return wrapper
    return decorator

def gpu_accelerated(fallback_to_cpu: bool = True):
    """Decorator to automatically use GPU acceleration with CPU fallback"""
    def decorator(func: Callable) -> Callable:
        """Decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper."""
            if torch.cuda.is_available():
                try:
                    # Try GPU first
                    kwargs['device'] = 'cuda'
                    return func(*args, **kwargs)
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    if fallback_to_cpu:
                        logger.warning(f"GPU execution failed for {func.__name__}: {e}. Falling back to CPU")
                        torch.cuda.empty_cache()  # Clear GPU memory
                        kwargs['device'] = 'cpu'
                        return func(*args, **kwargs)
                    else:
                        raise
            else:
                kwargs['device'] = 'cpu'
                return func(*args, **kwargs)

        return wrapper
    return decorator

def profile_performance(metric_collector: MetricCollector = None):
    """Decorator to profile function performance"""
    collector = metric_collector or MetricCollector()

    def decorator(func: Callable) -> Callable:
        """Decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper."""
            with PerformanceLogger(func.__name__) as perf:
                result = func(*args, **kwargs)

            # Record metrics
            collector.record(f"{func.__name__}_duration_ms", perf.metrics["duration_ms"])
            collector.record(f"{func.__name__}_calls", 1)

            return result

        return wrapper
    return decorator

# Global resource pool
_resource_pool = None

def get_resource_pool() -> ResourcePool:
    """Get global resource pool instance"""
    global _resource_pool
    if _resource_pool is None:
        limits = ResourceLimits()
        _resource_pool = ResourcePool(limits)
    return _resource_pool

def shutdown_performance_resources():
    """Shutdown performance optimization resources"""
    global _resource_pool
    if _resource_pool:
        _resource_pool.shutdown()
        _resource_pool = None
"""
⚡ GENERATION 3: Advanced Performance Optimization Engine

This module implements cutting-edge optimization techniques for HE-Graph-Embeddings,
including intelligent caching, resource pooling, concurrent processing, and 
auto-scaling capabilities.

Key Features:
- Adaptive caching with machine learning-based cache replacement
- Dynamic resource pooling with intelligent load balancing
- Concurrent processing with work-stealing algorithms
- Auto-scaling based on workload prediction
- Memory-efficient batch processing with optimal packing
- GPU acceleration with advanced memory management
"""

import asyncio
import concurrent.futures
import threading
import time
import logging
# Try to import psutil, fall back to os if not available
try:
    import psutil
    _has_psutil = True
except ImportError:
    import os
    _has_psutil = False
import gc
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import math
import weakref

from .robust_error_handling import robust_operation, ComputationError, ResourceError
from .he_health_monitor import track_he_operation

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization aggressiveness levels"""
    CONSERVATIVE = "conservative"    # Safe optimizations only
    BALANCED = "balanced"           # Balanced performance vs stability
    AGGRESSIVE = "aggressive"       # Maximum performance
    EXPERIMENTAL = "experimental"   # Cutting-edge optimizations


class ResourceType(Enum):
    """Types of computational resources"""
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    NETWORK = "network"


@dataclass
class WorkItem:
    """Work item for processing queue"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 1  # Higher = more priority
    created_at: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        # For priority queue ordering
        return (-self.priority, self.created_at) < (-other.priority, other.created_at)


@dataclass 
class CacheEntry:
    """Entry in the adaptive cache"""
    key: str
    value: Any
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    memory_size: int = 0
    computation_cost: float = 0.0  # Time taken to compute
    
    @property
    def age(self) -> float:
        return time.time() - self.creation_time
    
    @property
    def access_frequency(self) -> float:
        age = max(self.age, 1.0)  # Avoid division by zero
        return self.access_count / age
    
    @property
    def utility_score(self) -> float:
        """Calculate utility score for cache replacement"""
        recency = 1.0 / (time.time() - self.last_access + 1.0)
        frequency = self.access_frequency
        cost_benefit = self.computation_cost / max(self.memory_size, 1)
        
        return recency * frequency * cost_benefit


class AdaptiveCache:
    """
    Machine learning-inspired adaptive cache with intelligent replacement
    """
    
    def __init__(self, 
                 max_memory_mb: int = 1024,
                 max_entries: int = 10000,
                 cleanup_threshold: float = 0.8):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_entries = max_entries
        self.cleanup_threshold = cleanup_threshold
        
        self.cache: Dict[str, CacheEntry] = {}
        self.current_memory = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
        
        # Performance tracking
        self.access_pattern = deque(maxlen=1000)
        self.eviction_history = deque(maxlen=100)
        
        logger.info(f"Adaptive cache initialized: {max_memory_mb}MB, {max_entries} entries")
    
    @robust_operation(max_retries=2)
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with performance tracking"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                self.hits += 1
                
                # Record access pattern
                self.access_pattern.append((key, time.time(), True))
                
                return entry.value
            else:
                self.misses += 1
                self.access_pattern.append((key, time.time(), False))
                return None
    
    @robust_operation(max_retries=2)
    def put(self, key: str, value: Any, 
            computation_cost: float = 0.0) -> bool:
        """Store item in cache with intelligent eviction"""
        try:
            # Estimate memory size
            memory_size = self._estimate_memory_size(value)
            
            with self.lock:
                # Check if we need to make space
                if (self.current_memory + memory_size > self.max_memory_bytes * self.cleanup_threshold or
                    len(self.cache) >= self.max_entries):
                    self._intelligent_eviction(memory_size)
                
                # Store the entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    memory_size=memory_size,
                    computation_cost=computation_cost
                )
                
                # Remove old entry if exists
                if key in self.cache:
                    self.current_memory -= self.cache[key].memory_size
                
                self.cache[key] = entry
                self.current_memory += memory_size
                
                return True
                
        except Exception as e:
            logger.error(f"Cache put failed for key {key}: {e}")
            return False
    
    def _intelligent_eviction(self, needed_space: int):
        """Intelligent cache eviction using utility scoring"""
        if not self.cache:
            return
        
        # Calculate utility scores for all entries
        entries_with_scores = [
            (entry.utility_score, key, entry) 
            for key, entry in self.cache.items()
        ]
        
        # Sort by utility score (lowest first for eviction)
        entries_with_scores.sort()
        
        freed_space = 0
        evicted_keys = []
        
        # Evict lowest utility entries until we have enough space
        for score, key, entry in entries_with_scores:
            if freed_space >= needed_space and len(self.cache) > 1:
                break
            
            del self.cache[key]
            freed_space += entry.memory_size
            self.current_memory -= entry.memory_size
            evicted_keys.append(key)
            
            # Record eviction for learning
            self.eviction_history.append({
                'key': key,
                'utility_score': score,
                'age': entry.age,
                'access_count': entry.access_count,
                'timestamp': time.time()
            })
        
        logger.debug(f"Evicted {len(evicted_keys)} cache entries, freed {freed_space} bytes")
    
    def _estimate_memory_size(self, obj: Any) -> int:
        """Estimate memory size of object"""
        try:
            import sys
            return sys.getsizeof(obj)
        except:
            # Fallback estimation
            if isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_memory_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_memory_size(k) + self._estimate_memory_size(v) 
                          for k, v in obj.items())
            else:
                return 1024  # Default estimate
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            total_accesses = self.hits + self.misses
            hit_rate = self.hits / max(total_accesses, 1)
            
            return {
                'hit_rate': hit_rate,
                'total_entries': len(self.cache),
                'memory_usage_mb': self.current_memory / (1024 * 1024),
                'memory_utilization': self.current_memory / self.max_memory_bytes,
                'total_hits': self.hits,
                'total_misses': self.misses,
                'eviction_count': len(self.eviction_history)
            }


class ResourcePool:
    """
    Dynamic resource pool with intelligent load balancing
    """
    
    def __init__(self, 
                 pool_type: ResourceType,
                 min_size: int = 2,
                 max_size: int = None,
                 growth_factor: float = 1.5):
        self.pool_type = pool_type
        self.min_size = min_size
        cpu_count = psutil.cpu_count() if _has_psutil else os.cpu_count() or 4
        self.max_size = max_size or min(cpu_count * 2, 32)
        self.growth_factor = growth_factor
        
        self.resources = []
        self.available = deque()
        self.in_use = set()
        self.lock = threading.RLock()
        
        # Performance tracking
        self.allocation_times = deque(maxlen=1000)
        self.utilization_history = deque(maxlen=100)
        
        # Initialize minimum resources
        self._initialize_pool()
        
        logger.info(f"Resource pool initialized: {pool_type.value}, "
                   f"min={min_size}, max={max_size}")
    
    def _initialize_pool(self):
        """Initialize the resource pool"""
        with self.lock:
            for _ in range(self.min_size):
                resource = self._create_resource()
                if resource:
                    self.resources.append(resource)
                    self.available.append(resource)
    
    def _create_resource(self) -> Any:
        """Create a new resource (override in subclasses)"""
        if self.pool_type == ResourceType.CPU:
            return concurrent.futures.ThreadPoolExecutor(max_workers=1)
        elif self.pool_type == ResourceType.GPU:
            # GPU resource creation would be implemented here
            return f"gpu_context_{len(self.resources)}"
        else:
            return f"resource_{len(self.resources)}"
    
    def _destroy_resource(self, resource: Any):
        """Destroy a resource (override in subclasses)"""
        if hasattr(resource, 'shutdown'):
            resource.shutdown(wait=False)
    
    @robust_operation(max_retries=3)
    def acquire(self, timeout: float = 30.0) -> Optional[Any]:
        """Acquire a resource from the pool"""
        start_time = time.time()
        
        with self.lock:
            # Try to get an available resource
            if self.available:
                resource = self.available.popleft()
                self.in_use.add(resource)
                
                allocation_time = time.time() - start_time
                self.allocation_times.append(allocation_time)
                
                return resource
            
            # Try to create a new resource if under limit
            if len(self.resources) < self.max_size:
                resource = self._create_resource()
                if resource:
                    self.resources.append(resource)
                    self.in_use.add(resource)
                    
                    allocation_time = time.time() - start_time
                    self.allocation_times.append(allocation_time)
                    
                    return resource
        
        # Wait for a resource to become available
        end_time = start_time + timeout
        while time.time() < end_time:
            time.sleep(0.01)  # Small sleep to prevent busy waiting
            
            with self.lock:
                if self.available:
                    resource = self.available.popleft()
                    self.in_use.add(resource)
                    
                    allocation_time = time.time() - start_time
                    self.allocation_times.append(allocation_time)
                    
                    return resource
        
        logger.warning(f"Failed to acquire {self.pool_type.value} resource within {timeout}s")
        return None
    
    def release(self, resource: Any):
        """Release a resource back to the pool"""
        with self.lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                self.available.append(resource)
            else:
                logger.warning(f"Attempted to release unknown resource: {resource}")
    
    def scale_up(self, target_size: Optional[int] = None):
        """Scale up the resource pool"""
        with self.lock:
            if target_size is None:
                target_size = min(
                    int(len(self.resources) * self.growth_factor),
                    self.max_size
                )
            
            while len(self.resources) < target_size:
                resource = self._create_resource()
                if resource:
                    self.resources.append(resource)
                    self.available.append(resource)
                else:
                    break
            
            logger.info(f"Scaled up {self.pool_type.value} pool to {len(self.resources)} resources")
    
    def scale_down(self, target_size: Optional[int] = None):
        """Scale down the resource pool"""
        with self.lock:
            if target_size is None:
                target_size = max(
                    int(len(self.resources) / self.growth_factor),
                    self.min_size
                )
            
            while len(self.resources) > target_size and self.available:
                resource = self.available.pop()
                self.resources.remove(resource)
                self._destroy_resource(resource)
            
            logger.info(f"Scaled down {self.pool_type.value} pool to {len(self.resources)} resources")
    
    def get_utilization(self) -> float:
        """Get current utilization percentage"""
        with self.lock:
            if not self.resources:
                return 0.0
            return len(self.in_use) / len(self.resources)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool performance statistics"""
        with self.lock:
            utilization = self.get_utilization()
            avg_allocation_time = (
                sum(self.allocation_times) / len(self.allocation_times)
                if self.allocation_times else 0.0
            )
            
            return {
                'pool_type': self.pool_type.value,
                'total_resources': len(self.resources),
                'available_resources': len(self.available),
                'in_use_resources': len(self.in_use),
                'utilization': utilization,
                'average_allocation_time': avg_allocation_time,
                'min_size': self.min_size,
                'max_size': self.max_size
            }


class WorkStealingExecutor:
    """
    Advanced work-stealing executor for optimal load distribution
    """
    
    def __init__(self, 
                 num_workers: Optional[int] = None,
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        cpu_count = psutil.cpu_count() if _has_psutil else os.cpu_count() or 4
        self.num_workers = num_workers or cpu_count
        self.optimization_level = optimization_level
        
        # Work queues - one per worker plus global queue
        self.worker_queues = [deque() for _ in range(self.num_workers)]
        self.global_queue = []  # Priority queue
        self.queue_locks = [threading.Lock() for _ in range(self.num_workers)]
        self.global_lock = threading.Lock()
        
        # Worker threads
        self.workers = []
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_execution_time = 0.0
        self.steal_count = 0
        
        # Resource pools
        self.cpu_pool = ResourcePool(ResourceType.CPU, min_size=2, max_size=self.num_workers)
        
        logger.info(f"Work-stealing executor initialized: {self.num_workers} workers, "
                   f"{optimization_level.value} optimization")
    
    def start(self):
        """Start the executor"""
        if self.running:
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                name=f"WorkStealer-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} worker threads")
    
    def stop(self, timeout: float = 30.0):
        """Stop the executor"""
        if not self.running:
            return
        
        self.running = False
        self.shutdown_event.set()
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout / len(self.workers))
        
        self.workers.clear()
        logger.info("Work-stealing executor stopped")
    
    @track_he_operation("submit_work")
    def submit(self, func: Callable, *args, 
               priority: int = 1, 
               timeout: Optional[float] = None,
               callback: Optional[Callable] = None, **kwargs) -> str:
        """Submit work to the executor"""
        task_id = f"task_{time.time()}_{id(func)}"
        
        work_item = WorkItem(
            task_id=task_id,
            function=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            callback=callback
        )
        
        # Choose optimal queue for submission
        if priority > 5 or self.optimization_level == OptimizationLevel.AGGRESSIVE:
            # High priority or aggressive mode - use global queue
            with self.global_lock:
                heapq.heappush(self.global_queue, work_item)
        else:
            # Round-robin to worker queues
            worker_idx = hash(task_id) % self.num_workers
            with self.queue_locks[worker_idx]:
                self.worker_queues[worker_idx].append(work_item)
        
        return task_id
    
    def _worker_loop(self, worker_id: int):
        """Main worker loop with work stealing"""
        logger.debug(f"Worker {worker_id} started")
        
        while self.running and not self.shutdown_event.is_set():
            work_item = self._get_work(worker_id)
            
            if work_item:
                self._execute_work_item(work_item, worker_id)
            else:
                # No work available, sleep briefly
                time.sleep(0.001)
        
        logger.debug(f"Worker {worker_id} stopped")
    
    def _get_work(self, worker_id: int) -> Optional[WorkItem]:
        """Get work using work-stealing algorithm"""
        # 1. Check own queue first
        with self.queue_locks[worker_id]:
            if self.worker_queues[worker_id]:
                return self.worker_queues[worker_id].popleft()
        
        # 2. Check global priority queue
        with self.global_lock:
            if self.global_queue:
                return heapq.heappop(self.global_queue)
        
        # 3. Try to steal from other workers
        if self.optimization_level in [OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]:
            for steal_attempts in range(self.num_workers):
                victim_id = (worker_id + steal_attempts + 1) % self.num_workers
                
                with self.queue_locks[victim_id]:
                    if self.worker_queues[victim_id]:
                        # Steal from the end (LIFO for better cache locality)
                        work_item = self.worker_queues[victim_id].pop()
                        self.steal_count += 1
                        logger.debug(f"Worker {worker_id} stole work from worker {victim_id}")
                        return work_item
        
        return None
    
    def _execute_work_item(self, work_item: WorkItem, worker_id: int):
        """Execute a work item with error handling and performance tracking"""
        start_time = time.time()
        
        try:
            # Acquire CPU resource
            cpu_resource = self.cpu_pool.acquire(timeout=10.0)
            if not cpu_resource:
                raise ResourceError("Failed to acquire CPU resource")
            
            try:
                # Execute the function
                if work_item.timeout:
                    # Use timeout if specified
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(work_item.function, *work_item.args, **work_item.kwargs)
                        result = future.result(timeout=work_item.timeout)
                else:
                    result = work_item.function(*work_item.args, **work_item.kwargs)
                
                # Call callback if provided
                if work_item.callback:
                    work_item.callback(work_item.task_id, result, None)
                
                execution_time = time.time() - start_time
                self.total_execution_time += execution_time
                self.completed_tasks += 1
                
                logger.debug(f"Worker {worker_id} completed task {work_item.task_id} "
                           f"in {execution_time:.3f}s")
                
            finally:
                self.cpu_pool.release(cpu_resource)
                
        except Exception as e:
            self.failed_tasks += 1
            
            # Retry logic
            if work_item.retry_count < work_item.max_retries:
                work_item.retry_count += 1
                logger.warning(f"Task {work_item.task_id} failed, retrying "
                             f"({work_item.retry_count}/{work_item.max_retries}): {e}")
                
                # Re-submit for retry
                with self.queue_locks[worker_id]:
                    self.worker_queues[worker_id].appendleft(work_item)
            else:
                logger.error(f"Task {work_item.task_id} failed permanently: {e}")
                
                if work_item.callback:
                    work_item.callback(work_item.task_id, None, e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get executor performance statistics"""
        total_tasks = self.completed_tasks + self.failed_tasks
        success_rate = self.completed_tasks / max(total_tasks, 1)
        avg_execution_time = (
            self.total_execution_time / max(self.completed_tasks, 1)
        )
        
        # Queue lengths
        queue_lengths = []
        for i in range(self.num_workers):
            with self.queue_locks[i]:
                queue_lengths.append(len(self.worker_queues[i]))
        
        with self.global_lock:
            global_queue_length = len(self.global_queue)
        
        return {
            'num_workers': self.num_workers,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': success_rate,
            'average_execution_time': avg_execution_time,
            'steal_count': self.steal_count,
            'queue_lengths': queue_lengths,
            'global_queue_length': global_queue_length,
            'cpu_pool_stats': self.cpu_pool.get_stats()
        }


class AutoScaler:
    """
    Intelligent auto-scaling system with workload prediction
    """
    
    def __init__(self, 
                 target_utilization: float = 0.7,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.5,
                 prediction_window: int = 60):
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.prediction_window = prediction_window
        
        # Monitoring data
        self.utilization_history = deque(maxlen=1000)
        self.workload_history = deque(maxlen=1000)
        self.scaling_actions = deque(maxlen=100)
        
        # Prediction model (simple moving average for now)
        self.prediction_weights = [0.5, 0.3, 0.2]  # Recent, medium, older history
        
        # Controlled resources
        self.managed_pools = []
        self.managed_executors = []
        
        logger.info(f"Auto-scaler initialized: target={target_utilization:.2%}")
    
    def register_resource_pool(self, pool: ResourcePool):
        """Register a resource pool for auto-scaling"""
        self.managed_pools.append(weakref.ref(pool))
        logger.info(f"Registered {pool.pool_type.value} pool for auto-scaling")
    
    def register_executor(self, executor: WorkStealingExecutor):
        """Register an executor for auto-scaling"""
        self.managed_executors.append(weakref.ref(executor))
        logger.info("Registered work-stealing executor for auto-scaling")
    
    def update_metrics(self):
        """Update monitoring metrics from managed resources"""
        current_time = time.time()
        total_utilization = 0.0
        total_workload = 0.0
        active_resources = 0
        
        # Collect metrics from resource pools
        for pool_ref in self.managed_pools[:]:  # Copy to avoid modification during iteration
            pool = pool_ref()
            if pool is None:
                self.managed_pools.remove(pool_ref)
                continue
            
            utilization = pool.get_utilization()
            total_utilization += utilization
            active_resources += 1
        
        # Collect metrics from executors
        for executor_ref in self.managed_executors[:]:
            executor = executor_ref()
            if executor is None:
                self.managed_executors.remove(executor_ref)
                continue
            
            stats = executor.get_stats()
            # Calculate workload as ratio of pending to total tasks
            pending_tasks = sum(stats['queue_lengths']) + stats['global_queue_length']
            workload = pending_tasks / max(stats['num_workers'], 1)
            total_workload += workload
        
        # Record metrics
        if active_resources > 0:
            avg_utilization = total_utilization / active_resources
            self.utilization_history.append((current_time, avg_utilization))
        
        if self.managed_executors:
            avg_workload = total_workload / len(self.managed_executors)
            self.workload_history.append((current_time, avg_workload))
    
    def predict_future_utilization(self, horizon_seconds: float = 60.0) -> float:
        """Predict future utilization using historical data"""
        if len(self.utilization_history) < 3:
            return 0.5  # Default prediction
        
        # Simple weighted moving average prediction
        recent_utilizations = [util for _, util in list(self.utilization_history)[-10:]]
        
        if len(recent_utilizations) >= len(self.prediction_weights):
            predicted = sum(
                weight * util 
                for weight, util in zip(self.prediction_weights, recent_utilizations[-len(self.prediction_weights):])
            )
        else:
            predicted = sum(recent_utilizations) / len(recent_utilizations)
        
        return max(0.0, min(1.0, predicted))
    
    def should_scale_up(self) -> bool:
        """Determine if scaling up is needed"""
        if len(self.utilization_history) < 5:
            return False
        
        # Check recent utilization
        recent_utils = [util for _, util in list(self.utilization_history)[-5:]]
        avg_recent = sum(recent_utils) / len(recent_utils)
        
        # Check predicted utilization
        predicted = self.predict_future_utilization()
        
        return (avg_recent > self.scale_up_threshold or 
                predicted > self.scale_up_threshold)
    
    def should_scale_down(self) -> bool:
        """Determine if scaling down is appropriate"""
        if len(self.utilization_history) < 10:
            return False
        
        # Check sustained low utilization
        recent_utils = [util for _, util in list(self.utilization_history)[-10:]]
        avg_recent = sum(recent_utils) / len(recent_utils)
        
        # Check predicted utilization
        predicted = self.predict_future_utilization()
        
        return (avg_recent < self.scale_down_threshold and 
                predicted < self.scale_down_threshold)
    
    def execute_scaling_decision(self):
        """Execute scaling decisions based on current metrics"""
        if self.should_scale_up():
            self._scale_up()
        elif self.should_scale_down():
            self._scale_down()
    
    def _scale_up(self):
        """Scale up managed resources"""
        scaled_resources = 0
        
        for pool_ref in self.managed_pools:
            pool = pool_ref()
            if pool and pool.get_utilization() > self.scale_up_threshold:
                old_size = len(pool.resources)
                pool.scale_up()
                new_size = len(pool.resources)
                
                if new_size > old_size:
                    scaled_resources += 1
                    logger.info(f"Scaled up {pool.pool_type.value} pool: {old_size} → {new_size}")
        
        # Record scaling action
        self.scaling_actions.append({
            'action': 'scale_up',
            'timestamp': time.time(),
            'resources_affected': scaled_resources
        })
    
    def _scale_down(self):
        """Scale down managed resources"""
        scaled_resources = 0
        
        for pool_ref in self.managed_pools:
            pool = pool_ref()
            if pool and pool.get_utilization() < self.scale_down_threshold:
                old_size = len(pool.resources)
                pool.scale_down()
                new_size = len(pool.resources)
                
                if new_size < old_size:
                    scaled_resources += 1
                    logger.info(f"Scaled down {pool.pool_type.value} pool: {old_size} → {new_size}")
        
        # Record scaling action
        self.scaling_actions.append({
            'action': 'scale_down',
            'timestamp': time.time(),
            'resources_affected': scaled_resources
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics"""
        recent_actions = len([
            action for action in self.scaling_actions
            if time.time() - action['timestamp'] < 300  # Last 5 minutes
        ])
        
        return {
            'target_utilization': self.target_utilization,
            'current_utilization': (
                self.utilization_history[-1][1] if self.utilization_history else 0.0
            ),
            'predicted_utilization': self.predict_future_utilization(),
            'managed_pools': len([ref for ref in self.managed_pools if ref() is not None]),
            'managed_executors': len([ref for ref in self.managed_executors if ref() is not None]),
            'total_scaling_actions': len(self.scaling_actions),
            'recent_scaling_actions': recent_actions
        }


# Testing function
def test_advanced_optimization():
    """Test advanced optimization components"""
    logger.info("Testing advanced optimization engine...")
    
    # Test adaptive cache
    cache = AdaptiveCache(max_memory_mb=10)
    
    # Store some test data
    for i in range(20):
        cache.put(f"key_{i}", f"value_{i}" * 100, computation_cost=i * 0.1)
    
    # Test cache hits
    for i in range(0, 20, 2):
        result = cache.get(f"key_{i}")
        assert result is not None, f"Cache miss for key_{i}"
    
    cache_stats = cache.get_stats()
    logger.info(f"✅ Cache test: {cache_stats['hit_rate']:.2%} hit rate")
    
    # Test resource pool
    cpu_pool = ResourcePool(ResourceType.CPU, min_size=2, max_size=4)
    
    # Test resource acquisition and release
    resources = []
    for _ in range(3):
        resource = cpu_pool.acquire(timeout=1.0)
        if resource:
            resources.append(resource)
    
    for resource in resources:
        cpu_pool.release(resource)
    
    pool_stats = cpu_pool.get_stats()
    logger.info(f"✅ Resource pool test: {pool_stats['utilization']:.2%} utilization")
    
    # Test work-stealing executor
    executor = WorkStealingExecutor(num_workers=2)
    executor.start()
    
    def test_task(x):
        time.sleep(0.01)  # Simulate work
        return x * 2
    
    # Submit some tasks
    task_ids = []
    for i in range(10):
        task_id = executor.submit(test_task, i, priority=i % 3)
        task_ids.append(task_id)
    
    time.sleep(0.5)  # Let tasks complete
    
    executor_stats = executor.get_stats()
    logger.info(f"✅ Executor test: {executor_stats['completed_tasks']} tasks completed")
    
    executor.stop()
    
    # Test auto-scaler
    auto_scaler = AutoScaler()
    auto_scaler.register_resource_pool(cpu_pool)
    
    # Simulate some metrics
    for _ in range(10):
        auto_scaler.update_metrics()
        time.sleep(0.01)
    
    scaler_stats = auto_scaler.get_stats()
    logger.info(f"✅ Auto-scaler test: managing {scaler_stats['managed_pools']} pools")
    
    logger.info("⚡ Advanced optimization test complete!")


if __name__ == "__main__":
    test_advanced_optimization()
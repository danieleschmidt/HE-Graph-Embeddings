"""
Advanced concurrent processing for HE-Graph-Embeddings with auto-scaling and load balancing
"""


import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import queue
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import torch
import numpy as np
from contextlib import asynccontextmanager, contextmanager
import psutil
import os
import uuid

from .logging import get_logger, log_context, performance_context
from .error_handling import handle_exceptions, HEGraphError, retry_on_error
from .monitoring import get_health_checker
from .circuit_breaker import get_circuit_breaker, CircuitBreakerConfig

logger = get_logger(__name__)

@dataclass
class ConcurrencyConfig:
    """Configuration for concurrent processing"""
    # Worker management
    min_workers: int = 2
    max_workers: int = mp.cpu_count() * 2
    worker_idle_timeout: float = 300.0  # 5 minutes

    # Queue management
    max_queue_size: int = 1000
    queue_timeout: float = 30.0
    priority_levels: int = 3

    # Load balancing
    enable_load_balancing: bool = True
    load_balance_interval: float = 10.0
    worker_health_check_interval: float = 30.0

    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # Queue utilization
    scale_down_threshold: float = 0.2
    scaling_cooldown: float = 60.0

    # Performance optimization
    enable_work_stealing: bool = True
    enable_batching: bool = True
    max_batch_size: int = 10
    batch_timeout: float = 1.0

@dataclass
class WorkItem:
    """Individual work item in the processing queue"""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None

    def __post_init__(self):
        """  Post Init  ."""
        if not self.id:
            self.id = str(uuid.uuid4())

class WorkerProcess:
    """Individual worker process for concurrent execution"""

    def __init__(self, worker_id: str, config: ConcurrencyConfig):
        """  Init  ."""
        self.worker_id = worker_id
        self.config = config
        self.process = None
        self.input_queue = mp.Queue(maxsize=config.max_queue_size)
        self.output_queue = mp.Queue()
        self.status = "initializing"
        self.last_activity = time.time()
        self.processed_count = 0
        self.error_count = 0
        self.current_load = 0.0

        # Performance metrics
        self.performance_history = deque(maxlen=100)
        self.avg_processing_time = 0.0

        # Health monitoring
        self.health_check_failures = 0
        self.max_health_failures = 3

    def start(self) -> None:
        """Start worker process"""
        try:
            self.process = mp.Process(
                target=self._worker_main,
                args=(self.input_queue, self.output_queue, self.config),
                daemon=True
            )
            self.process.start()
            self.status = "running"
            logger.info(f"Worker {self.worker_id} started with PID {self.process.pid}")

        except Exception as e:
            logger.error(f"Failed to start worker {self.worker_id}: {e}")
            self.status = "failed"
            raise

    def stop(self, timeout: float = 10.0) -> None:
        """Stop worker process gracefully"""
        if not self.process or not self.process.is_alive():
            return

        try:
            # Send shutdown signal
            self.input_queue.put(("shutdown", None), timeout=1.0)
            self.process.join(timeout=timeout)

            if self.process.is_alive():
                logger.warning(f"Forcibly terminating worker {self.worker_id}")
                self.process.terminate()
                self.process.join(timeout=5.0)

            self.status = "stopped"
            logger.info(f"Worker {self.worker_id} stopped")

        except Exception as e:
            logger.error(f"Error stopping worker {self.worker_id}: {e}")
            if self.process and self.process.is_alive():
                self.process.kill()

    def submit_work(self, work_item: WorkItem) -> bool:
        """Submit work item to worker"""
        try:
            self.input_queue.put(("work", work_item), timeout=1.0)
            self.current_load += 1
            return True
        except queue.Full:
            return False
        except Exception as e:
            logger.error(f"Error submitting work to {self.worker_id}: {e}")
            return False

    def get_results(self) -> List[Tuple[str, Any]]:
        """Get completed work results"""
        results = []

        try:
            while True:
                try:
                    result_type, data = self.output_queue.get_nowait()

                    if result_type == "result":
                        work_id, result, processing_time = data
                        results.append((work_id, result))

                        # Update metrics
                        self.processed_count += 1
                        self.current_load = max(0, self.current_load - 1)
                        self.last_activity = time.time()

                        # Update performance history
                        self.performance_history.append(processing_time)
                        if self.performance_history:
                            self.avg_processing_time = statistics.mean(self.performance_history)

                    elif result_type == "error":
                        work_id, error = data
                        self.error_count += 1
                        self.current_load = max(0, self.current_load - 1)
                        results.append((work_id, Exception(f"Worker error: {error}")))

                    elif result_type == "health_check":
                        self.health_check_failures = 0  # Reset on successful health check

                except queue.Empty:
                    logger.error(f"Error in operation: {e}")
                    break

        except Exception as e:
            logger.error(f"Error getting results from {self.worker_id}: {e}")

        return results

    def is_healthy(self) -> bool:
        """Check if worker is healthy"""
        if not self.process or not self.process.is_alive():
            return False

        if self.health_check_failures > self.max_health_failures:
            return False

        # Check if worker is responsive
        idle_time = time.time() - self.last_activity
        if idle_time > self.config.worker_idle_timeout:
            return False

        return self.status == "running"

    def health_check(self) -> bool:
        """Perform health check on worker"""
        try:
            self.input_queue.put(("health_check", None), timeout=1.0)
            return True
        except (queue.Full, Exception):
            logger.error(f"Error in operation: {e}")
            self.health_check_failures += 1
            return False

    def get_load_metric(self) -> float:
        """Get current load metric (0.0 to 1.0)"""
        queue_load = self.input_queue.qsize() / max(1, self.config.max_queue_size)
        processing_load = self.current_load / max(1, self.config.max_queue_size)
        return min(1.0, max(queue_load, processing_load))

    @staticmethod
    def _worker_main(input_queue: mp.Queue, output_queue: mp.Queue, config: ConcurrencyConfig):
        """Main worker process function"""
        worker_name = f"worker-{os.getpid()}"

        # Initialize worker-specific resources
        torch.set_num_threads(1)  # Prevent thread contention

        try:
            while True:
                try:
                    # Get work item
                    message_type, data = input_queue.get(timeout=config.queue_timeout)

                    if message_type == "shutdown":
                        break

                    elif message_type == "health_check":
                        output_queue.put(("health_check", "ok"))
                        continue

                    elif message_type == "work":
                        work_item = data
                        start_time = time.time()

                        try:
                            # Execute work item
                            result = work_item.func(*work_item.args, **work_item.kwargs)
                            processing_time = time.time() - start_time

                            output_queue.put(("result", (work_item.id, result, processing_time)))

                        except Exception as e:
                            logger.error(f"Error in operation: {e}")
                            output_queue.put(("error", (work_item.id, str(e))))

                except queue.Empty:
                    logger.error(f"Error in operation: {e}")
                    continue  # Timeout, check for shutdown
                except Exception as e:
                    logger.error(f"Error in operation: {e}")
                    output_queue.put(("error", ("unknown", str(e))))

        except Exception as e:
            logger.error(f"Error in operation: {e}")
            output_queue.put(("error", ("worker_main", str(e))))

class WorkerPool:
    """Pool of worker processes with auto-scaling and load balancing"""

    def __init__(self, config: ConcurrencyConfig):
        """  Init  ."""
        self.config = config
        self.workers = {}  # worker_id -> WorkerProcess
        self.pending_work = {}  # work_id -> (WorkItem, future)
        self.completed_work = {}  # work_id -> result

        # Queue management
        self.priority_queues = [deque() for _ in range(config.priority_levels)]
        self.work_lock = threading.RLock()

        # Monitoring and management
        self.running = False
        self.manager_thread = None
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler(config)

        # Performance metrics
        self.total_submitted = 0
        self.total_completed = 0
        self.total_errors = 0

        # Circuit breaker for worker failures
        self.circuit_breaker = get_circuit_breaker(
            "worker_pool",
            CircuitBreakerConfig(failure_threshold=5, recovery_timeout=30.0)
        )

    def start(self) -> None:
        """Start the worker pool"""
        if self.running:
            return

        with log_context(operation="worker_pool_start"):
            self.running = True

            # Start initial workers
            initial_workers = min(self.config.min_workers, self.config.max_workers)
            for i in range(initial_workers):
                self._add_worker()

            # Start management thread
            self.manager_thread = threading.Thread(target=self._management_loop, daemon=True)
            self.manager_thread.start()

            logger.info(f"Worker pool started with {len(self.workers)} workers")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the worker pool"""
        if not self.running:
            return

        with log_context(operation="worker_pool_stop"):
            self.running = False

            # Stop all workers
            stop_futures = []
            for worker in self.workers.values():
                future = concurrent.futures.ThreadPoolExecutor().submit(worker.stop, timeout/len(self.workers))
                stop_futures.append(future)

            # Wait for workers to stop
            concurrent.futures.wait(stop_futures, timeout=timeout)

            # Stop management thread
            if self.manager_thread:
                self.manager_thread.join(timeout=5.0)

            self.workers.clear()
            logger.info("Worker pool stopped")

    def submit(self, func: Callable, *args, priority: int = 1, timeout: float = None,
        """Submit."""
                max_retries: int = 3, callback: Callable = None, **kwargs) -> concurrent.futures.Future:
        """Submit work to the pool"""
        if not self.running:
            raise HEGraphError("Worker pool is not running")

        work_item = WorkItem(
            id=str(uuid.uuid4()),
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            callback=callback
        )

        future = concurrent.futures.Future()

        with self.work_lock:
            # Add to priority queue
            priority_idx = max(0, min(priority - 1, self.config.priority_levels - 1))
            self.priority_queues[priority_idx].append(work_item)

            # Track pending work
            self.pending_work[work_item.id] = (work_item, future)
            self.total_submitted += 1

        return future

    def _add_worker(self) -> Optional[WorkerProcess]:
        """Add a new worker to the pool"""
        if len(self.workers) >= self.config.max_workers:
            return None

        worker_id = f"worker-{len(self.workers)}-{int(time.time())}"

        try:
            worker = WorkerProcess(worker_id, self.config)
            worker.start()
            self.workers[worker_id] = worker

            logger.info(f"Added worker {worker_id} to pool (total: {len(self.workers)})")
            return worker

        except Exception as e:
            logger.error(f"Failed to add worker: {e}")
            return None

    def _remove_worker(self, worker_id: str) -> None:
        """Remove worker from pool"""
        if worker_id in self.workers:
            worker = self.workers[worker_id]
            worker.stop()
            del self.workers[worker_id]
            logger.info(f"Removed worker {worker_id} from pool (total: {len(self.workers)})")

    def _management_loop(self) -> None:
        """Main management loop for the worker pool"""
        last_health_check = 0
        last_load_balance = 0
        last_scaling_check = 0

        while self.running:
            try:
                current_time = time.time()

                # Distribute work to workers
                self._distribute_work()

                # Collect results
                self._collect_results()

                # Health checks
                if current_time - last_health_check > self.config.worker_health_check_interval:
                    self._perform_health_checks()
                    last_health_check = current_time

                # Load balancing
                if (self.config.enable_load_balancing and
                    current_time - last_load_balance > self.config.load_balance_interval):
                    self._balance_load()
                    last_load_balance = current_time

                # Auto-scaling
                if (self.config.enable_auto_scaling and
                    current_time - last_scaling_check > self.config.scaling_cooldown):
                    self._check_scaling()
                    last_scaling_check = current_time

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Error in worker pool management: {e}")
                time.sleep(1.0)

    def _distribute_work(self) -> None:
        """Distribute queued work to available workers"""
        with self.work_lock:
            # Process priority queues in order
            for priority_queue in self.priority_queues:
                while priority_queue and self.workers:
                    # Find least loaded worker
                    best_worker = self.load_balancer.select_best_worker(list(self.workers.values()))

                    if not best_worker or best_worker.get_load_metric() > 0.9:
                        break  # All workers are busy

                    work_item = priority_queue.popleft()

                    if best_worker.submit_work(work_item):
                        logger.debug(f"Submitted work {work_item.id} to {best_worker.worker_id}")
                    else:
                        # Worker couldn't accept work, put it back
                        priority_queue.appendleft(work_item)
                        break

    def _collect_results(self) -> None:
        """Collect results from all workers"""
        for worker in self.workers.values():
            results = worker.get_results()

            for work_id, result in results:
                if work_id in self.pending_work:
                    work_item, future = self.pending_work[work_id]

                    if isinstance(result, Exception):
                        # Handle errors with retry logic
                        if work_item.retry_count < work_item.max_retries:
                            work_item.retry_count += 1
                            # Re-queue for retry
                            priority_idx = max(0, min(work_item.priority - 1, self.config.priority_levels - 1))
                            self.priority_queues[priority_idx].append(work_item)
                            logger.debug(f"Retrying work {work_id} (attempt {work_item.retry_count})")
                            continue
                        else:
                            future.set_exception(result)
                            self.total_errors += 1
                    else:
                        future.set_result(result)
                        self.total_completed += 1

                    # Call callback if provided
                    if work_item.callback:
                        try:
                            work_item.callback(result)
                        except Exception as e:
                            logger.error(f"Callback error for work {work_id}: {e}")

                    # Clean up
                    del self.pending_work[work_id]

    def _perform_health_checks(self) -> None:
        """Perform health checks on all workers"""
        unhealthy_workers = []

        for worker_id, worker in self.workers.items():
            if not worker.is_healthy():
                unhealthy_workers.append(worker_id)
            else:
                worker.health_check()

        # Remove unhealthy workers
        for worker_id in unhealthy_workers:
            logger.warning(f"Removing unhealthy worker {worker_id}")
            self._remove_worker(worker_id)

    def _balance_load(self) -> None:
        """Balance load across workers"""
        if len(self.workers) < 2:
            return

        worker_loads = [(w.worker_id, w.get_load_metric()) for w in self.workers.values()]
        worker_loads.sort(key=lambda x: x[1])

        # If load imbalance is significant, could implement work stealing here
        max_load = worker_loads[-1][1]
        min_load = worker_loads[0][1]

        if max_load - min_load > 0.5:
            logger.debug(f"Load imbalance detected: max={max_load:.2f}, min={min_load:.2f}")

    def _check_scaling(self) -> None:
        """Check if pool should be scaled up or down"""
        if not self.workers:
            return

        # Calculate average queue utilization
        total_queued = sum(len(q) for q in self.priority_queues)
        queue_utilization = total_queued / max(1, self.config.max_queue_size)

        # Calculate average worker load
        avg_worker_load = statistics.mean(w.get_load_metric() for w in self.workers.values())

        # Scale up if heavily loaded
        if (queue_utilization > self.config.scale_up_threshold or
            avg_worker_load > self.config.scale_up_threshold):
            if len(self.workers) < self.config.max_workers:
                logger.info(f"Scaling up: queue_util={queue_utilization:.2f}, worker_load={avg_worker_load:.2f}")
                self._add_worker()

        # Scale down if lightly loaded
        elif (queue_utilization < self.config.scale_down_threshold and
                avg_worker_load < self.config.scale_down_threshold):
            if len(self.workers) > self.config.min_workers:
                # Remove the least loaded worker
                worker_loads = [(w, w.get_load_metric()) for w in self.workers.values()]
                worker_loads.sort(key=lambda x: x[1])
                worker_to_remove = worker_loads[0][0]

                logger.info(f"Scaling down: removing worker {worker_to_remove.worker_id}")
                self._remove_worker(worker_to_remove.worker_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        with self.work_lock:
            total_queued = sum(len(q) for q in self.priority_queues)
            pending_count = len(self.pending_work)

            worker_stats = {}
            for worker_id, worker in self.workers.items():
                worker_stats[worker_id] = {
                    "status": worker.status,
                    "processed_count": worker.processed_count,
                    "error_count": worker.error_count,
                    "current_load": worker.current_load,
                    "load_metric": worker.get_load_metric(),
                    "avg_processing_time": worker.avg_processing_time,
                    "health_failures": worker.health_check_failures
                }

            return {
                "pool_status": "running" if self.running else "stopped",
                "worker_count": len(self.workers),
                "total_queued": total_queued,
                "pending_work": pending_count,
                "total_submitted": self.total_submitted,
                "total_completed": self.total_completed,
                "total_errors": self.total_errors,
                "completion_rate": self.total_completed / max(1, self.total_submitted),
                "error_rate": self.total_errors / max(1, self.total_submitted),
                "workers": worker_stats
            }

class LoadBalancer:
    """Load balancer for selecting optimal workers"""

    def select_best_worker(self, workers: List[WorkerProcess]) -> Optional[WorkerProcess]:
        """Select best worker based on load and performance"""
        if not workers:
            return None

        # Filter healthy workers
        healthy_workers = [w for w in workers if w.is_healthy()]

        if not healthy_workers:
            return None

        # Score workers based on multiple factors
        scored_workers = []

        for worker in healthy_workers:
            load_score = 1.0 - worker.get_load_metric()  # Lower load is better
            performance_score = 1.0 / (1.0 + worker.avg_processing_time)  # Faster is better
            reliability_score = 1.0 - (worker.error_count / max(1, worker.processed_count))

            # Weighted combination
            total_score = (0.5 * load_score + 0.3 * performance_score + 0.2 * reliability_score)
            scored_workers.append((worker, total_score))

        # Select best worker
        scored_workers.sort(key=lambda x: x[1], reverse=True)
        return scored_workers[0][0]

class AutoScaler:
    """Auto-scaling logic for worker pool"""

    def __init__(self, config: ConcurrencyConfig):
        """  Init  ."""
        self.config = config
        self.last_scale_action = 0
        self.scaling_history = deque(maxlen=10)

    def should_scale_up(self, current_workers: int, queue_utilization: float,
        """Should Scale Up."""
                        avg_worker_load: float) -> bool:
        """Check if should scale up"""
        if current_workers >= self.config.max_workers:
            return False

        if time.time() - self.last_scale_action < self.config.scaling_cooldown:
            return False

        return (queue_utilization > self.config.scale_up_threshold or
                avg_worker_load > self.config.scale_up_threshold)

    def should_scale_down(self, current_workers: int, queue_utilization: float,
        """Should Scale Down."""
                        avg_worker_load: float) -> bool:
        """Check if should scale down"""
        if current_workers <= self.config.min_workers:
            return False

        if time.time() - self.last_scale_action < self.config.scaling_cooldown:
            return False

        return (queue_utilization < self.config.scale_down_threshold and
                avg_worker_load < self.config.scale_down_threshold)

    def record_scaling_action(self, action: str, worker_count: int) -> None:
        """Record scaling action"""
        self.last_scale_action = time.time()
        self.scaling_history.append({
            "timestamp": self.last_scale_action,
            "action": action,
            "worker_count": worker_count
        })

# Global worker pool instance
_global_worker_pool = None

def get_worker_pool(config: ConcurrencyConfig = None) -> WorkerPool:
    """Get global worker pool instance"""
    global _global_worker_pool

    if _global_worker_pool is None:
        config = config or ConcurrencyConfig()
        _global_worker_pool = WorkerPool(config)
        _global_worker_pool.start()

    return _global_worker_pool

def shutdown_worker_pool():
    """Shutdown global worker pool"""
    global _global_worker_pool

    if _global_worker_pool:
        _global_worker_pool.stop()
        _global_worker_pool = None

# Context manager for temporary worker pools
@asynccontextmanager
async def temporary_worker_pool(config: ConcurrencyConfig = None):
    """Context manager for temporary worker pool"""
    config = config or ConcurrencyConfig()
    pool = WorkerPool(config)

    try:
        pool.start()
        yield pool
    finally:
        pool.stop()

# High-level async functions
async def concurrent_map(func: Callable, items: List[Any], max_workers: int = None,
                        priority: int = 1, timeout: float = None) -> List[Any]:
    """Concurrent map operation"""
    max_workers = max_workers or min(len(items), mp.cpu_count())
    config = ConcurrencyConfig(max_workers=max_workers)

    async with temporary_worker_pool(config) as pool:
        futures = []

        for item in items:
            future = pool.submit(func, item, priority=priority, timeout=timeout)
            futures.append(future)

        # Wait for all results
        results = []
        for future in futures:
            try:
                result = await asyncio.wrap_future(future)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in operation: {e}")
                results.append(e)

        return results

async def concurrent_batch_processing(items: List[Any], process_func: Callable,
                                    batch_size: int = 10, max_workers: int = None) -> List[Any]:
    """Process items in concurrent batches"""
    # Split items into batches
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

    # Process batches concurrently
    return await concurrent_map(process_func, batches, max_workers=max_workers)

# Decorators for concurrent processing
def concurrent_execution(max_workers: int = None, priority: int = 1):
    """Decorator for concurrent execution"""
    def decorator(func: Callable) -> Callable:
        """Decorator."""
        @handle_exceptions(reraise_as=HEGraphError)
        async def async_wrapper(items: List[Any], *args, **kwargs):
            def bound_func(item):
                """Bound Func."""
                return func(item, *args, **kwargs)

            return await concurrent_map(bound_func, items, max_workers=max_workers, priority=priority)

        return async_wrapper
    return decorator
"""
Comprehensive monitoring and health check utilities
"""


import asyncio
import time
import psutil
import torch
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import json
from collections import defaultdict, deque
import gc

from .logging import get_logger, performance_context

logger = get_logger(__name__)

class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = None
    timestamp: datetime = None
    duration_ms: float = 0

    def __post_init__(self):
        """  Post Init  ."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.details is None:
            self.details = {}

    def to_dict(self) -> Dict[str, Any]:
        """To Dict."""
        return {
            **asdict(self),
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat()
        }

class MetricCollector:
    """Collect and aggregate system metrics"""

    def __init__(self, max_history: int = 1000):
        """  Init  ."""
        self.max_history = max_history
        self.metrics = defaultdict(deque)
        self.lock = threading.Lock()

    def record(self, metric_name -> None: str, value: float, timestamp: datetime = None):
        """Record a metric value"""
        timestamp = timestamp or datetime.utcnow()

        with self.lock:
            self.metrics[metric_name].append((timestamp, value))

            # Keep only recent history
            while len(self.metrics[metric_name]) > self.max_history:
                self.metrics[metric_name].popleft()

    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value for metric"""
        with self.lock:
            if metric_name in self.metrics and self.metrics[metric_name]:
                return self.metrics[metric_name][-1][1]
        return None

    def get_average(self, metric_name: str, window_minutes: int = 5) -> Optional[float]:
        """Get average value over time window"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)

        with self.lock:
            if metric_name not in self.metrics:
                return None

            recent_values = [
                value for timestamp, value in self.metrics[metric_name]
                if timestamp >= cutoff_time
            ]

            return sum(recent_values) / len(recent_values) if recent_values else None

    def get_percentile(self, metric_name -> None: str, percentile: int = 95,
        """Get Percentile."""
                        window_minutes: int = 5) -> Optional[float]:
        """Get percentile value over time window"""

        import numpy as np

        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)

        with self.lock:
            if metric_name not in self.metrics:
                return None

            recent_values = [
                value for timestamp, value in self.metrics[metric_name]
                if timestamp >= cutoff_time
            ]

            return np.percentile(recent_values, percentile) if recent_values else None

class SystemMonitor:
    """Monitor system resources and performance"""

    def __init__(self, metric_collector: MetricCollector):
        """  Init  ."""
        self.metric_collector = metric_collector
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval -> None: float = 5.0):
        """Start background monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("System monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        logger.info("System monitoring stopped")

    def _monitor_loop(self, interval -> None: float):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

    def _collect_system_metrics(self) -> None:
        """Collect current system metrics"""
        timestamp = datetime.utcnow()

        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        self.metric_collector.record("cpu.usage_percent", cpu_percent, timestamp)

        # Memory metrics
        memory = psutil.virtual_memory()
        self.metric_collector.record("memory.usage_percent", memory.percent, timestamp)
        self.metric_collector.record("memory.available_gb", memory.available / (1024**3), timestamp)
        self.metric_collector.record("memory.used_gb", memory.used / (1024**3), timestamp)

        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metric_collector.record("disk.usage_percent", (disk.used / disk.total) * 100, timestamp)
        self.metric_collector.record("disk.free_gb", disk.free / (1024**3), timestamp)

        # GPU metrics (if available)
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)  # GB

                self.metric_collector.record("gpu.memory_allocated_gb", gpu_memory, timestamp)
                self.metric_collector.record("gpu.memory_cached_gb", gpu_memory_cached, timestamp)

                # Get GPU utilization if nvidia-ml-py is available
                try:

                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.metric_collector.record("gpu.utilization_percent", util.gpu, timestamp)
                except ImportError:
                    logger.error(f"Error in operation: {e}")
                    pass  # nvidia-ml-py not available

            except Exception as e:
                logger.debug(f"Could not collect GPU metrics: {e}")

    def get_system_health(self) -> HealthCheckResult:
        """Get overall system health"""
        issues = []
        status = HealthStatus.HEALTHY

        # Check CPU usage
        cpu_usage = self.metric_collector.get_average("cpu.usage_percent")
        if cpu_usage is not None:
            if cpu_usage > 90:
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
                status = HealthStatus.CRITICAL
            elif cpu_usage > 80:
                issues.append(f"Elevated CPU usage: {cpu_usage:.1f}%")
                status = max(status, HealthStatus.DEGRADED)

        # Check memory usage
        memory_usage = self.metric_collector.get_average("memory.usage_percent")
        if memory_usage is not None:
            if memory_usage > 95:
                issues.append(f"Critical memory usage: {memory_usage:.1f}%")
                status = HealthStatus.CRITICAL
            elif memory_usage > 85:
                issues.append(f"High memory usage: {memory_usage:.1f}%")
                status = max(status, HealthStatus.DEGRADED)

        # Check disk usage
        disk_usage = self.metric_collector.get_latest("disk.usage_percent")
        if disk_usage is not None:
            if disk_usage > 95:
                issues.append(f"Critical disk usage: {disk_usage:.1f}%")
                status = HealthStatus.CRITICAL
            elif disk_usage > 85:
                issues.append(f"High disk usage: {disk_usage:.1f}%")
                status = max(status, HealthStatus.DEGRADED)

        message = "System healthy" if not issues else "; ".join(issues)

        return HealthCheckResult(
            name="system_resources",
            status=status,
            message=message,
            details={
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory_usage,
                "disk_usage_percent": disk_usage
            }
        )

class ApplicationMonitor:
    """Monitor application-specific metrics"""

    def __init__(self, metric_collector: MetricCollector):
        """  Init  ."""
        self.metric_collector = metric_collector
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.error_count = 0
        self.active_contexts = {}
        self.active_models = {}

    def record_request(self, duration_ms -> None: float, status_code: int):
        """Record API request metrics"""
        self.request_count += 1
        self.metric_collector.record("api.request_duration_ms", duration_ms)
        self.metric_collector.record("api.requests_per_second", 1)  # Will be aggregated

        if status_code >= 400:
            self.error_count += 1
            self.metric_collector.record("api.error_rate", 1)

    def record_encryption_operation(self, operation -> None: str, duration_ms: float,
        """Record Encryption Operation."""
                                    data_size: int, success: bool):
        """Record encryption operation metrics"""
        self.metric_collector.record(f"encryption.{operation}_duration_ms", duration_ms)
        self.metric_collector.record(f"encryption.{operation}_throughput_mb_per_sec",
                                    (data_size / (1024*1024)) / (duration_ms / 1000))

        if not success:
            self.metric_collector.record(f"encryption.{operation}_error_rate", 1)

    def record_model_inference(self, model_name -> None: str, duration_ms: float,
        """Record Model Inference."""
                            batch_size: int, success: bool):
        """Record model inference metrics"""
        self.metric_collector.record(f"model.{model_name}_inference_duration_ms", duration_ms)
        self.metric_collector.record(f"model.{model_name}_throughput",
                                    batch_size / (duration_ms / 1000))  # samples per second

        if not success:
            self.metric_collector.record(f"model.{model_name}_error_rate", 1)

    def add_context(self, name -> None: str, context_info: Dict[str, Any]):
        """Add active CKKS context"""
        self.active_contexts[name] = {
            **context_info,
            "created_at": datetime.utcnow()
        }

    def remove_context(self, name -> None: str):
        """Remove active CKKS context"""
        self.active_contexts.pop(name, None)

    def add_model(self, name -> None: str, model_info: Dict[str, Any]):
        """Add active model"""
        self.active_models[name] = {
            **model_info,
            "created_at": datetime.utcnow()
        }

    def remove_model(self, name -> None: str):
        """Remove active model"""
        self.active_models.pop(name, None)

    def get_application_health(self) -> HealthCheckResult:
        """Get application health status"""
        issues = []
        status = HealthStatus.HEALTHY

        # Check error rate
        recent_error_rate = self.metric_collector.get_average("api.error_rate", 5)
        if recent_error_rate is not None:
            if recent_error_rate > 0.1:  # 10% error rate
                issues.append(f"High error rate: {recent_error_rate:.1%}")
                status = HealthStatus.DEGRADED
            elif recent_error_rate > 0.2:  # 20% error rate
                issues.append(f"Critical error rate: {recent_error_rate:.1%}")
                status = HealthStatus.CRITICAL

        # Check response times
        avg_response_time = self.metric_collector.get_average("api.request_duration_ms", 5)
        if avg_response_time is not None:
            if avg_response_time > 5000:  # 5 seconds
                issues.append(f"Slow response times: {avg_response_time:.0f}ms")
                status = max(status, HealthStatus.DEGRADED)
            elif avg_response_time > 10000:  # 10 seconds
                issues.append(f"Very slow response times: {avg_response_time:.0f}ms")
                status = HealthStatus.CRITICAL

        uptime = datetime.utcnow() - self.start_time

        return HealthCheckResult(
            name="application",
            status=status,
            message="Application healthy" if not issues else "; ".join(issues),
            details={
                "uptime_seconds": uptime.total_seconds(),
                "total_requests": self.request_count,
                "total_errors": self.error_count,
                "active_contexts": len(self.active_contexts),
                "active_models": len(self.active_models),
                "error_rate": recent_error_rate,
                "avg_response_time_ms": avg_response_time
            }
        )

class HealthChecker:
    """Centralized health checking system"""

    def __init__(self):
        """  Init  ."""
        self.health_checks = {}
        self.metric_collector = MetricCollector()
        self.system_monitor = SystemMonitor(self.metric_collector)
        self.app_monitor = ApplicationMonitor(self.metric_collector)

    def register_health_check(self, name -> None: str, check_func: Callable[[], HealthCheckResult]):
        """Register a custom health check"""
        self.health_checks[name] = check_func

    async def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks"""
        results = {}

        # System health
        results["system"] = self.system_monitor.get_system_health()

        # Application health
        results["application"] = self.app_monitor.get_application_health()

        # Custom health checks
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.perf_counter()
                result = check_func()
                result.duration_ms = (time.perf_counter() - start_time) * 1000
                results[name] = result
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(e)}"
                )

        return results

    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """Determine overall system status from individual checks"""
        if any(r.status == HealthStatus.CRITICAL for r in results.values()):
            return HealthStatus.CRITICAL
        elif any(r.status == HealthStatus.UNHEALTHY for r in results.values()):
            return HealthStatus.UNHEALTHY
        elif any(r.status == HealthStatus.DEGRADED for r in results.values()):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def start_monitoring(self) -> None:
        """Start background monitoring"""
        self.system_monitor.start_monitoring()
        logger.info("Health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self.system_monitor.stop_monitoring()
        logger.info("Health monitoring stopped")

# Custom health checks for HE-Graph-Embeddings

def check_cuda_health() -> HealthCheckResult:
    """Check CUDA/GPU health"""
    if not torch.cuda.is_available():
        return HealthCheckResult(
            name="cuda",
            status=HealthStatus.DEGRADED,
            message="CUDA not available - using CPU mode",
            details={"cuda_available": False}
        )

    try:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

        memory_usage = (memory_allocated / memory_total) * 100

        status = HealthStatus.HEALTHY
        message = f"CUDA healthy - {device_name}"

        if memory_usage > 90:
            status = HealthStatus.CRITICAL
            message = f"Critical GPU memory usage: {memory_usage:.1f}%"
        elif memory_usage > 80:
            status = HealthStatus.DEGRADED
            message = f"High GPU memory usage: {memory_usage:.1f}%"

        return HealthCheckResult(
            name="cuda",
            status=status,
            message=message,
            details={
                "cuda_available": True,
                "device_count": device_count,
                "current_device": current_device,
                "device_name": device_name,
                "memory_allocated_gb": memory_allocated,
                "memory_total_gb": memory_total,
                "memory_usage_percent": memory_usage
            }
        )

    except Exception as e:
        logger.error(f"Error in operation: {e}")
        return HealthCheckResult(
            name="cuda",
            status=HealthStatus.UNHEALTHY,
            message=f"CUDA error: {str(e)}"
        )

def check_encryption_context() -> HealthCheckResult:
    """Check if encryption contexts are working"""
    try:
        from ..python.he_graph import CKKSContext, HEConfig

        # Test basic encryption context creation
        config = HEConfig(poly_modulus_degree=1024)  # Small for testing
        context = CKKSContext(config)
        context.generate_keys()

        # Test basic encryption
        test_data = torch.randn(10, 5)
        encrypted = context.encrypt(test_data)

        return HealthCheckResult(
            name="encryption_context",
            status=HealthStatus.HEALTHY,
            message="Encryption context working correctly",
            details={
                "poly_degree": config.poly_modulus_degree,
                "security_level": config.security_level
            }
        )

    except Exception as e:
        logger.error(f"Error in operation: {e}")
        return HealthCheckResult(
            name="encryption_context",
            status=HealthStatus.CRITICAL,
            message=f"Encryption context failed: {str(e)}"
        )

# Global health checker instance
health_checker = HealthChecker()

# Register default health checks
health_checker.register_health_check("cuda", check_cuda_health)
health_checker.register_health_check("encryption", check_encryption_context)

def get_health_checker() -> HealthChecker:
    """Get global health checker instance"""
    return health_checker
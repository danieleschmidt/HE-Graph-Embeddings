"""
Advanced circuit breaker implementation with adaptive thresholds and recovery strategies
"""


import time
import threading
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
from collections import deque

from .logging import get_logger
from .error_handling import HEGraphError

logger = get_logger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout_duration: float = 30.0  # Request timeout
    volume_threshold: int = 10  # Minimum requests before evaluating
    error_threshold_percentage: float = 50.0  # Error rate threshold
    monitoring_window: float = 300.0  # 5 minutes
    adaptive_threshold: bool = True

    def __post_init__(self):
        """  Post Init  ."""
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if self.recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")

@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    timestamp: float
    duration: float
    success: bool
    error_type: Optional[str] = None

class AdaptiveThreshold:
    """Adaptive threshold calculator based on historical performance"""

    def __init__(self, window_size: int = 100):
        """  Init  ."""
        self.window_size = window_size
        self.success_rates = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.lock = threading.Lock()

    def record_request(self, success -> None: bool, duration: float):
        """Record request outcome and duration"""
        with self.lock:
            self.success_rates.append(1.0 if success else 0.0)
            self.response_times.append(duration)

    def get_adaptive_threshold(self, base_threshold: float) -> float:
        """Calculate adaptive threshold based on recent performance"""
        with self.lock:
            if len(self.success_rates) < 10:  # Need minimum data
                return base_threshold

            # Calculate recent success rate
            recent_success_rate = statistics.mean(list(self.success_rates)[-20:])

            # Calculate recent response time trend
            if len(self.response_times) >= 20:
                recent_times = list(self.response_times)[-20:]
                older_times = list(self.response_times)[-40:-20] if len(self.response_times) >= 40 else recent_times

                recent_avg = statistics.mean(recent_times)
                older_avg = statistics.mean(older_times)

                # If response times are increasing, lower threshold
                if recent_avg > older_avg * 1.5:
                    adaptive_factor = 0.7  # Make more sensitive
                elif recent_avg < older_avg * 0.8:
                    adaptive_factor = 1.3  # Make less sensitive
                else:
                    adaptive_factor = 1.0
            else:
                adaptive_factor = 1.0

            # Adjust based on success rate
            if recent_success_rate < 0.8:
                adaptive_factor *= 0.8  # More sensitive when success rate drops
            elif recent_success_rate > 0.95:
                adaptive_factor *= 1.2  # Less sensitive when performing well

            return max(base_threshold * adaptive_factor, base_threshold * 0.5)

class CircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds and detailed monitoring"""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """  Init  ."""
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State management
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_success_time = 0
        self.next_attempt_time = 0

        # Thread safety
        self.lock = threading.RLock()

        # Metrics tracking
        self.metrics = deque(maxlen=1000)
        self.total_requests = 0
        self.total_failures = 0

        # Adaptive threshold
        self.adaptive_threshold = AdaptiveThreshold() if config and config.adaptive_threshold else None

        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")

    def call(self, func: Callable, *args, timeout: Optional[float] = None, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            current_time = time.time()

            # Check if circuit is open and recovery timeout hasn't passed
            if self.state == CircuitState.OPEN:
                if current_time < self.next_attempt_time:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Next attempt in {self.next_attempt_time - current_time:.1f}s"
                    )
                else:
                    # Move to half-open state
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' moving to HALF_OPEN")

            self.total_requests += 1

        # Execute the function
        start_time = time.time()
        timeout = timeout or self.config.timeout_duration

        try:
            # TODO: Implement actual timeout mechanism
            result = func(*args, **kwargs)
            self._on_success(time.time() - start_time)
            return result

        except Exception as e:
            logger.error(f"Error in operation: {e}")
            self._on_failure(time.time() - start_time, e)
            raise

    def _on_success(self, duration -> None: float):
        """Handle successful execution"""
        current_time = time.time()

        with self.lock:
            self.success_count += 1
            self.last_success_time = current_time
            self.failure_count = 0  # Reset failure count on success

            # Record metrics
            self.metrics.append(RequestMetrics(
                timestamp=current_time,
                duration=duration,
                success=True
            ))

            if self.adaptive_threshold:
                self.adaptive_threshold.record_request(True, duration)

            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.success_count = 0
                    logger.info(f"Circuit breaker '{self.name}' closed after {self.config.success_threshold} successes")

            logger.debug(f"Circuit breaker '{self.name}' success: duration={duration:.3f}s, state={self.state.value}")

    def _on_failure(self, duration -> None: float, exception: Exception):
        """Handle failed execution"""
        current_time = time.time()

        with self.lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = current_time
            self.success_count = 0  # Reset success count on failure

            # Record metrics
            self.metrics.append(RequestMetrics(
                timestamp=current_time,
                duration=duration,
                success=False,
                error_type=type(exception).__name__
            ))

            if self.adaptive_threshold:
                self.adaptive_threshold.record_request(False, duration)

            # Check if circuit should open
            should_open = self._should_open_circuit()

            if should_open and self.state != CircuitState.OPEN:
                self.state = CircuitState.OPEN
                self.next_attempt_time = current_time + self.config.recovery_timeout
                logger.error(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures. "
                            f"Next attempt at {datetime.fromtimestamp(self.next_attempt_time)}")

            logger.debug(f"Circuit breaker '{self.name}' failure: {exception.__class__.__name__}, "
                        f"count={self.failure_count}, state={self.state.value}")

    def _should_open_circuit(self) -> bool:
        """Determine if circuit should open based on current metrics"""
        # Basic threshold check
        if self.failure_count >= self.config.failure_threshold:
            return True

        # Volume and error rate based check
        recent_window = time.time() - self.config.monitoring_window
        recent_metrics = [m for m in self.metrics if m.timestamp >= recent_window]

        if len(recent_metrics) < self.config.volume_threshold:
            return False  # Not enough volume

        # Calculate error rate
        error_count = sum(1 for m in recent_metrics if not m.success)
        error_rate = (error_count / len(recent_metrics)) * 100

        # Use adaptive threshold if available
        threshold = self.config.error_threshold_percentage
        if self.adaptive_threshold:
            threshold = self.adaptive_threshold.get_adaptive_threshold(threshold)

        return error_rate > threshold

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics"""
        with self.lock:
            current_time = time.time()
            recent_window = current_time - self.config.monitoring_window
            recent_metrics = [m for m in self.metrics if m.timestamp >= recent_window]

            if recent_metrics:
                recent_success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
                recent_avg_duration = statistics.mean(m.duration for m in recent_metrics)
                recent_error_rate = (1 - recent_success_rate) * 100
            else:
                recent_success_rate = 1.0
                recent_avg_duration = 0.0
                recent_error_rate = 0.0

            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "overall_error_rate": (self.total_failures / max(self.total_requests, 1)) * 100,
                "recent_success_rate": recent_success_rate * 100,
                "recent_error_rate": recent_error_rate,
                "recent_avg_duration_ms": recent_avg_duration * 1000,
                "recent_request_count": len(recent_metrics),
                "last_failure_time": datetime.fromtimestamp(self.last_failure_time).isoformat() if self.last_failure_time else None,
                "last_success_time": datetime.fromtimestamp(self.last_success_time).isoformat() if self.last_success_time else None,
                "next_attempt_time": datetime.fromtimestamp(self.next_attempt_time).isoformat() if self.next_attempt_time > current_time else None,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "error_threshold_percentage": self.config.error_threshold_percentage,
                    "adaptive_threshold": bool(self.adaptive_threshold)
                }
            }

    def reset(self) -> None:
        """Reset circuit breaker to initial state"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = 0
            self.next_attempt_time = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset")

    def force_open(self, duration -> None: float = None):
        """Force circuit breaker to open state"""
        with self.lock:
            self.state = CircuitState.OPEN
            duration = duration or self.config.recovery_timeout
            self.next_attempt_time = time.time() + duration
            logger.warning(f"Circuit breaker '{self.name}' forced open for {duration}s")

    def force_close(self) -> None:
        """Force circuit breaker to closed state"""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.next_attempt_time = 0
            logger.warning(f"Circuit breaker '{self.name}' forced closed")

class CircuitBreakerOpenError(HEGraphError):
    """Exception raised when circuit breaker is open"""
    pass

class CircuitBreakerManager:
    """Manager for multiple circuit breakers"""

    def __init__(self):
        """  Init  ."""
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()

    def get_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker"""
        with self.lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(name, config)
            return self.breakers[name]

    def remove_breaker(self, name -> None: str):
        """Remove circuit breaker"""
        with self.lock:
            if name in self.breakers:
                del self.breakers[name]
                logger.info(f"Circuit breaker '{name}' removed")

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers"""
        with self.lock:
            return {name: breaker.get_metrics() for name, breaker in self.breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers"""
        with self.lock:
            for breaker in self.breakers.values():
                breaker.reset()
            logger.info("All circuit breakers reset")

# Global circuit breaker manager
_circuit_breaker_manager = CircuitBreakerManager()

def get_circuit_breaker(name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
    """Get circuit breaker instance"""
    return _circuit_breaker_manager.get_breaker(name, config)

def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get metrics for all circuit breakers"""
    return _circuit_breaker_manager.get_all_metrics()

# Decorator for automatic circuit breaker protection
def circuit_breaker_protected(name: str, config: CircuitBreakerConfig = None,
    """Circuit Breaker Protected."""
                            timeout: float = None):
    """Decorator to protect function with circuit breaker"""
    def decorator(func: Callable) -> Callable:
        """Decorator."""
        breaker = get_circuit_breaker(name, config)

        def wrapper(*args, **kwargs):
            """Wrapper."""
            return breaker.call(func, *args, timeout=timeout, **kwargs)

        wrapper.__name__ = f"{func.__name__}_circuit_protected"
        wrapper.__doc__ = f"Circuit breaker protected version of {func.__name__}"
        wrapper._circuit_breaker = breaker

        return wrapper
    return decorator

# Predefined configurations for common scenarios
class CircuitBreakerConfigs:
    """Predefined circuit breaker configurations"""

    # For external API calls
    EXTERNAL_API = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=30.0,
        timeout_duration=10.0,
        error_threshold_percentage=30.0,
        adaptive_threshold=True
    )

    # For database operations
    DATABASE = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=60.0,
        timeout_duration=30.0,
        error_threshold_percentage=20.0,
        volume_threshold=20
    )

    # For encryption operations
    ENCRYPTION = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=45.0,
        timeout_duration=60.0,
        error_threshold_percentage=25.0,
        adaptive_threshold=True
    )

    # For GPU operations
    GPU_OPERATIONS = CircuitBreakerConfig(
        failure_threshold=2,
        recovery_timeout=30.0,
        timeout_duration=120.0,
        error_threshold_percentage=15.0,
        adaptive_threshold=True
    )

# Context manager for circuit breaker protection
class circuit_breaker_context:
    """Context manager for circuit breaker protection"""

    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """  Init  ."""
        self.breaker = get_circuit_breaker(name, config)
        self.start_time = None

    def __enter__(self):
        """  Enter  ."""
        current_time = time.time()

        with self.breaker.lock:
            if self.breaker.state == CircuitState.OPEN:
                if current_time < self.breaker.next_attempt_time:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.breaker.name}' is OPEN"
                    )
                else:
                    self.breaker.state = CircuitState.HALF_OPEN

            self.breaker.total_requests += 1

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """  Exit  ."""
        duration = time.time() - self.start_time if self.start_time else 0

        if exc_type is None:
            self.breaker._on_success(duration)
        else:
            self.breaker._on_failure(duration, exc_val)

        return False  # Don't suppress exceptions
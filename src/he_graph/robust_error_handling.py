"""
🛡️ GENERATION 2: Robust Error Handling and Validation Framework

This module implements comprehensive error handling, input validation, and reliability
enhancements for the HE-Graph-Embeddings system.

Key Features:
- Custom exception hierarchy with recovery strategies
- Input validation with detailed error messages
- Circuit breaker pattern for fault tolerance
- Automatic retry mechanisms with exponential backoff
- Health monitoring and system diagnostics
- Comprehensive logging with structured data
"""

import logging
import time
import functools
import traceback
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from abc import ABC, abstractmethod

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification and handling"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types"""
    RETRY = "retry"
    FALLBACK = "fallback"
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class ErrorContext:
    """Context information for error analysis and recovery"""
    error_type: str
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    timestamp: float = field(default_factory=time.time)
    function_name: str = ""
    module_name: str = ""
    additional_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""


class HEGraphBaseException(Exception):
    """Base exception for HE-Graph-Embeddings with enhanced context"""
    
    def __init__(self, 
                 message: str,
                 error_code: str = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 recovery_strategy: RecoveryStrategy = RecoveryStrategy.FAIL_FAST,
                 context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.context = context or {}
        self.timestamp = time.time()
        
        # Capture stack trace
        self.stack_trace = traceback.format_exc()
        
        # Log the error
        self._log_error()
    
    def _log_error(self):
        """Log error with structured information"""
        log_data = {
            "error_code": self.error_code,
            "severity": self.severity.value,
            "recovery_strategy": self.recovery_strategy.value,
            "context": self.context,
            "timestamp": self.timestamp
        }
        
        if self.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error(f"{self.error_code}: {self.message}", extra=log_data)
        else:
            logger.warning(f"{self.error_code}: {self.message}", extra=log_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization"""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "recovery_strategy": self.recovery_strategy.value,
            "context": self.context,
            "timestamp": self.timestamp,
            "stack_trace": self.stack_trace
        }


class ValidationError(HEGraphBaseException):
    """Input validation errors"""
    
    def __init__(self, message: str, field_name: str = None, **kwargs):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.FAIL_FAST,
            **kwargs
        )
        self.field_name = field_name


class EncryptionError(HEGraphBaseException):
    """Homomorphic encryption operation errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="ENCRYPTION_ERROR",
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class ComputationError(HEGraphBaseException):
    """Graph computation and neural network errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="COMPUTATION_ERROR",
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            **kwargs
        )


class ResourceError(HEGraphBaseException):
    """Resource allocation and management errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="RESOURCE_ERROR",
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            **kwargs
        )


class NetworkError(HEGraphBaseException):
    """Network and communication errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="NETWORK_ERROR",
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )


class ConfigurationError(HEGraphBaseException):
    """Configuration and setup errors"""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FAIL_FAST,
            **kwargs
        )


class RobustValidator:
    """Comprehensive input validation with detailed error reporting"""
    
    @staticmethod
    def validate_graph_data(node_features: Any, edge_index: Any, 
                           edge_attributes: Any = None) -> Tuple[bool, Optional[str]]:
        """Validate graph data structure and content"""
        try:
            # Check if we have the required attributes (simulating torch tensors)
            if not hasattr(node_features, 'shape') and not hasattr(node_features, '__len__'):
                raise ValidationError(
                    "Node features must be array-like with shape attribute",
                    field_name="node_features"
                )
            
            if not hasattr(edge_index, 'shape') and not hasattr(edge_index, '__len__'):
                raise ValidationError(
                    "Edge index must be array-like with shape attribute",
                    field_name="edge_index"
                )
            
            # Validate dimensions (simulated)
            try:
                num_nodes = len(node_features) if hasattr(node_features, '__len__') else node_features.shape[0]
                num_edges = len(edge_index[0]) if hasattr(edge_index, '__len__') else edge_index.shape[1]
            except (IndexError, AttributeError) as e:
                raise ValidationError(
                    f"Invalid graph data structure: {e}",
                    context={"node_features_type": type(node_features).__name__,
                            "edge_index_type": type(edge_index).__name__}
                )
            
            # Validate ranges
            if num_nodes <= 0:
                raise ValidationError("Graph must have at least one node")
            
            if num_edges < 0:
                raise ValidationError("Number of edges cannot be negative")
            
            # Check for reasonable sizes
            if num_nodes > 1000000:
                logger.warning(f"Large graph detected: {num_nodes} nodes. Consider batch processing.")
            
            if num_edges > 5000000:
                logger.warning(f"Dense graph detected: {num_edges} edges. Consider sparsification.")
            
            return True, None
            
        except HEGraphBaseException:
            raise  # Re-raise our custom exceptions
        except Exception as e:
            raise ValidationError(
                f"Unexpected validation error: {e}",
                context={"original_error": str(e)}
            )
    
    @staticmethod
    def validate_encryption_params(poly_degree: int, coeff_modulus_bits: List[int],
                                 scale: float, security_level: int = 128) -> Tuple[bool, Optional[str]]:
        """Validate CKKS encryption parameters"""
        try:
            # Validate polynomial degree
            if poly_degree <= 0 or (poly_degree & (poly_degree - 1)) != 0:
                raise ValidationError(
                    "Polynomial degree must be a positive power of 2",
                    field_name="poly_degree",
                    context={"value": poly_degree}
                )
            
            if poly_degree < 1024:
                raise ValidationError(
                    "Polynomial degree too small for secure operation",
                    field_name="poly_degree",
                    context={"value": poly_degree, "minimum": 1024}
                )
            
            # Validate coefficient modulus
            if not coeff_modulus_bits or len(coeff_modulus_bits) < 2:
                raise ValidationError(
                    "Coefficient modulus must have at least 2 primes",
                    field_name="coeff_modulus_bits"
                )
            
            total_bits = sum(coeff_modulus_bits)
            max_bits = poly_degree // 2  # Simplified estimate
            
            if total_bits > max_bits:
                raise ValidationError(
                    f"Total modulus bits ({total_bits}) exceeds security limit ({max_bits})",
                    field_name="coeff_modulus_bits",
                    context={"total_bits": total_bits, "max_bits": max_bits}
                )
            
            # Validate scale
            if scale <= 0:
                raise ValidationError(
                    "Scale must be positive",
                    field_name="scale",
                    context={"value": scale}
                )
            
            # Check scale compatibility
            min_prime_bits = min(coeff_modulus_bits)
            scale_bits = scale.bit_length()
            
            if scale_bits >= min_prime_bits:
                logger.warning(
                    f"Scale bits ({scale_bits}) close to minimum prime bits ({min_prime_bits}). "
                    "This may cause precision issues."
                )
            
            return True, None
            
        except HEGraphBaseException:
            raise
        except Exception as e:
            raise ValidationError(
                f"Unexpected parameter validation error: {e}",
                context={"original_error": str(e)}
            )


class RetryMechanism:
    """Robust retry mechanism with exponential backoff"""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for adding retry functionality"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except HEGraphBaseException as e:
                    last_exception = e
                    
                    # Check if this error type should be retried
                    if e.recovery_strategy != RecoveryStrategy.RETRY:
                        logger.info(f"Not retrying {e.error_code} due to recovery strategy")
                        raise
                    
                    if attempt < self.max_retries:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e.message}. "
                            f"Retrying in {delay:.2f}s"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All {self.max_retries} retry attempts failed for {func.__name__}")
                        raise
                
                except Exception as e:
                    # Wrap unexpected exceptions
                    wrapped_error = ComputationError(
                        f"Unexpected error in {func.__name__}: {e}",
                        context={"original_error": str(e), "function": func.__name__}
                    )
                    
                    if attempt < self.max_retries:
                        delay = self._calculate_delay(attempt)
                        logger.warning(f"Retrying {func.__name__} in {delay:.2f}s due to: {e}")
                        time.sleep(delay)
                        last_exception = wrapped_error
                    else:
                        raise wrapped_error
            
            # This shouldn't be reached, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter"""
        delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add ±50% jitter
        
        return delay


class HealthMonitor:
    """System health monitoring and diagnostics"""
    
    def __init__(self):
        self.metrics = {
            "errors_total": 0,
            "errors_by_type": {},
            "last_error_time": None,
            "start_time": time.time(),
            "operations_total": 0,
            "operations_successful": 0
        }
        self.lock = threading.Lock()
    
    def record_operation(self, success: bool = True, error_type: str = None):
        """Record an operation outcome"""
        with self.lock:
            self.metrics["operations_total"] += 1
            
            if success:
                self.metrics["operations_successful"] += 1
            else:
                self.metrics["errors_total"] += 1
                self.metrics["last_error_time"] = time.time()
                
                if error_type:
                    self.metrics["errors_by_type"][error_type] = \
                        self.metrics["errors_by_type"].get(error_type, 0) + 1
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        with self.lock:
            uptime = time.time() - self.metrics["start_time"]
            success_rate = (
                self.metrics["operations_successful"] / max(self.metrics["operations_total"], 1)
            )
            
            status = {
                "status": "healthy" if success_rate > 0.95 else "degraded" if success_rate > 0.8 else "unhealthy",
                "uptime_seconds": uptime,
                "success_rate": success_rate,
                "total_operations": self.metrics["operations_total"],
                "total_errors": self.metrics["errors_total"],
                "errors_by_type": self.metrics["errors_by_type"].copy(),
                "last_error_time": self.metrics["last_error_time"]
            }
            
            return status
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self.lock:
            self.metrics = {
                "errors_total": 0,
                "errors_by_type": {},
                "last_error_time": None,
                "start_time": time.time(),
                "operations_total": 0,
                "operations_successful": 0
            }


# Global health monitor instance
_health_monitor = HealthMonitor()


def robust_operation(max_retries: int = 3, monitor_health: bool = True):
    """
    Decorator combining retry mechanism and health monitoring
    """
    def decorator(func: Callable) -> Callable:
        retry_handler = RetryMechanism(max_retries=max_retries)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = retry_handler(func)(*args, **kwargs)
                
                if monitor_health:
                    _health_monitor.record_operation(success=True)
                
                return result
                
            except HEGraphBaseException as e:
                if monitor_health:
                    _health_monitor.record_operation(success=False, error_type=e.error_code)
                raise
            
            except Exception as e:
                if monitor_health:
                    _health_monitor.record_operation(success=False, error_type="UNEXPECTED_ERROR")
                raise ComputationError(
                    f"Unexpected error in {func.__name__}: {e}",
                    context={"function": func.__name__, "original_error": str(e)}
                )
        
        return wrapper
    return decorator


def get_system_health() -> Dict[str, Any]:
    """Get current system health status"""
    return _health_monitor.get_health_status()


def reset_health_metrics():
    """Reset health monitoring metrics"""
    _health_monitor.reset_metrics()


# Example usage and testing
@robust_operation(max_retries=2)
def example_he_operation(data: Any) -> Any:
    """Example function demonstrating robust error handling"""
    # Validate input
    if data is None:
        raise ValidationError("Input data cannot be None", field_name="data")
    
    # Simulate computation
    logger.info(f"Processing data of type {type(data).__name__}")
    
    # Simulate potential failure
    import random
    if random.random() < 0.3:  # 30% chance of failure
        raise ComputationError("Simulated computation failure")
    
    return f"Processed: {data}"


def test_robust_error_handling():
    """Test the robust error handling framework"""
    logger.info("Testing robust error handling framework...")
    
    # Test validation
    try:
        RobustValidator.validate_graph_data(None, None)
    except ValidationError as e:
        logger.info(f"✅ Validation test passed: {e.error_code}")
    
    # Test retry mechanism
    for i in range(5):
        try:
            result = example_he_operation(f"test_data_{i}")
            logger.info(f"✅ Operation succeeded: {result}")
        except Exception as e:
            logger.error(f"❌ Operation failed: {e}")
    
    # Check health status
    health = get_system_health()
    logger.info(f"📊 System health: {health['status']} "
               f"(success rate: {health['success_rate']:.2%})")
    
    logger.info("🛡️ Robust error handling test complete!")


if __name__ == "__main__":
    test_robust_error_handling()
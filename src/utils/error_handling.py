"""
Comprehensive error handling utilities for HE-Graph-Embeddings
"""

import functools
import logging
import time
import traceback
from typing import Any, Callable, Dict, Optional, Type, Union
from datetime import datetime
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class HEGraphError(Exception):
    """Base exception for HE-Graph-Embeddings"""
    def __init__(self, message: str, error_code: str = None, 
                 details: Dict[str, Any] = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.severity = severity
        self.timestamp = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/API responses"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }

class EncryptionError(HEGraphError):
    """Errors related to homomorphic encryption operations"""
    pass

class GraphProcessingError(HEGraphError):
    """Errors in graph neural network operations"""
    pass

class ValidationError(HEGraphError):
    """Input validation errors"""
    pass

class SecurityError(HEGraphError):
    """Security-related errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.HIGH, **kwargs)

class MemoryError(HEGraphError):
    """Memory allocation and management errors"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=ErrorSeverity.CRITICAL, **kwargs)

class CUDAError(HEGraphError):
    """CUDA/GPU related errors"""
    pass

class ConfigurationError(HEGraphError):
    """Configuration and setup errors"""
    pass

class NetworkError(HEGraphError):
    """Network and API related errors"""
    pass

def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Decorator to retry function calls on specified exceptions"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        break
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time:.2f}s")
                    time.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

def async_retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """Async version of retry decorator"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Async function {func.__name__} failed after {max_retries + 1} attempts: {e}")
                        break
                    
                    wait_time = delay * (backoff ** attempt)
                    logger.warning(f"Async attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

def handle_exceptions(
    default_return: Any = None,
    log_errors: bool = True,
    reraise_as: Type[Exception] = None
) -> Callable:
    """Decorator to handle exceptions with configurable behavior"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                
                if reraise_as:
                    raise reraise_as(f"Error in {func.__name__}: {str(e)}") from e
                
                if default_return is not None:
                    return default_return
                
                raise
        
        return wrapper
    return decorator

class ErrorRecovery:
    """Error recovery strategies"""
    
    @staticmethod
    def recover_memory_error(func: Callable, *args, **kwargs) -> Any:
        """Attempt to recover from memory errors by reducing batch sizes"""
        original_batch_size = kwargs.get('batch_size', 32)
        
        for reduction_factor in [0.5, 0.25, 0.1]:
            try:
                kwargs['batch_size'] = max(1, int(original_batch_size * reduction_factor))
                logger.info(f"Retrying with reduced batch size: {kwargs['batch_size']}")
                return func(*args, **kwargs)
            except MemoryError:
                continue
        
        raise MemoryError("Unable to recover from memory error even with minimum batch size")
    
    @staticmethod
    def recover_cuda_error(func: Callable, *args, **kwargs) -> Any:
        """Attempt to recover from CUDA errors by falling back to CPU"""
        try:
            return func(*args, **kwargs)
        except CUDAError as e:
            logger.warning(f"CUDA error encountered: {e}. Falling back to CPU computation")
            kwargs['device'] = 'cpu'
            kwargs['use_gpu'] = False
            return func(*args, **kwargs)
    
    @staticmethod
    def recover_network_error(func: Callable, *args, **kwargs) -> Any:
        """Attempt to recover from network errors with exponential backoff"""
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except NetworkError as e:
                if attempt == max_retries - 1:
                    raise
                
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Network error on attempt {attempt + 1}: {e}. Retrying in {delay}s")
                time.sleep(delay)

class CircuitBreaker:
    """Circuit breaker pattern for handling recurring failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time < self.recovery_timeout:
                raise HEGraphError("Circuit breaker is OPEN", error_code="CIRCUIT_BREAKER_OPEN")
            else:
                self.state = "HALF_OPEN"
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"Circuit breaker opened after {self.failure_count} failures")

class ErrorMetrics:
    """Track error metrics for monitoring"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_rates = {}
        self.last_error_time = {}
    
    def record_error(self, error: HEGraphError):
        """Record an error for metrics"""
        error_type = error.__class__.__name__
        
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_error_time[error_type] = time.time()
        
        # Calculate error rate (errors per minute)
        current_time = time.time()
        window_start = current_time - 60  # 1 minute window
        
        recent_errors = sum(
            1 for timestamp in self.last_error_time.values()
            if timestamp >= window_start
        )
        
        self.error_rates[error_type] = recent_errors
        
        logger.info(f"Error metrics updated: {error_type} count={self.error_counts[error_type]}, rate={recent_errors}/min")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current error metrics"""
        return {
            "error_counts": self.error_counts.copy(),
            "error_rates": self.error_rates.copy(),
            "total_errors": sum(self.error_counts.values())
        }

# Global error metrics instance
error_metrics = ErrorMetrics()

def log_and_track_error(error: HEGraphError):
    """Log error and update metrics"""
    logger.error(f"HEGraph Error: {error.to_dict()}")
    error_metrics.record_error(error)

# Context manager for error handling
class ErrorHandler:
    """Context manager for comprehensive error handling"""
    
    def __init__(self, operation_name: str, critical: bool = False):
        self.operation_name = operation_name
        self.critical = critical
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            logger.info(f"Operation completed successfully: {self.operation_name} ({duration:.3f}s)")
            return False
        
        # Convert to HEGraph error if not already
        if not isinstance(exc_val, HEGraphError):
            severity = ErrorSeverity.CRITICAL if self.critical else ErrorSeverity.MEDIUM
            he_error = HEGraphError(
                message=f"Error in {self.operation_name}: {str(exc_val)}",
                error_code=exc_type.__name__,
                details={"duration": duration, "traceback": traceback.format_exc()},
                severity=severity
            )
        else:
            he_error = exc_val
        
        log_and_track_error(he_error)
        
        # Don't suppress the exception
        return False

# Decorators for common error handling patterns
def secure_endpoint(func: Callable) -> Callable:
    """Decorator for API endpoints with security error handling"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except SecurityError as e:
            logger.critical(f"Security error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in secure endpoint {func.__name__}: {e}")
            raise SecurityError(f"Internal error in secure operation") from e
    return wrapper

def gpu_fallback(func: Callable) -> Callable:
    """Decorator to fallback to CPU on GPU errors"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (CUDAError, RuntimeError) as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                logger.warning(f"GPU error in {func.__name__}: {e}. Falling back to CPU")
                kwargs['device'] = 'cpu'
                return func(*args, **kwargs)
            raise
    return wrapper
"""
Advanced logging system with correlation IDs, structured logs, and monitoring integration
"""

import logging
import json
import time
import uuid
import threading
from typing import Any, Dict, Optional, List
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from enum import Enum
import os
import sys

class LogLevel(Enum):
    """Log levels with numeric values"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO  
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

@dataclass
class LogContext:
    """Structured log context"""
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add context if available
        if hasattr(record, 'context') and record.context:
            log_data.update(record.context.to_dict())
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)

class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Get correlation ID from thread local storage
        context = getattr(_local, 'log_context', None)
        if context:
            record.context = context
        else:
            record.context = None
        
        return True

class PerformanceFilter(logging.Filter):
    """Add performance metrics to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add performance context if available
        perf_context = getattr(_local, 'perf_context', None)
        if perf_context:
            if not hasattr(record, 'extra_fields'):
                record.extra_fields = {}
            record.extra_fields.update(perf_context)
        
        return True

# Thread-local storage for context
_local = threading.local()

class HEGraphLogger:
    """Enhanced logger with correlation tracking and structured logging"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
        self._setup_logger()
    
    def _setup_logger(self):
        """Configure logger with structured formatting and filters"""
        if not self.logger.handlers:
            # Console handler with structured format
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(StructuredFormatter())
            console_handler.addFilter(CorrelationFilter())
            console_handler.addFilter(PerformanceFilter())
            
            # File handler for persistent logs
            file_handler = logging.FileHandler('logs/hegraph.log', encoding='utf-8')
            file_handler.setFormatter(StructuredFormatter())
            file_handler.addFilter(CorrelationFilter())
            file_handler.addFilter(PerformanceFilter())
            
            # Error file handler
            error_handler = logging.FileHandler('logs/hegraph_errors.log', encoding='utf-8')
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(StructuredFormatter())
            error_handler.addFilter(CorrelationFilter())
            
            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler) 
            self.logger.addHandler(error_handler)
            self.logger.setLevel(logging.INFO)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with current context"""
        extra = kwargs.pop('extra', {})
        if kwargs:
            extra['extra_fields'] = kwargs
        
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        kwargs['exc_info'] = True
        self.error(message, **kwargs)

def set_log_context(context: LogContext):
    """Set logging context for current thread"""
    _local.log_context = context

def get_log_context() -> Optional[LogContext]:
    """Get current logging context"""
    return getattr(_local, 'log_context', None)

def clear_log_context():
    """Clear current logging context"""
    if hasattr(_local, 'log_context'):
        delattr(_local, 'log_context')

@contextmanager
def log_context(correlation_id: str = None, **kwargs):
    """Context manager for logging context"""
    correlation_id = correlation_id or str(uuid.uuid4())
    context = LogContext(correlation_id=correlation_id, **kwargs)
    
    old_context = get_log_context()
    set_log_context(context)
    
    try:
        yield context
    finally:
        if old_context:
            set_log_context(old_context)
        else:
            clear_log_context()

@contextmanager
def performance_context(**metrics):
    """Context manager for performance metrics"""
    _local.perf_context = metrics
    try:
        yield
    finally:
        if hasattr(_local, 'perf_context'):
            delattr(_local, 'perf_context')

class PerformanceLogger:
    """Logger for performance metrics and profiling"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.logger = get_logger(f"perf.{operation_name}")
        self.metrics = {}
    
    def start(self):
        """Start timing"""
        self.start_time = time.perf_counter()
        self.logger.info(f"Started {self.operation_name}")
    
    def stop(self, **extra_metrics):
        """Stop timing and log results"""
        if self.start_time is None:
            raise ValueError("PerformanceLogger not started")
        
        duration = time.perf_counter() - self.start_time
        self.metrics.update(extra_metrics)
        self.metrics['duration_ms'] = duration * 1000
        
        self.logger.info(
            f"Completed {self.operation_name}",
            **self.metrics
        )
        
        return duration
    
    def checkpoint(self, name: str, **metrics):
        """Log intermediate checkpoint"""
        if self.start_time is None:
            raise ValueError("PerformanceLogger not started")
        
        elapsed = time.perf_counter() - self.start_time
        self.logger.info(
            f"Checkpoint {name} in {self.operation_name}",
            elapsed_ms=elapsed * 1000,
            **metrics
        )
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "failed" if exc_type else "success"
        self.stop(status=status)

class SecurityLogger:
    """Specialized logger for security events"""
    
    def __init__(self):
        self.logger = get_logger("security")
    
    def authentication_attempt(self, user_id: str, success: bool, 
                             ip_address: str, user_agent: str = None):
        """Log authentication attempt"""
        self.logger.info(
            f"Authentication {'successful' if success else 'failed'}",
            event_type="authentication",
            user_id=user_id,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent
        )
    
    def access_attempt(self, user_id: str, resource: str, action: str, 
                      success: bool, reason: str = None):
        """Log access attempt"""
        self.logger.info(
            f"Access attempt: {action} on {resource}",
            event_type="access_control",
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            reason=reason
        )
    
    def suspicious_activity(self, description: str, user_id: str = None, 
                          ip_address: str = None, **details):
        """Log suspicious activity"""
        self.logger.warning(
            f"Suspicious activity: {description}",
            event_type="suspicious_activity",
            user_id=user_id,
            ip_address=ip_address,
            **details
        )
    
    def security_violation(self, violation_type: str, description: str,
                          severity: str = "medium", **details):
        """Log security violation"""
        log_method = self.logger.critical if severity == "high" else self.logger.error
        log_method(
            f"Security violation: {description}",
            event_type="security_violation",
            violation_type=violation_type,
            severity=severity,
            **details
        )

class AuditLogger:
    """Logger for audit trails"""
    
    def __init__(self):
        self.logger = get_logger("audit")
    
    def data_access(self, user_id: str, data_type: str, operation: str,
                   record_count: int = None, **metadata):
        """Log data access"""
        self.logger.info(
            f"Data access: {operation} on {data_type}",
            event_type="data_access",
            user_id=user_id,
            data_type=data_type,
            operation=operation,
            record_count=record_count,
            **metadata
        )
    
    def model_training(self, user_id: str, model_name: str, 
                      dataset_info: Dict[str, Any], **parameters):
        """Log model training"""
        self.logger.info(
            f"Model training started: {model_name}",
            event_type="model_training",
            user_id=user_id,
            model_name=model_name,
            dataset_info=dataset_info,
            **parameters
        )
    
    def encryption_operation(self, user_id: str, operation: str,
                           data_size: int, context_name: str):
        """Log encryption operations"""
        self.logger.info(
            f"Encryption operation: {operation}",
            event_type="encryption",
            user_id=user_id,
            operation=operation,
            data_size=data_size,
            context_name=context_name
        )

# Global loggers
_loggers = {}
security_logger = SecurityLogger()
audit_logger = AuditLogger()

def get_logger(name: str) -> HEGraphLogger:
    """Get or create logger instance"""
    if name not in _loggers:
        _loggers[name] = HEGraphLogger(name)
    return _loggers[name]

def setup_logging(log_level: str = "INFO", log_dir: str = "logs"):
    """Setup global logging configuration"""
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Set global log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Remove default handlers
    
    # Add structured console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    console_handler.addFilter(CorrelationFilter())
    root_logger.addHandler(console_handler)

# Decorators for logging
def log_function_call(logger_name: str = None):
    """Decorator to log function calls with performance metrics"""
    def decorator(func):
        logger = get_logger(logger_name or func.__module__)
        
        def wrapper(*args, **kwargs):
            with PerformanceLogger(func.__name__) as perf:
                try:
                    result = func(*args, **kwargs)
                    logger.debug(f"Function {func.__name__} completed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Function {func.__name__} failed: {str(e)}")
                    raise
        
        return wrapper
    return decorator

def log_async_function_call(logger_name: str = None):
    """Decorator to log async function calls"""
    def decorator(func):
        logger = get_logger(logger_name or func.__module__)
        
        async def wrapper(*args, **kwargs):
            with PerformanceLogger(func.__name__) as perf:
                try:
                    result = await func(*args, **kwargs)
                    logger.debug(f"Async function {func.__name__} completed successfully")
                    return result
                except Exception as e:
                    logger.error(f"Async function {func.__name__} failed: {str(e)}")
                    raise
        
        return wrapper
    return decorator

# Initialize logging on import
setup_logging()
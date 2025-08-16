"""
🛡️ GENERATION 2: HE Health Monitoring and Diagnostics

This module provides comprehensive health monitoring specifically for homomorphic
encryption operations, including noise budget tracking, performance monitoring,
and automatic optimization suggestions.

Key Features:
- Real-time noise budget monitoring with alerts
- Performance metrics collection and analysis
- Automatic parameter optimization recommendations
- Circuit breaker integration for HE operations
- Health dashboard and reporting
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict, deque

from .robust_error_handling import (
    HEGraphBaseException, ErrorSeverity, RecoveryStrategy,
    HealthMonitor
)

logger = logging.getLogger(__name__)


class NoiseLevel(Enum):
    """Noise budget levels for HE operations"""
    HEALTHY = "healthy"      # > 20 bits remaining
    WARNING = "warning"      # 10-20 bits remaining
    CRITICAL = "critical"    # 5-10 bits remaining
    EXHAUSTED = "exhausted"  # < 5 bits remaining


class PerformanceLevel(Enum):
    """Performance classification levels"""
    EXCELLENT = "excellent"  # < 100ms per operation
    GOOD = "good"           # 100ms - 1s per operation
    ACCEPTABLE = "acceptable"  # 1s - 10s per operation
    POOR = "poor"           # > 10s per operation


@dataclass
class NoiseMetrics:
    """Noise budget tracking metrics"""
    initial_budget: float
    current_budget: float
    operations_count: int
    average_consumption: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def remaining_percentage(self) -> float:
        """Calculate remaining noise budget as percentage"""
        if self.initial_budget <= 0:
            return 0.0
        return (self.current_budget / self.initial_budget) * 100
    
    @property
    def noise_level(self) -> NoiseLevel:
        """Classify current noise level"""
        if self.current_budget > 20:
            return NoiseLevel.HEALTHY
        elif self.current_budget > 10:
            return NoiseLevel.WARNING
        elif self.current_budget > 5:
            return NoiseLevel.CRITICAL
        else:
            return NoiseLevel.EXHAUSTED


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    operation_name: str
    duration: float
    timestamp: float = field(default_factory=time.time)
    memory_usage: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def performance_level(self) -> PerformanceLevel:
        """Classify performance level"""
        if self.duration < 0.1:
            return PerformanceLevel.EXCELLENT
        elif self.duration < 1.0:
            return PerformanceLevel.GOOD
        elif self.duration < 10.0:
            return PerformanceLevel.ACCEPTABLE
        else:
            return PerformanceLevel.POOR


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation data"""
    category: str
    priority: str  # high, medium, low
    message: str
    technical_details: Dict[str, Any]
    estimated_improvement: Optional[str] = None


class HEHealthMonitor:
    """
    Comprehensive health monitoring for homomorphic encryption operations
    """
    
    def __init__(self, 
                 max_history_size: int = 1000,
                 noise_warning_threshold: float = 20.0,
                 noise_critical_threshold: float = 10.0):
        self.max_history_size = max_history_size
        self.noise_warning_threshold = noise_warning_threshold
        self.noise_critical_threshold = noise_critical_threshold
        
        # Thread-safe data structures
        self.lock = threading.RLock()
        
        # Metrics storage
        self.noise_history: deque = deque(maxlen=max_history_size)
        self.performance_history: deque = deque(maxlen=max_history_size)
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        
        # Current state
        self.current_noise_budget = None
        self.initial_noise_budget = None
        self.operations_since_refresh = 0
        
        # Alert flags
        self.noise_alerts_enabled = True
        self.performance_alerts_enabled = True
        
        # Start time
        self.start_time = time.time()
        
        logger.info("HE Health Monitor initialized")
    
    def track_noise_budget(self, current_budget: float, operation_name: str = "unknown"):
        """Track noise budget consumption"""
        with self.lock:
            if self.initial_noise_budget is None:
                self.initial_noise_budget = current_budget
                logger.info(f"Initial noise budget set: {current_budget:.2f} bits")
            
            self.current_noise_budget = current_budget
            self.operations_since_refresh += 1
            
            # Calculate average consumption
            if self.operations_since_refresh > 0:
                total_consumed = self.initial_noise_budget - current_budget
                avg_consumption = total_consumed / self.operations_since_refresh
            else:
                avg_consumption = 0.0
            
            # Create noise metrics
            metrics = NoiseMetrics(
                initial_budget=self.initial_noise_budget,
                current_budget=current_budget,
                operations_count=self.operations_since_refresh,
                average_consumption=avg_consumption
            )
            
            self.noise_history.append(metrics)
            
            # Check for alerts
            if self.noise_alerts_enabled:
                self._check_noise_alerts(metrics, operation_name)
    
    def track_performance(self, operation_name: str, duration: float, 
                         success: bool = True, error_message: Optional[str] = None,
                         memory_usage: Optional[float] = None):
        """Track operation performance"""
        with self.lock:
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration=duration,
                success=success,
                error_message=error_message,
                memory_usage=memory_usage
            )
            
            self.performance_history.append(metrics)
            self.operation_counts[operation_name] += 1
            
            if not success:
                self.error_counts[operation_name] += 1
            
            # Check for performance alerts
            if self.performance_alerts_enabled:
                self._check_performance_alerts(metrics)
    
    def _check_noise_alerts(self, metrics: NoiseMetrics, operation_name: str):
        """Check and emit noise budget alerts"""
        level = metrics.noise_level
        
        if level == NoiseLevel.EXHAUSTED:
            logger.critical(
                f"NOISE EXHAUSTED: {metrics.current_budget:.2f} bits remaining "
                f"after {operation_name}. Immediate bootstrapping required!"
            )
        elif level == NoiseLevel.CRITICAL:
            logger.error(
                f"CRITICAL NOISE: {metrics.current_budget:.2f} bits remaining "
                f"after {operation_name}. Consider bootstrapping soon."
            )
        elif level == NoiseLevel.WARNING:
            logger.warning(
                f"LOW NOISE: {metrics.current_budget:.2f} bits remaining "
                f"after {operation_name}. Plan for bootstrapping."
            )
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check and emit performance alerts"""
        level = metrics.performance_level
        
        if level == PerformanceLevel.POOR:
            logger.warning(
                f"SLOW OPERATION: {metrics.operation_name} took {metrics.duration:.3f}s. "
                "Consider optimization."
            )
        
        if not metrics.success:
            logger.error(
                f"OPERATION FAILED: {metrics.operation_name} - {metrics.error_message}"
            )
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current health status"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            # Calculate success rates
            total_ops = sum(self.operation_counts.values())
            total_errors = sum(self.error_counts.values())
            success_rate = (total_ops - total_errors) / max(total_ops, 1)
            
            # Get recent performance metrics
            recent_perf = list(self.performance_history)[-10:] if self.performance_history else []
            avg_recent_duration = (
                sum(m.duration for m in recent_perf) / len(recent_perf)
                if recent_perf else 0.0
            )
            
            # Noise status
            noise_status = "unknown"
            if self.current_noise_budget is not None:
                if self.current_noise_budget > self.noise_warning_threshold:
                    noise_status = "healthy"
                elif self.current_noise_budget > self.noise_critical_threshold:
                    noise_status = "warning"
                else:
                    noise_status = "critical"
            
            return {
                "timestamp": time.time(),
                "uptime_seconds": uptime,
                "total_operations": total_ops,
                "success_rate": success_rate,
                "average_recent_duration": avg_recent_duration,
                "noise_budget_remaining": self.current_noise_budget,
                "noise_status": noise_status,
                "operations_since_refresh": self.operations_since_refresh,
                "operation_counts": dict(self.operation_counts),
                "error_counts": dict(self.error_counts)
            }
    
    def generate_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on collected metrics"""
        recommendations = []
        
        with self.lock:
            status = self.get_current_status()
            
            # Noise budget recommendations
            if self.current_noise_budget is not None:
                if self.current_noise_budget < self.noise_critical_threshold:
                    recommendations.append(OptimizationRecommendation(
                        category="noise_management",
                        priority="high",
                        message="Noise budget critically low. Bootstrap immediately.",
                        technical_details={
                            "current_budget": self.current_noise_budget,
                            "threshold": self.noise_critical_threshold,
                            "operations_count": self.operations_since_refresh
                        },
                        estimated_improvement="Restore full noise budget"
                    ))
                elif self.current_noise_budget < self.noise_warning_threshold:
                    recommendations.append(OptimizationRecommendation(
                        category="noise_management",
                        priority="medium",
                        message="Consider bootstrapping or reducing operation complexity.",
                        technical_details={
                            "current_budget": self.current_noise_budget,
                            "threshold": self.noise_warning_threshold
                        },
                        estimated_improvement="Extend computation capability"
                    ))
            
            # Performance recommendations
            if status["average_recent_duration"] > 5.0:
                recommendations.append(OptimizationRecommendation(
                    category="performance",
                    priority="medium",
                    message="Recent operations are slow. Consider parameter optimization.",
                    technical_details={
                        "avg_duration": status["average_recent_duration"],
                        "total_operations": status["total_operations"]
                    },
                    estimated_improvement="Up to 50% performance improvement"
                ))
            
            # Success rate recommendations
            if status["success_rate"] < 0.9:
                recommendations.append(OptimizationRecommendation(
                    category="reliability",
                    priority="high",
                    message="Low success rate detected. Review error patterns.",
                    technical_details={
                        "success_rate": status["success_rate"],
                        "error_counts": status["error_counts"]
                    },
                    estimated_improvement="Improved reliability"
                ))
            
            # Parameter recommendations
            if self.operations_since_refresh > 100:
                if self.current_noise_budget and self.current_noise_budget > 50:
                    recommendations.append(OptimizationRecommendation(
                        category="parameters",
                        priority="low",
                        message="Noise budget usage is very conservative. Consider smaller parameters.",
                        technical_details={
                            "operations_count": self.operations_since_refresh,
                            "remaining_budget": self.current_noise_budget
                        },
                        estimated_improvement="Faster operations with same security"
                    ))
        
        return recommendations
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for analysis"""
        with self.lock:
            return {
                "timestamp": time.time(),
                "noise_history": [
                    {
                        "timestamp": m.timestamp,
                        "current_budget": m.current_budget,
                        "remaining_percentage": m.remaining_percentage,
                        "operations_count": m.operations_count,
                        "noise_level": m.noise_level.value
                    }
                    for m in self.noise_history
                ],
                "performance_history": [
                    {
                        "timestamp": m.timestamp,
                        "operation_name": m.operation_name,
                        "duration": m.duration,
                        "success": m.success,
                        "performance_level": m.performance_level.value,
                        "memory_usage": m.memory_usage
                    }
                    for m in self.performance_history
                ],
                "operation_counts": dict(self.operation_counts),
                "error_counts": dict(self.error_counts),
                "current_status": self.get_current_status(),
                "recommendations": [
                    {
                        "category": r.category,
                        "priority": r.priority,
                        "message": r.message,
                        "technical_details": r.technical_details,
                        "estimated_improvement": r.estimated_improvement
                    }
                    for r in self.generate_recommendations()
                ]
            }
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        with self.lock:
            self.noise_history.clear()
            self.performance_history.clear()
            self.operation_counts.clear()
            self.error_counts.clear()
            self.current_noise_budget = None
            self.initial_noise_budget = None
            self.operations_since_refresh = 0
            self.start_time = time.time()
            
            logger.info("HE Health Monitor metrics reset")


# Global HE health monitor instance
_he_health_monitor = HEHealthMonitor()


def track_he_operation(operation_name: str):
    """Decorator for tracking HE operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error_msg = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error_msg = str(e)
                raise
            finally:
                duration = time.time() - start_time
                _he_health_monitor.track_performance(
                    operation_name=operation_name,
                    duration=duration,
                    success=success,
                    error_message=error_msg
                )
        
        return wrapper
    return decorator


def get_he_health_status() -> Dict[str, Any]:
    """Get current HE health status"""
    return _he_health_monitor.get_current_status()


def get_he_recommendations() -> List[Dict[str, Any]]:
    """Get HE optimization recommendations"""
    recommendations = _he_health_monitor.generate_recommendations()
    return [
        {
            "category": r.category,
            "priority": r.priority, 
            "message": r.message,
            "technical_details": r.technical_details,
            "estimated_improvement": r.estimated_improvement
        }
        for r in recommendations
    ]


def export_he_metrics() -> Dict[str, Any]:
    """Export all HE metrics"""
    return _he_health_monitor.export_metrics()


def reset_he_metrics():
    """Reset HE health metrics"""
    _he_health_monitor.reset_metrics()


# Testing function
def test_he_health_monitoring():
    """Test HE health monitoring functionality"""
    logger.info("Testing HE health monitoring...")
    
    # Simulate noise budget tracking
    initial_budget = 60.0
    for i in range(10):
        remaining = initial_budget - (i * 2.5)
        _he_health_monitor.track_noise_budget(remaining, f"operation_{i}")
    
    # Simulate performance tracking
    import random
    for i in range(20):
        duration = random.uniform(0.1, 2.0)
        success = random.random() > 0.1  # 90% success rate
        _he_health_monitor.track_performance(
            f"test_op_{i % 3}",
            duration,
            success=success,
            error_message="Test error" if not success else None
        )
    
    # Get status and recommendations
    status = get_he_health_status()
    recommendations = get_he_recommendations()
    
    logger.info(f"📊 Current status: {status['noise_status']}")
    logger.info(f"🎯 Success rate: {status['success_rate']:.2%}")
    logger.info(f"💡 Recommendations: {len(recommendations)}")
    
    for rec in recommendations:
        logger.info(f"   • {rec['priority'].upper()}: {rec['message']}")
    
    logger.info("🛡️ HE health monitoring test complete!")


if __name__ == "__main__":
    test_he_health_monitoring()
"""
Intelligent auto-scaling system for HE-Graph-Embeddings with predictive scaling
"""


import asyncio
import time
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
import statistics
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
import psutil
import torch

from .logging import get_logger, log_context, performance_context
from .error_handling import handle_exceptions, HEGraphError
from .monitoring import get_health_checker, MetricCollector

logger = get_logger(__name__)

@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    queue_depth: int = 0
    active_workers: int = 0
    request_rate: float = 0.0
    response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0

@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling policies"""
    # Resource thresholds
    cpu_scale_up_threshold: float = 70.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 80.0
    memory_scale_down_threshold: float = 40.0

    # Queue thresholds
    queue_scale_up_threshold: int = 50
    queue_scale_down_threshold: int = 5

    # Performance thresholds
    response_time_threshold: float = 5.0  # seconds
    error_rate_threshold: float = 0.05  # 5%

    # Scaling parameters
    min_instances: int = 1
    max_instances: int = 20
    scale_up_cooldown: float = 300.0  # 5 minutes
    scale_down_cooldown: float = 600.0  # 10 minutes

    # Advanced parameters
    enable_predictive_scaling: bool = True
    enable_scheduled_scaling: bool = True
    aggressiveness: float = 1.0  # Scaling aggressiveness factor

    # Instance types and capabilities
    instance_types: Dict[str, Dict] = field(default_factory=lambda: {
        "small": {"cpu_cores": 2, "memory_gb": 8, "gpu_memory_gb": 0},
        "medium": {"cpu_cores": 4, "memory_gb": 16, "gpu_memory_gb": 8},
        "large": {"cpu_cores": 8, "memory_gb": 32, "gpu_memory_gb": 16},
        "xlarge": {"cpu_cores": 16, "memory_gb": 64, "gpu_memory_gb": 32}
    })

class MetricsPrediction:
    """Predictive model for scaling metrics"""

    def __init__(self, history_size: int = 1440):  # 24 hours of minutes
        """  Init  ."""
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.model_weights = None
        self.last_training = 0
        self.training_interval = 3600  # Retrain every hour

    def add_metrics(self, metrics: ScalingMetrics) -> None:
        """Add metrics to history"""
        self.metrics_history.append(metrics)

    def predict_load(self, minutes_ahead: int = 15) -> Dict[str, float]:
        """Predict system load for future time"""
        if len(self.metrics_history) < 60:  # Need at least 1 hour of data
            return self._get_current_metrics_dict()

        # Simple time series forecasting using linear regression
        try:
            # Extract features for prediction
            cpu_values = [m.cpu_usage for m in list(self.metrics_history)[-60:]]
            memory_values = [m.memory_usage for m in list(self.metrics_history)[-60:]]
            queue_values = [m.queue_depth for m in list(self.metrics_history)[-60:]]

            # Simple trend calculation
            cpu_trend = self._calculate_trend(cpu_values)
            memory_trend = self._calculate_trend(memory_values)
            queue_trend = self._calculate_trend(queue_values)

            # Predict future values
            current_cpu = cpu_values[-1]
            current_memory = memory_values[-1]
            current_queue = queue_values[-1]

            predicted_cpu = max(0, min(100, current_cpu + cpu_trend * minutes_ahead))
            predicted_memory = max(0, min(100, current_memory + memory_trend * minutes_ahead))
            predicted_queue = max(0, current_queue + queue_trend * minutes_ahead)

            # Add seasonal patterns (simple daily cycle)
            hour_of_day = datetime.now().hour
            seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * hour_of_day / 24)

            return {
                "cpu_usage": predicted_cpu * seasonal_factor,
                "memory_usage": predicted_memory * seasonal_factor,
                "queue_depth": predicted_queue * seasonal_factor,
                "confidence": min(1.0, len(cpu_values) / 60.0)
            }

        except Exception as e:
            logger.error(f"Error in load prediction: {e}")
            return self._get_current_metrics_dict()

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using simple linear regression"""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x = list(range(n))
        y = values

        # Linear regression slope calculation
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope

    def _get_current_metrics_dict(self) -> Dict[str, float]:
        """Get current metrics as dict"""
        if not self.metrics_history:
            return {"cpu_usage": 50, "memory_usage": 50, "queue_depth": 0, "confidence": 0.0}

        latest = self.metrics_history[-1]
        return {
            "cpu_usage": latest.cpu_usage,
            "memory_usage": latest.memory_usage,
            "queue_depth": latest.queue_depth,
            "confidence": 1.0
        }

    def detect_anomalies(self) -> List[str]:
        """Detect anomalous patterns that might require scaling"""
        anomalies = []

        if len(self.metrics_history) < 20:
            return anomalies

        recent_metrics = list(self.metrics_history)[-20:]

        # Check for sudden spikes
        cpu_values = [m.cpu_usage for m in recent_metrics]
        if len(cpu_values) >= 5:
            recent_avg = statistics.mean(cpu_values[-5:])
            older_avg = statistics.mean(cpu_values[-15:-5]) if len(cpu_values) >= 15 else recent_avg

            if recent_avg > older_avg * 1.5:
                anomalies.append("cpu_spike")

        # Check for sustained high error rates
        error_rates = [m.error_rate for m in recent_metrics]
        if statistics.mean(error_rates) > 0.1:  # 10% error rate
            anomalies.append("high_error_rate")

        # Check for memory pressure
        memory_values = [m.memory_usage for m in recent_metrics]
        if statistics.mean(memory_values[-5:]) > 90:
            anomalies.append("memory_pressure")

        return anomalies

class ScalingDecisionEngine:
    """Engine for making intelligent scaling decisions"""

    def __init__(self, policy: ScalingPolicy):
        """  Init  ."""
        self.policy = policy
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.scaling_history = deque(maxlen=100)
        self.predictor = MetricsPrediction()

    def should_scale_up(self, metrics: ScalingMetrics, current_instances: int) -> Tuple[bool, str, int]:
        """Determine if should scale up and by how many instances"""
        if current_instances >= self.policy.max_instances:
            return False, "max_instances_reached", 0

        if time.time() - self.last_scale_up < self.policy.scale_up_cooldown:
            return False, "cooldown_period", 0

        # Add metrics to predictor
        self.predictor.add_metrics(metrics)

        # Check immediate thresholds
        scale_reasons = []
        urgency_score = 0

        # CPU-based scaling
        if metrics.cpu_usage > self.policy.cpu_scale_up_threshold:
            scale_reasons.append(f"high_cpu({metrics.cpu_usage:.1f}%)")
            urgency_score += (metrics.cpu_usage - self.policy.cpu_scale_up_threshold) / 100

        # Memory-based scaling
        if metrics.memory_usage > self.policy.memory_scale_up_threshold:
            scale_reasons.append(f"high_memory({metrics.memory_usage:.1f}%)")
            urgency_score += (metrics.memory_usage - self.policy.memory_scale_up_threshold) / 100

        # Queue-based scaling
        if metrics.queue_depth > self.policy.queue_scale_up_threshold:
            scale_reasons.append(f"high_queue({metrics.queue_depth})")
            urgency_score += metrics.queue_depth / 100

        # Performance-based scaling
        if metrics.response_time > self.policy.response_time_threshold:
            scale_reasons.append(f"slow_response({metrics.response_time:.2f}s)")
            urgency_score += metrics.response_time / 10

        if metrics.error_rate > self.policy.error_rate_threshold:
            scale_reasons.append(f"high_errors({metrics.error_rate:.1%})")
            urgency_score += metrics.error_rate * 10

        # Predictive scaling
        if self.policy.enable_predictive_scaling:
            predicted_load = self.predictor.predict_load(15)  # 15 minutes ahead
            if predicted_load["cpu_usage"] > self.policy.cpu_scale_up_threshold * 0.9:
                scale_reasons.append("predicted_high_cpu")
                urgency_score += 0.3

        # Anomaly detection
        anomalies = self.predictor.detect_anomalies()
        if anomalies:
            scale_reasons.extend(anomalies)
            urgency_score += len(anomalies) * 0.2

        # Determine scale amount based on urgency
        if scale_reasons:
            # Calculate scale amount
            if urgency_score > 2.0:
                scale_amount = min(3, self.policy.max_instances - current_instances)
            elif urgency_score > 1.0:
                scale_amount = min(2, self.policy.max_instances - current_instances)
            else:
                scale_amount = 1

            # Apply aggressiveness factor
            scale_amount = max(1, int(scale_amount * self.policy.aggressiveness))

            reason = f"scale_up: {', '.join(scale_reasons)} (urgency: {urgency_score:.2f})"
            return True, reason, scale_amount

        return False, "no_scale_needed", 0

    def should_scale_down(self, metrics: ScalingMetrics, current_instances: int) -> Tuple[bool, str, int]:
        """Determine if should scale down and by how many instances"""
        if current_instances <= self.policy.min_instances:
            return False, "min_instances_reached", 0

        if time.time() - self.last_scale_down < self.policy.scale_down_cooldown:
            return False, "cooldown_period", 0

        # Check if resources are underutilized
        underutilized_reasons = []

        if (metrics.cpu_usage < self.policy.cpu_scale_down_threshold and
            metrics.memory_usage < self.policy.memory_scale_down_threshold and
            metrics.queue_depth < self.policy.queue_scale_down_threshold):

            underutilized_reasons.append("low_resource_usage")

        # Check recent performance trends
        if len(self.scaling_history) > 0:
            recent_scale_ups = sum(1 for h in list(self.scaling_history)[-10:]
                                if h.get("action") == "scale_up")
            if recent_scale_ups == 0:  # No recent scale-ups
                underutilized_reasons.append("no_recent_scale_ups")

        if underutilized_reasons:
            # Conservative scale down - usually just 1 instance
            scale_amount = 1

            # More aggressive scale down if very underutilized
            if (metrics.cpu_usage < self.policy.cpu_scale_down_threshold / 2 and
                metrics.memory_usage < self.policy.memory_scale_down_threshold / 2):
                scale_amount = min(2, current_instances - self.policy.min_instances)

            reason = f"scale_down: {', '.join(underutilized_reasons)}"
            return True, reason, scale_amount

        return False, "resources_in_use", 0

    def record_scaling_action(self, action: str, instances_before: int,
                            instances_after: int, reason: str):
        """Record Scaling Action."""
        """Record scaling action for future reference"""
        scaling_record = {
            "timestamp": time.time(),
            "action": action,
            "instances_before": instances_before,
            "instances_after": instances_after,
            "reason": reason
        }

        self.scaling_history.append(scaling_record)

        if action == "scale_up":
            self.last_scale_up = time.time()
        elif action == "scale_down":
            self.last_scale_down = time.time()

        logger.info(f"Scaling action recorded: {action} {instances_before}->{instances_after} ({reason})")

class AutoScaler:
    """Main auto-scaling orchestrator"""

    def __init__(self, policy: ScalingPolicy = None):
        """  Init  ."""
        self.policy = policy or ScalingPolicy()
        self.decision_engine = ScalingDecisionEngine(self.policy)
        self.metric_collector = MetricCollector()

        # Resource management
        self.current_instances = 0
        self.instance_registry = {}  # instance_id -> instance_info
        self.scaling_callbacks = []  # Functions to call when scaling

        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.monitor_interval = 30.0  # 30 seconds

        # Scheduled scaling
        self.scheduled_policies = []  # List of scheduled scaling policies

        logger.info("AutoScaler initialized")

    def register_scaling_callback(self, callback: Callable[[str, int, str], None]) -> None:
        """Register callback for scaling events"""
        self.scaling_callbacks.append(callback)

    def start_monitoring(self) -> None:
        """Start auto-scaling monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("Auto-scaling monitoring started")

    def stop_monitoring(self) -> None:
        """Stop auto-scaling monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        logger.info("Auto-scaling monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                with log_context(operation="autoscale_check"):
                    # Collect current metrics
                    metrics = self._collect_metrics()

                    # Make scaling decisions
                    self._evaluate_scaling(metrics)

                    # Check scheduled scaling
                    if self.policy.enable_scheduled_scaling:
                        self._check_scheduled_scaling()

                    # Record metrics
                    self._record_metrics(metrics)

                time.sleep(self.monitor_interval)

            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring: {e}")
                time.sleep(60)  # Wait longer on error

    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics"""
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # GPU metrics
        gpu_usage = 0.0
        gpu_memory_usage = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_usage = (gpu_memory_allocated / gpu_memory_total) * 100

                # GPU utilization would need nvidia-ml-py for accurate measurement
                # For now, estimate based on memory usage
                gpu_usage = min(gpu_memory_usage * 1.2, 100)
            except Exception:
                logger.error(f"Error in operation: {e}")
                pass

        # Application metrics from health checker
        health_checker = get_health_checker()
        queue_depth = 0
        active_workers = 0
        request_rate = 0.0
        response_time = 0.0
        error_rate = 0.0
        throughput = 0.0

        if health_checker:
            # Get queue depth from metric collector
            latest_queue = health_checker.metric_collector.get_latest("queue.depth")
            if latest_queue:
                queue_depth = int(latest_queue)

            # Get worker count
            latest_workers = health_checker.metric_collector.get_latest("workers.active")
            if latest_workers:
                active_workers = int(latest_workers)

            # Get performance metrics
            response_time = health_checker.metric_collector.get_average("api.request_duration_ms", 5) or 0.0
            response_time = response_time / 1000  # Convert to seconds

            error_rate = health_checker.metric_collector.get_average("api.error_rate", 5) or 0.0

            # Calculate request rate and throughput
            request_rate = health_checker.metric_collector.get_average("api.requests_per_second", 1) or 0.0
            throughput = health_checker.metric_collector.get_average("throughput", 5) or 0.0

        return ScalingMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            queue_depth=queue_depth,
            active_workers=active_workers,
            request_rate=request_rate,
            response_time=response_time,
            error_rate=error_rate,
            throughput=throughput
        )

    def _evaluate_scaling(self, metrics: ScalingMetrics) -> None:
        """Evaluate and execute scaling decisions"""
        current_instances = self.current_instances

        # Check for scale up
        should_up, up_reason, up_amount = self.decision_engine.should_scale_up(metrics, current_instances)
        if should_up:
            new_instances = min(current_instances + up_amount, self.policy.max_instances)
            self._execute_scaling("scale_up", new_instances, up_reason)
            return

        # Check for scale down
        should_down, down_reason, down_amount = self.decision_engine.should_scale_down(metrics, current_instances)
        if should_down:
            new_instances = max(current_instances - down_amount, self.policy.min_instances)
            self._execute_scaling("scale_down", new_instances, down_reason)

    def _execute_scaling(self, action: str, target_instances: int, reason: str) -> None:
        """Execute scaling action"""
        current_instances = self.current_instances

        if target_instances == current_instances:
            return

        logger.info(f"Executing {action}: {current_instances} -> {target_instances} ({reason})")

        try:
            # Call registered callbacks
            for callback in self.scaling_callbacks:
                try:
                    callback(action, target_instances, reason)
                except Exception as e:
                    logger.error(f"Scaling callback error: {e}")

            # Update instance count
            self.current_instances = target_instances

            # Record the action
            self.decision_engine.record_scaling_action(
                action, current_instances, target_instances, reason
            )

        except Exception as e:
            logger.error(f"Failed to execute scaling {action}: {e}")

    def _check_scheduled_scaling(self) -> None:
        """Check and execute scheduled scaling policies"""
        current_time = datetime.now()

        for policy in self.scheduled_policies:
            if self._should_execute_scheduled_policy(policy, current_time):
                self._execute_scheduled_policy(policy)

    def _should_execute_scheduled_policy(self, policy: Dict, current_time: datetime) -> bool:
        """Check if scheduled policy should be executed"""
        # This would implement logic for scheduled scaling
        # e.g., scale up during business hours, scale down at night
        return False

    def _execute_scheduled_policy(self, policy: Dict) -> None:
        """Execute scheduled scaling policy"""
        pass  # Implement scheduled scaling logic

    def _record_metrics(self, metrics: ScalingMetrics) -> None:
        """Record metrics for monitoring and analysis"""
        if get_health_checker():
            collector = get_health_checker().metric_collector
            collector.record("autoscale.cpu_usage", metrics.cpu_usage)
            collector.record("autoscale.memory_usage", metrics.memory_usage)
            collector.record("autoscale.gpu_usage", metrics.gpu_usage)
            collector.record("autoscale.queue_depth", metrics.queue_depth)
            collector.record("autoscale.active_instances", self.current_instances)

    def add_scheduled_policy(self, policy: Dict) -> None:
        """Add scheduled scaling policy"""
        self.scheduled_policies.append(policy)

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics"""
        return {
            "current_instances": self.current_instances,
            "min_instances": self.policy.min_instances,
            "max_instances": self.policy.max_instances,
            "monitoring_active": self.monitoring_active,
            "last_scale_up": self.decision_engine.last_scale_up,
            "last_scale_down": self.decision_engine.last_scale_down,
            "scaling_history": list(self.decision_engine.scaling_history)[-10:],  # Last 10 actions
            "registered_callbacks": len(self.scaling_callbacks),
            "policy": {
                "cpu_thresholds": (self.policy.cpu_scale_down_threshold, self.policy.cpu_scale_up_threshold),
                "memory_thresholds": (self.policy.memory_scale_down_threshold, self.policy.memory_scale_up_threshold),
                "cooldown_periods": (self.policy.scale_down_cooldown, self.policy.scale_up_cooldown)
            }
        }

# Global auto-scaler instance
_global_auto_scaler = None

def get_auto_scaler(policy: ScalingPolicy = None) -> AutoScaler:
    """Get global auto-scaler instance"""
    global _global_auto_scaler

    if _global_auto_scaler is None:
        _global_auto_scaler = AutoScaler(policy)

    return _global_auto_scaler

def shutdown_auto_scaler():
    """Shutdown global auto-scaler"""
    global _global_auto_scaler

    if _global_auto_scaler:
        _global_auto_scaler.stop_monitoring()
        _global_auto_scaler = None

# Example scaling callback functions
def worker_pool_scaling_callback(action: str, target_instances: int, reason: str):
    """Example callback for scaling worker pool"""
    from .concurrent_processing import get_worker_pool

    worker_pool = get_worker_pool()
    current_workers = len(worker_pool.workers)

    if action == "scale_up" and target_instances > current_workers:
        # Add workers
        for _ in range(target_instances - current_workers):
            worker_pool._add_worker()

    elif action == "scale_down" and target_instances < current_workers:
        # Remove workers
        workers_to_remove = current_workers - target_instances
        worker_ids = list(worker_pool.workers.keys())[:workers_to_remove]

        for worker_id in worker_ids:
            worker_pool._remove_worker(worker_id)

# Utility functions for scaling analysis
def analyze_scaling_efficiency(scaler: AutoScaler, time_window_hours: int = 24) -> Dict[str, Any]:
    """Analyze scaling efficiency over time window"""
    cutoff_time = time.time() - (time_window_hours * 3600)

    recent_actions = [
        action for action in scaler.decision_engine.scaling_history
        if action["timestamp"] >= cutoff_time
    ]

    if not recent_actions:
        return {"analysis": "insufficient_data"}

    scale_ups = sum(1 for a in recent_actions if a["action"] == "scale_up")
    scale_downs = sum(1 for a in recent_actions if a["action"] == "scale_down")

    # Calculate scaling frequency
    total_time = time.time() - recent_actions[0]["timestamp"]
    avg_time_between_actions = total_time / len(recent_actions) if recent_actions else float('inf')

    return {
        "time_window_hours": time_window_hours,
        "total_actions": len(recent_actions),
        "scale_ups": scale_ups,
        "scale_downs": scale_downs,
        "avg_minutes_between_actions": avg_time_between_actions / 60,
        "scaling_stability": 1.0 - (abs(scale_ups - scale_downs) / max(1, len(recent_actions))),
        "efficiency_score": min(1.0, (scale_ups + scale_downs) / max(1, len(recent_actions)))
    }
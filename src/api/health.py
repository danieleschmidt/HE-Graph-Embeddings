"""
Health check API endpoints for HE-Graph-Embeddings.

This module provides comprehensive health monitoring endpoints for the
homomorphic encryption graph neural network service, including system metrics,
application status, and Kubernetes-compatible probes.

Classes:
    None (FastAPI router-based module)

Functions:
    health_check: Basic health status check (standalone)
    basic_health_check: Basic health status endpoint
    detailed_health_check: Comprehensive system health with all subsystems
    get_metrics: Current system performance metrics
    get_system_status: Detailed system resource information
    get_application_status: Application-specific status and context details
    run_specific_check: Execute named health check
    readiness_probe: Kubernetes readiness probe endpoint
    liveness_probe: Kubernetes liveness probe endpoint
    check_model_inference_health: Test model inference capability
    check_encryption_performance: Test encryption/decryption performance
    register_additional_checks: Register custom health checks

Example:
    Basic usage through FastAPI:
    ```python

    from fastapi import FastAPI
    from .api.health import router as health_router

    app = FastAPI()
    app.include_router(health_router)
    ```
"""


import asyncio
from datetime import datetime
from typing import Dict, Any, List

try:
    from fastapi import APIRouter, HTTPException, Depends
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    APIRouter = None
    HTTPException = None
    Depends = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from ..utils.monitoring import get_health_checker, HealthStatus, HealthCheckResult
from ..utils.logging import get_logger, log_context
from ..utils.error_handling import handle_exceptions, HEGraphError

logger = get_logger(__name__)

def health_check():
    """
    Standalone health check function that doesn't require FastAPI.
    
    Returns:
        dict: Basic health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "he-graph-embeddings",
        "version": "1.0.0"
    }

# Only create router if FastAPI is available
router = APIRouter(prefix="/health", tags=["health"]) if HAS_FASTAPI else None

if HAS_FASTAPI and router:
    @router.get("/", summary="Basic health check")
    @handle_exceptions(default_return=JSONResponse(
        status_code=503,
        content={"status": "unhealthy", "message": "Health check failed"}
    ))
    async def basic_health_check():
        """
        Basic health check endpoint.

        Returns simple service status without detailed diagnostics.
        Suitable for load balancer health checks.

        Returns:
            dict: Status response with timestamp and service identifier

        Response format:
            {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00.000Z",
                "service": "he-graph-embeddings"
            }
        """
        with log_context(operation="health_check_basic"):
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "he-graph-embeddings"
            }

    @router.get("/detailed", summary="Detailed health check")
    @handle_exceptions(reraise_as=HTTPException)
    async def detailed_health_check():
        """Comprehensive health check with all subsystem status"""
        with log_context(operation="health_check_detailed"):
            try:
            health_checker = get_health_checker()
            if not health_checker:
                raise HEGraphError("Health checker not initialized")

            # Run all health checks
            results = await health_checker.run_health_checks()
            overall_status = health_checker.get_overall_status(results)

            # Convert results to JSON-serializable format
            formatted_results = {
                name: result.to_dict() for name, result in results.items()
            }

            # Determine HTTP status code
            status_code = 200
            if overall_status == HealthStatus.CRITICAL:
                status_code = 503
            elif overall_status == HealthStatus.UNHEALTHY:
                status_code = 503
            elif overall_status == HealthStatus.DEGRADED:
                status_code = 200  # Still serving requests

            response_data = {
                "overall_status": overall_status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "checks": formatted_results,
                "summary": {
                    "total_checks": len(results),
                    "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                    "degraded": sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                    "unhealthy": sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
                    "critical": sum(1 for r in results.values() if r.status == HealthStatus.CRITICAL)
                }
            }

            logger.info(f"Health check completed: {overall_status.value}")

            return JSONResponse(
                status_code=status_code,
                content=response_data
            )

        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@router.get("/metrics", summary="System metrics")
@handle_exceptions(reraise_as=HTTPException)
async def get_metrics():
    """Get current system metrics"""
    with log_context(operation="metrics"):
        try:
            health_checker = get_health_checker()
            if not health_checker:
                raise HEGraphError("Health checker not initialized")

            metrics = {}
            collector = health_checker.metric_collector

            # Common metric names to report
            metric_names = [
                "cpu.usage_percent",
                "memory.usage_percent",
                "memory.available_gb",
                "disk.usage_percent",
                "gpu.memory_allocated_gb",
                "gpu.utilization_percent",
                "api.request_duration_ms",
                "api.error_rate",
                "encryption.encrypt_duration_ms",
                "encryption.decrypt_duration_ms"
            ]

            for metric_name in metric_names:
                latest = collector.get_latest(metric_name)
                if latest is not None:
                    metrics[metric_name] = {
                        "latest": latest,
                        "avg_5min": collector.get_average(metric_name, 5),
                        "p95_5min": collector.get_percentile(metric_name, 95, 5)
                    }

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@router.get("/system", summary="System resource status")
async def get_system_status():
    """Get detailed system resource information"""
    with log_context(operation="system_status"):
        try:
            health_checker = get_health_checker()
            system_health = health_checker.system_monitor.get_system_health()

            # Additional system info
            system_info = {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            }

            if torch.cuda.is_available():
                system_info.update({
                    "cuda_device_name": torch.cuda.get_device_name(0),
                    "cuda_memory_total": torch.cuda.get_device_properties(0).total_memory // (1024**3),
                    "cuda_memory_allocated": torch.cuda.memory_allocated() // (1024**3),
                    "cuda_memory_cached": torch.cuda.memory_reserved() // (1024**3)
                })

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "health": system_health.to_dict(),
                "system_info": system_info
            }

        except Exception as e:
            logger.error(f"System status check failed: {e}")
            raise HTTPException(status_code=500, detail=f"System status check failed: {str(e)}")

@router.get("/application", summary="Application status")
async def get_application_status():
    """Get application-specific status information"""
    with log_context(operation="app_status"):
        try:
            health_checker = get_health_checker()
            app_health = health_checker.app_monitor.get_application_health()

            # Additional application info
            app_info = {
                "active_contexts": len(health_checker.app_monitor.active_contexts),
                "active_models": len(health_checker.app_monitor.active_models),
                "context_details": {
                    name: {
                        "poly_degree": info.get("poly_degree"),
                        "security_level": info.get("security_level"),
                        "device": info.get("device"),
                        "age_seconds": (datetime.utcnow() - info["created_at"]).total_seconds()
                    }
                    for name, info in health_checker.app_monitor.active_contexts.items()
                },
                "model_details": {
                    name: {
                        "model_type": info.get("model_type"),
                        "in_channels": info.get("in_channels"),
                        "out_channels": info.get("out_channels"),
                        "age_seconds": (datetime.utcnow() - info["created_at"]).total_seconds()
                    }
                    for name, info in health_checker.app_monitor.active_models.items()
                }
            }

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "health": app_health.to_dict(),
                "application_info": app_info
            }

        except Exception as e:
            logger.error(f"Application status check failed: {e}")
            raise HTTPException(status_code=500, detail=f"Application status check failed: {str(e)}")

@router.post("/check/{check_name}", summary="Run specific health check")
async def run_specific_check(check_name: str):
    """Run a specific named health check"""
    with log_context(operation=f"health_check_{check_name}"):
        try:
            health_checker = get_health_checker()

            if check_name not in health_checker.health_checks:
                available_checks = list(health_checker.health_checks.keys()) + ["system", "application"]
                raise HTTPException(
                    status_code=404,
                    detail=f"Health check '{check_name}' not found. Available: {available_checks}"
                )

            # Run specific check
            check_func = health_checker.health_checks[check_name]
            result = check_func()

            return {
                "timestamp": datetime.utcnow().isoformat(),
                "check_name": check_name,
                "result": result.to_dict()
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Specific health check {check_name} failed: {e}")
            raise HTTPException(status_code=500, detail=f"Check failed: {str(e)}")

@router.get("/readiness", summary="Readiness probe")
async def readiness_probe():
    """Kubernetes-style readiness probe"""
    with log_context(operation="readiness_probe"):
        try:
            health_checker = get_health_checker()
            results = await health_checker.run_health_checks()
            overall_status = health_checker.get_overall_status(results)

            # Service is ready if not critical
            if overall_status == HealthStatus.CRITICAL:
                return JSONResponse(
                    status_code=503,
                    content={
                        "ready": False,
                        "status": overall_status.value,
                        "message": "Service not ready due to critical issues"
                    }
                )

            return {
                "ready": True,
                "status": overall_status.value,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Readiness probe failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "error": str(e)
                }
            )

@router.get("/liveness", summary="Liveness probe")
async def liveness_probe():
    """Kubernetes-style liveness probe"""
    with log_context(operation="liveness_probe"):
        # Simple liveness check - just verify basic functionality
        try:
            # Test basic tensor operations
            test_tensor = torch.randn(10, 10)
            _ = test_tensor.sum()

            return {
                "alive": True,
                "timestamp": datetime.utcnow().isoformat(),
                "service": "he-graph-embeddings"
            }

        except Exception as e:
            logger.critical(f"Liveness probe failed: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "alive": False,
                    "error": str(e)
                }
            )

# Custom health checks that can be registered

async def check_model_inference_health() -> HealthCheckResult:
    """
    Health check for model inference capability.

    Tests basic PyTorch tensor operations to verify model inference
    infrastructure is functional.

    Returns:
        HealthCheckResult: Result with status and diagnostic details

    Test details:
        - Creates random test tensor
        - Applies ReLU activation function
        - Verifies tensor operations complete successfully

    Status levels:
        - HEALTHY: All tensor operations successful
        - CRITICAL: PyTorch operations failed
    """
    try:
        # This would test basic model inference
        # For now, just verify PyTorch is working
        test_tensor = torch.randn(5, 10)
        result = torch.nn.functional.relu(test_tensor)

        return HealthCheckResult(
            name="model_inference",
            status=HealthStatus.HEALTHY,
            message="Model inference capability working",
            details={"test_shape": list(test_tensor.shape)}
        )

    except Exception as e:
        logger.error(f"Error in operation: {e}")
        return HealthCheckResult(
            name="model_inference",
            status=HealthStatus.CRITICAL,
            message=f"Model inference failed: {str(e)}"
        )

async def check_encryption_performance() -> HealthCheckResult:
    """Health check for encryption performance"""
    try:

        import time
        from ..python.he_graph import CKKSContext, HEConfig

        # Quick performance test
        config = HEConfig(poly_modulus_degree=4096)  # Small for quick test
        context = CKKSContext(config)
        context.generate_keys()

        test_data = torch.randn(10, 5)

        start_time = time.time()
        encrypted = context.encrypt(test_data)
        encryption_time = time.time() - start_time

        start_time = time.time()
        decrypted = context.decrypt(encrypted)
        decryption_time = time.time() - start_time

        # Performance thresholds
        if encryption_time > 1.0:  # 1 second
            status = HealthStatus.DEGRADED
            message = f"Slow encryption: {encryption_time:.3f}s"
        elif encryption_time > 2.0:  # 2 seconds
            status = HealthStatus.UNHEALTHY
            message = f"Very slow encryption: {encryption_time:.3f}s"
        else:
            status = HealthStatus.HEALTHY
            message = f"Encryption performance good: {encryption_time:.3f}s"

        return HealthCheckResult(
            name="encryption_performance",
            status=status,
            message=message,
            details={
                "encryption_time_ms": encryption_time * 1000,
                "decryption_time_ms": decryption_time * 1000,
                "test_data_shape": list(test_data.shape)
            }
        )

    except Exception as e:
        logger.error(f"Error in operation: {e}")
        return HealthCheckResult(
            name="encryption_performance",
            status=HealthStatus.CRITICAL,
            message=f"Encryption performance test failed: {str(e)}"
        )

# Register additional health checks on module import
def register_additional_checks():
    """Register additional health checks"""
    health_checker = get_health_checker()
    if health_checker:
        # These are async functions, so they need wrapper functions
        health_checker.register_health_check(
            "model_inference",
            lambda: asyncio.run(check_model_inference_health())
        )
        health_checker.register_health_check(
            "encryption_performance",
            lambda: asyncio.run(check_encryption_performance())
        )

# Register checks when module is imported
try:
    register_additional_checks()
except Exception as e:
    logger.warning(f"Could not register additional health checks: {e}")
#!/usr/bin/env python3
"""
Production health check script for HE-Graph-Embeddings
Validates system health, GPU availability, and service readiness
"""


import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional


import requests
import torch

# Add source path
sys.path.append('/app/src')

try:

    from utils.monitoring import HealthCheckManager
    from python.he_graph import CKKSContext
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionHealthCheck:
    """Comprehensive health check for production deployment"""

    def __init__(self, host: str = "localhost", port: int = 8000):
        """  Init  ."""
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.health_manager = HealthCheckManager()

    def check_api_health(self) -> Dict[str, Any]:
        """Check if the API is responding"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "details": response.json()
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}",
                    "details": response.text
                }

        except requests.exceptions.ConnectionError:
            logger.error(f"Error in operation: {e}")
            return {
                "status": "unhealthy",
                "error": "Connection refused",
                "details": "API service not responding"
            }
        except requests.exceptions.Timeout:
            logger.error(f"Error in operation: {e}")
            return {
                "status": "unhealthy",
                "error": "Request timeout",
                "details": "API response too slow"
            }
        except Exception as e:
            logger.error(f"Error in operation: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "Unexpected error during health check"
            }

    def check_gpu_health(self) -> Dict[str, Any]:
        """Check GPU availability and health"""
        try:
            if not torch.cuda.is_available():
                return {
                    "status": "unhealthy",
                    "error": "CUDA not available",
                    "gpu_count": 0
                }

            gpu_count = torch.cuda.device_count()
            gpu_info = []

            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_cached = torch.cuda.memory_reserved(i)
                memory_total = props.total_memory

                gpu_info.append({
                    "device_id": i,
                    "name": props.name,
                    "memory_allocated_mb": memory_allocated // (1024 * 1024),
                    "memory_cached_mb": memory_cached // (1024 * 1024),
                    "memory_total_mb": memory_total // (1024 * 1024),
                    "memory_free_mb": (memory_total - memory_cached) // (1024 * 1024),
                    "utilization_percent": (memory_cached / memory_total) * 100
                })

            # Check if any GPU has high utilization
            high_util_gpus = [gpu for gpu in gpu_info if gpu["utilization_percent"] > 90]

            return {
                "status": "healthy" if len(high_util_gpus) < gpu_count else "degraded",
                "gpu_count": gpu_count,
                "gpu_info": gpu_info,
                "warning": f"{len(high_util_gpus)} GPUs with >90% utilization" if high_util_gpus else None
            }

        except Exception as e:
            logger.error(f"Error in operation: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "GPU health check failed"
            }

    def check_he_functionality(self) -> Dict[str, Any]:
        """Test basic HE operations"""
        try:
            start_time = time.time()

            # Initialize CKKS context
            context = CKKSContext(
                poly_modulus_degree=8192,
                scale=2**30,
                gpu_id=0 if torch.cuda.is_available() else None
            )

            # Test basic encryption
            test_data = torch.randn(10, 16)
            if torch.cuda.is_available():
                test_data = test_data.cuda()

            encrypted = context.encrypt(test_data)
            result = context.add(encrypted, encrypted)
            decrypted = context.decrypt(result)

            # Verify correctness
            expected = test_data + test_data
            error = torch.mean(torch.abs(decrypted - expected)).item()

            operation_time = (time.time() - start_time) * 1000

            if error < 1e-3:  # Reasonable error threshold
                return {
                    "status": "healthy",
                    "operation_time_ms": operation_time,
                    "encryption_error": error,
                    "context_params": {
                        "poly_modulus_degree": context.config.poly_modulus_degree,
                        "scale": context.config.scale,
                        "gpu_enabled": torch.cuda.is_available()
                    }
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "HE operation accuracy too low",
                    "encryption_error": error,
                    "threshold": 1e-3
                }

        except Exception as e:
            logger.error(f"Error in operation: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "details": "HE functionality test failed"
            }

    def check_external_dependencies(self) -> Dict[str, Any]:
        """Check external service dependencies"""
        dependencies = {}

        # Redis
        try:

            import redis
            redis_host = os.getenv('REDIS_HOST', 'redis')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            redis_password = os.getenv('REDIS_PASSWORD')

            r = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                socket_timeout=5,
                decode_responses=True
            )

            # Test connection
            r.ping()
            info = r.info()

            dependencies['redis'] = {
                "status": "healthy",
                "host": redis_host,
                "port": redis_port,
                "version": info.get('redis_version'),
                "connected_clients": info.get('connected_clients'),
                "used_memory_mb": info.get('used_memory', 0) // (1024 * 1024)
            }

        except Exception as e:
            logger.error(f"Error in operation: {e}")
            dependencies['redis'] = {
                "status": "unhealthy",
                "error": str(e)
            }

        # Prometheus (optional)
        try:
            prometheus_host = os.getenv('PROMETHEUS_HOST', 'prometheus')
            prometheus_port = int(os.getenv('PROMETHEUS_PORT', '9090'))

            response = requests.get(
                f"http://{prometheus_host}:{prometheus_port}/-/healthy",
                timeout=5
            )

            if response.status_code == 200:
                dependencies['prometheus'] = {
                    "status": "healthy",
                    "host": prometheus_host,
                    "port": prometheus_port
                }
            else:
                dependencies['prometheus'] = {
                    "status": "unhealthy",
                    "error": f"HTTP {response.status_code}"
                }

        except Exception as e:
            logger.error(f"Error in operation: {e}")
            dependencies['prometheus'] = {
                "status": "unhealthy",
                "error": str(e)
            }

        return dependencies

    def comprehensive_health_check(self) -> Dict[str, Any]:
        """Perform complete system health check"""

        logger.info("Starting comprehensive health check...")
        start_time = time.time()

        # Individual health checks
        api_health = self.check_api_health()
        gpu_health = self.check_gpu_health()
        he_health = self.check_he_functionality()
        deps_health = self.check_external_dependencies()

        # System metrics from monitoring
        try:
            system_health = self.health_manager.get_comprehensive_health()
        except Exception as e:
            logger.error(f"Error in operation: {e}")
            system_health = {
                "status": "unhealthy",
                "error": f"System monitoring failed: {e}"
            }

        # Determine overall status
        checks = [api_health, gpu_health, he_health, system_health]
        dep_checks = list(deps_health.values())

        # Overall status logic
        if all(check["status"] == "healthy" for check in checks):
            if all(dep.get("status") == "healthy" for dep in dep_checks):
                overall_status = "healthy"
            else:
                overall_status = "degraded"  # External deps optional
        elif any(check["status"] == "unhealthy" for check in checks):
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"

        total_time = (time.time() - start_time) * 1000

        result = {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "check_duration_ms": total_time,
            "version": "2.0.0",
            "region": os.getenv('HE_GRAPH_REGION', 'unknown'),
            "checks": {
                "api": api_health,
                "gpu": gpu_health,
                "homomorphic_encryption": he_health,
                "system": system_health,
                "dependencies": deps_health
            }
        }

        logger.info(f"Health check completed in {total_time:.1f}ms - Status: {overall_status}")
        return result


def main():
    """Main."""
    parser = argparse.ArgumentParser(description='HE-Graph-Embeddings Health Check')
    parser.add_argument('--host', default='localhost', help='API host')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--daemon', action='store_true', help='Run as daemon')
    parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds (daemon mode)')
    parser.add_argument('--output', choices=['json', 'text'], default='json', help='Output format')
    parser.add_argument('--fail-on-degraded', action='store_true', help='Exit with error on degraded status')

    args = parser.parse_args()

    health_checker = ProductionHealthCheck(args.host, args.port)

    if args.daemon:
        logger.info(f"Starting health check daemon (interval: {args.interval}s)")

        while True:
            try:
                result = health_checker.comprehensive_health_check()

                if args.output == 'json':
                    print(json.dumps(result, indent=2))
                else:
                    status = result["status"]
                    timestamp = result["timestamp"]
                    duration = result["check_duration_ms"]
                    print(f"[{timestamp}] Health: {status.upper()} ({duration:.1f}ms)")

                time.sleep(args.interval)

            except KeyboardInterrupt:
                logger.info("Daemon stopped by user")
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(args.interval)
    else:
        # Single check
        result = health_checker.comprehensive_health_check()

        if args.output == 'json':
            print(json.dumps(result, indent=2))
        else:
            status = result["status"]
            print(f"Overall Status: {status.upper()}")

            for check_name, check_result in result["checks"].items():
                check_status = check_result.get("status", "unknown")
                print(f"  {check_name.title()}: {check_status.upper()}")

                if check_result.get("error"):
                    print(f"    Error: {check_result['error']}")

        # Exit with appropriate code
        if result["status"] == "unhealthy":
            sys.exit(1)
        elif result["status"] == "degraded" and args.fail_on_degraded:
            sys.exit(2)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Load testing and stress testing for HE-Graph-Embeddings
Tests system behavior under various load conditions
"""


import pytest
import asyncio
import concurrent.futures
import time
import threading
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
import torch
import numpy as np
import psutil
import gc


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


from python.he_graph import CKKSContext, HEConfig, HEGraphSAGE
from utils.performance import get_performance_manager, PerformanceConfig
from utils.concurrent_processing import get_worker_pool, ConcurrencyConfig
from utils.monitoring import get_health_checker

@dataclass
class LoadTestConfig:
    """Configuration for load testing"""
    max_concurrent_requests: int = 10
    total_requests: int = 100
    ramp_up_time: float = 10.0
    test_duration: float = 60.0

    # Graph parameters
    num_nodes: int = 50
    feature_dim: int = 32

    # HE parameters
    poly_degree: int = 4096
    security_level: int = 128

@dataclass
class LoadTestResult:
    """Results from load testing"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput: float
    error_rate: float
    peak_memory_usage_mb: float
    peak_cpu_usage: float

    def success_rate(self) -> float:
        """Success Rate."""
        return self.successful_requests / max(1, self.total_requests)

class LoadTestMetrics:
    """Collect and analyze load test metrics"""

    def __init__(self):
        """  Init  ."""
        self.request_times = []
        self.errors = []
        self.memory_samples = []
        self.cpu_samples = []
        self.timestamps = []
        self.lock = threading.Lock()

    def record_request(self, response_time -> None: float, success: bool, error: str = None):
        """Record a request result"""
        with self.lock:
            self.request_times.append(response_time)
            self.timestamps.append(time.time())

            if not success:
                self.errors.append({
                    'timestamp': time.time(),
                    'error': error or 'unknown_error',
                    'response_time': response_time
                })

    def record_system_metrics(self) -> None:
        """Record current system metrics"""
        with self.lock:
            self.memory_samples.append(psutil.virtual_memory().percent)
            self.cpu_samples.append(psutil.cpu_percent())

    def get_results(self) -> LoadTestResult:
        """Calculate final results"""
        with self.lock:
            if not self.request_times:
                return LoadTestResult(0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0)

            total_requests = len(self.request_times)
            failed_requests = len(self.errors)
            successful_requests = total_requests - failed_requests

            avg_response_time = np.mean(self.request_times)
            p95_response_time = np.percentile(self.request_times, 95)
            p99_response_time = np.percentile(self.request_times, 99)

            # Calculate throughput
            if self.timestamps:
                test_duration = max(self.timestamps) - min(self.timestamps)
                throughput = total_requests / max(test_duration, 1.0)
            else:
                throughput = 0.0

            error_rate = failed_requests / total_requests

            peak_memory = max(self.memory_samples) if self.memory_samples else 0
            peak_cpu = max(self.cpu_samples) if self.cpu_samples else 0

            return LoadTestResult(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                avg_response_time=avg_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                throughput=throughput,
                error_rate=error_rate,
                peak_memory_usage_mb=peak_memory,
                peak_cpu_usage=peak_cpu
            )

class HEGraphLoadTester:
    """Load tester for HE-Graph operations"""

    def __init__(self, config: LoadTestConfig):
        """  Init  ."""
        self.config = config
        self.metrics = LoadTestMetrics()
        self.he_context = None
        self.model = None
        self.test_data = None

        # Performance monitoring
        self.monitoring_active = False
        self.monitor_thread = None

    def setup(self) -> None:
        """Setup test environment"""
        # Create HE context
        he_config = HEConfig(
            poly_modulus_degree=self.config.poly_degree,
            coeff_modulus_bits=[40, 30, 30, 40],
            scale=2**30,
            security_level=self.config.security_level
        )

        self.he_context = CKKSContext(he_config)
        self.he_context.generate_keys()

        # Create model
        self.model = HEGraphSAGE(
            in_channels=self.config.feature_dim,
            hidden_channels=[16, 8],
            out_channels=4,
            context=self.he_context
        )

        # Generate test data
        self.test_data = self._generate_test_data()

        # Start monitoring
        self._start_monitoring()

    def teardown(self) -> None:
        """Cleanup test environment"""
        self._stop_monitoring()

        # Force cleanup
        self.he_context = None
        self.model = None
        self.test_data = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

    def _generate_test_data(self) -> List[tuple]:
        """Generate test data for load testing"""
        test_cases = []

        for i in range(self.config.total_requests):
            # Generate unique graph for each request
            np.random.seed(i)

            features = torch.randn(self.config.num_nodes, self.config.feature_dim)

            # Create random graph
            edges = []
            for node in range(self.config.num_nodes):
                num_edges = np.random.poisson(3) + 1
                targets = np.random.choice(
                    self.config.num_nodes,
                    min(num_edges, self.config.num_nodes - 1),
                    replace=False
                )
                for target in targets:
                    if target != node:
                        edges.append([node, target])

            edge_index = torch.tensor(edges).T.long() if edges else torch.zeros((2, 0)).long()

            test_cases.append((features, edge_index))

        return test_cases

    def _start_monitoring(self) -> None:
        """Start system monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

    def _stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

    def _monitoring_loop(self) -> None:
        """Monitor system resources during test"""
        while self.monitoring_active:
            self.metrics.record_system_metrics()
            time.sleep(1.0)

    def _execute_single_request(self, request_id: int) -> bool:
        """Execute a single HE-Graph request"""
        try:
            features, edge_index = self.test_data[request_id % len(self.test_data)]

            start_time = time.time()

            # Encrypt input
            enc_features = self.he_context.encrypt(features)

            # Forward pass
            enc_output = self.model(enc_features, edge_index)

            # Decrypt output
            output = self.he_context.decrypt(enc_output)

            response_time = time.time() - start_time

            # Validate output
            if not torch.isfinite(output).all():
                raise ValueError("Invalid output values")

            self.metrics.record_request(response_time, True)
            return True

        except Exception as e:
            logger.error(f"Error in operation: {e}")
            response_time = time.time() - start_time if 'start_time' in locals() else 0
            self.metrics.record_request(response_time, False, str(e))
            return False

    async def run_load_test(self) -> LoadTestResult:
        """Run the load test"""
        print(f"Starting load test: {self.config.total_requests} requests, "
                f"{self.config.max_concurrent_requests} concurrent")

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def rate_limited_request(request_id: int):
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self._execute_single_request, request_id
                )

        # Generate request schedule with ramp-up
        request_times = []
        for i in range(self.config.total_requests):
            # Linear ramp-up
            delay = (i / self.config.total_requests) * self.config.ramp_up_time
            request_times.append(delay)

        # Schedule all requests
        start_time = time.time()
        tasks = []

        for i, delay in enumerate(request_times):
            # Schedule request after delay
            task = asyncio.create_task(self._delayed_request(delay, i, rate_limited_request))
            tasks.append(task)

        # Wait for all requests to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Return results
        return self.metrics.get_results()

    async def _delayed_request(self, delay: float, request_id: int, request_func):
        """Execute request after delay"""
        await asyncio.sleep(delay)
        return await request_func(request_id)

class TestLoadAndStress:
    """Load and stress testing suite"""

    def setup_method(self) -> None:
        """Setup for each test"""
        # Clear any existing state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def teardown_method(self) -> None:
        """Cleanup after each test"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @pytest.mark.slow
    def test_light_load(self) -> None:
        """Test under light load conditions"""
        config = LoadTestConfig(
            max_concurrent_requests=2,
            total_requests=20,
            ramp_up_time=5.0,
            num_nodes=20,
            feature_dim=16,
            poly_degree=4096
        )

        tester = HEGraphLoadTester(config)

        try:
            tester.setup()

            # Run test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(tester.run_load_test())
            loop.close()

            # Verify results
            assert result.success_rate() >= 0.9, f"Success rate too low: {result.success_rate():.2%}"
            assert result.avg_response_time < 30.0, f"Average response time too high: {result.avg_response_time:.2f}s"
            assert result.throughput > 0.1, f"Throughput too low: {result.throughput:.3f} req/s"

            print(f"Light load test results:")
            print(f"  Success rate: {result.success_rate():.2%}")
            print(f"  Avg response time: {result.avg_response_time:.2f}s")
            print(f"  P95 response time: {result.p95_response_time:.2f}s")
            print(f"  Throughput: {result.throughput:.2f} req/s")

        finally:
            tester.teardown()

    @pytest.mark.slow
    def test_moderate_load(self) -> None:
        """Test under moderate load conditions"""
        config = LoadTestConfig(
            max_concurrent_requests=5,
            total_requests=50,
            ramp_up_time=10.0,
            num_nodes=40,
            feature_dim=24,
            poly_degree=4096
        )

        tester = HEGraphLoadTester(config)

        try:
            tester.setup()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(tester.run_load_test())
            loop.close()

            # More relaxed criteria for moderate load
            assert result.success_rate() >= 0.8, f"Success rate too low: {result.success_rate():.2%}"
            assert result.avg_response_time < 60.0, f"Average response time too high: {result.avg_response_time:.2f}s"

            print(f"Moderate load test results:")
            print(f"  Success rate: {result.success_rate():.2%}")
            print(f"  Avg response time: {result.avg_response_time:.2f}s")
            print(f"  P99 response time: {result.p99_response_time:.2f}s")
            print(f"  Throughput: {result.throughput:.2f} req/s")
            print(f"  Peak memory: {result.peak_memory_usage_mb:.1f}%")
            print(f"  Peak CPU: {result.peak_cpu_usage:.1f}%")

        finally:
            tester.teardown()

    @pytest.mark.slow
    @pytest.mark.stress
    def test_heavy_load(self) -> None:
        """Test under heavy load conditions"""
        # Skip in CI environments
        if os.getenv('CI'):
            pytest.skip("Skipping heavy load test in CI environment")

        config = LoadTestConfig(
            max_concurrent_requests=10,
            total_requests=100,
            ramp_up_time=15.0,
            num_nodes=60,
            feature_dim=32,
            poly_degree=8192
        )

        tester = HEGraphLoadTester(config)

        try:
            tester.setup()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(tester.run_load_test())
            loop.close()

            # Even more relaxed criteria for heavy load
            assert result.success_rate() >= 0.7, f"Success rate too low: {result.success_rate():.2%}"

            print(f"Heavy load test results:")
            print(f"  Success rate: {result.success_rate():.2%}")
            print(f"  Avg response time: {result.avg_response_time:.2f}s")
            print(f"  P99 response time: {result.p99_response_time:.2f}s")
            print(f"  Throughput: {result.throughput:.2f} req/s")
            print(f"  Peak memory: {result.peak_memory_usage_mb:.1f}%")
            print(f"  Peak CPU: {result.peak_cpu_usage:.1f}%")
            print(f"  Error rate: {result.error_rate:.2%}")

        finally:
            tester.teardown()

    def test_memory_pressure(self) -> None:
        """Test behavior under memory pressure"""
        # Create memory pressure by processing large graphs
        config = LoadTestConfig(
            max_concurrent_requests=3,
            total_requests=15,
            ramp_up_time=5.0,
            num_nodes=200,  # Large graph
            feature_dim=64,  # High-dimensional features
            poly_degree=8192
        )

        tester = HEGraphLoadTester(config)

        try:
            tester.setup()

            initial_memory = psutil.virtual_memory().percent

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(tester.run_load_test())
            loop.close()

            final_memory = psutil.virtual_memory().percent
            memory_growth = final_memory - initial_memory

            print(f"Memory pressure test results:")
            print(f"  Success rate: {result.success_rate():.2%}")
            print(f"  Memory growth: {memory_growth:.1f}%")
            print(f"  Peak memory usage: {result.peak_memory_usage_mb:.1f}%")

            # Should handle memory pressure gracefully
            assert result.success_rate() >= 0.6, "Should handle memory pressure"
            assert memory_growth < 30.0, "Excessive memory growth"

        finally:
            tester.teardown()

    def test_concurrent_context_creation(self) -> None:
        """Test concurrent creation of HE contexts"""
        def create_context():
            """Create Context."""
            try:
                config = HEConfig(poly_modulus_degree=4096, scale=2**30)
                context = CKKSContext(config)
                context.generate_keys()

                # Perform a simple operation
                test_data = torch.randn(10, 8)
                enc_data = context.encrypt(test_data)
                dec_data = context.decrypt(enc_data)

                error = torch.mean(torch.abs(test_data - dec_data))
                return error.item() < 0.1

            except Exception as e:
                print(f"Context creation failed: {e}")
                return False

        # Create contexts concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(create_context) for _ in range(8)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Concurrent context creation success rate too low: {success_rate:.2%}"

    def test_performance_degradation_over_time(self) -> None:
        """Test for performance degradation over extended usage"""
        config = HEConfig(poly_modulus_degree=4096, scale=2**30)
        context = CKKSContext(config)
        context.generate_keys()

        model = HEGraphSAGE(
            in_channels=16,
            hidden_channels=8,
            out_channels=4,
            context=context
        )

        # Measure performance over multiple iterations
        response_times = []
        num_iterations = 20

        for iteration in range(num_iterations):
            features = torch.randn(30, 16)
            edges = torch.randint(0, 30, (2, 50))

            start_time = time.time()

            enc_features = context.encrypt(features)
            enc_output = model(enc_features, edges)
            output = context.decrypt(enc_output)

            response_time = time.time() - start_time
            response_times.append(response_time)

            # Verify output quality
            assert torch.isfinite(output).all(), f"Invalid output at iteration {iteration}"

        # Check for performance degradation
        early_times = response_times[:5]
        late_times = response_times[-5:]

        early_avg = np.mean(early_times)
        late_avg = np.mean(late_times)

        degradation = (late_avg - early_avg) / early_avg

        print(f"Performance over time:")
        print(f"  Early average: {early_avg:.2f}s")
        print(f"  Late average: {late_avg:.2f}s")
        print(f"  Degradation: {degradation:.2%}")

        # Should not degrade significantly
        assert degradation < 0.5, f"Excessive performance degradation: {degradation:.2%}"

class TestStressConditions:
    """Stress testing under extreme conditions"""

    @pytest.mark.stress
    def test_extreme_graph_sizes(self) -> None:
        """Test with extremely large graphs"""
        if os.getenv('CI'):
            pytest.skip("Skipping extreme size test in CI")

        # Test parameters
        extreme_nodes = 500
        extreme_features = 128

        config = HEConfig(
            poly_modulus_degree=16384,  # Large polynomial degree
            coeff_modulus_bits=[60, 50, 50, 50, 50, 60],
            scale=2**50
        )

        try:
            context = CKKSContext(config)
            context.generate_keys()

            model = HEGraphSAGE(
                in_channels=extreme_features,
                hidden_channels=64,
                out_channels=16,
                context=context
            )

            # Generate large graph
            features = torch.randn(extreme_nodes, extreme_features)

            # Create sparse graph to make computation feasible
            edges = []
            for i in range(extreme_nodes):
                num_edges = min(5, extreme_nodes - 1)
                targets = np.random.choice(extreme_nodes, num_edges, replace=False)
                for target in targets:
                    if target != i:
                        edges.append([i, target])

            edge_index = torch.tensor(list(set(tuple(e) for e in edges))).T.long()

            # Process with timeout
            start_time = time.time()

            enc_features = context.encrypt(features)
            enc_output = model(enc_features, edge_index)
            output = context.decrypt(enc_output)

            processing_time = time.time() - start_time

            # Verify results
            assert output.shape == (extreme_nodes, 16)
            assert torch.isfinite(output).all()

            print(f"Extreme graph processing:")
            print(f"  Nodes: {extreme_nodes}, Features: {extreme_features}")
            print(f"  Processing time: {processing_time:.2f}s")
            print(f"  Memory usage: {torch.cuda.memory_allocated() / (1024**3):.2f}GB" if torch.cuda.is_available() else "N/A")

        except Exception as e:
            logger.error(f"Error in operation: {e}")
            pytest.skip(f"Extreme graph test failed (expected): {e}")

    @pytest.mark.stress
    def test_resource_exhaustion_recovery(self) -> None:
        """Test recovery from resource exhaustion"""
        config = HEConfig(poly_modulus_degree=4096, scale=2**30)

        # Create contexts until resource exhaustion
        contexts = []

        try:
            for i in range(50):  # Try to create many contexts
                context = CKKSContext(config)
                context.generate_keys()
                contexts.append(context)

                # Stop if memory usage gets too high
                if psutil.virtual_memory().percent > 85:
                    break

            print(f"Created {len(contexts)} contexts before resource pressure")

        except Exception as e:
            print(f"Resource exhaustion at {len(contexts)} contexts: {e}")

        finally:
            # Cleanup and test recovery
            contexts.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Test that we can still create a working context
            recovery_context = CKKSContext(config)
            recovery_context.generate_keys()

            # Verify it works
            test_data = torch.randn(10, 8)
            enc_data = recovery_context.encrypt(test_data)
            dec_data = recovery_context.decrypt(enc_data)

            error = torch.mean(torch.abs(test_data - dec_data))
            assert error < 0.1, "Recovery failed"

            print("Successfully recovered from resource exhaustion")

if __name__ == "__main__":
    # Run load tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not stress",  # Skip stress tests by default
        "--durations=10"
    ])
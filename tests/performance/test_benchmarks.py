"""
Performance benchmarks and stress tests for HE-Graph-Embeddings
"""


import pytest
import torch
import numpy as np
import time
import psutil
import gc
from typing import List, Dict, Tuple
from unittest.mock import patch
import threading
from concurrent.futures import ThreadPoolExecutor


from src.python.he_graph import (
    CKKSContext, HEConfig, HEGraphSAGE, HEGAT,
    EncryptedTensor, SecurityEstimator
)

from src.utils.performance import (
    ResourcePool, MemoryMonitor, BatchProcessor,
    ParallelExecutor, optimize_memory, adaptive_batch_size
)

from src.utils.caching import MemoryAwareCache, TensorCache, EncryptionCache


@pytest.mark.benchmark
class TestEncryptionPerformance:
    """Benchmark encryption operations"""

    @pytest.fixture(scope="class")
    def context(self) -> None:
        """Create optimized context for benchmarks"""
        config = HEConfig(
            poly_modulus_degree=16384,  # Balanced performance/security
            coeff_modulus_bits=[60, 40, 40, 60],
            scale=2**35
        )
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx

    def test_encryption_throughput(self, context, benchmark) -> None:
        """Benchmark encryption throughput"""
        data_sizes = [
            (10, 10),      # Small
            (100, 50),     # Medium
            (1000, 128),   # Large
        ]

        for rows, cols in data_sizes:
            data = torch.randn(rows, cols) * 0.1

            def encrypt_data():
                """Encrypt Data."""
                return context.encrypt(data)

            result = benchmark.pedantic(encrypt_data, rounds=5, iterations=3)

            # Calculate throughput
            data_size_mb = (rows * cols * 4) / (1024 * 1024)  # Float32 = 4 bytes
            throughput = data_size_mb / (benchmark.stats.mean)

            print(f"Encryption throughput for {rows}x{cols}: {throughput:.2f} MB/s")
            assert result is not None

    def test_homomorphic_operations_performance(self, context, benchmark) -> None:
        """Benchmark homomorphic operations"""
        # Test data
        a = torch.randn(100, 64) * 0.01
        b = torch.randn(100, 64) * 0.01

        enc_a = context.encrypt(a)
        enc_b = context.encrypt(b)

        # Benchmark addition
        def add_operation():
            """Add Operation."""
            return context.add(enc_a, enc_b)

        add_result = benchmark.pedantic(add_operation, rounds=10, iterations=1)
        assert add_result is not None

        # Benchmark multiplication (more expensive)
        def mul_operation():
            """Mul Operation."""
            return context.multiply(enc_a, enc_b)

        mul_result = benchmark.pedantic(mul_operation, rounds=5, iterations=1)
        assert mul_result is not None

    def test_batch_encryption_scaling(self, context) -> None:
        """Test batch encryption scaling"""
        batch_sizes = [1, 5, 10, 25, 50]
        times = []

        for batch_size in batch_sizes:
            data_batch = [torch.randn(50, 32) * 0.1 for _ in range(batch_size)]

            start_time = time.perf_counter()
            encrypted_batch = [context.encrypt(data) for data in data_batch]
            end_time = time.perf_counter()

            total_time = end_time - start_time
            times.append(total_time)

            # Verify results
            assert len(encrypted_batch) == batch_size
            for enc in encrypted_batch:
                assert isinstance(enc, EncryptedTensor)

            throughput = batch_size / total_time
            print(f"Batch size {batch_size}: {total_time:.3f}s ({throughput:.2f} encryptions/s)")

        # Check scaling efficiency (should be roughly linear)
        scaling_efficiency = times[-1] / times[0] / (batch_sizes[-1] / batch_sizes[0])
        assert 0.5 <= scaling_efficiency <= 2.0, f"Poor scaling efficiency: {scaling_efficiency}"

    def test_noise_budget_degradation(self, context) -> None:
        """Test noise budget degradation under operations"""
        data = torch.randn(10, 10) * 0.001  # Very small values
        encrypted = context.encrypt(data)

        noise_history = [encrypted.noise_budget]
        current = encrypted

        # Perform chain of operations
        for i in range(10):
            # Addition (low noise cost)
            current = context.add(current, encrypted)
            noise_history.append(current.noise_budget)

            # Multiplication (high noise cost)
            if i < 3:  # Limit multiplications to avoid exhaustion
                current = context.multiply(current, encrypted)
                noise_history.append(current.noise_budget)

        # Noise should generally decrease
        assert noise_history[-1] < noise_history[0]

        # Log noise degradation
        print(f"Noise degradation: {noise_history[0]:.2f} -> {noise_history[-1]:.2f}")
        for i, noise in enumerate(noise_history):
            print(f"  Step {i}: {noise:.2f}")


@pytest.mark.benchmark
class TestModelPerformance:
    """Benchmark model inference performance"""

    @pytest.fixture(scope="class")
    def context(self) -> None:
        """Create context for model benchmarks"""
        config = HEConfig(poly_modulus_degree=8192)  # Smaller for faster tests
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx

    @pytest.fixture(scope="class")
    def sample_graphs(self) -> None:
        """Generate sample graphs of different sizes"""
        graphs = {}

        # Small graph
        graphs['small'] = {
            'features': torch.randn(10, 32) * 0.01,
            'edges': torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9]])
        }

        # Medium graph
        graphs['medium'] = {
            'features': torch.randn(50, 64) * 0.01,
            'edges': torch.randint(0, 50, (2, 200))
        }

        # Large graph
        graphs['large'] = {
            'features': torch.randn(200, 128) * 0.005,
            'edges': torch.randint(0, 200, (2, 1000))
        }

        return graphs

    def test_graphsage_performance(self, context, sample_graphs, benchmark) -> None:
        """Benchmark GraphSAGE performance"""
        model = HEGraphSAGE(
            in_channels=32,
            hidden_channels=[16, 8],
            out_channels=4,
            context=context
        )

        # Test on small graph
        graph = sample_graphs['small']
        encrypted_features = context.encrypt(graph['features'])

        def graphsage_forward():
            """Graphsage Forward."""
            return model.forward(encrypted_features, graph['edges'])

        result = benchmark.pedantic(graphsage_forward, rounds=3, iterations=1)

        # Verify output
        output = context.decrypt(result)
        assert output.shape == (10, 4)

    def test_gat_performance(self, context, sample_graphs, benchmark) -> None:
        """Benchmark GAT performance"""
        model = HEGAT(
            in_channels=32,
            out_channels=4,
            heads=2,
            attention_type='additive',
            context=context
        )

        graph = sample_graphs['small']
        encrypted_features = context.encrypt(graph['features'])

        def gat_forward():
            """Gat Forward."""
            return model.forward(encrypted_features, graph['edges'])

        result = benchmark.pedantic(gat_forward, rounds=3, iterations=1)

        # Verify output
        output = context.decrypt(result)
        assert output.shape == (10, 4)

    def test_model_scaling_comparison(self, context, sample_graphs) -> None:
        """Compare model performance across different graph sizes"""
        models = {
            'GraphSAGE': HEGraphSAGE(in_channels=32, hidden_channels=16, out_channels=4, context=context),
            'GAT': HEGAT(in_channels=32, out_channels=4, heads=1, context=context)
        }

        results = {}

        for model_name, model in models.items():
            results[model_name] = {}

            for graph_size, graph in sample_graphs.items():
                if graph_size == 'large':
                    continue  # Skip large graphs for this test

                # Adjust input channels based on graph
                if graph_size == 'medium':
                    # Create compatible model
                    if model_name == 'GraphSAGE':
                        model = HEGraphSAGE(in_channels=64, hidden_channels=16, out_channels=4, context=context)
                    else:
                        model = HEGAT(in_channels=64, out_channels=4, heads=1, context=context)

                encrypted_features = context.encrypt(graph['features'])

                # Time inference
                start_time = time.perf_counter()
                encrypted_output = model.forward(encrypted_features, graph['edges'])
                end_time = time.perf_counter()

                inference_time = end_time - start_time
                results[model_name][graph_size] = inference_time

                # Verify output
                output = context.decrypt(encrypted_output)
                assert output.shape[1] == 4  # 4 output channels

                print(f"{model_name} on {graph_size}: {inference_time:.3f}s")

        # Compare relative performance
        for graph_size in ['small', 'medium']:
            if graph_size in results['GraphSAGE'] and graph_size in results['GAT']:
                sage_time = results['GraphSAGE'][graph_size]
                gat_time = results['GAT'][graph_size]
                ratio = gat_time / sage_time
                print(f"{graph_size} - GAT/GraphSAGE ratio: {ratio:.2f}")


@pytest.mark.benchmark
class TestMemoryPerformance:
    """Test memory usage and management"""

    def test_memory_usage_tracking(self) -> None:
        """Test memory usage during operations"""
        monitor = MemoryMonitor()
        initial_memory = monitor.get_memory_usage_percent()

        # Create large tensors
        large_tensors = []
        for i in range(10):
            tensor = torch.randn(1000, 1000)
            large_tensors.append(tensor)

        peak_memory = monitor.get_memory_usage_percent()
        memory_increase = peak_memory - initial_memory

        print(f"Memory usage increased by {memory_increase:.2f}%")

        # Cleanup
        del large_tensors
        gc.collect()

        final_memory = monitor.get_memory_usage_percent()
        memory_recovered = peak_memory - final_memory

        print(f"Memory recovered: {memory_recovered:.2f}%")

        # Should recover most memory
        assert memory_recovered > memory_increase * 0.5

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.get_device_properties')
    def test_gpu_memory_monitoring(self, mock_props, mock_reserved, mock_allocated, mock_cuda) -> None:
        """Test GPU memory monitoring"""
        # Mock GPU properties
        mock_props.return_value.total_memory = 16 * 1024**3  # 16GB
        mock_allocated.return_value = 4 * 1024**3  # 4GB allocated
        mock_reserved.return_value = 6 * 1024**3  # 6GB reserved

        monitor = MemoryMonitor()
        gpu_info = monitor.get_gpu_memory_info()

        assert gpu_info['total_gb'] == 16.0
        assert gpu_info['allocated_gb'] == 4.0
        assert gpu_info['reserved_gb'] == 6.0
        assert gpu_info['free_gb'] == 10.0  # 16 - 6

    def test_memory_cleanup_effectiveness(self) -> None:
        """Test memory cleanup mechanisms"""
        monitor = MemoryMonitor()
        initial_memory = psutil.Process().memory_info().rss

        # Create memory pressure
        large_data = []
        for _ in range(100):
            data = np.random.randn(10000, 100)
            large_data.append(data)

        peak_memory = psutil.Process().memory_info().rss

        # Trigger cleanup
        monitor.cleanup_memory()

        # Force garbage collection
        del large_data
        gc.collect()

        final_memory = psutil.Process().memory_info().rss

        # Calculate memory recovery
        memory_growth = peak_memory - initial_memory
        memory_recovered = peak_memory - final_memory
        recovery_rate = memory_recovered / memory_growth

        print(f"Memory growth: {memory_growth / 1024**2:.1f} MB")
        print(f"Memory recovered: {memory_recovered / 1024**2:.1f} MB")
        print(f"Recovery rate: {recovery_rate:.1%}")

        # Should recover significant memory
        assert recovery_rate > 0.5


@pytest.mark.benchmark
class TestCachePerformance:
    """Test caching system performance"""

    def test_cache_hit_rate_performance(self, benchmark) -> None:
        """Test cache performance under different hit rates"""
        cache = MemoryAwareCache(max_size_bytes=10*1024*1024)  # 10MB

        # Pre-populate cache
        for i in range(100):
            data = torch.randn(100, 100) * 0.1
            cache.put(f"key_{i}", data)

        # Test different access patterns
        def high_hit_rate_access():
            """High Hit Rate Access."""
            # 90% hit rate
            hits = 0
            for _ in range(100):
                key_id = np.random.randint(0, 10)  # Access first 10 keys frequently
                result = cache.get(f"key_{key_id}")
                if result is not None:
                    hits += 1
            return hits

        def low_hit_rate_access():
            """Low Hit Rate Access."""
            # 10% hit rate
            hits = 0
            for _ in range(100):
                key_id = np.random.randint(0, 1000)  # Most keys don't exist
                result = cache.get(f"key_{key_id}")
                if result is not None:
                    hits += 1
            return hits

        # Benchmark high hit rate
        high_hits = benchmark.pedantic(high_hit_rate_access, rounds=5, iterations=1)
        assert high_hits > 80  # Should have >80% hit rate

        # Benchmark low hit rate
        low_hits = benchmark.pedantic(low_hit_rate_access, rounds=5, iterations=1)
        assert low_hits < 20  # Should have <20% hit rate

    def test_tensor_cache_performance(self) -> None:
        """Test tensor cache performance"""
        tensor_cache = TensorCache(max_gpu_memory_gb=1.0, max_cpu_memory_gb=2.0)

        # Test caching different tensor sizes
        tensor_sizes = [
            (10, 10),
            (100, 100),
            (500, 500),
            (1000, 1000)
        ]

        for size in tensor_sizes:
            tensor = torch.randn(*size)

            # Measure put performance
            start_time = time.perf_counter()
            success = tensor_cache.put_tensor(f"tensor_{size}", tensor)
            put_time = time.perf_counter() - start_time

            # Measure get performance
            start_time = time.perf_counter()
            retrieved = tensor_cache.get_tensor(f"tensor_{size}")
            get_time = time.perf_counter() - start_time

            print(f"Size {size}: put={put_time*1000:.2f}ms, get={get_time*1000:.2f}ms")

            assert success or size == (1000, 1000)  # Large tensor might not fit
            if success:
                assert retrieved is not None
                torch.testing.assert_close(tensor, retrieved)

    def test_cache_memory_pressure_handling(self) -> None:
        """Test cache behavior under memory pressure"""
        cache = MemoryAwareCache(max_size_bytes=1024*1024)  # 1MB limit

        # Fill cache to capacity
        data_size = 1024 * 100  # 100KB each
        num_items = 15  # Should exceed capacity

        eviction_count = 0
        for i in range(num_items):
            data = np.random.randn(data_size // 8)  # 8 bytes per float64
            success = cache.put(f"item_{i}", data)

            if not success:
                eviction_count += 1

        # Check final state
        final_size = cache.stats.total_size_bytes
        assert final_size <= cache.max_size_bytes

        # Should have evicted some items
        print(f"Cache size: {final_size / 1024:.1f} KB")
        print(f"Items evicted: {cache.stats.evictions}")

        assert cache.stats.evictions > 0


@pytest.mark.benchmark
class TestConcurrencyPerformance:
    """Test concurrent operations performance"""

    @pytest.fixture(scope="class")
    def context(self) -> None:
        """Create context for concurrency tests"""
        config = HEConfig(poly_modulus_degree=4096)  # Smaller for faster operations
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx

    def test_concurrent_encryption(self, context) -> None:
        """Test concurrent encryption performance"""
        def encrypt_task(task_id):
            """Encrypt Task."""
            data = torch.randn(50, 20) * 0.01
            start_time = time.perf_counter()
            encrypted = context.encrypt(data)
            end_time = time.perf_counter()
            return task_id, end_time - start_time, encrypted

        # Test different numbers of concurrent tasks
        thread_counts = [1, 2, 4, 8]

        for num_threads in thread_counts:
            start_time = time.perf_counter()

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(encrypt_task, i) for i in range(num_threads * 2)]
                results = [f.result() for f in futures]

            total_time = time.perf_counter() - start_time

            # Verify all tasks completed
            assert len(results) == num_threads * 2
            for task_id, duration, encrypted in results:
                assert isinstance(encrypted, EncryptedTensor)
                assert duration > 0

            throughput = len(results) / total_time
            print(f"{num_threads} threads: {total_time:.3f}s ({throughput:.2f} ops/s)")

    def test_parallel_model_inference(self, context) -> None:
        """Test parallel model inference"""
        model = HEGraphSAGE(
            in_channels=16,
            hidden_channels=8,
            out_channels=4,
            context=context
        )

        def inference_task(task_id):
            """Inference Task."""
            features = torch.randn(20, 16) * 0.01
            edges = torch.randint(0, 20, (2, 30))

            encrypted_features = context.encrypt(features)

            start_time = time.perf_counter()
            encrypted_output = model.forward(encrypted_features, edges)
            end_time = time.perf_counter()

            output = context.decrypt(encrypted_output)
            return task_id, end_time - start_time, output.shape

        # Test with multiple workers
        num_tasks = 6
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(inference_task, i) for i in range(num_tasks)]
            results = [f.result() for f in futures]

        total_time = time.perf_counter() - start_time

        # Verify results
        assert len(results) == num_tasks
        for task_id, duration, shape in results:
            assert shape == (20, 4)  # Expected output shape
            assert duration > 0

        throughput = num_tasks / total_time
        print(f"Parallel inference: {total_time:.3f}s ({throughput:.2f} inferences/s)")

    def test_resource_pool_performance(self) -> None:
        """Test resource pool performance"""

        from src.utils.performance import ResourceLimits, ResourcePool

        limits = ResourceLimits(max_cpu_threads=4, max_concurrent_operations=8)
        pool = ResourcePool(limits)

        async def async_task(task_id):
            await asyncio.sleep(0.1)  # Simulate work
            return task_id * 2

        # Test resource pool efficiency

        import asyncio

        async def test_pool():
            tasks = []
            start_time = time.perf_counter()

            for i in range(10):
                task = pool.submit_cpu_task(lambda: time.sleep(0.1) or i)
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time

            return results, total_time

        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            results, total_time = loop.run_until_complete(test_pool())

            # Should complete faster than sequential (10 * 0.1 = 1.0s)
            assert total_time < 0.8
            assert len(results) == 10

            print(f"Resource pool: {total_time:.3f}s for 10 tasks")
        finally:
            pool.shutdown()
            loop.close()


@pytest.mark.benchmark
class TestScalabilityLimits:
    """Test system scalability limits"""

    @pytest.fixture(scope="class")
    def context(self) -> None:
        """Create context for scalability tests"""
        config = HEConfig(
            poly_modulus_degree=8192,
            coeff_modulus_bits=[60, 40, 60],
            scale=2**30
        )
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx

    def test_maximum_graph_size(self, context) -> None:
        """Test maximum processable graph size"""
        max_nodes = 1000

        # Binary search for maximum processable size
        low, high = 100, max_nodes
        largest_successful = 0

        while low <= high:
            mid = (low + high) // 2

            try:
                # Create graph of size 'mid'
                features = torch.randn(mid, 64) * 0.001  # Very small values
                edges = torch.randint(0, mid, (2, min(mid * 2, 5000)))

                # Test encryption
                start_time = time.perf_counter()
                encrypted_features = context.encrypt(features)
                encrypt_time = time.perf_counter() - start_time

                # Test simple model inference
                model = HEGraphSAGE(
                    in_channels=64,
                    hidden_channels=32,
                    out_channels=16,
                    context=context
                )

                start_time = time.perf_counter()
                encrypted_output = model.forward(encrypted_features, edges)
                inference_time = time.perf_counter() - start_time

                # Verify output
                output = context.decrypt(encrypted_output)
                assert output.shape == (mid, 16)

                print(f"✓ Size {mid}: encrypt={encrypt_time:.3f}s, inference={inference_time:.3f}s")

                largest_successful = mid
                low = mid + 1

            except (MemoryError, RuntimeError, torch.cuda.OutOfMemoryError) as e:
                print(f"✗ Size {mid}: {type(e).__name__}")
                high = mid - 1

            # Cleanup
            gc.collect()

        print(f"Maximum processable graph size: {largest_successful} nodes")
        assert largest_successful >= 100  # Should handle at least 100 nodes

    def test_encryption_depth_limits(self, context) -> None:
        """Test maximum homomorphic operation depth"""
        data = torch.randn(10, 10) * 0.0001  # Very small values
        encrypted = context.encrypt(data)

        depth = 0
        current = encrypted

        try:
            while current.noise_budget > 1.0:  # Continue while noise budget exists
                # Perform multiplication (expensive operation)
                current = context.multiply(current, encrypted)
                depth += 1

                noise_remaining = current.noise_budget
                print(f"Depth {depth}: noise budget = {noise_remaining:.2f}")

                if depth > 20:  # Safety limit
                    break

        except Exception as e:
            print(f"Failed at depth {depth}: {e}")

        print(f"Maximum operation depth: {depth}")
        assert depth >= 1  # Should support at least one multiplication

    @pytest.mark.slow
    def test_long_running_stability(self, context) -> None:
        """Test stability under long-running operations"""
        start_time = time.perf_counter()
        operations_completed = 0

        try:
            # Run for up to 30 seconds
            while time.perf_counter() - start_time < 30:
                # Create random data
                size = np.random.randint(10, 50)
                features = torch.randn(size, 32) * 0.01

                # Encrypt
                encrypted = context.encrypt(features)

                # Random operations
                operation = np.random.choice(['add', 'multiply'])

                if operation == 'add':
                    result = context.add(encrypted, encrypted)
                else:
                    result = context.multiply(encrypted, encrypted)

                # Verify operation completed
                assert isinstance(result, EncryptedTensor)
                operations_completed += 1

                # Periodic cleanup
                if operations_completed % 10 == 0:
                    gc.collect()
                    print(f"Completed {operations_completed} operations...")

        except Exception as e:
            print(f"Failed after {operations_completed} operations: {e}")

        total_time = time.perf_counter() - start_time
        ops_per_second = operations_completed / total_time

        print(f"Stability test: {operations_completed} ops in {total_time:.1f}s ({ops_per_second:.2f} ops/s)")
        assert operations_completed > 10  # Should complete at least 10 operations

    def test_memory_leak_detection(self) -> None:
        """Test for memory leaks during repeated operations"""

        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Create and destroy contexts repeatedly
        for i in range(20):
            config = HEConfig(poly_modulus_degree=4096)

            with patch('torch.cuda.is_available', return_value=False):
                context = CKKSContext(config)
                context.generate_keys()

                # Perform operations
                data = torch.randn(20, 20) * 0.01
                encrypted = context.encrypt(data)
                result = context.add(encrypted, encrypted)
                decrypted = context.decrypt(result)

                # Verify result
                assert decrypted.shape == (20, 20)

            # Force cleanup
            del context, encrypted, result, decrypted
            gc.collect()

            # Check memory periodically
            if i % 5 == 0:
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory
                print(f"Iteration {i}: memory growth = {memory_growth / 1024**2:.1f} MB")

        # Final memory check
        final_memory = process.memory_info().rss
        total_growth = final_memory - initial_memory
        growth_mb = total_growth / (1024**2)

        print(f"Total memory growth: {growth_mb:.1f} MB")

        # Should not have significant memory growth (< 50MB)
        assert growth_mb < 50, f"Potential memory leak: {growth_mb:.1f} MB growth"


@pytest.mark.benchmark
class TestRealWorldScenarios:
    """Test performance under realistic usage scenarios"""

    @pytest.fixture(scope="class")
    def production_context(self) -> None:
        """Create production-like context"""
        config = HEConfig(
            poly_modulus_degree=32768,  # Production size
            coeff_modulus_bits=[60, 40, 40, 40, 40, 60],
            scale=2**40,
            security_level=128
        )
        with patch('torch.cuda.is_available', return_value=False):
            ctx = CKKSContext(config)
            ctx.generate_keys()
            return ctx

    def test_financial_fraud_detection_scenario(self, production_context) -> None:
        """Simulate financial fraud detection workload"""
        # Simulate transaction graph (accounts and transactions)
        num_accounts = 100
        num_transactions = 500

        # Account features (balance, age, transaction_count, etc.)
        account_features = torch.randn(num_accounts, 16) * 0.01

        # Transaction edges (random connections)
        transaction_edges = torch.randint(0, num_accounts, (2, num_transactions))

        # Create fraud detection model
        model = HEGraphSAGE(
            in_channels=16,
            hidden_channels=[32, 16, 8],
            out_channels=2,  # fraud/legitimate
            aggregator='mean',
            context=production_context
        )

        # Benchmark complete workflow
        start_time = time.perf_counter()

        # 1. Encrypt account features
        encrypted_features = production_context.encrypt(account_features)
        encrypt_time = time.perf_counter()

        # 2. Run fraud detection
        risk_scores = model.forward(encrypted_features, transaction_edges)
        inference_time = time.perf_counter()

        # 3. Decrypt results (in practice, only high-risk accounts)
        decrypted_scores = production_context.decrypt(risk_scores)
        decrypt_time = time.perf_counter()

        total_time = decrypt_time - start_time

        # Verify output
        assert decrypted_scores.shape == (num_accounts, 2)

        print(f"Fraud Detection Scenario:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Encryption: {(encrypt_time - start_time):.3f}s")
        print(f"  Inference: {(inference_time - encrypt_time):.3f}s")
        print(f"  Decryption: {(decrypt_time - inference_time):.3f}s")
        print(f"  Throughput: {num_accounts / total_time:.2f} accounts/s")

    def test_healthcare_drug_interaction_scenario(self, production_context) -> None:
        """Simulate healthcare drug interaction prediction"""
        # Drug molecule features
        num_drugs = 50
        drug_features = torch.randn(num_drugs, 256) * 0.005  # Large feature vectors

        # Drug interaction edges
        num_interactions = 200
        interaction_edges = torch.randint(0, num_drugs, (2, num_interactions))

        # Drug interaction prediction model (GAT for attention)
        model = HEGAT(
            in_channels=256,
            out_channels=3,  # no_interaction, mild_interaction, severe_interaction
            heads=4,
            attention_type='additive',
            context=production_context
        )

        start_time = time.perf_counter()

        # Encrypt drug features
        encrypted_features = production_context.encrypt(drug_features)
        encrypt_time = time.perf_counter()

        # Predict interactions
        interaction_predictions = model.forward(encrypted_features, interaction_edges)
        inference_time = time.perf_counter()

        total_time = inference_time - start_time

        # Verify predictions
        decrypted_predictions = production_context.decrypt(interaction_predictions)
        assert decrypted_predictions.shape == (num_drugs, 3)

        print(f"Healthcare Drug Interaction Scenario:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Encryption: {(encrypt_time - start_time):.3f}s")
        print(f"  Inference: {(inference_time - encrypt_time):.3f}s")
        print(f"  Throughput: {num_drugs / total_time:.2f} drugs/s")

    @pytest.mark.slow
    def test_high_frequency_inference_scenario(self, production_context) -> None:
        """Test high-frequency inference scenario"""
        # Create lightweight model for frequent inference
        model = HEGraphSAGE(
            in_channels=32,
            hidden_channels=16,
            out_channels=4,
            context=production_context
        )

        # Pre-encrypt common data
        base_features = torch.randn(20, 32) * 0.01
        base_edges = torch.randint(0, 20, (2, 40))
        encrypted_base = production_context.encrypt(base_features)

        # Simulate high-frequency inference requests
        num_requests = 100
        latencies = []

        start_time = time.perf_counter()

        for i in range(num_requests):
            request_start = time.perf_counter()

            # Small variations in input
            noise = torch.randn_like(base_features) * 0.001
            varied_features = production_context.encrypt(base_features + noise)

            # Run inference
            output = model.forward(varied_features, base_edges)

            request_end = time.perf_counter()
            latency = request_end - request_start
            latencies.append(latency)

            if (i + 1) % 20 == 0:
                avg_latency = np.mean(latencies[-20:])
                print(f"Requests {i-19}-{i}: avg latency = {avg_latency*1000:.1f}ms")

        total_time = time.perf_counter() - start_time

        # Calculate statistics
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        throughput = num_requests / total_time

        print(f"High-Frequency Inference Results:")
        print(f"  Total requests: {num_requests}")
        print(f"  Average latency: {avg_latency*1000:.1f}ms")
        print(f"  P95 latency: {p95_latency*1000:.1f}ms")
        print(f"  Throughput: {throughput:.2f} requests/s")

        # Performance assertions
        assert avg_latency < 1.0  # Less than 1 second average
        assert throughput > 5     # More than 5 requests per second
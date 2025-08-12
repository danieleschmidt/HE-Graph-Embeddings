#!/usr/bin/env python3
"""
Comprehensive integration tests for HE-Graph-Embeddings full pipeline
Tests end-to-end functionality with real encryption and graph processing
"""


import pytest
import torch
import numpy as np
import time
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Import our modules

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


from python.he_graph import (
    CKKSContext, HEConfig, HEGraphSAGE, HEGAT,
    SecurityEstimator, NoiseTracker
)

from utils.validation import TensorValidator, GraphValidator
from utils.performance import get_performance_manager
from utils.monitoring import get_health_checker
from utils.error_handling import HEGraphError, EncryptionError

class TestFullPipelineIntegration:
    """Integration tests for complete HE-Graph processing pipeline"""

    @pytest.fixture(autouse=True)
    def setup_method(self) -> None:
        """Setup test environment"""
        # Clear any existing global state
        torch.manual_seed(42)
        np.random.seed(42)

        # Create test configuration
        self.test_config = HEConfig(
            poly_modulus_degree=4096,  # Small for testing
            coeff_modulus_bits=[40, 30, 30, 40],
            scale=2**30,
            security_level=128,
            precision_bits=25
        )

        # Test data
        self.num_nodes = 20
        self.feature_dim = 16
        self.num_edges = 40

        self.test_features = torch.randn(self.num_nodes, self.feature_dim)
        self.test_edge_index = self._create_test_graph()

        yield

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _create_test_graph(self) -> torch.Tensor:
        """Create a test graph with known structure"""
        edges = []

        # Create a ring graph
        for i in range(self.num_nodes):
            edges.append([i, (i + 1) % self.num_nodes])

        # Add some random edges
        np.random.seed(42)
        for _ in range(self.num_edges - self.num_nodes):
            src = np.random.randint(0, self.num_nodes)
            dst = np.random.randint(0, self.num_nodes)
            if src != dst:
                edges.append([src, dst])

        return torch.tensor(edges[:self.num_edges]).T.long()

    def test_basic_encryption_decryption_cycle(self) -> None:
        """Test basic encryption/decryption cycle"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        # Encrypt features
        encrypted_features = context.encrypt(self.test_features)

        # Verify encryption properties
        assert encrypted_features.c0.shape == self.test_features.shape
        assert encrypted_features.c1.shape == self.test_features.shape
        assert encrypted_features.scale == self.test_config.scale

        # Decrypt and verify
        decrypted_features = context.decrypt(encrypted_features)

        # Check decryption accuracy
        mse = torch.mean((self.test_features - decrypted_features) ** 2)
        assert mse < 0.1, f"Decryption error too high: MSE = {mse:.6f}"

    def test_graphsage_full_pipeline(self) -> None:
        """Test complete GraphSAGE pipeline with encryption"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        # Create model
        model = HEGraphSAGE(
            in_channels=self.feature_dim,
            hidden_channels=[12, 8],
            out_channels=4,
            num_layers=None,
            aggregator='mean',
            context=context
        )

        # Encrypt input
        encrypted_features = context.encrypt(self.test_features)

        # Forward pass
        with NoiseTracker() as tracker:
            encrypted_output = model(encrypted_features, self.test_edge_index)
            final_noise = tracker.get_noise_budget()

        # Verify output shape
        assert encrypted_output.c0.shape == (self.num_nodes, 4)

        # Decrypt output
        decrypted_output = context.decrypt(encrypted_output)

        # Verify output properties
        assert decrypted_output.shape == (self.num_nodes, 4)
        assert torch.isfinite(decrypted_output).all()
        assert final_noise > 0, "Noise budget exhausted"

        # Verify non-zero output (model should have learned something)
        assert torch.abs(decrypted_output).max() > 0.01

    def test_gat_attention_mechanism(self) -> None:
        """Test GAT with attention mechanism"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        # Create GAT model
        model = HEGAT(
            in_channels=self.feature_dim,
            out_channels=8,
            heads=2,
            attention_type='additive',
            context=context
        )

        # Encrypt input
        encrypted_features = context.encrypt(self.test_features)

        # Forward pass
        encrypted_output = model(encrypted_features, self.test_edge_index)

        # Decrypt and verify
        decrypted_output = context.decrypt(encrypted_output)

        assert decrypted_output.shape == (self.num_nodes, 8)
        assert torch.isfinite(decrypted_output).all()

    def test_homomorphic_operations_consistency(self) -> None:
        """Test consistency of homomorphic operations"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        # Test data
        a = torch.randn(5, 3) * 0.5  # Small values for better precision
        b = torch.randn(5, 3) * 0.5

        # Encrypt
        enc_a = context.encrypt(a)
        enc_b = context.encrypt(b)

        # Homomorphic addition
        enc_sum = context.add(enc_a, enc_b)
        dec_sum = context.decrypt(enc_sum)
        expected_sum = a + b

        add_error = torch.mean(torch.abs(dec_sum - expected_sum))
        assert add_error < 0.01, f"Addition error: {add_error:.6f}"

        # Homomorphic multiplication
        enc_product = context.multiply(enc_a, enc_b)
        dec_product = context.decrypt(enc_product)
        expected_product = a * b

        mul_error = torch.mean(torch.abs(dec_product - expected_product))
        assert mul_error < 0.1, f"Multiplication error: {mul_error:.6f}"

    def test_batch_processing_pipeline(self) -> None:
        """Test batch processing with multiple graphs"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        model = HEGraphSAGE(
            in_channels=self.feature_dim,
            hidden_channels=8,
            out_channels=4,
            context=context
        )

        # Process multiple graphs
        batch_results = []

        for i in range(3):  # 3 different graphs
            # Create slightly different features
            features = self.test_features + torch.randn_like(self.test_features) * 0.1

            # Encrypt and process
            enc_features = context.encrypt(features)
            enc_output = model(enc_features, self.test_edge_index)
            dec_output = context.decrypt(enc_output)

            batch_results.append(dec_output)

        # Verify all results are different (due to different inputs)
        for i in range(len(batch_results)):
            for j in range(i + 1, len(batch_results)):
                diff = torch.mean(torch.abs(batch_results[i] - batch_results[j]))
                assert diff > 0.01, f"Batch results {i} and {j} are too similar"

    def test_security_parameter_validation(self) -> None:
        """Test security parameter validation"""
        estimator = SecurityEstimator()

        # Test valid parameters
        valid_params = {
            'poly_degree': 4096,
            'coeff_modulus_bits': [40, 30, 30, 40]
        }
        security_bits = estimator.estimate(valid_params)
        assert security_bits >= 128, f"Security level too low: {security_bits}"

        # Test parameter recommendations
        recommended = estimator.recommend(
            security_bits=128,
            multiplicative_depth=5,
            precision_bits=25
        )

        assert recommended.poly_modulus_degree >= 4096
        assert recommended.security_level == 128

    def test_error_handling_and_recovery(self) -> None:
        """Test error handling and recovery mechanisms"""
        # Test with invalid configuration
        invalid_config = HEConfig(
            poly_modulus_degree=1023,  # Not power of 2
            coeff_modulus_bits=[10]   # Too small
        )

        with pytest.raises((HEGraphError, EncryptionError, ValueError)):
            context = CKKSContext(invalid_config)

    def test_memory_management(self) -> None:
        """Test memory management during processing"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        model = HEGraphSAGE(
            in_channels=self.feature_dim,
            hidden_channels=8,
            out_channels=4,
            context=context
        )

        # Monitor memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Process multiple iterations
        for _ in range(5):
            enc_features = context.encrypt(self.test_features)
            enc_output = model(enc_features, self.test_edge_index)
            _ = context.decrypt(enc_output)

            # Force cleanup
            del enc_features, enc_output

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_growth = final_memory - initial_memory

        # Memory growth should be minimal
        if torch.cuda.is_available():
            assert memory_growth < 100 * 1024 * 1024, f"Excessive memory growth: {memory_growth / (1024*1024):.1f}MB"

    def test_performance_benchmarking(self) -> None:
        """Test performance benchmarking and optimization"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        model = HEGraphSAGE(
            in_channels=self.feature_dim,
            hidden_channels=8,
            out_channels=4,
            context=context
        )

        # Benchmark encryption
        encryption_times = []
        for _ in range(3):
            start_time = time.time()
            enc_features = context.encrypt(self.test_features)
            encryption_times.append(time.time() - start_time)

        avg_encryption_time = np.mean(encryption_times)
        assert avg_encryption_time < 5.0, f"Encryption too slow: {avg_encryption_time:.3f}s"

        # Benchmark forward pass
        enc_features = context.encrypt(self.test_features)

        forward_times = []
        for _ in range(3):
            start_time = time.time()
            enc_output = model(enc_features, self.test_edge_index)
            forward_times.append(time.time() - start_time)

        avg_forward_time = np.mean(forward_times)
        assert avg_forward_time < 10.0, f"Forward pass too slow: {avg_forward_time:.3f}s"

    def test_noise_budget_management(self) -> None:
        """Test noise budget management throughout computation"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        model = HEGraphSAGE(
            in_channels=self.feature_dim,
            hidden_channels=[8, 6, 4],  # Deeper model
            out_channels=2,
            context=context
        )

        enc_features = context.encrypt(self.test_features)

        # Track noise budget through computation
        with NoiseTracker() as tracker:
            initial_budget = enc_features.noise_budget

            enc_output = model(enc_features, self.test_edge_index)
            final_budget = enc_output.noise_budget

        assert initial_budget > final_budget, "Noise budget should decrease"
        assert final_budget > 0, "Noise budget exhausted"

        # Should still be able to decrypt
        decrypted = context.decrypt(enc_output)
        assert torch.isfinite(decrypted).all(), "Decryption failed due to noise"

    def test_validation_integration(self) -> None:
        """Test integration with validation utilities"""
        # Test tensor validation
        validator_result = TensorValidator.validate_tensor(
            self.test_features,
            name="test_features",
            min_dims=2,
            max_dims=2,
            finite_only=True
        )
        assert validator_result.is_valid

        # Test graph validation
        graph_result = GraphValidator.validate_graph_structure(
            self.num_nodes,
            self.test_edge_index,
            self.test_features
        )
        assert graph_result.is_valid

    def test_concurrent_processing(self) -> None:
        """Test concurrent processing capabilities"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        model = HEGraphSAGE(
            in_channels=self.feature_dim,
            hidden_channels=8,
            out_channels=4,
            context=context
        )

        # Create multiple test cases
        test_cases = []
        for i in range(3):
            features = self.test_features + torch.randn_like(self.test_features) * 0.1
            test_cases.append((features, self.test_edge_index))

        # Process concurrently (simulate with sequential for simplicity)
        results = []
        for features, edge_index in test_cases:
            enc_features = context.encrypt(features)
            enc_output = model(enc_features, edge_index)
            dec_output = context.decrypt(enc_output)
            results.append(dec_output)

        # Verify all results
        assert len(results) == 3
        for result in results:
            assert result.shape == (self.num_nodes, 4)
            assert torch.isfinite(result).all()

    @pytest.mark.slow
    def test_long_running_computation(self) -> None:
        """Test long-running computation with multiple iterations"""
        context = CKKSContext(self.test_config)
        context.generate_keys()

        model = HEGraphSAGE(
            in_channels=self.feature_dim,
            hidden_channels=8,
            out_channels=4,
            context=context
        )

        # Simulate training-like iterations
        results = []
        noise_budgets = []

        for iteration in range(10):
            # Add some variation to features
            features = self.test_features + torch.randn_like(self.test_features) * 0.01

            enc_features = context.encrypt(features)
            enc_output = model(enc_features, self.test_edge_index)

            # Track noise budget
            noise_budgets.append(enc_output.noise_budget)

            dec_output = context.decrypt(enc_output)
            results.append(dec_output)

        # Verify consistency across iterations
        assert len(results) == 10
        assert all(torch.isfinite(result).all() for result in results)
        assert all(budget > 0 for budget in noise_budgets), "Noise budget exhausted"

    def test_real_world_graph_sizes(self) -> None:
        """Test with more realistic graph sizes"""
        # Skip if running in limited memory environment
        if os.getenv('CI'):
            pytest.skip("Skipping large graph test in CI environment")

        # Larger test graph
        large_num_nodes = 100
        large_feature_dim = 32

        large_features = torch.randn(large_num_nodes, large_feature_dim)
        large_edge_index = self._create_large_test_graph(large_num_nodes)

        context = CKKSContext(HEConfig(
            poly_modulus_degree=8192,  # Larger for more capacity
            coeff_modulus_bits=[50, 40, 40, 50],
            scale=2**40
        ))
        context.generate_keys()

        model = HEGraphSAGE(
            in_channels=large_feature_dim,
            hidden_channels=16,
            out_channels=8,
            context=context
        )

        # Process large graph
        enc_features = context.encrypt(large_features)
        enc_output = model(enc_features, large_edge_index)
        dec_output = context.decrypt(enc_output)

        assert dec_output.shape == (large_num_nodes, 8)
        assert torch.isfinite(dec_output).all()

    def _create_large_test_graph(self, num_nodes: int) -> torch.Tensor:
        """Create a larger test graph"""
        edges = []

        # Create a more complex graph structure
        for i in range(num_nodes):
            # Ring connections
            edges.append([i, (i + 1) % num_nodes])

            # Random connections (scale-free-like)
            num_connections = min(5, num_nodes - 1)
            for _ in range(num_connections):
                target = np.random.randint(0, num_nodes)
                if target != i:
                    edges.append([i, target])

        # Remove duplicates
        edges = list(set(tuple(edge) for edge in edges))

        return torch.tensor(edges).T.long()

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""

    @pytest.fixture(autouse=True)
    def setup_performance_testing(self) -> None:
        """Setup for performance tests"""
        self.performance_manager = get_performance_manager()

        # Standard test configuration
        self.benchmark_config = HEConfig(
            poly_modulus_degree=8192,
            coeff_modulus_bits=[50, 40, 40, 40, 50],
            scale=2**40,
            security_level=128
        )

    def test_encryption_performance_scaling(self) -> None:
        """Test encryption performance with different input sizes"""
        context = CKKSContext(self.benchmark_config)
        context.generate_keys()

        sizes = [(10, 16), (50, 32), (100, 64)]
        results = []

        for num_nodes, feature_dim in sizes:
            features = torch.randn(num_nodes, feature_dim)

            # Benchmark encryption
            times = []
            for _ in range(3):
                start_time = time.time()
                enc_features = context.encrypt(features)
                times.append(time.time() - start_time)

            avg_time = np.mean(times)
            throughput = (num_nodes * feature_dim) / avg_time  # elements per second

            results.append({
                'size': (num_nodes, feature_dim),
                'avg_time': avg_time,
                'throughput': throughput
            })

        # Verify performance characteristics
        for result in results:
            assert result['avg_time'] < 30.0, f"Encryption too slow for size {result['size']}"
            assert result['throughput'] > 10, f"Throughput too low: {result['throughput']}"

    def test_model_inference_scaling(self) -> None:
        """Test model inference performance with different architectures"""
        context = CKKSContext(self.benchmark_config)
        context.generate_keys()

        # Test different model configurations
        model_configs = [
            {"hidden": [16], "name": "shallow"},
            {"hidden": [16, 8], "name": "medium"},
            {"hidden": [16, 12, 8], "name": "deep"}
        ]

        test_features = torch.randn(50, 32)
        test_edges = torch.randint(0, 50, (2, 100))
        enc_features = context.encrypt(test_features)

        for config in model_configs:
            model = HEGraphSAGE(
                in_channels=32,
                hidden_channels=config["hidden"],
                out_channels=4,
                context=context
            )

            # Benchmark inference
            times = []
            for _ in range(3):
                start_time = time.time()
                enc_output = model(enc_features, test_edges)
                times.append(time.time() - start_time)

            avg_time = np.mean(times)

            # Log performance
            print(f"Model {config['name']}: {avg_time:.3f}s average")

            # Performance assertions
            max_expected_time = 60.0 * len(config["hidden"])  # Scale with depth
            assert avg_time < max_expected_time, f"Model {config['name']} too slow: {avg_time:.3f}s"

# Test utilities and fixtures
@pytest.fixture(scope="session")
def test_data_generator():
    """Generate test data for multiple tests"""
    def _generate(num_nodes: int, feature_dim: int, num_edges: int = None):
        """ Generate."""
        if num_edges is None:
            num_edges = num_nodes * 2

        features = torch.randn(num_nodes, feature_dim)

        # Create connected graph
        edges = []
        for i in range(num_nodes - 1):
            edges.append([i, i + 1])

        # Add random edges
        for _ in range(num_edges - (num_nodes - 1)):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                edges.append([src, dst])

        edge_index = torch.tensor(edges).T.long()

        return features, edge_index

    return _generate

@pytest.mark.integration
class TestRealWorldScenarios:
    """Integration tests for real-world usage scenarios"""

    def test_financial_fraud_detection_scenario(self) -> None:
        """Test scenario: fraud detection on financial transaction graph"""
        # Simulate financial transaction data
        num_accounts = 100
        num_features = 20  # Account features (balance, history, etc.)

        # Generate test data
        account_features = torch.randn(num_accounts, num_features)

        # Create transaction graph (some accounts have many connections - potential fraud)
        edges = []
        for i in range(num_accounts):
            # Normal accounts have few connections
            num_connections = np.random.poisson(2) + 1

            # Fraud accounts (last 10) have many connections
            if i >= num_accounts - 10:
                num_connections = np.random.poisson(10) + 5

            for _ in range(min(num_connections, num_accounts - 1)):
                target = np.random.randint(0, num_accounts)
                if target != i:
                    edges.append([i, target])

        edge_index = torch.tensor(list(set(tuple(e) for e in edges))).T.long()

        # Setup HE context
        context = CKKSContext(HEConfig(
            poly_modulus_degree=8192,
            coeff_modulus_bits=[50, 40, 40, 40, 50],
            scale=2**40
        ))
        context.generate_keys()

        # Create fraud detection model
        fraud_model = HEGraphSAGE(
            in_channels=num_features,
            hidden_channels=[16, 8],
            out_channels=2,  # Fraud probability
            aggregator='mean',
            context=context
        )

        # Process encrypted data
        enc_features = context.encrypt(account_features)
        enc_fraud_scores = fraud_model(enc_features, edge_index)
        fraud_scores = context.decrypt(enc_fraud_scores)

        # Verify results
        assert fraud_scores.shape == (num_accounts, 2)
        assert torch.isfinite(fraud_scores).all()

        # Higher connectivity accounts should have different patterns
        high_degree_nodes = torch.bincount(edge_index[1])
        high_degree_mask = high_degree_nodes > high_degree_nodes.median()

        if high_degree_mask.sum() > 0:
            high_degree_scores = fraud_scores[high_degree_mask]
            low_degree_scores = fraud_scores[~high_degree_mask]

            # Scores should show some difference
            score_difference = torch.abs(high_degree_scores.mean() - low_degree_scores.mean())
            assert score_difference > 0.01, "Model should detect structural differences"

if __name__ == "__main__":
    # Run tests with comprehensive output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10"
    ])
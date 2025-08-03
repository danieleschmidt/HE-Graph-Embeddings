"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API integration test"
    )

@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture(scope="session")
def sample_graph_data():
    """Sample graph data for testing"""
    return {
        "node_features": [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0], 
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ],
        "edge_index": [
            [0, 1, 2, 1, 3],
            [1, 2, 3, 0, 2]
        ],
        "edge_attributes": [
            [0.5],
            [0.7],
            [0.3],
            [0.9],
            [0.1]
        ],
        "node_labels": [0, 1, 0, 1]
    }

@pytest.fixture(scope="session")
def large_graph_data():
    """Large graph data for performance testing"""
    num_nodes = 1000
    feature_dim = 64
    num_edges = 5000
    
    # Random node features
    node_features = np.random.randn(num_nodes, feature_dim).tolist()
    
    # Random edges
    source_nodes = np.random.randint(0, num_nodes, num_edges).tolist()
    target_nodes = np.random.randint(0, num_nodes, num_edges).tolist()
    edge_index = [source_nodes, target_nodes]
    
    # Random edge attributes
    edge_attributes = np.random.randn(num_edges, 1).tolist()
    
    return {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_attributes": edge_attributes
    }

@pytest.fixture
def mock_cuda_available():
    """Mock CUDA availability"""
    with patch('torch.cuda.is_available', return_value=False):
        yield

@pytest.fixture
def mock_cuda_enabled():
    """Mock CUDA as available and working"""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.set_device'), \
         patch('torch.cuda.get_device_properties') as mock_props:
        
        # Mock GPU properties
        mock_props.return_value.total_memory = 16 * 1024**3  # 16GB
        mock_props.return_value.name = "Tesla V100"
        
        with patch('torch.cuda.current_device', return_value=0), \
             patch('torch.cuda.device_count', return_value=1):
            yield

@pytest.fixture
def mock_he_config():
    """Mock HE configuration for testing"""
    return {
        "poly_modulus_degree": 4096,  # Smaller for faster tests
        "coeff_modulus_bits": [60, 40, 60],
        "scale": 2**30,
        "security_level": 128,
        "precision_bits": 20,
        "gpu_memory_pool_gb": 2,
        "enable_ntt_cache": True,
        "batch_size": 32,
        "bootstrap_threshold": 5,
        "auto_mod_switch": True
    }

@pytest.fixture
def small_he_config():
    """Very small HE configuration for unit tests"""
    return {
        "poly_modulus_degree": 1024,  # Very small for fast tests
        "coeff_modulus_bits": [40, 30, 40],
        "scale": 2**20,
        "security_level": 128,
        "precision_bits": 15
    }

@pytest.fixture
def mock_ckks_context(mock_cuda_available, mock_he_config):
    """Mock CKKS context for testing"""
    from src.python.he_graph import CKKSContext, HEConfig
    
    config = HEConfig(**mock_he_config)
    context = CKKSContext(config)
    context.generate_keys()
    return context

@pytest.fixture
def sample_encrypted_data(mock_ckks_context):
    """Sample encrypted data for testing"""
    context = mock_ckks_context
    
    # Create sample plaintext data
    data = torch.randn(5, 3) * 0.1  # Small values to avoid overflow
    
    # Encrypt
    encrypted = context.encrypt(data)
    
    return {
        "plaintext": data,
        "encrypted": encrypted,
        "context": context
    }

@pytest.fixture
def sample_model_configs():
    """Sample model configurations for testing"""
    return {
        "graphsage": {
            "model_type": "graphsage",
            "in_channels": 8,
            "out_channels": 4,
            "hidden_channels": [6, 5],
            "num_layers": 2,
            "aggregator": "mean",
            "learning_rate": 0.01,
            "epochs": 10
        },
        "gat": {
            "model_type": "gat",
            "in_channels": 8,
            "out_channels": 4,
            "heads": 2,
            "attention_type": "additive",
            "learning_rate": 0.005,
            "epochs": 10
        },
        "simple_graphsage": {
            "model_type": "graphsage",
            "in_channels": 3,
            "out_channels": 2,
            "hidden_channels": 4,
            "num_layers": 1,
            "aggregator": "mean"
        }
    }

@pytest.fixture
def api_auth_headers():
    """Authorization headers for API testing"""
    return {"X-API-Key": "dev-key-12345"}

@pytest.fixture
def mock_api_client():
    """Mock API client for testing"""
    from fastapi.testclient import TestClient
    from src.api.routes import app
    
    with patch('torch.cuda.is_available', return_value=False):
        client = TestClient(app)
        yield client

class MockTensor:
    """Mock tensor for testing without actual PyTorch operations"""
    
    def __init__(self, shape, dtype=torch.float32):
        self.shape = shape
        self.dtype = dtype
        self.data = torch.zeros(shape, dtype=dtype)
    
    def __add__(self, other):
        return MockTensor(self.shape, self.dtype)
    
    def __mul__(self, other):
        return MockTensor(self.shape, self.dtype)
    
    def to(self, device):
        return self
    
    def cpu(self):
        return self
    
    def cuda(self):
        return self

@pytest.fixture
def mock_tensor_factory():
    """Factory for creating mock tensors"""
    return MockTensor

# Test data generators
def generate_random_graph(num_nodes=10, num_edges=20, feature_dim=5):
    """Generate random graph for testing"""
    node_features = torch.randn(num_nodes, feature_dim)
    
    # Generate random edges
    sources = torch.randint(0, num_nodes, (num_edges,))
    targets = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([sources, targets])
    
    return node_features, edge_index

def generate_chain_graph(num_nodes=5, feature_dim=3):
    """Generate chain graph for testing"""
    node_features = torch.randn(num_nodes, feature_dim)
    
    # Create chain: 0-1-2-3-4
    sources = torch.arange(num_nodes - 1)
    targets = torch.arange(1, num_nodes)
    edge_index = torch.stack([sources, targets])
    
    return node_features, edge_index

def generate_complete_graph(num_nodes=4, feature_dim=2):
    """Generate complete graph for testing"""
    node_features = torch.randn(num_nodes, feature_dim)
    
    # Create all possible edges
    sources = []
    targets = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                sources.append(i)
                targets.append(j)
    
    edge_index = torch.tensor([sources, targets])
    
    return node_features, edge_index

@pytest.fixture
def graph_generators():
    """Graph generation utilities"""
    return {
        "random": generate_random_graph,
        "chain": generate_chain_graph,
        "complete": generate_complete_graph
    }

# Performance testing utilities
@pytest.fixture
def performance_monitor():
    """Performance monitoring utility"""
    import time
    import psutil
    import threading
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_usage = []
            self.monitoring = False
            
        def start(self):
            self.start_time = time.time()
            self.monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_memory)
            self._monitor_thread.start()
            
        def stop(self):
            self.end_time = time.time()
            self.monitoring = False
            if hasattr(self, '_monitor_thread'):
                self._monitor_thread.join()
            
        def _monitor_memory(self):
            process = psutil.Process()
            while self.monitoring:
                self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
                time.sleep(0.1)
        
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        @property
        def peak_memory_mb(self):
            return max(self.memory_usage) if self.memory_usage else 0
    
    return PerformanceMonitor

# Error injection utilities
@pytest.fixture
def error_injector():
    """Utility for injecting errors during testing"""
    class ErrorInjector:
        def __init__(self):
            self.patches = []
        
        def inject_cuda_error(self):
            """Inject CUDA unavailable error"""
            patcher = patch('torch.cuda.is_available', return_value=False)
            self.patches.append(patcher)
            return patcher.start()
        
        def inject_memory_error(self):
            """Inject memory error"""
            def memory_error(*args, **kwargs):
                raise MemoryError("Simulated memory error")
            
            patcher = patch('torch.randn', side_effect=memory_error)
            self.patches.append(patcher)
            return patcher.start()
        
        def inject_value_error(self, target, message="Simulated value error"):
            """Inject value error"""
            def value_error(*args, **kwargs):
                raise ValueError(message)
            
            patcher = patch(target, side_effect=value_error)
            self.patches.append(patcher)
            return patcher.start()
        
        def cleanup(self):
            """Cleanup all patches"""
            for patcher in self.patches:
                patcher.stop()
            self.patches.clear()
    
    injector = ErrorInjector()
    yield injector
    injector.cleanup()

# Database testing utilities (for future database tests)
@pytest.fixture
def mock_database():
    """Mock database for testing"""
    class MockDatabase:
        def __init__(self):
            self.data = {}
            self.connected = False
        
        def connect(self):
            self.connected = True
        
        def disconnect(self):
            self.connected = False
        
        def store(self, key, value):
            if not self.connected:
                raise RuntimeError("Not connected to database")
            self.data[key] = value
        
        def retrieve(self, key):
            if not self.connected:
                raise RuntimeError("Not connected to database")
            return self.data.get(key)
        
        def clear(self):
            self.data.clear()
    
    db = MockDatabase()
    db.connect()
    yield db
    db.disconnect()

# Cleanup utilities
@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically cleanup GPU memory after each test"""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible tests"""
    torch.manual_seed(42)
    np.random.seed(42)
    yield

# Test configuration validation
def pytest_runtest_setup(item):
    """Setup function run before each test"""
    # Skip GPU tests if CUDA is not available
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Skip slow tests unless specifically requested
    if item.get_closest_marker("slow") and not item.config.getoption("--runslow"):
        pytest.skip("Slow test skipped (use --runslow to run)")

def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--runintegration", action="store_true", default=False,
        help="Run integration tests"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection based on options"""
    if not config.getoption("--runintegration"):
        skip_integration = pytest.mark.skip(reason="Integration tests skipped (use --runintegration)")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip_integration)
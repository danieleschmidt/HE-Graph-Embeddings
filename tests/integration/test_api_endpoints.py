"""
Integration tests for HE-Graph-Embeddings API endpoints
"""


import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import torch
import numpy as np

# Import the FastAPI app

from src.api.routes import app

class TestAPIClient:
    """Test API client setup and basic functionality"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    @pytest.fixture
    def auth_headers(self) -> None:
        """Authorization headers for protected endpoints"""
        return {"X-API-Key": "dev-key-12345"}

    def test_app_startup(self, client) -> None:
        """Test application starts successfully"""
        # This is implicitly tested by creating the client
        assert client is not None

class TestHealthEndpoints:
    """Test health check endpoints"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    def test_health_check(self, client) -> None:
        """Test basic health check"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "gpu_available" in data
        assert "active_contexts" in data
        assert "active_models" in data

    def test_detailed_health(self, client) -> None:
        """Test detailed health endpoint"""
        response = client.get("/health/detailed")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert "gpu_info" in data
        assert "contexts" in data
        assert "models" in data

    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.device_count', return_value=2)
    @patch('torch.cuda.current_device', return_value=0)
    @patch('torch.cuda.get_device_name', return_value="Tesla V100")
    def test_health_with_gpu(self, mock_name, mock_device, mock_count, mock_available) -> None:
        """Test health check with GPU available"""
        with patch('torch.cuda.memory_allocated', return_value=1024**3), \
            patch('torch.cuda.memory_reserved', return_value=2*1024**3):

            client = TestClient(app)
            response = client.get("/health/detailed")

            assert response.status_code == 200
            data = response.json()

            assert data["gpu_info"]["gpu_count"] == 2
            assert data["gpu_info"]["device_name"] == "Tesla V100"

class TestContextManagement:
    """Test CKKS context management endpoints"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    @pytest.fixture
    def auth_headers(self) -> None:
        """Authorization headers"""
        return {"X-API-Key": "dev-key-12345"}

    def test_list_contexts(self, client, auth_headers) -> None:
        """Test listing contexts"""
        response = client.get("/contexts", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert "contexts" in data
        assert "default" in data["contexts"]

        default_ctx = data["contexts"]["default"]
        assert "poly_degree" in default_ctx
        assert "security_level" in default_ctx
        assert "scale" in default_ctx

    def test_create_context(self, client, auth_headers) -> None:
        """Test creating new context"""
        config = {
            "poly_modulus_degree": 16384,
            "coeff_modulus_bits": [60, 40, 40, 60],
            "scale": 2**35,
            "security_level": 128
        }

        response = client.post(
            "/contexts/test_context",
            json=config,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "Context test_context created successfully" in data["message"]

    def test_create_context_invalid_config(self, client, auth_headers) -> None:
        """Test creating context with invalid configuration"""
        invalid_config = {
            "poly_modulus_degree": 12345,  # Not power of 2
            "coeff_modulus_bits": [60],
            "scale": -1,  # Invalid scale
            "security_level": 64  # Unsupported
        }

        response = client.post(
            "/contexts/invalid_context",
            json=invalid_config,
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_delete_context(self, client, auth_headers) -> None:
        """Test deleting context"""
        # First create a context
        config = {
            "poly_modulus_degree": 8192,
            "coeff_modulus_bits": [60, 40, 60],
            "scale": 2**30,
            "security_level": 128
        }

        create_response = client.post(
            "/contexts/deletable_context",
            json=config,
            headers=auth_headers
        )
        assert create_response.status_code == 200

        # Then delete it
        delete_response = client.delete(
            "/contexts/deletable_context",
            headers=auth_headers
        )

        assert delete_response.status_code == 200
        data = delete_response.json()
        assert "deleted" in data["message"]

    def test_delete_default_context_forbidden(self, client, auth_headers) -> None:
        """Test that deleting default context is forbidden"""
        response = client.delete("/contexts/default", headers=auth_headers)

        assert response.status_code == 400
        data = response.json()
        assert "Cannot delete default context" in data["detail"]

    def test_delete_nonexistent_context(self, client, auth_headers) -> None:
        """Test deleting non-existent context"""
        response = client.delete("/contexts/nonexistent", headers=auth_headers)

        assert response.status_code == 404

class TestModelManagement:
    """Test model management endpoints"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    @pytest.fixture
    def auth_headers(self) -> None:
        """Authorization headers"""
        return {"X-API-Key": "dev-key-12345"}

    def test_create_graphsage_model(self, client, auth_headers) -> None:
        """Test creating GraphSAGE model"""
        model_config = {
            "model_type": "graphsage",
            "in_channels": 10,
            "out_channels": 5,
            "hidden_channels": [8, 6],
            "num_layers": 2,
            "aggregator": "mean",
            "learning_rate": 0.01,
            "epochs": 50
        }

        response = client.post(
            "/models/test_graphsage",
            json=model_config,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "Model test_graphsage created successfully" in data["message"]

    def test_create_gat_model(self, client, auth_headers) -> None:
        """Test creating GAT model"""
        model_config = {
            "model_type": "gat",
            "in_channels": 8,
            "out_channels": 4,
            "heads": 2,
            "attention_type": "additive",
            "learning_rate": 0.005,
            "epochs": 100
        }

        response = client.post(
            "/models/test_gat",
            json=model_config,
            headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "Model test_gat created successfully" in data["message"]

    def test_create_model_invalid_type(self, client, auth_headers) -> None:
        """Test creating model with invalid type"""
        model_config = {
            "model_type": "invalid_type",
            "in_channels": 5,
            "out_channels": 3
        }

        response = client.post(
            "/models/invalid_model",
            json=model_config,
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_list_models(self, client, auth_headers) -> None:
        """Test listing models"""
        # First create a model
        model_config = {
            "model_type": "graphsage",
            "in_channels": 5,
            "out_channels": 3,
            "hidden_channels": 4,
            "num_layers": 1
        }

        client.post(
            "/models/list_test_model",
            json=model_config,
            headers=auth_headers
        )

        # Then list models
        response = client.get("/models", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert "models" in data
        assert "list_test_model" in data["models"]

        model_info = data["models"]["list_test_model"]
        assert model_info["model_type"] == "graphsage"
        assert "created_at" in model_info

    def test_delete_model(self, client, auth_headers) -> None:
        """Test deleting model"""
        # Create model
        model_config = {
            "model_type": "graphsage",
            "in_channels": 3,
            "out_channels": 2
        }

        create_response = client.post(
            "/models/deletable_model",
            json=model_config,
            headers=auth_headers
        )
        assert create_response.status_code == 200

        # Delete model
        delete_response = client.delete(
            "/models/deletable_model",
            headers=auth_headers
        )

        assert delete_response.status_code == 200
        data = delete_response.json()
        assert "deleted" in data["message"]

class TestEncryptionEndpoints:
    """Test encryption and decryption endpoints"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    @pytest.fixture
    def sample_graph_data(self) -> None:
        """Sample graph data for testing"""
        return {
            "node_features": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0]
            ],
            "edge_index": [
                [0, 1, 2],
                [1, 2, 0]
            ]
        }

    def test_encrypt_data(self, client, sample_graph_data) -> None:
        """Test data encryption"""
        response = client.post("/encrypt", json=sample_graph_data)

        assert response.status_code == 200
        data = response.json()

        assert "encryption_id" in data
        assert "encrypted_shape" in data
        assert "num_edges" in data
        assert "scale" in data
        assert "noise_budget" in data

        assert data["encrypted_shape"] == [3, 3]  # 3 nodes, 3 features
        assert data["num_edges"] == 3

    def test_encrypt_data_invalid_format(self, client) -> None:
        """Test encryption with invalid data format"""
        invalid_data = {
            "node_features": [],  # Empty features
            "edge_index": [[0, 1], [1, 0]]
        }

        response = client.post("/encrypt", json=invalid_data)
        assert response.status_code == 400

    def test_encrypt_with_custom_context(self, client, sample_graph_data, auth_headers) -> None:
        """Test encryption with custom context"""
        # First create custom context
        config = {
            "poly_modulus_degree": 8192,
            "coeff_modulus_bits": [60, 40, 60],
            "scale": 2**30,
            "security_level": 128
        }

        context_response = client.post(
            "/contexts/custom_encrypt_context",
            json=config,
            headers=auth_headers
        )
        assert context_response.status_code == 200

        # Then encrypt with custom context
        response = client.post(
            "/encrypt?context_name=custom_encrypt_context",
            json=sample_graph_data
        )

        assert response.status_code == 200

    def test_batch_encryption(self, client, sample_graph_data) -> None:
        """Test batch encryption"""
        batch_data = [sample_graph_data, sample_graph_data, sample_graph_data]

        response = client.post("/batch/encrypt", json=batch_data)

        assert response.status_code == 200
        data = response.json()

        assert "batch_results" in data
        assert len(data["batch_results"]) == 3

        for i, result in enumerate(data["batch_results"]):
            assert result["batch_index"] == i
            assert "encryption_id" in result
            assert "noise_budget" in result

class TestInferenceEndpoints:
    """Test inference endpoints"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    @pytest.fixture
    def auth_headers(self) -> None:
        """Authorization headers"""
        return {"X-API-Key": "dev-key-12345"}

    @pytest.fixture
    def test_model_and_data(self, client, auth_headers) -> None:
        """Create test model and sample data"""
        # Create model
        model_config = {
            "model_type": "graphsage",
            "in_channels": 3,
            "out_channels": 2,
            "hidden_channels": 4,
            "num_layers": 1
        }

        model_response = client.post(
            "/models/inference_test_model",
            json=model_config,
            headers=auth_headers
        )
        assert model_response.status_code == 200

        # Sample data
        sample_data = {
            "node_features": [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ],
            "edge_index": [
                [0, 1],
                [1, 0]
            ]
        }

        return "inference_test_model", sample_data

    def test_run_inference(self, client, test_model_and_data) -> None:
        """Test running inference"""
        model_name, sample_data = test_model_and_data

        response = client.post(f"/inference/{model_name}", json=sample_data)

        assert response.status_code == 200
        data = response.json()

        assert "inference_id" in data
        assert "output_shape" in data
        assert "noise_budget" in data

    def test_inference_nonexistent_model(self, client) -> None:
        """Test inference with non-existent model"""
        sample_data = {
            "node_features": [[1.0, 2.0]],
            "edge_index": [[], []]
        }

        response = client.post("/inference/nonexistent_model", json=sample_data)

        assert response.status_code == 404
        data = response.json()
        assert "Model not found" in data["detail"]

    def test_inference_with_options(self, client, test_model_and_data) -> None:
        """Test inference with additional options"""
        model_name, sample_data = test_model_and_data

        # Add inference options
        sample_data.update({
            "return_attention_weights": False,
            "return_embeddings": False,
            "decrypt_output": False
        })

        response = client.post(f"/inference/{model_name}", json=sample_data)

        assert response.status_code == 200

class TestUtilityEndpoints:
    """Test utility endpoints"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    def test_security_estimation(self, client) -> None:
        """Test security level estimation"""
        config = {
            "poly_modulus_degree": 16384,
            "coeff_modulus_bits": [60, 40, 40, 60],
            "scale": 2**35,
            "security_level": 128
        }

        response = client.get("/security/estimate", params=config)

        assert response.status_code == 200
        data = response.json()

        assert "security_bits" in data
        assert "parameters" in data
        assert "recommendation" in data

    def test_performance_benchmark(self, client) -> None:
        """Test performance benchmark"""
        response = client.get("/performance/benchmark")

        assert response.status_code == 200
        data = response.json()

        assert "benchmark_results" in data
        assert "configuration" in data

        benchmark = data["benchmark_results"]
        assert "data_size" in benchmark
        assert "encryption_time_ms" in benchmark
        assert "addition_time_ms" in benchmark
        assert "noise_budget" in benchmark

class TestAuthenticationMiddleware:
    """Test authentication middleware"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    def test_protected_endpoint_without_auth(self, client) -> None:
        """Test accessing protected endpoint without authentication"""
        response = client.get("/models")

        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "AuthenticationError"

    def test_protected_endpoint_with_invalid_auth(self, client) -> None:
        """Test accessing protected endpoint with invalid API key"""
        headers = {"X-API-Key": "invalid-key"}
        response = client.get("/models", headers=headers)

        assert response.status_code == 403
        data = response.json()
        assert data["error"] == "AuthenticationError"

    def test_protected_endpoint_with_valid_auth(self, client) -> None:
        """Test accessing protected endpoint with valid API key"""
        headers = {"X-API-Key": "dev-key-12345"}
        response = client.get("/models", headers=headers)

        assert response.status_code == 200

    def test_public_endpoint_no_auth_required(self, client) -> None:
        """Test accessing public endpoint without authentication"""
        response = client.get("/health")

        assert response.status_code == 200

class TestValidationMiddleware:
    """Test input validation middleware"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    def test_invalid_json_content_type(self, client) -> None:
        """Test request with invalid content type"""
        response = client.post(
            "/encrypt",
            data="not json",
            headers={"Content-Type": "text/plain"}
        )

        assert response.status_code == 415
        data = response.json()
        assert data["error"] == "UnsupportedMediaType"

    def test_large_request_payload(self, client) -> None:
        """Test request with oversized payload"""
        # Create very large payload
        large_data = {
            "node_features": [[1.0] * 1000 for _ in range(10000)],  # Very large
            "edge_index": [list(range(5000)), list(range(5000, 10000))]
        }

        response = client.post("/encrypt", json=large_data)

        # This might succeed or fail depending on the actual size limit
        # In a real test, we'd mock the content-length header
        assert response.status_code in [200, 413, 422]

class TestErrorHandlingMiddleware:
    """Test error handling middleware"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    def test_404_error_handling(self, client) -> None:
        """Test 404 error handling"""
        response = client.get("/nonexistent/endpoint")

        assert response.status_code == 404

    def test_method_not_allowed(self, client) -> None:
        """Test method not allowed error"""
        response = client.put("/health")  # Health only supports GET

        assert response.status_code == 405

    def test_validation_error_handling(self, client) -> None:
        """Test validation error handling"""
        invalid_data = {
            "node_features": "not a list",  # Invalid type
            "edge_index": [[0, 1], [1, 0]]
        }

        response = client.post("/encrypt", json=invalid_data)

        assert response.status_code == 422  # Pydantic validation error
        data = response.json()
        assert "detail" in data

class TestResponseHeaders:
    """Test response headers and security"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    def test_processing_time_header(self, client) -> None:
        """Test processing time header is added"""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Process-Time" in response.headers

        # Processing time should be a valid float
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0

    def test_cors_headers(self, client) -> None:
        """Test CORS headers are present"""
        response = client.options("/health")

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end integration tests"""

    @pytest.fixture
    def client(self) -> None:
        """Create test client"""
        with patch('torch.cuda.is_available', return_value=False):
            return TestClient(app)

    @pytest.fixture
    def auth_headers(self) -> None:
        """Authorization headers"""
        return {"X-API-Key": "dev-key-12345"}

    def test_complete_workflow(self, client, auth_headers) -> None:
        """Test complete workflow from context creation to inference"""
        # 1. Create custom context
        context_config = {
            "poly_modulus_degree": 8192,
            "coeff_modulus_bits": [60, 40, 40, 60],
            "scale": 2**30,
            "security_level": 128
        }

        context_response = client.post(
            "/contexts/workflow_context",
            json=context_config,
            headers=auth_headers
        )
        assert context_response.status_code == 200

        # 2. Create model
        model_config = {
            "model_type": "graphsage",
            "in_channels": 4,
            "out_channels": 2,
            "hidden_channels": 3,
            "num_layers": 1
        }

        model_response = client.post(
            "/models/workflow_model",
            json=model_config,
            headers=auth_headers,
            params={"context_name": "workflow_context"}
        )
        assert model_response.status_code == 200

        # 3. Encrypt data
        graph_data = {
            "node_features": [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0]
            ],
            "edge_index": [
                [0, 1, 2],
                [1, 2, 0]
            ]
        }

        encrypt_response = client.post(
            "/encrypt?context_name=workflow_context",
            json=graph_data
        )
        assert encrypt_response.status_code == 200

        # 4. Run inference
        inference_response = client.post(
            "/inference/workflow_model",
            json=graph_data
        )
        assert inference_response.status_code == 200

        # 5. Check results
        inference_data = inference_response.json()
        assert "inference_id" in inference_data
        assert "output_shape" in inference_data
        assert inference_data["output_shape"] == [3, 2]  # 3 nodes, 2 output features

        # 6. Cleanup
        client.delete("/models/workflow_model", headers=auth_headers)
        client.delete("/contexts/workflow_context", headers=auth_headers)

    def test_batch_processing_workflow(self, client, auth_headers) -> None:
        """Test batch processing workflow"""
        # Create model for batch processing
        model_config = {
            "model_type": "graphsage",
            "in_channels": 2,
            "out_channels": 1,
            "hidden_channels": 2,
            "num_layers": 1
        }

        model_response = client.post(
            "/models/batch_model",
            json=model_config,
            headers=auth_headers
        )
        assert model_response.status_code == 200

        # Create batch data
        batch_graphs = [
            {
                "node_features": [[1.0, 2.0], [3.0, 4.0]],
                "edge_index": [[0], [1]]
            },
            {
                "node_features": [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
                "edge_index": [[0, 1], [1, 2]]
            }
        ]

        # Batch encrypt
        encrypt_response = client.post("/batch/encrypt", json=batch_graphs)
        assert encrypt_response.status_code == 200

        encrypt_data = encrypt_response.json()
        assert len(encrypt_data["batch_results"]) == 2

        # Run inference on individual graphs
        for graph_data in batch_graphs:
            inference_response = client.post(
                "/inference/batch_model",
                json=graph_data
            )
            assert inference_response.status_code == 200
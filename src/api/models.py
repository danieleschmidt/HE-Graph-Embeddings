"""
Pydantic models for API request/response validation
"""


try:
    from pydantic import BaseModel as PydanticBaseModel, Field, validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    PydanticBaseModel = None
    Field = None
    validator = None

from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Fallback BaseModel if Pydantic not available
class BaseModel:
    """Fallback BaseModel when Pydantic is not available"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Use Pydantic BaseModel if available, otherwise fallback
if HAS_PYDANTIC:
    BaseModel = PydanticBaseModel

class ModelType(str, Enum):
    """Supported model types"""
    GRAPHSAGE = "graphsage"
    GAT = "gat"
    GRAPH_CONV = "graph_conv"

class AggregatorType(str, Enum):
    """Aggregation types for GraphSAGE"""
    MEAN = "mean"
    MAX = "max"
    SUM = "sum"
    LSTM = "lstm"

class AttentionType(str, Enum):
    """Attention mechanisms for GAT"""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    SCALED_DOT_PRODUCT = "scaled_dot_product"

class GraphRequest(BaseModel):
    """Request model for graph data"""
    node_features: List[List[float]] = Field(
        ...,
        description="Node feature matrix (N x D)",
        min_items=1
    )
    edge_index: List[List[int]] = Field(
        ...,
        description="Edge indices (2 x E)",
        min_items=2,
        max_items=2
    )
    edge_attributes: Optional[List[List[float]]] = Field(
        None,
        description="Edge feature matrix (E x D_edge)"
    )
    node_labels: Optional[List[int]] = Field(
        None,
        description="Node labels for supervised learning"
    )
    graph_labels: Optional[List[int]] = Field(
        None,
        description="Graph-level labels"
    )

    @validator('node_features')
    def validate_node_features(cls, v):
        """Validate Node Features."""
        if not v:
            raise ValueError("Node features cannot be empty")

        # Check that all nodes have same feature dimension
        feature_dim = len(v[0])
        for i, node_feat in enumerate(v):
            if len(node_feat) != feature_dim:
                raise ValueError(f"Inconsistent feature dimension at node {i}")

        return v

    @validator('edge_index')
    def validate_edge_index(cls, v):
        """Validate Edge Index."""
        if len(v) != 2:
            raise ValueError("Edge index must have exactly 2 rows (source, target)")

        if len(v[0]) != len(v[1]):
            raise ValueError("Source and target edge lists must have same length")

        return v

    class Config:
        """Config class."""
        schema_extra = {
            "example": {
                "node_features": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                "edge_index": [[0, 1, 2], [1, 2, 0]],
                "edge_attributes": [[0.5], [0.7], [0.3]],
                "node_labels": [0, 1, 0]
            }
        }

class ModelTrainingRequest(BaseModel):
    """Request model for creating/training models"""
    model_type: ModelType = Field(..., description="Type of GNN model")
    in_channels: int = Field(..., gt=0, description="Input feature dimension")
    out_channels: int = Field(..., gt=0, description="Output feature dimension")
    hidden_channels: Union[int, List[int]] = Field(
        default=64,
        description="Hidden layer dimensions"
    )
    num_layers: Optional[int] = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of layers"
    )

    # GraphSAGE specific
    aggregator: Optional[AggregatorType] = Field(
        default=AggregatorType.MEAN,
        description="Aggregation function for GraphSAGE"
    )

    # GAT specific
    heads: Optional[int] = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of attention heads for GAT"
    )
    attention_type: Optional[AttentionType] = Field(
        default=AttentionType.ADDITIVE,
        description="Attention mechanism type"
    )
    edge_dim: Optional[int] = Field(
        default=None,
        ge=1,
        description="Edge feature dimension"
    )

    # Training parameters
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    epochs: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=32, ge=1, le=10000)
    dropout: float = Field(default=0.0, ge=0, le=0.9)
    weight_decay: float = Field(default=0.0, ge=0)

    # Encryption parameters
    noise_budget_threshold: float = Field(
        default=10.0,
        ge=1.0,
        description="Minimum noise budget before bootstrapping"
    )
    auto_rescale: bool = Field(
        default=True,
        description="Automatically rescale during training"
    )

    @validator('hidden_channels')
    def validate_hidden_channels(cls, v, values):
        """Validate Hidden Channels."""
        if isinstance(v, int):
            num_layers = values.get('num_layers', 2)
            return [v] * num_layers
        elif isinstance(v, list):
            if 'num_layers' in values and len(v) != values['num_layers']:
                raise ValueError("Length of hidden_channels must match num_layers")
            return v
        else:
            raise ValueError("hidden_channels must be int or list of ints")

    class Config:
        """Config class."""
        schema_extra = {
            "example": {
                "model_type": "graphsage",
                "in_channels": 128,
                "out_channels": 32,
                "hidden_channels": [64, 64],
                "num_layers": 2,
                "aggregator": "mean",
                "learning_rate": 0.01,
                "epochs": 100,
                "batch_size": 32
            }
        }

class InferenceRequest(BaseModel):
    """Request model for inference"""
    node_features: List[List[float]] = Field(
        ...,
        description="Node feature matrix"
    )
    edge_index: List[List[int]] = Field(
        ...,
        description="Edge indices"
    )
    edge_attributes: Optional[List[List[float]]] = Field(
        None,
        description="Edge features"
    )
    return_attention_weights: bool = Field(
        default=False,
        description="Return attention weights (GAT only)"
    )
    return_embeddings: bool = Field(
        default=False,
        description="Return intermediate embeddings"
    )
    decrypt_output: bool = Field(
        default=False,
        description="Decrypt output (for testing only)"
    )

    class Config:
        """Config class."""
        schema_extra = {
            "example": {
                "node_features": [[1.0, 2.0], [3.0, 4.0]],
                "edge_index": [[0], [1]],
                "return_attention_weights": False,
                "decrypt_output": False
            }
        }

class BatchInferenceRequest(BaseModel):
    """Request model for batch inference"""
    graphs: List[InferenceRequest] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of graphs for batch processing"
    )
    parallel_processing: bool = Field(
        default=True,
        description="Process graphs in parallel"
    )

    class Config:
        """Config class."""
        schema_extra = {
            "example": {
                "graphs": [
                    {
                        "node_features": [[1.0, 2.0], [3.0, 4.0]],
                        "edge_index": [[0], [1]]
                    }
                ],
                "parallel_processing": True
            }
        }

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")
    gpu_available: bool = Field(..., description="GPU availability")
    active_contexts: int = Field(..., description="Number of active CKKS contexts")
    active_models: int = Field(..., description="Number of loaded models")

class EncryptionResponse(BaseModel):
    """Response for encryption operations"""
    encryption_id: str = Field(..., description="Unique encryption identifier")
    encrypted_shape: List[int] = Field(..., description="Shape of encrypted data")
    num_edges: Optional[int] = Field(None, description="Number of edges")
    scale: float = Field(..., description="CKKS scale parameter")
    noise_budget: float = Field(..., description="Remaining noise budget")
    context_name: str = Field(default="default", description="CKKS context used")

class InferenceResponse(BaseModel):
    """Response for inference operations"""
    inference_id: str = Field(..., description="Unique inference identifier")
    output_shape: List[int] = Field(..., description="Shape of output")
    noise_budget: float = Field(..., description="Remaining noise budget")
    processing_time_ms: Optional[float] = Field(
        None,
        description="Processing time in milliseconds"
    )
    attention_weights: Optional[List[List[float]]] = Field(
        None,
        description="Attention weights (if requested)"
    )
    embeddings: Optional[List[List[float]]] = Field(
        None,
        description="Intermediate embeddings (if requested)"
    )
    decrypted_output: Optional[List[List[float]]] = Field(
        None,
        description="Decrypted output (if requested, testing only)"
    )

class BatchInferenceResponse(BaseModel):
    """Response for batch inference"""
    batch_id: str = Field(..., description="Batch processing identifier")
    results: List[InferenceResponse] = Field(..., description="Individual results")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    successful_inferences: int = Field(..., description="Number of successful inferences")
    failed_inferences: int = Field(..., description="Number of failed inferences")

class SecurityEstimateResponse(BaseModel):
    """Response for security estimation"""
    security_bits: int = Field(..., description="Estimated security in bits")
    parameters: Dict[str, Any] = Field(..., description="Input parameters")
    recommendation: str = Field(..., description="Security recommendation")
    warnings: Optional[List[str]] = Field(None, description="Security warnings")

class BenchmarkResponse(BaseModel):
    """Response for performance benchmarks"""
    benchmark_results: Dict[str, Any] = Field(..., description="Benchmark metrics")
    configuration: Dict[str, Any] = Field(..., description="Test configuration")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Config class."""
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input parameters",
                "details": {"field": "node_features", "issue": "cannot be empty"},
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }

class ContextInfo(BaseModel):
    """CKKS context information"""
    name: str = Field(..., description="Context name")
    poly_degree: int = Field(..., description="Polynomial modulus degree")
    security_level: int = Field(..., description="Security level in bits")
    scale: float = Field(..., description="CKKS scale parameter")
    coeff_modulus_bits: List[int] = Field(..., description="Coefficient modulus chain")
    created_at: datetime = Field(..., description="Creation timestamp")

class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    model_type: ModelType = Field(..., description="Model type")
    context_name: str = Field(..., description="Associated CKKS context")
    in_channels: int = Field(..., description="Input feature dimension")
    out_channels: int = Field(..., description="Output feature dimension")
    parameters: Dict[str, Any] = Field(..., description="Model parameters")
    created_at: datetime = Field(..., description="Creation timestamp")
    training_status: str = Field(default="initialized", description="Training status")

class TrainingProgress(BaseModel):
    """Training progress information"""
    epoch: int = Field(..., description="Current epoch")
    total_epochs: int = Field(..., description="Total epochs")
    loss: float = Field(..., description="Current loss")
    noise_budget: float = Field(..., description="Current noise budget")
    elapsed_time_ms: float = Field(..., description="Elapsed training time")
    estimated_remaining_ms: Optional[float] = Field(
        None,
        description="Estimated remaining time"
    )
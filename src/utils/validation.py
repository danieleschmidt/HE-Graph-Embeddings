"""
Comprehensive input validation utilities for HE-Graph-Embeddings
"""


from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import torch

from .error_handling import ValidationError, SecurityError
from .logging import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __post_init__(self):
        """  Post Init  ."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    def add_error(self, error: str) -> None:
        """Add validation error"""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str) -> None:
        """Add validation warning"""
        self.warnings.append(warning)

    def raise_if_invalid(self) -> None:
        """Raise ValidationError if validation failed"""
        if not self.is_valid:
            raise ValidationError(f"Validation failed: {'; '.join(self.errors)}")

class TensorValidator:
    """Validator for PyTorch tensors"""

    @staticmethod
    def validate_tensor(tensor: Any, name: str = "tensor",
                        min_dims: int = None, max_dims: int = None,
                        min_size: int = None, max_size: int = None,
                        dtype: torch.dtype = None,
                        finite_only: bool = True,
                        allow_empty: bool = False) -> ValidationResult:
        """Comprehensive tensor validation"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # Type check
        if not isinstance(tensor, torch.Tensor):
            result.add_error(f"{name} must be a torch.Tensor, got {type(tensor)}")
            return result

        # Empty tensor check
        if tensor.numel() == 0 and not allow_empty:
            result.add_error(f"{name} cannot be empty")

        # Dimension checks
        if min_dims is not None and tensor.dim() < min_dims:
            result.add_error(f"{name} must have at least {min_dims} dimensions, got {tensor.dim()}")

        if max_dims is not None and tensor.dim() > max_dims:
            result.add_error(f"{name} must have at most {max_dims} dimensions, got {tensor.dim()}")

        # Size checks
        if min_size is not None and tensor.numel() < min_size:
            result.add_error(f"{name} must have at least {min_size} elements, got {tensor.numel()}")

        if max_size is not None and tensor.numel() > max_size:
            result.add_error(f"{name} must have at most {max_size} elements, got {tensor.numel()}")

        # Data type check
        if dtype is not None and tensor.dtype != dtype:
            result.add_warning(f"{name} expected dtype {dtype}, got {tensor.dtype}")

        # Finite values check
        if finite_only and tensor.numel() > 0:
            if not torch.isfinite(tensor).all():
                result.add_error(f"{name} contains infinite or NaN values")

        # Range warnings for large values
        if tensor.numel() > 0 and torch.is_floating_point(tensor):
            max_val = torch.abs(tensor).max().item()
            if max_val > 1e6:
                result.add_warning(f"{name} contains very large values (max: {max_val:.2e})")

        return result

    @staticmethod
    def validate_graph_tensor(tensor: torch.Tensor, num_nodes: int,
                            feature_dim: int, name: str = "graph_tensor") -> ValidationResult:
        """Validate graph-specific tensor shapes"""
        result = TensorValidator.validate_tensor(
            tensor, name, min_dims=2, max_dims=2,
            finite_only=True, allow_empty=False
        )

        if result.is_valid:
            expected_shape = (num_nodes, feature_dim)
            if tensor.shape != expected_shape:
                result.add_error(f"{name} shape {tensor.shape} doesn't match expected {expected_shape}")

        return result

    @staticmethod
    def validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> ValidationResult:
        """Validate graph edge index tensor"""
        result = TensorValidator.validate_tensor(
            edge_index, "edge_index", min_dims=2, max_dims=2,
            dtype=torch.long, finite_only=True, allow_empty=True
        )

        if result.is_valid and edge_index.numel() > 0:
            # Check shape is [2, num_edges]
            if edge_index.size(0) != 2:
                result.add_error(f"edge_index must have shape [2, num_edges], got {edge_index.shape}")

            # Check node indices are valid
            max_idx = edge_index.max().item() if edge_index.numel() > 0 else -1
            min_idx = edge_index.min().item() if edge_index.numel() > 0 else 0

            if min_idx < 0:
                result.add_error("edge_index contains negative node indices")

            if max_idx >= num_nodes:
                result.add_error(f"edge_index contains node index {max_idx} >= num_nodes {num_nodes}")

            # Check for self-loops (warning only)
            if edge_index.size(1) > 0:
                self_loops = (edge_index[0] == edge_index[1]).sum().item()
                if self_loops > 0:
                    result.add_warning(f"edge_index contains {self_loops} self-loops")

        return result

class EncryptionValidator:
    """Validator for encryption-related operations"""

    @staticmethod
    def validate_ckks_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate CKKS configuration parameters"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # Polynomial degree validation
        if "poly_modulus_degree" in config:
            degree = config["poly_modulus_degree"]
            if not isinstance(degree, int) or degree <= 0:
                result.add_error("poly_modulus_degree must be positive integer")
            elif degree & (degree - 1) != 0:
                result.add_error("poly_modulus_degree must be power of 2")
            elif degree < 1024:
                result.add_error("poly_modulus_degree too small for security (minimum 1024)")
            elif degree > 65536:
                result.add_error("poly_modulus_degree too large (maximum 65536)")

        # Coefficient modulus validation
        if "coeff_modulus_bits" in config:
            coeff_bits = config["coeff_modulus_bits"]
            if not isinstance(coeff_bits, list) or len(coeff_bits) < 2:
                result.add_error("coeff_modulus_bits must be list with at least 2 elements")
            else:
                for i, bits in enumerate(coeff_bits):
                    if not isinstance(bits, int) or bits < 20 or bits > 60:
                        result.add_error(f"coeff_modulus_bits[{i}] must be between 20 and 60")

        # Scale validation
        if "scale" in config:
            scale = config["scale"]
            if not isinstance(scale, (int, float)) or scale <= 0:
                result.add_error("scale must be positive number")
            elif scale < 2**20:
                result.add_warning("scale may be too small for precision")
            elif scale > 2**60:
                result.add_error("scale too large (maximum 2^60)")

        # Security level validation
        if "security_level" in config:
            security = config["security_level"]
            if security not in [128, 192, 256]:
                result.add_error("security_level must be 128, 192, or 256")

        # Cross-parameter validation
        if "coeff_modulus_bits" in config and "security_level" in config:
            total_bits = sum(config["coeff_modulus_bits"])
            security = config["security_level"]

            limits = {128: 438, 192: 305, 256: 239}
            if total_bits > limits.get(security, 438):
                result.add_error(f"Total modulus bits {total_bits} exceeds {security}-bit security limit")

        return result

    @staticmethod
    def validate_noise_budget(ciphertext: Any, min_budget: float = 10.0) -> ValidationResult:
        """Validate ciphertext has sufficient noise budget"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        try:
            if hasattr(ciphertext, 'noise_budget'):
                budget = ciphertext.noise_budget
                if budget < min_budget:
                    result.add_error(f"Insufficient noise budget: {budget:.1f} < {min_budget}")
                elif budget < min_budget * 2:
                    result.add_warning(f"Low noise budget: {budget:.1f}")
            else:
                result.add_warning("Cannot validate noise budget - method not available")
        except Exception as e:
            logger.error(f"Error in operation: {e}")
            result.add_warning(f"Noise budget validation failed: {e}")

        return result

class GraphValidator:
    """Validator for graph neural network operations"""

    @staticmethod
    def validate_graph_structure(num_nodes: int, edge_index: torch.Tensor,
                                features: torch.Tensor) -> ValidationResult:
        """Validate complete graph structure"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # Basic parameter validation
        if num_nodes <= 0:
            result.add_error("num_nodes must be positive")
            return result

        # Validate features
        feature_result = TensorValidator.validate_tensor(
            features, "features", min_dims=2, max_dims=2, finite_only=True
        )
        result.errors.extend(feature_result.errors)
        result.warnings.extend(feature_result.warnings)

        if feature_result.is_valid:
            if features.size(0) != num_nodes:
                result.add_error(f"features first dimension {features.size(0)} != num_nodes {num_nodes}")

            feature_dim = features.size(1)
            if feature_dim <= 0:
                result.add_error("feature dimension must be positive")

        # Validate edge index
        edge_result = TensorValidator.validate_edge_index(edge_index, num_nodes)
        result.errors.extend(edge_result.errors)
        result.warnings.extend(edge_result.warnings)

        # Graph connectivity warnings
        if edge_result.is_valid and edge_index.numel() > 0:
            num_edges = edge_index.size(1)

            # Check for isolated nodes
            unique_nodes = torch.unique(edge_index)
            if len(unique_nodes) < num_nodes:
                isolated_count = num_nodes - len(unique_nodes)
                result.add_warning(f"Graph has {isolated_count} isolated nodes")

            # Density check
            max_edges = num_nodes * (num_nodes - 1) // 2  # Undirected graph
            density = num_edges / max_edges if max_edges > 0 else 0

            if density < 0.01:
                result.add_warning(f"Very sparse graph (density: {density:.3f})")
            elif density > 0.5:
                result.add_warning(f"Very dense graph (density: {density:.3f})")

        return result

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate graph neural network model configuration"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        # Input/output dimension validation
        required_params = ["in_channels", "out_channels"]
        for param in required_params:
            if param not in config:
                result.add_error(f"Missing required parameter: {param}")
            elif not isinstance(config[param], int) or config[param] <= 0:
                result.add_error(f"{param} must be positive integer")

        # Hidden channels validation
        if "hidden_channels" in config:
            hidden = config["hidden_channels"]
            if isinstance(hidden, int):
                if hidden <= 0:
                    result.add_error("hidden_channels must be positive")
            elif isinstance(hidden, list):
                if not hidden:
                    result.add_error("hidden_channels list cannot be empty")
                for i, dim in enumerate(hidden):
                    if not isinstance(dim, int) or dim <= 0:
                        result.add_error(f"hidden_channels[{i}] must be positive integer")
            else:
                result.add_error("hidden_channels must be int or list of ints")

        # Number of layers validation
        if "num_layers" in config:
            layers = config["num_layers"]
            if not isinstance(layers, int) or layers <= 0:
                result.add_error("num_layers must be positive integer")
            elif layers > 10:
                result.add_warning(f"Large number of layers ({layers}) may cause noise budget issues")

        # Aggregator validation
        if "aggregator" in config:
            aggregator = config["aggregator"]
            valid_aggregators = ["mean", "sum", "max", "min"]
            if aggregator not in valid_aggregators:
                result.add_error(f"aggregator must be one of {valid_aggregators}")

        # Activation validation
        if "activation" in config:
            activation = config["activation"]
            valid_activations = ["relu_poly", "tanh_poly", "sigmoid_poly", "none"]
            if activation not in valid_activations:
                result.add_error(f"activation must be one of {valid_activations}")

        # Dropout validation
        if "dropout_enc" in config:
            dropout = config["dropout_enc"]
            if not isinstance(dropout, (int, float)) or dropout < 0 or dropout >= 1:
                result.add_error("dropout_enc must be in [0, 1)")

        return result

class SecurityValidator:
    """Validator for security-related checks"""

    @staticmethod
    def validate_security_level(poly_degree: int, coeff_modulus_bits: List[int],
                                target_security: int = 128) -> ValidationResult:
        """Validate security level using simplified LWE estimator"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        total_bits = sum(coeff_modulus_bits)

        # Simplified security estimation (based on common parameters)
        if target_security == 128:
            if poly_degree >= 32768 and total_bits <= 438:
                estimated_security = 128
            elif poly_degree >= 16384 and total_bits <= 218:
                estimated_security = 128
            elif poly_degree >= 8192 and total_bits <= 109:
                estimated_security = 128
            else:
                estimated_security = 80  # Conservative estimate
        elif target_security == 192:
            if poly_degree >= 32768 and total_bits <= 305:
                estimated_security = 192
            else:
                estimated_security = 128
        else:  # 256-bit security
            if poly_degree >= 32768 and total_bits <= 239:
                estimated_security = 256
            else:
                estimated_security = 192

        if estimated_security < target_security:
            result.add_error(f"Estimated security {estimated_security} bits < target {target_security} bits")
        elif estimated_security < target_security + 20:
            result.add_warning(f"Security level {estimated_security} bits is close to minimum")

        return result

    @staticmethod
    def validate_key_security(secret_key: torch.Tensor) -> ValidationResult:
        """Validate secret key has sufficient entropy"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        if secret_key.numel() == 0:
            result.add_error("Secret key cannot be empty")
            return result

        # Check for ternary distribution
        unique_vals = torch.unique(secret_key)
        if len(unique_vals) != 3:
            result.add_error("Secret key should have ternary distribution (-1, 0, 1)")

        # Check distribution balance
        counts = torch.bincount(secret_key + 1)  # Shift to [0, 1, 2]
        if len(counts) == 3:
            total = secret_key.numel()
            ratios = counts.float() / total

            # Each value should appear roughly 1/3 of the time
            for i, ratio in enumerate(ratios):
                expected = 1.0 / 3.0
                if abs(ratio - expected) > 0.1:  # 10% tolerance
                    result.add_warning(f"Unbalanced secret key distribution: value {i-1} appears {ratio:.2%}")

        return result

# Global validation functions

def validate_input(data: Any, validation_type: str = "tensor", **kwargs) -> ValidationResult:
    """General input validation dispatcher"""
    if validation_type == "tensor":
        return TensorValidator.validate_tensor(data, **kwargs)
    elif validation_type == "graph":
        return GraphValidator.validate_graph_structure(**kwargs)
    elif validation_type == "ckks_config":
        return EncryptionValidator.validate_ckks_config(data)
    elif validation_type == "model_config":
        return GraphValidator.validate_model_config(data)
    else:
        result = ValidationResult(is_valid=False, errors=[], warnings=[])
        result.add_error(f"Unknown validation type: {validation_type}")
        return result

def safe_validate(validator_func, *args, **kwargs) -> ValidationResult:
    """Safely run validation with exception handling"""
    try:
        return validator_func(*args, **kwargs)
    except Exception as e:
        result = ValidationResult(is_valid=False, errors=[], warnings=[])
        result.add_error(f"Validation failed with exception: {str(e)}")
        logger.error(f"Validation exception: {e}")
        return result

# Decorators for automatic validation

def validate_tensor_input(tensor_name: str = "tensor", **validation_kwargs):
    """Decorator to validate tensor inputs"""
    def decorator(func):
        """Decorator."""
        def wrapper(*args, **kwargs):
            """Wrapper."""
            # Extract tensor from arguments (simple implementation)
            if tensor_name in kwargs:
                tensor = kwargs[tensor_name]
                result = TensorValidator.validate_tensor(tensor, tensor_name, **validation_kwargs)
                result.raise_if_invalid()

            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_graph_input(func):
    """Decorator to validate graph structure inputs"""
    def wrapper(*args, **kwargs):
        """Wrapper."""
        # Look for common graph parameters
        if 'features' in kwargs and 'edge_index' in kwargs:
            features = kwargs['features']
            edge_index = kwargs['edge_index']
            num_nodes = features.size(0) if hasattr(features, 'size') else kwargs.get('num_nodes', 0)

            result = GraphValidator.validate_graph_structure(num_nodes, edge_index, features)
            result.raise_if_invalid()

        return func(*args, **kwargs)
    return wrapper
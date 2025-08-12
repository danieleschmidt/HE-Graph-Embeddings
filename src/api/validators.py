"""
Input validation utilities for HE-Graph-Embeddings API
"""


import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from pydantic import validator
import logging

logger = logging.getLogger(__name__)

class GraphDataValidator:
    """Validator for graph data structures"""

    @staticmethod
    def validate_node_features(features: List[List[float]]) -> Tuple[bool, Optional[str]]:
        """Validate node feature matrix"""
        if not features:
            return False, "Node features cannot be empty"

        if not isinstance(features, list):
            return False, "Node features must be a list"

        # Check dimensions consistency
        if not all(isinstance(node, list) for node in features):
            return False, "All node features must be lists"

        feature_dim = len(features[0])
        if feature_dim == 0:
            return False, "Feature dimension cannot be zero"

        for i, node_feat in enumerate(features):
            if len(node_feat) != feature_dim:
                return False, f"Inconsistent feature dimension at node {i}: expected {feature_dim}, got {len(node_feat)}"

            if not all(isinstance(val, (int, float)) for val in node_feat):
                return False, f"All feature values must be numeric at node {i}"

            # Check for NaN/Inf
            if any(np.isnan(val) or np.isinf(val) for val in node_feat):
                return False, f"Invalid values (NaN/Inf) found at node {i}"

        return True, None

    @staticmethod
    def validate_edge_index(edge_index: List[List[int]], num_nodes: int) -> Tuple[bool, Optional[str]]:
        """Validate edge index structure"""
        if not edge_index:
            return False, "Edge index cannot be empty"

        if len(edge_index) != 2:
            return False, "Edge index must have exactly 2 rows (source, target)"

        source_nodes, target_nodes = edge_index

        if len(source_nodes) != len(target_nodes):
            return False, "Source and target node lists must have same length"

        if not source_nodes:  # Empty edge list is valid
            return True, None

        # Validate node indices
        for i, (src, tgt) in enumerate(zip(source_nodes, target_nodes)):
            if not isinstance(src, int) or not isinstance(tgt, int):
                return False, f"Edge indices must be integers at edge {i}"

            if src < 0 or src >= num_nodes:
                return False, f"Source node index {src} out of range [0, {num_nodes}) at edge {i}"

            if tgt < 0 or tgt >= num_nodes:
                return False, f"Target node index {tgt} out of range [0, {num_nodes}) at edge {i}"

        return True, None

    @staticmethod
    def validate_edge_attributes(edge_attr: List[List[float]], num_edges: int) -> Tuple[bool, Optional[str]]:
        """Validate edge attribute matrix"""
        if not edge_attr:
            return True, None  # Edge attributes are optional

        if len(edge_attr) != num_edges:
            return False, f"Number of edge attributes ({len(edge_attr)}) must match number of edges ({num_edges})"

        if not edge_attr[0]:
            return False, "Edge attribute dimension cannot be zero"

        attr_dim = len(edge_attr[0])

        for i, edge_feat in enumerate(edge_attr):
            if len(edge_feat) != attr_dim:
                return False, f"Inconsistent edge attribute dimension at edge {i}: expected {attr_dim}, got {len(edge_feat)}"

            if not all(isinstance(val, (int, float)) for val in edge_feat):
                return False, f"All edge attribute values must be numeric at edge {i}"

            if any(np.isnan(val) or np.isinf(val) for val in edge_feat):
                return False, f"Invalid values (NaN/Inf) found in edge attributes at edge {i}"

        return True, None

    @staticmethod
    def validate_graph_structure(node_features: List[List[float]],
        """Validate Graph Structure."""
                                edge_index: List[List[int]],
                                edge_attributes: Optional[List[List[float]]] = None) -> Tuple[bool, Optional[str]]:
        """Validate complete graph structure"""
        # Validate node features
        valid, error = GraphDataValidator.validate_node_features(node_features)
        if not valid:
            return False, f"Node features validation failed: {error}"

        num_nodes = len(node_features)

        # Validate edge index
        valid, error = GraphDataValidator.validate_edge_index(edge_index, num_nodes)
        if not valid:
            return False, f"Edge index validation failed: {error}"

        num_edges = len(edge_index[0]) if edge_index[0] else 0

        # Validate edge attributes if provided
        if edge_attributes is not None:
            valid, error = GraphDataValidator.validate_edge_attributes(edge_attributes, num_edges)
            if not valid:
                return False, f"Edge attributes validation failed: {error}"

        return True, None

    @staticmethod
    def check_graph_connectivity(edge_index: List[List[int]], num_nodes: int) -> Dict[str, Any]:
        """Analyze graph connectivity properties"""
        if not edge_index[0]:  # No edges
            return {
                "is_connected": False,
                "num_components": num_nodes,
                "num_self_loops": 0,
                "is_directed": False,
                "density": 0.0
            }

        source_nodes, target_nodes = edge_index
        edges = set(zip(source_nodes, target_nodes))

        # Count self-loops
        self_loops = sum(1 for src, tgt in edges if src == tgt)

        # Check if directed (existence of both (u,v) and (v,u) for all edges)
        reverse_edges = {(tgt, src) for src, tgt in edges}
        is_directed = not edges.issubset(reverse_edges)

        # Calculate density
        max_edges = num_nodes * (num_nodes - 1)
        if not is_directed:
            max_edges //= 2
        density = len(edges) / max_edges if max_edges > 0 else 0.0

        # Simple connectivity check (BFS)
        if num_nodes == 0:
            is_connected = True
            num_components = 0
        else:
            adj_list = {i: [] for i in range(num_nodes)}
            for src, tgt in edges:
                adj_list[src].append(tgt)
                if not is_directed:
                    adj_list[tgt].append(src)

            visited = [False] * num_nodes
            num_components = 0

            for start in range(num_nodes):
                if not visited[start]:
                    # BFS from this node
                    queue = [start]
                    visited[start] = True
                    num_components += 1

                    while queue:
                        node = queue.pop(0)
                        for neighbor in adj_list[node]:
                            if not visited[neighbor]:
                                visited[neighbor] = True
                                queue.append(neighbor)

            is_connected = (num_components == 1)

        return {
            "is_connected": is_connected,
            "num_components": num_components,
            "num_self_loops": self_loops,
            "is_directed": is_directed,
            "density": density,
            "num_edges": len(edges),
            "num_nodes": num_nodes
        }

class ModelConfigValidator:
    """Validator for model configuration parameters"""

    @staticmethod
    def validate_dimensions(in_channels: int, hidden_channels: List[int],
        """Validate Dimensions."""
                            out_channels: int) -> Tuple[bool, Optional[str]]:
        """Validate model dimensions"""
        if in_channels <= 0:
            return False, "Input channels must be positive"

        if out_channels <= 0:
            return False, "Output channels must be positive"

        if not hidden_channels:
            return False, "Hidden channels cannot be empty"

        for i, dim in enumerate(hidden_channels):
            if dim <= 0:
                return False, f"Hidden channel dimension at layer {i} must be positive"

        # Check for reasonable dimensions (prevent memory issues)
        max_dim = 10000
        if in_channels > max_dim:
            return False, f"Input channels {in_channels} exceeds maximum {max_dim}"

        if out_channels > max_dim:
            return False, f"Output channels {out_channels} exceeds maximum {max_dim}"

        for i, dim in enumerate(hidden_channels):
            if dim > max_dim:
                return False, f"Hidden channel dimension {dim} at layer {i} exceeds maximum {max_dim}"

        return True, None

    @staticmethod
    def validate_training_params(learning_rate: float, epochs: int,
        """Validate Training Params."""
                                batch_size: int, dropout: float) -> Tuple[bool, Optional[str]]:
        """Validate training parameters"""
        if learning_rate <= 0 or learning_rate > 1:
            return False, "Learning rate must be in (0, 1]"

        if epochs <= 0 or epochs > 10000:
            return False, "Epochs must be in [1, 10000]"

        if batch_size <= 0 or batch_size > 10000:
            return False, "Batch size must be in [1, 10000]"

        if dropout < 0 or dropout >= 1:
            return False, "Dropout must be in [0, 1)"

        return True, None

    @staticmethod
    def validate_encryption_params(poly_degree: int, coeff_modulus_bits: List[int],
        """Validate Encryption Params."""
                                scale: float, security_level: int) -> Tuple[bool, Optional[str]]:
        """Validate CKKS encryption parameters"""
        # Check poly_degree is power of 2
        if poly_degree <= 0 or (poly_degree & (poly_degree - 1)) != 0:
            return False, "Polynomial degree must be a positive power of 2"

        # Check reasonable range
        if poly_degree < 1024 or poly_degree > 65536:
            return False, "Polynomial degree must be in range [1024, 65536]"

        # Validate coefficient modulus
        if not coeff_modulus_bits:
            return False, "Coefficient modulus cannot be empty"

        if len(coeff_modulus_bits) < 2:
            return False, "Coefficient modulus must have at least 2 primes"

        for i, bits in enumerate(coeff_modulus_bits):
            if bits < 30 or bits > 60:
                return False, f"Coefficient modulus bits at index {i} must be in range [30, 60]"

        # Check total modulus size for security
        total_bits = sum(coeff_modulus_bits)
        if security_level == 128 and total_bits > 438:
            return False, f"Total coefficient modulus {total_bits} bits exceeds 128-bit security limit (438 bits)"

        if security_level == 192 and total_bits > 305:
            return False, f"Total coefficient modulus {total_bits} bits exceeds 192-bit security limit (305 bits)"

        # Validate scale
        if scale <= 0:
            return False, "Scale must be positive"

        if scale > 2**60:
            return False, "Scale exceeds maximum value (2^60)"

        # Security level validation
        if security_level not in [128, 192, 256]:
            return False, "Security level must be 128, 192, or 256"

        return True, None

class SecurityValidator:
    """Security-focused validation"""

    @staticmethod
    def validate_input_size(data_size: int, max_size: int = 100 * 1024 * 1024) -> Tuple[bool, Optional[str]]:
        """Validate input data size to prevent DoS attacks"""
        if data_size > max_size:
            return False, f"Input size {data_size} exceeds maximum allowed {max_size} bytes"

        return True, None

    @staticmethod
    def validate_api_key(api_key: str) -> Tuple[bool, Optional[str]]:
        """Validate API key format and strength"""
        if not api_key:
            return False, "API key cannot be empty"

        if len(api_key) < 8:
            return False, "API key must be at least 8 characters"

        # In production, implement proper key validation
        # Check against database, validate JWT, etc.

        return True, None

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""

        import re
        import os

        # Remove path separators and special characters
        sanitized = re.sub(r'[^\w\-_\.]', '', filename)

        # Remove leading dots and path components
        sanitized = os.path.basename(sanitized)

        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]

        return sanitized

    @staticmethod
    def validate_tensor_memory(tensor_shape: List[int], dtype_size: int = 4) -> Tuple[bool, Optional[str]]:
        """Validate tensor memory requirements"""
        if not tensor_shape:
            return False, "Tensor shape cannot be empty"

        # Calculate memory requirement
        total_elements = 1
        for dim in tensor_shape:
            if dim <= 0:
                return False, f"Tensor dimension {dim} must be positive"
            total_elements *= dim

        memory_bytes = total_elements * dtype_size
        max_memory = 8 * 1024 * 1024 * 1024  # 8GB limit

        if memory_bytes > max_memory:
            return False, f"Tensor requires {memory_bytes} bytes, exceeds limit {max_memory}"

        return True, None
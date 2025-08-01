# Architecture Documentation

## System Overview

HE-Graph-Embeddings implements a layered architecture that combines CKKS homomorphic encryption with graph neural networks for privacy-preserving graph computation.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  Python API  │  Model Training  │  Inference Pipeline           │
├─────────────────────────────────────────────────────────────────┤
│                    HE-GNN Models                                │
├─────────────────────────────────────────────────────────────────┤
│  HEGraphSAGE │     HEGAT        │    HEGraphConv               │
├─────────────────────────────────────────────────────────────────┤
│               Homomorphic Operations Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  Encrypted   │  Message         │  Attention     │  Activation   │
│  Linear      │  Passing         │  Mechanisms    │  Functions    │
├─────────────────────────────────────────────────────────────────┤
│                    CKKS Engine                                  │
├─────────────────────────────────────────────────────────────────┤
│  Encryption  │  Arithmetic      │  Bootstrap     │  Modulus      │
│  Context     │  Operations      │  Operations    │  Switching    │
├─────────────────────────────────────────────────────────────────┤
│                    CUDA Kernels                                 │
├─────────────────────────────────────────────────────────────────┤
│  NTT/INTT    │  Polynomial      │  Memory        │  Graph        │
│  Operations  │  Arithmetic      │  Management    │  Operations   │
├─────────────────────────────────────────────────────────────────┤
│                   Hardware Layer                                │
│                     (NVIDIA GPU)                                │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. CKKS Engine

**Purpose**: Core homomorphic encryption functionality
**Technologies**: SEAL/PALISADE-inspired CKKS implementation
**Key Components**:
- Context management with security parameter configuration
- Ciphertext and plaintext handling
- Key generation and management
- Noise budget tracking

### 2. CUDA Kernels

**Purpose**: GPU-accelerated homomorphic operations
**Technologies**: CUDA 12.0+, custom kernel implementations
**Key Components**:
- NTT (Number Theoretic Transform) operations
- Polynomial arithmetic in encrypted domain
- Memory coalescing for ciphertext operations
- Graph-specific scatter/gather operations

### 3. Homomorphic Operations Layer

**Purpose**: Bridge between CKKS primitives and GNN operations
**Key Components**:
- Encrypted matrix multiplication
- Homomorphic aggregation functions
- Polynomial approximations for non-linear functions
- Ciphertext packing/unpacking strategies

### 4. HE-GNN Models

**Purpose**: Graph neural network models operating on encrypted data
**Supported Models**:
- **HEGraphSAGE**: Sample and aggregate with encrypted operations
- **HEGAT**: Graph attention with polynomial softmax approximation
- **HEGraphConv**: Basic graph convolution in encrypted space

## Data Flow

### Encryption Pipeline
```
Plaintext Graph → Feature Encoding → CKKS Encryption → Packed Ciphertexts
```

### Forward Pass Pipeline
```
Encrypted Features → Message Passing → Linear Transform → Activation → Output
```

### Multi-GPU Distribution
```
Graph Partitioning → Context Distribution → Parallel Processing → Result Aggregation
```

## Security Architecture

### Threat Model
- **Honest-but-curious server**: Server executes computations correctly but may try to learn from intermediate values
- **Data never decrypted**: All operations performed in encrypted space
- **Key management**: Private keys never leave client environment

### Security Parameters
- **Polynomial degree**: 2^15 (32768) for 128-bit security
- **Coefficient modulus**: Carefully chosen prime chain
- **Scale**: Balances precision vs. noise growth
- **Bootstrap threshold**: Automatic noise management

## Performance Characteristics

### Computational Complexity
- **Encryption**: O(n log n) per feature vector
- **Homomorphic operations**: O(d²) where d is multiplicative depth  
- **Graph operations**: O(|E| + |V|) encrypted complexity
- **Communication**: Minimal - computation happens on encrypted data

### Memory Architecture
- **GPU memory pools**: Pre-allocated ciphertext buffers
- **Streaming**: Large graphs processed in batches
- **Cache optimization**: NTT table caching for repeated operations

## Scalability Design

### Horizontal Scaling
- **Multi-GPU**: Graph partitioning across multiple GPUs
- **Distributed**: MPI-style communication for large clusters
- **Load balancing**: Dynamic work distribution based on GPU capacity

### Vertical Scaling
- **Memory optimization**: Gradient checkpointing for large models
- **Precision scaling**: Adaptive precision based on noise budget
- **Batch processing**: Efficient ciphertext packing strategies

## Extension Points

### Custom Models
- **HEModule base class**: Framework for implementing new encrypted layers
- **Kernel interface**: CUDA kernel registration system
- **Parameter sharing**: Encrypted parameter management

### Integration Points
- **Federated learning**: Multi-party computation extensions
- **Other HE schemes**: Pluggable encryption backend
- **Hardware acceleration**: TPU/other accelerator support

## Dependencies

### Core Dependencies
- **CUDA Runtime**: 12.0+
- **cuDNN**: For optimized operations
- **PyTorch**: 2.0+ with CUDA support
- **Python**: 3.9+

### Optional Dependencies
- **NetworkX**: Graph utilities and examples
- **DGL/PyG**: Graph data loading compatibility
- **Weights & Biases**: Experiment tracking
- **Docker**: Containerized deployment

## Configuration Management

### Encryption Parameters
```python
{
    "poly_modulus_degree": 32768,
    "coeff_modulus_bits": [60, 40, 40, 40, 40, 60],
    "scale": 2**40,
    "security_level": 128
}
```

### Performance Tuning
```python
{
    "gpu_memory_pool_gb": 40,
    "batch_size": 1024,
    "enable_ntt_cache": True,
    "bootstrap_threshold": 10
}
```

This architecture enables secure, scalable graph neural network computation while maintaining strong cryptographic guarantees through homomorphic encryption.
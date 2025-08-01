# ADR-001: CKKS Homomorphic Encryption Scheme Selection

## Status
Accepted

## Context
The project requires homomorphic encryption to enable privacy-preserving graph neural network computation. Multiple homomorphic encryption schemes exist with different trade-offs:

1. **BGV**: Good for integer arithmetic, limited depth
2. **BFV**: Similar to BGV, better for batching
3. **CKKS**: Supports approximate arithmetic on real/complex numbers
4. **TFHE**: Fast bootstrapping, limited to Boolean circuits
5. **GSW**: Compact ciphertexts, expensive operations

Graph neural networks require:
- Floating-point arithmetic for features and weights
- Matrix multiplications and linear transformations
- Non-linear activation functions (approximated)
- Aggregation operations (sum, mean, max)
- Reasonable computational overhead

## Decision
We choose **CKKS (Cheon-Kim-Kim-Song)** homomorphic encryption scheme.

## Rationale

### Advantages of CKKS:
1. **Native floating-point support**: GNNs work with real-valued features and weights
2. **SIMD operations**: Single Instruction, Multiple Data through polynomial packing
3. **Approximate computation**: Acceptable for machine learning applications
4. **Established security**: Well-studied scheme with known security parameters
5. **GPU optimization**: NTT operations map well to parallel computation

### Implementation Parameters:
- **Polynomial degree**: 2^15 (32,768) for 128-bit security
- **Coefficient modulus**: 438-bit chain for 10+ multiplication depth
- **Scale**: 2^40 for 40-bit precision
- **Bootstrap**: HEaaN-style optimized bootstrapping when needed

### Trade-offs Accepted:
- **Approximate results**: Small numerical errors from encryption noise
- **Computational overhead**: 50-100x slowdown vs. plaintext operations
- **Memory requirements**: Larger ciphertext sizes vs. plaintext
- **Depth limitations**: Requires noise management and potential bootstrapping

## Alternatives Considered

### BGV/BFV
- **Rejected**: Requires integer quantization, loses ML precision
- **Impact**: Would need complex fixed-point arithmetic simulation

### TFHE
- **Rejected**: Boolean gates don't efficiently implement ML operations
- **Impact**: Prohibitive computational cost for GNN forward passes

### Plaintext with Secure Aggregation
- **Rejected**: Doesn't protect intermediate computations
- **Impact**: Server could infer sensitive patterns from partial results

## Consequences

### Positive:
- Enables truly private graph neural network inference
- Maintains reasonable computational performance for practical use
- Leverages existing CUDA optimization techniques
- Compatible with standard ML training pipelines

### Negative:
- Requires careful parameter tuning for security vs. performance
- Limits model depth due to noise accumulation
- Introduces numerical approximation errors
- Requires specialized cryptographic expertise

### Mitigations:
- Automated parameter selection based on model requirements
- Noise budget tracking and automatic bootstrapping
- Comprehensive testing against plaintext baselines
- Clear documentation of precision limitations

## Implementation Notes
- Use Microsoft SEAL-style API for familiarity
- Implement custom CUDA kernels for critical path operations
- Provide Python bindings with PyTorch integration
- Include security parameter estimation tools

## Related ADRs
- ADR-002: CUDA Kernel Optimization Strategy
- ADR-003: Graph Partitioning for Multi-GPU Scaling
- ADR-004: Activation Function Approximation Methods
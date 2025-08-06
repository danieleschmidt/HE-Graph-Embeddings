# HE-Graph-Embeddings User Guide

## Introduction

HE-Graph-Embeddings is a privacy-preserving graph neural network platform that enables machine learning on sensitive graph data without compromising privacy. Using homomorphic encryption, your data remains encrypted throughout the entire computation process.

## Getting Started

### Installation

#### Prerequisites

- Python 3.9 or higher
- 16GB RAM minimum (32GB recommended for large graphs)
- CUDA-compatible GPU (optional but recommended)
- OpenMP support for parallel processing

#### Installing from PyPI

```bash
pip install he-graph-embeddings
```

#### Installing from Source

```bash
git clone https://github.com/terragon/he-graph-embeddings.git
cd he-graph-embeddings
pip install -e .
```

#### Docker Installation

```bash
docker pull terragon/he-graph-embeddings:latest
docker run -it --gpus all terragon/he-graph-embeddings:latest
```

### Quick Start Example

```python
from he_graph import CKKSContext, HEGraphSAGE, HEConfig
import numpy as np

# Create graph data
features = np.random.randn(100, 64).astype(np.float32)
edge_index = np.random.randint(0, 100, size=(2, 500))

# Setup homomorphic encryption
config = HEConfig()
context = CKKSContext(config)
context.generate_keys()

# Create and use GraphSAGE model
model = HEGraphSAGE(
    in_channels=64,
    hidden_channels=32,
    out_channels=16,
    context=context
)

# Encrypt, process, and decrypt
encrypted_features = context.encrypt(features)
encrypted_output = model(encrypted_features, edge_index)
output = context.decrypt(encrypted_output)

print(f"Input shape: {features.shape}")
print(f"Output shape: {output.shape}")
```

## Core Concepts

### Homomorphic Encryption

**What is Homomorphic Encryption?**
Homomorphic encryption allows computations to be performed directly on encrypted data without decrypting it first. The results, when decrypted, match those obtained from operations on the original unencrypted data.

**Why CKKS?**
The CKKS (Cheon-Kim-Kim-Song) scheme is particularly suited for machine learning applications because:
- Supports approximate arithmetic on real numbers
- Enables SIMD (Single Instruction, Multiple Data) operations
- Provides configurable precision and performance trade-offs

### Graph Neural Networks

**GraphSAGE (Graph Sample and Aggregate)**
- Inductive learning on graphs
- Samples and aggregates information from node neighborhoods
- Scalable to large graphs through sampling

**Graph Attention Networks (GAT)**
- Uses attention mechanisms to weight neighbor contributions
- Learns different importance weights for different neighbors
- Multi-head attention for capturing diverse relationships

### Noise Budget

**Understanding Noise Budget**
Homomorphic encryption introduces controlled noise to maintain security. Each operation consumes some noise budget:
- **Addition**: Minimal noise consumption (~0.1 bits)
- **Multiplication**: Moderate noise consumption (~20-40 bits)
- **Activation Functions**: Higher noise consumption (~15-25 bits)

**Managing Noise Budget**
```python
from he_graph import NoiseTracker

with NoiseTracker() as tracker:
    # Perform encrypted operations
    encrypted_result = model(encrypted_data, edge_index)
    
    print(f"Remaining noise budget: {tracker.get_noise_budget()} bits")
    
    if tracker.get_noise_budget() < 10:
        print("Warning: Low noise budget!")
```

## Configuration Guide

### HEConfig Parameters

```python
from he_graph import HEConfig

# Conservative configuration (high security, slower)
conservative_config = HEConfig(
    poly_modulus_degree=32768,
    coeff_modulus_bits=[60, 40, 40, 40, 40, 40, 40, 60],
    scale=2**40,
    precision_bits=35
)

# Balanced configuration (default)
balanced_config = HEConfig(
    poly_modulus_degree=16384,
    coeff_modulus_bits=[60, 40, 40, 40, 60],
    scale=2**40,
    precision_bits=30
)

# Performance configuration (lower security, faster)
performance_config = HEConfig(
    poly_modulus_degree=8192,
    coeff_modulus_bits=[60, 40, 60],
    scale=2**35,
    precision_bits=25
)
```

### Parameter Guidelines

| Parameter | Small Graphs | Medium Graphs | Large Graphs |
|-----------|--------------|---------------|--------------|
| `poly_modulus_degree` | 8192 | 16384 | 32768 |
| `coeff_modulus_bits` | [60, 40, 60] | [60, 40, 40, 40, 60] | [60, 40, 40, 40, 40, 40, 40, 60] |
| `scale` | 2^35 | 2^40 | 2^40 |
| `precision_bits` | 25 | 30 | 35 |

**Graph Size Guidelines**:
- **Small**: <1,000 nodes, <10,000 edges
- **Medium**: 1,000-10,000 nodes, 10,000-100,000 edges  
- **Large**: >10,000 nodes, >100,000 edges

## Working with Graph Data

### Data Format Requirements

**Node Features**
```python
# Features should be a 2D numpy array
# Shape: (num_nodes, feature_dimension)
features = np.array([
    [0.1, 0.2, 0.3],  # Node 0 features
    [0.4, 0.5, 0.6],  # Node 1 features
    [0.7, 0.8, 0.9],  # Node 2 features
], dtype=np.float32)
```

**Edge Index**
```python
# Edge index should be a 2D numpy array
# Shape: (2, num_edges)
# First row: source nodes, Second row: target nodes
edge_index = np.array([
    [0, 1, 2],  # Source nodes
    [1, 2, 0],  # Target nodes
], dtype=np.int64)
```

**Edge Attributes (Optional)**
```python
# Edge attributes should be a 2D numpy array
# Shape: (num_edges, edge_feature_dimension)
edge_attr = np.array([
    [0.1, 0.2],  # Edge 0->1 attributes
    [0.3, 0.4],  # Edge 1->2 attributes
    [0.5, 0.6],  # Edge 2->0 attributes
], dtype=np.float32)
```

### Loading Common Graph Formats

#### NetworkX Graphs

```python
import networkx as nx
from he_graph.utils import networkx_to_hegraph

# Create or load NetworkX graph
G = nx.karate_club_graph()

# Add node features
for node in G.nodes():
    G.nodes[node]['features'] = np.random.randn(16)

# Convert to HE-Graph format
edge_index, features = networkx_to_hegraph(G)
```

#### PyTorch Geometric Data

```python
from torch_geometric.datasets import Cora
from he_graph.utils import pyg_to_hegraph

# Load PyTorch Geometric dataset
dataset = Cora(root='/tmp/Cora')
data = dataset[0]

# Convert to HE-Graph format
edge_index, features = pyg_to_hegraph(data)
```

#### CSV Files

```python
import pandas as pd
from he_graph.utils import csv_to_hegraph

# Load edges from CSV
edges_df = pd.read_csv('edges.csv')  # Columns: source, target
features_df = pd.read_csv('features.csv')  # Columns: node_id, feature1, feature2, ...

# Convert to HE-Graph format
edge_index, features = csv_to_hegraph(edges_df, features_df)
```

## Model Usage

### GraphSAGE

```python
from he_graph import HEGraphSAGE

# Create model
model = HEGraphSAGE(
    in_channels=64,           # Input feature dimension
    hidden_channels=[32, 16], # Hidden layer dimensions
    out_channels=8,           # Output dimension
    num_layers=3,             # Number of GraphSAGE layers
    aggr='mean',             # Aggregation function: 'mean', 'max', 'add'
    context=context          # CKKS context
)

# Process encrypted data
encrypted_features = context.encrypt(features)
encrypted_output = model(encrypted_features, edge_index)
output = context.decrypt(encrypted_output)
```

### Graph Attention Networks (GAT)

```python
from he_graph import HEGAT

# Create GAT model
model = HEGAT(
    in_channels=64,      # Input feature dimension
    out_channels=32,     # Output dimension per head
    heads=4,             # Number of attention heads
    attention_type='additive',  # 'additive' or 'multiplicative'
    dropout=0.1,         # Dropout rate (applied before encryption)
    edge_dim=2,          # Edge feature dimension (optional)
    context=context      # CKKS context
)

# Process with edge attributes
encrypted_features = context.encrypt(features)
encrypted_edge_attr = context.encrypt(edge_attr)
encrypted_output = model(
    encrypted_features, 
    edge_index, 
    edge_attr=encrypted_edge_attr
)
output = context.decrypt(encrypted_output)
```

### Custom Models

```python
from he_graph import HELayer

class CustomModel:
    def __init__(self, context):
        self.context = context
        self.layer1 = HEGraphSAGE(64, 32, 16, context=context)
        self.layer2 = HEGAT(16, 8, heads=2, context=context)
    
    def __call__(self, x_enc, edge_index):
        # First GraphSAGE layer
        x_enc = self.layer1(x_enc, edge_index)
        
        # Apply activation function
        x_enc = self.context.approximate_activation(x_enc, 'relu')
        
        # Second GAT layer
        x_enc = self.layer2(x_enc, edge_index)
        
        return x_enc

# Use custom model
model = CustomModel(context)
encrypted_output = model(encrypted_features, edge_index)
```

## Advanced Features

### Batch Processing

```python
from he_graph import BatchProcessor

# Process multiple graphs efficiently
batch_processor = BatchProcessor(context, max_batch_size=10)

graphs = [
    (features1, edge_index1),
    (features2, edge_index2),
    (features3, edge_index3),
]

model = HEGraphSAGE(64, 32, 16, context=context)

# Process all graphs
results = batch_processor.process_batch(model, graphs)
```

### Concurrent Processing

```python
from he_graph import ConcurrentProcessor
import threading

# Setup concurrent processing
processor = ConcurrentProcessor(
    num_workers=4,
    context=context
)

# Process graphs concurrently
def process_graph(graph_data):
    features, edge_index = graph_data
    encrypted_features = context.encrypt(features)
    encrypted_output = model(encrypted_features, edge_index)
    return context.decrypt(encrypted_output)

# Submit tasks
futures = []
for graph in graphs:
    future = processor.submit(process_graph, graph)
    futures.append(future)

# Collect results
results = [future.result() for future in futures]
```

### Performance Optimization

#### GPU Acceleration

```python
# Enable GPU acceleration
config = HEConfig(use_gpu=True, gpu_device=0)
context = CKKSContext(config)

# Check GPU availability
if context.is_gpu_available():
    print("GPU acceleration enabled")
else:
    print("Falling back to CPU")
```

#### Memory Management

```python
from he_graph import MemoryManager

# Enable automatic memory management
with MemoryManager(context) as mm:
    # Perform operations
    encrypted_features = context.encrypt(features)
    encrypted_output = model(encrypted_features, edge_index)
    
    # Memory automatically freed at context exit
```

#### Noise Budget Optimization

```python
from he_graph import NoiseOptimizer

optimizer = NoiseOptimizer(context)

# Optimize operations order for better noise budget usage
operations = [
    ('encrypt', features),
    ('graphsage_layer', edge_index),
    ('activation', 'relu'),
    ('gat_layer', edge_index),
    ('decrypt', None)
]

# Execute optimized sequence
result = optimizer.execute_optimized(operations)
```

## CLI Usage

### Basic Commands

```bash
# Encrypt a graph dataset
he-graph encrypt --input data.npz --config config.json --output encrypted.he

# Process encrypted graph with GraphSAGE
he-graph process --model graphsage --input encrypted.he --output results.he

# Decrypt results
he-graph decrypt --input results.he --keys keys.json --output final_results.npz
```

### Configuration File

Create `config.json`:
```json
{
  "encryption": {
    "poly_modulus_degree": 16384,
    "coeff_modulus_bits": [60, 40, 40, 40, 60],
    "scale": 1099511627776,
    "precision_bits": 30
  },
  "model": {
    "type": "graphsage",
    "in_channels": 64,
    "hidden_channels": [32, 16],
    "out_channels": 8,
    "num_layers": 3
  },
  "processing": {
    "use_gpu": true,
    "batch_size": 32,
    "num_workers": 4
  }
}
```

### Batch Processing CLI

```bash
# Process multiple graphs
he-graph batch-process \
  --input-dir /path/to/graphs/ \
  --config config.json \
  --output-dir /path/to/results/ \
  --parallel 4
```

## Troubleshooting

### Common Issues

#### Noise Budget Exhausted

**Problem**: `NoiseExhaustionError: Insufficient noise budget for operation`

**Solution**:
```python
# Increase initial noise budget
config = HEConfig(
    poly_modulus_degree=32768,  # Larger degree
    coeff_modulus_bits=[60, 40, 40, 40, 40, 40, 40, 60]  # More primes
)

# Or reduce operation complexity
model = HEGraphSAGE(
    in_channels=64,
    hidden_channels=16,  # Smaller hidden dimension
    out_channels=8,
    num_layers=2         # Fewer layers
)
```

#### Memory Errors

**Problem**: `OutOfMemoryError: Cannot allocate tensor`

**Solution**:
```python
# Reduce batch size
config = HEConfig(batch_size=8)  # Smaller batches

# Use memory-efficient processing
from he_graph import MemoryEfficientProcessor
processor = MemoryEfficientProcessor(context)
```

#### Slow Performance

**Problem**: Processing takes too long

**Solutions**:
```python
# Enable GPU acceleration
config = HEConfig(use_gpu=True)

# Use lower precision
config = HEConfig(precision_bits=25)

# Reduce polynomial degree for small graphs
config = HEConfig(poly_modulus_degree=8192)

# Enable parallel processing
from he_graph import ParallelProcessor
processor = ParallelProcessor(num_workers=8)
```

### Debugging Tools

#### Noise Budget Monitoring

```python
from he_graph import DebugMode

with DebugMode(context) as debug:
    encrypted_features = context.encrypt(features)
    debug.log_noise_budget("After encryption")
    
    encrypted_output = model(encrypted_features, edge_index)
    debug.log_noise_budget("After model")
    
    output = context.decrypt(encrypted_output)
    debug.log_noise_budget("After decryption")
```

#### Performance Profiling

```python
from he_graph import Profiler

with Profiler() as prof:
    # Your code here
    encrypted_output = model(encrypted_features, edge_index)

# Print profiling results
prof.print_stats()
```

#### Verification Mode

```python
# Enable verification against plaintext computation
context = CKKSContext(config, verify_mode=True)

# Operations will be verified against plaintext equivalents
encrypted_output = model(encrypted_features, edge_index)
output = context.decrypt(encrypted_output)

# Verification results logged automatically
```

## Best Practices

### Security

1. **Never log encryption keys or intermediate results**
2. **Use fresh contexts for different datasets**
3. **Monitor noise budget throughout computation**
4. **Rotate encryption keys regularly**
5. **Validate input data before encryption**

### Performance

1. **Choose appropriate parameters for your graph size**
2. **Use GPU acceleration when available**
3. **Process graphs in batches when possible**
4. **Monitor memory usage and optimize accordingly**
5. **Profile code to identify bottlenecks**

### Development

1. **Start with small graphs for testing**
2. **Use verification mode during development**
3. **Implement proper error handling**
4. **Add comprehensive logging**
5. **Write unit tests for custom components**

## API Reference

For detailed API documentation, see [API_REFERENCE.md](API_REFERENCE.md).

## Examples

Additional examples are available in the `examples/` directory:

- `basic_usage.py`: Simple GraphSAGE example
- `advanced_usage.py`: Complex multi-model pipeline
- `batch_processing.py`: Processing multiple graphs
- `custom_models.py`: Building custom architectures
- `performance_tuning.py`: Optimization techniques

## Support

- **Documentation**: https://docs.he-graph.terragon.ai
- **Community Forum**: https://community.terragon.ai  
- **Issues**: https://github.com/terragon/he-graph-embeddings/issues
- **Email**: support@terragon.ai

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
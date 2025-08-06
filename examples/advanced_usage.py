#!/usr/bin/env python3
"""
Advanced Usage Example for HE-Graph-Embeddings
Demonstrates Graph Attention Networks, batch processing, and performance optimization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from python.he_graph import (
    CKKSContext, HEConfig, HEGAT, HEGraphSAGE,
    SecurityEstimator, NoiseTracker
)

def create_citation_network(num_papers=200, num_features=128):
    """Create a synthetic citation network graph"""
    print(f"Creating citation network: {num_papers} papers, {num_features} features")
    
    # Create citation edges (roughly following a scale-free pattern)
    edges = []
    
    # Each paper cites some earlier papers
    for i in range(1, num_papers):
        # Number of citations follows a power law (simplified)
        num_citations = min(np.random.poisson(3) + 1, i)
        
        # Select papers to cite (bias towards recent papers)
        probabilities = np.array([1.0 / (i - j + 1) for j in range(i)])
        probabilities = probabilities / probabilities.sum()
        
        cited_papers = np.random.choice(i, size=num_citations, replace=False, p=probabilities)
        
        for cited in cited_papers:
            edges.append([cited, i])  # cited -> citing
            edges.append([i, cited])  # Make undirected
    
    edge_index = np.array(edges).T
    
    # Generate features representing paper topics
    # Use a mixture of topics
    num_topics = 8
    topic_weights = np.random.dirichlet(np.ones(num_topics) * 0.5, size=num_papers)
    
    # Each topic has characteristic features
    topic_vectors = np.random.randn(num_topics, num_features)
    
    features = np.zeros((num_papers, num_features))
    for i in range(num_papers):
        for j in range(num_topics):
            features[i] += topic_weights[i, j] * topic_vectors[j]
    
    features = features.astype(np.float32)
    
    print(f"   📊 Generated {len(edges)} edges ({len(set(tuple(e) for e in edges))} unique)")
    
    return edge_index, features

def demonstrate_gat_attention():
    """Demonstrate Graph Attention Networks with homomorphic encryption"""
    print("\n🎯 Graph Attention Network (GAT) Example")
    print("-" * 45)
    
    # Create test data
    edge_index, features = create_citation_network(num_papers=50, num_features=64)
    num_nodes, feature_dim = features.shape
    
    # Setup encryption with higher precision for attention
    config = HEConfig(
        poly_modulus_degree=32768,  # Higher for attention computations
        coeff_modulus_bits=[60, 40, 40, 40, 40, 40, 60],
        scale=2**40,
        precision_bits=35
    )
    
    context = CKKSContext(config)
    context.generate_keys()
    
    # Create GAT model
    gat_model = HEGAT(
        in_channels=feature_dim,
        out_channels=32,
        heads=4,  # Multi-head attention
        attention_type='additive',  # More HE-friendly
        edge_dim=None,
        context=context
    )
    
    print(f"   🧠 GAT Model: {feature_dim} → 32 features, 4 heads")
    
    # Encrypt and process
    print("   🔐 Encrypting features...")
    with NoiseTracker() as tracker:
        enc_features = context.encrypt(features)
        initial_noise = tracker.get_noise_budget()
    
    print("   🔍 Computing attention...")
    start_time = time.time()
    
    with NoiseTracker() as tracker:
        enc_output = gat_model(enc_features, edge_index)
        attention_noise = tracker.get_noise_budget()
    
    attention_time = time.time() - start_time
    
    print(f"   ⚡ Attention computed in {attention_time:.3f}s")
    print(f"   📊 Noise budget: {initial_noise:.1f} → {attention_noise:.1f} bits")
    
    # Decrypt results
    output = context.decrypt(enc_output)
    print(f"   📊 Output shape: {output.shape}")
    print(f"   📈 Attention preserved feature diversity: std={output.std():.3f}")
    
    return attention_time, initial_noise - attention_noise

def demonstrate_batch_processing():
    """Demonstrate batch processing for multiple graphs"""
    print("\n📦 Batch Processing Example")
    print("-" * 30)
    
    # Create multiple small graphs
    batch_size = 5
    graphs = []
    
    for i in range(batch_size):
        edge_index, features = create_citation_network(num_papers=30, num_features=32)
        graphs.append((edge_index, features))
    
    print(f"   📊 Processing batch of {batch_size} graphs")
    
    # Setup encryption
    config = HEConfig(poly_modulus_degree=16384)
    context = CKKSContext(config)
    context.generate_keys()
    
    # Process each graph
    model = HEGraphSAGE(
        in_channels=32,
        hidden_channels=16,
        out_channels=8,
        context=context
    )
    
    batch_times = []
    total_start = time.time()
    
    for i, (edge_index, features) in enumerate(graphs):
        print(f"   📝 Processing graph {i+1}/{batch_size}...")
        
        start_time = time.time()
        
        # Encrypt, process, decrypt
        enc_features = context.encrypt(features)
        enc_output = model(enc_features, edge_index)
        output = context.decrypt(enc_output)
        
        graph_time = time.time() - start_time
        batch_times.append(graph_time)
        
        print(f"      ⚡ Graph {i+1}: {graph_time:.3f}s, output: {output.shape}")
    
    total_time = time.time() - total_start
    avg_time = np.mean(batch_times)
    
    print(f"   📈 Batch summary:")
    print(f"      Total time: {total_time:.3f}s")
    print(f"      Average per graph: {avg_time:.3f}s")
    print(f"      Throughput: {batch_size/total_time:.2f} graphs/second")
    
    return total_time, avg_time

def demonstrate_noise_management():
    """Demonstrate noise budget management and optimization"""
    print("\n🔧 Noise Management Example")
    print("-" * 32)
    
    edge_index, features = create_citation_network(num_papers=40, num_features=48)
    
    # Test different configurations
    configs = [
        {"name": "Conservative", "poly": 32768, "bits": [60, 40, 40, 40, 40, 40, 60]},
        {"name": "Aggressive", "poly": 16384, "bits": [60, 40, 40, 60]},
        {"name": "Deep", "poly": 32768, "bits": [60, 40, 40, 40, 40, 40, 40, 40, 60]}
    ]
    
    for config_def in configs:
        print(f"\n   🧪 Testing {config_def['name']} configuration:")
        
        config = HEConfig(
            poly_modulus_degree=config_def['poly'],
            coeff_modulus_bits=config_def['bits'],
            scale=2**40
        )
        
        context = CKKSContext(config)
        context.generate_keys()
        
        # Create deeper model to consume more noise
        model = HEGraphSAGE(
            in_channels=48,
            hidden_channels=[32, 24, 16],  # 3 layers
            out_channels=8,
            context=context
        )
        
        # Track noise throughout computation
        noise_history = []
        
        with NoiseTracker() as tracker:
            enc_features = context.encrypt(features)
            noise_history.append(("Initial", tracker.get_noise_budget()))
            
            # Process through each layer
            x_enc = enc_features
            for i, conv in enumerate(model.convs):
                x_enc = conv(x_enc, edge_index)
                if i < len(model.convs) - 1:  # Apply activation except last layer
                    x_enc = model._apply_activation(x_enc)
                
                tracker.update(x_enc)
                noise_history.append((f"Layer {i+1}", tracker.get_noise_budget()))
        
        print(f"      📊 Noise budget progression:")
        for stage, budget in noise_history:
            status = "✅" if budget > 10 else "⚠️" if budget > 5 else "❌"
            print(f"         {stage:10}: {budget:6.1f} bits {status}")
        
        final_noise = noise_history[-1][1]
        success = final_noise > 5  # Minimum viable noise budget
        
        if success:
            output = context.decrypt(x_enc)
            print(f"      ✅ Success! Output: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
        else:
            print(f"      ❌ Failed! Insufficient noise budget ({final_noise:.1f} bits)")

def demonstrate_performance_optimization():
    """Demonstrate various performance optimization techniques"""
    print("\n⚡ Performance Optimization Example")
    print("-" * 38)
    
    edge_index, features = create_citation_network(num_papers=100, num_features=64)
    
    # Test different optimization strategies
    optimizations = [
        {"name": "Baseline", "poly": 16384, "scale": 2**40, "precision": 30},
        {"name": "High Precision", "poly": 32768, "scale": 2**50, "precision": 40},
        {"name": "Fast Scale", "poly": 16384, "scale": 2**30, "precision": 25},
    ]
    
    results = []
    
    for opt in optimizations:
        print(f"\n   🔬 Testing {opt['name']}:")
        
        config = HEConfig(
            poly_modulus_degree=opt['poly'],
            scale=opt['scale'],
            precision_bits=opt['precision']
        )
        
        context = CKKSContext(config)
        context.generate_keys()
        
        model = HEGraphSAGE(
            in_channels=64,
            hidden_channels=32,
            out_channels=16,
            context=context
        )
        
        # Time the full pipeline
        start_time = time.time()
        
        enc_features = context.encrypt(features)
        enc_output = model(enc_features, edge_index)
        output = context.decrypt(enc_output)
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        accuracy_proxy = output.std()  # Higher std suggests better feature preservation
        
        result = {
            'name': opt['name'],
            'time': total_time,
            'accuracy_proxy': accuracy_proxy,
            'efficiency': accuracy_proxy / total_time  # Quality per second
        }
        results.append(result)
        
        print(f"      ⏱️  Total time: {total_time:.3f}s")
        print(f"      📊 Feature preservation: {accuracy_proxy:.4f}")
        print(f"      ⚡ Efficiency: {result['efficiency']:.4f}")
    
    # Summary
    print(f"\n   📈 Optimization Summary:")
    best_time = min(results, key=lambda x: x['time'])
    best_accuracy = max(results, key=lambda x: x['accuracy_proxy'])
    best_efficiency = max(results, key=lambda x: x['efficiency'])
    
    print(f"      🏃 Fastest: {best_time['name']} ({best_time['time']:.3f}s)")
    print(f"      🎯 Most accurate: {best_accuracy['name']} ({best_accuracy['accuracy_proxy']:.4f})")
    print(f"      ⚡ Most efficient: {best_efficiency['name']} ({best_efficiency['efficiency']:.4f})")

def main():
    print("🚀 HE-Graph-Embeddings Advanced Usage Example")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        gat_time, gat_noise = demonstrate_gat_attention()
        batch_total, batch_avg = demonstrate_batch_processing()
        demonstrate_noise_management()
        demonstrate_performance_optimization()
        
        # Final summary
        print("\n🎉 Advanced Example Completed Successfully!")
        print("\n📋 Performance Summary:")
        print(f"   • GAT forward pass: {gat_time:.3f}s (noise used: {gat_noise:.1f} bits)")
        print(f"   • Batch processing: {batch_avg:.3f}s per graph")
        print(f"   • Noise management: Demonstrated across multiple configurations")
        print(f"   • Optimization strategies: Compared time vs. accuracy tradeoffs")
        
        return {
            'success': True,
            'gat_time': gat_time,
            'batch_avg': batch_avg,
            'demonstrations': ['GAT', 'Batch Processing', 'Noise Management', 'Optimization']
        }
        
    except Exception as e:
        print(f"\n❌ Advanced example failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    result = main()
    if result['success']:
        print(f"\n✅ All advanced features demonstrated successfully")
    else:
        print(f"\n❌ Advanced example failed with error")
        sys.exit(1)
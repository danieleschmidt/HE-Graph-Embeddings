#!/usr/bin/env python3
"""
HE-Graph-Embeddings Command Line Interface
Provides easy access to homomorphic graph neural network operations
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Import our HE graph library
try:
    from ..python.he_graph import (
        CKKSContext, HEConfig, HEGraphSAGE, HEGAT, 
        SecurityEstimator, NoiseTracker
    )
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from python.he_graph import (
        CKKSContext, HEConfig, HEGraphSAGE, HEGAT, 
        SecurityEstimator, NoiseTracker
    )

import numpy as np

def create_sample_graph(num_nodes: int = 34, feature_dim: int = 128) -> tuple:
    """Create a sample graph for testing (Karate Club-style)"""
    print(f"Creating sample graph with {num_nodes} nodes, {feature_dim} features")
    
    # Create simple ring graph + some random connections
    edges = []
    for i in range(num_nodes):
        edges.append([i, (i + 1) % num_nodes])  # Ring
        if i % 3 == 0 and i + 3 < num_nodes:
            edges.append([i, i + 3])  # Some shortcuts
    
    # Convert to numpy array format
    edge_index = np.array(edges).T
    
    # Generate random node features
    features = np.random.randn(num_nodes, feature_dim).astype(np.float32)
    
    print(f"Graph created: {len(edges)} edges, feature shape: {features.shape}")
    return edge_index, features

def benchmark_mode(args):
    """Run performance benchmarks"""
    print("🚀 Running HE-Graph-Embeddings Benchmarks")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        {"name": "Small (128-bit)", "nodes": 100, "features": 32, "security": 128},
        {"name": "Medium (128-bit)", "nodes": 500, "features": 64, "security": 128},
        {"name": "Large (128-bit)", "nodes": 1000, "features": 128, "security": 128},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n📊 Testing {config['name']}:")
        print(f"   Nodes: {config['nodes']}, Features: {config['features']}")
        
        # Create test data
        edge_index, features = create_sample_graph(config['nodes'], config['features'])
        
        # Setup HE context
        he_config = HEConfig(
            poly_modulus_degree=16384 if config['nodes'] < 500 else 32768,
            security_level=config['security']
        )
        
        context = CKKSContext(he_config)
        context.generate_keys()
        
        # Create model
        model = HEGraphSAGE(
            in_channels=config['features'],
            hidden_channels=32,
            out_channels=16,
            num_layers=2,
            context=context
        )
        
        # Benchmark encryption
        start_time = time.time()
        enc_features = context.encrypt(features)
        encrypt_time = time.time() - start_time
        
        # Benchmark forward pass
        start_time = time.time()
        try:
            enc_output = model(enc_features, edge_index)
            forward_time = time.time() - start_time
            success = True
        except Exception as e:
            forward_time = 0
            success = False
            print(f"   ❌ Forward pass failed: {e}")
        
        # Benchmark decryption
        if success:
            start_time = time.time()
            output = context.decrypt(enc_output)
            decrypt_time = time.time() - start_time
        else:
            decrypt_time = 0
        
        result = {
            'config': config['name'],
            'nodes': config['nodes'],
            'features': config['features'],
            'encrypt_time': encrypt_time,
            'forward_time': forward_time,
            'decrypt_time': decrypt_time,
            'total_time': encrypt_time + forward_time + decrypt_time,
            'success': success
        }
        results.append(result)
        
        if success:
            print(f"   ✅ Encryption: {encrypt_time:.3f}s")
            print(f"   ✅ Forward pass: {forward_time:.3f}s") 
            print(f"   ✅ Decryption: {decrypt_time:.3f}s")
            print(f"   🎯 Total time: {result['total_time']:.3f}s")
        
    # Summary
    print(f"\n📈 Benchmark Summary:")
    print("-" * 50)
    for result in results:
        if result['success']:
            print(f"{result['config']:20} | {result['total_time']:8.3f}s | ✅")
        else:
            print(f"{result['config']:20} | {'failed':8} | ❌")
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

def demo_mode(args):
    """Run interactive demonstration"""
    print("🎭 HE-Graph-Embeddings Demo Mode")
    print("=" * 40)
    
    # Create sample data
    print("\n1. Creating sample graph...")
    edge_index, features = create_sample_graph(args.nodes, args.features)
    
    # Setup encryption
    print("\n2. Setting up homomorphic encryption...")
    he_config = HEConfig(
        poly_modulus_degree=16384,
        security_level=128,
        scale=2**40
    )
    
    # Validate security
    estimator = SecurityEstimator()
    params = {
        'poly_degree': he_config.poly_modulus_degree,
        'coeff_modulus_bits': he_config.coeff_modulus_bits,
        'scale': he_config.scale
    }
    security_bits = estimator.estimate(params)
    print(f"   🔒 Estimated security level: {security_bits} bits")
    
    context = CKKSContext(he_config)
    print("   🔑 Generating encryption keys...")
    context.generate_keys()
    print("   ✅ Encryption context ready")
    
    # Create model
    print(f"\n3. Creating {args.model.upper()} model...")
    if args.model == 'graphsage':
        model = HEGraphSAGE(
            in_channels=args.features,
            hidden_channels=64,
            out_channels=32,
            num_layers=2,
            aggregator='mean',
            context=context
        )
    elif args.model == 'gat':
        model = HEGAT(
            in_channels=args.features,
            out_channels=32,
            heads=4,
            context=context
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    print(f"   ✅ {args.model.upper()} model created")
    
    # Encrypt features
    print("\n4. Encrypting node features...")
    with NoiseTracker() as tracker:
        enc_features = context.encrypt(features)
        print(f"   📊 Initial noise budget: {tracker.get_noise_budget():.2f} bits")
    
    # Forward pass
    print("\n5. Running encrypted forward pass...")
    start_time = time.time()
    with NoiseTracker() as tracker:
        enc_output = model(enc_features, edge_index)
        forward_time = time.time() - start_time
        final_noise = tracker.get_noise_budget()
    
    print(f"   ⚡ Forward pass completed in {forward_time:.3f}s")
    print(f"   📊 Final noise budget: {final_noise:.2f} bits")
    
    # Decrypt result
    print("\n6. Decrypting results...")
    output = context.decrypt(enc_output)
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   📈 Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\n🎉 Demo completed successfully!")
    
    if args.save_output:
        output_data = {
            'model_type': args.model,
            'graph_nodes': args.nodes,
            'feature_dim': args.features,
            'output_shape': list(output.shape),
            'output_stats': {
                'mean': float(output.mean()),
                'std': float(output.std()),
                'min': float(output.min()),
                'max': float(output.max())
            },
            'performance': {
                'forward_time': forward_time,
                'final_noise_budget': final_noise
            }
        }
        
        with open(args.save_output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Results saved to {args.save_output}")

def security_mode(args):
    """Analyze security parameters"""
    print("🔒 HE-Graph-Embeddings Security Analysis")
    print("=" * 45)
    
    estimator = SecurityEstimator()
    
    # Test different parameter sets
    test_configs = [
        {"name": "Conservative", "poly": 16384, "coeff_bits": [60, 40, 40, 60]},
        {"name": "Standard", "poly": 32768, "coeff_bits": [60, 40, 40, 40, 40, 60]},
        {"name": "High-Depth", "poly": 32768, "coeff_bits": [60, 40, 40, 40, 40, 40, 40, 60]},
    ]
    
    print("\n📋 Security Level Analysis:")
    print("-" * 60)
    print(f"{'Config':15} {'Poly Degree':12} {'Total Bits':12} {'Security':10}")
    print("-" * 60)
    
    for config in test_configs:
        params = {
            'poly_degree': config['poly'],
            'coeff_modulus_bits': config['coeff_bits']
        }
        
        security_bits = estimator.estimate(params)
        total_bits = sum(config['coeff_bits'])
        
        print(f"{config['name']:15} {config['poly']:12} {total_bits:12} {security_bits:10}")
    
    # Recommendation
    print(f"\n💡 Parameter Recommendations:")
    for depth in [5, 10, 15]:
        try:
            recommended = estimator.recommend(
                security_bits=128,
                multiplicative_depth=depth,
                precision_bits=30
            )
            print(f"   Depth {depth:2d}: poly_degree={recommended.poly_modulus_degree}, "
                  f"modulus_count={len(recommended.coeff_modulus_bits)}")
        except Exception as e:
            print(f"   Depth {depth:2d}: {e}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="HE-Graph-Embeddings: Privacy-preserving graph neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  he-graph demo --nodes 100 --features 64 --model graphsage
  he-graph benchmark --output results.json
  he-graph security
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demonstration')
    demo_parser.add_argument('--nodes', type=int, default=34, 
                           help='Number of nodes in test graph (default: 34)')
    demo_parser.add_argument('--features', type=int, default=128,
                           help='Feature dimension (default: 128)')
    demo_parser.add_argument('--model', choices=['graphsage', 'gat'], default='graphsage',
                           help='Model type to use (default: graphsage)')
    demo_parser.add_argument('--save-output', type=str,
                           help='Save demo results to JSON file')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmarks')
    benchmark_parser.add_argument('--output', '-o', type=str, default='benchmark_results.json',
                                help='Output file for results (default: benchmark_results.json)')
    
    # Security command
    security_parser = subparsers.add_parser('security', help='Analyze security parameters')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'demo':
            demo_mode(args)
        elif args.command == 'benchmark':
            benchmark_mode(args)
        elif args.command == 'security':
            security_mode(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
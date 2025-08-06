#!/usr/bin/env python3
"""
Basic Usage Example for HE-Graph-Embeddings
Demonstrates core functionality with a simple graph
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from python.he_graph import (
    CKKSContext, HEConfig, HEGraphSAGE, HEGAT,
    SecurityEstimator, NoiseTracker
)

def create_karate_club_graph():
    """Create Zachary's Karate Club graph edges"""
    # Simplified version of the famous karate club graph
    edges = [
        [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 10], [0, 11], [0, 12], [0, 13], [0, 17], [0, 19], [0, 21], [0, 31],
        [1, 2], [1, 3], [1, 7], [1, 13], [1, 17], [1, 19], [1, 21], [1, 30],
        [2, 3], [2, 7], [2, 8], [2, 9], [2, 13], [2, 27], [2, 28], [2, 32],
        [3, 7], [3, 12], [3, 13],
        [4, 6], [4, 10],
        [5, 6], [5, 10], [5, 16],
        [6, 16],
        [8, 30], [8, 32], [8, 33],
        [9, 33],
        [13, 33],
        [14, 32], [14, 33],
        [15, 32], [15, 33],
        [18, 32], [18, 33],
        [19, 33],
        [20, 32], [20, 33],
        [22, 32], [22, 33],
        [23, 25], [23, 27], [23, 29], [23, 32], [23, 33],
        [24, 25], [24, 27], [24, 31],
        [25, 31],
        [26, 29], [26, 33],
        [27, 33],
        [28, 31], [28, 33],
        [29, 32], [29, 33],
        [30, 32], [30, 33],
        [31, 32], [31, 33],
        [32, 33]
    ]
    
    return np.array(edges).T

def main():
    print("🎯 HE-Graph-Embeddings Basic Usage Example")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("\n1. Creating Karate Club graph...")
    edge_index = create_karate_club_graph()
    num_nodes = 34
    feature_dim = 64
    
    # Generate random node features
    np.random.seed(42)  # For reproducibility
    node_features = np.random.randn(num_nodes, feature_dim).astype(np.float32)
    
    print(f"   📊 Graph: {num_nodes} nodes, {edge_index.shape[1]} edges")
    print(f"   📊 Features: {node_features.shape}")
    
    # Step 2: Configure homomorphic encryption
    print("\n2. Setting up homomorphic encryption...")
    
    # Create HE configuration
    config = HEConfig(
        poly_modulus_degree=16384,
        coeff_modulus_bits=[60, 40, 40, 40, 60],
        scale=2**40,
        security_level=128,
        precision_bits=30
    )
    
    # Validate security
    estimator = SecurityEstimator()
    params = {
        'poly_degree': config.poly_modulus_degree,
        'coeff_modulus_bits': config.coeff_modulus_bits
    }
    security_bits = estimator.estimate(params)
    print(f"   🔒 Estimated security level: {security_bits} bits")
    
    # Create CKKS context
    context = CKKSContext(config)
    print("   🔑 Generating encryption keys...")
    context.generate_keys()
    print("   ✅ Encryption context ready")
    
    # Step 3: Create and configure GraphSAGE model
    print("\n3. Creating HE-GraphSAGE model...")
    
    model = HEGraphSAGE(
        in_channels=feature_dim,
        hidden_channels=[32, 16],  # Two hidden layers
        out_channels=8,
        num_layers=None,  # Inferred from hidden_channels
        aggregator='mean',
        activation='relu_poly',
        dropout_enc=0.0,  # No dropout for this example
        context=context
    )
    
    print(f"   🧠 Model: {feature_dim} → 32 → 16 → 8")
    print(f"   📐 Layers: {len(model.convs)}")
    
    # Step 4: Encrypt node features
    print("\n4. Encrypting node features...")
    
    with NoiseTracker() as tracker:
        encrypted_features = context.encrypt(node_features)
        initial_noise = tracker.get_noise_budget()
    
    print(f"   🔐 Features encrypted successfully")
    print(f"   📊 Initial noise budget: {initial_noise:.1f} bits")
    
    # Step 5: Run encrypted forward pass
    print("\n5. Running encrypted forward pass...")
    
    import time
    start_time = time.time()
    
    with NoiseTracker() as tracker:
        encrypted_output = model(encrypted_features, edge_index)
        final_noise = tracker.get_noise_budget()
    
    forward_time = time.time() - start_time
    
    print(f"   ⚡ Forward pass completed in {forward_time:.3f} seconds")
    print(f"   📊 Final noise budget: {final_noise:.1f} bits")
    print(f"   🔍 Noise consumed: {initial_noise - final_noise:.1f} bits")
    
    # Step 6: Decrypt and analyze results
    print("\n6. Decrypting results...")
    
    output_embeddings = context.decrypt(encrypted_output)
    
    print(f"   📊 Output shape: {output_embeddings.shape}")
    print(f"   📈 Output range: [{output_embeddings.min():.3f}, {output_embeddings.max():.3f}]")
    print(f"   📊 Output mean: {output_embeddings.mean():.3f}")
    print(f"   📊 Output std: {output_embeddings.std():.3f}")
    
    # Step 7: Demonstrate homomorphic operations
    print("\n7. Demonstrating homomorphic operations...")
    
    # Test homomorphic addition
    enc_features_copy = context.encrypt(node_features * 0.5)
    enc_sum = context.add(encrypted_features, enc_features_copy)
    decrypted_sum = context.decrypt(enc_sum)
    
    expected_sum = node_features + (node_features * 0.5)
    actual_sum = decrypted_sum
    
    error = np.mean(np.abs(expected_sum - actual_sum))
    print(f"   ➕ Addition error: {error:.6f}")
    
    # Test homomorphic multiplication
    enc_double = context.encrypt(node_features[:10, :10])  # Smaller for efficiency
    enc_product = context.multiply(enc_double, enc_double)
    decrypted_product = context.decrypt(enc_product)
    
    expected_product = (node_features[:10, :10]) ** 2
    actual_product = decrypted_product
    
    mult_error = np.mean(np.abs(expected_product - actual_product))
    print(f"   ✖️  Multiplication error: {mult_error:.6f}")
    
    # Step 8: Compare with plaintext computation (for verification)
    print("\n8. Comparing with plaintext computation...")
    
    # Create a simplified plaintext version for comparison
    # Note: This is approximate since we can't exactly replicate HE operations
    print("   📝 This would require plaintext GraphSAGE implementation")
    print("   📝 Encrypted computation preserves privacy throughout")
    
    # Final summary
    print("\n🎉 Example completed successfully!")
    print("\n📋 Summary:")
    print(f"   • Processed {num_nodes} nodes with {feature_dim}-dim features")
    print(f"   • Reduced to {output_embeddings.shape[1]}-dim embeddings") 
    print(f"   • Forward pass time: {forward_time:.3f}s")
    print(f"   • Noise budget used: {initial_noise - final_noise:.1f} bits")
    print(f"   • Security level: {security_bits} bits")
    
    return {
        'success': True,
        'output_shape': output_embeddings.shape,
        'forward_time': forward_time,
        'noise_used': initial_noise - final_noise,
        'security_level': security_bits
    }

if __name__ == '__main__':
    try:
        result = main()
        print(f"\n✅ Example ran successfully: {result}")
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
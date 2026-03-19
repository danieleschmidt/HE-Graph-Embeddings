#!/usr/bin/env python3
"""
HE-GraphSAGE Demo
=================

Demonstrates that GraphSAGE inference on *encrypted* node features produces
embeddings that closely agree with plaintext inference.

Setup
-----
- 10-node random graph, 4-dimensional features
- 2-layer PrivateGraphSAGE (in=4, hidden=8, out=4)
- HomomorphicSimulator with security_param=128

Key result
----------
The server computes graph embeddings without ever seeing plaintext features.
After decryption the encrypted output differs from plaintext only by the
accumulated noise floor (typically < 1e-4 for these parameters).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from src.he_graph.homomorphic_simulator import HomomorphicSimulator
from src.he_graph.private_graphsage import PrivateGraphSAGE, SimpleGraph


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. Build a 10-node random graph
    # ------------------------------------------------------------------
    NUM_NODES = 10
    IN_DIM = 4
    HIDDEN_DIM = 8
    OUT_DIM = 4

    graph = SimpleGraph.random(NUM_NODES, m=15, seed=0)
    print(f"Graph: {NUM_NODES} nodes, edges: {sum(len(v) for v in graph.neighbours.values())//2}")

    # Random node features (simulate private user data)
    x = torch.randn(NUM_NODES, IN_DIM)
    print(f"Node features: shape {tuple(x.shape)}\n")

    # ------------------------------------------------------------------
    # 2. Plaintext inference
    # ------------------------------------------------------------------
    model = PrivateGraphSAGE(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_layers=2)
    model.eval()

    with torch.no_grad():
        plain_out = model.forward(graph, x)

    print("=== Plaintext Inference ===")
    print(f"Output shape: {tuple(plain_out.shape)}")
    print(f"First 3 node embeddings (plaintext):")
    for i in range(3):
        print(f"  Node {i}: {plain_out[i].numpy().round(4)}")

    # ------------------------------------------------------------------
    # 3. Encrypted inference
    # ------------------------------------------------------------------
    sim = HomomorphicSimulator(security_param=128, seed=7)
    print(f"\n{sim}")

    # Client encrypts its features
    print("\n[Client] Encrypting node features...")
    enc_features = {v: sim.encrypt(x[v]) for v in range(NUM_NODES)}
    print(f"  Encrypted {NUM_NODES} feature vectors.")
    print(f"  Server sees: {enc_features[0]}")

    # Server runs GraphSAGE on encrypted data
    print("\n[Server] Running GraphSAGE on encrypted features (no decryption)...")
    with torch.no_grad():
        enc_out = model.forward_encrypted(graph, enc_features)
    print("  Done.")

    # Client decrypts output
    print("\n[Client] Decrypting output embeddings...")
    dec_out = torch.stack([sim.decrypt(enc_out[v]) for v in range(NUM_NODES)])

    print("\n=== Encrypted Inference (after decryption) ===")
    print(f"Output shape: {tuple(dec_out.shape)}")
    print(f"First 3 node embeddings (decrypted):")
    for i in range(3):
        print(f"  Node {i}: {dec_out[i].numpy().round(4)}")

    # ------------------------------------------------------------------
    # 4. Agreement check
    # ------------------------------------------------------------------
    abs_diff = (plain_out - dec_out).abs()
    max_err = abs_diff.max().item()
    mean_err = abs_diff.mean().item()

    print("\n=== Agreement ===")
    print(f"Max absolute error  : {max_err:.6f}")
    print(f"Mean absolute error : {mean_err:.6f}")
    print(f"Noise tolerance     : 1e-2")

    tolerance = 1e-2
    agreed = max_err < tolerance
    print(f"\n✓ Plaintext and encrypted inference agree within {tolerance}: {agreed}")

    if agreed:
        print("\n✅ Demo PASSED — privacy-preserving inference works correctly.")
    else:
        print("\n❌ Demo FAILED — outputs diverge beyond noise tolerance.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Noise budget summary
    # ------------------------------------------------------------------
    print("\n=== Noise Budget (per output node) ===")
    for v in range(min(5, NUM_NODES)):
        budget = sim.noise_budget(enc_out[v])
        noise_sigma = enc_out[v].noise_variance ** 0.5
        print(f"  Node {v}: noise_σ={noise_sigma:.2e}, budget={budget:.1f} bits")


if __name__ == "__main__":
    main()

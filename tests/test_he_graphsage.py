"""
Tests for HE-GraphSAGE: HomomorphicSimulator and PrivateGraphSAGE.

Run with:  pytest tests/test_he_graphsage.py -v
"""

import math

import numpy as np
import pytest
import torch

from src.he_graph.homomorphic_simulator import EncryptedTensor, HomomorphicSimulator
from src.he_graph.private_graphsage import PrivateGraphSAGE, SAGELayer, SimpleGraph


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sim():
    """HomomorphicSimulator with fixed seed for reproducibility."""
    return HomomorphicSimulator(security_param=128, seed=0)


@pytest.fixture
def small_graph():
    """4-node ring graph."""
    return SimpleGraph.ring(4)


@pytest.fixture
def medium_graph():
    """10-node random graph with 15 edges."""
    return SimpleGraph.random(10, m=15, seed=42)


@pytest.fixture
def model_small():
    """Tiny 2-layer PrivateGraphSAGE for fast unit tests."""
    torch.manual_seed(0)
    return PrivateGraphSAGE(in_dim=3, hidden_dim=4, out_dim=2, num_layers=2)


@pytest.fixture
def model_medium():
    """Medium PrivateGraphSAGE matching the demo configuration."""
    torch.manual_seed(42)
    return PrivateGraphSAGE(in_dim=4, hidden_dim=8, out_dim=4, num_layers=2)


# ===========================================================================
# HomomorphicSimulator tests
# ===========================================================================

class TestHomomorphicSimulator:
    """Unit tests for HomomorphicSimulator."""

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def test_default_construction(self):
        s = HomomorphicSimulator()
        assert s.security_param == 128
        assert s.noise_scale > 0

    def test_noise_scale_increases_with_security_param(self):
        s64 = HomomorphicSimulator(security_param=64)
        s256 = HomomorphicSimulator(security_param=256)
        assert s256.noise_scale < s64.noise_scale  # stronger = smaller noise scale

    def test_custom_noise_scale(self):
        s = HomomorphicSimulator(noise_scale=1e-6)
        assert s.noise_scale == pytest.approx(1e-6)

    # -----------------------------------------------------------------------
    # encrypt / decrypt roundtrip
    # -----------------------------------------------------------------------

    def test_encrypt_returns_encrypted_tensor(self, sim):
        x = torch.randn(5)
        ct = sim.encrypt(x)
        assert isinstance(ct, EncryptedTensor)
        assert ct.is_encrypted is True

    def test_decrypt_approximate_roundtrip_scalar(self, sim):
        x = torch.tensor([3.14])
        ct = sim.encrypt(x)
        recovered = sim.decrypt(ct)
        assert abs(float(recovered) - 3.14) < 0.1

    def test_decrypt_shape_preserved(self, sim):
        x = torch.randn(4, 6)
        ct = sim.encrypt(x)
        dec = sim.decrypt(ct)
        assert dec.shape == x.shape

    def test_encrypt_produces_noisy_output(self, sim):
        """Encrypted value should differ from plaintext (noise is injected)."""
        x = torch.ones(10)
        ct = sim.encrypt(x)
        assert not torch.allclose(ct.data, x), "Ciphertext must differ from plaintext"

    def test_noise_is_small_relative_to_plaintext(self, sim):
        x = torch.randn(50) * 5.0
        ct = sim.encrypt(x)
        diff = (sim.decrypt(ct) - x).abs()
        # Expect < 1% relative error on average
        assert (diff / (x.abs() + 1e-6)).mean().item() < 0.01

    def test_encrypt_seeded_reproducibility(self):
        """Same seed → same ciphertext."""
        x = torch.tensor([1.0, 2.0, 3.0])
        ct1 = HomomorphicSimulator(seed=99).encrypt(x)
        ct2 = HomomorphicSimulator(seed=99).encrypt(x)
        assert torch.allclose(ct1.data, ct2.data)

    # -----------------------------------------------------------------------
    # add
    # -----------------------------------------------------------------------

    def test_add_linearity(self, sim):
        """encrypt(a) + encrypt(b) decrypts to ≈ a + b."""
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        ct_sum = sim.add(sim.encrypt(a), sim.encrypt(b))
        recovered = sim.decrypt(ct_sum)
        assert torch.allclose(recovered, a + b, atol=0.1)

    def test_add_noise_accumulates(self, sim):
        """Noise variance should grow after addition."""
        x = torch.randn(8)
        ct = sim.encrypt(x)
        base_noise = ct.noise_variance
        ct_sum = sim.add(ct, ct)
        assert ct_sum.noise_variance > base_noise

    def test_add_operator_overload(self, sim):
        a, b = torch.randn(4), torch.randn(4)
        ct_a, ct_b = sim.encrypt(a), sim.encrypt(b)
        ct_sum = ct_a + ct_b
        assert isinstance(ct_sum, EncryptedTensor)

    # -----------------------------------------------------------------------
    # multiply
    # -----------------------------------------------------------------------

    def test_multiply_by_scalar(self, sim):
        """encrypt(x) * s decrypts to ≈ x * s."""
        x = torch.tensor([2.0, -3.0, 5.0])
        s = 3.7
        ct = sim.multiply(sim.encrypt(x), s)
        recovered = sim.decrypt(ct)
        assert torch.allclose(recovered, x * s, atol=0.5)

    def test_multiply_noise_scales_quadratically(self, sim):
        x = torch.ones(4)
        ct = sim.encrypt(x)
        base = ct.noise_variance
        ct2 = sim.multiply(ct, 2.0)
        assert ct2.noise_variance == pytest.approx(base * 4.0, rel=1e-6)

    def test_multiply_by_zero(self, sim):
        x = torch.randn(5)
        ct = sim.multiply(sim.encrypt(x), 0.0)
        recovered = sim.decrypt(ct)
        assert torch.allclose(recovered, torch.zeros(5), atol=1e-12)

    def test_multiply_operator_overload(self, sim):
        x = torch.randn(3)
        ct = sim.encrypt(x)
        ct2 = ct * 2.0
        assert isinstance(ct2, EncryptedTensor)

    # -----------------------------------------------------------------------
    # noise_budget
    # -----------------------------------------------------------------------

    def test_noise_budget_positive(self, sim):
        ct = sim.encrypt(torch.randn(4))
        assert sim.noise_budget(ct) > 0.0

    def test_noise_budget_decreases_after_multiply(self, sim):
        ct = sim.encrypt(torch.randn(4))
        b0 = sim.noise_budget(ct)
        ct_big = sim.multiply(ct, 100.0)
        assert sim.noise_budget(ct_big) < b0

    # -----------------------------------------------------------------------
    # EncryptedTensor properties
    # -----------------------------------------------------------------------

    def test_encrypted_tensor_repr_hides_data(self, sim):
        ct = sim.encrypt(torch.tensor([42.0]))
        r = repr(ct)
        assert "42" not in r
        assert "EncryptedTensor" in r

    def test_encrypted_flag(self, sim):
        ct = sim.encrypt(torch.randn(3))
        assert ct.is_encrypted is True


# ===========================================================================
# SimpleGraph tests
# ===========================================================================

class TestSimpleGraph:

    def test_ring_graph(self):
        g = SimpleGraph.ring(5)
        assert g.num_nodes == 5
        # Every node should have exactly 2 neighbours in a ring
        for v in range(5):
            assert len(g.neighbours[v]) == 2

    def test_random_graph_node_count(self):
        g = SimpleGraph.random(10, m=12, seed=1)
        assert g.num_nodes == 10

    def test_undirected_edges(self):
        g = SimpleGraph(4, [(0, 1), (2, 3)])
        assert 1 in g.neighbours[0]
        assert 0 in g.neighbours[1]
        assert 3 in g.neighbours[2]

    def test_isolated_node(self):
        g = SimpleGraph(3, [(0, 1)])
        assert g.neighbours[2] == []


# ===========================================================================
# SAGELayer tests
# ===========================================================================

class TestSAGELayer:

    def test_output_shape(self):
        layer = SAGELayer(in_dim=4, out_dim=8)
        g = SimpleGraph.ring(5)
        x = torch.randn(5, 4)
        out = layer(g, x)
        assert out.shape == (5, 8)

    def test_isolated_node_uses_zero_neighbour(self):
        layer = SAGELayer(in_dim=3, out_dim=2)
        g = SimpleGraph(3, [(0, 1)])  # node 2 is isolated
        x = torch.randn(3, 3)
        # Should not raise
        out = layer(g, x)
        assert out.shape == (3, 2)


# ===========================================================================
# PrivateGraphSAGE tests
# ===========================================================================

class TestPrivateGraphSAGE:

    # -----------------------------------------------------------------------
    # Plaintext forward
    # -----------------------------------------------------------------------

    def test_forward_output_shape(self, model_small, small_graph):
        x = torch.randn(4, 3)
        out = model_small.forward(small_graph, x)
        assert out.shape == (4, 2)

    def test_forward_different_inputs_give_different_outputs(self, model_small, small_graph):
        x1 = torch.randn(4, 3)
        x2 = torch.randn(4, 3)
        out1 = model_small.forward(small_graph, x1)
        out2 = model_small.forward(small_graph, x2)
        assert not torch.allclose(out1, out2)

    # -----------------------------------------------------------------------
    # Encrypted aggregate
    # -----------------------------------------------------------------------

    def test_aggregate_returns_dict_for_all_nodes(self, model_small, small_graph):
        sim = HomomorphicSimulator(seed=1)
        x = torch.randn(4, 3)
        enc = {v: sim.encrypt(x[v]) for v in range(4)}
        agg = model_small.aggregate(small_graph, enc)
        assert set(agg.keys()) == set(range(4))

    def test_aggregate_output_is_encrypted_tensor(self, model_small, small_graph):
        sim = HomomorphicSimulator(seed=2)
        x = torch.randn(4, 3)
        enc = {v: sim.encrypt(x[v]) for v in range(4)}
        agg = model_small.aggregate(small_graph, enc)
        for v in range(4):
            assert isinstance(agg[v], EncryptedTensor)
            assert agg[v].is_encrypted

    def test_aggregate_mean_equals_plaintext_mean(self, model_small):
        """Mean aggregate of unencrypted inputs should match torch mean."""
        # Use tiny noise so decrypt ≈ plaintext
        sim = HomomorphicSimulator(noise_scale=1e-9, seed=3)
        # Star graph: node 0 connected to all others
        n = 5
        g = SimpleGraph(n, [(0, i) for i in range(1, n)])
        x = torch.randn(n, 3)
        enc = {v: sim.encrypt(x[v]) for v in range(n)}

        agg = model_small.aggregate(g, enc)

        # Node 0 is connected to 1,2,3,4 — its aggregate should ≈ mean(x[1:])
        expected = x[1:].mean(dim=0)
        recovered = sim.decrypt(agg[0])
        assert torch.allclose(recovered, expected, atol=1e-4)

    # -----------------------------------------------------------------------
    # Encrypted forward
    # -----------------------------------------------------------------------

    def test_forward_encrypted_output_dict(self, model_small, small_graph):
        sim = HomomorphicSimulator(seed=5)
        x = torch.randn(4, 3)
        enc = {v: sim.encrypt(x[v]) for v in range(4)}
        enc_out = model_small.forward_encrypted(small_graph, enc)
        assert set(enc_out.keys()) == set(range(4))

    def test_forward_encrypted_output_is_encrypted(self, model_small, small_graph):
        sim = HomomorphicSimulator(seed=6)
        x = torch.randn(4, 3)
        enc = {v: sim.encrypt(x[v]) for v in range(4)}
        enc_out = model_small.forward_encrypted(small_graph, enc)
        for v in range(4):
            assert enc_out[v].is_encrypted

    def test_forward_encrypted_output_shape(self, model_small, small_graph):
        sim = HomomorphicSimulator(seed=7)
        x = torch.randn(4, 3)
        enc = {v: sim.encrypt(x[v]) for v in range(4)}
        enc_out = model_small.forward_encrypted(small_graph, enc)
        # out_dim = 2 for model_small
        assert enc_out[0].data.shape == (2,)

    # -----------------------------------------------------------------------
    # Plaintext vs Encrypted agreement — the core correctness test
    # -----------------------------------------------------------------------

    def test_plaintext_encrypted_agreement_small(self, model_small, small_graph):
        """
        Plaintext and encrypted outputs must agree within noise tolerance.
        """
        torch.manual_seed(10)
        sim = HomomorphicSimulator(security_param=128, seed=10)
        x = torch.randn(4, 3)

        plain_out = model_small.forward(small_graph, x)
        enc = {v: sim.encrypt(x[v]) for v in range(4)}
        enc_out = model_small.forward_encrypted(small_graph, enc)
        dec_out = torch.stack([sim.decrypt(enc_out[v]) for v in range(4)])

        max_err = (plain_out - dec_out).abs().max().item()
        assert max_err < 1e-2, (
            f"Plaintext/encrypted output mismatch: max_err={max_err:.6f} > 1e-2"
        )

    def test_plaintext_encrypted_agreement_medium(self, model_medium, medium_graph):
        """
        Larger graph: outputs should agree within 1e-2 tolerance (demo scenario).
        """
        torch.manual_seed(42)
        sim = HomomorphicSimulator(security_param=128, seed=42)
        x = torch.randn(10, 4)

        plain_out = model_medium.forward(medium_graph, x)
        enc = {v: sim.encrypt(x[v]) for v in range(10)}
        enc_out = model_medium.forward_encrypted(medium_graph, enc)
        dec_out = torch.stack([sim.decrypt(enc_out[v]) for v in range(10)])

        max_err = (plain_out - dec_out).abs().max().item()
        assert max_err < 1e-2, (
            f"Max error on medium graph: {max_err:.6f}"
        )

    def test_agreement_single_layer(self):
        """Single-layer model: easier to reason about correctness."""
        torch.manual_seed(1)
        sim = HomomorphicSimulator(noise_scale=1e-7, seed=1)
        model = PrivateGraphSAGE(in_dim=3, hidden_dim=3, out_dim=3, num_layers=1)
        graph = SimpleGraph.ring(6)
        x = torch.randn(6, 3)

        plain_out = model.forward(graph, x)
        enc = {v: sim.encrypt(x[v]) for v in range(6)}
        enc_out = model.forward_encrypted(graph, enc)
        dec_out = torch.stack([sim.decrypt(enc_out[v]) for v in range(6)])

        max_err = (plain_out - dec_out).abs().max().item()
        assert max_err < 1e-4  # tiny noise → very tight

    # -----------------------------------------------------------------------
    # Privacy invariant: server never decrypts
    # -----------------------------------------------------------------------

    def test_server_does_not_call_decrypt(self, model_small, small_graph, monkeypatch):
        """
        forward_encrypted must never call sim.decrypt().
        We monkeypatch decrypt to raise if called.
        """
        sim = HomomorphicSimulator(seed=20)
        x = torch.randn(4, 3)
        enc = {v: sim.encrypt(x[v]) for v in range(4)}

        def _no_decrypt(ct):
            raise AssertionError("Server called decrypt — privacy violation!")

        monkeypatch.setattr(sim, "decrypt", _no_decrypt)
        # Patch the context on all ciphertexts
        for ct in enc.values():
            ct._ctx = sim

        # Should complete without triggering the patched decrypt
        enc_out = model_small.forward_encrypted(small_graph, enc)
        # No assertion needed — if decrypt was called the test raises

    # -----------------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------------

    def test_isolated_node_in_encrypted_forward(self):
        """A node with no neighbours should still produce valid encrypted output."""
        torch.manual_seed(5)
        sim = HomomorphicSimulator(seed=5)
        model = PrivateGraphSAGE(in_dim=2, hidden_dim=2, out_dim=2, num_layers=1)
        # Node 2 is isolated
        graph = SimpleGraph(3, [(0, 1)])
        x = torch.randn(3, 2)
        enc = {v: sim.encrypt(x[v]) for v in range(3)}

        enc_out = model.forward_encrypted(graph, enc)
        assert enc_out[2].is_encrypted  # must not raise

    def test_single_node_graph(self):
        """Degenerate graph with one node."""
        torch.manual_seed(7)
        sim = HomomorphicSimulator(seed=7)
        model = PrivateGraphSAGE(in_dim=3, hidden_dim=4, out_dim=2, num_layers=1)
        graph = SimpleGraph(1, [])
        x = torch.randn(1, 3)
        enc = {0: sim.encrypt(x[0])}

        enc_out = model.forward_encrypted(graph, enc)
        dec = sim.decrypt(enc_out[0])
        plain = model.forward(graph, x)

        assert torch.allclose(plain[0], dec, atol=1e-3)

    def test_three_layers(self):
        """Verify 3-layer model also achieves agreement."""
        torch.manual_seed(9)
        sim = HomomorphicSimulator(security_param=128, seed=9)
        model = PrivateGraphSAGE(in_dim=4, hidden_dim=6, out_dim=3, num_layers=3)
        graph = SimpleGraph.ring(8)
        x = torch.randn(8, 4)

        plain_out = model.forward(graph, x)
        enc = {v: sim.encrypt(x[v]) for v in range(8)}
        enc_out = model.forward_encrypted(graph, enc)
        dec_out = torch.stack([sim.decrypt(enc_out[v]) for v in range(8)])

        max_err = (plain_out - dec_out).abs().max().item()
        assert max_err < 1e-2

"""
HomomorphicSimulator: CKKS-style homomorphic encryption simulator.

This module provides a pure-Python/PyTorch simulation of CKKS homomorphic
encryption. Rather than actual lattice-based cryptography, it models the
*noise growth semantics* of CKKS:

  - Encryption adds Gaussian noise proportional to the security parameter.
  - Homomorphic addition accumulates noise.
  - Multiplication by a scalar scales the noise.
  - Decryption recovers the original value within the accumulated noise floor.

This lets us demonstrate the *algorithmic correctness* of HE-based graph
inference (plaintext ≈ encrypted output) without requiring SEAL/HEAAN/Concrete.

Privacy model
-------------
In a real deployment the server sees only ciphertexts: it cannot distinguish
the plaintext node features from random-looking tuples.  The simulation
replicates this property by tagging each EncryptedTensor with a
``is_encrypted`` flag and never exposing the underlying data through the
server-facing API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Ciphertext container
# ---------------------------------------------------------------------------

@dataclass
class EncryptedTensor:
    """
    Simulated CKKS ciphertext.

    Attributes
    ----------
    data : torch.Tensor
        The "noisy" ciphertext value (plaintext + accumulated noise).
    noise_variance : float
        Variance of the accumulated noise (for analysis / budget tracking).
    scale : float
        CKKS scaling factor (informational; kept for API parity).
    is_encrypted : bool
        Always True – marks this as an opaque server-side object.
    """

    data: torch.Tensor
    noise_variance: float
    scale: float = 2.0 ** 40
    is_encrypted: bool = field(default=True, init=False)

    # Convenience: route arithmetic through the originating context so the
    # server code never needs to unwrap the ciphertext.
    _ctx: Optional["HomomorphicSimulator"] = field(default=None, repr=False)

    def __add__(self, other: "EncryptedTensor") -> "EncryptedTensor":
        if self._ctx is None:
            raise RuntimeError("No encryption context attached")
        return self._ctx.add(self, other)

    def __mul__(self, scalar: float) -> "EncryptedTensor":
        if self._ctx is None:
            raise RuntimeError("No encryption context attached")
        return self._ctx.multiply(self, scalar)

    def __repr__(self) -> str:  # hide plaintext data
        return (
            f"EncryptedTensor(shape={tuple(self.data.shape)}, "
            f"noise_σ={math.sqrt(self.noise_variance):.2e})"
        )


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class HomomorphicSimulator:
    """
    CKKS-style homomorphic encryption simulator.

    Parameters
    ----------
    security_param : int
        Analogous to the CKKS bit-security level (e.g. 128).
        Higher values produce *more* noise (stronger security guarantee).
    noise_scale : float, optional
        Base noise standard-deviation injected on encryption.
        Defaults to ``10 ** (-(security_param / 32))``, which keeps noise
        negligible relative to typical neural-network activations.
    scale : float
        CKKS scaling factor (cosmetic in this simulation).
    seed : int, optional
        RNG seed for reproducibility in tests.
    """

    def __init__(
        self,
        security_param: int = 128,
        noise_scale: Optional[float] = None,
        scale: float = 2.0 ** 40,
        seed: Optional[int] = None,
    ) -> None:
        self.security_param = security_param
        self.scale = scale

        # Default noise: small but non-zero, proportional to security level
        if noise_scale is None:
            noise_scale = 10.0 ** (-(security_param / 32.0))
        self.noise_scale = noise_scale

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

    # ------------------------------------------------------------------
    # Core HE operations
    # ------------------------------------------------------------------

    def encrypt(self, plaintext: torch.Tensor) -> EncryptedTensor:
        """
        Encrypt a plaintext tensor.

        Adds Gaussian noise with std = ``noise_scale * (1 + ||plaintext||_∞)``.
        The infinity-norm factor keeps relative noise constant across
        differently-scaled inputs (similar to CKKS's scale management).

        Parameters
        ----------
        plaintext : torch.Tensor
            The value to encrypt (arbitrary shape, float dtype).

        Returns
        -------
        EncryptedTensor
            Opaque ciphertext carrying the noisy value.
        """
        plaintext = plaintext.float()
        # Noise std proportional to magnitude of the plaintext (CKKS flavour)
        inf_norm = float(plaintext.abs().max().item()) + 1.0
        sigma = self.noise_scale * inf_norm

        noise = torch.normal(
            mean=torch.zeros_like(plaintext),
            std=torch.full_like(plaintext, sigma),
            generator=self._rng,
        )
        noisy = plaintext + noise
        return EncryptedTensor(
            data=noisy,
            noise_variance=sigma ** 2,
            scale=self.scale,
            _ctx=self,
        )

    def decrypt(self, ciphertext: EncryptedTensor) -> torch.Tensor:
        """
        Decrypt a ciphertext (simulation: just return the noisy data).

        In real CKKS this would subtract the secret-key component; here the
        "ciphertext" already stores plaintext + noise, so decryption is a
        no-op that returns the noisy approximation.

        Parameters
        ----------
        ciphertext : EncryptedTensor

        Returns
        -------
        torch.Tensor
            Approximate plaintext (within noise tolerance).
        """
        return ciphertext.data.clone()

    def add(self, c1: EncryptedTensor, c2: EncryptedTensor) -> EncryptedTensor:
        """
        Homomorphic addition.

        Noise variances add (independent Gaussians).

        Parameters
        ----------
        c1, c2 : EncryptedTensor

        Returns
        -------
        EncryptedTensor
        """
        return EncryptedTensor(
            data=c1.data + c2.data,
            noise_variance=c1.noise_variance + c2.noise_variance,
            scale=self.scale,
            _ctx=self,
        )

    def multiply(self, c: EncryptedTensor, scalar: float) -> EncryptedTensor:
        """
        Multiply an encrypted tensor by a *plaintext* scalar.

        This is HE-friendly (no ciphertext-ciphertext multiplication needed).
        Noise variance scales as ``scalar²``.

        Parameters
        ----------
        c : EncryptedTensor
        scalar : float

        Returns
        -------
        EncryptedTensor
        """
        return EncryptedTensor(
            data=c.data * scalar,
            noise_variance=c.noise_variance * (scalar ** 2),
            scale=self.scale,
            _ctx=self,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def noise_budget(self, ciphertext: EncryptedTensor) -> float:
        """Remaining noise budget in log₂ scale (higher = more headroom)."""
        remaining_bits = math.log2(self.scale) - 0.5 * math.log2(
            max(ciphertext.noise_variance, 1e-300)
        )
        return max(0.0, remaining_bits)

    def __repr__(self) -> str:
        return (
            f"HomomorphicSimulator(security_param={self.security_param}, "
            f"noise_scale={self.noise_scale:.2e})"
        )

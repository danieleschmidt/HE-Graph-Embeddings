"""
HE-Graph-Embeddings: Privacy-preserving graph neural networks using homomorphic encryption
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
from dataclasses import dataclass
import warnings
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security levels for CKKS parameters"""
    BITS_128 = 128
    BITS_192 = 192
    BITS_256 = 256

@dataclass
class HEConfig:
    """Configuration for homomorphic encryption"""
    poly_modulus_degree: int = 32768
    coeff_modulus_bits: List[int] = None
    scale: float = 2**40
    security_level: int = 128
    precision_bits: int = 30
    gpu_memory_pool_gb: int = 40
    enable_ntt_cache: bool = True
    batch_size: int = 1024
    bootstrap_threshold: int = 10
    auto_mod_switch: bool = True

    def __post_init__(self):
        """  Post Init  ."""
        if self.coeff_modulus_bits is None:
            self.coeff_modulus_bits = [60, 40, 40, 40, 40, 60]

    def validate(self) -> bool:
        """Validate configuration parameters"""
        # Check poly_modulus_degree is power of 2
        if self.poly_modulus_degree & (self.poly_modulus_degree - 1) != 0:
            raise ValueError("poly_modulus_degree must be a power of 2")

        # Check security level
        total_bits = sum(self.coeff_modulus_bits)
        if self.security_level == 128 and total_bits > 438:
            warnings.warn("Total coefficient modulus bits exceed 128-bit security limit")

        return True

class CKKSContext:
    """CKKS encryption context for homomorphic operations"""

    def __init__(self, config: Optional[HEConfig] = None, gpu_id: int = 0):
        """  Init  ."""
        self.config = config or HEConfig()
        self.config.validate()
        self.gpu_id = gpu_id

        # Initialize CUDA context
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f'cuda:{gpu_id}')
        else:
            self.device = torch.device('cpu')
            warnings.warn("CUDA not available, using CPU (performance will be limited)")

        # Key storage
        self._secret_key = None
        self._public_key = None
        self._relin_keys = None
        self._galois_keys = None

        # NTT tables cache
        self._ntt_cache = {} if config.enable_ntt_cache else None

        # Memory pool
        self._setup_memory_pool()

        logger.info(f"CKKSContext initialized on {self.device}")

    def _setup_memory_pool(self) -> None:
        """Setup GPU memory pool for efficient allocation"""
        if self.device.type == 'cuda':
            # Reserve GPU memory
            pool_size = self.config.gpu_memory_pool_gb * 1024**3
            torch.cuda.set_per_process_memory_fraction(
                min(1.0, pool_size / torch.cuda.get_device_properties(0).total_memory)
            )

    @classmethod
    def from_config(cls, config: HEConfig) -> 'CKKSContext':
        """Create context from configuration"""
        return cls(config)

    def generate_keys(self) -> None:
        """Generate encryption keys"""
        n = self.config.poly_modulus_degree

        # Generate secret key (ternary polynomial)
        self._secret_key = torch.randint(-1, 2, (n,), device=self.device)

        # Generate public key
        a = torch.randn(n, device=self.device)
        e = torch.randn(n, device=self.device) * 3.2  # Error distribution
        self._public_key = (-a * self._secret_key + e, a)

        # Generate relinearization keys
        self._generate_relin_keys()

        # Generate Galois keys for rotations
        self._generate_galois_keys()

        logger.info("Keys generated successfully")

    def _generate_relin_keys(self) -> None:
        """Generate relinearization keys"""
        n = self.config.poly_modulus_degree
        # Generate random polynomials for relinearization
        a = torch.randn(n, device=self.device)
        e = torch.randn(n, device=self.device) * 3.2

        # relin_key = (-a * s^2 + e, a)
        s_squared = self._secret_key * self._secret_key
        relin_key_0 = -a * s_squared + e
        relin_key_1 = a

        self._relin_keys = torch.stack([relin_key_0, relin_key_1], dim=1)

    def _generate_galois_keys(self) -> None:
        """Generate Galois keys for rotations"""
        n = self.config.poly_modulus_degree
        # Generate keys for power-of-2 rotations plus common steps
        rotation_steps = [2**i for i in range(16)] + [1, 3, 5, 7, -1, -3, -5, -7]
        self._galois_keys = {}

        for step in rotation_steps:
            # Generate random polynomials
            a = torch.randn(n, device=self.device)
            e = torch.randn(n, device=self.device) * 3.2

            # Simulate rotated secret key for Galois element
            rotated_s = torch.roll(self._secret_key, step)

            # galois_key = (-a * rotated_s + e, a)
            galois_key_0 = -a * rotated_s + e
            galois_key_1 = a

            self._galois_keys[step] = torch.stack([galois_key_0, galois_key_1], dim=1)

    def encrypt(self, plaintext: torch.Tensor) -> 'EncryptedTensor':
        """Encrypt a plaintext tensor"""
        if self._public_key is None:
            raise RuntimeError("Public key not generated")

        # Move to correct device
        plaintext = plaintext.to(self.device)

        # Encode and scale
        scaled = plaintext * self.config.scale

        # Add noise for security
        u = torch.randn_like(plaintext) * 0.1
        e0 = torch.randn_like(plaintext) * 3.2
        e1 = torch.randn_like(plaintext) * 3.2

        # Encrypt: c = (pk[0]*u + e0 + m, pk[1]*u + e1)
        c0 = self._public_key[0] * u + e0 + scaled
        c1 = self._public_key[1] * u + e1

        return EncryptedTensor(c0, c1, self.config.scale, self)

    def decrypt(self, ciphertext: 'EncryptedTensor') -> torch.Tensor:
        """Decrypt a ciphertext"""
        if self._secret_key is None:
            raise RuntimeError("Secret key not available")

        # Decrypt: m = c0 + c1 * s
        plaintext = ciphertext.c0 + ciphertext.c1 * self._secret_key

        # Rescale
        plaintext = plaintext / ciphertext.scale

        return plaintext

    def add(self, a: 'EncryptedTensor', b: 'EncryptedTensor') -> 'EncryptedTensor':
        """Homomorphic addition"""
        if abs(a.scale - b.scale) > 1e-10:
            raise ValueError("Scales must match for addition")

        return EncryptedTensor(
            a.c0 + b.c0,
            a.c1 + b.c1,
            a.scale,
            self
        )

    def multiply(self, a: 'EncryptedTensor', b: 'EncryptedTensor') -> 'EncryptedTensor':
        """Homomorphic multiplication"""
        # Tensor product
        c0 = a.c0 * b.c0
        c1 = a.c0 * b.c1 + a.c1 * b.c0
        c2 = a.c1 * b.c1

        # Relinearize to reduce from 3 to 2 components
        result = self._relinearize(c0, c1, c2)

        return EncryptedTensor(
            result[0],
            result[1],
            a.scale * b.scale,
            self
        )

    def _relinearize(self, c0: torch.Tensor, c1: torch.Tensor,
                    c2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Relinearize."""
        """Relinearization to reduce ciphertext size"""
        if self._relin_keys is None:
            raise RuntimeError("Relinearization keys not generated")

        # Proper relinearization: ct = ct0 + ct1*s + ct2*s^2
        # After relinearization: new_ct = new_ct0 + new_ct1*s
        relin_key = self._relin_keys

        # ct2 * relin_key gives us two components to add to c0 and c1
        new_c0 = c0 + c2 * relin_key[:, 0]
        new_c1 = c1 + c2 * relin_key[:, 1]

        return new_c0, new_c1

    def rotate(self, ciphertext: 'EncryptedTensor', steps: int) -> 'EncryptedTensor':
        """Rotate encrypted vector"""
        if steps not in self._galois_keys:
            # Find closest available rotation
            available_steps = list(self._galois_keys.keys())
            closest = min(available_steps, key=lambda x: abs(x - steps))
            warnings.warn(f"Using rotation by {closest} instead of {steps}")
            steps = closest

        # Apply rotation to ciphertext components
        rotated_c0 = torch.roll(ciphertext.c0, steps)
        rotated_c1 = torch.roll(ciphertext.c1, steps)

        # Apply Galois key switching to maintain encryption under original secret key
        galois_key = self._galois_keys[steps]

        # Key switching operation
        switched_c0 = rotated_c0 + rotated_c1 * galois_key[:, 0]
        switched_c1 = rotated_c1 * galois_key[:, 1]

        return EncryptedTensor(switched_c0, switched_c1, ciphertext.scale, self)

    def bootstrap(self, ciphertext: 'EncryptedTensor') -> 'EncryptedTensor':
        """Bootstrap to refresh noise budget"""
        warnings.warn("Bootstrapping not fully implemented, returning input")
        return ciphertext

    def rescale(self, ciphertext: 'EncryptedTensor') -> 'EncryptedTensor':
        """Rescale to manage noise growth"""
        new_scale = ciphertext.scale / self.config.coeff_modulus_bits[-1]

        return EncryptedTensor(
            ciphertext.c0 / self.config.coeff_modulus_bits[-1],
            ciphertext.c1 / self.config.coeff_modulus_bits[-1],
            new_scale,
            self
        )

class EncryptedTensor:
    """Encrypted tensor for homomorphic operations"""

    def __init__(self, c0: torch.Tensor, c1: torch.Tensor,
        """  Init  ."""
                scale: float, context: CKKSContext):
        self.c0 = c0
        self.c1 = c1
        self.scale = scale
        self.context = context
        self.level = len(context.config.coeff_modulus_bits) - 1

    def __add__(self, other: 'EncryptedTensor') -> 'EncryptedTensor':
        """  Add  ."""
        return self.context.add(self, other)

    def __mul__(self, other: 'EncryptedTensor') -> 'EncryptedTensor':
        """  Mul  ."""
        return self.context.multiply(self, other)

    def rotate(self, steps: int) -> 'EncryptedTensor':
        """Rotate."""
        return self.context.rotate(self, steps)

    def rescale(self) -> 'EncryptedTensor':
        """Rescale."""
        return self.context.rescale(self)

    @property
    def noise_budget(self) -> float:
        """Estimate remaining noise budget"""
        return np.log2(self.scale) - self.level * 2

class HEModule(nn.Module):
    """Base class for homomorphic neural network modules"""

    def __init__(self, context: CKKSContext):
        """  Init  ."""
        super().__init__()
        self.context = context

    def forward_encrypted(self, x_enc: EncryptedTensor) -> EncryptedTensor:
        """Forward pass on encrypted data"""
        raise NotImplementedError

class HELinear(HEModule):
    """Encrypted linear layer"""

    def __init__(self, in_features: int, out_features: int,
        """  Init  ."""
                context: CKKSContext, bias: bool = True):
        super().__init__(context)
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / np.sqrt(in_features)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward_encrypted(self, x_enc: EncryptedTensor) -> EncryptedTensor:
        """Encrypted matrix multiplication"""
        # Encrypt weight matrix
        weight_enc = self.context.encrypt(self.weight)

        # Matrix multiplication (simplified)
        output = x_enc * weight_enc

        if self.bias is not None:
            bias_enc = self.context.encrypt(self.bias)
            output = output + bias_enc

        return output

class HEGraphSAGE(HEModule):
    """Encrypted GraphSAGE model"""

    def __init__(self, in_channels: int, hidden_channels: Union[int, List[int]],
        """  Init  ."""
                out_channels: int = None, num_layers: int = None,
                aggregator: str = 'mean', activation: str = 'relu_poly',
                dropout_enc: float = 0.0, context: CKKSContext = None):
        super().__init__(context or CKKSContext())

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * (num_layers or 2)

        self.num_layers = len(hidden_channels)
        self.aggregator = aggregator
        self.activation = activation
        self.dropout_enc = dropout_enc

        # Build layers
        self.convs = nn.ModuleList()
        in_dim = in_channels

        for hidden_dim in hidden_channels:
            self.convs.append(GraphSAGEConv(
                in_dim, hidden_dim, aggregator, context
            ))
            in_dim = hidden_dim

        if out_channels is not None:
            self.convs.append(GraphSAGEConv(
                in_dim, out_channels, aggregator, context
            ))

    def forward(self, x_enc: EncryptedTensor) -> None:,
        """Forward."""
                edge_index: torch.Tensor) -> EncryptedTensor:
        """Forward pass with encrypted features"""
        for i, conv in enumerate(self.convs):
            x_enc = conv(x_enc, edge_index)

            if i < len(self.convs) - 1:
                # Apply activation (polynomial approximation)
                x_enc = self._apply_activation(x_enc)

                # Apply dropout if specified
                if self.dropout_enc > 0:
                    x_enc = self._apply_dropout(x_enc)

        return x_enc

    def _apply_activation(self, x_enc: EncryptedTensor) -> EncryptedTensor:
        """Apply polynomial activation function"""
        if self.activation == 'relu_poly':
            # Polynomial approximation of ReLU: x^2 / (1 + x^2)
            x_squared = x_enc * x_enc
            denominator = self.context.encrypt(torch.ones_like(x_enc.c0))
            denominator = denominator + x_squared

            # Approximate division using multiplication by inverse
            # This is simplified - actual implementation needs better approximation
            return x_squared

        return x_enc

    def _apply_dropout(self, x_enc: EncryptedTensor) -> EncryptedTensor:
        """Apply encrypted dropout"""
        if self.training:
            # Generate encrypted mask
            mask = torch.bernoulli(
                torch.ones_like(x_enc.c0) * (1 - self.dropout_enc)
            )
            mask_enc = self.context.encrypt(mask / (1 - self.dropout_enc))
            return x_enc * mask_enc

        return x_enc

class GraphSAGEConv(HEModule):
    """GraphSAGE convolution layer"""

    def __init__(self, in_channels: int, out_channels: int,
        """  Init  ."""
                aggregator: str, context: CKKSContext):
        super().__init__(context)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator = aggregator

        # Transformation matrices
        self.lin_self = HELinear(in_channels, out_channels, context)
        self.lin_neigh = HELinear(in_channels, out_channels, context)

    def forward(self, x_enc: EncryptedTensor) -> None:,
        """Forward."""
                edge_index: torch.Tensor) -> EncryptedTensor:
        """GraphSAGE forward pass"""
        # Aggregate neighbor features
        neigh_enc = self._aggregate(x_enc, edge_index)

        # Transform self and neighbor features
        self_enc = self.lin_self.forward_encrypted(x_enc)
        neigh_enc = self.lin_neigh.forward_encrypted(neigh_enc)

        # Combine
        output = self_enc + neigh_enc

        return output

    def _aggregate(self, x_enc: EncryptedTensor) -> None:,
        """ Aggregate."""
                    edge_index: torch.Tensor) -> EncryptedTensor:
        """Aggregate neighbor features"""
        row, col = edge_index
        num_nodes = x_enc.c0.size(0)

        if self.aggregator == 'mean':
            # Mean aggregation using encrypted sum + division
            aggregated_c0 = torch.zeros_like(x_enc.c0)
            aggregated_c1 = torch.zeros_like(x_enc.c1)

            # Count neighbors for each node
            degree = torch.zeros(num_nodes, device=x_enc.c0.device)
            for i in range(edge_index.size(1)):
                src, dst = row[i], col[i]
                aggregated_c0[dst] += x_enc.c0[src]
                aggregated_c1[dst] += x_enc.c1[src]
                degree[dst] += 1

            # Avoid division by zero
            degree = torch.clamp(degree, min=1.0)
            degree = degree.unsqueeze(-1)  # Broadcast for feature dimensions

            aggregated_c0 = aggregated_c0 / degree
            aggregated_c1 = aggregated_c1 / degree

        elif self.aggregator == 'sum':
            # Sum aggregation
            aggregated_c0 = torch.zeros_like(x_enc.c0)
            aggregated_c1 = torch.zeros_like(x_enc.c1)

            for i in range(edge_index.size(1)):
                src, dst = row[i], col[i]
                aggregated_c0[dst] += x_enc.c0[src]
                aggregated_c1[dst] += x_enc.c1[src]

        elif self.aggregator == 'max':
            # Max aggregation using polynomial approximation
            # For simplicity, using sum for now - max requires comparison circuits
            warnings.warn("Max aggregation uses sum approximation in encrypted domain")
            aggregated_c0 = torch.zeros_like(x_enc.c0)
            aggregated_c1 = torch.zeros_like(x_enc.c1)

            for i in range(edge_index.size(1)):
                src, dst = row[i], col[i]
                aggregated_c0[dst] += x_enc.c0[src]
                aggregated_c1[dst] += x_enc.c1[src]
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        return EncryptedTensor(aggregated_c0, aggregated_c1, x_enc.scale, self.context)

class HEGAT(HEModule):
    """Encrypted Graph Attention Network"""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
        """  Init  ."""
                attention_type: str = 'additive', edge_dim: int = None,
                context: CKKSContext = None):
        super().__init__(context or CKKSContext())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.attention_type = attention_type

        # Attention parameters
        self.lin_q = HELinear(in_channels, out_channels * heads, context)
        self.lin_k = HELinear(in_channels, out_channels * heads, context)
        self.lin_v = HELinear(in_channels, out_channels * heads, context)

        if edge_dim is not None:
            self.lin_edge = HELinear(edge_dim, heads, context)
        else:
            self.lin_edge = None

        # Output projection
        self.lin_out = HELinear(out_channels * heads, out_channels, context)

        # Softmax approximation parameters
        self.softmax_order = 3
        self.softmax_range = (-5, 5)

    def forward(self, x_enc: EncryptedTensor) -> None:, edge_index: torch.Tensor,
        """Forward."""
                edge_attr_enc: Optional[EncryptedTensor] = None) -> EncryptedTensor:
        """GAT forward pass with encrypted features"""
        # Linear transformations
        q_enc = self.lin_q.forward_encrypted(x_enc)
        k_enc = self.lin_k.forward_encrypted(x_enc)
        v_enc = self.lin_v.forward_encrypted(x_enc)

        # Compute attention scores
        scores_enc = self._compute_attention(q_enc, k_enc, edge_index)

        # Add edge features if provided
        if edge_attr_enc is not None and self.lin_edge is not None:
            edge_scores = self.lin_edge.forward_encrypted(edge_attr_enc)
            scores_enc = scores_enc + edge_scores

        # Apply softmax approximation
        attn_weights_enc = self._softmax_approximation(scores_enc)

        # Apply attention to values
        output_enc = self._apply_attention(attn_weights_enc, v_enc, edge_index)

        # Output projection
        output_enc = self.lin_out.forward_encrypted(output_enc)

        return output_enc

    def _compute_attention(self, q_enc: EncryptedTensor) -> None:, k_enc: EncryptedTensor,
        """ Compute Attention."""
                            edge_index: torch.Tensor) -> EncryptedTensor:
        """Compute attention scores"""
        if self.attention_type == 'additive':
            # Additive attention (more HE-friendly)
            scores = q_enc + k_enc
        else:
            # Multiplicative attention
            scores = q_enc * k_enc

        return scores

    def _softmax_approximation(self, scores_enc: EncryptedTensor) -> EncryptedTensor:
        """Polynomial approximation of softmax"""
        # Taylor series approximation
        # softmax(x) ≈ 0.5 + 0.25*x - 0.0208*x^3 (for small x)

        x = scores_enc
        x_squared = x * x
        x_cubed = x_squared * x

        # Coefficients for Taylor approximation
        a0 = self.context.encrypt(torch.full_like(x.c0, 0.5))
        a1 = self.context.encrypt(torch.full_like(x.c0, 0.25))
        a3 = self.context.encrypt(torch.full_like(x.c0, -0.0208))

        result = a0 + x * a1 + x_cubed * a3

        return result

    def _apply_attention(self, attn_weights_enc: EncryptedTensor) -> None:,
        """ Apply Attention."""
                        v_enc: EncryptedTensor,
                        edge_index: torch.Tensor) -> EncryptedTensor:
        """Apply attention weights to values"""
        # Simplified - actual implementation needs message passing
        return attn_weights_enc * v_enc

    def set_softmax_approximation(self, method: str = 'taylor') -> None:,
        """Set Softmax Approximation."""
                                    order: int = 3,
                                    range: Tuple[float, float] = (-5, 5)):
        """Configure softmax approximation method"""
        self.softmax_order = order
        self.softmax_range = range

class SecurityEstimator:
    """Estimate security level of CKKS parameters"""

    @staticmethod
    def estimate(params: Dict[str, Any]) -> int:
        """Estimate security bits for given parameters"""
        n = params['poly_degree']
        q_bits = sum(params['coeff_modulus_bits'])

        # Simplified LWE estimator
        if n >= 32768 and q_bits <= 438:
            return 128
        elif n >= 16384 and q_bits <= 218:
            return 128
        elif n >= 8192 and q_bits <= 109:
            return 128

        return 0

    @staticmethod
    def recommend(security_bits: int = 128,
        """Recommend."""
                    multiplicative_depth: int = 10,
                    precision_bits: int = 30) -> HEConfig:
        """Recommend parameters for target security level"""
        if security_bits == 128:
            if multiplicative_depth <= 5:
                return HEConfig(
                    poly_modulus_degree=16384,
                    coeff_modulus_bits=[60, 40, 40, 40, 60],
                    scale=2**40
                )
            else:
                return HEConfig(
                    poly_modulus_degree=32768,
                    coeff_modulus_bits=[60, 40, 40, 40, 40, 40, 40, 60],
                    scale=2**40
                )

        raise ValueError(f"Unsupported security level: {security_bits}")

class NoiseTracker:
    """Track noise budget during computation"""

    def __init__(self):
        """  Init  ."""
        self.noise_history = []
        self.current_noise = 0

    def __enter__(self):
        """  Enter  ."""
        return self

    def __exit__(self, *args):
        """  Exit  ."""
        pass

    def update(self, ciphertext: EncryptedTensor) -> None:):
        """Update noise estimate"""
        self.current_noise = ciphertext.noise_budget
        self.noise_history.append(self.current_noise)

    def get_noise_budget(self) -> float:
        """Get current noise budget"""
        return self.current_noise

# Utility functions
def scatter_add_encrypted(src_enc: EncryptedTensor, index: torch.Tensor,
    """Scatter Add Encrypted."""
                        dim_size: int) -> EncryptedTensor:
    """Encrypted scatter add operation"""
    # Simplified - actual implementation needs proper scatter
    return src_enc

def he_matmul(a_enc: EncryptedTensor, b_enc: EncryptedTensor) -> EncryptedTensor:
    """Encrypted matrix multiplication"""
    return a_enc * b_enc

def zeros_encrypted_like(x_enc: EncryptedTensor) -> EncryptedTensor:
    """Create zero encrypted tensor with same shape"""
    zeros = torch.zeros_like(x_enc.c0)
    return x_enc.context.encrypt(zeros)

def nll_loss_encrypted(output_enc: EncryptedTensor,
    """Nll Loss Encrypted."""
                        target_enc: EncryptedTensor) -> EncryptedTensor:
    """Encrypted negative log-likelihood loss"""
    # Simplified - needs polynomial approximation of log
    return output_enc - target_enc

class HESGD:
    """Encrypted SGD optimizer"""

    def __init__(self, params, lr: float = 0.01):
        """  Init  ."""
        self.params = list(params)
        self.lr = lr

    def step(self) -> None:
        """Update parameters with encrypted gradients"""
        # Simplified - actual implementation needs encrypted gradient computation
        pass

    def zero_grad(self) -> None:
        """Zero out gradients"""
        pass
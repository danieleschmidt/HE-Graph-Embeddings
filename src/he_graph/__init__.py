"""
he_graph: Privacy-preserving graph neural networks via simulated homomorphic encryption.
"""

from .homomorphic_simulator import EncryptedTensor, HomomorphicSimulator
from .private_graphsage import PrivateGraphSAGE, SAGELayer, SimpleGraph

__all__ = [
    "HomomorphicSimulator",
    "EncryptedTensor",
    "PrivateGraphSAGE",
    "SAGELayer",
    "SimpleGraph",
]

"""Database models for HE-Graph-Embeddings"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, ForeignKey, Index, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, ARRAY

Base = declarative_base()

class EncryptionStatus(Enum):
    """Status of encryption operations"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class GraphType(Enum):
    """Types of supported graphs"""
    SOCIAL = "social"
    FINANCIAL = "financial"
    MOLECULAR = "molecular"
    KNOWLEDGE = "knowledge"
    SUPPLY_CHAIN = "supply_chain"
    CUSTOM = "custom"

class Graph(Base):
    """Graph metadata storage"""
    __tablename__ = "graphs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    graph_type = Column(String(50), default=GraphType.CUSTOM.value)
    num_nodes = Column(Integer)
    num_edges = Column(Integer)
    node_features_dim = Column(Integer)
    edge_features_dim = Column(Integer)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    
    # Relationships
    owner = relationship("User", back_populates="graphs")
    nodes = relationship("Node", back_populates="graph", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="graph", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="graph", cascade="all, delete-orphan")
    computations = relationship("Computation", back_populates="graph")
    
    __table_args__ = (
        Index("idx_graph_owner", "owner_id"),
        Index("idx_graph_created", "created_at"),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "graph_type": self.graph_type,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "node_features_dim": self.node_features_dim,
            "edge_features_dim": self.edge_features_dim,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class Node(Base):
    """Graph node storage"""
    __tablename__ = "nodes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(UUID(as_uuid=True), ForeignKey("graphs.id"), nullable=False)
    node_id = Column(Integer, nullable=False)  # Node ID within the graph
    features = Column(ARRAY(Float))  # Plain features
    encrypted_features = Column(Text)  # Base64 encoded encrypted features
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    graph = relationship("Graph", back_populates="nodes")
    source_edges = relationship("Edge", foreign_keys="Edge.source_node_id", back_populates="source_node")
    target_edges = relationship("Edge", foreign_keys="Edge.target_node_id", back_populates="target_node")
    
    __table_args__ = (
        Index("idx_node_graph", "graph_id"),
        Index("idx_node_graph_node", "graph_id", "node_id", unique=True),
    )

class Edge(Base):
    """Graph edge storage"""
    __tablename__ = "edges"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(UUID(as_uuid=True), ForeignKey("graphs.id"), nullable=False)
    source_node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=False)
    target_node_id = Column(UUID(as_uuid=True), ForeignKey("nodes.id"), nullable=False)
    weight = Column(Float, default=1.0)
    features = Column(ARRAY(Float))
    encrypted_features = Column(Text)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    graph = relationship("Graph", back_populates="edges")
    source_node = relationship("Node", foreign_keys=[source_node_id], back_populates="source_edges")
    target_node = relationship("Node", foreign_keys=[target_node_id], back_populates="target_edges")
    
    __table_args__ = (
        Index("idx_edge_graph", "graph_id"),
        Index("idx_edge_source", "source_node_id"),
        Index("idx_edge_target", "target_node_id"),
    )

class Embedding(Base):
    """Computed embeddings storage"""
    __tablename__ = "embeddings"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(UUID(as_uuid=True), ForeignKey("graphs.id"), nullable=False)
    computation_id = Column(UUID(as_uuid=True), ForeignKey("computations.id"))
    model_type = Column(String(50), nullable=False)  # GraphSAGE, GAT, etc.
    model_params = Column(JSON)
    embedding_dim = Column(Integer)
    embeddings_data = Column(Text)  # Serialized embeddings
    encrypted_embeddings = Column(Text)  # Encrypted version
    is_encrypted = Column(Boolean, default=False)
    noise_budget = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    graph = relationship("Graph", back_populates="embeddings")
    computation = relationship("Computation", back_populates="embeddings")
    
    __table_args__ = (
        Index("idx_embedding_graph", "graph_id"),
        Index("idx_embedding_computation", "computation_id"),
        Index("idx_embedding_created", "created_at"),
    )

class Computation(Base):
    """Homomorphic computation tracking"""
    __tablename__ = "computations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    graph_id = Column(UUID(as_uuid=True), ForeignKey("graphs.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    operation_type = Column(String(100), nullable=False)
    status = Column(String(50), default=EncryptionStatus.PENDING.value)
    parameters = Column(JSON)
    
    # CKKS parameters
    poly_modulus_degree = Column(Integer)
    scale_bits = Column(Integer)
    security_level = Column(Integer)
    
    # Performance metrics
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    gpu_time_ms = Column(Float)
    cpu_time_ms = Column(Float)
    memory_usage_mb = Column(Float)
    noise_consumed = Column(Float)
    
    # Results
    result = Column(JSON)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    graph = relationship("Graph", back_populates="computations")
    user = relationship("User", back_populates="computations")
    embeddings = relationship("Embedding", back_populates="computation")
    
    __table_args__ = (
        Index("idx_computation_user", "user_id"),
        Index("idx_computation_graph", "graph_id"),
        Index("idx_computation_status", "status"),
        Index("idx_computation_created", "created_at"),
    )

class User(Base):
    """User account management"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255))
    organization = Column(String(255))
    api_key = Column(String(255), unique=True, index=True)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    
    # Usage limits
    max_graphs = Column(Integer, default=10)
    max_nodes_per_graph = Column(Integer, default=100000)
    max_computations_per_day = Column(Integer, default=100)
    
    # Usage tracking
    total_computations = Column(Integer, default=0)
    total_gpu_time_ms = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    graphs = relationship("Graph", back_populates="owner")
    computations = relationship("Computation", back_populates="user")
    keys = relationship("EncryptionKey", back_populates="user")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "organization": self.organization,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }

class EncryptionKey(Base):
    """Encryption key management"""
    __tablename__ = "encryption_keys"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    key_type = Column(String(50), nullable=False)  # public, relin, galois
    key_data = Column(Text)  # Base64 encoded key
    parameters = Column(JSON)  # CKKS parameters used
    fingerprint = Column(String(255), unique=True, index=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="keys")
    
    __table_args__ = (
        Index("idx_key_user", "user_id"),
        Index("idx_key_type", "key_type"),
        Index("idx_key_created", "created_at"),
    )

class AuditLog(Base):
    """Audit trail for security-sensitive operations"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    details = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_action", "action"),
        Index("idx_audit_timestamp", "timestamp"),
    )

# Data classes for non-ORM operations
@dataclass
class GraphData:
    """In-memory graph representation"""
    id: str
    nodes: List[Dict[str, Any]]
    edges: List[Tuple[int, int]]
    node_features: Optional[Any] = None
    edge_features: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComputationRequest:
    """Request for homomorphic computation"""
    graph_id: str
    operation: str
    model_type: str
    model_params: Dict[str, Any]
    encryption_params: Dict[str, Any]
    user_id: str

@dataclass
class ComputationResult:
    """Result of homomorphic computation"""
    computation_id: str
    status: EncryptionStatus
    embeddings: Optional[Any] = None
    encrypted_embeddings: Optional[str] = None
    noise_budget: Optional[float] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
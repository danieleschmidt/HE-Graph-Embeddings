"""
Autonomous SDLC System
Terragon Labs Enhanced Software Development Lifecycle Framework

This package provides a comprehensive autonomous SDLC implementation with:
- Progressive Quality Orchestration
- Intelligent Checkpoint Selection
- Self-Improving Quality Metrics
- Research-Driven Development
- Global Compliance Engine

Version: 1.0.0
"""

from .progressive_quality_orchestrator import (
    AdvancedProgressiveQualityOrchestrator,
    ExecutionResult,
    QualityMetric,
    SDLCGeneration,
    QualityGateStatus
)

from .intelligent_checkpoint_selector import (
    IntelligentCheckpointSelector,
    ExecutionPath,
    ProjectCharacteristics,
    CheckpointDefinition
)

from .self_improving_quality_metrics import (
    SelfImprovingQualityMetrics,
    QualityMeasurement,
    MetricType,
    AdaptiveThreshold
)

from .research_driven_development import (
    ResearchDrivenDevelopmentFramework,
    ResearchHypothesis,
    ExperimentalDesign,
    ExperimentalResult,
    ReplicationStudy
)

from .global_compliance_engine import (
    GlobalComplianceEngine,
    ComplianceAssessment,
    RegulationType,
    DataInventoryItem,
    DataSubjectRequest
)

__version__ = "1.0.0"
__author__ = "Terragon Labs"
__description__ = "Autonomous SDLC System with Progressive Enhancement"

__all__ = [
    # Main orchestrator
    "AdvancedProgressiveQualityOrchestrator",
    "ExecutionResult",
    "QualityMetric",
    "SDLCGeneration",
    "QualityGateStatus",
    
    # Checkpoint selection
    "IntelligentCheckpointSelector",
    "ExecutionPath",
    "ProjectCharacteristics",
    "CheckpointDefinition",
    
    # Quality metrics
    "SelfImprovingQualityMetrics",
    "QualityMeasurement",
    "MetricType",
    "AdaptiveThreshold",
    
    # Research framework
    "ResearchDrivenDevelopmentFramework",
    "ResearchHypothesis",
    "ExperimentalDesign",
    "ExperimentalResult",
    "ReplicationStudy",
    
    # Compliance engine
    "GlobalComplianceEngine",
    "ComplianceAssessment",
    "RegulationType",
    "DataInventoryItem",
    "DataSubjectRequest"
]
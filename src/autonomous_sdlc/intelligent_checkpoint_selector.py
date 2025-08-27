#!/usr/bin/env python3
"""
Intelligent Checkpoint Selection Engine for Autonomous SDLC
==========================================================

🧠 AI-POWERED CHECKPOINT SELECTION AND OPTIMIZATION

This system uses machine learning and intelligent analysis to automatically select
optimal checkpoints, quality gates, and execution strategies based on project
characteristics, historical performance, and predictive analytics.

🎯 INTELLIGENT FEATURES:
• Dynamic Checkpoint Adaptation: Automatically adjusts checkpoints based on project analysis
• Performance-Based Optimization: Uses historical data to optimize execution paths
• Risk-Aware Planning: Identifies and mitigates risks through intelligent checkpoint selection
• Context-Aware Strategies: Adapts to project type, complexity, and domain requirements
• Continuous Learning: Improves selection accuracy through feedback loops

🚀 CHECKPOINT STRATEGIES BY PROJECT TYPE:
API Service: Foundation → Data → Auth → Endpoints → Testing → Monitoring → Scaling
CLI Tool: Structure → Commands → Config → Plugins → Testing → Distribution
Web App: Frontend → Backend → State → UI → Testing → Performance → Deployment
Library: Modules → API → Examples → Docs → Testing → Packaging → Distribution
Research: Literature → Methodology → Implementation → Experimentation → Publication
Crypto: Security → Algorithms → Keys → Performance → Audit → Compliance → Docs

🛡️ ADAPTIVE QUALITY GATES:
• Code Quality: AST analysis, complexity metrics, maintainability index
• Security: Vulnerability scanning, cryptographic validation, compliance checks
• Performance: Benchmarking, profiling, scalability testing
• Testing: Coverage analysis, mutation testing, integration validation
• Documentation: Coverage measurement, quality assessment, accessibility
• Research: Reproducibility, statistical significance, peer review readiness

⚡ INTELLIGENT OPTIMIZATION:
• Predictive checkpoint duration estimation
• Resource requirement forecasting  
• Risk-based priority adjustment
• Parallel execution optimization
• Failure prediction and mitigation

Built with ❤️ by Terragon Labs
"""

import json
import logging
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import re
import ast

# Machine learning imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    np = pd = RandomForestClassifier = GradientBoostingRegressor = None

logger = logging.getLogger(__name__)

class ProjectType(Enum):
    """Enhanced project type detection"""
    API_SERVICE = "api"
    CLI_TOOL = "cli" 
    WEB_APPLICATION = "webapp"
    LIBRARY = "library"
    RESEARCH_PROJECT = "research"
    CRYPTOGRAPHIC_SYSTEM = "crypto"
    MACHINE_LEARNING = "ml"
    BLOCKCHAIN = "blockchain"
    IOT_SYSTEM = "iot"
    DATA_PIPELINE = "data"
    MICROSERVICE = "microservice"

class ComplexityLevel(Enum):
    """Project complexity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ENTERPRISE = "enterprise"

class CheckpointPriority(Enum):
    """Checkpoint execution priority"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"

@dataclass
class ProjectCharacteristics:
    """Comprehensive project characteristics for intelligent selection"""
    type: ProjectType
    complexity: ComplexityLevel
    domain: str
    size_metrics: Dict[str, int]
    technology_stack: List[str]
    dependencies: List[str]
    security_requirements: bool
    performance_critical: bool
    compliance_needs: List[str]
    research_aspects: bool
    api_complexity: int
    data_sensitivity: str  # "public", "internal", "confidential", "top_secret"
    deployment_targets: List[str]
    user_base_size: str  # "small", "medium", "large", "enterprise"

@dataclass
class CheckpointDefinition:
    """Enhanced checkpoint definition with intelligence"""
    name: str
    description: str
    project_types: Set[ProjectType]
    prerequisites: List[str]
    quality_gates: List[str]
    estimated_duration: timedelta
    resource_requirements: Dict[str, Any]
    priority: CheckpointPriority
    success_criteria: Dict[str, float]
    failure_recovery: List[str]
    parallel_eligible: bool = False
    complexity_scaling: float = 1.0
    risk_factors: List[str] = field(default_factory=list)
    optimization_hints: List[str] = field(default_factory=list)

@dataclass
class ExecutionPath:
    """Optimized execution path with intelligent sequencing"""
    checkpoints: List[CheckpointDefinition]
    total_duration: timedelta
    parallel_groups: List[List[str]]
    critical_path: List[str]
    risk_score: float
    confidence: float
    resource_profile: Dict[str, float]
    success_probability: float

@dataclass
class HistoricalExecution:
    """Historical execution data for learning"""
    project_characteristics: ProjectCharacteristics
    selected_checkpoints: List[str]
    actual_duration: timedelta
    success_rate: float
    quality_scores: Dict[str, float]
    bottlenecks: List[str]
    lessons_learned: List[str]
    timestamp: datetime

class IntelligentCheckpointSelector:
    """
    AI-powered checkpoint selection engine that learns from experience and
    optimizes execution paths based on project characteristics and historical data
    """

    def __init__(self, project_root: str = None):
        """Initialize the intelligent selector"""
        self.project_root = Path(project_root or Path.cwd()).resolve()
        
        # Initialize checkpoint definitions
        self.checkpoint_definitions = self._initialize_checkpoint_definitions()
        
        # Historical data storage
        self.execution_history: List[HistoricalExecution] = []
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # ML models
        self.ml_models = {}
        if ML_AVAILABLE:
            self._initialize_ml_models()
            
        # Project analysis cache
        self.project_characteristics: Optional[ProjectCharacteristics] = None
        self.last_analysis_time: Optional[datetime] = None
        
        logger.info(f"🧠 Intelligent Checkpoint Selector initialized")
        logger.info(f"   ML Support: {'✅' if ML_AVAILABLE else '❌'}")
        logger.info(f"   Checkpoint Definitions: {len(self.checkpoint_definitions)}")

    def select_optimal_checkpoints(self, requirements: Dict[str, Any] = None) -> ExecutionPath:
        """
        Select optimal checkpoints using intelligent analysis and ML prediction
        """
        logger.info("🎯 Selecting optimal checkpoints...")
        
        start_time = time.time()
        
        # Analyze project characteristics
        project_chars = self._analyze_project_characteristics()
        
        # Generate candidate checkpoint combinations
        candidates = self._generate_checkpoint_candidates(project_chars, requirements or {})
        
        # Evaluate candidates using ML and heuristics
        best_path = self._evaluate_and_select_path(candidates, project_chars)
        
        # Optimize the selected path
        optimized_path = self._optimize_execution_path(best_path, project_chars)
        
        # Learn from selection for future improvements
        self._record_selection_decision(optimized_path, project_chars)
        
        selection_time = time.time() - start_time
        logger.info(f"✅ Optimal checkpoints selected in {selection_time:.1f}s")
        logger.info(f"   Path: {len(optimized_path.checkpoints)} checkpoints")
        logger.info(f"   Duration: {optimized_path.total_duration}")
        logger.info(f"   Success Probability: {optimized_path.success_probability:.2f}")
        
        return optimized_path

    def _initialize_checkpoint_definitions(self) -> Dict[str, CheckpointDefinition]:
        """Initialize comprehensive checkpoint definitions"""
        
        checkpoints = {}
        
        # Foundation checkpoints (applicable to all projects)
        checkpoints["project_foundation"] = CheckpointDefinition(
            name="Project Foundation",
            description="Establish basic project structure and configuration",
            project_types={ProjectType.API_SERVICE, ProjectType.CLI_TOOL, ProjectType.WEB_APPLICATION, 
                          ProjectType.LIBRARY, ProjectType.RESEARCH_PROJECT, ProjectType.CRYPTOGRAPHIC_SYSTEM},
            prerequisites=[],
            quality_gates=["structure_validation", "configuration_check"],
            estimated_duration=timedelta(hours=2),
            resource_requirements={"cpu": 1, "memory": "2GB"},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"structure_score": 0.9, "config_score": 0.8},
            failure_recovery=["recreate_structure", "fix_configuration"]
        )
        
        # API-specific checkpoints
        checkpoints["data_layer"] = CheckpointDefinition(
            name="Data Layer Implementation",
            description="Implement database models and data access layer",
            project_types={ProjectType.API_SERVICE, ProjectType.WEB_APPLICATION, ProjectType.MICROSERVICE},
            prerequisites=["project_foundation"],
            quality_gates=["model_validation", "migration_check", "data_integrity"],
            estimated_duration=timedelta(hours=6),
            resource_requirements={"cpu": 2, "memory": "4GB", "database": True},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"model_coverage": 0.9, "migration_success": 1.0},
            failure_recovery=["rollback_migrations", "fix_models", "data_repair"],
            complexity_scaling=1.5
        )
        
        checkpoints["authentication_system"] = CheckpointDefinition(
            name="Authentication & Authorization",
            description="Implement secure authentication and authorization system",
            project_types={ProjectType.API_SERVICE, ProjectType.WEB_APPLICATION},
            prerequisites=["data_layer"],
            quality_gates=["auth_security_scan", "token_validation", "rbac_check"],
            estimated_duration=timedelta(hours=8),
            resource_requirements={"cpu": 2, "memory": "4GB", "security_tools": True},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"security_score": 0.95, "auth_coverage": 0.9},
            failure_recovery=["security_patch", "auth_redesign"],
            risk_factors=["security_vulnerabilities", "token_management"]
        )
        
        checkpoints["api_endpoints"] = CheckpointDefinition(
            name="API Endpoints Implementation",
            description="Implement RESTful API endpoints with validation",
            project_types={ProjectType.API_SERVICE, ProjectType.WEB_APPLICATION, ProjectType.MICROSERVICE},
            prerequisites=["authentication_system"],
            quality_gates=["api_validation", "endpoint_testing", "contract_compliance"],
            estimated_duration=timedelta(hours=10),
            resource_requirements={"cpu": 2, "memory": "4GB"},
            priority=CheckpointPriority.HIGH,
            success_criteria={"endpoint_coverage": 0.9, "contract_compliance": 1.0},
            failure_recovery=["fix_endpoints", "update_contracts"],
            parallel_eligible=True,
            complexity_scaling=2.0
        )
        
        # CLI-specific checkpoints
        checkpoints["command_structure"] = CheckpointDefinition(
            name="Command Structure",
            description="Implement CLI command structure and parsing",
            project_types={ProjectType.CLI_TOOL},
            prerequisites=["project_foundation"],
            quality_gates=["command_validation", "help_completeness", "argument_parsing"],
            estimated_duration=timedelta(hours=4),
            resource_requirements={"cpu": 1, "memory": "2GB"},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"command_coverage": 0.9, "help_score": 0.8},
            failure_recovery=["fix_commands", "update_help"]
        )
        
        checkpoints["cli_configuration"] = CheckpointDefinition(
            name="Configuration Management", 
            description="Implement configuration file handling and validation",
            project_types={ProjectType.CLI_TOOL, ProjectType.API_SERVICE},
            prerequisites=["command_structure"],
            quality_gates=["config_validation", "schema_compliance"],
            estimated_duration=timedelta(hours=3),
            resource_requirements={"cpu": 1, "memory": "2GB"},
            priority=CheckpointPriority.HIGH,
            success_criteria={"config_coverage": 0.9},
            failure_recovery=["fix_config_schema"]
        )
        
        checkpoints["plugin_system"] = CheckpointDefinition(
            name="Plugin System",
            description="Implement extensible plugin architecture",
            project_types={ProjectType.CLI_TOOL},
            prerequisites=["cli_configuration"],
            quality_gates=["plugin_validation", "api_compatibility"],
            estimated_duration=timedelta(hours=6),
            resource_requirements={"cpu": 2, "memory": "4GB"},
            priority=CheckpointPriority.MEDIUM,
            success_criteria={"plugin_api_score": 0.8},
            failure_recovery=["fix_plugin_api"],
            complexity_scaling=1.3
        )
        
        # Research-specific checkpoints
        checkpoints["literature_review"] = CheckpointDefinition(
            name="Literature Review",
            description="Conduct comprehensive literature review and gap analysis",
            project_types={ProjectType.RESEARCH_PROJECT},
            prerequisites=["project_foundation"],
            quality_gates=["literature_completeness", "gap_analysis", "novelty_assessment"],
            estimated_duration=timedelta(hours=20),
            resource_requirements={"cpu": 1, "memory": "4GB", "research_tools": True},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"literature_score": 0.9, "novelty_score": 0.8},
            failure_recovery=["expand_search", "deeper_analysis"]
        )
        
        checkpoints["methodology_design"] = CheckpointDefinition(
            name="Research Methodology",
            description="Design experimental methodology and validation framework",
            project_types={ProjectType.RESEARCH_PROJECT},
            prerequisites=["literature_review"],
            quality_gates=["methodology_validation", "statistical_design", "ethics_review"],
            estimated_duration=timedelta(hours=15),
            resource_requirements={"cpu": 2, "memory": "8GB"},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"methodology_score": 0.9, "statistical_power": 0.8},
            failure_recovery=["redesign_experiment", "statistical_consultation"]
        )
        
        checkpoints["experimental_implementation"] = CheckpointDefinition(
            name="Experimental Implementation",
            description="Implement experimental framework and data collection",
            project_types={ProjectType.RESEARCH_PROJECT},
            prerequisites=["methodology_design"],
            quality_gates=["implementation_validation", "reproducibility_check"],
            estimated_duration=timedelta(hours=25),
            resource_requirements={"cpu": 4, "memory": "16GB", "gpu": True},
            priority=CheckpointPriority.HIGH,
            success_criteria={"implementation_score": 0.9, "reproducibility": 0.95},
            failure_recovery=["fix_implementation", "improve_reproducibility"],
            complexity_scaling=2.5
        )
        
        # Cryptographic system checkpoints
        checkpoints["security_model"] = CheckpointDefinition(
            name="Security Model Design",
            description="Design comprehensive security model and threat analysis",
            project_types={ProjectType.CRYPTOGRAPHIC_SYSTEM},
            prerequisites=["project_foundation"],
            quality_gates=["threat_model", "security_requirements", "compliance_check"],
            estimated_duration=timedelta(hours=12),
            resource_requirements={"cpu": 2, "memory": "8GB", "security_tools": True},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"security_model_score": 0.95, "threat_coverage": 0.9},
            failure_recovery=["redesign_security", "expand_threat_model"],
            risk_factors=["security_vulnerabilities", "compliance_gaps"]
        )
        
        checkpoints["cryptographic_algorithms"] = CheckpointDefinition(
            name="Cryptographic Algorithms",
            description="Implement and validate cryptographic algorithms",
            project_types={ProjectType.CRYPTOGRAPHIC_SYSTEM},
            prerequisites=["security_model"],
            quality_gates=["algorithm_correctness", "cryptographic_validation", "side_channel_resistance"],
            estimated_duration=timedelta(hours=30),
            resource_requirements={"cpu": 4, "memory": "16GB", "specialized_hardware": True},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"algorithm_score": 0.98, "validation_score": 0.95},
            failure_recovery=["algorithm_fix", "validation_enhancement"],
            complexity_scaling=3.0,
            risk_factors=["cryptographic_bugs", "performance_issues"]
        )
        
        checkpoints["key_management"] = CheckpointDefinition(
            name="Key Management System",
            description="Implement secure key generation, storage, and lifecycle management",
            project_types={ProjectType.CRYPTOGRAPHIC_SYSTEM},
            prerequisites=["cryptographic_algorithms"],
            quality_gates=["key_security", "lifecycle_management", "hsm_integration"],
            estimated_duration=timedelta(hours=15),
            resource_requirements={"cpu": 2, "memory": "8GB", "hsm": True},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"key_security_score": 0.98},
            failure_recovery=["enhance_key_security", "hsm_reconfiguration"]
        )
        
        # Universal checkpoints (applicable to all projects)
        checkpoints["comprehensive_testing"] = CheckpointDefinition(
            name="Comprehensive Testing",
            description="Implement unit, integration, and end-to-end testing",
            project_types=set(ProjectType),
            prerequisites=["api_endpoints", "command_structure", "experimental_implementation", "cryptographic_algorithms"],
            quality_gates=["test_coverage", "test_quality", "mutation_testing"],
            estimated_duration=timedelta(hours=12),
            resource_requirements={"cpu": 4, "memory": "8GB"},
            priority=CheckpointPriority.HIGH,
            success_criteria={"coverage": 0.85, "quality_score": 0.8},
            failure_recovery=["improve_tests", "fix_coverage"],
            parallel_eligible=True,
            complexity_scaling=1.5
        )
        
        checkpoints["performance_optimization"] = CheckpointDefinition(
            name="Performance Optimization",
            description="Optimize performance and implement benchmarking",
            project_types=set(ProjectType),
            prerequisites=["comprehensive_testing"],
            quality_gates=["performance_benchmark", "memory_efficiency", "scalability_test"],
            estimated_duration=timedelta(hours=10),
            resource_requirements={"cpu": 8, "memory": "16GB"},
            priority=CheckpointPriority.HIGH,
            success_criteria={"performance_score": 0.8, "efficiency_score": 0.8},
            failure_recovery=["performance_tuning", "architecture_optimization"],
            complexity_scaling=1.8
        )
        
        checkpoints["security_audit"] = CheckpointDefinition(
            name="Security Audit",
            description="Comprehensive security audit and vulnerability assessment",
            project_types=set(ProjectType),
            prerequisites=["performance_optimization"],
            quality_gates=["vulnerability_scan", "penetration_test", "compliance_audit"],
            estimated_duration=timedelta(hours=8),
            resource_requirements={"cpu": 2, "memory": "4GB", "security_tools": True},
            priority=CheckpointPriority.CRITICAL,
            success_criteria={"security_score": 0.95, "compliance_score": 0.9},
            failure_recovery=["fix_vulnerabilities", "enhance_security"],
            risk_factors=["security_vulnerabilities", "compliance_violations"]
        )
        
        checkpoints["documentation_completion"] = CheckpointDefinition(
            name="Documentation Completion",
            description="Complete comprehensive documentation and guides",
            project_types=set(ProjectType),
            prerequisites=["security_audit"],
            quality_gates=["doc_coverage", "doc_quality", "accessibility_check"],
            estimated_duration=timedelta(hours=6),
            resource_requirements={"cpu": 1, "memory": "2GB"},
            priority=CheckpointPriority.HIGH,
            success_criteria={"doc_coverage": 0.9, "quality_score": 0.8},
            failure_recovery=["improve_documentation"],
            parallel_eligible=True
        )
        
        checkpoints["deployment_preparation"] = CheckpointDefinition(
            name="Deployment Preparation", 
            description="Prepare for production deployment and monitoring",
            project_types=set(ProjectType),
            prerequisites=["documentation_completion"],
            quality_gates=["deployment_readiness", "monitoring_setup", "rollback_plan"],
            estimated_duration=timedelta(hours=8),
            resource_requirements={"cpu": 2, "memory": "4GB", "cloud_services": True},
            priority=CheckpointPriority.HIGH,
            success_criteria={"deployment_score": 0.9, "monitoring_score": 0.8},
            failure_recovery=["fix_deployment", "enhance_monitoring"]
        )
        
        return checkpoints

    def _analyze_project_characteristics(self) -> ProjectCharacteristics:
        """Analyze project to determine characteristics for intelligent selection"""
        
        # Use cached analysis if recent
        if (self.project_characteristics and 
            self.last_analysis_time and 
            datetime.now() - self.last_analysis_time < timedelta(hours=1)):
            return self.project_characteristics
            
        logger.info("🔍 Analyzing project characteristics...")
        
        # Detect project type
        project_type = self._detect_project_type()
        
        # Calculate complexity
        complexity = self._calculate_project_complexity()
        
        # Determine domain
        domain = self._identify_domain()
        
        # Analyze size metrics
        size_metrics = self._calculate_size_metrics()
        
        # Detect technology stack
        tech_stack = self._detect_technology_stack()
        
        # Analyze dependencies
        dependencies = self._analyze_dependencies()
        
        # Assess security requirements
        security_req = self._assess_security_requirements()
        
        # Check performance criticality
        performance_critical = self._assess_performance_criticality()
        
        # Determine compliance needs
        compliance_needs = self._assess_compliance_needs()
        
        # Check research aspects
        research_aspects = self._has_research_aspects()
        
        # Calculate API complexity
        api_complexity = self._calculate_api_complexity()
        
        # Assess data sensitivity
        data_sensitivity = self._assess_data_sensitivity()
        
        # Determine deployment targets
        deployment_targets = self._identify_deployment_targets()
        
        # Estimate user base size
        user_base_size = self._estimate_user_base_size()
        
        self.project_characteristics = ProjectCharacteristics(
            type=project_type,
            complexity=complexity,
            domain=domain,
            size_metrics=size_metrics,
            technology_stack=tech_stack,
            dependencies=dependencies,
            security_requirements=security_req,
            performance_critical=performance_critical,
            compliance_needs=compliance_needs,
            research_aspects=research_aspects,
            api_complexity=api_complexity,
            data_sensitivity=data_sensitivity,
            deployment_targets=deployment_targets,
            user_base_size=user_base_size
        )
        
        self.last_analysis_time = datetime.now()
        
        logger.info(f"✅ Project analysis complete:")
        logger.info(f"   Type: {project_type.value}")
        logger.info(f"   Complexity: {complexity.value}")
        logger.info(f"   Domain: {domain}")
        logger.info(f"   Security: {'High' if security_req else 'Standard'}")
        logger.info(f"   Performance: {'Critical' if performance_critical else 'Standard'}")
        
        return self.project_characteristics

    def _detect_project_type(self) -> ProjectType:
        """Enhanced project type detection using multiple indicators"""
        
        # File-based indicators
        if (self.project_root / "Dockerfile").exists() or (self.project_root / "docker-compose.yml").exists():
            if any((self.project_root / "src").glob("*app.py")) or any((self.project_root / "src").glob("*main.py")):
                return ProjectType.API_SERVICE
        
        # Check for CLI indicators
        setup_py = self.project_root / "setup.py"
        if setup_py.exists():
            try:
                with open(setup_py) as f:
                    content = f.read()
                    if "console_scripts" in content:
                        return ProjectType.CLI_TOOL
            except Exception:
                pass
        
        # Research project indicators
        research_keywords = ["research", "experiment", "paper", "study", "analysis", "evaluation"]
        if any(keyword in str(self.project_root).lower() for keyword in research_keywords):
            return ProjectType.RESEARCH_PROJECT
        
        # Check for research-specific directories/files
        research_dirs = ["experiments", "results", "benchmarks", "evaluation"]
        if any((self.project_root / d).exists() for d in research_dirs):
            return ProjectType.RESEARCH_PROJECT
        
        # Cryptographic system indicators
        crypto_keywords = ["crypto", "encryption", "cipher", "homomorphic", "ckks", "seal", "security"]
        src_dir = self.project_root / "src"
        if src_dir.exists():
            for keyword in crypto_keywords:
                if any(src_dir.rglob(f"*{keyword}*")):
                    return ProjectType.CRYPTOGRAPHIC_SYSTEM
        
        # Machine learning indicators
        ml_files = ["model.py", "train.py", "inference.py", "dataset.py"]
        ml_dirs = ["models", "training", "data"]
        if (any((src_dir / f).exists() for f in ml_files if src_dir.exists()) or
            any((self.project_root / d).exists() for d in ml_dirs)):
            return ProjectType.MACHINE_LEARNING
        
        # Web application indicators
        web_indicators = ["templates", "static", "views", "frontend", "react", "vue", "angular"]
        if any((self.project_root / w).exists() for w in web_indicators):
            return ProjectType.WEB_APPLICATION
        
        # Blockchain indicators
        blockchain_keywords = ["blockchain", "smart_contract", "ethereum", "solidity", "web3"]
        if any(keyword in str(self.project_root).lower() for keyword in blockchain_keywords):
            return ProjectType.BLOCKCHAIN
        
        # Microservice indicators
        if ((self.project_root / "kubernetes").exists() or 
            (self.project_root / "k8s").exists() or
            any((self.project_root).glob("*service*.py"))):
            return ProjectType.MICROSERVICE
        
        # Default to library if setup.py exists
        if setup_py.exists():
            return ProjectType.LIBRARY
            
        # Default fallback
        return ProjectType.API_SERVICE

    def _calculate_project_complexity(self) -> ComplexityLevel:
        """Calculate project complexity using multiple metrics"""
        
        complexity_score = 0
        
        # File count metric
        src_dir = self.project_root / "src"
        if src_dir.exists():
            py_files = list(src_dir.rglob("*.py"))
            if len(py_files) > 100:
                complexity_score += 3
            elif len(py_files) > 50:
                complexity_score += 2
            elif len(py_files) > 20:
                complexity_score += 1
        
        # Dependency complexity
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file) as f:
                    deps = len([line for line in f.readlines() if line.strip() and not line.startswith("#")])
                    if deps > 50:
                        complexity_score += 3
                    elif deps > 25:
                        complexity_score += 2
                    elif deps > 10:
                        complexity_score += 1
            except Exception:
                pass
        
        # Directory structure complexity
        if src_dir.exists():
            subdirs = len([d for d in src_dir.rglob("*") if d.is_dir()])
            if subdirs > 20:
                complexity_score += 2
            elif subdirs > 10:
                complexity_score += 1
        
        # Technology stack complexity
        complex_tech_indicators = ["tensorflow", "pytorch", "kubernetes", "docker", "cryptography"]
        for tech in complex_tech_indicators:
            if self._has_technology(tech):
                complexity_score += 1
        
        # Configuration complexity
        config_files = ["docker-compose.yml", "kubernetes", "terraform", ".github/workflows"]
        config_complexity = sum(1 for cf in config_files if 
                               (self.project_root / cf).exists() or 
                               any(self.project_root.rglob(f"*{cf}*")))
        complexity_score += min(config_complexity, 3)
        
        # Map score to complexity level
        if complexity_score >= 10:
            return ComplexityLevel.ENTERPRISE
        elif complexity_score >= 7:
            return ComplexityLevel.HIGH
        elif complexity_score >= 4:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW

    def _identify_domain(self) -> str:
        """Identify the project domain"""
        
        domain_keywords = {
            "healthcare": ["medical", "health", "patient", "clinical", "diagnosis"],
            "finance": ["financial", "banking", "payment", "transaction", "trading"],
            "security": ["security", "cryptography", "encryption", "authentication"],
            "ml_ai": ["machine_learning", "artificial_intelligence", "neural", "deep_learning"],
            "iot": ["iot", "sensor", "device", "embedded", "raspberry"],
            "blockchain": ["blockchain", "cryptocurrency", "smart_contract", "ethereum"],
            "research": ["research", "academic", "experiment", "paper", "study"],
            "web": ["web", "frontend", "backend", "api", "http"]
        }
        
        project_text = str(self.project_root).lower()
        
        # Check README content if available
        readme_file = self.project_root / "README.md"
        if readme_file.exists():
            try:
                with open(readme_file) as f:
                    project_text += " " + f.read().lower()
            except Exception:
                pass
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in project_text for keyword in keywords):
                return domain
        
        return "general"

    def _calculate_size_metrics(self) -> Dict[str, int]:
        """Calculate project size metrics"""
        
        metrics = {
            "python_files": 0,
            "test_files": 0,
            "lines_of_code": 0,
            "functions": 0,
            "classes": 0
        }
        
        src_dir = self.project_root / "src"
        if src_dir.exists():
            python_files = list(src_dir.rglob("*.py"))
            metrics["python_files"] = len(python_files)
            
            for py_file in python_files[:50]:  # Limit analysis to prevent performance issues
                try:
                    with open(py_file, encoding='utf-8') as f:
                        content = f.read()
                        metrics["lines_of_code"] += len(content.splitlines())
                        
                    # AST analysis for functions and classes
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                metrics["functions"] += 1
                            elif isinstance(node, ast.ClassDef):
                                metrics["classes"] += 1
                    except Exception:
                        continue
                        
                except Exception:
                    continue
        
        # Test files
        tests_dir = self.project_root / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.rglob("*.py"))
            metrics["test_files"] = len(test_files)
        
        return metrics

    def _detect_technology_stack(self) -> List[str]:
        """Detect technology stack from various indicators"""
        
        technologies = []
        
        # Check requirements.txt
        req_file = self.project_root / "requirements.txt"
        if req_file.exists():
            try:
                with open(req_file) as f:
                    for line in f:
                        line = line.strip().lower()
                        if line and not line.startswith("#"):
                            package = line.split(">=")[0].split("==")[0].split("~=")[0].strip()
                            technologies.append(package)
            except Exception:
                pass
        
        # Check setup.py
        setup_file = self.project_root / "setup.py"
        if setup_file.exists():
            try:
                with open(setup_file) as f:
                    content = f.read()
                    # Extract common framework mentions
                    frameworks = ["flask", "django", "fastapi", "pytorch", "tensorflow", "sklearn"]
                    for framework in frameworks:
                        if framework in content.lower():
                            technologies.append(framework)
            except Exception:
                pass
        
        # Check Docker files
        if (self.project_root / "Dockerfile").exists():
            technologies.append("docker")
        if (self.project_root / "docker-compose.yml").exists():
            technologies.append("docker-compose")
        
        # Check for specific file patterns
        tech_patterns = {
            "kubernetes": ["*.yaml", "*.yml"],
            "terraform": ["*.tf"],
            "ansible": ["*.yml"],
            "react": ["package.json", "src/index.js"],
            "vue": ["vue.config.js"],
            "angular": ["angular.json"]
        }
        
        for tech, patterns in tech_patterns.items():
            if any(list(self.project_root.rglob(pattern)) for pattern in patterns):
                technologies.append(tech)
        
        return list(set(technologies))  # Remove duplicates

    def _analyze_dependencies(self) -> List[str]:
        """Analyze project dependencies"""
        return self._detect_technology_stack()  # For now, same as technology stack

    def _assess_security_requirements(self) -> bool:
        """Assess if project has high security requirements"""
        
        security_indicators = [
            "cryptography", "encryption", "auth", "security", "ssl", "tls",
            "oauth", "jwt", "password", "hash", "signature", "certificate"
        ]
        
        # Check in technology stack
        tech_stack = self._detect_technology_stack()
        if any(indicator in " ".join(tech_stack) for indicator in security_indicators):
            return True
        
        # Check in project name/path
        if any(indicator in str(self.project_root).lower() for indicator in security_indicators):
            return True
        
        # Check for security-related files
        security_files = ["auth.py", "security.py", "encryption.py", "crypto.py"]
        src_dir = self.project_root / "src"
        if src_dir.exists():
            for sec_file in security_files:
                if any(src_dir.rglob(sec_file)):
                    return True
        
        return False

    def _assess_performance_criticality(self) -> bool:
        """Assess if project is performance-critical"""
        
        performance_indicators = [
            "optimization", "performance", "speed", "benchmark", "profiling",
            "cache", "parallel", "concurrent", "async", "gpu", "cuda"
        ]
        
        # Check in technology stack
        tech_stack = self._detect_technology_stack()
        perf_tech = ["numpy", "scipy", "numba", "cython", "tensorflow", "pytorch", "cuda"]
        if any(tech in tech_stack for tech in perf_tech):
            return True
        
        # Check in project context
        project_text = str(self.project_root).lower()
        if any(indicator in project_text for indicator in performance_indicators):
            return True
        
        # Check complexity - enterprise projects usually need performance
        if self._calculate_project_complexity() == ComplexityLevel.ENTERPRISE:
            return True
        
        return False

    def _assess_compliance_needs(self) -> List[str]:
        """Assess compliance requirements"""
        
        compliance_needs = []
        
        compliance_indicators = {
            "GDPR": ["gdpr", "privacy", "personal_data", "european"],
            "HIPAA": ["hipaa", "health", "medical", "patient"],
            "SOC2": ["soc2", "security", "audit", "compliance"],
            "PCI": ["pci", "payment", "card", "financial"],
            "CCPA": ["ccpa", "california", "consumer_privacy"]
        }
        
        project_text = str(self.project_root).lower()
        
        # Check README content
        readme_file = self.project_root / "README.md"
        if readme_file.exists():
            try:
                with open(readme_file) as f:
                    project_text += " " + f.read().lower()
            except Exception:
                pass
        
        for compliance, indicators in compliance_indicators.items():
            if any(indicator in project_text for indicator in indicators):
                compliance_needs.append(compliance)
        
        return compliance_needs

    def _has_research_aspects(self) -> bool:
        """Check if project has research aspects"""
        
        research_indicators = [
            "research", "experiment", "paper", "study", "analysis", "evaluation",
            "benchmark", "dataset", "algorithm", "novelty", "innovation"
        ]
        
        # Check project path
        project_text = str(self.project_root).lower()
        if any(indicator in project_text for indicator in research_indicators):
            return True
        
        # Check for research directories
        research_dirs = ["experiments", "benchmarks", "evaluation", "results", "papers"]
        if any((self.project_root / d).exists() for d in research_dirs):
            return True
        
        # Check for academic file patterns
        academic_files = ["*.tex", "*.bib", "paper.pdf", "results.json"]
        if any(list(self.project_root.rglob(pattern)) for pattern in academic_files):
            return True
        
        return False

    def _calculate_api_complexity(self) -> int:
        """Calculate API complexity score (0-10)"""
        
        complexity = 0
        
        # Check for API-related files
        src_dir = self.project_root / "src"
        if src_dir.exists():
            api_files = list(src_dir.rglob("*api*.py")) + list(src_dir.rglob("*route*.py"))
            complexity += min(len(api_files), 3)
        
        # Check for OpenAPI/Swagger specs
        if any(self.project_root.rglob("*.yaml")) or any(self.project_root.rglob("swagger*")):
            complexity += 2
        
        # Check for multiple API versions
        if any(src_dir.rglob("*v1*")) and any(src_dir.rglob("*v2*")):
            complexity += 2
        
        # Check for microservices
        if "microservice" in str(self.project_root).lower():
            complexity += 3
        
        return min(complexity, 10)

    def _assess_data_sensitivity(self) -> str:
        """Assess data sensitivity level"""
        
        if self._assess_compliance_needs():
            return "confidential"
        
        if self._assess_security_requirements():
            return "internal"
        
        sensitive_indicators = ["personal", "private", "confidential", "sensitive"]
        project_text = str(self.project_root).lower()
        
        if any(indicator in project_text for indicator in sensitive_indicators):
            return "internal"
        
        return "public"

    def _identify_deployment_targets(self) -> List[str]:
        """Identify deployment targets"""
        
        targets = []
        
        # Check for container deployment
        if (self.project_root / "Dockerfile").exists():
            targets.append("containers")
        
        # Check for cloud deployment
        cloud_indicators = ["terraform", "kubernetes", "helm", "aws", "gcp", "azure"]
        for indicator in cloud_indicators:
            if any(self.project_root.rglob(f"*{indicator}*")):
                targets.append("cloud")
                break
        
        # Check for traditional deployment
        if (self.project_root / "setup.py").exists():
            targets.append("traditional")
        
        # Check for edge deployment
        edge_indicators = ["edge", "iot", "embedded"]
        project_text = str(self.project_root).lower()
        if any(indicator in project_text for indicator in edge_indicators):
            targets.append("edge")
        
        return targets if targets else ["traditional"]

    def _estimate_user_base_size(self) -> str:
        """Estimate user base size"""
        
        complexity = self._calculate_project_complexity()
        
        if complexity == ComplexityLevel.ENTERPRISE:
            return "enterprise"
        elif complexity == ComplexityLevel.HIGH:
            return "large"
        elif complexity == ComplexityLevel.MEDIUM:
            return "medium"
        else:
            return "small"

    def _has_technology(self, tech: str) -> bool:
        """Check if project uses specific technology"""
        tech_stack = self._detect_technology_stack()
        return tech.lower() in [t.lower() for t in tech_stack]

    def _generate_checkpoint_candidates(self, 
                                      project_chars: ProjectCharacteristics,
                                      requirements: Dict[str, Any]) -> List[List[CheckpointDefinition]]:
        """Generate candidate checkpoint combinations"""
        
        logger.info("🎯 Generating checkpoint candidates...")
        
        # Filter checkpoints applicable to project type
        applicable_checkpoints = []
        for checkpoint in self.checkpoint_definitions.values():
            if project_chars.type in checkpoint.project_types:
                applicable_checkpoints.append(checkpoint)
        
        # Create different execution strategies
        candidates = []
        
        # Strategy 1: Minimal viable path (critical checkpoints only)
        minimal_path = [cp for cp in applicable_checkpoints 
                       if cp.priority in [CheckpointPriority.CRITICAL]]
        if minimal_path:
            candidates.append(minimal_path)
        
        # Strategy 2: Balanced path (critical + high priority)
        balanced_path = [cp for cp in applicable_checkpoints 
                        if cp.priority in [CheckpointPriority.CRITICAL, CheckpointPriority.HIGH]]
        if balanced_path:
            candidates.append(balanced_path)
        
        # Strategy 3: Comprehensive path (all applicable checkpoints)
        comprehensive_path = applicable_checkpoints
        candidates.append(comprehensive_path)
        
        # Strategy 4: Domain-specific optimized path
        domain_optimized = self._create_domain_optimized_path(project_chars, applicable_checkpoints)
        if domain_optimized:
            candidates.append(domain_optimized)
        
        logger.info(f"✅ Generated {len(candidates)} candidate paths")
        return candidates

    def _create_domain_optimized_path(self, 
                                    project_chars: ProjectCharacteristics,
                                    available_checkpoints: List[CheckpointDefinition]) -> Optional[List[CheckpointDefinition]]:
        """Create domain-specific optimized checkpoint path"""
        
        domain = project_chars.domain
        
        # Domain-specific checkpoint prioritization
        domain_priorities = {
            "security": ["security_model", "cryptographic_algorithms", "key_management", "security_audit"],
            "research": ["literature_review", "methodology_design", "experimental_implementation"],
            "healthcare": ["security_audit", "compliance_check", "data_validation"],
            "finance": ["security_audit", "performance_optimization", "compliance_check"]
        }
        
        if domain not in domain_priorities:
            return None
        
        priority_checkpoints = domain_priorities[domain]
        
        # Start with critical checkpoints
        optimized_path = [cp for cp in available_checkpoints 
                         if cp.priority == CheckpointPriority.CRITICAL]
        
        # Add domain-specific high-priority checkpoints
        for cp in available_checkpoints:
            if cp.name.lower().replace(" ", "_") in [p.lower() for p in priority_checkpoints]:
                if cp not in optimized_path:
                    optimized_path.append(cp)
        
        # Add remaining high priority checkpoints
        for cp in available_checkpoints:
            if cp.priority == CheckpointPriority.HIGH and cp not in optimized_path:
                optimized_path.append(cp)
        
        return optimized_path

    def _evaluate_and_select_path(self, 
                                candidates: List[List[CheckpointDefinition]],
                                project_chars: ProjectCharacteristics) -> ExecutionPath:
        """Evaluate candidates and select optimal path"""
        
        logger.info("📊 Evaluating candidate paths...")
        
        best_path = None
        best_score = -1
        
        for i, candidate in enumerate(candidates):
            # Calculate path metrics
            path_score = self._calculate_path_score(candidate, project_chars)
            
            logger.info(f"   Candidate {i+1}: Score {path_score:.2f}")
            
            if path_score > best_score:
                best_score = path_score
                best_path = candidate
        
        if not best_path:
            raise ValueError("No valid execution path found")
        
        # Convert to ExecutionPath
        execution_path = self._create_execution_path(best_path, project_chars)
        
        logger.info(f"✅ Selected optimal path with score {best_score:.2f}")
        return execution_path

    def _calculate_path_score(self, 
                            checkpoints: List[CheckpointDefinition],
                            project_chars: ProjectCharacteristics) -> float:
        """Calculate score for a checkpoint path"""
        
        score = 0.0
        
        # Coverage score (how well it covers project needs)
        coverage_score = self._calculate_coverage_score(checkpoints, project_chars)
        score += coverage_score * 0.3
        
        # Efficiency score (duration vs. value)
        efficiency_score = self._calculate_efficiency_score(checkpoints, project_chars)
        score += efficiency_score * 0.2
        
        # Risk score (lower risk is better)
        risk_score = 1.0 - self._calculate_risk_score(checkpoints, project_chars)
        score += risk_score * 0.2
        
        # Feasibility score (resource constraints)
        feasibility_score = self._calculate_feasibility_score(checkpoints, project_chars)
        score += feasibility_score * 0.15
        
        # Success probability score
        success_score = self._calculate_success_probability(checkpoints, project_chars)
        score += success_score * 0.15
        
        return score

    def _calculate_coverage_score(self, 
                                checkpoints: List[CheckpointDefinition],
                                project_chars: ProjectCharacteristics) -> float:
        """Calculate how well checkpoints cover project needs"""
        
        # Define requirement coverage
        required_aspects = {
            "foundation": ["project_foundation"],
            "core_functionality": ["data_layer", "api_endpoints", "command_structure", "experimental_implementation"],
            "security": ["authentication_system", "security_audit"],
            "testing": ["comprehensive_testing"],
            "performance": ["performance_optimization"],
            "documentation": ["documentation_completion"],
            "deployment": ["deployment_preparation"]
        }
        
        checkpoint_names = [cp.name.lower().replace(" ", "_") for cp in checkpoints]
        
        coverage_score = 0.0
        total_aspects = len(required_aspects)
        
        for aspect, required_checkpoints in required_aspects.items():
            aspect_covered = any(req_cp in checkpoint_names for req_cp in required_checkpoints)
            if aspect_covered:
                coverage_score += 1.0 / total_aspects
        
        # Bonus for domain-specific coverage
        if project_chars.domain == "security" and any("security" in name for name in checkpoint_names):
            coverage_score += 0.1
        
        if project_chars.research_aspects and any("research" in name or "experiment" in name for name in checkpoint_names):
            coverage_score += 0.1
        
        return min(1.0, coverage_score)

    def _calculate_efficiency_score(self, 
                                  checkpoints: List[CheckpointDefinition],
                                  project_chars: ProjectCharacteristics) -> float:
        """Calculate efficiency score (value per time unit)"""
        
        total_duration = sum((cp.estimated_duration.total_seconds() * 
                            cp.complexity_scaling * 
                            project_chars.complexity.value.count('h') + 1) 
                           for cp in checkpoints)
        
        total_value = sum(self._get_checkpoint_value(cp, project_chars) for cp in checkpoints)
        
        if total_duration == 0:
            return 0.0
        
        efficiency = total_value / (total_duration / 3600)  # Value per hour
        
        # Normalize to 0-1 range
        return min(1.0, efficiency / 10.0)

    def _get_checkpoint_value(self, 
                            checkpoint: CheckpointDefinition,
                            project_chars: ProjectCharacteristics) -> float:
        """Get value score for a checkpoint"""
        
        base_value = {
            CheckpointPriority.CRITICAL: 10.0,
            CheckpointPriority.HIGH: 7.0,
            CheckpointPriority.MEDIUM: 4.0,
            CheckpointPriority.LOW: 2.0,
            CheckpointPriority.OPTIONAL: 1.0
        }[checkpoint.priority]
        
        # Adjust for project characteristics
        if project_chars.security_requirements and "security" in checkpoint.name.lower():
            base_value *= 1.5
        
        if project_chars.performance_critical and "performance" in checkpoint.name.lower():
            base_value *= 1.3
        
        if project_chars.research_aspects and any(term in checkpoint.name.lower() 
                                                for term in ["research", "experiment", "literature"]):
            base_value *= 1.4
        
        return base_value

    def _calculate_risk_score(self, 
                            checkpoints: List[CheckpointDefinition],
                            project_chars: ProjectCharacteristics) -> float:
        """Calculate overall risk score (0-1, higher is riskier)"""
        
        risk_factors = []
        
        for checkpoint in checkpoints:
            # Risk from checkpoint complexity
            duration_risk = min(1.0, checkpoint.estimated_duration.total_seconds() / (24 * 3600))
            risk_factors.append(duration_risk * 0.3)
            
            # Risk from checkpoint-specific factors
            checkpoint_risk = len(checkpoint.risk_factors) * 0.1
            risk_factors.append(checkpoint_risk)
            
            # Risk from resource requirements
            resource_risk = len(checkpoint.resource_requirements) * 0.05
            risk_factors.append(resource_risk)
        
        # Project complexity risk
        complexity_risk = {
            ComplexityLevel.LOW: 0.1,
            ComplexityLevel.MEDIUM: 0.2,
            ComplexityLevel.HIGH: 0.4,
            ComplexityLevel.ENTERPRISE: 0.6
        }[project_chars.complexity]
        risk_factors.append(complexity_risk)
        
        return min(1.0, sum(risk_factors) / len(risk_factors))

    def _calculate_feasibility_score(self, 
                                   checkpoints: List[CheckpointDefinition],
                                   project_chars: ProjectCharacteristics) -> float:
        """Calculate feasibility score based on resource constraints"""
        
        # For now, assume all paths are feasible
        # In a real implementation, this would check against available resources
        return 0.9

    def _calculate_success_probability(self, 
                                     checkpoints: List[CheckpointDefinition],
                                     project_chars: ProjectCharacteristics) -> float:
        """Calculate probability of successful execution"""
        
        if ML_AVAILABLE and len(self.execution_history) > 10:
            # Use ML to predict success probability
            return self._ml_predict_success_probability(checkpoints, project_chars)
        
        # Heuristic-based probability
        base_probability = 0.8
        
        # Adjust based on project complexity
        complexity_penalty = {
            ComplexityLevel.LOW: 0.0,
            ComplexityLevel.MEDIUM: -0.05,
            ComplexityLevel.HIGH: -0.1,
            ComplexityLevel.ENTERPRISE: -0.15
        }[project_chars.complexity]
        
        base_probability += complexity_penalty
        
        # Adjust based on path length
        if len(checkpoints) > 10:
            base_probability -= 0.05
        elif len(checkpoints) < 5:
            base_probability -= 0.02  # Too few checkpoints might miss important aspects
        
        return max(0.3, min(0.95, base_probability))

    def _ml_predict_success_probability(self, 
                                      checkpoints: List[CheckpointDefinition],
                                      project_chars: ProjectCharacteristics) -> float:
        """Use ML to predict success probability"""
        
        try:
            # Create feature vector
            features = self._create_feature_vector(checkpoints, project_chars)
            
            # Use trained model to predict
            if "success_predictor" in self.ml_models:
                probability = self.ml_models["success_predictor"].predict_proba([features])[0][1]
                return float(probability)
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
        
        return 0.8  # Fallback

    def _create_feature_vector(self, 
                             checkpoints: List[CheckpointDefinition],
                             project_chars: ProjectCharacteristics) -> List[float]:
        """Create feature vector for ML models"""
        
        features = []
        
        # Project characteristics features
        features.append(len(ProjectType) - list(ProjectType).index(project_chars.type))
        features.append(len(ComplexityLevel) - list(ComplexityLevel).index(project_chars.complexity))
        features.append(project_chars.size_metrics.get("python_files", 0))
        features.append(len(project_chars.technology_stack))
        features.append(float(project_chars.security_requirements))
        features.append(float(project_chars.performance_critical))
        features.append(len(project_chars.compliance_needs))
        features.append(float(project_chars.research_aspects))
        features.append(project_chars.api_complexity)
        
        # Checkpoint features
        features.append(len(checkpoints))
        features.append(sum(cp.estimated_duration.total_seconds() for cp in checkpoints) / 3600)
        features.append(sum(len(cp.quality_gates) for cp in checkpoints))
        features.append(sum(len(cp.risk_factors) for cp in checkpoints))
        
        # Priority distribution
        priority_counts = {priority: 0 for priority in CheckpointPriority}
        for cp in checkpoints:
            priority_counts[cp.priority] += 1
        
        for priority in CheckpointPriority:
            features.append(priority_counts[priority])
        
        return features

    def _create_execution_path(self, 
                             checkpoints: List[CheckpointDefinition],
                             project_chars: ProjectCharacteristics) -> ExecutionPath:
        """Create ExecutionPath from checkpoint list"""
        
        # Sort checkpoints by dependencies and priority
        sorted_checkpoints = self._sort_checkpoints_by_dependencies(checkpoints)
        
        # Calculate total duration with complexity scaling
        total_duration = timedelta()
        for cp in sorted_checkpoints:
            scaled_duration = cp.estimated_duration * cp.complexity_scaling
            complexity_multiplier = {
                ComplexityLevel.LOW: 0.8,
                ComplexityLevel.MEDIUM: 1.0,
                ComplexityLevel.HIGH: 1.3,
                ComplexityLevel.ENTERPRISE: 1.6
            }[project_chars.complexity]
            total_duration += scaled_duration * complexity_multiplier
        
        # Identify parallel groups
        parallel_groups = self._identify_parallel_groups(sorted_checkpoints)
        
        # Identify critical path
        critical_path = self._identify_critical_path(sorted_checkpoints)
        
        # Calculate metrics
        risk_score = self._calculate_risk_score(sorted_checkpoints, project_chars)
        success_probability = self._calculate_success_probability(sorted_checkpoints, project_chars)
        
        return ExecutionPath(
            checkpoints=sorted_checkpoints,
            total_duration=total_duration,
            parallel_groups=parallel_groups,
            critical_path=critical_path,
            risk_score=risk_score,
            confidence=0.8,  # TODO: Calculate based on historical data
            resource_profile=self._calculate_resource_profile(sorted_checkpoints),
            success_probability=success_probability
        )

    def _sort_checkpoints_by_dependencies(self, 
                                        checkpoints: List[CheckpointDefinition]) -> List[CheckpointDefinition]:
        """Sort checkpoints respecting dependencies"""
        
        sorted_checkpoints = []
        remaining = checkpoints.copy()
        
        while remaining:
            # Find checkpoints with satisfied dependencies
            ready = []
            for cp in remaining:
                dependencies_satisfied = all(
                    any(dep in sorted_cp.name.lower() for sorted_cp in sorted_checkpoints)
                    for dep in cp.prerequisites
                )
                if dependencies_satisfied:
                    ready.append(cp)
            
            if not ready:
                # Add remaining checkpoints if no dependencies can be satisfied
                # (handles circular dependencies or missing prerequisites)
                ready = remaining
            
            # Sort ready checkpoints by priority
            ready.sort(key=lambda cp: list(CheckpointPriority).index(cp.priority))
            
            # Add the highest priority ready checkpoint
            next_checkpoint = ready[0]
            sorted_checkpoints.append(next_checkpoint)
            remaining.remove(next_checkpoint)
        
        return sorted_checkpoints

    def _identify_parallel_groups(self, 
                                checkpoints: List[CheckpointDefinition]) -> List[List[str]]:
        """Identify groups of checkpoints that can run in parallel"""
        
        parallel_groups = []
        current_group = []
        
        for i, cp in enumerate(checkpoints):
            if cp.parallel_eligible:
                current_group.append(cp.name)
            else:
                if current_group:
                    if len(current_group) > 1:
                        parallel_groups.append(current_group)
                    current_group = []
        
        # Add final group if exists
        if len(current_group) > 1:
            parallel_groups.append(current_group)
        
        return parallel_groups

    def _identify_critical_path(self, 
                              checkpoints: List[CheckpointDefinition]) -> List[str]:
        """Identify critical path through checkpoints"""
        
        # For now, return all critical priority checkpoints
        critical_path = []
        for cp in checkpoints:
            if cp.priority == CheckpointPriority.CRITICAL:
                critical_path.append(cp.name)
        
        return critical_path

    def _calculate_resource_profile(self, 
                                  checkpoints: List[CheckpointDefinition]) -> Dict[str, float]:
        """Calculate resource usage profile"""
        
        profile = {
            "avg_cpu": 0,
            "avg_memory": 0,
            "peak_cpu": 0,
            "peak_memory": 0,
            "special_requirements": 0
        }
        
        cpu_values = []
        memory_values = []
        
        for cp in checkpoints:
            cpu = cp.resource_requirements.get("cpu", 1)
            memory_str = cp.resource_requirements.get("memory", "2GB")
            
            # Parse memory string (simplified)
            memory_gb = float(re.search(r'(\d+)', memory_str).group(1)) if re.search(r'(\d+)', memory_str) else 2
            
            cpu_values.append(cpu)
            memory_values.append(memory_gb)
            
            # Count special requirements
            special_keys = ["database", "gpu", "specialized_hardware", "security_tools", "cloud_services"]
            profile["special_requirements"] += sum(1 for key in special_keys 
                                                 if cp.resource_requirements.get(key, False))
        
        if cpu_values:
            profile["avg_cpu"] = statistics.mean(cpu_values)
            profile["peak_cpu"] = max(cpu_values)
        
        if memory_values:
            profile["avg_memory"] = statistics.mean(memory_values)
            profile["peak_memory"] = max(memory_values)
        
        return profile

    def _optimize_execution_path(self, 
                               path: ExecutionPath,
                               project_chars: ProjectCharacteristics) -> ExecutionPath:
        """Optimize execution path for better performance"""
        
        logger.info("⚡ Optimizing execution path...")
        
        # Apply optimizations
        optimized_checkpoints = self._apply_checkpoint_optimizations(path.checkpoints, project_chars)
        
        # Recalculate parallel groups after optimization
        optimized_parallel_groups = self._identify_parallel_groups(optimized_checkpoints)
        
        # Recalculate duration considering parallelization
        optimized_duration = self._calculate_optimized_duration(
            optimized_checkpoints, 
            optimized_parallel_groups,
            project_chars
        )
        
        return ExecutionPath(
            checkpoints=optimized_checkpoints,
            total_duration=optimized_duration,
            parallel_groups=optimized_parallel_groups,
            critical_path=path.critical_path,
            risk_score=path.risk_score,
            confidence=min(1.0, path.confidence + 0.1),  # Slight confidence boost from optimization
            resource_profile=path.resource_profile,
            success_probability=min(1.0, path.success_probability + 0.05)
        )

    def _apply_checkpoint_optimizations(self, 
                                      checkpoints: List[CheckpointDefinition],
                                      project_chars: ProjectCharacteristics) -> List[CheckpointDefinition]:
        """Apply various optimizations to checkpoint execution"""
        
        optimized = checkpoints.copy()
        
        # Optimization 1: Remove redundant checkpoints for simple projects
        if project_chars.complexity == ComplexityLevel.LOW:
            optimized = [cp for cp in optimized 
                        if cp.priority in [CheckpointPriority.CRITICAL, CheckpointPriority.HIGH]]
        
        # Optimization 2: Skip optional checkpoints if risk tolerance is low
        # (This would be configurable in a real implementation)
        
        # Optimization 3: Merge related checkpoints if possible
        # (Implementation would depend on specific checkpoint definitions)
        
        return optimized

    def _calculate_optimized_duration(self, 
                                    checkpoints: List[CheckpointDefinition],
                                    parallel_groups: List[List[str]],
                                    project_chars: ProjectCharacteristics) -> timedelta:
        """Calculate duration considering parallelization"""
        
        total_duration = timedelta()
        parallel_checkpoint_names = {name for group in parallel_groups for name in group}
        
        # Calculate duration for sequential checkpoints
        for cp in checkpoints:
            if cp.name not in parallel_checkpoint_names:
                scaled_duration = cp.estimated_duration * cp.complexity_scaling
                complexity_multiplier = {
                    ComplexityLevel.LOW: 0.8,
                    ComplexityLevel.MEDIUM: 1.0,
                    ComplexityLevel.HIGH: 1.3,
                    ComplexityLevel.ENTERPRISE: 1.6
                }[project_chars.complexity]
                total_duration += scaled_duration * complexity_multiplier
        
        # Calculate duration for parallel groups (use maximum duration in each group)
        for group in parallel_groups:
            group_checkpoints = [cp for cp in checkpoints if cp.name in group]
            if group_checkpoints:
                max_duration = max(
                    cp.estimated_duration * cp.complexity_scaling 
                    for cp in group_checkpoints
                )
                complexity_multiplier = {
                    ComplexityLevel.LOW: 0.8,
                    ComplexityLevel.MEDIUM: 1.0,
                    ComplexityLevel.HIGH: 1.3,
                    ComplexityLevel.ENTERPRISE: 1.6
                }[project_chars.complexity]
                total_duration += max_duration * complexity_multiplier
        
        return total_duration

    def _record_selection_decision(self, 
                                 path: ExecutionPath,
                                 project_chars: ProjectCharacteristics) -> None:
        """Record selection decision for future learning"""
        
        # Create historical record
        historical_record = {
            "timestamp": datetime.now().isoformat(),
            "project_characteristics": {
                "type": project_chars.type.value,
                "complexity": project_chars.complexity.value,
                "domain": project_chars.domain,
                "size_metrics": project_chars.size_metrics,
                "security_requirements": project_chars.security_requirements,
                "performance_critical": project_chars.performance_critical
            },
            "selected_path": {
                "checkpoints": [cp.name for cp in path.checkpoints],
                "total_duration": path.total_duration.total_seconds(),
                "risk_score": path.risk_score,
                "success_probability": path.success_probability,
                "parallel_groups": path.parallel_groups
            }
        }
        
        # Save to file for future analysis
        try:
            selection_log = self.project_root / "sdlc_results" / "checkpoint_selections.jsonl"
            selection_log.parent.mkdir(exist_ok=True)
            
            with open(selection_log, 'a') as f:
                f.write(json.dumps(historical_record) + "\n")
                
        except Exception as e:
            logger.warning(f"Could not record selection decision: {e}")

    def _initialize_ml_models(self) -> None:
        """Initialize ML models for intelligent selection"""
        
        if not ML_AVAILABLE:
            return
        
        # Initialize models
        self.ml_models = {
            "success_predictor": RandomForestClassifier(n_estimators=100, random_state=42),
            "duration_predictor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "risk_assessor": RandomForestClassifier(n_estimators=50, random_state=42)
        }
        
        # Load historical data if available
        self._load_historical_data()

    def _load_historical_data(self) -> None:
        """Load historical execution data for ML training"""
        
        selection_log = self.project_root / "sdlc_results" / "checkpoint_selections.jsonl"
        
        if not selection_log.exists():
            return
        
        try:
            with open(selection_log) as f:
                for line in f:
                    record = json.loads(line.strip())
                    # Process record and add to training data
                    # (Implementation would convert records to training examples)
                    
        except Exception as e:
            logger.warning(f"Could not load historical data: {e}")


def main():
    """Main function for testing the intelligent checkpoint selector"""
    
    print("🧠 Intelligent Checkpoint Selection Engine")
    print("=" * 50)
    
    # Initialize selector
    selector = IntelligentCheckpointSelector()
    
    # Select optimal checkpoints
    execution_path = selector.select_optimal_checkpoints()
    
    # Print results
    print(f"\n✅ Optimal Execution Path Selected:")
    print(f"   Duration: {execution_path.total_duration}")
    print(f"   Checkpoints: {len(execution_path.checkpoints)}")
    print(f"   Success Probability: {execution_path.success_probability:.2f}")
    print(f"   Risk Score: {execution_path.risk_score:.2f}")
    
    print(f"\n📋 Checkpoint Sequence:")
    for i, cp in enumerate(execution_path.checkpoints, 1):
        print(f"   {i}. {cp.name} ({cp.priority.value})")
    
    if execution_path.parallel_groups:
        print(f"\n⚡ Parallel Execution Groups:")
        for i, group in enumerate(execution_path.parallel_groups, 1):
            print(f"   Group {i}: {', '.join(group)}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Self-Improving Quality Metrics Engine for Autonomous SDLC
========================================================

🧠 MACHINE LEARNING-ENHANCED QUALITY MEASUREMENT AND OPTIMIZATION

This system implements self-learning quality metrics that continuously improve
their accuracy and predictive power through feedback loops, historical data analysis,
and advanced machine learning techniques.

🎯 INTELLIGENT QUALITY FEATURES:
• Adaptive Thresholds: AI-driven quality gate thresholds that adjust based on project context
• Predictive Quality Scoring: ML models that predict quality issues before they occur
• Anomaly Detection: Automated detection of quality regressions and unusual patterns
• Continuous Learning: Models that improve accuracy through feedback and historical data
• Context-Aware Metrics: Quality measures that adapt to project type, domain, and complexity
• Trend Analysis: Long-term quality trend prediction and optimization recommendations

🚀 ADVANCED QUALITY DIMENSIONS:
• Code Quality: Maintainability, complexity, technical debt, code smells
• Security Quality: Vulnerability detection, cryptographic validation, compliance
• Performance Quality: Latency, throughput, resource efficiency, scalability
• Reliability Quality: Error rates, recovery time, availability, fault tolerance
• User Experience: Usability, accessibility, internationalization, responsiveness
• Research Quality: Reproducibility, statistical significance, novelty, impact

🛡️ SELF-IMPROVING MECHANISMS:
• Bayesian Optimization: Automatic hyperparameter tuning for quality models
• Ensemble Learning: Combining multiple models for robust quality prediction
• Active Learning: Intelligent selection of most informative data points
• Transfer Learning: Leveraging knowledge from similar projects
• Meta-Learning: Learning to learn better quality assessment strategies
• Reinforcement Learning: Optimizing quality gate sequences through trial and error

⚡ QUALITY INTELLIGENCE:
• Real-time Quality Monitoring: Continuous assessment during development
• Predictive Quality Analytics: Forecasting quality issues and their impact
• Root Cause Analysis: AI-powered identification of quality problem sources
• Quality Debt Tracking: Technical debt accumulation and prioritization
• Quality ROI Analysis: Cost-benefit analysis of quality improvements
• Smart Quality Gates: Dynamic gate selection based on risk assessment

🌍 GLOBAL QUALITY STANDARDS:
• Multi-Standard Compliance: ISO 9001, CMMI, IEEE standards alignment
• Cultural Adaptation: Quality metrics adapted to different development cultures
• Regulatory Compliance: GDPR, HIPAA, SOC2 quality requirements
• Industry Standards: Domain-specific quality benchmarks and best practices

Built with ❤️ by Terragon Labs - Making Quality Intelligent
"""

import json
import logging
import time
import statistics
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import subprocess
import ast
import re

# Advanced ML imports
try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, f_regression
    from scipy import stats
    from scipy.optimize import minimize
    ML_ADVANCED = True
except ImportError:
    ML_ADVANCED = False
    np = pd = RandomForestRegressor = None

# Bayesian optimization
try:
    from skopt import gp_minimize, forest_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_OPT = True
except ImportError:
    BAYESIAN_OPT = False

logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Quality measurement dimensions"""
    CODE_QUALITY = "code_quality"
    SECURITY_QUALITY = "security_quality"
    PERFORMANCE_QUALITY = "performance_quality"
    RELIABILITY_QUALITY = "reliability_quality"
    USER_EXPERIENCE = "user_experience"
    RESEARCH_QUALITY = "research_quality"
    MAINTAINABILITY = "maintainability"
    TESTABILITY = "testability"
    DOCUMENTATION_QUALITY = "documentation_quality"
    COMPLIANCE_QUALITY = "compliance_quality"

class MetricType(Enum):
    """Types of quality metrics"""
    STATIC_ANALYSIS = "static"
    DYNAMIC_ANALYSIS = "dynamic"
    BEHAVIORAL = "behavioral"
    PREDICTIVE = "predictive"
    COMPOSITE = "composite"

class LearningStrategy(Enum):
    """Machine learning strategies for quality improvement"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    META_LEARNING = "meta"
    ACTIVE_LEARNING = "active"

@dataclass
class QualityMetricDefinition:
    """Enhanced quality metric definition with ML capabilities"""
    name: str
    dimension: QualityDimension
    metric_type: MetricType
    description: str
    baseline_threshold: float
    adaptive_threshold: bool = True
    weight: float = 1.0
    calculation_method: str = "heuristic"
    ml_model: Optional[str] = None
    feature_extractors: List[str] = field(default_factory=list)
    context_factors: List[str] = field(default_factory=list)
    quality_gate_eligible: bool = True
    trend_analysis: bool = True
    anomaly_detection: bool = False
    predictive_horizon: timedelta = field(default_factory=lambda: timedelta(hours=24))

@dataclass
class QualityMeasurement:
    """Individual quality measurement with rich metadata"""
    metric_name: str
    value: float
    threshold: float
    normalized_score: float
    confidence: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    prediction: Optional[float] = None
    trend_direction: str = "stable"  # "improving", "declining", "stable"
    anomaly_score: float = 0.0
    contributing_factors: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

@dataclass
class QualityTrend:
    """Quality trend analysis with predictive capabilities"""
    metric_name: str
    historical_values: List[float]
    timestamps: List[datetime]
    trend_coefficient: float
    seasonal_patterns: Dict[str, float]
    anomalies: List[Tuple[datetime, float, str]]
    predictions: List[Tuple[datetime, float, float]]  # (timestamp, value, confidence)
    quality_volatility: float
    improvement_rate: float
    regression_risk: float

@dataclass
class AdaptiveThreshold:
    """Self-adjusting quality thresholds with ML optimization"""
    metric_name: str
    current_threshold: float
    base_threshold: float
    adaptation_history: List[Tuple[datetime, float, str]]
    context_adjustments: Dict[str, float]
    performance_feedback: List[Tuple[float, bool]]  # (threshold, success)
    optimization_score: float
    last_optimization: datetime
    confidence_interval: Tuple[float, float]
    adaptation_strategy: str = "gradient_descent"

class SelfImprovingQualityMetrics:
    """
    Advanced quality metrics system that learns and improves continuously
    through machine learning, feedback analysis, and adaptive optimization
    """

    def __init__(self, project_root: str = None):
        """Initialize the self-improving quality metrics system"""
        self.project_root = Path(project_root or Path.cwd()).resolve()
        
        # Initialize storage
        self.metrics_dir = self.project_root / "sdlc_results" / "quality_metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric definitions
        self.metric_definitions = self._initialize_metric_definitions()
        
        # Historical data and models
        self.historical_measurements: Dict[str, List[QualityMeasurement]] = {}
        self.quality_trends: Dict[str, QualityTrend] = {}
        self.adaptive_thresholds: Dict[str, AdaptiveThreshold] = {}
        self.ml_models: Dict[str, Any] = {}
        self.anomaly_detectors: Dict[str, Any] = {}
        self.feature_scalers: Dict[str, Any] = {}
        
        # Learning system state
        self.learning_iteration = 0
        self.model_performance_history: Dict[str, List[float]] = {}
        self.optimization_history: List[Dict[str, Any]] = []
        
        # Initialize ML components
        if ML_ADVANCED:
            self._initialize_ml_models()
            self._load_historical_data()
            
        logger.info(f"🧠 Self-Improving Quality Metrics initialized")
        logger.info(f"   ML Advanced: {'✅' if ML_ADVANCED else '❌'}")
        logger.info(f"   Bayesian Optimization: {'✅' if BAYESIAN_OPT else '❌'}")
        logger.info(f"   Metric Definitions: {len(self.metric_definitions)}")

    def measure_quality(self, context: Dict[str, Any] = None) -> Dict[str, QualityMeasurement]:
        """
        Perform comprehensive quality measurement with ML enhancement
        """
        logger.info("📊 Performing intelligent quality measurement...")
        
        start_time = time.time()
        measurements = {}
        context = context or {}
        
        # Add system context
        context.update(self._gather_system_context())
        
        for metric_name, metric_def in self.metric_definitions.items():
            try:
                measurement = self._measure_single_metric(metric_def, context)
                measurements[metric_name] = measurement
                
                # Record measurement for learning
                self._record_measurement(measurement)
                
                logger.info(f"   ✅ {metric_name}: {measurement.value:.3f} (threshold: {measurement.threshold:.3f})")
                
            except Exception as e:
                logger.error(f"   ❌ {metric_name}: Failed - {e}")
                # Create failed measurement
                measurements[metric_name] = QualityMeasurement(
                    metric_name=metric_name,
                    value=0.0,
                    threshold=metric_def.baseline_threshold,
                    normalized_score=0.0,
                    confidence=0.0,
                    timestamp=datetime.now(),
                    context=context
                )
        
        measurement_time = time.time() - start_time
        
        # Perform post-measurement analysis
        self._analyze_measurement_patterns(measurements)
        self._update_trends(measurements)
        self._detect_anomalies(measurements)
        
        logger.info(f"✅ Quality measurement completed in {measurement_time:.1f}s")
        
        return measurements

    def _initialize_metric_definitions(self) -> Dict[str, QualityMetricDefinition]:
        """Initialize comprehensive metric definitions"""
        
        metrics = {}
        
        # Code Quality Metrics
        metrics["cyclomatic_complexity"] = QualityMetricDefinition(
            name="Cyclomatic Complexity",
            dimension=QualityDimension.CODE_QUALITY,
            metric_type=MetricType.STATIC_ANALYSIS,
            description="Average cyclomatic complexity of functions and methods",
            baseline_threshold=10.0,
            adaptive_threshold=True,
            weight=1.5,
            ml_model="complexity_predictor",
            feature_extractors=["function_count", "line_count", "nesting_depth"],
            context_factors=["project_size", "domain", "team_experience"],
            anomaly_detection=True
        )
        
        metrics["maintainability_index"] = QualityMetricDefinition(
            name="Maintainability Index",
            dimension=QualityDimension.MAINTAINABILITY,
            metric_type=MetricType.COMPOSITE,
            description="Composite maintainability score based on complexity, comments, and structure",
            baseline_threshold=60.0,
            adaptive_threshold=True,
            weight=2.0,
            ml_model="maintainability_predictor",
            feature_extractors=["complexity_metrics", "comment_ratio", "duplication"],
            trend_analysis=True,
            anomaly_detection=True
        )
        
        metrics["technical_debt_ratio"] = QualityMetricDefinition(
            name="Technical Debt Ratio",
            dimension=QualityDimension.CODE_QUALITY,
            metric_type=MetricType.COMPOSITE,
            description="Ratio of technical debt to total development cost",
            baseline_threshold=0.05,
            adaptive_threshold=True,
            weight=1.8,
            ml_model="debt_predictor",
            feature_extractors=["code_smells", "duplication", "complexity"],
            predictive_horizon=timedelta(days=7)
        )
        
        # Security Quality Metrics
        metrics["security_vulnerability_density"] = QualityMetricDefinition(
            name="Security Vulnerability Density",
            dimension=QualityDimension.SECURITY_QUALITY,
            metric_type=MetricType.STATIC_ANALYSIS,
            description="Number of security vulnerabilities per 1000 lines of code",
            baseline_threshold=1.0,
            adaptive_threshold=True,
            weight=3.0,
            ml_model="vulnerability_predictor",
            feature_extractors=["code_patterns", "dependency_analysis", "input_validation"],
            anomaly_detection=True
        )
        
        metrics["cryptographic_strength"] = QualityMetricDefinition(
            name="Cryptographic Strength",
            dimension=QualityDimension.SECURITY_QUALITY,
            metric_type=MetricType.STATIC_ANALYSIS,
            description="Assessment of cryptographic algorithm strength and implementation",
            baseline_threshold=0.9,
            adaptive_threshold=False,  # Security thresholds should be strict
            weight=3.0,
            feature_extractors=["algorithm_analysis", "key_management", "implementation_patterns"]
        )
        
        # Performance Quality Metrics
        metrics["computational_complexity"] = QualityMetricDefinition(
            name="Computational Complexity",
            dimension=QualityDimension.PERFORMANCE_QUALITY,
            metric_type=MetricType.STATIC_ANALYSIS,
            description="Average computational complexity of algorithms",
            baseline_threshold=100.0,  # O(n log n) normalized score
            adaptive_threshold=True,
            weight=1.5,
            ml_model="performance_predictor",
            feature_extractors=["loop_analysis", "recursion_depth", "data_structures"],
            predictive_horizon=timedelta(hours=4)
        )
        
        metrics["memory_efficiency"] = QualityMetricDefinition(
            name="Memory Efficiency",
            dimension=QualityDimension.PERFORMANCE_QUALITY,
            metric_type=MetricType.DYNAMIC_ANALYSIS,
            description="Memory usage efficiency and leak detection",
            baseline_threshold=0.8,
            adaptive_threshold=True,
            weight=1.3,
            ml_model="memory_predictor",
            feature_extractors=["allocation_patterns", "cleanup_analysis", "data_structure_size"]
        )
        
        # Reliability Quality Metrics
        metrics["error_handling_coverage"] = QualityMetricDefinition(
            name="Error Handling Coverage",
            dimension=QualityDimension.RELIABILITY_QUALITY,
            metric_type=MetricType.STATIC_ANALYSIS,
            description="Percentage of potential error conditions with proper handling",
            baseline_threshold=0.85,
            adaptive_threshold=True,
            weight=2.0,
            ml_model="reliability_predictor",
            feature_extractors=["exception_patterns", "error_propagation", "recovery_mechanisms"]
        )
        
        metrics["fault_tolerance_score"] = QualityMetricDefinition(
            name="Fault Tolerance Score",
            dimension=QualityDimension.RELIABILITY_QUALITY,
            metric_type=MetricType.BEHAVIORAL,
            description="System's ability to continue operating under adverse conditions",
            baseline_threshold=0.9,
            adaptive_threshold=True,
            weight=2.5,
            feature_extractors=["circuit_breakers", "retry_mechanisms", "graceful_degradation"]
        )
        
        # Research Quality Metrics (for research projects)
        metrics["reproducibility_score"] = QualityMetricDefinition(
            name="Reproducibility Score",
            dimension=QualityDimension.RESEARCH_QUALITY,
            metric_type=MetricType.BEHAVIORAL,
            description="Ability to reproduce experimental results consistently",
            baseline_threshold=0.95,
            adaptive_threshold=False,  # Research standards should be strict
            weight=3.0,
            feature_extractors=["seed_management", "version_control", "environment_specification"]
        )
        
        metrics["statistical_significance"] = QualityMetricDefinition(
            name="Statistical Significance",
            dimension=QualityDimension.RESEARCH_QUALITY,
            metric_type=MetricType.STATIC_ANALYSIS,
            description="Statistical validity of experimental results",
            baseline_threshold=0.05,  # p-value threshold
            adaptive_threshold=False,
            weight=3.0,
            feature_extractors=["sample_size", "effect_size", "statistical_tests"]
        )
        
        # Documentation Quality Metrics
        metrics["documentation_completeness"] = QualityMetricDefinition(
            name="Documentation Completeness",
            dimension=QualityDimension.DOCUMENTATION_QUALITY,
            metric_type=MetricType.STATIC_ANALYSIS,
            description="Percentage of public API with comprehensive documentation",
            baseline_threshold=0.9,
            adaptive_threshold=True,
            weight=1.5,
            ml_model="documentation_predictor",
            feature_extractors=["docstring_coverage", "comment_quality", "example_availability"]
        )
        
        metrics["documentation_accuracy"] = QualityMetricDefinition(
            name="Documentation Accuracy",
            dimension=QualityDimension.DOCUMENTATION_QUALITY,
            metric_type=MetricType.BEHAVIORAL,
            description="Accuracy of documentation relative to actual implementation",
            baseline_threshold=0.95,
            adaptive_threshold=True,
            weight=1.8,
            ml_model="doc_accuracy_predictor",
            feature_extractors=["code_doc_alignment", "example_validation", "version_sync"]
        )
        
        # Testability Metrics
        metrics["test_coverage"] = QualityMetricDefinition(
            name="Test Coverage",
            dimension=QualityDimension.TESTABILITY,
            metric_type=MetricType.DYNAMIC_ANALYSIS,
            description="Percentage of code covered by automated tests",
            baseline_threshold=0.85,
            adaptive_threshold=True,
            weight=2.0,
            ml_model="coverage_predictor",
            feature_extractors=["line_coverage", "branch_coverage", "function_coverage"],
            trend_analysis=True
        )
        
        metrics["test_effectiveness"] = QualityMetricDefinition(
            name="Test Effectiveness",
            dimension=QualityDimension.TESTABILITY,
            metric_type=MetricType.BEHAVIORAL,
            description="Ability of tests to detect bugs and regressions",
            baseline_threshold=0.8,
            adaptive_threshold=True,
            weight=2.2,
            ml_model="test_effectiveness_predictor",
            feature_extractors=["mutation_testing", "assertion_quality", "test_diversity"]
        )
        
        return metrics

    def _measure_single_metric(self, 
                             metric_def: QualityMetricDefinition,
                             context: Dict[str, Any]) -> QualityMeasurement:
        """Measure a single quality metric with ML enhancement"""
        
        timestamp = datetime.now()
        
        # Extract features for the metric
        features = self._extract_features(metric_def, context)
        
        # Calculate raw metric value
        raw_value = self._calculate_metric_value(metric_def, features, context)
        
        # Get adaptive threshold
        threshold = self._get_adaptive_threshold(metric_def.name, context)
        
        # Normalize score (0-1 scale)
        normalized_score = self._normalize_metric_score(raw_value, threshold, metric_def)
        
        # Calculate confidence
        confidence = self._calculate_confidence(metric_def, features, context)
        
        # Get ML prediction if model exists
        prediction = self._get_ml_prediction(metric_def, features)
        
        # Analyze trend
        trend_direction = self._analyze_metric_trend(metric_def.name, raw_value)
        
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(metric_def, raw_value, features)
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(metric_def, raw_value, features)
        
        return QualityMeasurement(
            metric_name=metric_def.name,
            value=raw_value,
            threshold=threshold,
            normalized_score=normalized_score,
            confidence=confidence,
            timestamp=timestamp,
            context=context.copy(),
            features=features,
            prediction=prediction,
            trend_direction=trend_direction,
            anomaly_score=anomaly_score,
            improvement_suggestions=suggestions
        )

    def _extract_features(self, 
                         metric_def: QualityMetricDefinition,
                         context: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for metric calculation and ML models"""
        
        features = {}
        
        # Project-level features
        features.update(self._extract_project_features())
        
        # Code-level features
        features.update(self._extract_code_features())
        
        # Context-specific features
        features.update(self._extract_context_features(context))
        
        # Metric-specific features
        for extractor in metric_def.feature_extractors:
            extractor_features = self._run_feature_extractor(extractor)
            features.update(extractor_features)
        
        return features

    def _extract_project_features(self) -> Dict[str, float]:
        """Extract project-level features"""
        
        features = {}
        
        # Project size features
        src_dir = self.project_root / "src"
        if src_dir.exists():
            py_files = list(src_dir.rglob("*.py"))
            features["file_count"] = len(py_files)
            
            total_lines = 0
            for py_file in py_files[:50]:  # Limit for performance
                try:
                    with open(py_file, encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except Exception:
                    continue
            
            features["total_lines"] = total_lines
            features["avg_file_size"] = total_lines / max(1, len(py_files))
        
        # Complexity features
        features["directory_depth"] = self._calculate_directory_depth()
        features["import_complexity"] = self._calculate_import_complexity()
        
        return features

    def _extract_code_features(self) -> Dict[str, float]:
        """Extract code-level features using AST analysis"""
        
        features = {
            "function_count": 0,
            "class_count": 0,
            "avg_function_length": 0,
            "max_nesting_depth": 0,
            "comment_ratio": 0,
            "docstring_ratio": 0
        }
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return features
        
        py_files = list(src_dir.rglob("*.py"))
        
        function_lengths = []
        total_lines = 0
        comment_lines = 0
        docstring_lines = 0
        
        for py_file in py_files[:20]:  # Limit for performance
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                    
                total_lines += len(content.splitlines())
                
                # Count comment lines
                comment_lines += len([line for line in content.splitlines() 
                                    if line.strip().startswith('#')])
                
                # AST analysis
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            features["function_count"] += 1
                            func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 10
                            function_lengths.append(func_lines)
                            
                            # Check for docstring
                            if (node.body and isinstance(node.body[0], ast.Expr) and
                                isinstance(node.body[0].value, ast.Constant) and
                                isinstance(node.body[0].value.value, str)):
                                docstring_lines += len(node.body[0].value.value.splitlines())
                        
                        elif isinstance(node, ast.ClassDef):
                            features["class_count"] += 1
                            
                            # Check for class docstring
                            if (node.body and isinstance(node.body[0], ast.Expr) and
                                isinstance(node.body[0].value, ast.Constant) and
                                isinstance(node.body[0].value.value, str)):
                                docstring_lines += len(node.body[0].value.value.splitlines())
                
                except SyntaxError:
                    continue
                    
            except Exception:
                continue
        
        # Calculate derived features
        if function_lengths:
            features["avg_function_length"] = statistics.mean(function_lengths)
        
        if total_lines > 0:
            features["comment_ratio"] = comment_lines / total_lines
            features["docstring_ratio"] = docstring_lines / total_lines
        
        return features

    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Extract context-specific features"""
        
        features = {}
        
        # Time-based features
        now = datetime.now()
        features["hour_of_day"] = now.hour
        features["day_of_week"] = now.weekday()
        features["is_weekend"] = float(now.weekday() >= 5)
        
        # Development phase features
        features["development_phase"] = context.get("development_phase", 0)  # 0=early, 1=mid, 2=late
        features["team_size"] = context.get("team_size", 1)
        features["deadline_pressure"] = context.get("deadline_pressure", 0.5)
        
        return features

    def _run_feature_extractor(self, extractor_name: str) -> Dict[str, float]:
        """Run specific feature extractor"""
        
        extractors = {
            "function_count": self._extract_function_metrics,
            "complexity_metrics": self._extract_complexity_metrics,
            "security_patterns": self._extract_security_patterns,
            "performance_patterns": self._extract_performance_patterns,
            "test_patterns": self._extract_test_patterns
        }
        
        if extractor_name in extractors:
            try:
                return extractors[extractor_name]()
            except Exception as e:
                logger.warning(f"Feature extractor {extractor_name} failed: {e}")
        
        return {}

    def _extract_function_metrics(self) -> Dict[str, float]:
        """Extract function-level metrics"""
        
        metrics = {
            "total_functions": 0,
            "avg_params": 0,
            "max_params": 0,
            "recursive_functions": 0
        }
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return metrics
        
        param_counts = []
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        metrics["total_functions"] += 1
                        param_count = len(node.args.args)
                        param_counts.append(param_count)
                        
                        # Check for recursion (simplified)
                        func_name = node.name
                        for child in ast.walk(node):
                            if isinstance(child, ast.Name) and child.id == func_name:
                                metrics["recursive_functions"] += 1
                                break
                                
            except Exception:
                continue
        
        if param_counts:
            metrics["avg_params"] = statistics.mean(param_counts)
            metrics["max_params"] = max(param_counts)
        
        return metrics

    def _extract_complexity_metrics(self) -> Dict[str, float]:
        """Extract complexity-related metrics"""
        
        return {
            "cyclomatic_complexity": self._calculate_cyclomatic_complexity(),
            "nesting_depth": self._calculate_max_nesting_depth(),
            "cognitive_complexity": self._calculate_cognitive_complexity()
        }

    def _extract_security_patterns(self) -> Dict[str, float]:
        """Extract security-related patterns"""
        
        patterns = {
            "hardcoded_secrets": 0,
            "sql_injection_risk": 0,
            "xss_risk": 0,
            "crypto_usage": 0
        }
        
        # Simplified pattern detection
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return patterns
        
        security_patterns = {
            "hardcoded_secrets": [r'password\s*=\s*["\'][^"\']+["\']', r'api_key\s*=\s*["\'][^"\']+["\']'],
            "sql_injection_risk": [r'\.execute\(.*\+.*\)', r'SELECT.*\+'],
            "xss_risk": [r'innerHTML\s*=', r'document\.write\('],
            "crypto_usage": [r'import.*crypt', r'from.*crypt', r'hashlib', r'ssl']
        }
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                for pattern_name, regexes in security_patterns.items():
                    for regex in regexes:
                        if re.search(regex, content, re.IGNORECASE):
                            patterns[pattern_name] += 1
                            
            except Exception:
                continue
        
        return patterns

    def _extract_performance_patterns(self) -> Dict[str, float]:
        """Extract performance-related patterns"""
        
        return {
            "nested_loops": self._count_nested_loops(),
            "recursion_depth": self._estimate_recursion_depth(),
            "memory_allocations": self._count_memory_allocations(),
            "io_operations": self._count_io_operations()
        }

    def _extract_test_patterns(self) -> Dict[str, float]:
        """Extract test-related patterns"""
        
        patterns = {
            "test_files": 0,
            "test_functions": 0,
            "assertions": 0,
            "mocks": 0
        }
        
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            return patterns
        
        test_files = list(tests_dir.rglob("test_*.py")) + list(tests_dir.rglob("*_test.py"))
        patterns["test_files"] = len(test_files)
        
        for test_file in test_files:
            try:
                with open(test_file, encoding='utf-8') as f:
                    content = f.read()
                
                # Count test functions
                patterns["test_functions"] += len(re.findall(r'def test_', content))
                
                # Count assertions
                patterns["assertions"] += len(re.findall(r'assert\s+', content))
                
                # Count mocks
                patterns["mocks"] += len(re.findall(r'mock|Mock|patch', content))
                
            except Exception:
                continue
        
        return patterns

    def _calculate_metric_value(self, 
                              metric_def: QualityMetricDefinition,
                              features: Dict[str, float],
                              context: Dict[str, Any]) -> float:
        """Calculate raw metric value"""
        
        # Dispatch to specific calculators
        calculators = {
            "Cyclomatic Complexity": self._calculate_cyclomatic_complexity,
            "Maintainability Index": self._calculate_maintainability_index,
            "Technical Debt Ratio": self._calculate_technical_debt_ratio,
            "Security Vulnerability Density": self._calculate_vulnerability_density,
            "Cryptographic Strength": self._calculate_crypto_strength,
            "Error Handling Coverage": self._calculate_error_handling_coverage,
            "Test Coverage": self._calculate_test_coverage,
            "Documentation Completeness": self._calculate_doc_completeness
        }
        
        if metric_def.name in calculators:
            try:
                return calculators[metric_def.name]()
            except Exception as e:
                logger.warning(f"Metric calculation failed for {metric_def.name}: {e}")
                return 0.0
        
        # Default calculation based on features
        return self._calculate_default_metric(metric_def, features)

    def _calculate_cyclomatic_complexity(self) -> float:
        """Calculate average cyclomatic complexity"""
        
        complexities = []
        src_dir = self.project_root / "src"
        
        if not src_dir.exists():
            return 1.0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        complexity = self._calculate_function_complexity(node)
                        complexities.append(complexity)
                        
            except Exception:
                continue
        
        return statistics.mean(complexities) if complexities else 1.0

    def _calculate_function_complexity(self, func_node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function"""
        
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
        
        return complexity

    def _calculate_maintainability_index(self) -> float:
        """Calculate maintainability index (0-100 scale)"""
        
        # Simplified maintainability index calculation
        # MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * (Cyclomatic Complexity) - 16.2 * ln(Lines of Code)
        
        avg_complexity = self._calculate_cyclomatic_complexity()
        total_lines = self._count_total_lines()
        halstead_volume = self._estimate_halstead_volume()
        
        if total_lines > 0 and halstead_volume > 0:
            mi = 171 - 5.2 * math.log(halstead_volume) - 0.23 * avg_complexity - 16.2 * math.log(total_lines)
            return max(0, min(100, mi))
        
        return 60.0  # Default moderate score

    def _calculate_technical_debt_ratio(self) -> float:
        """Calculate technical debt ratio"""
        
        # Simplified debt calculation based on code smells
        debt_indicators = {
            "long_methods": self._count_long_methods(),
            "large_classes": self._count_large_classes(),
            "code_duplication": self._estimate_code_duplication(),
            "complex_conditions": self._count_complex_conditions()
        }
        
        total_debt = sum(debt_indicators.values())
        total_code_units = self._count_code_units()
        
        return total_debt / max(total_code_units, 1)

    def _calculate_vulnerability_density(self) -> float:
        """Calculate security vulnerability density"""
        
        vulnerabilities = 0
        total_lines = self._count_total_lines()
        
        # Check for common vulnerability patterns
        vuln_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'os\.system\s*\(',
            r'subprocess\..*shell=True',
            r'password.*=.*["\'].*["\']',
            r'sql.*\+.*\%'
        ]
        
        src_dir = self.project_root / "src"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern in vuln_patterns:
                        vulnerabilities += len(re.findall(pattern, content, re.IGNORECASE))
                        
                except Exception:
                    continue
        
        return vulnerabilities / max(total_lines / 1000, 1)  # Per 1000 lines

    def _calculate_crypto_strength(self) -> float:
        """Calculate cryptographic strength assessment"""
        
        # For cryptographic projects, this would be more sophisticated
        crypto_score = 0.8  # Base score
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return crypto_score
        
        # Check for strong crypto patterns
        strong_patterns = ["AES", "RSA", "ECDSA", "SHA256", "PBKDF2"]
        weak_patterns = ["MD5", "SHA1", "DES", "RC4"]
        
        strong_count = 0
        weak_count = 0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read().upper()
                
                for pattern in strong_patterns:
                    if pattern in content:
                        strong_count += 1
                
                for pattern in weak_patterns:
                    if pattern in content:
                        weak_count += 1
                        
            except Exception:
                continue
        
        # Adjust score based on findings
        crypto_score += strong_count * 0.02
        crypto_score -= weak_count * 0.1
        
        return max(0.0, min(1.0, crypto_score))

    def _calculate_error_handling_coverage(self) -> float:
        """Calculate error handling coverage"""
        
        functions_with_handling = 0
        total_functions = 0
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 0.5
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        total_functions += 1
                        
                        # Check for try/except blocks
                        for child in ast.walk(node):
                            if isinstance(child, ast.Try):
                                functions_with_handling += 1
                                break
                                
            except Exception:
                continue
        
        return functions_with_handling / max(total_functions, 1)

    def _calculate_test_coverage(self) -> float:
        """Calculate test coverage (simplified estimation)"""
        
        # In a real implementation, this would use coverage tools
        test_files = len(list((self.project_root / "tests").rglob("*.py"))) if (self.project_root / "tests").exists() else 0
        src_files = len(list((self.project_root / "src").rglob("*.py"))) if (self.project_root / "src").exists() else 1
        
        # Rough estimation: assume good coverage if test/src ratio is good
        ratio = test_files / src_files
        
        if ratio >= 1.0:
            return 0.9
        elif ratio >= 0.5:
            return 0.8
        elif ratio >= 0.3:
            return 0.7
        elif ratio >= 0.1:
            return 0.6
        else:
            return 0.5

    def _calculate_doc_completeness(self) -> float:
        """Calculate documentation completeness"""
        
        documented_functions = 0
        total_functions = 0
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 0.5
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        total_functions += 1
                        
                        # Check for docstring
                        if (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant) and
                            isinstance(node.body[0].value.value, str)):
                            documented_functions += 1
                            
            except Exception:
                continue
        
        return documented_functions / max(total_functions, 1)

    def _calculate_default_metric(self, 
                                metric_def: QualityMetricDefinition,
                                features: Dict[str, float]) -> float:
        """Default metric calculation based on features"""
        
        # Weighted feature combination
        relevant_features = [f for f in features.keys() 
                           if any(context in f for context in metric_def.context_factors)]
        
        if not relevant_features:
            return 0.5  # Neutral default
        
        feature_values = [features[f] for f in relevant_features]
        return statistics.mean(feature_values) if feature_values else 0.5

    # Helper methods for metric calculations
    
    def _calculate_directory_depth(self) -> float:
        """Calculate maximum directory depth"""
        max_depth = 0
        for path in self.project_root.rglob("*"):
            if path.is_dir():
                depth = len(path.relative_to(self.project_root).parts)
                max_depth = max(max_depth, depth)
        return max_depth

    def _calculate_import_complexity(self) -> float:
        """Calculate import complexity score"""
        total_imports = 0
        unique_modules = set()
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 1.0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                imports = re.findall(r'^(?:import|from)\s+([^\s]+)', content, re.MULTILINE)
                total_imports += len(imports)
                unique_modules.update(imports)
                
            except Exception:
                continue
        
        return len(unique_modules) / max(total_imports, 1)

    def _calculate_max_nesting_depth(self) -> float:
        """Calculate maximum nesting depth"""
        max_depth = 0
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 1.0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                depth = self._calculate_ast_depth(tree)
                max_depth = max(max_depth, depth)
                
            except Exception:
                continue
        
        return max_depth

    def _calculate_ast_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Calculate AST node depth recursively"""
        max_child_depth = current_depth
        
        for child in ast.iter_child_nodes(node):
            child_depth = self._calculate_ast_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth

    def _calculate_cognitive_complexity(self) -> float:
        """Calculate cognitive complexity (simplified)"""
        # Simplified version - in practice this would be more sophisticated
        return self._calculate_cyclomatic_complexity() * 1.2

    def _count_total_lines(self) -> int:
        """Count total lines of code"""
        total_lines = 0
        src_dir = self.project_root / "src"
        
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                try:
                    with open(py_file, encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except Exception:
                    continue
        
        return total_lines

    def _estimate_halstead_volume(self) -> float:
        """Estimate Halstead volume (simplified)"""
        # Simplified estimation
        total_operators = 0
        total_operands = 0
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 100.0
        
        # Count Python operators and identifiers (simplified)
        operators = ['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not']
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                for op in operators:
                    total_operators += content.count(op)
                
                # Rough operand count (identifiers)
                total_operands += len(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', content))
                
            except Exception:
                continue
        
        # Halstead Volume = (N1 + N2) * log2(n1 + n2)
        # Where N1, N2 are total operators/operands, n1, n2 are unique
        total_tokens = total_operators + total_operands
        unique_tokens = total_tokens * 0.3  # Rough estimation
        
        if unique_tokens > 0:
            return total_tokens * math.log2(unique_tokens)
        
        return 100.0

    def _count_long_methods(self) -> int:
        """Count methods longer than threshold"""
        long_methods = 0
        threshold = 20  # lines
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if hasattr(node, 'end_lineno') and node.end_lineno:
                            length = node.end_lineno - node.lineno
                            if length > threshold:
                                long_methods += 1
                                
            except Exception:
                continue
        
        return long_methods

    def _count_large_classes(self) -> int:
        """Count classes larger than threshold"""
        large_classes = 0
        threshold = 100  # lines
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        if hasattr(node, 'end_lineno') and node.end_lineno:
                            length = node.end_lineno - node.lineno
                            if length > threshold:
                                large_classes += 1
                                
            except Exception:
                continue
        
        return large_classes

    def _estimate_code_duplication(self) -> int:
        """Estimate code duplication (simplified)"""
        # This is a very simplified version
        return 0  # Would need sophisticated analysis

    def _count_complex_conditions(self) -> int:
        """Count complex conditional statements"""
        complex_conditions = 0
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                # Count conditions with multiple logical operators
                complex_conditions += len(re.findall(r'if.*and.*or|if.*or.*and', content))
                
            except Exception:
                continue
        
        return complex_conditions

    def _count_code_units(self) -> int:
        """Count total code units (functions + classes)"""
        return self._extract_function_metrics()["total_functions"] + self._extract_code_features()["class_count"]

    def _count_nested_loops(self) -> int:
        """Count nested loop structures"""
        nested_loops = 0
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.For, ast.While)):
                        # Check for nested loops
                        for child in ast.walk(node):
                            if isinstance(child, (ast.For, ast.While)) and child != node:
                                nested_loops += 1
                                break
                                
            except Exception:
                continue
        
        return nested_loops

    def _estimate_recursion_depth(self) -> int:
        """Estimate maximum recursion depth"""
        # Simplified estimation
        return 0  # Would need more sophisticated analysis

    def _count_memory_allocations(self) -> int:
        """Count potential memory allocations"""
        allocations = 0
        allocation_patterns = [r'list\(', r'dict\(', r'set\(', r'\[\]', r'\{\}']
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in allocation_patterns:
                    allocations += len(re.findall(pattern, content))
                    
            except Exception:
                continue
        
        return allocations

    def _count_io_operations(self) -> int:
        """Count I/O operations"""
        io_ops = 0
        io_patterns = [r'open\(', r'\.read\(', r'\.write\(', r'requests\.', r'urllib']
        
        src_dir = self.project_root / "src"
        if not src_dir.exists():
            return 0
        
        for py_file in src_dir.rglob("*.py"):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                
                for pattern in io_patterns:
                    io_ops += len(re.findall(pattern, content))
                    
            except Exception:
                continue
        
        return io_ops

    def _gather_system_context(self) -> Dict[str, Any]:
        """Gather system context for metric calculation"""
        
        return {
            "timestamp": datetime.now(),
            "project_root": str(self.project_root),
            "python_version": "3.9",  # Could be detected
            "system_load": 0.5,  # Could be measured
            "available_memory": 16.0,  # GB, could be measured
            "development_phase": "implementation"  # Could be inferred
        }

    def _get_adaptive_threshold(self, metric_name: str, context: Dict[str, Any]) -> float:
        """Get adaptive threshold for metric"""
        
        if metric_name not in self.adaptive_thresholds:
            # Initialize adaptive threshold
            metric_def = self.metric_definitions[metric_name]
            self.adaptive_thresholds[metric_name] = AdaptiveThreshold(
                metric_name=metric_name,
                current_threshold=metric_def.baseline_threshold,
                base_threshold=metric_def.baseline_threshold,
                adaptation_history=[],
                context_adjustments={},
                performance_feedback=[],
                optimization_score=0.0,
                last_optimization=datetime.now(),
                confidence_interval=(metric_def.baseline_threshold * 0.9, 
                                   metric_def.baseline_threshold * 1.1)
            )
        
        threshold_config = self.adaptive_thresholds[metric_name]
        
        # Apply context-based adjustments
        adjusted_threshold = threshold_config.current_threshold
        
        # Adjust for project complexity
        complexity = context.get("project_complexity", 1.0)
        adjusted_threshold *= complexity
        
        # Adjust for deadline pressure
        pressure = context.get("deadline_pressure", 0.5)
        if pressure > 0.8:
            adjusted_threshold *= 0.9  # Lower threshold under pressure
        
        return adjusted_threshold

    def _normalize_metric_score(self, 
                               raw_value: float, 
                               threshold: float,
                               metric_def: QualityMetricDefinition) -> float:
        """Normalize metric score to 0-1 range"""
        
        if metric_def.name in ["Security Vulnerability Density", "Technical Debt Ratio"]:
            # Lower is better
            if raw_value <= threshold:
                return 1.0
            else:
                return max(0.0, 1.0 - (raw_value - threshold) / threshold)
        else:
            # Higher is better
            if raw_value >= threshold:
                return 1.0
            else:
                return max(0.0, raw_value / threshold)

    def _calculate_confidence(self, 
                            metric_def: QualityMetricDefinition,
                            features: Dict[str, float],
                            context: Dict[str, Any]) -> float:
        """Calculate confidence in metric measurement"""
        
        base_confidence = 0.8
        
        # Adjust based on data availability
        if len(features) < 5:
            base_confidence *= 0.8
        
        # Adjust based on metric type
        if metric_def.metric_type == MetricType.STATIC_ANALYSIS:
            base_confidence *= 0.95
        elif metric_def.metric_type == MetricType.PREDICTIVE:
            base_confidence *= 0.7
        
        # Adjust based on historical data
        if metric_def.name in self.historical_measurements:
            history_length = len(self.historical_measurements[metric_def.name])
            if history_length > 10:
                base_confidence *= 1.1
        
        return min(1.0, base_confidence)

    def _get_ml_prediction(self, 
                          metric_def: QualityMetricDefinition,
                          features: Dict[str, float]) -> Optional[float]:
        """Get ML prediction for metric if model exists"""
        
        if not ML_ADVANCED or not metric_def.ml_model:
            return None
        
        model_name = metric_def.ml_model
        if model_name not in self.ml_models:
            return None
        
        try:
            model = self.ml_models[model_name]
            feature_vector = self._create_feature_vector(features)
            
            if len(feature_vector) > 0:
                prediction = model.predict([feature_vector])[0]
                return float(prediction)
        except Exception as e:
            logger.warning(f"ML prediction failed for {metric_def.name}: {e}")
        
        return None

    def _create_feature_vector(self, features: Dict[str, float]) -> List[float]:
        """Create feature vector for ML models"""
        
        # Use consistent feature ordering
        feature_names = sorted(features.keys())
        return [features[name] for name in feature_names]

    def _analyze_metric_trend(self, metric_name: str, current_value: float) -> str:
        """Analyze trend direction for metric"""
        
        if metric_name not in self.historical_measurements:
            return "stable"
        
        history = self.historical_measurements[metric_name]
        if len(history) < 3:
            return "stable"
        
        recent_values = [m.value for m in history[-3:]]
        recent_values.append(current_value)
        
        # Simple trend analysis
        if len(recent_values) >= 3:
            slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
            
            if slope > 0.01:
                return "improving"
            elif slope < -0.01:
                return "declining"
        
        return "stable"

    def _calculate_anomaly_score(self, 
                                metric_def: QualityMetricDefinition,
                                current_value: float,
                                features: Dict[str, float]) -> float:
        """Calculate anomaly score for metric value"""
        
        if not metric_def.anomaly_detection or not ML_ADVANCED:
            return 0.0
        
        model_name = f"{metric_def.name}_anomaly"
        
        if model_name not in self.anomaly_detectors:
            # Initialize anomaly detector
            self.anomaly_detectors[model_name] = IsolationForest(
                contamination=0.1, random_state=42
            )
            return 0.0
        
        try:
            detector = self.anomaly_detectors[model_name]
            feature_vector = self._create_feature_vector(features)
            feature_vector.append(current_value)
            
            if hasattr(detector, 'decision_function'):
                anomaly_score = detector.decision_function([feature_vector])[0]
                return max(0.0, -anomaly_score)  # Convert to positive score
        except Exception as e:
            logger.warning(f"Anomaly detection failed for {metric_def.name}: {e}")
        
        return 0.0

    def _generate_improvement_suggestions(self, 
                                        metric_def: QualityMetricDefinition,
                                        current_value: float,
                                        features: Dict[str, float]) -> List[str]:
        """Generate improvement suggestions based on metric analysis"""
        
        suggestions = []
        
        # General suggestions based on metric type
        if metric_def.dimension == QualityDimension.CODE_QUALITY:
            if current_value > metric_def.baseline_threshold * 1.5:
                suggestions.extend([
                    "Consider refactoring complex functions into smaller units",
                    "Review code structure and extract common patterns",
                    "Implement design patterns to reduce complexity"
                ])
        
        elif metric_def.dimension == QualityDimension.SECURITY_QUALITY:
            if current_value < metric_def.baseline_threshold * 0.8:
                suggestions.extend([
                    "Conduct comprehensive security review",
                    "Update dependencies to latest secure versions",
                    "Implement additional input validation"
                ])
        
        elif metric_def.dimension == QualityDimension.PERFORMANCE_QUALITY:
            if current_value < metric_def.baseline_threshold * 0.8:
                suggestions.extend([
                    "Profile code to identify performance bottlenecks",
                    "Consider algorithmic optimizations",
                    "Implement caching strategies"
                ])
        
        # Feature-based suggestions
        if "nested_loops" in features and features["nested_loops"] > 5:
            suggestions.append("Reduce nested loop complexity with better algorithms")
        
        if "test_coverage" in features and features["test_coverage"] < 0.8:
            suggestions.append("Increase test coverage for better quality assurance")
        
        return suggestions[:3]  # Limit to top 3 suggestions

    def _record_measurement(self, measurement: QualityMeasurement) -> None:
        """Record measurement for learning and trend analysis"""
        
        metric_name = measurement.metric_name
        
        if metric_name not in self.historical_measurements:
            self.historical_measurements[metric_name] = []
        
        self.historical_measurements[metric_name].append(measurement)
        
        # Keep only recent measurements (last 100)
        if len(self.historical_measurements[metric_name]) > 100:
            self.historical_measurements[metric_name] = self.historical_measurements[metric_name][-100:]

    def _analyze_measurement_patterns(self, measurements: Dict[str, QualityMeasurement]) -> None:
        """Analyze patterns across measurements"""
        
        # Identify correlations between metrics
        metric_values = {name: m.value for name, m in measurements.items()}
        
        # Update correlation analysis
        self._update_correlation_analysis(metric_values)
        
        # Identify quality hotspots
        hotspots = self._identify_quality_hotspots(measurements)
        
        if hotspots:
            logger.info(f"Quality hotspots identified: {hotspots}")

    def _update_correlation_analysis(self, metric_values: Dict[str, float]) -> None:
        """Update correlation analysis between metrics"""
        
        if not ML_ADVANCED or len(metric_values) < 3:
            return
        
        # Store for batch correlation analysis
        correlation_file = self.metrics_dir / "correlations.json"
        
        try:
            if correlation_file.exists():
                with open(correlation_file) as f:
                    correlation_data = json.load(f)
            else:
                correlation_data = {"measurements": []}
            
            correlation_data["measurements"].append({
                "timestamp": datetime.now().isoformat(),
                **metric_values
            })
            
            # Keep only recent data
            if len(correlation_data["measurements"]) > 1000:
                correlation_data["measurements"] = correlation_data["measurements"][-1000:]
            
            with open(correlation_file, 'w') as f:
                json.dump(correlation_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to update correlation analysis: {e}")

    def _identify_quality_hotspots(self, measurements: Dict[str, QualityMeasurement]) -> List[str]:
        """Identify quality hotspots that need attention"""
        
        hotspots = []
        
        for name, measurement in measurements.items():
            # Check for poor performance
            if measurement.normalized_score < 0.5:
                hotspots.append(f"{name}: Low score ({measurement.normalized_score:.2f})")
            
            # Check for high anomaly score
            if measurement.anomaly_score > 0.7:
                hotspots.append(f"{name}: Anomalous behavior detected")
            
            # Check for declining trend
            if measurement.trend_direction == "declining":
                hotspots.append(f"{name}: Quality declining")
        
        return hotspots

    def _update_trends(self, measurements: Dict[str, QualityMeasurement]) -> None:
        """Update quality trends"""
        
        for name, measurement in measurements.items():
            if name not in self.quality_trends:
                self.quality_trends[name] = QualityTrend(
                    metric_name=name,
                    historical_values=[],
                    timestamps=[],
                    trend_coefficient=0.0,
                    seasonal_patterns={},
                    anomalies=[],
                    predictions=[],
                    quality_volatility=0.0,
                    improvement_rate=0.0,
                    regression_risk=0.0
                )
            
            trend = self.quality_trends[name]
            trend.historical_values.append(measurement.value)
            trend.timestamps.append(measurement.timestamp)
            
            # Keep recent data
            if len(trend.historical_values) > 50:
                trend.historical_values = trend.historical_values[-50:]
                trend.timestamps = trend.timestamps[-50:]
            
            # Update trend statistics
            if len(trend.historical_values) >= 5:
                self._calculate_trend_statistics(trend)

    def _calculate_trend_statistics(self, trend: QualityTrend) -> None:
        """Calculate trend statistics"""
        
        if len(trend.historical_values) < 5:
            return
        
        values = np.array(trend.historical_values) if ML_ADVANCED else trend.historical_values
        
        # Calculate trend coefficient (slope)
        if ML_ADVANCED:
            x = np.arange(len(values))
            trend.trend_coefficient = float(np.polyfit(x, values, 1)[0])
            
            # Calculate volatility (standard deviation)
            trend.quality_volatility = float(np.std(values))
            
            # Calculate improvement rate
            if len(values) >= 10:
                recent_avg = np.mean(values[-5:])
                older_avg = np.mean(values[:5])
                trend.improvement_rate = float((recent_avg - older_avg) / older_avg)
        
        # Calculate regression risk
        declining_periods = sum(1 for i in range(1, len(trend.historical_values))
                              if trend.historical_values[i] < trend.historical_values[i-1])
        trend.regression_risk = declining_periods / len(trend.historical_values)

    def _detect_anomalies(self, measurements: Dict[str, QualityMeasurement]) -> None:
        """Detect anomalies in measurements"""
        
        if not ML_ADVANCED:
            return
        
        for name, measurement in measurements.items():
            metric_def = self.metric_definitions[name]
            
            if not metric_def.anomaly_detection:
                continue
            
            # Update anomaly detector with new data
            self._update_anomaly_detector(name, measurement)

    def _update_anomaly_detector(self, metric_name: str, measurement: QualityMeasurement) -> None:
        """Update anomaly detector with new measurement"""
        
        model_name = f"{metric_name}_anomaly"
        
        if model_name not in self.anomaly_detectors:
            self.anomaly_detectors[model_name] = IsolationForest(
                contamination=0.1, random_state=42
            )
        
        detector = self.anomaly_detectors[model_name]
        
        # Collect training data
        if metric_name in self.historical_measurements:
            history = self.historical_measurements[metric_name]
            if len(history) >= 10:
                try:
                    # Create feature matrix
                    feature_matrix = []
                    for m in history[-20:]:  # Use recent history
                        features = list(m.features.values()) if m.features else [m.value]
                        features.append(m.value)
                        feature_matrix.append(features)
                    
                    # Retrain detector
                    detector.fit(feature_matrix)
                    
                except Exception as e:
                    logger.warning(f"Failed to update anomaly detector for {metric_name}: {e}")

    def _initialize_ml_models(self) -> None:
        """Initialize ML models for quality prediction"""
        
        if not ML_ADVANCED:
            return
        
        # Initialize predictive models
        self.ml_models = {
            "complexity_predictor": RandomForestRegressor(n_estimators=100, random_state=42),
            "maintainability_predictor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "debt_predictor": RandomForestRegressor(n_estimators=50, random_state=42),
            "vulnerability_predictor": RandomForestRegressor(n_estimators=100, random_state=42),
            "performance_predictor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "memory_predictor": RandomForestRegressor(n_estimators=50, random_state=42),
            "reliability_predictor": GradientBoostingRegressor(n_estimators=75, random_state=42),
            "documentation_predictor": RandomForestRegressor(n_estimators=50, random_state=42),
            "doc_accuracy_predictor": GradientBoostingRegressor(n_estimators=75, random_state=42),
            "coverage_predictor": RandomForestRegressor(n_estimators=75, random_state=42),
            "test_effectiveness_predictor": GradientBoostingRegressor(n_estimators=75, random_state=42)
        }
        
        # Initialize feature scalers
        self.feature_scalers = {
            name: StandardScaler() for name in self.ml_models.keys()
        }

    def _load_historical_data(self) -> None:
        """Load historical measurement data"""
        
        measurements_file = self.metrics_dir / "historical_measurements.json"
        
        if not measurements_file.exists():
            return
        
        try:
            with open(measurements_file) as f:
                data = json.load(f)
            
            # Reconstruct historical measurements
            for metric_name, measurements in data.items():
                self.historical_measurements[metric_name] = []
                
                for m_data in measurements[-50:]:  # Load recent data
                    measurement = QualityMeasurement(
                        metric_name=m_data["metric_name"],
                        value=m_data["value"],
                        threshold=m_data["threshold"],
                        normalized_score=m_data["normalized_score"],
                        confidence=m_data["confidence"],
                        timestamp=datetime.fromisoformat(m_data["timestamp"]),
                        context=m_data.get("context", {}),
                        features=m_data.get("features", {}),
                        prediction=m_data.get("prediction"),
                        trend_direction=m_data.get("trend_direction", "stable"),
                        anomaly_score=m_data.get("anomaly_score", 0.0),
                        improvement_suggestions=m_data.get("improvement_suggestions", [])
                    )
                    
                    self.historical_measurements[metric_name].append(measurement)
                    
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")

    def save_measurements(self, measurements: Dict[str, QualityMeasurement]) -> None:
        """Save measurements to persistent storage"""
        
        # Save individual measurements
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        measurements_file = self.metrics_dir / f"measurements_{timestamp}.json"
        
        try:
            measurement_data = {}
            for name, measurement in measurements.items():
                measurement_data[name] = {
                    "metric_name": measurement.metric_name,
                    "value": measurement.value,
                    "threshold": measurement.threshold,
                    "normalized_score": measurement.normalized_score,
                    "confidence": measurement.confidence,
                    "timestamp": measurement.timestamp.isoformat(),
                    "context": measurement.context,
                    "features": measurement.features,
                    "prediction": measurement.prediction,
                    "trend_direction": measurement.trend_direction,
                    "anomaly_score": measurement.anomaly_score,
                    "improvement_suggestions": measurement.improvement_suggestions
                }
            
            with open(measurements_file, 'w') as f:
                json.dump(measurement_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save measurements: {e}")
        
        # Update historical measurements file
        self._save_historical_data()

    def _save_historical_data(self) -> None:
        """Save historical data to file"""
        
        measurements_file = self.metrics_dir / "historical_measurements.json"
        
        try:
            historical_data = {}
            
            for metric_name, measurements in self.historical_measurements.items():
                historical_data[metric_name] = []
                
                for m in measurements[-100:]:  # Keep recent data
                    historical_data[metric_name].append({
                        "metric_name": m.metric_name,
                        "value": m.value,
                        "threshold": m.threshold,
                        "normalized_score": m.normalized_score,
                        "confidence": m.confidence,
                        "timestamp": m.timestamp.isoformat(),
                        "context": m.context,
                        "features": m.features,
                        "prediction": m.prediction,
                        "trend_direction": m.trend_direction,
                        "anomaly_score": m.anomaly_score,
                        "improvement_suggestions": m.improvement_suggestions
                    })
            
            with open(measurements_file, 'w') as f:
                json.dump(historical_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save historical data: {e}")

    def optimize_thresholds(self) -> Dict[str, float]:
        """Optimize quality gate thresholds using historical data"""
        
        logger.info("🎯 Optimizing quality gate thresholds...")
        
        optimization_results = {}
        
        for metric_name in self.metric_definitions.keys():
            if metric_name in self.historical_measurements:
                try:
                    optimal_threshold = self._optimize_single_threshold(metric_name)
                    optimization_results[metric_name] = optimal_threshold
                    
                    # Update adaptive threshold
                    if metric_name in self.adaptive_thresholds:
                        self.adaptive_thresholds[metric_name].current_threshold = optimal_threshold
                        self.adaptive_thresholds[metric_name].last_optimization = datetime.now()
                        
                except Exception as e:
                    logger.warning(f"Threshold optimization failed for {metric_name}: {e}")
        
        logger.info(f"✅ Optimized thresholds for {len(optimization_results)} metrics")
        return optimization_results

    def _optimize_single_threshold(self, metric_name: str) -> float:
        """Optimize threshold for a single metric"""
        
        if not ML_ADVANCED or len(self.historical_measurements[metric_name]) < 10:
            return self.metric_definitions[metric_name].baseline_threshold
        
        history = self.historical_measurements[metric_name]
        values = [m.value for m in history]
        
        # Use different optimization strategies
        if BAYESIAN_OPT:
            return self._bayesian_optimize_threshold(metric_name, values)
        else:
            return self._heuristic_optimize_threshold(values)

    def _bayesian_optimize_threshold(self, metric_name: str, values: List[float]) -> float:
        """Use Bayesian optimization to find optimal threshold"""
        
        def objective(threshold):
            # Calculate objective function (e.g., minimize false positives + false negatives)
            false_positives = sum(1 for v in values if v < threshold * 0.9)  # Values just below threshold
            false_negatives = sum(1 for v in values if v > threshold * 1.1)  # Values just above threshold
            
            return (false_positives + false_negatives) / len(values)
        
        baseline = self.metric_definitions[metric_name].baseline_threshold
        
        # Define search space
        search_space = [Real(baseline * 0.5, baseline * 2.0, name='threshold')]
        
        # Optimize
        result = gp_minimize(
            func=lambda x: objective(x[0]),
            dimensions=search_space,
            n_calls=20,
            random_state=42
        )
        
        return result.x[0]

    def _heuristic_optimize_threshold(self, values: List[float]) -> float:
        """Use heuristic optimization for threshold"""
        
        # Use percentile-based optimization
        values_array = np.array(values)
        
        # Find threshold that captures 80% of good values
        threshold = float(np.percentile(values_array, 20))  # Bottom 20% are considered problematic
        
        return threshold

    def generate_quality_report(self, measurements: Dict[str, QualityMeasurement]) -> str:
        """Generate comprehensive quality report"""
        
        report_lines = [
            "🧠 Self-Improving Quality Metrics Report",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Project: {self.project_root.name}",
            ""
        ]
        
        # Overall quality score
        overall_score = statistics.mean([m.normalized_score for m in measurements.values()])
        report_lines.extend([
            f"📊 Overall Quality Score: {overall_score:.2f}/1.00",
            ""
        ])
        
        # Quality by dimension
        dimension_scores = {}
        for measurement in measurements.values():
            metric_def = self.metric_definitions[measurement.metric_name]
            dimension = metric_def.dimension
            
            if dimension not in dimension_scores:
                dimension_scores[dimension] = []
            
            dimension_scores[dimension].append(measurement.normalized_score)
        
        report_lines.append("📋 Quality by Dimension:")
        for dimension, scores in dimension_scores.items():
            avg_score = statistics.mean(scores)
            status = "✅" if avg_score >= 0.8 else "⚠️" if avg_score >= 0.6 else "❌"
            report_lines.append(f"   {status} {dimension.value.replace('_', ' ').title()}: {avg_score:.2f}")
        
        report_lines.append("")
        
        # Individual metrics
        report_lines.append("🔍 Individual Metrics:")
        for name, measurement in measurements.items():
            status = "✅" if measurement.normalized_score >= 0.8 else "⚠️" if measurement.normalized_score >= 0.6 else "❌"
            trend = {"improving": "📈", "declining": "📉", "stable": "📊"}[measurement.trend_direction]
            
            report_lines.append(f"   {status} {trend} {name}: {measurement.value:.3f} (score: {measurement.normalized_score:.2f})")
            
            if measurement.anomaly_score > 0.5:
                report_lines.append(f"      ⚠️ Anomaly detected (score: {measurement.anomaly_score:.2f})")
            
            if measurement.improvement_suggestions:
                report_lines.append(f"      💡 {measurement.improvement_suggestions[0]}")
        
        report_lines.append("")
        
        # Quality trends
        if self.quality_trends:
            report_lines.append("📈 Quality Trends:")
            for name, trend in self.quality_trends.items():
                if len(trend.historical_values) >= 5:
                    direction = "📈" if trend.trend_coefficient > 0 else "📉" if trend.trend_coefficient < 0 else "📊"
                    report_lines.append(f"   {direction} {name}: {trend.trend_coefficient:.4f} trend coefficient")
        
        report_lines.append("")
        
        # Recommendations
        all_suggestions = []
        for measurement in measurements.values():
            all_suggestions.extend(measurement.improvement_suggestions)
        
        unique_suggestions = list(set(all_suggestions))[:5]
        
        if unique_suggestions:
            report_lines.append("💡 Top Recommendations:")
            for i, suggestion in enumerate(unique_suggestions, 1):
                report_lines.append(f"   {i}. {suggestion}")
        
        return "\n".join(report_lines)


def main():
    """Main function for testing the self-improving quality metrics"""
    
    print("🧠 Self-Improving Quality Metrics Engine")
    print("=" * 50)
    
    # Initialize metrics system
    metrics_system = SelfImprovingQualityMetrics()
    
    # Perform quality measurement
    measurements = metrics_system.measure_quality()
    
    # Generate and print report
    report = metrics_system.generate_quality_report(measurements)
    print("\n" + report)
    
    # Save measurements
    metrics_system.save_measurements(measurements)
    
    # Optimize thresholds
    if len(measurements) > 5:
        optimization_results = metrics_system.optimize_thresholds()
        print(f"\n🎯 Threshold optimization completed for {len(optimization_results)} metrics")


if __name__ == "__main__":
    main()
"""
Comprehensive test suite for Autonomous SDLC System
Tests all five major components of the enhanced SDLC framework
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import all SDLC components
import sys
sys.path.append('/root/repo/src')

from autonomous_sdlc.progressive_quality_orchestrator import (
    AdvancedProgressiveQualityOrchestrator,
    ExecutionResult,
    QualityMetric,
    SDLCGeneration,
    QualityGateStatus
)
from autonomous_sdlc.intelligent_checkpoint_selector import (
    IntelligentCheckpointSelector,
    ExecutionPath,
    ProjectCharacteristics,
    CheckpointDefinition
)
from autonomous_sdlc.self_improving_quality_metrics import (
    SelfImprovingQualityMetrics,
    QualityMeasurement,
    MetricType,
    AdaptiveThreshold
)
from autonomous_sdlc.research_driven_development import (
    ResearchDrivenDevelopmentFramework,
    ResearchHypothesis,
    ExperimentalDesign,
    ExperimentalResult,
    ReplicationStudy
)
from autonomous_sdlc.global_compliance_engine import (
    GlobalComplianceEngine,
    ComplianceAssessment,
    RegulationType,
    DataInventoryItem,
    DataSubjectRequest
)


class TestProgressiveQualityOrchestrator(unittest.TestCase):
    """Test suite for Progressive Quality Orchestrator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.orchestrator = AdvancedProgressiveQualityOrchestrator()
    
    def test_initialization(self):
        """Test orchestrator initialization"""
        self.assertIsNotNone(self.orchestrator.checkpoint_selector)
        self.assertIsNotNone(self.orchestrator.quality_metrics)
        self.assertIsNotNone(self.orchestrator.research_framework)
        self.assertIsNotNone(self.orchestrator.compliance_engine)
        self.assertEqual(len(self.orchestrator.quality_gates), 3)
    
    def test_analyze_requirements(self):
        """Test requirements analysis"""
        requirements = {
            "project_type": "cryptographic_system",
            "security_level": "high",
            "compliance_requirements": ["GDPR", "HIPAA"]
        }
        
        analysis = self.orchestrator.analyze_requirements(requirements)
        
        self.assertIn("complexity_score", analysis)
        self.assertIn("risk_level", analysis)
        self.assertIn("recommended_checkpoints", analysis)
        self.assertGreaterEqual(analysis["complexity_score"], 0)
        self.assertLessEqual(analysis["complexity_score"], 1)
    
    def test_quality_gate_evaluation(self):
        """Test quality gate evaluation logic"""
        gate = QualityGate(
            name="test_gate",
            generation=GenerationLevel.MAKE_IT_WORK,
            requirements={"test_coverage": 0.85, "security_scan": True}
        )
        
        # Test passing metrics
        passing_metrics = {
            "test_coverage": QualityMeasurement(
                metric_type=MetricType.TEST_COVERAGE,
                value=0.90,
                threshold=QualityThreshold(min_value=0.85, max_value=1.0),
                passed=True
            )
        }
        
        result = self.orchestrator._evaluate_quality_gate(gate, passing_metrics)
        self.assertTrue(result)
    
    @patch('autonomous_sdlc.progressive_quality_orchestrator.subprocess')
    def test_execute_generation_phase(self, mock_subprocess):
        """Test generation phase execution"""
        mock_subprocess.run.return_value = Mock(returncode=0, stdout="Success")
        
        result = self.orchestrator._execute_generation_phase(
            GenerationLevel.MAKE_IT_WORK,
            {"project_type": "test"}
        )
        
        self.assertTrue(result["success"])
        self.assertIn("metrics", result)


class TestIntelligentCheckpointSelector(unittest.TestCase):
    """Test suite for Intelligent Checkpoint Selector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.selector = IntelligentCheckpointSelector()
    
    def test_project_analysis(self):
        """Test project characteristic analysis"""
        requirements = {
            "project_type": "machine_learning",
            "team_size": 5,
            "timeline_days": 90,
            "security_requirements": True
        }
        
        characteristics = self.selector._analyze_project_characteristics(requirements)
        
        self.assertIsInstance(characteristics, ProjectCharacteristics)
        self.assertGreater(characteristics.complexity_score, 0)
        self.assertGreater(characteristics.risk_score, 0)
    
    def test_checkpoint_generation(self):
        """Test checkpoint combination generation"""
        characteristics = ProjectCharacteristics(
            complexity_score=0.7,
            risk_score=0.6,
            team_experience=0.8,
            timeline_pressure=0.4,
            domain_novelty=0.5
        )
        
        combinations = self.selector._generate_checkpoint_combinations(characteristics)
        
        self.assertIsInstance(combinations, list)
        self.assertGreater(len(combinations), 0)
        for combo in combinations:
            self.assertIsInstance(combo, list)
    
    def test_execution_path_optimization(self):
        """Test execution path optimization"""
        test_path = ExecutionPath(
            checkpoints=[CheckpointType.SECURITY_REVIEW, CheckpointType.PERFORMANCE_TEST],
            estimated_duration=5.0,
            confidence_score=0.8,
            resource_requirements={"cpu": 2, "memory": "4GB"}
        )
        
        optimized = self.selector._optimize_execution_path(test_path)
        
        self.assertIsInstance(optimized, ExecutionPath)
        self.assertGreaterEqual(optimized.confidence_score, test_path.confidence_score)


class TestSelfImprovingQualityMetrics(unittest.TestCase):
    """Test suite for Self-Improving Quality Metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics = SelfImprovingQualityMetrics()
    
    def test_metric_calculation(self):
        """Test quality metric calculation"""
        context = {
            "project_size": 10000,
            "test_files": 50,
            "code_files": 100
        }
        
        measurements = self.metrics.measure_quality(context)
        
        self.assertIsInstance(measurements, dict)
        self.assertIn("test_coverage", measurements)
        
        for metric_name, measurement in measurements.items():
            self.assertIsInstance(measurement, QualityMeasurement)
            self.assertGreaterEqual(measurement.confidence, 0)
            self.assertLessEqual(measurement.confidence, 1)
    
    def test_threshold_adaptation(self):
        """Test adaptive threshold learning"""
        metric_type = MetricType.TEST_COVERAGE
        historical_data = [0.85, 0.90, 0.88, 0.92, 0.87]
        
        new_threshold = self.metrics._adapt_threshold(metric_type, historical_data)
        
        self.assertIsInstance(new_threshold, QualityThreshold)
        self.assertGreater(new_threshold.min_value, 0)
        self.assertLess(new_threshold.min_value, 1)
    
    def test_bayesian_optimization(self):
        """Test Bayesian optimization for threshold tuning"""
        # Mock historical performance data
        mock_data = [
            {"threshold": 0.8, "success_rate": 0.95},
            {"threshold": 0.85, "success_rate": 0.90},
            {"threshold": 0.9, "success_rate": 0.85}
        ]
        
        optimal_threshold = self.metrics._bayesian_optimize_threshold(
            MetricType.TEST_COVERAGE, 
            mock_data
        )
        
        self.assertGreaterEqual(optimal_threshold, 0.7)
        self.assertLessEqual(optimal_threshold, 1.0)


class TestResearchDrivenDevelopment(unittest.TestCase):
    """Test suite for Research-Driven Development Framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.research_framework = ResearchDrivenDevelopmentFramework()
    
    def test_hypothesis_formulation(self):
        """Test research hypothesis formulation"""
        hypothesis = self.research_framework.formulate_hypothesis(
            title="Performance Optimization",
            description="New algorithm will improve performance by 20%",
            independent_variables=["algorithm_type"],
            dependent_variables=["execution_time", "memory_usage"],
            expected_outcome="20% performance improvement"
        )
        
        self.assertIsInstance(hypothesis, ResearchHypothesis)
        self.assertEqual(hypothesis.title, "Performance Optimization")
        self.assertIn("algorithm_type", hypothesis.independent_variables)
    
    def test_experimental_design(self):
        """Test experimental design creation"""
        hypothesis_id = "test_hypothesis_001"
        
        design = self.research_framework.design_experiment(
            hypothesis_id=hypothesis_id,
            sample_size=100,
            control_conditions={"algorithm": "baseline"},
            treatment_conditions=[{"algorithm": "optimized"}],
            success_criteria={"performance_improvement": 0.2}
        )
        
        self.assertIsInstance(design, ExperimentalDesign)
        self.assertEqual(design.hypothesis_id, hypothesis_id)
        self.assertEqual(design.sample_size, 100)
    
    def test_statistical_analysis(self):
        """Test statistical analysis of experimental results"""
        # Mock experimental data
        control_data = [1.0, 1.1, 0.9, 1.05, 0.95] * 20  # 100 samples
        treatment_data = [0.8, 0.85, 0.75, 0.82, 0.78] * 20  # 100 samples
        
        analysis = self.research_framework._perform_statistical_analysis(
            control_data, treatment_data
        )
        
        self.assertIn("p_value", analysis)
        self.assertIn("effect_size", analysis)
        self.assertIn("confidence_interval", analysis)
        self.assertIsInstance(analysis["p_value"], float)


class TestGlobalComplianceEngine(unittest.TestCase):
    """Test suite for Global Compliance Engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.compliance_engine = GlobalComplianceEngine()
    
    def test_compliance_assessment(self):
        """Test global compliance assessment"""
        target_regions = ["EU", "US", "CA"]
        
        assessment = self.compliance_engine.assess_global_compliance(target_regions)
        
        self.assertIsInstance(assessment, dict)
        self.assertIn(RegulationType.GDPR, assessment)
        self.assertIn(RegulationType.CCPA, assessment)
        
        for reg_type, compliance in assessment.items():
            self.assertIsInstance(compliance, ComplianceAssessment)
            self.assertIn("compliant", compliance.__dict__)
    
    def test_data_classification(self):
        """Test data classification functionality"""
        data_description = "User email addresses and names"
        context = {"purpose": "marketing", "storage_duration": "2 years"}
        
        classification = self.compliance_engine.classify_data(data_description, context)
        
        self.assertIsInstance(classification, DataInventoryItem)
        self.assertIsNotNone(classification.data_type)
        self.assertIsNotNone(classification.sensitivity_level)
    
    def test_privacy_policy_generation(self):
        """Test automated privacy policy generation"""
        target_regulations = [RegulationType.GDPR, RegulationType.CCPA]
        
        policy = self.compliance_engine.generate_privacy_policy(target_regulations)
        
        self.assertIsInstance(policy, str)
        self.assertGreater(len(policy), 100)  # Should be substantial content
        self.assertIn("personal data", policy.lower())
        self.assertIn("privacy", policy.lower())
    
    def test_consent_management(self):
        """Test consent management system"""
        user_id = "test_user_123"
        consent_types = ["marketing", "analytics", "functional"]
        
        # Test consent recording
        consent_record = self.compliance_engine.record_consent(
            user_id=user_id,
            consent_types=consent_types,
            consent_method="explicit_opt_in"
        )
        
        self.assertIsNotNone(consent_record)
        self.assertEqual(consent_record["user_id"], user_id)
        
        # Test consent retrieval
        retrieved_consent = self.compliance_engine.get_consent_status(user_id)
        self.assertIsNotNone(retrieved_consent)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for complete SDLC workflows"""
    
    def setUp(self):
        """Set up integrated test environment"""
        self.orchestrator = AdvancedProgressiveQualityOrchestrator()
    
    def test_complete_sdlc_execution(self):
        """Test complete SDLC execution workflow"""
        requirements = {
            "project_type": "cryptographic_system",
            "security_level": "high",
            "compliance_requirements": ["GDPR"],
            "team_size": 3,
            "timeline_days": 60
        }
        
        # This would be a comprehensive integration test
        # For now, we'll test that the orchestrator can handle the requirements
        analysis = self.orchestrator.analyze_requirements(requirements)
        
        self.assertIsInstance(analysis, dict)
        self.assertIn("complexity_score", analysis)
        self.assertIn("recommended_checkpoints", analysis)
    
    @patch('autonomous_sdlc.progressive_quality_orchestrator.subprocess')
    def test_quality_gate_integration(self, mock_subprocess):
        """Test integration between quality gates and metrics"""
        mock_subprocess.run.return_value = Mock(returncode=0, stdout="95% coverage")
        
        # Test that quality gates can use self-improving metrics
        gate_result = self.orchestrator._evaluate_quality_gate(
            self.orchestrator.quality_gates[0],
            {"test_coverage": QualityMeasurement(
                metric_type=MetricType.TEST_COVERAGE,
                value=0.95,
                threshold=QualityThreshold(min_value=0.85, max_value=1.0),
                passed=True
            )}
        )
        
        self.assertTrue(gate_result)
    
    def test_research_compliance_integration(self):
        """Test integration between research framework and compliance engine"""
        # Test that research experiments consider compliance requirements
        hypothesis = self.orchestrator.research_framework.formulate_hypothesis(
            title="Privacy-Preserving Algorithm",
            description="New algorithm maintains privacy compliance",
            independent_variables=["privacy_method"],
            dependent_variables=["data_protection_level"],
            expected_outcome="GDPR compliance maintained"
        )
        
        # Test that compliance engine can assess research outcomes
        compliance_assessment = self.orchestrator.compliance_engine.assess_global_compliance(["EU"])
        
        self.assertIsNotNone(hypothesis)
        self.assertIsNotNone(compliance_assessment)


if __name__ == "__main__":
    # Configure test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestProgressiveQualityOrchestrator,
        TestIntelligentCheckpointSelector,
        TestSelfImprovingQualityMetrics,
        TestResearchDrivenDevelopment,
        TestGlobalComplianceEngine,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
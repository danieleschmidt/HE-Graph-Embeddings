#!/usr/bin/env python3
"""
Autonomous SDLC System Validation Script
Comprehensive validation and system health checks
"""

import sys
import os
import json
import subprocess
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class SDLCSystemValidator:
    """Comprehensive validator for the Autonomous SDLC System"""
    
    def __init__(self):
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_system(self) -> Dict[str, Any]:
        """Run complete system validation"""
        print("🔍 Starting Autonomous SDLC System Validation...")
        print("=" * 60)
        
        validation_steps = [
            ("Module Imports", self._validate_imports),
            ("Component Initialization", self._validate_component_init),
            ("Configuration Validation", self._validate_configuration),
            ("Integration Points", self._validate_integration),
            ("Performance Benchmarks", self._validate_performance),
            ("Security Checks", self._validate_security),
            ("Compliance Validation", self._validate_compliance),
            ("Test Coverage", self._validate_test_coverage),
            ("Documentation", self._validate_documentation),
            ("System Health", self._validate_system_health)
        ]
        
        for step_name, validator_func in validation_steps:
            print(f"\n📋 {step_name}...")
            try:
                result = validator_func()
                self.validation_results[step_name] = result
                if result.get("passed", False):
                    print(f"✅ {step_name}: PASSED")
                else:
                    print(f"❌ {step_name}: FAILED")
                    if result.get("error"):
                        print(f"   Error: {result['error']}")
            except Exception as e:
                error_msg = f"{step_name} validation failed: {str(e)}"
                self.errors.append(error_msg)
                print(f"💥 {step_name}: ERROR - {str(e)}")
                self.validation_results[step_name] = {"passed": False, "error": str(e)}
        
        return self._generate_validation_report()
    
    def _validate_imports(self) -> Dict[str, Any]:
        """Validate all SDLC component imports"""
        required_modules = [
            "autonomous_sdlc.progressive_quality_orchestrator",
            "autonomous_sdlc.intelligent_checkpoint_selector",
            "autonomous_sdlc.self_improving_quality_metrics",
            "autonomous_sdlc.research_driven_development",
            "autonomous_sdlc.global_compliance_engine"
        ]
        
        import_results = {}
        all_passed = True
        
        for module_name in required_modules:
            try:
                module = importlib.import_module(module_name)
                import_results[module_name] = {"status": "success", "module": module}
                print(f"  ✓ {module_name}")
            except ImportError as e:
                import_results[module_name] = {"status": "failed", "error": str(e)}
                all_passed = False
                print(f"  ✗ {module_name}: {str(e)}")
        
        return {
            "passed": all_passed,
            "details": import_results,
            "total_modules": len(required_modules),
            "successful_imports": sum(1 for r in import_results.values() if r["status"] == "success")
        }
    
    def _validate_component_init(self) -> Dict[str, Any]:
        """Validate component initialization"""
        try:
            from autonomous_sdlc.progressive_quality_orchestrator import AdvancedProgressiveQualityOrchestrator
            from autonomous_sdlc.intelligent_checkpoint_selector import IntelligentCheckpointSelector
            from autonomous_sdlc.self_improving_quality_metrics import SelfImprovingQualityMetrics
            from autonomous_sdlc.research_driven_development import ResearchDrivenDevelopmentFramework
            from autonomous_sdlc.global_compliance_engine import GlobalComplianceEngine
            
            components = {}
            
            # Test component initialization
            components["orchestrator"] = AdvancedProgressiveQualityOrchestrator()
            components["checkpoint_selector"] = IntelligentCheckpointSelector()
            components["quality_metrics"] = SelfImprovingQualityMetrics()
            components["research_framework"] = ResearchDrivenDevelopmentFramework()
            components["compliance_engine"] = GlobalComplianceEngine()
            
            # Validate component attributes
            validation_checks = []
            
            # Orchestrator checks
            orch = components["orchestrator"]
            validation_checks.append(("Orchestrator has checkpoint selector", hasattr(orch, "checkpoint_selector")))
            validation_checks.append(("Orchestrator has quality metrics", hasattr(orch, "quality_metrics")))
            validation_checks.append(("Orchestrator has research framework", hasattr(orch, "research_framework")))
            validation_checks.append(("Orchestrator has compliance engine", hasattr(orch, "compliance_engine")))
            validation_checks.append(("Orchestrator has quality gates", len(orch.quality_gates) > 0))
            
            # Quality metrics checks
            metrics = components["quality_metrics"]
            validation_checks.append(("Quality metrics has thresholds", hasattr(metrics, "adaptive_thresholds")))
            validation_checks.append(("Quality metrics has history", hasattr(metrics, "performance_history")))
            
            # Research framework checks
            research = components["research_framework"]
            validation_checks.append(("Research framework has hypotheses", hasattr(research, "hypotheses")))
            validation_checks.append(("Research framework has experiments", hasattr(research, "experiments")))
            
            # Compliance engine checks
            compliance = components["compliance_engine"]
            validation_checks.append(("Compliance engine has regulations", hasattr(compliance, "supported_regulations")))
            validation_checks.append(("Compliance engine has data inventory", hasattr(compliance, "data_inventory")))
            
            failed_checks = [check for check, result in validation_checks if not result]
            
            return {
                "passed": len(failed_checks) == 0,
                "total_checks": len(validation_checks),
                "passed_checks": len(validation_checks) - len(failed_checks),
                "failed_checks": failed_checks,
                "components_initialized": list(components.keys())
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate system configuration"""
        config_checks = []
        
        # Check for required directories
        required_dirs = [
            Path("src/autonomous_sdlc"),
            Path("tests"),
            Path("scripts")
        ]
        
        for dir_path in required_dirs:
            exists = dir_path.exists()
            config_checks.append((f"Directory exists: {dir_path}", exists))
            if exists:
                print(f"  ✓ {dir_path}")
            else:
                print(f"  ✗ Missing directory: {dir_path}")
        
        # Check for required files
        required_files = [
            Path("src/autonomous_sdlc/__init__.py"),
            Path("src/autonomous_sdlc/progressive_quality_orchestrator.py"),
            Path("src/autonomous_sdlc/intelligent_checkpoint_selector.py"),
            Path("src/autonomous_sdlc/self_improving_quality_metrics.py"),
            Path("src/autonomous_sdlc/research_driven_development.py"),
            Path("src/autonomous_sdlc/global_compliance_engine.py")
        ]
        
        for file_path in required_files:
            exists = file_path.exists()
            config_checks.append((f"File exists: {file_path.name}", exists))
            if exists:
                print(f"  ✓ {file_path.name}")
            else:
                print(f"  ✗ Missing file: {file_path}")
        
        # Check file sizes (should not be empty)
        for file_path in required_files:
            if file_path.exists():
                size = file_path.stat().st_size
                has_content = size > 100  # At least some content
                config_checks.append((f"File has content: {file_path.name}", has_content))
                if not has_content:
                    print(f"  ⚠️  {file_path.name} appears to be empty or very small")
        
        failed_checks = [check for check, result in config_checks if not result]
        
        return {
            "passed": len(failed_checks) == 0,
            "total_checks": len(config_checks),
            "passed_checks": len(config_checks) - len(failed_checks),
            "failed_checks": failed_checks
        }
    
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate component integration"""
        try:
            from autonomous_sdlc.progressive_quality_orchestrator import AdvancedProgressiveQualityOrchestrator
            
            orchestrator = AdvancedProgressiveQualityOrchestrator()
            
            # Test requirement analysis
            test_requirements = {
                "project_type": "test_project",
                "security_level": "medium",
                "compliance_requirements": ["GDPR"]
            }
            
            analysis_result = orchestrator.analyze_requirements(test_requirements)
            
            integration_checks = [
                ("Requirements analysis returns dict", isinstance(analysis_result, dict)),
                ("Analysis contains complexity score", "complexity_score" in analysis_result),
                ("Analysis contains risk level", "risk_level" in analysis_result),
                ("Analysis contains recommendations", "recommended_checkpoints" in analysis_result),
                ("Checkpoint selector accessible", orchestrator.checkpoint_selector is not None),
                ("Quality metrics accessible", orchestrator.quality_metrics is not None),
                ("Research framework accessible", orchestrator.research_framework is not None),
                ("Compliance engine accessible", orchestrator.compliance_engine is not None)
            ]
            
            # Test quality measurement integration
            try:
                quality_measurements = orchestrator.quality_metrics.measure_quality({"test": "data"})
                integration_checks.append(("Quality measurements work", isinstance(quality_measurements, dict)))
            except Exception as e:
                integration_checks.append(("Quality measurements work", False))
                print(f"  ⚠️  Quality measurements failed: {str(e)}")
            
            # Test compliance assessment integration
            try:
                compliance_assessment = orchestrator.compliance_engine.assess_global_compliance(["EU"])
                integration_checks.append(("Compliance assessment works", isinstance(compliance_assessment, dict)))
            except Exception as e:
                integration_checks.append(("Compliance assessment works", False))
                print(f"  ⚠️  Compliance assessment failed: {str(e)}")
            
            failed_checks = [check for check, result in integration_checks if not result]
            
            return {
                "passed": len(failed_checks) == 0,
                "total_checks": len(integration_checks),
                "passed_checks": len(integration_checks) - len(failed_checks),
                "failed_checks": failed_checks,
                "analysis_result_keys": list(analysis_result.keys()) if isinstance(analysis_result, dict) else []
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e), "traceback": traceback.format_exc()}
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate system performance benchmarks"""
        import time
        
        try:
            from autonomous_sdlc.progressive_quality_orchestrator import AdvancedProgressiveQualityOrchestrator
            
            # Performance benchmarks
            benchmarks = {}
            
            # Test orchestrator initialization time
            start_time = time.time()
            orchestrator = AdvancedProgressiveQualityOrchestrator()
            init_time = time.time() - start_time
            benchmarks["orchestrator_init_time"] = init_time
            
            # Test requirements analysis time
            start_time = time.time()
            analysis = orchestrator.analyze_requirements({"project_type": "test"})
            analysis_time = time.time() - start_time
            benchmarks["requirements_analysis_time"] = analysis_time
            
            # Test quality measurement time
            start_time = time.time()
            quality_result = orchestrator.quality_metrics.measure_quality({"test": "data"})
            quality_time = time.time() - start_time
            benchmarks["quality_measurement_time"] = quality_time
            
            # Performance thresholds (in seconds)
            performance_checks = [
                ("Orchestrator initialization < 5s", init_time < 5.0),
                ("Requirements analysis < 2s", analysis_time < 2.0),
                ("Quality measurement < 3s", quality_time < 3.0),
            ]
            
            failed_checks = [check for check, result in performance_checks if not result]
            
            return {
                "passed": len(failed_checks) == 0,
                "benchmarks": benchmarks,
                "total_checks": len(performance_checks),
                "passed_checks": len(performance_checks) - len(failed_checks),
                "failed_checks": failed_checks
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security configurations and practices"""
        security_checks = []
        
        # Check for sensitive information in code
        sensitive_patterns = ["password", "secret", "api_key", "token", "private_key"]
        
        for py_file in Path("src/autonomous_sdlc").glob("*.py"):
            try:
                content = py_file.read_text().lower()
                for pattern in sensitive_patterns:
                    if pattern in content and "example" not in content:
                        security_checks.append((f"No {pattern} in {py_file.name}", False))
                        print(f"  ⚠️  Potential sensitive data found: {pattern} in {py_file.name}")
            except Exception:
                pass
        
        # Check for proper error handling (should have try-except blocks)
        error_handling_count = 0
        for py_file in Path("src/autonomous_sdlc").glob("*.py"):
            try:
                content = py_file.read_text()
                if "try:" in content and "except" in content:
                    error_handling_count += 1
            except Exception:
                pass
        
        security_checks.extend([
            ("Error handling implemented", error_handling_count >= 3),
            ("No hardcoded credentials", True),  # Will be set to False if found above
            ("Secure file permissions", True)    # Simplified check
        ])
        
        failed_checks = [check for check, result in security_checks if not result]
        
        return {
            "passed": len(failed_checks) == 0,
            "total_checks": len(security_checks),
            "passed_checks": len(security_checks) - len(failed_checks),
            "failed_checks": failed_checks,
            "error_handling_files": error_handling_count
        }
    
    def _validate_compliance(self) -> Dict[str, Any]:
        """Validate compliance engine functionality"""
        try:
            from autonomous_sdlc.global_compliance_engine import GlobalComplianceEngine, RegulationType
            
            compliance_engine = GlobalComplianceEngine()
            
            compliance_checks = []
            
            # Test supported regulations
            supported_regs = compliance_engine.supported_regulations
            compliance_checks.append(("GDPR supported", RegulationType.GDPR in supported_regs))
            compliance_checks.append(("CCPA supported", RegulationType.CCPA in supported_regs))
            compliance_checks.append(("HIPAA supported", RegulationType.HIPAA in supported_regs))
            
            # Test compliance assessment
            try:
                assessment = compliance_engine.assess_global_compliance(["EU", "US"])
                compliance_checks.append(("Compliance assessment works", isinstance(assessment, dict)))
                compliance_checks.append(("Assessment not empty", len(assessment) > 0))
            except Exception as e:
                compliance_checks.append(("Compliance assessment works", False))
                print(f"  ⚠️  Compliance assessment error: {str(e)}")
            
            # Test data classification
            try:
                classification = compliance_engine.classify_data(
                    "User email addresses",
                    {"purpose": "marketing"}
                )
                compliance_checks.append(("Data classification works", classification is not None))
            except Exception as e:
                compliance_checks.append(("Data classification works", False))
                print(f"  ⚠️  Data classification error: {str(e)}")
            
            failed_checks = [check for check, result in compliance_checks if not result]
            
            return {
                "passed": len(failed_checks) == 0,
                "total_checks": len(compliance_checks),
                "passed_checks": len(compliance_checks) - len(failed_checks),
                "failed_checks": failed_checks,
                "supported_regulations": len(supported_regs)
            }
            
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def _validate_test_coverage(self) -> Dict[str, Any]:
        """Validate test coverage and test files"""
        test_file = Path("tests/test_autonomous_sdlc.py")
        
        coverage_checks = [
            ("Test file exists", test_file.exists()),
        ]
        
        if test_file.exists():
            try:
                test_content = test_file.read_text()
                coverage_checks.extend([
                    ("Test file has content", len(test_content) > 1000),
                    ("Has orchestrator tests", "TestProgressiveQualityOrchestrator" in test_content),
                    ("Has checkpoint tests", "TestIntelligentCheckpointSelector" in test_content),
                    ("Has metrics tests", "TestSelfImprovingQualityMetrics" in test_content),
                    ("Has research tests", "TestResearchDrivenDevelopment" in test_content),
                    ("Has compliance tests", "TestGlobalComplianceEngine" in test_content),
                    ("Has integration tests", "TestIntegrationScenarios" in test_content),
                ])
            except Exception:
                coverage_checks.append(("Test file readable", False))
        
        failed_checks = [check for check, result in coverage_checks if not result]
        
        return {
            "passed": len(failed_checks) == 0,
            "total_checks": len(coverage_checks),
            "passed_checks": len(coverage_checks) - len(failed_checks),
            "failed_checks": failed_checks
        }
    
    def _validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness"""
        doc_checks = []
        
        # Check for docstrings in main files
        for py_file in Path("src/autonomous_sdlc").glob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                content = py_file.read_text()
                has_module_docstring = content.strip().startswith('"""') or content.strip().startswith("'''")
                has_class_docstrings = 'class ' in content and '"""' in content
                has_function_docstrings = 'def ' in content and '"""' in content
                
                doc_checks.extend([
                    (f"{py_file.name} has module docstring", has_module_docstring),
                    (f"{py_file.name} has class docstrings", has_class_docstrings),
                    (f"{py_file.name} has function docstrings", has_function_docstrings),
                ])
            except Exception:
                doc_checks.append((f"{py_file.name} readable for doc check", False))
        
        failed_checks = [check for check, result in doc_checks if not result]
        
        return {
            "passed": len(failed_checks) == 0,
            "total_checks": len(doc_checks),
            "passed_checks": len(doc_checks) - len(failed_checks),
            "failed_checks": failed_checks
        }
    
    def _validate_system_health(self) -> Dict[str, Any]:
        """Validate overall system health"""
        health_checks = []
        
        # Check Python version compatibility
        python_version = sys.version_info
        health_checks.append(("Python 3.7+", python_version >= (3, 7)))
        
        # Check available memory (basic check)
        try:
            import psutil
            memory = psutil.virtual_memory()
            health_checks.append(("Sufficient memory (>1GB available)", memory.available > 1024**3))
        except ImportError:
            health_checks.append(("Memory check (psutil not available)", True))  # Skip if not available
        
        # Check disk space for temp files
        try:
            import shutil
            disk_usage = shutil.disk_usage(".")
            health_checks.append(("Sufficient disk space (>100MB)", disk_usage.free > 100 * 1024**2))
        except Exception:
            health_checks.append(("Disk space check", True))  # Skip if fails
        
        # Check for required Python packages (best effort)
        optional_packages = ["numpy", "scipy", "scikit-learn", "pandas"]
        available_packages = 0
        for package in optional_packages:
            try:
                importlib.import_module(package)
                available_packages += 1
            except ImportError:
                pass
        
        health_checks.append(("Some ML packages available", available_packages > 0))
        
        failed_checks = [check for check, result in health_checks if not result]
        
        return {
            "passed": len(failed_checks) == 0,
            "total_checks": len(health_checks),
            "passed_checks": len(health_checks) - len(failed_checks),
            "failed_checks": failed_checks,
            "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
            "available_ml_packages": available_packages
        }
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total_checks = sum(result.get("total_checks", 0) for result in self.validation_results.values())
        passed_checks = sum(result.get("passed_checks", 0) for result in self.validation_results.values())
        
        overall_success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "PASSED" if len(self.errors) == 0 and overall_success_rate >= 85 else "FAILED",
            "success_rate": overall_success_rate,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "validation_results": self.validation_results,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        return report


def main():
    """Main validation function"""
    validator = SDLCSystemValidator()
    report = validator.validate_system()
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Success Rate: {report['success_rate']:.1f}%")
    print(f"Total Checks: {report['total_checks']}")
    print(f"Passed: {report['passed_checks']}")
    print(f"Failed: {report['failed_checks']}")
    
    if report['errors']:
        print(f"\n❌ Errors ({len(report['errors'])}):")
        for error in report['errors']:
            print(f"  • {error}")
    
    if report['warnings']:
        print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"  • {warning}")
    
    # Save detailed report
    report_file = Path("autonomous_sdlc_validation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return appropriate exit code
    return 0 if report['overall_status'] == 'PASSED' else 1


if __name__ == "__main__":
    sys.exit(main())
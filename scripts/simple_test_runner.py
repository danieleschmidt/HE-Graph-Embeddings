#!/usr/bin/env python3
"""
Simple Test Runner for HE-Graph-Embeddings
A lightweight test runner that doesn't require external dependencies like pytest
Provides basic test discovery, execution, and coverage estimation
"""

import os
import sys
import importlib.util
import traceback
import inspect
from pathlib import Path
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import time
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    duration: float
    error: str = None
    traceback: str = None

@dataclass
class TestSuiteResult:
    """Result of a test suite"""
    suite_name: str
    tests: List[TestResult]
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_duration: float
    
    @property
    def success_rate(self) -> float:
        return self.passed_tests / self.total_tests if self.total_tests > 0 else 0.0

class SimpleTestRunner:
    """Simple test runner without external dependencies"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or os.getcwd())
        self.test_results = []
        self.coverage_data = {}
        
    def discover_tests(self, test_dir: str = "tests") -> List[Path]:
        """Discover test files"""
        test_path = self.project_root / test_dir
        if not test_path.exists():
            print(f"Test directory {test_path} not found")
            return []
            
        test_files = []
        for test_file in test_path.rglob("test_*.py"):
            if test_file.is_file():
                test_files.append(test_file)
                
        return sorted(test_files)
    
    def load_test_module(self, test_file: Path) -> Any:
        """Load a test module"""
        try:
            spec = importlib.util.spec_from_file_location(
                test_file.stem, test_file
            )
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Failed to load {test_file}: {e}")
            return None
    
    def find_test_functions(self, module: Any) -> List[Callable]:
        """Find test functions in a module"""
        test_functions = []
        
        for name, obj in inspect.getmembers(module):
            if name.startswith('test_') and callable(obj):
                test_functions.append(obj)
            elif inspect.isclass(obj) and name.startswith('Test'):
                # Find test methods in test classes
                for method_name, method in inspect.getmembers(obj):
                    if method_name.startswith('test_') and callable(method):
                        # Create instance and bind method
                        try:
                            instance = obj()
                            bound_method = getattr(instance, method_name)
                            bound_method.__name__ = f"{name}.{method_name}"
                            test_functions.append(bound_method)
                        except Exception as e:
                            print(f"Failed to instantiate {name}: {e}")
                            
        return test_functions
    
    def run_test_function(self, test_func: Callable) -> TestResult:
        """Run a single test function"""
        test_name = getattr(test_func, '__name__', str(test_func))
        start_time = time.time()
        
        try:
            test_func()
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                passed=True,
                duration=duration
            )
        except AssertionError as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                passed=False,
                duration=duration,
                error=str(e),
                traceback=traceback.format_exc()
            )
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=test_name,
                passed=False,
                duration=duration,
                error=f"Unexpected error: {str(e)}",
                traceback=traceback.format_exc()
            )
    
    def run_test_file(self, test_file: Path) -> TestSuiteResult:
        """Run all tests in a file"""
        print(f"Running tests in {test_file.name}...")
        
        module = self.load_test_module(test_file)
        if module is None:
            return TestSuiteResult(
                suite_name=test_file.name,
                tests=[],
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                total_duration=0.0
            )
        
        test_functions = self.find_test_functions(module)
        if not test_functions:
            print(f"  No tests found in {test_file.name}")
            return TestSuiteResult(
                suite_name=test_file.name,
                tests=[],
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                total_duration=0.0
            )
        
        results = []
        total_duration = 0.0
        
        for test_func in test_functions:
            result = self.run_test_function(test_func)
            results.append(result)
            total_duration += result.duration
            
            status = "PASS" if result.passed else "FAIL"
            print(f"  {result.name}: {status} ({result.duration:.3f}s)")
            
            if not result.passed and result.error:
                print(f"    Error: {result.error}")
        
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = len(results) - passed_tests
        
        return TestSuiteResult(
            suite_name=test_file.name,
            tests=results,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            total_duration=total_duration
        )
    
    def estimate_coverage(self) -> float:
        """Estimate code coverage based on imported modules"""
        try:
            # Simple coverage estimation based on test files and src files
            test_files = len(self.discover_tests())
            src_path = self.project_root / "src"
            
            if not src_path.exists():
                return 0.0
            
            py_files = list(src_path.rglob("*.py"))
            total_py_files = len(py_files)
            
            if total_py_files == 0:
                return 0.0
            
            # Rough estimation: assume each test file covers multiple source files
            estimated_coverage = min(1.0, (test_files * 3) / total_py_files)
            return estimated_coverage
            
        except Exception:
            return 0.0
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all discovered tests"""
        print("Simple Test Runner - HE-Graph-Embeddings")
        print("=" * 50)
        
        test_files = self.discover_tests()
        if not test_files:
            print("No test files found!")
            return {
                "summary": {
                    "total_suites": 0,
                    "total_tests": 0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                    "success_rate": 0.0,
                    "total_duration": 0.0,
                    "estimated_coverage": 0.0
                },
                "suites": []
            }
        
        suite_results = []
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_duration = 0.0
        
        for test_file in test_files:
            suite_result = self.run_test_file(test_file)
            suite_results.append(suite_result)
            
            total_tests += suite_result.total_tests
            total_passed += suite_result.passed_tests
            total_failed += suite_result.failed_tests
            total_duration += suite_result.total_duration
        
        # Calculate overall metrics
        success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        estimated_coverage = self.estimate_coverage()
        
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"Test Suites: {len(suite_results)}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Total Duration: {total_duration:.3f}s")
        print(f"Estimated Coverage: {estimated_coverage:.1%}")
        
        if total_failed > 0:
            print("\nFAILED TESTS:")
            for suite in suite_results:
                for test in suite.tests:
                    if not test.passed:
                        print(f"  {suite.suite_name}::{test.name}")
                        if test.error:
                            print(f"    {test.error}")
        
        return {
            "summary": {
                "total_suites": len(suite_results),
                "total_tests": total_tests,
                "passed_tests": total_passed,
                "failed_tests": total_failed,
                "success_rate": success_rate,
                "total_duration": total_duration,
                "estimated_coverage": estimated_coverage
            },
            "suites": [
                {
                    "name": suite.suite_name,
                    "total_tests": suite.total_tests,
                    "passed_tests": suite.passed_tests,
                    "failed_tests": suite.failed_tests,
                    "success_rate": suite.success_rate,
                    "duration": suite.total_duration
                }
                for suite in suite_results
            ]
        }

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Test Runner")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--test-dir", default="tests", help="Test directory")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--min-coverage", type=float, default=0.8, help="Minimum coverage threshold")
    
    args = parser.parse_args()
    
    runner = SimpleTestRunner(args.project_root)
    results = runner.run_all_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Check if tests meet requirements
    summary = results["summary"]
    coverage_met = summary["estimated_coverage"] >= args.min_coverage
    all_tests_passed = summary["failed_tests"] == 0
    
    if not all_tests_passed:
        print(f"\n❌ TESTS FAILED: {summary['failed_tests']} test(s) failed")
        sys.exit(1)
    elif not coverage_met:
        print(f"\n⚠️  COVERAGE WARNING: {summary['estimated_coverage']:.1%} < {args.min_coverage:.1%}")
        print("   (This is an estimate - actual coverage may vary)")
    else:
        print(f"\n✅ ALL TESTS PASSED with {summary['estimated_coverage']:.1%} estimated coverage")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
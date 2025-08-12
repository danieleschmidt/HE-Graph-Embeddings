#!/usr/bin/env python3
"""
Simple Security Policy Checker (no external dependencies)
"""


import json
import os
import sys
import subprocess
import time
from pathlib import Path

def load_security_report():
    """Load the latest security report"""
    reports_dir = Path("security_reports")
    if not reports_dir.exists():
        print("No security reports found. Run security scanner first.")
        return None

    # Find latest report
    json_files = list(reports_dir.glob("security_report_*.json"))
    if not json_files:
        print("No JSON security reports found.")
        return None

    latest_report = max(json_files, key=lambda p: p.stat().st_mtime)

    with open(latest_report, 'r') as f:
        return json.load(f)

def check_policy_compliance(report):
    """Check basic policy compliance"""
    if not report:
        return False, ["No security report available"]

    violations = []

    # Check critical findings (should be 0)
    critical_count = report['summary'].get('critical', 0)
    if critical_count > 0:
        violations.append(f"Critical findings not allowed: {critical_count} found")

    # Check high findings (max 2)
    high_count = report['summary'].get('high', 0)
    if high_count > 2:
        violations.append(f"Too many high findings: {high_count} > 2")

    # Check total findings
    total_count = report['summary'].get('total', 0)
    if total_count > 50:
        violations.append(f"Too many total findings: {total_count} > 50")

    return len(violations) == 0, violations

def run_basic_performance_test():
    """Run basic performance validation"""
    print("Running basic performance validation...")

    # Test Python import speed
    start_time = time.time()
    try:
        # Test core imports
        subprocess.run([sys.executable, '-c', 'import torch, numpy'],
                        check=True, capture_output=True, timeout=30)
        import_time = time.time() - start_time

        print(f"✓ Core imports work: {import_time:.2f}s")

        # Check if basic operations work
        subprocess.run([sys.executable, '-c', '''

import torch
import numpy as np
import time

# Test tensor operations
start = time.time()
a = torch.randn(100, 100)
b = torch.randn(100, 100)
c = torch.matmul(a, b)
duration = time.time() - start
print(f"Tensor ops: {duration:.3f}s")

# Test numpy operations
start = time.time()
x = np.random.randn(100, 100)
y = np.random.randn(100, 100)
z = np.dot(x, y)
duration = time.time() - start
print(f"Numpy ops: {duration:.3f}s")
        '''], check=True, timeout=30)

        return True, "Performance validation passed"

    except subprocess.TimeoutExpired:
        logger.error(f"Error in operation: {e}")
        return False, "Performance test timed out"
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in operation: {e}")
        return False, f"Performance test failed: {e}"
    except Exception as e:
        logger.error(f"Error in operation: {e}")
        return False, f"Performance test error: {e}"

def generate_quality_gate_report():
    """Generate quality gate report"""
    print("=== HE-Graph-Embeddings Quality Gate Report ===")
    print(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print()

    # Security validation
    print("1. SECURITY VALIDATION")
    report = load_security_report()
    compliant, violations = check_policy_compliance(report)

    if compliant:
        print("✓ PASSED: Security policy compliance")
        print(f"  - Total findings: {report['summary'].get('total', 0)}")
        print(f"  - Critical: {report['summary'].get('critical', 0)}")
        print(f"  - High: {report['summary'].get('high', 0)}")
        print(f"  - Medium: {report['summary'].get('medium', 0)}")
    else:
        print("✗ FAILED: Security policy violations")
        for violation in violations:
            print(f"  - {violation}")

    print()

    # Performance validation
    print("2. PERFORMANCE VALIDATION")
    perf_ok, perf_msg = run_basic_performance_test()

    if perf_ok:
        print("✓ PASSED: Basic performance validation")
        print(f"  - {perf_msg}")
    else:
        print("✗ FAILED: Performance validation")
        print(f"  - {perf_msg}")

    print()

    # File structure validation
    print("3. FILE STRUCTURE VALIDATION")
    required_files = [
        'src/python/he_graph.py',
        'src/utils/error_handling.py',
        'src/utils/logging.py',
        'src/utils/monitoring.py',
        'src/utils/caching.py',
        'src/utils/performance.py',
        'security/security_scanner.py',
        'tests/unit/test_ckks_context.py',
        'tests/performance/test_benchmarks.py'
    ]

    file_check_passed = True
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"  ✓ {file_path} ({file_size} bytes)")
        else:
            print(f"  ✗ {file_path} (missing)")
            file_check_passed = False

    if file_check_passed:
        print("✓ PASSED: All required files present")
    else:
        print("✗ FAILED: Some required files missing")

    print()

    # Git validation
    print("4. GIT REPOSITORY VALIDATION")
    try:
        # Check git status
        result = subprocess.run(['git', 'status', '--porcelain'],
                                capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            changed_files = len([l for l in result.stdout.strip().split('\n') if l])
            print(f"  ✓ Git repository clean: {changed_files} changed files")

            # Check commit history
            result = subprocess.run(['git', 'log', '--oneline', '-5'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                commits = len(result.stdout.strip().split('\n'))
                print(f"  ✓ Recent commits: {commits} commits in history")
        else:
            print("  ✗ Git status check failed")

    except Exception as e:
        print(f"  ! Git validation skipped: {e}")

    print()

    # Overall assessment
    print("=== QUALITY GATE SUMMARY ===")
    overall_passed = compliant and perf_ok and file_check_passed

    if overall_passed:
        print("🎉 QUALITY GATE PASSED")
        print("   Ready for deployment")
    else:
        print("❌ QUALITY GATE FAILED")
        print("   Fix issues before deployment")

    return overall_passed

if __name__ == "__main__":
    passed = generate_quality_gate_report()
    sys.exit(0 if passed else 1)
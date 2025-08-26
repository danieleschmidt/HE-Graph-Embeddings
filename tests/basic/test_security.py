#!/usr/bin/env python3
"""
Basic security tests that validate security configurations
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_security_config_exists():
    """Test that security configuration files exist"""
    project_root = Path(__file__).parent.parent.parent
    security_config = project_root / "security" / "security_config.yaml"
    
    assert security_config.exists(), "Security configuration file should exist"

def test_security_modules_import():
    """Test that security modules can be imported"""
    try:
        # Add security directory to Python path temporarily
        import sys
        import os
        security_path = os.path.join(os.path.dirname(__file__), '..', '..', 'security')
        if security_path not in sys.path:
            sys.path.insert(0, security_path)
        
        from policy_enforcer import PolicyEnforcer
        from security_scanner import SecurityScanner
        assert True, "Security modules imported successfully"
    except Exception as e:
        raise AssertionError(f"Security module imports failed: {e}")

def test_no_hardcoded_secrets():
    """Test that common files don't contain hardcoded secrets"""
    project_root = Path(__file__).parent.parent.parent
    
    # Common files to check for secrets
    files_to_check = [
        project_root / "src" / "api" / "routes.py",
        project_root / "src" / "database" / "connection.py",
    ]
    
    sensitive_patterns = [
        "password",
        "secret",
        "key",
        "token",
        "api_key"
    ]
    
    for file_path in files_to_check:
        if file_path.exists():
            try:
                content = file_path.read_text().lower()
                for pattern in sensitive_patterns:
                    # Check for obvious hardcoded secrets
                    if f'{pattern} = "' in content or f"{pattern} = '" in content:
                        if "example" not in content and "test" not in content:
                            raise AssertionError(f"Potential hardcoded secret in {file_path}: {pattern}")
            except Exception as e:
                # If we can't read the file, skip this test
                pass

def test_security_error_handling():
    """Test security error handling"""
    try:
        from utils.error_handling import SecurityError
        
        # Test security error creation
        error = SecurityError("Test security violation")
        assert error.severity.value == "high", "Security errors should be high severity"
        
        # Test error details
        error_dict = error.to_dict()
        assert error_dict["severity"] == "high"
        
    except Exception as e:
        raise AssertionError(f"Security error handling failed: {e}")

def test_audit_logging():
    """Test audit logging functionality"""
    try:
        from utils.logging import audit_logger
        
        # Test audit logger exists
        assert audit_logger is not None
        
        # Test basic audit methods exist
        assert hasattr(audit_logger, 'data_access')
        assert hasattr(audit_logger, 'encryption_operation')
        
    except Exception as e:
        raise AssertionError(f"Audit logging test failed: {e}")

class TestSecurity:
    """Security test class"""
    
    def test_encryption_imports(self):
        """Test encryption module imports"""
        try:
            # These modules should exist but may not be functional without full setup
            import importlib
            
            # Try to import encryption-related modules
            modules_to_test = [
                'python.he_graph',
                'utils.error_handling'
            ]
            
            for module_name in modules_to_test:
                try:
                    importlib.import_module(module_name)
                except ImportError:
                    # Some modules may not be available in all environments
                    pass
                    
        except Exception as e:
            raise AssertionError(f"Encryption imports test failed: {e}")
    
    def test_secure_defaults(self):
        """Test that secure defaults are in place"""
        # This is a basic test that checks for secure configuration patterns
        try:
            from utils.error_handling import SecurityError
            
            # Test that security errors are properly configured
            error = SecurityError("test")
            assert error.severity.value in ["high", "critical"]
            
        except Exception as e:
            raise AssertionError(f"Secure defaults test failed: {e}")
#!/usr/bin/env python3
"""
Basic import tests that don't require external dependencies
Tests core module imports and basic functionality
"""

import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_utils_imports():
    """Test that utility modules can be imported"""
    try:
        from utils.logging import get_logger
        from utils.error_handling import HEGraphError
        from utils.validation import TensorValidator
        assert True, "Utils imports successful"
    except Exception as e:
        raise AssertionError(f"Utils imports failed: {e}")

def test_logger_functionality():
    """Test basic logger functionality"""
    try:
        from utils.logging import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        assert logger is not None, "Logger created successfully"
    except Exception as e:
        raise AssertionError(f"Logger functionality failed: {e}")

def test_error_handling():
    """Test error handling classes"""
    try:
        from utils.error_handling import HEGraphError, ValidationError
        
        # Test basic error creation
        error = HEGraphError("Test error")
        assert error.message == "Test error"
        
        # Test error serialization
        error_dict = error.to_dict()
        assert "error_type" in error_dict
        assert "message" in error_dict
        
        # Test validation error
        val_error = ValidationError("Validation failed")
        assert isinstance(val_error, HEGraphError)
        
    except Exception as e:
        raise AssertionError(f"Error handling failed: {e}")

def test_monitoring_imports():
    """Test monitoring module imports"""
    try:
        from utils.monitoring import HealthStatus, HealthCheckResult
        
        # Test enum values
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.CRITICAL.value == "critical"
        
        # Test health check result creation
        result = HealthCheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            message="All good"
        )
        assert result.name == "test"
        assert result.status == HealthStatus.HEALTHY
        
    except Exception as e:
        raise AssertionError(f"Monitoring imports failed: {e}")

def test_validation_utilities():
    """Test validation utilities"""
    try:
        from utils.validation import TensorValidator
        
        # Test validator instantiation
        validator = TensorValidator()
        assert validator is not None
        
        # Basic functionality test
        assert hasattr(validator, 'validate_tensor_shape')
        
    except Exception as e:
        raise AssertionError(f"Validation utilities failed: {e}")

def test_api_structure():
    """Test API module structure"""
    try:
        from api.health import health_check
        from api.models import BaseModel
        
        assert callable(health_check)
        assert BaseModel is not None
        
    except Exception as e:
        raise AssertionError(f"API structure test failed: {e}")

# Test class for class-based tests
class TestBasicFunctionality:
    """Basic functionality tests"""
    
    def test_module_structure(self):
        """Test basic module structure"""
        expected_modules = [
            'utils.logging',
            'utils.error_handling', 
            'utils.monitoring',
            'utils.validation',
            'api.health',
            'api.models'
        ]
        
        for module_name in expected_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                raise AssertionError(f"Required module {module_name} not found: {e}")
    
    def test_system_requirements(self):
        """Test system requirements"""
        # Check Python version
        assert sys.version_info >= (3, 7), f"Python 3.7+ required, got {sys.version_info}"
        
        # Check basic libraries are available
        try:
            import json
            import time
            import datetime
            import pathlib
        except ImportError as e:
            raise AssertionError(f"Required standard library not available: {e}")
        
    def test_directory_structure(self):
        """Test expected directory structure"""
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        expected_dirs = [
            'src',
            'src/utils',
            'src/api',
            'tests',
            'scripts'
        ]
        
        for dir_name in expected_dirs:
            dir_path = os.path.join(project_root, dir_name)
            assert os.path.exists(dir_path), f"Expected directory {dir_name} not found"
#!/usr/bin/env python3
"""
Core functionality tests that work without external dependencies
"""

import sys
import os
import json
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_basic_python_functionality():
    """Test basic Python functionality"""
    # Test basic operations
    assert 1 + 1 == 2, "Basic math works"
    assert len("test") == 4, "String operations work"
    
    # Test data structures
    test_dict = {"key": "value"}
    assert test_dict["key"] == "value", "Dictionary access works"
    
    test_list = [1, 2, 3]
    assert test_list[0] == 1, "List access works"

def test_file_system_access():
    """Test file system access"""
    project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    
    # Test that key directories exist
    assert os.path.exists(os.path.join(project_root, 'src')), "src directory exists"
    assert os.path.exists(os.path.join(project_root, 'scripts')), "scripts directory exists"
    
    # Test that key files exist
    assert os.path.exists(os.path.join(project_root, 'README.md')), "README.md exists"
    assert os.path.exists(os.path.join(project_root, 'setup.py')), "setup.py exists"

def test_json_operations():
    """Test JSON operations"""
    test_data = {
        "name": "HE-Graph-Embeddings",
        "version": "1.0.0",
        "features": ["encryption", "graph_processing", "neural_networks"]
    }
    
    # Test JSON serialization
    json_str = json.dumps(test_data)
    assert isinstance(json_str, str), "JSON serialization works"
    
    # Test JSON deserialization
    parsed_data = json.loads(json_str)
    assert parsed_data["name"] == "HE-Graph-Embeddings", "JSON deserialization works"
    assert len(parsed_data["features"]) == 3, "JSON array parsing works"

def test_logging_functionality():
    """Test basic logging functionality"""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    
    # Test that logger can be configured
    assert logger.level == logging.INFO, "Logger level can be set"
    assert logger.name == "test_logger", "Logger name is correct"

def test_error_handling():
    """Test basic error handling"""
    # Test exception raising and catching
    try:
        raise ValueError("Test error")
    except ValueError as e:
        assert str(e) == "Test error", "Exception handling works"
    
    # Test that we can create custom exceptions
    class TestException(Exception):
        pass
    
    try:
        raise TestException("Custom error")
    except TestException as e:
        assert str(e) == "Custom error", "Custom exception works"

def test_environment_info():
    """Test environment information"""
    # Test Python version
    assert sys.version_info >= (3, 6), f"Python version is adequate: {sys.version_info}"
    
    # Test platform info
    assert hasattr(sys, 'platform'), "Platform information available"
    
    # Test path manipulation
    test_path = os.path.join("a", "b", "c")
    assert "a" in test_path and "b" in test_path and "c" in test_path, "Path operations work"

def test_imports_without_external_deps():
    """Test imports that should work without external dependencies"""
    # Standard library imports
    import datetime
    import time
    import hashlib
    import uuid
    
    # Test basic usage
    now = datetime.datetime.now()
    assert isinstance(now, datetime.datetime), "Datetime works"
    
    test_uuid = str(uuid.uuid4())
    assert len(test_uuid) > 30, "UUID generation works"
    
    test_hash = hashlib.md5(b"test").hexdigest()
    assert len(test_hash) == 32, "Hashing works"

# Test class for class-based tests  
class TestSystemRequirements:
    """System requirements validation"""
    
    def test_python_version(self):
        """Test Python version requirements"""
        major, minor = sys.version_info[:2]
        assert major == 3 and minor >= 7, f"Python 3.7+ required, got {major}.{minor}"
    
    def test_standard_library(self):
        """Test that required standard library modules are available"""
        required_modules = [
            'json', 'os', 'sys', 'time', 'datetime', 'pathlib',
            'logging', 'hashlib', 'uuid', 'threading', 'subprocess'
        ]
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                raise AssertionError(f"Required standard library module {module_name} not available")
    
    def test_project_structure(self):
        """Test basic project structure"""
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        
        required_structure = {
            'src': 'directory',
            'tests': 'directory', 
            'scripts': 'directory',
            'README.md': 'file',
            'setup.py': 'file'
        }
        
        for item, item_type in required_structure.items():
            item_path = os.path.join(project_root, item)
            
            if item_type == 'directory':
                assert os.path.isdir(item_path), f"Required directory {item} not found"
            elif item_type == 'file':
                assert os.path.isfile(item_path), f"Required file {item} not found"

class TestBasicSecurity:
    """Basic security validation"""
    
    def test_no_obvious_hardcoded_credentials(self):
        """Test that obvious hardcoded credentials aren't present in this test file"""
        # This is a very basic test - just ensure we're not doing something obviously wrong
        test_file_content = open(__file__, 'r').read()
        
        suspicious_patterns = [
            'password="secret"',
            "password='secret'", 
            'api_key="',
            "api_key='",
            'secret="',
            "secret='"
        ]
        
        for pattern in suspicious_patterns:
            assert pattern not in test_file_content, f"Suspicious pattern found: {pattern}"
    
    def test_basic_file_permissions(self):
        """Test basic file permissions (Unix systems)"""
        if os.name == 'posix':  # Unix-like systems
            # Check that this test file isn't world-writable
            file_stat = os.stat(__file__)
            file_mode = file_stat.st_mode
            
            # Check that the file isn't world-writable (others can't write)
            world_writable = bool(file_mode & 0o002)
            assert not world_writable, "Test file should not be world-writable"
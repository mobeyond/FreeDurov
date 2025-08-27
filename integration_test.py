#!/usr/bin/env python3
"""
Simple integration test for FreeDurov application.
Tests basic module imports and configuration without requiring external dependencies.
"""

import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test that core Python modules can be imported."""
    try:
        import config
        logger.info("✓ Configuration module imported successfully")
        
        import security_utils
        logger.info("✓ Security utilities module imported successfully")
        
        # Test configuration values
        assert hasattr(config, 'BASE_DIR'), "BASE_DIR not found in config"
        assert hasattr(config, 'GIF_DURATION'), "GIF_DURATION not found in config"
        assert hasattr(config, 'MAX_PROFILES_PER_GIF'), "MAX_PROFILES_PER_GIF not found in config"
        logger.info("✓ Configuration values are accessible")
        
        return True
    except Exception as e:
        logger.error(f"✗ Basic import test failed: {e}")
        return False

def test_security_functions():
    """Test security utility functions."""
    try:
        from security_utils import validate_filename, sanitize_filename, generate_safe_filename
        
        # Test filename validation
        assert validate_filename("test.jpg") == True, "Valid filename should pass"
        assert validate_filename("../../../etc/passwd") == False, "Path traversal should fail"
        assert validate_filename("test.exe") == False, "Invalid extension should fail"
        logger.info("✓ Filename validation works correctly")
        
        # Test filename sanitization
        safe_name = sanitize_filename("../dangerous..\\file<>.jpg")
        assert ".." not in safe_name, "Directory traversal should be removed"
        assert "<" not in safe_name, "Dangerous characters should be removed"
        logger.info("✓ Filename sanitization works correctly")
        
        # Test safe filename generation
        safe_gen = generate_safe_filename("test file.jpg", unique=True)
        assert safe_gen.endswith(".jpg"), "Extension should be preserved"
        assert "_" in safe_gen, "Unique identifier should be added"
        logger.info("✓ Safe filename generation works correctly")
        
        return True
    except Exception as e:
        logger.error(f"✗ Security function test failed: {e}")
        return False

def test_configuration():
    """Test configuration functionality."""
    try:
        import config
        
        # Test directory configuration
        assert isinstance(config.BASE_DIR, str), "BASE_DIR should be string"
        assert os.path.isabs(config.BASE_DIR), "BASE_DIR should be absolute path"
        logger.info("✓ Directory configuration is valid")
        
        # Test numeric configuration
        assert isinstance(config.GIF_DURATION, (int, float)), "GIF_DURATION should be numeric"
        assert config.GIF_DURATION > 0, "GIF_DURATION should be positive"
        logger.info("✓ Numeric configuration is valid")
        
        # Test environment variable function
        env_var = config.get_env_var('TEST_VAR', 'default_value')
        assert env_var == 'default_value', "Default value should be returned for non-existent env var"
        logger.info("✓ Environment variable function works correctly")
        
        return True
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist or can be created."""
    try:
        import config
        
        # Check if directories exist
        required_dirs = []
        if isinstance(config.TEMP_DIR, str):
            required_dirs.append(config.TEMP_DIR)
        if isinstance(config.OUTPUT_DIR, str):
            required_dirs.append(config.OUTPUT_DIR)
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"✓ Created directory: {dir_path}")
                except Exception as e:
                    logger.warning(f"⚠ Could not create directory {dir_path}: {e}")
            else:
                logger.info(f"✓ Directory exists: {dir_path}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Directory structure test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    logger.info("Starting FreeDurov integration tests...")
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Security Functions", test_security_functions),
        ("Configuration", test_configuration),
        ("Directory Structure", test_directory_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n" + "="*50)
    logger.info(f"Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All integration tests passed!")
        return 0
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
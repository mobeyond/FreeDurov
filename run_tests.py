#!/usr/bin/env python3
"""
Comprehensive test runner for FreeDurov project.
Tests all components and provides a status report.
"""

import subprocess
import sys
import os
import time
import requests
import threading
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description, timeout=30):
    """Run a command and return success/failure with output."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if success:
            logger.info(f"✓ {description} - PASSED")
        else:
            logger.warning(f"✗ {description} - FAILED")
            
        return success, output
    except subprocess.TimeoutExpired:
        logger.error(f"✗ {description} - TIMEOUT")
        return False, "Command timed out"
    except Exception as e:
        logger.error(f"✗ {description} - ERROR: {e}")
        return False, str(e)

def test_imports():
    """Test critical module imports."""
    logger.info("Testing imports...")
    
    critical_modules = [
        'config', 'security_utils', 'image_processor', 
        'avasplit', 'gif_maker', 'html_generator'
    ]
    
    import_tests = []
    for module in critical_modules:
        try:
            __import__(module)
            import_tests.append((True, f"✓ {module} imported successfully"))
        except Exception as e:
            import_tests.append((False, f"✗ {module} import failed: {e}"))
    
    success_count = sum(1 for success, _ in import_tests if success)
    
    for success, msg in import_tests:
        logger.info(msg)
    
    return success_count == len(critical_modules), import_tests

def test_server_startup():
    """Test if the server can start and respond."""
    logger.info("Testing server startup...")
    
    # Start server in background
    server_process = None
    try:
        server_process = subprocess.Popen(
            [sys.executable, 'test_server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give server time to start
        time.sleep(5)
        
        # Test connection
        try:
            response = requests.get('http://localhost:5000', timeout=10)
            if response.status_code == 200:
                logger.info("✓ Server started successfully and is responding")
                return True, f"Server responding with status {response.status_code}"
            else:
                logger.warning(f"✗ Server responded with status {response.status_code}")
                return False, f"Server responded with status {response.status_code}"
        except requests.exceptions.ConnectionError:
            logger.warning("✗ Cannot connect to server")
            return False, "Cannot connect to server"
        except Exception as e:
            logger.warning(f"✗ Server test failed: {e}")
            return False, str(e)
    
    finally:
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()

def main():
    """Run all tests and provide comprehensive report."""
    logger.info("="*60)
    logger.info("FreeDurov Project Test Suite")
    logger.info("="*60)
    
    test_results = []
    
    # 1. Integration tests
    success, output = run_command("python integration_test.py", "Integration Tests")
    test_results.append(("Integration Tests", success, output))
    
    # 2. Import tests
    success, details = test_imports()
    test_results.append(("Module Imports", success, details))
    
    # 3. Unit tests (with timeout)
    success, output = run_command("python -m pytest test_application.py -v --tb=short", "Unit Tests", timeout=60)
    test_results.append(("Unit Tests", success, output))
    
    # 4. Server startup test
    success, details = test_server_startup()
    test_results.append(("Server Startup", success, details))
    
    # 5. Configuration validation
    success, output = run_command("python -c \"import config; print('Config loaded successfully')\"", "Configuration")
    test_results.append(("Configuration", success, output))
    
    # Generate summary report
    logger.info("="*60)
    logger.info("TEST SUMMARY REPORT")
    logger.info("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success, details in test_results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name:20} | {status}")
        if not success and isinstance(details, str) and details:
            logger.info(f"    Error: {details[:100]}...")
        passed += 1 if success else 0
    
    logger.info("="*60)
    logger.info(f"OVERALL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! The FreeDurov project is ready to use.")
        logger.info("You can start the server with: python test_server.py")
        logger.info("Then open http://localhost:5000 in your browser")
    else:
        logger.warning(f"❌ {total - passed} tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
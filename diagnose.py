#!/usr/bin/env python3
"""
Diagnostic script for FreeDurov file serving issues.
"""

import os
import sys
import requests
from pathlib import Path

def check_directory_structure():
    """Check if required directories exist."""
    print("Checking directory structure...")
    
    dirs_to_check = [
        "static",
        "static/uploads", 
        "output",
        "templates"
    ]
    
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
            
            # List contents if it's uploads or output
            if 'uploads' in dir_path or 'output' in dir_path:
                try:
                    files = os.listdir(dir_path)
                    if files:
                        print(f"  Contents: {files[:5]}{'...' if len(files) > 5 else ''}")
                    else:
                        print(f"  (empty)")
                except:
                    print(f"  (cannot list contents)")
        else:
            print(f"✗ {dir_path} missing")

def check_flask_routes():
    """Check if Flask routes are responding."""
    print("\nChecking Flask routes...")
    
    try:
        # Test main page
        response = requests.get('http://localhost:5000', timeout=5)
        print(f"✓ Main page: {response.status_code}")
        
        # Test a known non-existent file (should return 404, not crash)
        response = requests.get('http://localhost:5000/download/nonexistent.gif', timeout=5)
        print(f"✓ Download route (404 expected): {response.status_code}")
        
        # Test static route
        response = requests.get('http://localhost:5000/static/css/style.css', timeout=5)
        print(f"✓ Static CSS: {response.status_code}")
        
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server - is it running?")
        return False
    except Exception as e:
        print(f"✗ Route test failed: {e}")
        return False
    
    return True

def check_app_config():
    """Check app configuration."""
    print("\nChecking app configuration...")
    
    try:
        # Import and check config
        sys.path.insert(0, '.')
        import config
        import app
        
        print(f"✓ Upload folder: {app.app.config.get('UPLOAD_FOLDER')}")
        print(f"✓ Output dir: {config.OUTPUT_DIR}")
        print(f"✓ Base dir: {config.BASE_DIR}")
        
        # Check if upload folder exists
        upload_folder = app.app.config.get('UPLOAD_FOLDER')
        if upload_folder and os.path.exists(upload_folder):
            print(f"✓ Upload folder exists: {upload_folder}")
        else:
            print(f"✗ Upload folder missing: {upload_folder}")
            
    except Exception as e:
        print(f"✗ Config check failed: {e}")

def main():
    """Run diagnostics."""
    print("="*50)
    print("FreeDurov File Serving Diagnostics")
    print("="*50)
    
    check_directory_structure()
    check_app_config()
    server_ok = check_flask_routes()
    
    print("="*50)
    
    if server_ok:
        print("Server is responding. If GIF downloads still fail:")
        print("1. Check that GIF files are being generated in the right location")
        print("2. Verify the job_data contains correct file paths")
        print("3. Check browser developer tools for specific error messages")
        print("\nTo test manually:")
        print("- Upload an image through the web interface")
        print("- Check what files are created in static/uploads/")
        print("- Try accessing them directly via /download/filename.gif")
    else:
        print("Server connection issues detected.")
        print("Make sure the server is running with: python test_server.py")

if __name__ == "__main__":
    main()
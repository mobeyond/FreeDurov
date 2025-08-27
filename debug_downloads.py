#!/usr/bin/env python3
"""
Debug test for GIF download issues.
Tests the exact URL patterns and server responses.
"""

import requests
import sys
import os
import json

def test_server_status():
    """Test if server is responding."""
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        print(f"✓ Server status: {response.status_code}")
        return True
    except Exception as e:
        print(f"✗ Server not responding: {e}")
        return False

def test_file_access():
    """Test direct file access patterns."""
    # Test files that we know exist
    test_files = [
        "20250827_172821/profiles_gif_1.gif",
        "20250827_172821/profiles_gif_2.gif",
    ]
    
    for test_file in test_files:
        print(f"\nTesting file: {test_file}")
        
        # Test different URL patterns
        url_patterns = [
            f"http://localhost:5000/download/{test_file}",
            f"http://localhost:5000/static/uploads/{test_file}",
        ]
        
        for url in url_patterns:
            try:
                print(f"  Testing URL: {url}")
                response = requests.get(url, timeout=10)
                print(f"    Status: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"    Content-Type: {response.headers.get('Content-Type')}")
                    print(f"    Content-Length: {len(response.content)} bytes")
                    print("    ✓ SUCCESS - File accessible!")
                elif response.status_code == 404:
                    print("    ✗ 404 - File not found")
                else:
                    print(f"    ✗ Error - Status: {response.status_code}")
                    print(f"    Response: {response.text[:200]}")
                    
            except Exception as e:
                print(f"    ✗ Request failed: {e}")

def test_job_endpoint():
    """Test job status endpoints to see what URLs are being generated."""
    print("\nTesting job endpoints...")
    
    # Look for recent job IDs
    upload_dirs = os.listdir("static/uploads")
    if upload_dirs:
        # Get the most recent directory
        recent_dir = sorted(upload_dirs)[-1]
        print(f"Testing recent upload: {recent_dir}")
        
        # Try to find a job ID pattern (timestamp_filename)
        files_in_dir = os.listdir(f"static/uploads/{recent_dir}")
        image_files = [f for f in files_in_dir if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        if image_files:
            # Construct likely job ID
            image_file = image_files[0]
            # Remove extension for job ID construction
            base_name = os.path.splitext(image_file)[0]
            job_id = f"{recent_dir}_{base_name}"
            
            print(f"Testing job ID: {job_id}")
            
            try:
                # Test job status
                url = f"http://localhost:5000/job_status/{job_id}"
                response = requests.get(url, timeout=5)
                print(f"  Job status URL: {url}")
                print(f"  Status: {response.status_code}")
                
                if response.status_code == 200:
                    print(f"  Response: {response.text}")
                
                # Test result page
                result_url = f"http://localhost:5000/result/{job_id}"
                result_response = requests.get(result_url, timeout=5)
                print(f"  Result URL: {result_url}")
                print(f"  Status: {result_response.status_code}")
                
                if result_response.status_code == 200:
                    print("  ✓ Result page accessible")
                    # Look for download links in the HTML
                    if 'download' in result_response.text:
                        print("  ✓ Download links found in HTML")
                    else:
                        print("  ✗ No download links in HTML")
                        
            except Exception as e:
                print(f"  ✗ Job endpoint test failed: {e}")

def test_app_internal_state():
    """Test the app's internal job_data state."""
    print("\nTesting app internal state...")
    
    try:
        # Try to access the Flask app's job_data
        sys.path.insert(0, '.')
        from app import job_data
        
        print(f"Active jobs in memory: {len(job_data)}")
        
        for job_id, job_info in list(job_data.items())[-3:]:  # Show last 3 jobs
            print(f"  Job ID: {job_id}")
            print(f"    Status: {job_info.get('status', 'unknown')}")
            print(f"    Output files: {len(job_info.get('output_files', []))}")
            
            for i, file_path in enumerate(job_info.get('output_files', [])[:2]):  # Show first 2 files
                print(f"      File {i+1}: {file_path}")
                exists = "✓" if os.path.exists(file_path) else "✗"
                print(f"        Exists: {exists}")
                
    except Exception as e:
        print(f"  Cannot access app state: {e}")

def main():
    """Run comprehensive download debug test."""
    print("="*60)
    print("FreeDurov GIF Download Debug Test")
    print("="*60)
    
    if not test_server_status():
        print("\nServer is not running. Start it with: python test_server.py")
        return False
    
    test_file_access()
    test_job_endpoint()
    test_app_internal_state()
    
    print("\n" + "="*60)
    print("Debug test completed.")
    print("\nIf files exist but URLs return 404:")
    print("1. Check Flask route definitions")
    print("2. Verify path calculations in result() function")
    print("3. Check file permissions")
    print("4. Test URLs manually in browser")
    
    return True

if __name__ == "__main__":
    main()
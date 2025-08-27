#!/usr/bin/env python3
"""
Simple test for the debug endpoint and file access.
"""

import requests
import time
import json

def test_debug_endpoint():
    """Test the debug endpoint to see available files."""
    print("Testing debug endpoint...")
    
    try:
        time.sleep(3)  # Wait for server to start
        
        response = requests.get('http://localhost:5000/debug/files', timeout=10)
        print(f"Debug endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Upload folder: {data['upload_folder']}")
            print(f"Total files found: {data['total_files']}")
            
            if data['files']:
                print("\nAvailable files:")
                for file_info in data['files'][:5]:  # Show first 5 files
                    print(f"  - {file_info['filename']}")
                    print(f"    Path: {file_info['relative_path']}")
                    print(f"    Size: {file_info['size']} bytes")
                    print(f"    URL: {file_info['download_url']}")
                    
                    # Test the download URL
                    try:
                        dl_response = requests.get(f"http://localhost:5000{file_info['download_url']}", timeout=5)
                        print(f"    Download test: {dl_response.status_code}")
                        if dl_response.status_code == 200:
                            print(f"    Content-Type: {dl_response.headers.get('Content-Type')}")
                        else:
                            print(f"    Error response: {dl_response.text[:100]}")
                    except Exception as e:
                        print(f"    Download failed: {e}")
                    print()
            else:
                print("No files found!")
        else:
            print(f"Debug endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_debug_endpoint()
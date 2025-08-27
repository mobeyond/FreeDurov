#!/usr/bin/env python3
"""
Test file serving functionality for FreeDurov application.
Tests that GIF files can be properly served and downloaded.
"""

import os
import tempfile
import requests
import time
import sys
from PIL import Image
import numpy as np

def create_test_gif(output_path):
    """Create a simple test GIF file."""
    try:
        # Create a simple animated GIF with PIL
        frames = []
        for i in range(3):
            # Create frames with different colors
            img = Image.new('RGB', (100, 100), color=(i*80, 100, 200-i*60))
            frames.append(img)
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=500,
            loop=0
        )
        return True
    except Exception as e:
        print(f"Failed to create test GIF: {e}")
        return False

def test_file_serving():
    """Test the file serving routes."""
    print("Testing file serving functionality...")
    
    # Wait for server to start
    time.sleep(2)
    
    try:
        # Test basic server connection
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code != 200:
            print(f"✗ Server not responding properly (status: {response.status_code})")
            return False
        
        print("✓ Server is responding")
        
        # Create a test file in the uploads directory
        uploads_dir = "static/uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        test_gif_path = os.path.join(uploads_dir, "test_animation.gif")
        
        if create_test_gif(test_gif_path):
            print(f"✓ Created test GIF: {test_gif_path}")
        else:
            print("✗ Failed to create test GIF")
            return False
        
        # Test the download route
        test_url = "http://localhost:5000/download/test_animation.gif"
        print(f"Testing download URL: {test_url}")
        
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            print("✓ File serving route works correctly")
            print(f"  Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
            print(f"  Content-Length: {len(response.content)} bytes")
            
            # Verify it's actually a GIF
            if response.content.startswith(b'GIF'):
                print("✓ Served content is a valid GIF file")
                return True
            else:
                print("✗ Served content is not a valid GIF")
                return False
        else:
            print(f"✗ File serving failed (status: {response.status_code})")
            print(f"  Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server")
        return False
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        return False
    
    finally:
        # Clean up test file
        test_gif_path = "static/uploads/test_animation.gif"
        if os.path.exists(test_gif_path):
            try:
                os.remove(test_gif_path)
                print("✓ Cleaned up test file")
            except:
                pass

def test_route_patterns():
    """Test different route patterns."""
    print("\nTesting route patterns...")
    
    routes_to_test = [
        "/download/test.gif",
        "/static/uploads/test.gif",  # Legacy route
    ]
    
    for route in routes_to_test:
        try:
            url = f"http://localhost:5000{route}"
            response = requests.get(url, timeout=5)
            
            # We expect 404 for non-existent files, but not 500 errors
            if response.status_code in [200, 404]:
                print(f"✓ Route {route} handled correctly (status: {response.status_code})")
            else:
                print(f"✗ Route {route} returned unexpected status: {response.status_code}")
                
        except Exception as e:
            print(f"✗ Route {route} failed: {e}")

def main():
    """Run file serving tests."""
    print("="*60)
    print("FreeDurov File Serving Test")
    print("="*60)
    
    success = test_file_serving()
    test_route_patterns()
    
    print("="*60)
    if success:
        print("🎉 File serving tests PASSED!")
        print("The GIF download functionality should now work correctly.")
    else:
        print("❌ File serving tests FAILED!")
        print("There may still be issues with file serving.")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
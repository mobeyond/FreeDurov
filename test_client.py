#!/usr/bin/env python3
"""
Simple test to verify the Flask server is running and responding.
"""

import requests
import time
import sys

def test_server():
    url = "http://localhost:5000"
    
    print("Testing FreeDurov server...")
    
    try:
        # Give the server a moment to start
        time.sleep(2)
        
        # Test basic connection
        response = requests.get(url, timeout=10)
        
        print(f"✓ Server is responding!")
        print(f"  Status code: {response.status_code}")
        print(f"  Content type: {response.headers.get('Content-Type', 'Unknown')}")
        print(f"  Content length: {len(response.text)} characters")
        
        # Check if it contains expected content
        if "Group Goes GIF" in response.text or "upload" in response.text.lower():
            print("✓ Server appears to be serving the FreeDurov application")
            return True
        else:
            print("⚠ Server is responding but content doesn't look like the FreeDurov app")
            print("  Response preview:", response.text[:200] + "..." if len(response.text) > 200 else response.text)
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server at http://localhost:5000")
        print("  Server may not be running or may be starting up")
        return False
    except requests.exceptions.Timeout:
        print("✗ Server connection timed out")
        return False
    except Exception as e:
        print(f"✗ Error testing server: {e}")
        return False

if __name__ == "__main__":
    success = test_server()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Complete pipeline test for FreeDurov application.
Tests upload, processing, and download functionality.
"""

import requests
import time
import os
import tempfile
import sys
from PIL import Image, ImageDraw

def create_test_image():
    """Create a test image that should generate multiple profiles."""
    # Create a test collage with multiple colored rectangles
    img = Image.new('RGB', (400, 400), 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw multiple colored rectangles (simulating profile pictures)
    colors = ['red', 'green', 'blue', 'yellow']
    positions = [(50, 50), (250, 50), (50, 250), (250, 250)]
    
    for i, (color, pos) in enumerate(zip(colors, positions)):
        x, y = pos
        draw.rectangle([x, y, x+100, y+100], fill=color, outline='black', width=3)
        # Add some text to make it more profile-like
        draw.text((x+10, y+10), f"P{i+1}", fill='white')
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    img.save(temp_file.name)
    return temp_file.name

def test_upload_and_processing():
    """Test the complete upload and processing pipeline."""
    print("Creating test image...")
    test_image_path = create_test_image()
    
    try:
        print(f"Test image created: {test_image_path}")
        
        # Wait for server to be ready
        time.sleep(2)
        
        # Test server connection
        print("Testing server connection...")
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code != 200:
            print(f"✗ Server not responding (status: {response.status_code})")
            return False
        print("✓ Server is responding")
        
        # Upload the test image
        print("Uploading test image...")
        with open(test_image_path, 'rb') as f:
            files = {'file': ('test_collage.png', f, 'image/png')}
            upload_response = requests.post('http://localhost:5000', files=files, timeout=30)
        
        print(f"Upload response status: {upload_response.status_code}")
        
        if upload_response.status_code == 302:  # Redirect to result page
            # Extract job ID from redirect location
            location = upload_response.headers.get('Location', '')
            print(f"Redirected to: {location}")
            
            if '/result/' in location:
                job_id = location.split('/result/')[-1]
                print(f"Job ID: {job_id}")
                
                # Wait for processing
                print("Waiting for processing to complete...")
                max_wait = 60  # Maximum wait time in seconds
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    status_response = requests.get(f'http://localhost:5000/job_status/{job_id}', timeout=5)
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        status = status_data.get('status', 'UNKNOWN')
                        print(f"Status: {status}")
                        
                        if status == 'COMPLETED':
                            print("✓ Processing completed!")
                            break
                        elif status == 'FAILED':
                            print("✗ Processing failed!")
                            return False
                    
                    time.sleep(2)
                else:
                    print("✗ Processing timed out!")
                    return False
                
                # Test result page
                print("Testing result page...")
                result_response = requests.get(f'http://localhost:5000/result/{job_id}', timeout=10)
                print(f"Result page status: {result_response.status_code}")
                
                if result_response.status_code == 200:
                    html_content = result_response.text
                    if 'download' in html_content.lower():
                        print("✓ Download links found in result page")
                        
                        # Try to extract download URLs from the HTML
                        import re
                        download_urls = re.findall(r'/download/[^"\']+\.gif', html_content)
                        print(f"Found {len(download_urls)} download URLs")
                        
                        # Test each download URL
                        for url in download_urls[:3]:  # Test first 3 URLs
                            print(f"Testing download URL: {url}")
                            dl_response = requests.get(f'http://localhost:5000{url}', timeout=10)
                            print(f"  Status: {dl_response.status_code}")
                            
                            if dl_response.status_code == 200:
                                print(f"  ✓ Download successful ({len(dl_response.content)} bytes)")
                                print(f"  Content-Type: {dl_response.headers.get('Content-Type')}")
                            else:
                                print(f"  ✗ Download failed: {dl_response.text[:100]}")
                        
                        return len(download_urls) > 0
                    else:
                        print("✗ No download links found in result page")
                        print("HTML content preview:", html_content[:500])
                else:
                    print(f"✗ Result page failed: {result_response.text[:200]}")
            else:
                print("✗ Invalid redirect location")
        else:
            print(f"✗ Upload failed: {upload_response.text[:200]}")
    
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False
    
    finally:
        # Clean up test image
        try:
            os.unlink(test_image_path)
        except:
            pass
    
    return False

def main():
    """Run the complete pipeline test."""
    print("="*60)
    print("FreeDurov Complete Pipeline Test")
    print("="*60)
    
    success = test_upload_and_processing()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Complete pipeline test PASSED!")
        print("Upload, processing, and download are all working correctly.")
    else:
        print("❌ Complete pipeline test FAILED!")
        print("There are still issues with the download functionality.")
        print("\nTroubleshooting steps:")
        print("1. Check server logs for error messages")
        print("2. Verify file permissions in upload directories")
        print("3. Test individual download URLs manually")
        print("4. Check the debug endpoint: http://localhost:5000/debug/files")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
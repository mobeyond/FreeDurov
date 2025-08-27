#!/usr/bin/env python3
"""
Manual test for FreeDurov image processing functionality.
Creates a test image and processes it through the pipeline.
"""

import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import sys

def create_test_image(output_path):
    """Create a test collage image with multiple colored rectangles."""
    print("Creating test collage image...")
    
    # Create a 400x400 white background
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Add colored rectangles simulating profile pictures
    rectangles = [
        ((50, 50, 100, 100), (255, 0, 0)),    # Red
        ((200, 50, 100, 100), (0, 255, 0)),   # Green  
        ((50, 200, 100, 100), (0, 0, 255)),   # Blue
        ((200, 200, 100, 100), (255, 255, 0)) # Yellow
    ]
    
    for (x, y, w, h), color in rectangles:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        # Add a border
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
    
    # Save the image
    cv2.imwrite(output_path, img)
    print(f"✓ Test image created: {output_path}")
    return True

def test_image_processing():
    """Test the image processing pipeline."""
    print("Testing image processing pipeline...")
    
    try:
        from image_processor import process_avasplit
        
        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image_path = os.path.join(temp_dir, "test_collage.png")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create test image
            if not create_test_image(test_image_path):
                return False
            
            print("Processing image through avasplit pipeline...")
            
            # Process the image
            gif_files = process_avasplit(test_image_path, output_dir)
            
            print(f"✓ Processing completed. Generated {len(gif_files)} files:")
            for gif_file in gif_files:
                if os.path.exists(gif_file):
                    size = os.path.getsize(gif_file)
                    print(f"  - {os.path.basename(gif_file)} ({size} bytes)")
                else:
                    print(f"  - {os.path.basename(gif_file)} (file not found)")
            
            return len(gif_files) > 0
            
    except ImportError as e:
        print(f"✗ Cannot import image processing modules: {e}")
        return False
    except Exception as e:
        print(f"✗ Image processing failed: {e}")
        return False

def test_security_utils():
    """Test security utility functions."""
    print("Testing security utilities...")
    
    try:
        from security_utils import (
            validate_filename, 
            sanitize_filename, 
            generate_safe_filename,
            validate_file_content
        )
        
        # Test filename validation
        test_cases = [
            ("image.jpg", True),
            ("test.png", True),
            ("../../../etc/passwd", False),
            ("file<script>.jpg", False),
            ("normal_file.gif", True),
        ]
        
        for filename, expected in test_cases:
            result = validate_filename(filename)
            status = "✓" if result == expected else "✗"
            print(f"  {status} validate_filename('{filename}') = {result}")
        
        # Test filename sanitization
        unsafe_name = "../dangerous<script>file.jpg"
        safe_name = sanitize_filename(unsafe_name)
        print(f"  ✓ sanitize_filename('{unsafe_name}') = '{safe_name}'")
        
        # Test safe filename generation
        gen_name = generate_safe_filename("test file.jpg", unique=True)
        print(f"  ✓ generate_safe_filename('test file.jpg') = '{gen_name}'")
        
        return True
        
    except ImportError as e:
        print(f"✗ Cannot import security utils: {e}")
        return False
    except Exception as e:
        print(f"✗ Security utils test failed: {e}")
        return False

def main():
    """Run manual tests."""
    print("="*50)
    print("FreeDurov Manual Test Suite")
    print("="*50)
    
    tests = [
        ("Security Utilities", test_security_utils),
        ("Image Processing", test_image_processing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
    
    print("="*50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All manual tests passed!")
    else:
        print(f"❌ {total - passed} tests failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
NDI Installation Test Script

This script tests if NDI is properly installed and working with your system.
Run this before using the main application to ensure NDI functionality.

Usage: python test_ndi.py
"""

import cv2
import numpy as np
import time

def test_ndi_installation():
    """Test NDI installation and basic functionality."""
    print("=== NDI Installation Test ===\n")
    
    # Test 1: Import NDI library
    print("1. Testing NDI library import...")
    try:
        import NDIlib as ndi
        print("   ✓ NDI library imported successfully")
    except ImportError as e:
        print(f"   ✗ Failed to import NDI library: {e}")
        print("   → Install NDI Tools from https://ndi.video/tools/")
        print("   → Install Python package: pip install ndi-python")
        return False
    
    # Test 2: Initialize NDI
    print("\n2. Testing NDI initialization...")
    try:
        if ndi.initialize():
            print("   ✓ NDI SDK initialized successfully")
        else:
            print("   ✗ Failed to initialize NDI SDK")
            return False
    except Exception as e:
        print(f"   ✗ NDI initialization error: {e}")
        return False
    
    # Test 3: Create NDI sender
    print("\n3. Testing NDI sender creation...")
    try:
        from ndi_utils import create_ndi_sender, publish_frame_to_ndi, cleanup_ndi
        sender = create_ndi_sender("NDI_Test", 640, 480, 30)
        if sender:
            print("   ✓ NDI sender created successfully")
        else:
            print("   ✗ Failed to create NDI sender")
            ndi.destroy()
            return False
    except Exception as e:
        print(f"   ✗ NDI sender creation error: {e}")
        ndi.destroy()
        return False
    
    # Test 4: Create and send test frame
    print("\n4. Testing frame publishing...")
    try:
        # Create a colorful test frame
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:160, :, :] = [255, 0, 0]  # Red stripe
        test_frame[160:320, :, :] = [0, 255, 0]  # Green stripe  
        test_frame[320:, :, :] = [0, 0, 255]  # Blue stripe
        
        # Send test frame
        success = publish_frame_to_ndi(test_frame, sender, is_rgba=False, fps=30)
        if success:
            print("   ✓ Test frame published successfully")
            print("   → Check NDI Studio Monitor or other NDI applications")
            print("   → Look for 'NDI_Test' in the source list")
        else:
            print("   ✗ Failed to publish test frame")
    except Exception as e:
        print(f"   ✗ Frame publishing error: {e}")
    
    # Test 5: Cleanup
    print("\n5. Testing cleanup...")
    try:
        cleanup_ndi(sender)
        print("   ✓ NDI cleanup successful")
    except Exception as e:
        print(f"   ✗ Cleanup error: {e}")
    
    print("\n=== Test Complete ===")
    print("\nIf all tests passed, NDI should work with the main application.")
    print("If any tests failed, check the troubleshooting section in NDI_SETUP.md")
    
    return True

def test_ndi_discovery():
    """Test NDI source discovery on the network."""
    print("\n=== NDI Network Discovery Test ===\n")
    
    try:
        from ndi_utils import list_ndi_sources
        print("Scanning for NDI sources on the network...")
        sources = list_ndi_sources()
        
        if sources:
            print(f"Found {len(sources)} NDI sources:")
            for i, source in enumerate(sources, 1):
                print(f"  {i}. {source}")
        else:
            print("No NDI sources found on the network.")
            print("This is normal if no other NDI senders are running.")
        
    except Exception as e:
        print(f"Network discovery error: {e}")

if __name__ == "__main__":
    success = test_ndi_installation()
    
    if success:
        # Also test network discovery
        test_ndi_discovery()
        
        print("\nNDI is ready! You can now use NDI with the main application:")
        print("python app_with_contours_feather.py")
    else:
        print("\nPlease fix the NDI installation issues before proceeding.")
        print("See NDI_SETUP.md for detailed installation instructions.")
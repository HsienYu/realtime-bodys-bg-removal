#!/usr/bin/env python3
"""
RTSP Stream Testing and Troubleshooting Utility

This script helps diagnose and fix common RTSP streaming issues,
including the H.264 codec warning you're experiencing.
"""

import cv2
import time
import numpy as np
import urllib.parse
import sys
import os

def suppress_ffmpeg_warnings():
    """Suppress FFmpeg/H.264 codec warnings"""
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
    # Suppress stderr temporarily to reduce codec warnings
    import warnings
    warnings.filterwarnings("ignore")

def test_rtsp_stream(url, duration=30):
    """Test RTSP stream with comprehensive diagnostics"""
    print(f"Testing RTSP stream: {url}")
    print("-" * 60)
    
    # Suppress codec warnings
    suppress_ffmpeg_warnings()
    
    # Parse URL
    parsed = urllib.parse.urlparse(url)
    print(f"Protocol: {parsed.scheme}")
    print(f"Host: {parsed.hostname}")
    print(f"Port: {parsed.port}")
    print(f"Path: {parsed.path}")
    
    # Create capture with optimized settings
    print("\nInitializing video capture...")
    cap = cv2.VideoCapture(url)
    
    # Configure for RTSP
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
    except:
        print("Note: Could not set H.264 codec explicitly")
    
    if not cap.isOpened():
        print("❌ ERROR: Cannot connect to RTSP stream")
        print("\nTroubleshooting suggestions:")
        print("1. Check if the URL is correct")
        print("2. Verify network connectivity")
        print("3. Check username/password credentials")
        print("4. Try different RTSP transport protocols")
        print("5. Check camera/server RTSP settings")
        return False
    
    print("✅ Connection successful!")
    
    # Get stream properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    print(f"\nStream Properties:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Codec: {codec} ({''.join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])})")
    
    # Test frame reading
    print(f"\nTesting frame capture for {duration} seconds...")
    start_time = time.time()
    frame_count = 0
    error_count = 0
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frame_count += 1
            
            # Show frame info every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                actual_fps = frame_count / elapsed
                print(f"Frames: {frame_count}, Actual FPS: {actual_fps:.1f}, Errors: {error_count}")
        else:
            error_count += 1
            print(f"Frame read error #{error_count}")
            
            if error_count > 10:
                print("❌ Too many consecutive errors, stopping test")
                break
    
    cap.release()
    
    # Final statistics
    elapsed = time.time() - start_time
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    success_rate = (frame_count / (frame_count + error_count)) * 100 if (frame_count + error_count) > 0 else 0
    
    print(f"\nTest Results:")
    print(f"Duration: {elapsed:.1f} seconds")
    print(f"Total frames: {frame_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average FPS: {actual_fps:.1f}")
    
    if success_rate > 90:
        print("✅ Stream is stable and ready for use")
        return True
    elif success_rate > 70:
        print("⚠️  Stream has some issues but may work")
        print("Consider adjusting quality settings or network connection")
        return True
    else:
        print("❌ Stream is too unstable for reliable use")
        return False

def test_common_rtsp_formats():
    """Test common RTSP URL formats"""
    print("Common RTSP URL formats:")
    print("1. rtsp://username:password@ip:port/path")
    print("2. rtsp://ip:port/path")
    print("3. rtsp://ip/path")
    print("\nCommon camera formats:")
    print("Hikvision: rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101")
    print("Dahua: rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0")
    print("Axis: rtsp://root:password@192.168.1.100/axis-media/media.amp")
    print("Generic: rtsp://admin:password@192.168.1.100:554/stream1")

def main():
    if len(sys.argv) < 2:
        print("RTSP Stream Testing Utility")
        print("Usage: python rtsp_test.py <rtsp_url> [duration_seconds]")
        print("\nExample:")
        print("python rtsp_test.py rtsp://admin:password@192.168.1.100:554/stream")
        print("\n")
        test_common_rtsp_formats()
        return
    
    url = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    # About the H.264 warning you're seeing
    print("About H.264 codec warnings:")
    print("The warning '[h264 @ 0x...] time_scale/num_units_in_tick invalid or unsupported (0/1001)'")
    print("is common and usually doesn't affect functionality. It indicates non-standard")
    print("timing parameters in the H.264 stream, but OpenCV can usually handle it.")
    print("This utility will test if your stream works despite the warning.\n")
    
    success = test_rtsp_stream(url, duration)
    
    if not success:
        print("\nTroubleshooting steps for your specific error:")
        print("1. The H.264 timing warning is usually harmless")
        print("2. Try different RTSP transport protocols:")
        print("   - Add ?tcp to URL: rtsp://...?tcp")
        print("   - Or try UDP: rtsp://...?udp")
        print("3. Check camera H.264 profile settings")
        print("4. Try lower resolution/bitrate on camera")
        print("5. Update camera firmware")
        print("6. Test with VLC media player first")
    else:
        print("\n✅ Stream should work with app_enhanced.py despite codec warnings!")

if __name__ == "__main__":
    main()
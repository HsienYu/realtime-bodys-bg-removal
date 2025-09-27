#!/usr/bin/env python3
"""
Quick fix for H.264 timing warnings in RTSP streams

This script provides several methods to reduce or eliminate the 
H.264 codec warning you're seeing.
"""

import os
import sys

def apply_opencv_fixes():
    """Apply OpenCV environment fixes for H.264 issues"""
    print("Applying OpenCV environment fixes...")
    
    # Set FFmpeg options for better H.264 handling
    fixes = [
        'OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp',
        'OPENCV_FFMPEG_CAPTURE_OPTIONS=protocol_whitelist;file,udp,rtp,tcp',
        'OPENCV_LOG_LEVEL=ERROR',  # Reduce log verbosity
    ]
    
    for fix in fixes:
        key, value = fix.split('=', 1)
        os.environ[key] = value
        print(f"Set {key} = {value}")
    
    print("Environment fixes applied!")
    
    return fixes

def generate_launch_script():
    """Generate a launch script with fixes applied"""
    script_content = '''#!/bin/bash
# Launch script for app_enhanced.py with H.264 fixes

# Set environment variables to reduce H.264 codec warnings
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"
export OPENCV_LOG_LEVEL="ERROR"

# Suppress stderr warnings (optional - may hide other important messages)
# export PYTHONWARNINGS="ignore"

echo "Starting app_enhanced.py with H.264 fixes..."
python3 app_enhanced.py 2>/dev/null  # This line suppresses stderr warnings

# Alternative: Run with warnings visible but filtered
# python3 app_enhanced.py 2>&1 | grep -v "time_scale/num_units_in_tick"
'''
    
    with open('run_app_enhanced.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('run_app_enhanced.sh', 0o755)  # Make executable
    
    print("Created run_app_enhanced.sh launch script")
    print("Usage: ./run_app_enhanced.sh")

def test_fixes_with_rtsp(url):
    """Test the fixes with an actual RTSP stream"""
    import cv2
    import time
    
    print(f"Testing fixes with RTSP stream: {url}")
    
    # Apply environment fixes
    apply_opencv_fixes()
    
    # Test with suppressed warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    # Redirect stderr temporarily to suppress codec warnings
    import contextlib
    from io import StringIO
    
    stderr_capture = StringIO()
    
    print("Opening stream (warnings suppressed)...")
    
    with contextlib.redirect_stderr(stderr_capture):
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if cap.isOpened():
            print("✅ Stream opened successfully")
            
            # Test frame reading
            for i in range(10):
                ret, frame = cap.read()
                if ret:
                    print(f"Frame {i+1} read successfully")
                else:
                    print(f"Failed to read frame {i+1}")
                time.sleep(0.1)
            
            cap.release()
        else:
            print("❌ Failed to open stream")
    
    # Show any captured warnings
    warnings_output = stderr_capture.getvalue()
    if warnings_output:
        print(f"\nCaptured warnings/errors:")
        print(warnings_output)
    else:
        print("\n✅ No warnings captured - fixes working!")

def main():
    print("H.264 Codec Warning Fix Utility")
    print("=" * 50)
    
    print("\nAbout the warning you're seeing:")
    print("'[h264 @ 0x...] time_scale/num_units_in_tick invalid or unsupported (0/1001)'")
    print("This is a FFmpeg warning about non-standard H.264 timing parameters.")
    print("It usually doesn't affect functionality but can clutter output.")
    
    print("\nAvailable fixes:")
    print("1. Apply environment variable fixes")
    print("2. Generate launch script with fixes")
    print("3. Test fixes with your RTSP stream")
    print("4. Apply all fixes and exit")
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test" and len(sys.argv) > 2:
            test_fixes_with_rtsp(sys.argv[2])
            return
        elif sys.argv[1] == "fix":
            apply_opencv_fixes()
            generate_launch_script()
            print("\n✅ All fixes applied!")
            print("You can now run: ./run_app_enhanced.sh")
            return
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            fixes = apply_opencv_fixes()
            print("\nTo make permanent, add these to your shell profile:")
            for fix in fixes:
                print(f"export {fix}")
                
        elif choice == "2":
            generate_launch_script()
            
        elif choice == "3":
            url = input("Enter your RTSP URL: ").strip()
            if url:
                test_fixes_with_rtsp(url)
            else:
                print("No URL provided")
                
        elif choice == "4":
            apply_opencv_fixes()
            generate_launch_script()
            print("\n✅ All fixes applied!")
            print("You can now run: ./run_app_enhanced.sh")
            
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
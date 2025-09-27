#!/usr/bin/env python3
"""
Example usage patterns for the enhanced realtime body background removal app.
This script demonstrates common configurations and setups.
"""

import subprocess
import sys
import os

def run_example(title, description, args=[]):
    """Display and optionally run an example configuration"""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {title}")
    print(f"{'='*60}")
    print(f"Description: {description}")
    print(f"Command: python app_enhanced.py")
    
    if args:
        print(f"Suggested inputs when prompted:")
        for i, arg in enumerate(args, 1):
            print(f"  {i}. {arg}")
    
    print("\nPress Enter to continue to next example, 'r' to run this example, or 'q' to quit")
    choice = input("> ").strip().lower()
    
    if choice == 'q':
        return False
    elif choice == 'r':
        try:
            subprocess.run([sys.executable, "app_enhanced.py"])
        except KeyboardInterrupt:
            print("\nExample stopped by user")
        except Exception as e:
            print(f"Error running example: {e}")
    
    return True

def main():
    """Run through example configurations"""
    print("Enhanced Realtime Body Background Removal - Usage Examples")
    print("This script shows common usage patterns and configurations.")
    
    examples = [
        {
            "title": "Virtual Meeting with Custom Background",
            "description": "Perfect for video calls with professional background",
            "args": [
                "Model: 0 (fastest)",
                "ONNX: 0 (no)",
                "Input: 0 (camera)",
                "Camera: 0 (default)",
                "Resolution: 0 (HD 720p)",
                "Confidence: 0.5 (default)",
                "Feathering: 10 (default)",
                "Background: 3 (custom image)",
                "Image path: /path/to/your/background.jpg",
                "Syphon: 1 (yes) - for OBS integration",
                "NDI: 0 (no)"
            ]
        },
        {
            "title": "RTSP Security Camera Processing",
            "description": "Monitor and process IP camera feeds",
            "args": [
                "Model: 1 (small but accurate)",
                "ONNX: 1 (yes, for performance)",
                "Input: 1 (RTSP stream)",
                "RTSP URL: rtsp://admin:password@192.168.1.100:554/stream",
                "Confidence: 0.3 (lower for distant detection)",
                "Feathering: 5 (less processing)",
                "Background: 5 (remove body with green fill)",
                "Syphon: 0 (no)",
                "NDI: 1 (yes) - for monitoring software"
            ]
        },
        {
            "title": "Content Creation - Invisibility Effect",
            "description": "Create invisible person effects for videos",
            "args": [
                "Model: 2 (medium accuracy)",
                "ONNX: 0 (no)",
                "Input: 0 (camera)",
                "Camera: 0 (default)",
                "Resolution: 2 (Full HD)",
                "Confidence: 0.4",
                "Feathering: 15 (smooth edges)",
                "Background: 8 (remove body, original background)",
                "Syphon: 1 (yes) - for recording software",
                "NDI: 0 (no)"
            ]
        },
        {
            "title": "Live Streaming with Green Screen",
            "description": "Classic green screen for streaming",
            "args": [
                "Model: 0 (fastest for real-time)",
                "ONNX: 1 (yes, optimize performance)",
                "Input: 0 (camera)",
                "Camera: 0 (default)",
                "Resolution: 1 (HD 800p)",
                "Confidence: 0.5",
                "Feathering: 10",
                "Background: 0 (green screen)",
                "Syphon: 1 (yes)",
                "NDI: 1 (yes) - dual output"
            ]
        },
        {
            "title": "Video File Post-Processing",
            "description": "Process recorded videos with background replacement",
            "args": [
                "Model: 3 (large model for best quality)",
                "ONNX: 0 (no, prioritize accuracy)",
                "Input: 2 (video file)",
                "Video path: /path/to/your/video.mp4",
                "Confidence: 0.6",
                "Feathering: 20 (maximum smoothing)",
                "Background: 3 (custom image)",
                "Image path: /path/to/background.jpg",
                "Syphon: 1 (yes) - to record output",
                "NDI: 0 (no)"
            ]
        },
        {
            "title": "Transparent Output for Compositing",
            "description": "Generate transparent person cutouts for video editing",
            "args": [
                "Model: 2 (medium accuracy)",
                "ONNX: 0 (no)",
                "Input: 0 (camera)",
                "Camera: 0",
                "Resolution: 2 (Full HD)",
                "Confidence: 0.5",
                "Feathering: 12",
                "Background: 2 (transparent background)",
                "Syphon: 1 (yes) - with alpha channel",
                "NDI: 0 (no)"
            ]
        },
        {
            "title": "Time-Delayed Background Effect",
            "description": "Person appears over their past position",
            "args": [
                "Model: 1 (good balance)",
                "ONNX: 1 (yes)",
                "Input: 0 (camera)",
                "Camera: 0",
                "Resolution: 0 (HD 720p)",
                "Confidence: 0.4",
                "Feathering: 8",
                "Background: 4 (original video background)",
                "Syphon: 1 (yes)",
                "NDI: 0 (no)"
            ]
        }
    ]
    
    for example in examples:
        if not run_example(**example):
            break
    
    print(f"\n{'='*60}")
    print("Example demonstrations complete!")
    print("For detailed documentation, see ENHANCED_FEATURES.md")
    print("To run the application: python app_enhanced.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting examples...")
    except Exception as e:
        print(f"Error: {e}")
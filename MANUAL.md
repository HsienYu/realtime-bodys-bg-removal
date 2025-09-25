# Real-Time Body Background Removal - User Manual

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Background Modes](#background-modes)
5. [Configuration Options](#configuration-options)
6. [Keyboard Controls](#keyboard-controls)
7. [Performance Optimization](#performance-optimization)
8. [Syphon Integration](#syphon-integration)
9. [Troubleshooting](#troubleshooting)
10. [Technical Details](#technical-details)

## Overview

This application provides real-time person detection and background removal/replacement using YOLO segmentation models. It offers multiple modes for different use cases including virtual backgrounds, background-only streaming, and creative effects.

### Key Features
- Real-time person segmentation using YOLOv8
- 6 different background modes (Keep Body and Remove Body variants)
- Adjustable edge feathering for smooth transitions
- Syphon output for integration with streaming software
- Live mode switching via keyboard shortcuts
- Performance optimization controls
- Multi-person support with area-between-bodies detection

## Installation

### Prerequisites
- macOS (tested on macOS 10.14+)
- Python 3.8 or higher
- Webcam or external camera

### Required Packages
Install the required packages using:
```bash
pip install -r requirements.txt
```

Required packages include:
- `ultralytics` (YOLO models)
- `opencv-python` (Computer vision)
- `numpy` (Array operations)
- Additional dependencies as listed in requirements.txt

### YOLO Models
The application will automatically download YOLO models on first use. Available models:
- `yolov8n-seg.pt` (Nano - fastest, least accurate)
- `yolov8s-seg.pt` (Small - good balance)
- `yolov8m-seg.pt` (Medium - better accuracy)
- `yolov8l-seg.pt` (Large - high accuracy)
- `yolov8x-seg.pt` (Extra Large - highest accuracy, slowest)

## Quick Start

1. **Run the application:**
   ```bash
   python app_with_contours_feather.py
   ```

2. **Configure settings:**
   - Select YOLO model (0 for fastest, higher numbers for better accuracy)
   - Choose camera (usually 0 for built-in camera)
   - Select resolution (0 for HD 1280x720 recommended)
   - Set confidence threshold (0.3-0.5 recommended)
   - Choose edge feathering amount (10 recommended)
   - Select background mode (see modes below)

3. **Use keyboard controls during runtime:**
   - Press `H` to show/hide information overlay
   - Press `M` to cycle through common modes
   - Press `Q` to quit

## Background Modes

### Keep Body Modes (Shows person, replaces background)

#### Mode 0: Keep Body - Green Background
- **Use case:** Standard green screen effect
- **Output:** Person visible on solid green background
- **Best for:** Streaming, virtual backgrounds in video calls

#### Mode 1: Keep Body - Blue Background  
- **Use case:** Blue screen effect (alternative to green)
- **Output:** Person visible on solid blue background
- **Best for:** When green conflicts with clothing/environment

#### Mode 2: Keep Body - Custom Color Background
- **Use case:** Specific color requirements
- **Output:** Person visible on user-defined color background
- **Configuration:** Prompts for RGB values (0-255 each)

#### Mode 3: Transparent Body
- **Use case:** Advanced compositing workflows
- **Output:** RGBA output with person shape transparent, background opaque
- **Best for:** Professional video editing, OBS transparency effects

### Remove Body Modes (Hides person, shows background)

#### Mode 4: Remove Body - Show Original Background Only
- **Use case:** Background-only streaming, security applications
- **Output:** Original camera feed with person areas blacked out
- **Best for:** Showing environments without people, privacy applications

#### Mode 5: Remove Body - Show Background with Custom Color Fill
- **Use case:** Creative effects, artistic applications
- **Output:** Original background with person areas filled with custom color
- **Configuration:** Prompts for custom fill color RGB values

## Configuration Options

### Model Selection
- **yolov8n-seg.pt:** Fastest processing, lowest accuracy
- **yolov8s-seg.pt:** Good balance of speed and accuracy (recommended)
- **yolov8m/l/x-seg.pt:** Higher accuracy but slower processing

### ONNX Export
- Enables hardware acceleration where supported
- Generally faster inference after initial conversion
- Choose "1" if you have compatible hardware

### Camera Settings
- **Camera Selection:** Choose from detected cameras
- **Resolution Options:**
  - HD 1280x720 (16:9) - Recommended for most use cases
  - HD 1280x800 (16:10)
  - Full HD 1920x1080 (16:9) - Higher quality but slower
  - Full HD 1920x1200 (16:10)

### Detection Parameters
- **Confidence Threshold:** 0.1-0.9 range
  - Lower values (0.3): Detect more people, possible false positives
  - Higher values (0.7): More conservative detection, may miss people
  - Recommended: 0.3-0.5

### Edge Feathering
- **Range:** 0-20 pixels
- **0:** Sharp edges (fastest processing)
- **5-10:** Moderate smoothing (recommended)
- **15-20:** Very smooth edges (slower processing)

## Keyboard Controls

### Essential Controls
| Key | Function | Description |
|-----|----------|-------------|
| `Q` | Quit | Exit the application |
| `H` | Hide/Show Info | Toggle information overlay |

### Mode Switching
| Key | Function | Description |
|-----|----------|-------------|
| `M` | Cycle Modes | Green Keep → Blue Keep → Remove Body |
| `K` | Toggle Keep Body | Switch between Green ↔ Blue keep body modes |
| `R` | Remove Body | Switch to Remove Body - Original Background mode |

### Detection Controls
| Key | Function | Description |
|-----|----------|-------------|
| `B` | Area Between Bodies | Toggle including area between multiple people |

### Performance Controls
| Key | Function | Description |
|-----|----------|-------------|
| `+` or `=` | Increase Frequency | Process more frames (higher quality, slower) |
| `-` | Decrease Frequency | Skip more frames (lower quality, faster) |

## Performance Optimization

### Frame Processing Control
- **Process Every N Frames:** Adjustable via +/- keys
- **Default:** Process every frame for best quality
- **Performance mode:** Process every 2-3 frames for slower hardware

### Hardware Considerations
- **CPU:** Faster processors handle higher resolutions better
- **Memory:** 8GB+ recommended for Full HD processing
- **Camera:** Higher quality cameras may require more processing power

### Optimization Tips
1. **Start with lower resolution** (720p) and increase if performance allows
2. **Use yolov8n or yolov8s models** for real-time performance
3. **Enable ONNX** if supported by your hardware
4. **Adjust confidence threshold** - higher values process faster
5. **Reduce feathering** for better performance

## Syphon Integration

### What is Syphon?
Syphon allows real-time video sharing between applications on macOS.

### Setup for OBS Studio
1. **Enable Syphon output** when prompted during startup
2. **In OBS Studio:**
   - Add Source → Syphon Client
   - Select "PersonBackgroundRemoval" from the dropdown
   - The processed video feed will appear in OBS

### Setup for Other Applications
- Look for "Syphon Client" or "Syphon Input" options
- Select "PersonBackgroundRemoval" as the source
- Supported by: OBS Studio, vMix, Wirecast, VJ software, etc.

### Syphon Modes
- **Standard modes (0-2, 4-5):** Output RGB video
- **Transparent mode (3):** Output RGBA with transparency support

## Troubleshooting

### Common Issues

#### "Camera not found" or "Failed to open camera"
**Solutions:**
- Check camera permissions in System Preferences → Security & Privacy → Camera
- Ensure no other applications are using the camera
- Try different camera numbers (0, 1, 2, etc.)

#### Low FPS / Sluggish performance
**Solutions:**
- Use a smaller model (yolov8n instead of yolov8s/m/l/x)
- Reduce resolution to 720p
- Increase frame skip with `-` key
- Lower confidence threshold
- Disable edge feathering (set to 0)

#### Person detection not working
**Solutions:**
- Ensure good lighting conditions
- Lower confidence threshold (try 0.3 or lower)
- Check if person is fully visible in frame
- Try different YOLO model

#### Syphon not working in OBS
**Solutions:**
- Ensure OBS has Syphon plugin installed
- Restart OBS after starting the background removal app
- Check that "PersonBackgroundRemoval" appears in Syphon Client list
- Try toggling Syphon off/on in the application

#### Edge artifacts or poor masking
**Solutions:**
- Increase edge feathering amount
- Ensure good lighting with minimal shadows
- Try a larger YOLO model for better segmentation
- Enable "area between bodies" mode for multiple people

### Error Messages

#### "ValueError: invalid literal for int() with base 10: ''"
- **Cause:** Empty input when number expected
- **Solution:** Simply press Enter to use default values, or enter a valid number

#### "CUDA out of memory" (if using GPU)
- **Cause:** GPU memory exhausted
- **Solution:** Use smaller model, reduce resolution, or disable GPU acceleration

#### "Model download failed"
- **Cause:** Network issues or insufficient disk space
- **Solution:** Check internet connection and available disk space (models are ~50MB each)

### Performance Benchmarks

#### Typical FPS on MacBook Pro M1:
- **720p, yolov8n:** 25-30 FPS
- **720p, yolov8s:** 15-20 FPS
- **1080p, yolov8n:** 15-20 FPS
- **1080p, yolov8s:** 8-12 FPS

#### Typical FPS on Intel MacBook Pro (2019):
- **720p, yolov8n:** 10-15 FPS
- **720p, yolov8s:** 5-8 FPS
- **1080p, yolov8n:** 5-8 FPS

## Technical Details

### Architecture
- **Main Thread:** Camera capture and display
- **Processing Thread:** YOLO inference and mask generation
- **Thread Synchronization:** Lock-based frame sharing

### Image Processing Pipeline
1. **Frame Capture:** OpenCV camera input
2. **YOLO Inference:** Person segmentation
3. **Mask Processing:** Morphological operations, feathering
4. **Background Replacement:** Mask-based blending
5. **Output:** Display and Syphon publishing

### File Structure
```
realtime-bodys-bg-removal/
├── app_with_contours_feather.py  # Main application
├── syphon_utils.py              # Syphon integration
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
├── MANUAL.md                    # This manual
├── yolov8*-seg.pt              # YOLO model files
└── __pycache__/                # Python cache files
```

### Dependencies Explained
- **ultralytics:** YOLOv8 model implementation and utilities
- **opencv-python:** Camera capture, image processing, display
- **numpy:** Efficient array operations for image data
- **threading:** Multi-threaded processing for performance
- **subprocess:** System camera detection
- **collections.deque:** Efficient ring buffers for performance metrics

---

## Support and Development

For issues, feature requests, or contributions, please refer to the project repository or contact the development team.

**Version:** 2.0.0  
**Last Updated:** September 2024  
**Compatibility:** macOS 10.14+, Python 3.8+
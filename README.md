# Real-time Person Background Removal

A powerful real-time video processing application that uses YOLOv8 to detect people and remove or replace their backgrounds. Designed for streamers, content creators, and video conferencing users who need professional-looking video without a physical green screen.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Configuration Options](#configuration-options)
  - [Keyboard Controls](#keyboard-controls)
- [Output Methods](#output-methods)
  - [Syphon Integration](#syphon-integration)
  - [Alternative Output Methods](#alternative-output-methods)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Accurate Person Detection & Segmentation**: Leverages YOLOv8's advanced computer vision capabilities to precisely identify and isolate people in the frame
- **Real-Time Processing**: Optimized algorithms ensure low-latency processing suitable for live applications
- **Flexible Background Options**:
  - Complete background removal (transparent)
  - Static image replacement
  - Video background replacement
  - Blur effect option
- **Multiple Output Methods**: 
  - Standard window display for direct viewing
  - Syphon server output for integration with streaming software (macOS)
  - Virtual camera integration for video conferencing
- **Customizable Configuration**: Adjust resolution, processing parameters, and model selection to balance quality and performance

## Requirements

- Python 3.10 or newer
- Operating System: macOS, Windows, or Linux
- Webcam or other video input source
- For GPU acceleration: CUDA-compatible NVIDIA GPU (recommended)
- For Syphon output: macOS only

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/realtime-bodys-bg-removal.git
   cd realtime-bodys-bg-removal
   ```

2. **Create a conda environment:**
   ```bash
   conda create -n background-removal python=3.10
   conda activate background-removal
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **For Syphon support (macOS only):**
   ```bash
   pip install syphon-python
   ```

## Quick Start

```bash
# Basic usage with default settings
python main.py

# With custom settings
python main.py --model yolov8n-seg --resolution 1280x720 --background path/to/image.jpg
```

## Usage Guide

### Configuration Options

#### Model Selection
Choose from different YOLOv8 segmentation models based on your performance needs:

| Model | Speed | Accuracy | Memory Usage | Recommended Hardware |
|-------|-------|----------|--------------|---------------------|
| yolov8n-seg | Fastest | Lower | ~2GB | Integrated GPU |
| yolov8s-seg | Medium | Medium | ~2.5GB | Mid-range GPU |
| yolov8m-seg | Slower | Higher | ~5GB | Good GPU |
| yolov8l-seg | Slowest | Highest | ~8GB | High-end GPU |

Example: `--model yolov8s-seg`

#### Resolution Options
Set your preferred resolution to balance quality and performance:

- 640x480 (Basic, fastest)
- 1280x720 (16:9 HD)
- 1280x800 (16:10)
- 1920x1080 (16:9 Full HD)
- 1920x1200 (16:10)

Example: `--resolution 1280x720`

#### Background Options
Specify what replaces the removed background:

- Transparent: `--background none`
- Solid color: `--background "#00FF00"` (green)
- Image file: `--background path/to/image.jpg`
- Video file: `--background path/to/video.mp4`
- Blur: `--background blur --blur-strength 15`

### Keyboard Controls

During application runtime:
- `Q` or `ESC`: Quit application
- `S`: Save current frame as image
- `B`: Cycle between configured background options
- `M`: Toggle model visualization mode (shows detection boundaries)
- `+` / `-`: Increase/decrease detection confidence threshold
- `F`: Toggle fullscreen mode

## Output Methods

### Syphon Integration

Syphon allows you to send your processed video to other applications on macOS:

1. Start the application with: `python main.py --output syphon`
2. In OBS Studio:
   - Add a "Syphon Client" source
   - Select the "BackgroundRemoval" Syphon server
   - Configure as needed in OBS

### Alternative Output Methods

#### Virtual Camera
```bash
# Start with virtual camera output
python main.py --output virtual-camera
```
Then select "Background Removal Camera" in your video conferencing app

#### Screen Capture
If direct integration isn't working:
1. Run the application in window mode
2. Use OBS or similar software to capture the application window
3. Add post-processing as needed

## Performance Optimization

- **For better speed**: Use `--model yolov8n-seg --resolution 640x480`
- **For better quality**: Use `--model yolov8m-seg --resolution 1280x720`
- **For GPU acceleration**: Use `--device cuda` (requires NVIDIA GPU with CUDA)
- **For CPU-only**: Use `--device cpu` (slower but works on all systems)

## Troubleshooting

### Common Issues

1. **Metal Device Errors (macOS)**:
   - Try: `python main.py --device cpu` to bypass GPU
   - Ensure macOS is updated to latest version

2. **Empty/Black Frames in OBS**:
   - Verify the correct Syphon server is selected
   - Try restarting both the application and OBS
   - Check terminal for error messages

3. **Performance Issues**:
   - Select a lighter model: `--model yolov8n-seg`
   - Reduce resolution: `--resolution 640x480`
   - Close other GPU-intensive applications
   - Try `--optimization speed` flag to prioritize performance

4. **Camera Access Problems**:
   - Check system permissions for camera access
   - Try specifying a different camera: `--camera 1` (try 0, 1, 2, etc.)
   - Make sure no other application is using the camera

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open Issues to report bugs and request features.

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 by Ultralytics for the excellent object detection models
- Syphon Project for the macOS inter-application video sharing framework
- Contributors and users who have provided valuable feedback and improvements

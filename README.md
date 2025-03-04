# Real-time Person Background Removal

A real-time video processing application that detects people using YOLOv8 and removes or replaces their background. Perfect for live streaming, video conferencing, and content creation.

## Features

- **Advanced Person Detection**: Uses YOLOv8 for accurate person detection and segmentation
- **Real-Time Processing**: Optimized for low-latency video processing
- **Background Options**: Remove or replace backgrounds with images/videos
- **Multiple Output Methods**: 
  - Standard window display
  - Syphon output for streaming software (macOS)
  - Virtual camera integration
- **Flexible Configuration**: Adjustable resolution, model selection, and processing parameters

## Requirements

- Python 3.10+
- macOS, Windows, or Linux
- Webcam or video input source
- For Syphon: macOS only

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/real-time-person-background-removal.git
   cd real-time-person-background-removal
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
# Basic usage
python main.py

# With custom settings
python main.py --model yolov8n-seg --resolution 1280x720 --background path/to/image.jpg
```

## Usage Guide

### Model Selection

The application supports different YOLOv8 segmentation models:

- `yolov8n-seg`: Fastest, lower accuracy
- `yolov8s-seg`: Balanced performance
- `yolov8m-seg`: Higher accuracy, slower
- `yolov8l-seg`: Best accuracy, requires good GPU

### Resolution Options

Supported resolutions include:
- 1280x720 (16:9 HD)
- 1280x800 (16:10)
- 1920x1080 (16:9 Full HD)
- 1920x1200 (16:10)

### Keyboard Controls

- `Q`: Quit application
- `S`: Save current frame
- `B`: Toggle between background options (if configured)
- `M`: Toggle model visualization

## Syphon Integration

### Setup for OBS

1. Start the application with Syphon enabled
2. In OBS, add a "Syphon Client" source
3. Select the "BackgroundRemoval" Syphon server
4. Adjust settings as needed

### Alternative Output Methods

If you encounter issues with Syphon:

1. **PyGame Syphon Bridge**:
   ```bash
   pip install pygame pygame_syphon
   python syphon_alternative.py
   ```

2. **OBS Virtual Camera**:
   - Capture the application window in OBS
   - Use OBS Virtual Camera feature

3. **Screen Recording**:
   - Use QuickTime or other screen recording software

## Troubleshooting

### Common Issues

1. **Metal Device Errors**:
   - Try the PyGame alternative implementation
   - Update your GPU drivers

2. **Empty Frames in OBS**:
   - Verify the correct Syphon server is selected
   - Check the application is running and publishing frames

3. **Performance Issues**:
   - Select a lighter model (yolov8n-seg)
   - Reduce resolution
   - Close other GPU-intensive applications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Realtime Body Background Removal

A Python application that uses YOLOv8 segmentation models to remove backgrounds from video in real-time, providing various output options including green screen, custom backgrounds, and transparency for streaming and virtual camera applications.


## Features

### Standard Version
- **Real-time processing**: Optimized for smooth real-time video performance
- **Multiple YOLOv8 models**: Choose from different model sizes based on your hardware capabilities
- **ONNX support**: Option to convert and use ONNX models for potentially better performance
- **Edge feathering**: Smooth edges with adjustable feathering for professional-looking results
- **Multiple output modes**:
  - Green screen
  - Blue screen
  - Transparent background (alpha channel)
- **Multi-person handling**: Intelligently handles multiple people with option to include/exclude space between bodies
- **Threaded processing**: Uses a separate processing thread to maintain UI responsiveness
- **Syphon integration**: Output to other applications on macOS like OBS Studio using Syphon
- **NDI integration**: Network Device Interface support for professional streaming with async frame publishing
- **Performance tuning**: Adjustable processing frequency for balancing quality and performance

### Enhanced Version (app_enhanced.py)
- **All standard features plus:**
- **RTSP/RTMP input**: Connect to IP cameras and streaming sources
- **Video file processing**: Process pre-recorded video files with automatic looping
- **Custom background images**: Use any image file as background
- **Original video backgrounds**: Time-delayed background effects
- **12 background modes**: Extended options including invisibility effects and normal modes
- **Enhanced streaming**: Better integration with professional streaming workflows
- **FP16/MPS optimization**: Automatic Apple Silicon optimization with 2-3x performance boost
- **Async NDI publishing**: Stable NDI frame transmission with separate publishing thread
- **Dynamic mode switching**: Cycle through all 12 modes in real-time (press 'M')
- **Background frame offset control**: Adjust time-delay effect with '[' and ']' keys
- **Performance configuration**: YAML-based settings for hardware-specific tuning

### M3 Max Performance Optimized Version (app_enhanced_m3_max_stable.py) 🚀
- **All enhanced features plus:**
- **3-4x Performance Improvement**: From 8 FPS to 25-45+ FPS on MacBook Pro M3 Max
- **Reliable PyTorch Optimization**: Proven performance improvements over ONNX
- **Adaptive Quality Control**: Dynamic frame skipping and quality adjustment
- **Optimized Threading**: Enhanced multi-threading for M3 Max architecture
- **Smart Inference Scaling**: Processes at optimal resolution (60% scale) for M3 Max performance
- **All 12 Background Modes**: Keep all enhanced features with maximum performance
- **Conservative FPS Target**: Stable 25 FPS target with bursts up to 40+ FPS

### Split-Screen Version (app_enhanced_split.py) 🎬 NEW!
- **All enhanced features plus:**
- **Split-Screen Effects**: Apply effects to left side, right side, or full frame
- **Real-time Switching**: Toggle between split modes while running (I/O/P keys)
- **Visual Split Line**: Yellow center line shows split boundary
- **Creative Effects**: Perfect for before/after comparisons and demonstrations
- **All 12 Background Modes**: Works with all background effects
- **Full Integration**: Includes RTSP, video files, Syphon, and NDI support

## Requirements

- Python 3.7+
- macOS (for Syphon support)
- Webcam, IP camera (RTSP/RTMP), or video file input
- Packages listed in requirements.txt:
  - numpy, opencv-python, ultralytics (YOLOv8)
  - syphon-python (macOS Syphon output)
  - ndi-python (NDI output, requires NDI Tools)
  - torch, torchvision (YOLO inference)
  - onnx, onnxruntime (optional, for ONNX acceleration)
  - pyyaml (configuration files)

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/realtime-bodys-bg-removal.git
   cd realtime-bodys-bg-removal
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. For Syphon support, see [SYPHON_SETUP.md](SYPHON_SETUP.md) for detailed instructions.

## Usage

### Standard Version
Run the standard application with:

```bash
python app_with_contours_feather.py
```

### Enhanced Version
Run the enhanced version with RTSP support and custom backgrounds:

```bash
python app_enhanced.py
```

### M3 Max Performance Optimized Version (RECOMMENDED for MacBook Pro M3 Max) 🚀
Run the performance-optimized version with all features plus 3-4x speed improvement:

```bash
python app_enhanced_m3_max_stable.py
```

**Expected Performance on M3 Max:**
- YOLOv8n-seg: 30-40 FPS (vs 8-12 FPS original) ⭐ Recommended
- YOLOv8s-seg: 20-30 FPS (vs 6-10 FPS original)
- YOLOv8m-seg: 12-20 FPS (vs 4-8 FPS original)
- All 12 background modes supported
- RTSP/RTMP + Video files + Syphon + NDI
- Stable async NDI publishing at 25 FPS target

### Split-Screen Version (For Creative Effects & Demonstrations) 🎬
Run the split-screen version for before/after comparisons:

```bash
python app_enhanced_split.py
```

**Split-Screen Features:**
- Press **I**: Effect on LEFT side only (center to left)
- Press **O**: Effect on RIGHT side only (center to right)  
- Press **P**: FULL effect (default)
- Yellow center line shows split boundary
- Perfect for:
  - Live demonstrations
  - Before/after comparisons
  - Social media content creation
  - Testing different settings side-by-side

See [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) for detailed documentation of the features including:
- RTSP/RTMP video input support
- Custom background images
- Original video background modes
- Video file processing
- 12 total background modes

Follow the interactive prompts to configure the application:

1. **Model selection**: 
   - Choose a YOLOv8 segmentation model (0-4, smaller number = faster but less accurate)

2. **ONNX option**:
   - Enter 1 to use ONNX, 0 to use standard PyTorch model

3. **Camera selection**:
   - Select from available cameras

4. **Resolution**:
   - Choose from preset resolutions (HD, Full HD)

5. **Detection confidence**:
   - Set the confidence threshold (0.1-0.9, recommended 0.3-0.5)
   - Lower values detect bodies at greater distances but may increase false positives

6. **Edge feathering**:
   - Set the amount of edge feathering (0-20)
   - Higher values create smoother transitions at edges
   - 0 for sharp edges, 10 recommended for natural look

7. **Background mode**:
   - 0: Green screen
   - 1: Blue screen
   - 2: Custom color (you'll be prompted for RGB values)
   - 3: Transparent background (outputs RGBA with transparent bodies)

8. **Syphon output**:
   - Enable Syphon to output to other applications (macOS only)

9. **NDI output**:
   - Enable NDI for professional streaming over network
   - Requires NDI Tools installation (https://ndi.video/tools/)
   - Test NDI setup with: `python test_ndi.py`

## Keyboard Controls

While the application is running:

- **Q**: Quit the application
- **H**: Toggle display of information overlay
- **B**: Toggle inclusion of area between multiple bodies
- **M**: Cycle through all 12 background modes (in enhanced versions)
- **1-8**: Direct mode selection hotkeys (in enhanced versions)
- **+/-**: Increase/decrease processing frequency (lower frequency = higher performance)
- **[ / ]**: Adjust background frame offset in original background modes (time-delay effect)

**Split-Screen Controls (app_enhanced_split.py only):**
- **I**: Effect on LEFT side only (center to left)
- **O**: Effect on RIGHT side only (center to right)
- **P**: FULL effect across entire frame

## Performance Tips

- Use smaller models (yolov8n-seg, yolov8s-seg) for better performance
- Enable ONNX with CoreML on Apple Silicon for 2-3x speed boost (automatic in app_enhanced.py)
- Use the processing frequency control (+/-) to balance quality and performance
- Run at lower resolutions for better framerates
- On Apple Silicon, app_enhanced.py automatically uses FP16 and MPS optimization
- Use `app_enhanced_m3_max_stable.py` for maximum performance on M3 Max
- Configure hardware-specific settings in `performance_config.yaml` or `performance_config_m3_max.yaml`

## Syphon Integration

When Syphon is enabled, the output can be used in applications like OBS Studio:

1. Install the obs-syphon plugin in OBS
2. Add a Syphon Client source
3. Select "PersonBackgroundRemoval" as the source
4. For transparent mode, ensure "Allow Transparency" is checked

See [SYPHON_SETUP.md](SYPHON_SETUP.md) for detailed instructions.

## NDI Integration

NDI (Network Device Interface) allows professional streaming over network:

1. Install NDI Tools from https://ndi.video/tools/
2. Install Python package: `pip install ndi-python`
3. Test NDI setup: `python test_ndi.py`
4. Enable NDI when prompted in the application
5. Look for "PersonBackgroundRemoval" in NDI-compatible apps (OBS, vMix, etc.)

**Features:**
- Async frame publishing for stable transmission
- Automatic BGRA format conversion
- Queue management to prevent frame backup
- Performance statistics logging
- Supports both RGB and RGBA (transparent) modes

## Performance Configuration

The application includes YAML configuration files for hardware-specific tuning:

### General Configuration (`performance_config.yaml`)
- Balanced settings for most hardware
- Adaptive quality control
- 30 FPS target

### M3 Max Configuration (`performance_config_m3_max.yaml`)
- Optimized for Apple M3 Max chips
- 60 FPS target
- Larger buffers for stability
- CoreML execution provider

**Settings Tool:**
```bash
python optimize_settings.py
```
Generates recommended configuration based on your use case.

## Troubleshooting

- **Low FPS**: Try a smaller model, lower resolution, or increase the processing interval
- **Poor segmentation**: Try a larger model (yolov8m-seg, yolov8l-seg) or adjust confidence (try 0.3)
- **Edge artifacts**: Increase the feathering amount for smoother edges
- **Syphon not working**: Ensure you have the correct plugins installed in the receiving application
- **NDI not working**: Run `python test_ndi.py` to verify installation
- **Incomplete background replacement**: Lower confidence threshold to 0.3 and try yolov8m-seg model
- **H.264 codec warnings (RTSP)**: These are usually harmless; try adding `?tcp` to RTSP URL if issues persist
- **On Apple Silicon**: app_enhanced.py automatically uses MPS and FP16 for 2-3x performance boost

## License

0x5a@pimiya.rocks

## Documentation

- [ENHANCED_FEATURES.md](ENHANCED_FEATURES.md) - Detailed documentation of enhanced features
- [FP16_MPS_OPTIMIZATION.md](FP16_MPS_OPTIMIZATION.md) - Apple Silicon optimization guide
- [SYPHON_SETUP.md](SYPHON_SETUP.md) - Syphon configuration guide
- `performance_config.yaml` - General performance settings
- `performance_config_m3_max.yaml` - M3 Max specific settings

## Testing Tools

- `test_ndi.py` - Test NDI installation and functionality
- `optimize_settings.py` - Generate optimal configuration recommendations
- `rtsp_test.py` - Test RTSP stream connections

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Syphon Project](https://syphon.github.io/)
- [NDI by NewTek](https://ndi.video/)

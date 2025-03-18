# Realtime Body Background Removal

A Python application that uses YOLOv8 segmentation models to remove backgrounds from video in real-time, providing various output options including green screen, custom backgrounds, and transparency for streaming and virtual camera applications.


## Features

- **Real-time processing**: Optimized for smooth real-time video performance
- **Multiple YOLOv8 models**: Choose from different model sizes based on your hardware capabilities
- **ONNX support**: Option to convert and use ONNX models for potentially better performance
- **Edge feathering**: Smooth edges with adjustable feathering for professional-looking results
- **Multiple output modes**:
  - Green screen
  - Blue screen
  - Custom color
  - Transparent background (alpha channel)
- **Multi-person handling**: Intelligently handles multiple people with option to include/exclude space between bodies
- **Threaded processing**: Uses a separate processing thread to maintain UI responsiveness
- **Syphon integration**: Output to other applications on macOS like OBS Studio using Syphon
- **Performance tuning**: Adjustable processing frequency for balancing quality and performance

## Requirements

- Python 3.7+
- macOS (for Syphon support)
- Webcam or other camera input
- Packages listed in requirements.txt

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

Run the application with:

```bash
python app_with_contours_feather.py
```

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

## Keyboard Controls

While the application is running:

- **Q**: Quit the application
- **H**: Toggle display of information overlay
- **B**: Toggle inclusion of area between multiple bodies
- **+/-**: Increase/decrease processing frequency (lower frequency = higher performance)

## Performance Tips

- Use smaller models (yolov8n-seg, yolov8s-seg) for better performance
- Enable ONNX for potential performance gains
- Use the processing frequency control (+/-) to balance quality and performance
- Run at lower resolutions for better framerates

## Syphon Integration

When Syphon is enabled, the output can be used in applications like OBS Studio:

1. Install the obs-syphon plugin in OBS
2. Add a Syphon Client source
3. Select "PersonBackgroundRemoval" as the source
4. For transparent mode, ensure "Allow Transparency" is checked

See [SYPHON_SETUP.md](SYPHON_SETUP.md) for detailed instructions.

## Troubleshooting

- **Low FPS**: Try a smaller model, lower resolution, or increase the processing interval
- **Poor segmentation**: Try a larger model (yolov8m-seg, yolov8l-seg) or adjust confidence
- **Edge artifacts**: Increase the feathering amount for smoother edges
- **Syphon not working**: Ensure you have the correct plugins installed in the receiving application

## License

0x5a@pimiya.rocks

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Syphon Project](https://syphon.github.io/)

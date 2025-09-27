# Enhanced Features Documentation

This document describes the new features added to the realtime body background removal application in `app_enhanced.py`.

## New Features Overview

### 1. RTSP/RTMP Video Input Support
- Support for IP cameras and streaming protocols
- Real-time validation of stream connections
- Automatic fallback to default camera if stream fails

### 2. Enhanced Background Modes
- Custom background images
- Original video background mode
- All original background modes maintained

### 3. Video File Input Support
- Process pre-recorded video files
- Automatic video looping
- Maintains all processing capabilities

## Video Input Options

When running `python app_enhanced.py`, you'll be prompted to choose from three input types:

### Option 0: Physical Camera (Default)
- Uses built-in or USB cameras
- Includes resolution selection
- Horizontal flipping for natural mirror view

### Option 1: RTSP/RTMP Stream
- Supports various streaming protocols
- Includes connection validation
- Examples of supported URLs:

```
rtsp://username:password@192.168.1.100:554/stream
rtsp://192.168.1.100:8554/live
rtsp://admin:password@camera.local/h264
rtmp://live-server.com/live/stream_key
http://192.168.1.100:8080/video.mjpg
https://example.com/live/stream.m3u8
```

#### RTSP Setup Examples

**Generic IP Camera:**
```
rtsp://admin:password@192.168.1.100:554/stream1
```

**Hikvision Camera:**
```
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101
```

**Dahua Camera:**
```
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

**Axis Camera:**
```
rtsp://root:password@192.168.1.100/axis-media/media.amp
```

### Option 2: Video File
- Process any video file supported by OpenCV
- Supports common formats: MP4, AVI, MOV, MKV, etc.
- Automatically loops when video ends

## Enhanced Background Modes

The application now supports 12 different background modes (0-11):

### Keep Body Modes (0-5)
Shows the detected person while replacing the background:

- **Mode 0**: Keep Body - Green Background
- **Mode 1**: Keep Body - Blue Background  
- **Mode 2**: Keep Body - Transparent Background (RGBA output)
- **Mode 3**: Keep Body - Custom Background Image
- **Mode 4**: Keep Body - Original Video Background
- **Mode 5**: Keep Body - Normal (No Effect)

### Remove Body Modes (6-11)
Hides the detected person while showing the background:

- **Mode 6**: Remove Body - Green Fill in Body Area
- **Mode 7**: Remove Body - Blue Fill in Body Area
- **Mode 8**: Remove Body - Transparent Body Area (RGBA output)
- **Mode 9**: Remove Body - Custom Background Image
- **Mode 10**: Remove Body - Original Background
- **Mode 11**: Remove Body - Normal (No Effect)

## Custom Background Image Setup

When selecting Mode 3 (Keep Body - Custom Background Image):

1. Prepare your background image in common formats (JPG, PNG, BMP)
2. When prompted, enter the full path to your image:
   ```
   /Users/username/Pictures/background.jpg
   ```
3. The image will be automatically resized to match your video resolution
4. If the image can't be loaded, it will fallback to green background

### Background Image Tips:
- Use high-resolution images for best quality
- Images will be stretched to fit video dimensions
- Consider the aspect ratio of your video feed
- PNG files with transparency are supported but will be flattened

## Original Video Background Mode

Modes 4 and 8 use the original video feed as background:

- **Mode 4**: Shows person over a slightly delayed version of the video
- **Mode 8**: Shows the background while hiding the person (like invisibility effect)

This creates interesting effects like:
- Time-delayed background (person appears over their past position)
- Invisibility cloak effect (person disappears but background remains)

## Normal Mode (No Effect)

Modes 5 and 11 provide "Normal" operation:

- **Mode 5**: Keep Body - Normal shows the original video feed without any processing
- **Mode 11**: Remove Body - Normal also shows the original video feed without any processing

These modes are useful for:
- Comparing processed vs unprocessed video
- Quick switching to normal view during streaming
- Testing and calibration purposes
- When you want to temporarily disable all effects

## Keyboard Controls

Enhanced keyboard controls for real-time adjustments:

- **Q**: Quit application
- **H**: Hide/show information overlay
- **B**: Toggle area between bodies mode (for multiple people)
- **M**: Cycle through all 12 background modes (0→1→2→3→4→5→6→7→8→9→10→11)
- **+/-**: Adjust processing frequency (performance tuning)

## Performance Considerations

### RTSP Streams
- Network latency affects real-time performance
- Higher resolution streams require more processing power
- Consider using lower resolution streams for better performance
- Wired connections generally perform better than WiFi

### Custom Backgrounds
- Large background images may impact performance
- Pre-resize images to your video resolution for optimal performance
- Use compressed image formats (JPEG) for faster loading

### Video Files
- Processing speed depends on video resolution and complexity
- Use lower resolution videos for real-time effects
- H.264 encoded videos generally perform better

## Troubleshooting

### RTSP Connection Issues
```
Error: Cannot connect to RTSP stream
```
**Solutions:**
- Verify the RTSP URL format
- Check network connectivity to the camera
- Ensure camera supports the specified resolution
- Verify username/password credentials
- Check firewall settings

### H.264 Codec Warnings
```
[h264 @ 0x...] time_scale/num_units_in_tick invalid or unsupported (0/1001)
```
**About this warning:**
This is a common FFmpeg warning that indicates non-standard timing parameters in the H.264 stream. It's usually harmless and doesn't affect functionality.

**Solutions:**
- The warning can be safely ignored if video works properly
- Try different RTSP transport protocols:
  - Add `?tcp` to your URL: `rtsp://camera_ip/stream?tcp`
  - Or try UDP: `rtsp://camera_ip/stream?udp`
- Adjust camera H.264 profile settings (use baseline profile)
- Lower the bitrate/resolution on your camera
- Update camera firmware if available

**Test your stream:**
```bash
python rtsp_test.py rtsp://your_camera_url
```

### Custom Background Issues
```
Error: Cannot load background image
```
**Solutions:**
- Verify the image file path is correct
- Check image file format is supported
- Ensure sufficient disk space and memory
- Try a different image file

### Performance Issues
**For better performance:**
- Use smaller YOLO models (yolov8n-seg.pt)
- Enable ONNX optimization
- Reduce processing frequency with +/- keys
- Lower video resolution
- Close other applications

## Example Use Cases

### Virtual Meeting Setup
1. Use Mode 3 with a professional background image
2. Connect via RTSP from a high-quality IP camera
3. Output to OBS via Syphon/NDI for video conferencing

### Content Creation
1. Use Mode 8 for invisibility effects
2. Record with Mode 4 for time-delay effects  
3. Combine with video files for post-processing

### Live Streaming
1. Use RTSP input from multiple camera angles
2. Switch between background modes in real-time
3. Output to streaming software via NDI

### Security/Monitoring
1. Process RTSP feeds from security cameras
2. Use Mode 5-8 to highlight or hide people
3. Monitor multiple camera feeds

## Integration with Streaming Software

### OBS Studio Setup
1. Install obs-syphon plugin (macOS) or obs-ndi plugin
2. Add "Syphon Client" or "NDI Source"
3. Select "PersonBackgroundRemoval" as the source
4. For transparent modes, enable "Allow Transparency"

### vMix Setup
1. Ensure NDI Tools are installed
2. Add "NDI/IP Input"
3. Select "PersonBackgroundRemoval" from the list
4. Configure as needed for your production

## Command Line Examples

```bash
# Run with enhanced features
python app_enhanced.py

# Example workflow:
# 1. Choose model (0 for fastest)
# 2. Choose RTSP input (1)
# 3. Enter: rtsp://admin:password@192.168.1.100:554/stream
# 4. Choose custom background (3)
# 5. Enter: /Users/username/Pictures/office_background.jpg
# 6. Enable Syphon output for OBS
```

## Technical Notes

### RTSP Protocol Support
- Supports RTSP over TCP and UDP
- Compatible with most IP camera brands
- Handles authentication (username/password)
- Automatic stream reconnection on failures

### Background Processing
- Background images cached in memory for performance
- Original video backgrounds use frame buffering (30 frames)
- Automatic fallback to solid colors if images fail

### Threading Architecture
- Separate processing thread maintains UI responsiveness
- Frame buffering prevents dropped frames
- Lock-based synchronization ensures thread safety

This enhanced version maintains full backward compatibility while adding powerful new capabilities for professional video processing workflows.
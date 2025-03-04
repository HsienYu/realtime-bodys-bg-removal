# Syphon Integration Setup Guide

This application supports Syphon, which allows you to share the processed video frames with other macOS applications like OBS Studio, video mixers, or VJ software.

## Installation

1. Install the required Python package:

```bash
pip install syphon-python
```

Note: The `syphon-python` package requires Python 3.7+ and macOS 10.14+.

## Usage

1. When prompted, enter `1` to enable Syphon output.
2. The application will create a Syphon server named "PersonBackgroundRemoval".
3. In your receiving application (e.g., OBS), add a Syphon source and select "PersonBackgroundRemoval".

## For OBS Users

1. Install the "obs-syphon" plugin for OBS:
   - Download from: https://github.com/zakk4223/obs-syphon/releases
   - Extract and copy to your OBS plugins folder

2. Restart OBS after installing the plugin

3. Add a Syphon source:
   - Click the "+" button in the Sources panel
   - Select "Syphon Client"
   - Create a new source and name it (e.g., "Green Screen Video")
   - In the properties, select "PersonBackgroundRemoval" from the Source dropdown
   - Make sure "Allow Transparency" is checked
   - Click OK

4. If you don't see the source or it appears black:
   - Make sure your app is running and Syphon is enabled
   - Try restarting OBS
   - Check that the plugin is properly installed

## Troubleshooting

If you cannot get video in your Syphon client application:

1. **Verify Syphon server is running**:
   - The application will print "Syphon server created successfully" when working correctly
   - Ensure you see no errors related to Syphon in the console output

2. **Check with Syphon Viewer**:
   - Download Simple Syphon Recorder: https://github.com/Syphon/Simple/releases
   - Open it and see if you can view your "PersonBackgroundRemoval" server
   - If it shows up here but not in your target application, the issue is with that application

3. **Format compatibility**:
   - Some applications may require specific pixel formats
   - The app converts to RGB format which should be compatible with most Syphon clients

4. **Permission issues**:
   - Make sure your application has proper permissions to access the camera and GPU

5. **Restart applications**:
   - Sometimes closing and reopening both the sender and receiver applications resolves issues

If problems persist, try running a basic Syphon example to verify your system can support Syphon properly.


"""
Enhanced version of app_with_contours_feather.py that adds:
1. RTSP video input support
2. Normal background functionality (custom background images and original feed)
3. All existing features maintained
"""
import os
import random
import cv2
import time
import numpy as np
from ultralytics import YOLO
import subprocess
from syphon_utils import create_syphon_server, publish_frame_to_syphon, cleanup_syphon
from ndi_utils import create_ndi_sender, publish_frame_to_ndi, cleanup_ndi
import threading
from collections import deque
import urllib.parse
import warnings

# Suppress common OpenCV/FFmpeg warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

# YOLO model selection
list_model = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
]

print("Choose a model: ")
for i, model in enumerate(list_model):
    print(f"{i}: {model}")
model_input = input("Enter the model number [default: 0]: ").strip()
model_choice = int(model_input) if model_input else 0

onnx_input = input("Do you want to use ONNX? (0/1) [default: 0]: ").strip()
is_onnx = bool(int(onnx_input)) if onnx_input else False

model = YOLO(list_model[model_choice])

yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
colors = [random.choices(range(256), k=3) for _ in classes_ids]

if is_onnx and not os.path.exists(list_model[model_choice][:-3] + ".onnx"):
    model = model.export(format="onnx")
    model = YOLO(list_model[model_choice][:-3] + ".onnx")


def list_cameras():
    """List available cameras on macOS"""
    try:
        result = subprocess.run(
            ['system_profiler', 'SPCameraDataType'], stdout=subprocess.PIPE)
        output = result.stdout.decode('utf-8')
        devices = [line.strip()
                   for line in output.split('\n') if 'Model ID' in line]
        return devices
    except Exception as e:
        print(f"Error listing cameras: {e}")
        return []


def validate_rtsp_url(url):
    """Validate and test RTSP URL connection with enhanced codec handling"""
    try:
        # Parse URL to check if it's properly formatted
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ['rtsp', 'rtmp', 'http', 'https']:
            return False, "Invalid protocol. Supported: rtsp://, rtmp://, http://, https://"

        # Try to open the stream with specific options for better H.264 handling
        test_cap = cv2.VideoCapture(url)

        # Set buffer size to reduce latency and improve stability
        test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set specific codec properties for better RTSP handling
        test_cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('H', '2', '6', '4'))

        if not test_cap.isOpened():
            test_cap.release()
            return False, "Cannot connect to RTSP stream"

        # Try to read multiple frames to ensure stability
        success_count = 0
        for i in range(5):
            ret, frame = test_cap.read()
            if ret and frame is not None:
                success_count += 1

        test_cap.release()

        if success_count == 0:
            return False, "Cannot read frames from RTSP stream"
        elif success_count < 3:
            return True, f"RTSP stream connected but may be unstable ({success_count}/5 frames read successfully)"
        else:
            return True, "RTSP stream is valid and stable"

    except Exception as e:
        return False, f"Error validating RTSP URL: {e}"


def configure_rtsp_capture(cap, is_rtsp=False):
    """Configure video capture with optimized settings for RTSP streams"""
    if is_rtsp:
        # Reduce buffer size to minimize latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set transport protocol (try TCP first for stability)
        # Note: This may not work with all OpenCV builds
        try:
            cap.set(cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc('H', '2', '6', '4'))
        except:
            pass  # Ignore if not supported

        # Set additional properties for better RTSP handling
        cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS

        # Suppress codec warnings by redirecting stderr temporarily
        import os
        import sys
        from contextlib import redirect_stderr

        # This helps reduce codec warning spam
        try:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        except:
            pass

    return cap


def load_background_image(image_path, target_width, target_height):
    """Load and resize background image"""
    try:
        if not os.path.exists(image_path):
            return None, "Background image file not found"

        bg_image = cv2.imread(image_path)
        if bg_image is None:
            return None, "Cannot load background image. Check file format."

        # Resize to match video dimensions
        bg_image = cv2.resize(bg_image, (target_width, target_height))
        return bg_image, "Background image loaded successfully"

    except Exception as e:
        return None, f"Error loading background image: {e}"


# Enhanced video input selection
print("\nChoose video input:")
print("0: Use physical camera")
print("1: Use RTSP/RTMP stream")
print("2: Use video file")

input_choice = input("Enter input type [default: 0]: ").strip()
input_type = int(input_choice) if input_choice else 0

vid = None
video_source = None

if input_type == 0:  # Physical camera
    camera_devices = list_cameras()
    print("Choose a camera: ")
    for i, device in enumerate(camera_devices):
        print(f"{i}: {device}")
    camera_input = input("Enter the camera number [default: 0]: ").strip()
    camera_choice = int(camera_input) if camera_input else 0
    video_source = camera_choice
    print(
        f"Using camera: {camera_devices[camera_choice] if camera_devices else 'Default camera'}")

elif input_type == 1:  # RTSP/RTMP stream
    print("Enter RTSP/RTMP URL:")
    print("Examples:")
    print("  rtsp://username:password@192.168.1.100:554/stream")
    print("  rtsp://192.168.1.100:8554/live")
    print("  rtmp://live-server.com/live/stream_key")

    while True:
        rtsp_url = input("RTSP/RTMP URL: ").strip()
        if not rtsp_url:
            print("URL cannot be empty")
            continue

        print("Validating stream connection...")
        is_valid, message = validate_rtsp_url(rtsp_url)
        print(message)

        if is_valid:
            video_source = rtsp_url
            break
        else:
            retry = input("Try again? (y/n) [default: y]: ").strip().lower()
            if retry == 'n':
                print("Falling back to default camera")
                video_source = 0
                break

elif input_type == 2:  # Video file
    video_path = input("Enter path to video file: ").strip()
    if os.path.exists(video_path):
        video_source = video_path
        print(f"Using video file: {video_path}")
    else:
        print("Video file not found. Falling back to default camera")
        video_source = 0

# Initialize video capture with enhanced RTSP handling
print("Initializing video capture...")
vid = cv2.VideoCapture(video_source)

# Configure capture settings based on input type
vid = configure_rtsp_capture(vid, is_rtsp=(input_type == 1))

if not vid.isOpened():
    print("Error: Cannot open video source. Trying default camera...")
    vid = cv2.VideoCapture(0)
    vid = configure_rtsp_capture(vid, is_rtsp=False)
    input_type = 0  # Reset to camera type
    if not vid.isOpened():
        print("Error: Cannot open any video source!")
        exit(1)

# Resolution selection (only for physical cameras)
if input_type == 0:
    print("\nChoose resolution:")
    resolutions = [
        {"name": "HD (1280x720) - 16:9", "width": 1280, "height": 720},
        {"name": "HD (1280x800) - 16:10", "width": 1280, "height": 800},
        {"name": "Full HD (1920x1080) - 16:9", "width": 1920, "height": 1080},
        {"name": "Full HD (1920x1200) - 16:10", "width": 1920, "height": 1200}
    ]

    for i, res in enumerate(resolutions):
        print(f"{i}: {res['name']}")

    resolution_input = input("Enter resolution number [default: 0]: ").strip()
    resolution_choice = int(resolution_input) if resolution_input else 0
    frame_width = resolutions[resolution_choice]["width"]
    frame_height = resolutions[resolution_choice]["height"]

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Get actual dimensions
actual_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual capture resolution: {actual_width}x{actual_height}")

# Person class ID
if model_choice == 0 or model_choice == 1:
    person_class_id = 0
else:
    person_class_id = yolo_classes.index("person")

# Detection confidence
conf_input = input(
    "Enter detection confidence threshold (0.1-0.9, recommended 0.3-0.5) [default: 0.5]: ").strip()
conf_threshold = float(conf_input) if conf_input else 0.5
conf_threshold = max(0.1, min(0.9, conf_threshold))

# Feather amount selection for edges
feather_input = input(
    "Enter edge feathering amount (0-20, 0 for sharp edges, 10 recommended) [default: 10]: ").strip()
feather_amount = int(feather_input) if feather_input else 10
feather_amount = max(0, min(20, feather_amount))

# Enhanced background mode selection
print("\nChoose background mode:")
print("Keep Body modes (show person, replace background):")
print("  0: Keep Body - Green Background")
print("  1: Keep Body - Blue Background")
print("  2: Keep Body - Transparent Background")
print("  3: Keep Body - Custom Background Image")
print("  4: Keep Body - Original Video Background")
print("  5: Keep Body - Normal (No Effect)")
print("\nRemove Body modes (hide person, show background):")
print("  6: Remove Body - Green Fill in Body Area")
print("  7: Remove Body - Blue Fill in Body Area")
print("  8: Remove Body - Transparent Body Area")
print("  9: Remove Body - Custom Background Image")
print("  10: Remove Body - Original Background")
print("  11: Remove Body - Normal (No Effect)")

bg_input = input("Choose background mode [default: 0]: ").strip()
bg_mode = int(bg_input) if bg_input else 0

# Handle background setup for different modes
background_color = None
background_image = None
# Store recent frames for background
original_background_buffer = deque(maxlen=30)

if bg_mode == 2 or bg_mode == 8:  # Transparent modes
    # Transparent modes - no background setup needed
    background_color = None
elif bg_mode == 0 or bg_mode == 6:  # Green
    background_color = (0, 255, 0)  # Green in BGR
elif bg_mode == 1 or bg_mode == 7:  # Blue
    background_color = (255, 0, 0)  # Blue in BGR format
elif bg_mode == 3 or bg_mode == 9:  # Custom background image (both modes)
    bg_image_path = input("Enter path to background image: ").strip()
    if bg_image_path:
        bg_image, msg = load_background_image(
            bg_image_path, actual_width, actual_height)
        if bg_image is not None:
            background_image = bg_image
            print(msg)
        else:
            print(f"Error: {msg}. Falling back to green background.")
            background_color = (0, 255, 0)
            bg_mode = 0 if bg_mode == 3 else 6  # Fallback to appropriate green mode
    else:
        print("No background image specified. Falling back to green background.")
        background_color = (0, 255, 0)
        bg_mode = 0 if bg_mode == 3 else 6  # Fallback to appropriate green mode
elif bg_mode == 4 or bg_mode == 10:  # Original video background
    print("Using original video feed as background")
    # Background will be captured from video frames
elif bg_mode == 5 or bg_mode == 11:  # Normal (no effect)
    print("Using normal mode (no background effect)")
    # No background processing needed

# Initialize Syphon
syphon_input = input(
    "Do you want to enable Syphon output? (0/1) [default: 0]: ").strip()
use_syphon = bool(int(syphon_input)) if syphon_input else False
syphon_server, mtl_device = None, None

if use_syphon:
    syphon_name = "PersonBackgroundRemoval"
    print(f"Creating Syphon server '{syphon_name}'...")
    syphon_server, mtl_device = create_syphon_server(syphon_name)

    if syphon_server and mtl_device:
        print(
            f"Syphon ready. In OBS, add a 'Syphon Client' source and select '{syphon_name}'")
    else:
        print("Failed to initialize Syphon. Continuing without Syphon output.")
        use_syphon = False

# Initialize NDI
ndi_input = input(
    "Do you want to enable NDI output? (0/1) [default: 0]: ").strip()
use_ndi = bool(int(ndi_input)) if ndi_input else False
ndi_sender = None

if use_ndi:
    ndi_name = "PersonBackgroundRemoval"
    print(f"Creating NDI sender '{ndi_name}'...")
    ndi_sender = create_ndi_sender(ndi_name, actual_width, actual_height)

    if ndi_sender:
        print(
            f"NDI ready. Look for '{ndi_name}' in NDI-compatible applications (OBS, vMix, etc.)")
        print("Make sure NDI Tools are installed and firewall allows NDI traffic.")
    else:
        print("Failed to initialize NDI. Continuing without NDI output.")
        use_ndi = False

# Kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Timing variables
start_time = time.time()
frame_id = 0
show_indicators = False
include_area_between = True

# Performance optimization variables
skip_frames = 0
process_every_n_frames = 1
last_key_time = time.time()
key_press_buffer = deque(maxlen=5)
fps_values = deque(maxlen=30)
processing_times = deque(maxlen=30)

# Thread-safe variables
is_running = True
current_frame = None
processed_frame = None
frame_ready = threading.Event()
processing_done = threading.Event()

# NDI stable transmission variables
ndi_frame_queue = deque(maxlen=5)  # Small buffer to prevent backup but ensure stability
ndi_frame_ready = threading.Event()
ndi_lock = threading.Lock()
ndi_stats = {'sent': 0, 'dropped': 0, 'queue_full': 0}

# Lock for thread synchronization
frame_lock = threading.Lock()


def get_background_frame(original_frame):
    """Get the appropriate background based on the current mode"""
    if (bg_mode == 3 or bg_mode == 9) and background_image is not None:
        # Custom background image (both keep body and remove body modes)
        return background_image.copy()
    elif bg_mode == 4 or bg_mode == 10:
        # Original video background - use a slightly older frame or current frame
        if len(original_background_buffer) > 5:
            # Use frame from 5 frames ago
            return original_background_buffer[-5]
        else:
            return original_frame.copy()
    elif bg_mode == 5 or bg_mode == 11:
        # Normal mode - return original frame unchanged
        return original_frame.copy()
    elif background_color is not None:
        # Solid color background
        return np.full_like(original_frame, background_color)
    else:
        # Fallback
        return original_frame.copy()


def process_frame(frame, process_id):
    """Process a single frame - this is the computationally heavy part"""
    start_proc_time = time.time()

    # Store frame for background use
    original_background_buffer.append(frame.copy())

    # Model inference
    results = model.predict(frame, stream=True, conf=conf_threshold)

    # Process segmentation masks
    combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    all_person_contours = []
    person_detected = False

    for result in results:
        if result.masks is None:
            continue

        for mask, box in zip(result.masks.xy, result.boxes):
            if int(box.cls[0]) == person_class_id:
                person_detected = True
                points = np.int32([mask])
                all_person_contours.append(points)

    # If persons are detected, create a unified mask
    if person_detected and all_person_contours:
        hull_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        for contour in all_person_contours:
            cv2.fillPoly(combined_mask, contour, 255)

        if len(all_person_contours) > 1 and include_area_between:
            all_points = np.vstack([contour.reshape(-1, 2)
                                   for contour in all_person_contours])
            hull = cv2.convexHull(all_points)
            cv2.fillConvexPoly(hull_mask, hull, 255)
            combined_mask = cv2.bitwise_or(combined_mask, hull_mask)

        # Apply morphological operations
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        if feather_amount > 0:
            dist_transform = cv2.distanceTransform(
                combined_mask, cv2.DIST_L2, 5)
            dist_transform = cv2.normalize(
                dist_transform, None, 0, 255, cv2.NORM_MINMAX)
            feathered_mask = cv2.GaussianBlur(
                combined_mask, (feather_amount*2+1, feather_amount*2+1), 0)
            combined_mask = feathered_mask
        else:
            combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    if person_detected:
        mask_3channel = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR) / 255.0

        if bg_mode <= 5:  # Keep Body modes (0-5)
            if bg_mode == 2:  # Transparent Background
                frame_transparent = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                frame_transparent[:, :, :3] = (
                    frame * mask_3channel).astype(np.uint8)
                frame_transparent[:, :, 3] = combined_mask
                output_frame = frame_transparent
            elif bg_mode == 5:  # Normal (no effect) - show original frame
                output_frame = frame.copy()
            else:  # Color/Image/Original Background (0, 1, 3, 4)
                bg_frame = get_background_frame(frame)
                output_frame = frame * mask_3channel + \
                    bg_frame * (1 - mask_3channel)
                output_frame = output_frame.astype(np.uint8)

        else:  # Remove Body modes (6-11)
            inverted_mask_3channel = 1 - mask_3channel

            if bg_mode == 8:  # Transparent Body Area
                frame_transparent = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                frame_transparent[:, :, :3] = (
                    frame * inverted_mask_3channel).astype(np.uint8)
                inverted_mask = cv2.bitwise_not(combined_mask)
                frame_transparent[:, :, 3] = inverted_mask
                output_frame = frame_transparent
            elif bg_mode == 11:  # Normal (no effect) - show original frame
                output_frame = frame.copy()
            elif bg_mode == 10:  # Remove Body - Original Background
                bg_frame = get_background_frame(frame)
                output_frame = bg_frame * inverted_mask_3channel + frame * mask_3channel
                output_frame = output_frame.astype(np.uint8)
            elif bg_mode == 9:  # Remove Body - Custom Background
                bg_frame = get_background_frame(frame)
                output_frame = bg_frame * inverted_mask_3channel + bg_frame * mask_3channel
                output_frame = output_frame.astype(np.uint8)
            else:  # Color Fill (6, 7)
                color_fill = np.full_like(frame, background_color)
                output_frame = frame * inverted_mask_3channel + color_fill * mask_3channel
                output_frame = output_frame.astype(np.uint8)
    else:
        # No person detected
        if bg_mode == 2:  # Keep Body Transparent - show nothing
            output_frame = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
        elif bg_mode == 8:  # Remove Body Transparent - show full background
            output_frame = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            output_frame[:, :, :3] = frame
            output_frame[:, :, 3] = 255
        elif bg_mode == 5 or bg_mode == 11:  # Normal modes - show original frame
            output_frame = frame.copy()
        elif bg_mode >= 6:  # Remove Body modes - show original frame
            output_frame = frame.copy()
        elif bg_mode == 4:  # Keep Body Original - show background when no person
            output_frame = get_background_frame(frame)
        else:  # Keep Body modes - show background/color
            output_frame = get_background_frame(frame)

    # Flip horizontally for natural view (except for RTSP streams)
    if input_type == 0:  # Only flip for physical cameras
        output_frame = cv2.flip(output_frame, 1)

    proc_time = time.time() - start_proc_time
    processing_times.append(proc_time)

    return output_frame, person_detected, proc_time


def processing_thread():
    """Thread function for frame processing"""
    global processed_frame, current_frame
    local_frame_id = 0

    while is_running:
        if frame_ready.wait(timeout=0.1):
            frame_ready.clear()

            with frame_lock:
                if current_frame is None:
                    processing_done.set()
                    continue
                frame_to_process = current_frame.copy()

            if local_frame_id % process_every_n_frames == 0:
                try:
                    output_frame, detected, proc_time = process_frame(frame_to_process, local_frame_id)

                    with frame_lock:
                        processed_frame = (output_frame, detected, proc_time)
                except Exception as e:
                    print(f"Error in processing thread: {e}")

            local_frame_id += 1
            processing_done.set()


def ndi_publishing_thread():
    """Stable NDI publishing thread with proper queue management and cleanup"""
    global ndi_frame_queue, ndi_stats
    target_ndi_fps = 25  # Higher FPS for better stability
    frame_interval = 1.0 / target_ndi_fps
    last_publish_time = 0
    
    print(f"NDI thread started - targeting {target_ndi_fps} FPS")
    
    while is_running:
        try:
            current_time = time.time()
            
            # Wait for frames or timeout to check is_running
            if ndi_frame_ready.wait(timeout=0.05):  # Shorter timeout for responsive cleanup
                ndi_frame_ready.clear()
            
            # Rate limiting
            if current_time - last_publish_time < frame_interval:
                continue
            
            # Get frame from queue
            frame_data = None
            with ndi_lock:
                if ndi_frame_queue:
                    frame_data = ndi_frame_queue.popleft()
            
            if frame_data is None:
                continue
                
            frame_to_publish, is_transparent = frame_data
            
            # Publish frame
            try:
                if is_transparent:
                    success = publish_frame_to_ndi(frame_to_publish, ndi_sender, is_rgba=True, fps=target_ndi_fps)
                else:
                    success = publish_frame_to_ndi(frame_to_publish, ndi_sender, fps=target_ndi_fps)
                    
                if success:
                    ndi_stats['sent'] += 1
                    last_publish_time = current_time
                else:
                    ndi_stats['dropped'] += 1
                    
                # Status update every 100 frames
                if ndi_stats['sent'] % 100 == 0 and ndi_stats['sent'] > 0:
                    total = ndi_stats['sent'] + ndi_stats['dropped']
                    success_rate = (ndi_stats['sent'] / total) * 100
                    print(f"NDI: {ndi_stats['sent']} sent, {success_rate:.1f}% success, queue_full: {ndi_stats['queue_full']}")
                    
            except Exception as e:
                ndi_stats['dropped'] += 1
                if ndi_stats['dropped'] % 20 == 0:
                    print(f"NDI publish error: {e}")
                    
        except Exception as e:
            if is_running:  # Only print if we're not shutting down
                print(f"NDI thread error: {e}")
            break
    
    print("NDI thread shutting down...")


def update_background_color():
    """Update background color based on current bg_mode"""
    global background_color

    if bg_mode in [2, 8]:  # Transparent modes
        background_color = None
    elif bg_mode in [0, 6]:  # Green modes
        background_color = (0, 255, 0)
    elif bg_mode in [1, 7]:  # Blue modes
        background_color = (255, 0, 0)
    elif bg_mode in [3, 4, 5, 9, 10, 11]:  # Custom/Original/Normal modes
        background_color = None


def handle_keys(key):
    """Handle keyboard inputs"""
    global show_indicators, include_area_between, is_running, process_every_n_frames, bg_mode

    if key == -1:
        return False

    key = key & 0xFF

    if key == ord('q'):
        print("Quit command received")
        is_running = False
        return True
    elif key == ord('h'):
        show_indicators = not show_indicators
        print(f"Indicators {'hidden' if not show_indicators else 'shown'}")
    elif key == ord('b'):
        include_area_between = not include_area_between
        print(
            f"Include area between bodies: {'On' if include_area_between else 'Off'}")
    elif key == ord('m'):
        # Cycle through all background modes
        bg_mode = (bg_mode + 1) % 12  # 12 total modes (0-11)
        update_background_color()
        mode_names = {
            0: "Keep Body - Green BG", 1: "Keep Body - Blue BG", 2: "Keep Body - Transparent BG",
            3: "Keep Body - Custom BG", 4: "Keep Body - Original BG", 5: "Keep Body - Normal",
            6: "Remove Body - Green Fill", 7: "Remove Body - Blue Fill", 8: "Remove Body - Transparent Area",
            9: "Remove Body - Custom BG", 10: "Remove Body - Original BG", 11: "Remove Body - Normal"
        }
        print(f"Switched to mode: {mode_names[bg_mode]}")
    elif key == ord('+') or key == ord('='):
        process_every_n_frames = max(1, process_every_n_frames - 1)
        print(
            f"Processing frequency increased: Every {process_every_n_frames} frame(s)")
    elif key == ord('-'):
        process_every_n_frames += 1
        print(
            f"Processing frequency decreased: Every {process_every_n_frames} frame(s)")

    return False


# Print startup information
print("\n=== Enhanced Real-time Body Background Removal ===")
print("New Features:")
print("  - RTSP/RTMP stream support")
print("  - Custom background images")
print("  - Original video background mode")
print("\nKeyboard Controls:")
print("  Q - Quit application")
print("  H - Hide/show info overlay")
print("  B - Toggle area between bodies mode")
print("  M - Cycle through all modes (0→1→2→3→4→5→6→7→8)")
print("  +/- - Adjust processing frequency")
print("\nModes: Keep Body (0-5) | Remove Body (6-11)")
print("Keep Body: 0: Green, 1: Blue, 2: Transparent, 3: Custom Image, 4: Original, 5: Normal")
print("Remove Body: 6: Green Fill, 7: Blue Fill, 8: Transparent, 9: Custom Image, 10: Original, 11: Normal")

if use_syphon:
    print("\nSyphon: Enabled - Broadcasting to 'PersonBackgroundRemoval'")
if use_ndi:
    print("\nNDI: Enabled - Broadcasting to 'PersonBackgroundRemoval'")

print("\nStarting video feed...\n")

# Start the processing thread
processing_thread = threading.Thread(target=processing_thread)
processing_thread.daemon = True
processing_thread.start()

# Start NDI publishing thread if NDI is enabled
ndi_thread = None
if use_ndi:
    ndi_thread = threading.Thread(target=ndi_publishing_thread)
    ndi_thread.daemon = True
    ndi_thread.start()
    print("NDI async publishing thread started")

# Main loop
while is_running:
    frame_start_time = time.time()
    frame_id += 1

    # Capture frame with RTSP reconnection logic
    ret, frame = vid.read()
    if not ret:
        if input_type == 1:  # RTSP stream - try to reconnect
            print("RTSP connection lost, attempting to reconnect...")
            vid.release()
            time.sleep(1)  # Wait a moment before reconnecting
            vid = cv2.VideoCapture(video_source)
            vid = configure_rtsp_capture(vid, is_rtsp=True)

            # Try to read again after reconnection
            ret, frame = vid.read()
            if not ret:
                print("Failed to reconnect to RTSP stream")
                break
            else:
                print("RTSP reconnection successful")

        elif input_type == 2:  # Video file - loop or exit
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to beginning
            continue
        else:
            print("Failed to capture frame from camera")
            break

    # Check for key presses
    key = cv2.waitKey(1)
    if key != -1:
        key_press_buffer.append(key)
        last_key_time = time.time()

    if key_press_buffer:
        key = key_press_buffer.popleft()
        if handle_keys(key):
            break

    # Pass the new frame to the processing thread
    with frame_lock:
        current_frame = frame.copy()

    frame_ready.set()

    # Wait until frame is processed
    processing_done.wait(timeout=1/30)
    processing_done.clear()

    # Get the processed frame if available
    display_frame = None
    person_detected = False
    proc_time = 0

    with frame_lock:
        if processed_frame is not None:
            display_frame, person_detected, proc_time = processed_frame

    if display_frame is None:
        display_frame = cv2.flip(frame, 1) if input_type == 0 else frame

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time
    fps_values.append(fps)
    avg_fps = sum(fps_values) / len(fps_values)
    avg_proc_time = sum(processing_times) / \
        max(1, len(processing_times)) * 1000

    # Add information overlay
    if display_frame is not None and show_indicators:
        mode_names = {
            0: "Keep Body - Green BG", 1: "Keep Body - Blue BG", 2: "Keep Body - Transparent BG",
            3: "Keep Body - Custom BG", 4: "Keep Body - Original BG", 5: "Keep Body - Normal",
            6: "Remove Body - Green Fill", 7: "Remove Body - Blue Fill", 8: "Remove Body - Transparent Area",
            9: "Remove Body - Custom BG", 10: "Remove Body - Original BG", 11: "Remove Body - Normal"
        }

        if bg_mode not in [2, 8]:  # Non-transparent modes
            text_color = (255, 255, 255) if bg_mode >= 5 else (50, 170, 50)

            cv2.putText(display_frame, f"Mode: {mode_names.get(bg_mode, 'Unknown')} | FPS: {round(avg_fps, 1)} (Proc: {round(avg_proc_time, 1)}ms)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(display_frame, f"Input: {'Camera' if input_type == 0 else 'RTSP/Stream' if input_type == 1 else 'Video File'} | Person: {'Yes' if person_detected else 'No'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(display_frame, f"Area between: {'On' if include_area_between else 'Off'} | Feather: {feather_amount} | Conf: {conf_threshold}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(display_frame, "Q: quit | H: hide | B: area mode | M: cycle modes | +/-: frequency",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Publish to Syphon
    if use_syphon and syphon_server and mtl_device and display_frame is not None:
        if frame_id % 100 == 0:
            print(f"Publishing frame {frame_id} to Syphon")

        if bg_mode in [2, 8]:
            success = publish_frame_to_syphon(
                display_frame, syphon_server, mtl_device, is_rgba=True)
        else:
            success = publish_frame_to_syphon(
                display_frame, syphon_server, mtl_device)

        if not success and frame_id % 30 == 0:
            print("Failed to publish to Syphon")

    # Queue frame for stable NDI publishing
    if use_ndi and ndi_sender and display_frame is not None:
        is_transparent = bg_mode in [2, 8]
        
        # Try to add to queue without blocking
        if ndi_lock.acquire(blocking=False):
            try:
                # Add to queue if not full
                if len(ndi_frame_queue) < ndi_frame_queue.maxlen:
                    ndi_frame_queue.append((display_frame.copy(), is_transparent))
                    ndi_frame_ready.set()
                else:
                    # Queue is full - replace oldest frame to maintain freshness
                    ndi_frame_queue.clear()  # Clear to prevent backup
                    ndi_frame_queue.append((display_frame.copy(), is_transparent))
                    ndi_stats['queue_full'] += 1
                    ndi_frame_ready.set()
            finally:
                ndi_lock.release()

    # Display the frame
    if bg_mode in [2, 8] and display_frame is not None:
        # Create checkerboard preview for transparent modes
        checker_size = 20
        checker_frame = np.zeros(
            (display_frame.shape[0], display_frame.shape[1], 3), dtype=np.uint8)

        checker_pattern = np.zeros(
            (checker_size*2, checker_size*2, 3), dtype=np.uint8)
        checker_pattern[:checker_size, :checker_size] = [255, 255, 255]
        checker_pattern[checker_size:, checker_size:] = [255, 255, 255]

        for i in range(0, display_frame.shape[0], checker_size*2):
            for j in range(0, display_frame.shape[1], checker_size*2):
                h = min(checker_size*2, display_frame.shape[0]-i)
                w = min(checker_size*2, display_frame.shape[1]-j)
                checker_frame[i:i+h, j:j+w] = checker_pattern[:h, :w]

        if display_frame.shape[2] == 4:
            alpha = display_frame[:, :, 3:4] / 255.0
            display_frame_rgb = display_frame[:, :, :3]
            preview = checker_frame * (1 - alpha) + (display_frame_rgb * alpha)
            preview = preview.astype(np.uint8)
            cv2.imshow('Enhanced Person Background Removal', preview)
    elif display_frame is not None:
        cv2.imshow('Enhanced Person Background Removal', display_frame)

    # Frame rate control
    frame_time = time.time() - frame_start_time
    if frame_time < 1/60:
        time.sleep(1/60 - frame_time)

# Cleanup
print("\nShutting down...")
is_running = False

# Clean up processing thread
processing_thread.join(timeout=1.0)
if processing_thread.is_alive():
    print("Warning: Processing thread did not terminate cleanly")

# Clean up NDI thread properly
if use_ndi and ndi_thread:
    print("Stopping NDI thread...")
    # Signal thread to wake up and check is_running
    ndi_frame_ready.set()
    
    # Wait for thread to terminate
    ndi_thread.join(timeout=2.0)
    if ndi_thread.is_alive():
        print("Warning: NDI thread did not terminate cleanly")
    else:
        print("NDI thread stopped successfully")

vid.release()

# Clean up external resources
if use_syphon:
    cleanup_syphon(syphon_server)

if use_ndi:
    print("Cleaning up NDI sender...")
    cleanup_ndi(ndi_sender)
    print("NDI cleanup complete")

cv2.destroyAllWindows()
print("Application shutdown complete")

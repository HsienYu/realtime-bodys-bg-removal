"""
Alternative version of app.py that fully integrates the working Syphon implementation from mask.py
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

# Camera selection


def list_cameras():
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


camera_devices = list_cameras()
print("Choose a camera: ")
for i, device in enumerate(camera_devices):
    print(f"{i}: {device}")
camera_input = input("Enter the camera number [default: 0]: ").strip()
camera_choice = int(camera_input) if camera_input else 0
print(f"Using camera: {camera_devices[camera_choice]}")

# Resolution selection
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

vid = cv2.VideoCapture(camera_choice)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

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

# Background mode selection with consistent options
print("\nChoose background mode:")
print("Keep Body modes (show person, replace background):")
print("  0: Keep Body - Green Background")
print("  1: Keep Body - Blue Background")
print("  2: Keep Body - Transparent Background")
print("\nRemove Body modes (hide person, show background):")
print("  3: Remove Body - Green Fill in Body Area")
print("  4: Remove Body - Blue Fill in Body Area")
print("  5: Remove Body - Transparent Body Area")
bg_input = input(
    "Choose background mode [default: 0]: ").strip()
bg_mode = int(bg_input) if bg_input else 0

# Handle background color for different modes
if bg_mode == 2 or bg_mode == 5:
    # Transparent modes - no background color needed
    background_color = None
else:
    # Set colors for green/blue modes
    if bg_mode == 0 or bg_mode == 3:  # Green (Keep Body or Remove Body)
        background_color = (0, 255, 0)  # Green in BGR
    elif bg_mode == 1 or bg_mode == 4:  # Blue (Keep Body or Remove Body)
        background_color = (255, 0, 0)  # Blue in BGR format (B, G, R)
    else:
        background_color = (0, 255, 0)  # Default to green

# Initialize Syphon using the working implementation from mask.py
syphon_input = input("Do you want to enable Syphon output? (0/1) [default: 0]: ").strip()
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
ndi_input = input("Do you want to enable NDI output? (0/1) [default: 0]: ").strip()
use_ndi = bool(int(ndi_input)) if ndi_input else False
ndi_sender = None

if use_ndi:
    ndi_name = "PersonBackgroundRemoval"
    print(f"Creating NDI sender '{ndi_name}'...")
    ndi_sender = create_ndi_sender(ndi_name, actual_width, actual_height)
    
    if ndi_sender:
        print(f"NDI ready. Look for '{ndi_name}' in NDI-compatible applications (OBS, vMix, etc.)")
        print("Make sure NDI Tools are installed and firewall allows NDI traffic.")
    else:
        print("Failed to initialize NDI. Continuing without NDI output.")
        use_ndi = False

# Kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Timing variables
start_time = time.time()
frame_id = 0
show_indicators = False  # New toggle variable for text indicators visibility
include_area_between = True  # New toggle variable for including area between bodies

# Performance optimization variables
skip_frames = 0  # How many frames to skip inference on
process_every_n_frames = 1  # Process only every n frames
last_key_time = time.time()  # Track last key press time
key_press_buffer = deque(maxlen=5)  # Buffer for key presses
fps_values = deque(maxlen=30)  # Track recent FPS values
processing_times = deque(maxlen=30)  # Track processing times

# Thread-safe variables
is_running = True
current_frame = None
processed_frame = None
frame_ready = threading.Event()
processing_done = threading.Event()

# Lock for thread synchronization
frame_lock = threading.Lock()


def process_frame(frame, process_id):
    """Process a single frame - this is the computationally heavy part"""
    start_proc_time = time.time()

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
                # Instead of filling individual masks, collect all contours
                all_person_contours.append(points)

    # If persons are detected, create a unified mask that includes area between bodies
    if person_detected and all_person_contours:
        # Create a blank mask and draw all contours
        hull_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        # Option 1: Simple approach - draw all contours
        for contour in all_person_contours:
            cv2.fillPoly(combined_mask, contour, 255)

        # Option 2: If multiple people, try to include area between them (only if toggled on)
        if len(all_person_contours) > 1 and include_area_between:
            # Convert all points to a single flat array
            all_points = np.vstack([contour.reshape(-1, 2)
                                   for contour in all_person_contours])

            # Compute convex hull to enclose all detected people
            hull = cv2.convexHull(all_points)
            cv2.fillConvexPoly(hull_mask, hull, 255)

            # Combine individual masks with hull mask for final result
            combined_mask = cv2.bitwise_or(combined_mask, hull_mask)

        # Apply morphological operations to improve mask edges
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        if feather_amount > 0:
            # Create feathered edges for smoother transitions
            # Convert binary mask to distance field
            dist_transform = cv2.distanceTransform(
                combined_mask, cv2.DIST_L2, 5)

            # Normalize and scale the distance transform to create a gradient at edges
            dist_transform = cv2.normalize(
                dist_transform, None, 0, 255, cv2.NORM_MINMAX)

            # Apply Gaussian blur to create smooth gradients at edges
            feathered_mask = cv2.GaussianBlur(
                combined_mask, (feather_amount*2+1, feather_amount*2+1), 0)

            # Use the feathered mask instead of the binary mask
            combined_mask = feathered_mask
        else:
            # Original behavior - just apply a small Gaussian blur
            combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    if person_detected:
        # Create mask channels for blending
        mask_3channel = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR) / 255.0

        if bg_mode <= 2:  # Keep Body modes (0: Green, 1: Blue, 2: Transparent)
            if bg_mode == 2:  # Keep Body - Transparent Background
                # Create an RGBA image
                frame_transparent = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                
                # Copy the person (RGB channels)
                frame_transparent[:, :, :3] = (frame * mask_3channel).astype(np.uint8)
                
                # Set alpha channel: 255 for person (opaque), 0 for background (transparent)
                frame_transparent[:, :, 3] = combined_mask
                
                output_frame = frame_transparent
            else:  # Keep Body - Green (0) or Blue (1) Background
                colored_bg = np.full_like(frame, background_color)
                output_frame = frame * mask_3channel + colored_bg * (1 - mask_3channel)
                output_frame = output_frame.astype(np.uint8)
        
        else:  # Remove Body modes (3: Green, 4: Blue, 5: Transparent)
            # Invert the mask to show everything EXCEPT the body
            inverted_mask_3channel = 1 - mask_3channel
            
            if bg_mode == 5:  # Remove Body - Transparent Body Area
                # Create an RGBA image
                frame_transparent = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                
                # Copy the background (RGB channels)
                frame_transparent[:, :, :3] = (frame * inverted_mask_3channel).astype(np.uint8)
                
                # Set alpha channel: 255 for background (opaque), 0 for person area (transparent)
                inverted_mask = cv2.bitwise_not(combined_mask)
                frame_transparent[:, :, 3] = inverted_mask
                
                output_frame = frame_transparent
            else:  # Remove Body - Green (3) or Blue (4) Fill
                # Show original background, fill body areas with color
                color_fill = np.full_like(frame, background_color)
                output_frame = frame * inverted_mask_3channel + color_fill * mask_3channel
                output_frame = output_frame.astype(np.uint8)
    else:
        # No person detected
        if bg_mode == 2 or bg_mode == 5:  # Transparent modes
            # Create a fully opaque frame for transparent modes when no person detected
            if bg_mode == 2:  # Keep Body Transparent - show nothing (fully transparent)
                output_frame = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                # Alpha channel stays 0 (fully transparent)
            else:  # Remove Body Transparent - show full background (fully opaque)
                output_frame = np.zeros(
                    (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                output_frame[:, :, :3] = frame
                output_frame[:, :, 3] = 255  # Fully opaque
        elif bg_mode >= 3:  # Remove Body modes (3: Green, 4: Blue)
            # No body detected, so show original frame (nothing to remove)
            output_frame = frame.copy()
        else:
            # Keep Body modes (0: Green, 1: Blue) - no body detected, show background color
            output_frame = np.full_like(frame, background_color)

    # Flip horizontally for natural view
    output_frame = cv2.flip(output_frame, 1)

    proc_time = time.time() - start_proc_time
    processing_times.append(proc_time)

    return output_frame, person_detected, proc_time


def processing_thread():
    """Thread function for frame processing"""
    global processed_frame, current_frame
    local_frame_id = 0

    while is_running:
        # Wait for a new frame to be ready
        if frame_ready.wait(timeout=0.1):
            frame_ready.clear()  # Reset event

            # Get the frame to process with lock protection
            with frame_lock:
                if current_frame is None:
                    processing_done.set()
                    continue
                frame_to_process = current_frame.copy()

            # Only process certain frames to avoid overloading
            if local_frame_id % process_every_n_frames == 0:
                try:
                    # Process the frame
                    output_frame, detected, proc_time = process_frame(
                        frame_to_process, local_frame_id)

                    # Store the processed frame with lock protection
                    with frame_lock:
                        processed_frame = (output_frame, detected, proc_time)
                except Exception as e:
                    print(f"Error in processing thread: {e}")

            local_frame_id += 1
            processing_done.set()  # Signal that processing is done


def update_background_color():
    """Update background color based on current bg_mode"""
    global background_color
    
    if bg_mode == 2 or bg_mode == 5:
        # Transparent modes - no background color needed
        background_color = None
    else:
        # Set colors for green/blue modes
        if bg_mode == 0 or bg_mode == 3:  # Green (Keep Body or Remove Body)
            background_color = (0, 255, 0)  # Green in BGR
        elif bg_mode == 1 or bg_mode == 4:  # Blue (Keep Body or Remove Body)
            background_color = (255, 0, 0)  # Blue in BGR format (B, G, R)
        else:
            background_color = (0, 255, 0)  # Default to green

def handle_keys(key):
    """Handle keyboard inputs separately for more responsive hotkeys"""
    global show_indicators, include_area_between, is_running, process_every_n_frames, skip_frames, bg_mode

    if key == -1:  # No key pressed
        return False

    key = key & 0xFF  # Get the ASCII value

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
        all_modes = [0, 1, 2, 3, 4, 5]  # All 6 modes
        current_index = bg_mode if bg_mode in all_modes else 0
        next_index = (current_index + 1) % len(all_modes)
        bg_mode = all_modes[next_index]
        update_background_color()  # Update color for new mode
        mode_names = {
            0: "Keep Body - Green BG", 1: "Keep Body - Blue BG", 2: "Keep Body - Transparent BG",
            3: "Remove Body - Green Fill", 4: "Remove Body - Blue Fill", 5: "Remove Body - Transparent Area"
        }
        print(f"Switched to mode: {mode_names[bg_mode]}")
    elif key == ord('k'):
        # Cycle through Keep Body modes (0: Green, 1: Blue, 2: Transparent)
        if bg_mode <= 2:
            bg_mode = (bg_mode + 1) % 3  # Cycle through 0, 1, 2
        else:
            bg_mode = 0  # Switch from Remove Body to Keep Body
        update_background_color()  # Update color for new mode
        keep_names = {0: "Green", 1: "Blue", 2: "Transparent"}
        print(f"Keep Body mode: {keep_names[bg_mode]} background")
    elif key == ord('r'):
        # Cycle through Remove Body modes (3: Green, 4: Blue, 5: Transparent)
        if bg_mode >= 3:
            bg_mode = 3 + ((bg_mode - 3 + 1) % 3)  # Cycle through 3, 4, 5
        else:
            bg_mode = 3  # Switch from Keep Body to Remove Body
        update_background_color()  # Update color for new mode
        remove_names = {3: "Green Fill", 4: "Blue Fill", 5: "Transparent Area"}
        print(f"Remove Body mode: {remove_names[bg_mode]}")
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
print("\n=== Real-time Body Background Removal ===")
print("Keyboard Controls:")
print("  Q - Quit application")
print("  H - Hide/show info overlay")
print("  B - Toggle area between bodies mode")
print("  M - Cycle through all modes (0→1→2→3→4→5)")
print("  K - Cycle Keep Body modes (Green→Blue→Transparent)")
print("  R - Cycle Remove Body modes (Green→Blue→Transparent)")
print("  +/- - Adjust processing frequency")
print("\nModes: Keep Body (0-2) | Remove Body (3-5)")
print("Colors: Green, Blue, Transparent for each category")

if use_syphon:
    print("\nSyphon: Enabled - Broadcasting to 'PersonBackgroundRemoval'")
if use_ndi:
    print("\nNDI: Enabled - Broadcasting to 'PersonBackgroundRemoval'")
    print("Note: Ensure NDI Tools are installed and firewall allows NDI")
    
print("\nStarting camera feed...\n")

# Start the processing thread
processing_thread = threading.Thread(target=processing_thread)
processing_thread.daemon = True
processing_thread.start()

# Main loop
while is_running:
    frame_start_time = time.time()
    frame_id += 1

    # Capture frame
    ret, frame = vid.read()
    if not ret:
        break

    # Check for key presses more frequently for responsiveness
    key = cv2.waitKey(1)
    if key != -1:  # If a key was pressed
        key_press_buffer.append(key)
        last_key_time = time.time()

    # Process any pending key presses
    if key_press_buffer:
        key = key_press_buffer.popleft()
        if handle_keys(key):
            break  # Exit if handle_keys returns True (quit)

    # Pass the new frame to the processing thread
    with frame_lock:
        current_frame = frame.copy()

    # Signal that a new frame is ready
    frame_ready.set()

    # Wait until frame is processed or timeout (ensures UI responsiveness)
    processing_done.wait(timeout=1/30)  # Limit to 30fps for UI
    processing_done.clear()

    # Get the processed frame if available
    display_frame = None
    person_detected = False
    proc_time = 0

    with frame_lock:
        if processed_frame is not None:
            display_frame, person_detected, proc_time = processed_frame

    # Use the processed frame or the last one if not available
    if display_frame is None:
        # If no processed frame is ready, just show the raw frame
        display_frame = cv2.flip(frame, 1)

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time
    fps_values.append(fps)
    avg_fps = sum(fps_values) / len(fps_values)
    avg_proc_time = sum(processing_times) / \
        max(1, len(processing_times)) * 1000  # ms

    # Add information overlay
    if display_frame is not None and show_indicators:
        # Define mode names for display
        mode_names = {
            0: "Keep Body - Green BG",
            1: "Keep Body - Blue BG", 
            2: "Keep Body - Transparent BG",
            3: "Remove Body - Green Fill",
            4: "Remove Body - Blue Fill",
            5: "Remove Body - Transparent Area"
        }
        
        if bg_mode not in [2, 5]:  # Standard display modes (not transparent)
            # Choose text color based on mode for better visibility
            text_color = (255, 255, 255) if bg_mode >= 3 else (50, 170, 50)
            
            cv2.putText(display_frame, f"Mode: {mode_names.get(bg_mode, 'Unknown')} | FPS: {round(avg_fps, 1)} (Proc: {round(avg_proc_time, 1)}ms)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(display_frame, f"Confidence: {conf_threshold} | Person detected: {'Yes' if person_detected else 'No'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(display_frame, f"Include area between: {'On' if include_area_between else 'Off'} | Feather: {feather_amount}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(display_frame, f"Process every: {process_every_n_frames} frames (+/- to adjust)",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            cv2.putText(display_frame, "Q: quit | H: hide info | B: area mode | M: cycle modes | K: keep body | R: remove body",
                        (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        elif bg_mode in [2, 5] and show_indicators:  # Transparent modes
            # Put text in a position that's less likely to interfere with the content
            mode_name = "Keep Body Transparent" if bg_mode == 2 else "Remove Body Transparent"
            cv2.putText(display_frame, f"{mode_name} | FPS: {round(avg_fps, 1)} | Area: {'On' if include_area_between else 'Off'} | Feather: {feather_amount}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Publish to Syphon
    if use_syphon and syphon_server and mtl_device and display_frame is not None:
        # Only log every 100 frames to avoid console spam
        if frame_id % 100 == 0:
            print(f"Publishing frame {frame_id} to Syphon")

        # Adjust publish_frame_to_syphon function call based on mode
        if bg_mode in [2, 5]:  # Transparent modes
            # For transparent mode, we need to handle RGBA
            success = publish_frame_to_syphon(
                display_frame, syphon_server, mtl_device, is_rgba=True)
        else:
            success = publish_frame_to_syphon(
                display_frame, syphon_server, mtl_device)

        if not success and frame_id % 30 == 0:  # Only log failures occasionally
            print("Failed to publish to Syphon")
    
    # Publish to NDI
    if use_ndi and ndi_sender and display_frame is not None:
        # Only log every 100 frames to avoid console spam
        if frame_id % 100 == 0:
            print(f"Publishing frame {frame_id} to NDI")
        
        # Calculate current FPS for NDI timing
        target_fps = min(60, int(avg_fps)) if avg_fps > 0 else 30
        
        # Adjust publish_frame_to_ndi function call based on mode
        if bg_mode in [2, 5]:  # Transparent modes
            # For transparent mode, we need to handle RGBA
            success = publish_frame_to_ndi(
                display_frame, ndi_sender, is_rgba=True, fps=target_fps)
        else:
            success = publish_frame_to_ndi(
                display_frame, ndi_sender, fps=target_fps)
        
        if not success and frame_id % 30 == 0:  # Only log failures occasionally
            print("Failed to publish to NDI")

    # Display the frame
    if bg_mode in [2, 5] and display_frame is not None:
        # For display purposes, create a preview with checkered background
        # to make transparent areas visible
        checker_size = 20
        checker_frame = np.zeros(
            (display_frame.shape[0], display_frame.shape[1], 3), dtype=np.uint8)

        # Create checker pattern (optimized)
        checker_pattern = np.zeros(
            (checker_size*2, checker_size*2, 3), dtype=np.uint8)
        checker_pattern[:checker_size, :checker_size] = [255, 255, 255]
        checker_pattern[checker_size:, checker_size:] = [255, 255, 255]

        # Tile the pattern (faster than nested loops)
        for i in range(0, display_frame.shape[0], checker_size*2):
            for j in range(0, display_frame.shape[1], checker_size*2):
                h = min(checker_size*2, display_frame.shape[0]-i)
                w = min(checker_size*2, display_frame.shape[1]-j)
                checker_frame[i:i+h, j:j+w] = checker_pattern[:h, :w]

        # Apply alpha blending
        if display_frame.shape[2] == 4:  # Make sure we have an alpha channel
            alpha = display_frame[:, :, 3:4] / 255.0
            display_frame_rgb = display_frame[:, :, :3]
            # Black pixels with full alpha replace checker pattern
            preview = checker_frame * (1 - alpha) + (display_frame_rgb * alpha)
            preview = preview.astype(np.uint8)
            cv2.imshow('Person Background Removal', preview)
    elif display_frame is not None:
        cv2.imshow('Person Background Removal', display_frame)

    # Frame rate control - sleep if we're processing too fast
    frame_time = time.time() - frame_start_time
    if frame_time < 1/60:  # Cap at 60fps for efficiency
        time.sleep(1/60 - frame_time)

# Stop the processing thread
is_running = False
processing_thread.join(timeout=1.0)

# Cleanup
vid.release()

if use_syphon:
    cleanup_syphon(syphon_server)

if use_ndi:
    cleanup_ndi(ndi_sender)

cv2.destroyAllWindows()

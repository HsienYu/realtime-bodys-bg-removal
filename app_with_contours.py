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
model_choice = int(input("Enter the model number: "))

is_onnx = bool(int(input("Do you want to use ONNX? (0/1): ")))

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
camera_choice = int(input("Enter the camera number: "))
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

resolution_choice = int(input("Enter resolution number: "))
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
conf_threshold = float(
    input("Enter detection confidence threshold (0.1-0.9, recommended 0.3-0.5): "))
conf_threshold = max(0.1, min(0.9, conf_threshold))

# Background color selection with transparent option
bg_mode = int(
    input("Choose background mode (0: Green, 1: Blue, 2: Custom, 3: Transparent Body): "))
if bg_mode == 2:
    r = int(input("Enter red component (0-255): "))
    g = int(input("Enter green component (0-255): "))
    b = int(input("Enter blue component (0-255): "))
    background_color = (b, g, r)  # OpenCV uses BGR
elif bg_mode == 3:
    # Transparent mode - will show body shape in green with transparent background
    background_color = None  # No background color
else:
    background_color = (0, 255, 0) if bg_mode == 0 else (255, 0, 0)

# Initialize Syphon using the working implementation from mask.py
use_syphon = bool(int(input("Do you want to enable Syphon output? (0/1): ")))
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

# Kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Timing variables
start_time = time.time()
frame_id = 0
show_indicators = True  # New toggle variable for text indicators visibility
include_area_between = True  # New toggle variable for including area between bodies

# Main loop
while True:
    frame_id += 1

    # Capture frame
    ret, frame = vid.read()
    if not ret:
        break

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
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    if person_detected:
        # Create mask channels for blending
        mask_3channel = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR) / 255.0

        if bg_mode == 3:  # Transparent mode
            # Create an RGBA image
            frame_transparent = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

            # Make everything black by default (already zeros)

            # Set alpha channel: 0 for bodies (transparent), 255 for background (opaque)
            # The logic is inverted compared to what you might expect:
            # - combined_mask has 255 where bodies are, 0 elsewhere
            # - For alpha, we want 0 where bodies are, 255 elsewhere
            inverted_mask = cv2.bitwise_not(
                combined_mask)  # 255 - combined_mask

            # Assign the alpha channel
            frame_transparent[:, :, 3] = inverted_mask

            # Use this as our frame
            frame = frame_transparent
        else:
            # Standard solid background mode
            colored_bg = np.full_like(frame, background_color)
            frame = frame * mask_3channel + colored_bg * (1 - mask_3channel)
            frame = frame.astype(np.uint8)
    else:
        # No person detected
        if bg_mode == 3:
            # Create a fully black opaque frame (not transparent)
            frame = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            # Set alpha to 255 (fully opaque)
            frame[:, :, 3] = 255
        else:
            # Use the default background color for the entire frame
            frame = np.full_like(frame, background_color)

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time

    # Flip horizontally for natural view
    frame = cv2.flip(frame, 1)

    # Add information overlay (only if not transparent)
    if bg_mode != 3 and show_indicators:  # Modified to check show_indicators
        cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, f"Confidence: {conf_threshold}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, f"Person detected: {'Yes' if person_detected else 'No'}",
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, f"Include area between: {'On' if include_area_between else 'Off'}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "Press Q to quit, H to hide info, B to toggle area mode", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Add minimal indicator even in transparent mode
    elif bg_mode == 3 and show_indicators:
        # Put text in a position that's less likely to interfere with the content
        cv2.putText(frame, f"Include area: {'On' if include_area_between else 'Off'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # Publish to Syphon using the method from mask.py
    if use_syphon and syphon_server and mtl_device:
        # Only log every 100 frames to avoid console spam
        if frame_id % 100 == 0:
            print(f"Publishing frame {frame_id} to Syphon")

        # Adjust publish_frame_to_syphon function call based on mode
        if bg_mode == 3:
            # For transparent mode, we need to handle RGBA
            success = publish_frame_to_syphon(
                frame, syphon_server, mtl_device, is_rgba=True)
        else:
            success = publish_frame_to_syphon(frame, syphon_server, mtl_device)

        if not success and frame_id % 30 == 0:  # Only log failures occasionally
            print("Failed to publish to Syphon")

    # Display the frame
    if bg_mode == 3:
        # For display purposes, create a preview with checkered background
        # to make transparent areas visible
        checker_size = 20
        display_frame = np.zeros(
            (frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        # Create checker pattern
        for i in range(0, frame.shape[0], checker_size * 2):
            for j in range(0, frame.shape[1], checker_size * 2):
                # White squares
                display_frame[i:min(i+checker_size, frame.shape[0]),
                              j:min(j+checker_size, frame.shape[1])] = [255, 255, 255]
                if j+checker_size < frame.shape[1] and i+checker_size < frame.shape[0]:
                    display_frame[i+checker_size:min(i+checker_size*2, frame.shape[0]),
                                  j+checker_size:min(j+checker_size*2, frame.shape[1])] = [255, 255, 255]

        # Apply alpha blending
        alpha = frame[:, :, 3:4] / 255.0

        # Black pixels (where RGB is 0,0,0) with full alpha (255)
        # should replace the checker pattern
        display_frame = display_frame * (1 - alpha) + (frame[:, :, :3] * alpha)
        display_frame = display_frame.astype(np.uint8)

        cv2.imshow('Person Background Removal', display_frame)
    else:
        cv2.imshow('Person Background Removal', frame)

    # Improved keyboard handling for more responsive hotkeys
    # Increased wait time from 1ms to 5ms for better key capture
    key = cv2.waitKey(5) & 0xFF

    # Process key presses - use elif structure for more efficient handling
    if key == ord('q'):
        print("Quit command received")
        break
    elif key == ord('h'):
        show_indicators = not show_indicators
        print(f"Indicators {'hidden' if not show_indicators else 'shown'}")
    elif key == ord('b'):
        include_area_between = not include_area_between
        print(
            f"Include area between bodies: {'On' if include_area_between else 'Off'}")

    # Remove the extra cv2.pollKey() call as it's not needed and might cause issues

# Cleanup
vid.release()

if use_syphon:
    cleanup_syphon(syphon_server)

cv2.destroyAllWindows()

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
    person_detected = False

    for result in results:
        if result.masks is None:
            continue

        for mask, box in zip(result.masks.xy, result.boxes):
            if int(box.cls[0]) == person_class_id:
                person_detected = True
                points = np.int32([mask])
                cv2.fillPoly(combined_mask, points, 255)

    # Apply mask and background
    if person_detected:
        # Apply morphological operations to improve mask edges
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

        # Create mask channels for blending
        mask_3channel = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR) / 255.0

        if bg_mode == 3:  # Transparent mode
            # Create output with only green bodies and transparency (RGBA)
            frame_transparent = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            green_bodies = np.zeros_like(frame)
            green_bodies[:, :] = [0, 255, 0]  # BGR green

            # Set the RGB channels using the mask
            frame_transparent[:, :, :3] = green_bodies * mask_3channel

            # Set the alpha channel based on the mask
            frame_transparent[:, :, 3] = combined_mask

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
            # Create a fully transparent frame (all zeros in RGBA)
            frame = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
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
        cv2.putText(frame, "Press Q to quit, H to hide info", (10, 170),  # Updated text
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

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
        # Create a white background for display purposes only
        display_frame = np.ones(
            (frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255

        # Use alpha channel to blend
        alpha = frame[:, :, 3:4] / 255.0
        rgb = frame[:, :, :3]

        # Blend RGB channels with white background
        display_frame = (rgb * alpha + display_frame *
                         (1.0 - alpha)).astype(np.uint8)
        cv2.imshow('Person Background Removal', display_frame)
    else:
        cv2.imshow('Person Background Removal', frame)

    # Exit on 'q' press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        # Toggle the indicator visibility
        show_indicators = not show_indicators
        print(f"Indicators {'hidden' if not show_indicators else 'shown'}")

# Cleanup
vid.release()

if use_syphon:
    cleanup_syphon(syphon_server)

cv2.destroyAllWindows()

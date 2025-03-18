import os
import random
import cv2
import time
import numpy as np
from ultralytics import YOLO
import subprocess
from syphon import SyphonMetalServer

list_model = [
    "yolov8n-seg.pt",
    "yolov8s-seg.pt",
    "yolov8m-seg.pt",
    "yolov8l-seg.pt",
    "yolov8x-seg.pt",
]


# add choice for model selection
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

# define a video capture object


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
camera_device = list(camera_devices)[camera_choice]
print(f"Using camera: {camera_device}")

# Add resolution selection
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

# Verify actual capture resolution (camera might not support requested resolution)
actual_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Actual capture resolution: {actual_width}x{actual_height}")

if model_choice == 0 or model_choice == 1:
    person_class_id = 0
else:
    person_class_id = yolo_classes.index(
        "person")  # Get the class ID for "person"

# Confidence threshold for detection - lower values help detect people at greater distances
# but may increase false positives
conf_threshold = float(
    input("Enter detection confidence threshold (0.1-0.9, recommended 0.3-0.5): "))
# Ensure value is between 0.1 and 0.9
conf_threshold = max(0.1, min(0.9, conf_threshold))

# After model selection and before video capture
bg_mode = int(
    input("Choose background mode (0: Green, 1: Blue, 2: Custom Color, 3: Transparent Body): "))
if bg_mode == 2:
    # Allow custom RGB background color
    r = int(input("Enter red component (0-255): "))
    g = int(input("Enter green component (0-255): "))
    b = int(input("Enter blue component (0-255): "))
    background_color = (b, g, r)  # OpenCV uses BGR
elif bg_mode == 3:
    # Transparent mode - will show transparent bodies with black background
    background_color = None  # No background color
else:
    background_color = (0, 255, 0) if bg_mode == 0 else (255, 0, 0)

# Add helper function for Metal texture creation (from mask.py)


def create_texture_descriptor(width, height):
    """Create a Metal texture descriptor with the correct attributes"""
    descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
        Metal.MTLPixelFormatRGBA8Unorm,
        width,
        height,
        False
    )
    descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
    return descriptor


# Ask user if they want to use Syphon
use_syphon = bool(int(input("Do you want to enable Syphon output? (0/1): ")))

# Initialize Syphon server if requested - using approach from mask.py
syphon_server = None
if use_syphon:
    try:
        import Metal

        # Initialize Metal device - simpler approach from mask.py
        print("Initializing Metal device for Syphon...")
        mtl_device = Metal.MTLCreateSystemDefaultDevice()

        if mtl_device is None:
            print("Failed to create Metal device. Syphon requires Metal support.")
            use_syphon = False
        else:
            # Create Syphon server - using approach from mask.py
            syphon_name = "PersonBackgroundRemoval"
            print(
                f"Creating Syphon server '{syphon_name}' with dimensions {actual_width}x{actual_height}...")

            try:
                from syphon import SyphonMetalServer
                # Create server with just the name, not passing device directly
                syphon_server = SyphonMetalServer(syphon_name)
                print("Syphon server created successfully")
                print(
                    f"IMPORTANT: If using OBS, add a 'Syphon Client' source and select '{syphon_name}'")
            except Exception as e:
                print(f"Failed to create Syphon server: {e}")
                use_syphon = False

    except ImportError as e:
        print(f"Import error: {e}")
        print("Metal module not available. Syphon requires Metal support.")
        use_syphon = False
    except Exception as e:
        print(f"Error creating Syphon server: {e}")
        use_syphon = False

# Kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

start_time = time.time()
frame_id = 0
show_indicators = True  # New toggle variable for text indicators visibility

while (True):
    frame_id += 1

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    if not ret:
        break

    # Run model inference with user-defined confidence threshold
    results = model.predict(frame, stream=True, conf=conf_threshold)

    # Initialize combined mask for all people
    combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    # Flag to track if any person was detected
    person_detected = False

    # Filter masks for "person" class and combine them into a single mask
    for result in results:
        if result.masks is None:
            continue  # Skip this result if no masks were found

        for mask, box in zip(result.masks.xy, result.boxes):
            if int(box.cls[0]) == person_class_id:  # If the class is "person"
                person_detected = True
                points = np.int32([mask])

                # Add this person to the combined mask
                cv2.fillPoly(combined_mask, points, 255)

    # If at least one person was detected, process the combined mask
    if person_detected:
        # Improve mask edges using morphological operations
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

        # Create mask channels for blending
        mask_3channel = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR) / 255.0

        if bg_mode == 3:  # Transparent mode
            # Create output with black background and transparent bodies
            frame_transparent = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)

            # Set the alpha channel to be transparent where the mask is
            # (0 for body areas, 255 for background)
            frame_transparent[:, :, 3] = 255 - combined_mask

            # Use this as our frame for display
            frame = frame_transparent
        else:
            # Normal solid background mode
            colored_bg = np.full_like(frame, background_color)
            frame = frame * mask_3channel + colored_bg * (1 - mask_3channel)
            frame = frame.astype(np.uint8)
    else:
        # No person detected
        if bg_mode == 3:
            # Create a black frame with full alpha (opaque black)
            frame = np.zeros(
                (frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            frame[:, :, 3] = 255  # Full alpha for background
        else:
            # Use the default background color for the entire frame
            frame = np.full_like(frame, background_color)

    # Calculate and display FPS
    elapsed_time = time.time() - start_time
    fps = frame_id / elapsed_time

    # Flip the frame horizontally for a more natural view
    frame = cv2.flip(frame, 1)

    # Display status information on the frame (only if not transparent)
    if bg_mode != 3 and show_indicators:  # Modified to respect the show_indicators flag
        cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, f"Confidence: {conf_threshold}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "Person detected: {}".format("Yes" if person_detected else "No"),
                    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.putText(frame, "Press Q to quit, H to hide info", (10, 170),  # Updated text
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # Send the frame to Syphon if enabled - using approach from mask.py
    if use_syphon and syphon_server and hasattr(syphon_server, 'has_clients'):
        try:
            # Only publish if there are clients to save resources
            if syphon_server.has_clients:
                # Handle transparency appropriately
                if bg_mode == 3:
                    # For transparent mode, frame is already RGBA
                    rgba = frame
                else:
                    # Convert BGR to RGBA for standard modes
                    rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

                # Flip vertically for Syphon
                rgba = cv2.flip(rgba, 0)
                h, w = rgba.shape[:2]

                # Debug print every 100 frames
                if frame_id % 100 == 0:
                    print(f"Publishing frame with shape: {rgba.shape}")

                # Create Metal texture using the helper function
                descriptor = create_texture_descriptor(w, h)
                texture = mtl_device.newTextureWithDescriptor_(descriptor)

                # Copy frame data to texture
                rgba = np.ascontiguousarray(rgba)
                region = Metal.MTLRegionMake2D(0, 0, w, h)
                texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                    region, 0, rgba.tobytes(), w * 4)

                # Publish texture to Syphon
                syphon_server.publish_frame_texture(texture)
        except Exception as e:
            print(f"Syphon publishing error: {e}")
            use_syphon = False
    elif use_syphon and syphon_server:
        try:
            # Fallback to standard publish method if has_clients is not available
            if bg_mode == 3:
                # For transparent mode - need to handle RGBA
                syphon_server.publish(cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA))
            else:
                # Standard RGB conversion
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                syphon_server.publish(rgb_frame)
        except Exception as e:
            print(f"Standard Syphon publishing failed: {e}")
            use_syphon = False

    # Handle displaying transparent images properly
    if bg_mode == 3:
        # Create a black background for display purposes
        display_frame = np.zeros(
            (frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

        # Use alpha channel to determine where to show the background
        alpha = frame[:, :, 3:4] / 255.0

        # Since we want bodies to be transparent (no RGB content),
        # just use the alpha to determine what to show
        display_frame = (display_frame * alpha).astype(np.uint8)
        cv2.imshow('frame', display_frame)
    else:
        cv2.imshow('frame', frame)

    # Modify the key press detection to handle 'h' as well as 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('h'):
        # Toggle the indicator visibility
        show_indicators = not show_indicators
        print(f"Indicators {'hidden' if not show_indicators else 'shown'}")

# After the loop release the cap object
vid.release()

# Clean up Syphon resources if they were used
if use_syphon and syphon_server:
    try:
        syphon_server.stop()
        print("Syphon server stopped and resources cleaned up")
    except Exception as e:
        print(f"Error stopping Syphon server: {e}")

# Destroy all the windows
cv2.destroyAllWindows()

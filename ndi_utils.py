"""
Utility functions for NDI (Network Device Interface) implementation
"""
import cv2
import numpy as np
import time


def create_ndi_sender(sender_name, width=1920, height=1080, fps=30):
    """
    Create an NDI sender for publishing frames
    
    Args:
        sender_name (str): Name of the NDI sender
        width (int): Frame width
        height (int): Frame height
        fps (int): Frames per second
    
    Returns:
        NDI sender object or None if failed
    """
    try:
        import NDIlib as ndi
        
        # Initialize NDI
        if not ndi.initialize():
            print("Failed to initialize NDI SDK")
            return None
            
        print("NDI SDK initialized successfully")
        
        # Create NDI send settings
        send_settings = ndi.SendCreate()
        send_settings.ndi_name = sender_name.encode('utf-8')  # NDI expects bytes
        send_settings.clock_video = True
        send_settings.clock_audio = False
        
        # Create sender with settings
        sender = ndi.send_create(send_settings)
        
        if sender:
            print(f"NDI sender '{sender_name}' created successfully")
            print(f"Resolution: {width}x{height} @ {fps}fps")
            return sender
        else:
            print("Failed to create NDI sender")
            return None
            
    except ImportError as e:
        print(f"NDI import error: {e}")
        print("Please install the ndi-python package: pip install ndi-python")
        print("Also ensure NDI Tools are installed from https://ndi.video/tools/")
        return None
    except Exception as e:
        print(f"Error creating NDI sender: {e}")
        return None


def publish_frame_to_ndi(frame, sender, is_rgba=False, fps=30):
    """
    Publish a frame to NDI
    
    Args:
        frame (numpy.ndarray): The frame to publish
        sender: The NDI sender object
        is_rgba (bool): Whether the frame is already in RGBA format
        fps (int): Target frame rate for timing
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import NDIlib as ndi
        
        if sender is None:
            return False
        
        # Convert frame format if needed
        if is_rgba:
            # Frame is already RGBA
            if frame.shape[2] == 4:
                # Convert RGBA to BGRA for NDI
                bgra = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
            else:
                # Add alpha channel if missing
                bgra = cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA)
        else:
            # Convert BGR to BGRA for NDI
            bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        
        # Ensure frame is contiguous in memory
        bgra = np.ascontiguousarray(bgra)
        
        h, w = bgra.shape[:2]
        
        # Create NDI video frame
        video_frame = ndi.VideoFrameV2()
        video_frame.data = bgra
        video_frame.line_stride_in_bytes = w * 4
        video_frame.xres = w
        video_frame.yres = h
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRA
        video_frame.frame_rate_N = fps
        video_frame.frame_rate_D = 1
        video_frame.picture_aspect_ratio = float(w) / float(h)
        video_frame.frame_format_type = ndi.FRAME_FORMAT_TYPE_PROGRESSIVE
        video_frame.timecode = ndi.SEND_TIMECODE_SYNTHESIZE
        video_frame.timestamp = 0  # Let NDI handle timing
        
        # Send the frame
        ndi.send_send_video_v2(sender, video_frame)
        return True
        
    except Exception as e:
        # Only print errors occasionally to avoid spam
        if hasattr(publish_frame_to_ndi, '_error_count'):
            publish_frame_to_ndi._error_count += 1
        else:
            publish_frame_to_ndi._error_count = 1
            
        if publish_frame_to_ndi._error_count <= 3:  # Print first 3 errors
            print(f"NDI publish error: {e}")
        return False


def cleanup_ndi(sender):
    """
    Clean up NDI resources
    
    Args:
        sender: The NDI sender to clean up
    """
    if sender:
        try:
            import NDIlib as ndi
            ndi.send_destroy(sender)
            ndi.destroy()
            print("NDI sender cleaned up successfully")
        except Exception as e:
            print(f"Error cleaning up NDI sender: {e}")


def list_ndi_sources():
    """
    List available NDI sources on the network
    
    Returns:
        list: List of NDI source names
    """
    try:
        import NDIlib as ndi
        
        if not ndi.initialize():
            print("Failed to initialize NDI SDK")
            return []
        
        # Create finder
        find_settings = ndi.FindCreate()
        finder = ndi.find_create_v2(find_settings)
        if finder is None:
            print("Failed to create NDI finder")
            return []
        
        # Wait for sources to be discovered
        time.sleep(1)
        
        # Get sources
        sources = ndi.find_get_current_sources(finder)
        source_names = []
        for source in sources:
            if hasattr(source, 'ndi_name'):
                name = source.ndi_name
                if isinstance(name, bytes):
                    source_names.append(name.decode('utf-8'))
                else:
                    source_names.append(str(name))
            else:
                source_names.append('<unknown>')
        
        # Clean up
        ndi.find_destroy(finder)
        ndi.destroy()
        
        return source_names
        
    except ImportError:
        print("NDI not available - ndi-python package not installed")
        return []
    except Exception as e:
        print(f"Error listing NDI sources: {e}")
        return []

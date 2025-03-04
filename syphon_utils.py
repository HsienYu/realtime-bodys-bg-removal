"""
Utility functions for Syphon implementation
"""
import cv2
import numpy as np


def create_syphon_server(server_name):
    """
    Create a Syphon server for publishing frames

    Args:
        server_name (str): Name of the Syphon server

    Returns:
        tuple: (syphon_server, mtl_device) or (None, None) if failed
    """
    try:
        import Metal
        from syphon import SyphonMetalServer

        # Initialize Metal device
        mtl_device = Metal.MTLCreateSystemDefaultDevice()
        if mtl_device is None:
            print("Failed to create Metal device. Syphon requires Metal support.")
            return None, None

        # Create Syphon server
        syphon_server = SyphonMetalServer(server_name)
        return syphon_server, mtl_device

    except ImportError as e:
        print(f"Import error: {e}")
        print("Metal or Syphon modules not available.")
        return None, None

    except Exception as e:
        print(f"Error creating Syphon server: {e}")
        return None, None


def create_texture_descriptor(width, height):
    """
    Create a Metal texture descriptor for Syphon

    Args:
        width (int): Texture width
        height (int): Texture height

    Returns:
        MTLTextureDescriptor: The Metal texture descriptor
    """
    import Metal

    descriptor = Metal.MTLTextureDescriptor.texture2DDescriptorWithPixelFormat_width_height_mipmapped_(
        Metal.MTLPixelFormatRGBA8Unorm,
        width,
        height,
        False
    )
    descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
    return descriptor


def publish_frame_to_syphon(frame, syphon_server, mtl_device, is_rgba=False):
    """
    Publish a frame to Syphon using Metal

    Args:
        frame (numpy.ndarray): The frame to publish
        syphon_server: The Syphon server
        mtl_device: The Metal device
        is_rgba (bool): Whether the frame is already in RGBA format

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import Metal

        # Skip if no clients
        if hasattr(syphon_server, 'has_clients') and not syphon_server.has_clients:
            return True

        # Convert frame to RGBA if needed
        if is_rgba:
            rgba = frame  # Frame is already RGBA
        else:
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        # Flip vertically for Syphon
        rgba = cv2.flip(rgba, 0)
        h, w = rgba.shape[:2]

        # Create Metal texture
        descriptor = create_texture_descriptor(w, h)
        texture = mtl_device.newTextureWithDescriptor_(descriptor)

        # Copy frame data to texture
        rgba = np.ascontiguousarray(rgba)
        region = Metal.MTLRegionMake2D(0, 0, w, h)
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
            region, 0, rgba.tobytes(), w * 4)

        # Publish texture
        syphon_server.publish_frame_texture(texture)
        return True

    except Exception as e:
        # Don't print every error to avoid console spam
        return False


def cleanup_syphon(syphon_server):
    """
    Clean up Syphon resources

    Args:
        syphon_server: The Syphon server to clean up
    """
    if syphon_server:
        try:
            syphon_server.stop()
            print("Syphon server stopped")
        except Exception as e:
            print(f"Error stopping Syphon server: {e}")

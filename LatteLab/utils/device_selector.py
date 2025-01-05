# utils/device_selector.py

# External dependencies
import pyopencl as cl

def create_context_and_queue():
    """
    Tries to create a context/queue on a GPU. 
    If no GPU is found, falls back to CPU.
    Returns (context, queue, device_info_string).
    """
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found!")

    # Try to find a GPU in any platform
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            # Found at least one GPU
            device = devices[0]  # pick the first GPU
            context = cl.Context([device])
            queue = cl.CommandQueue(context)
            info_str = f"Using GPU: {device.name}"
            return context, queue, info_str
    
    # No GPU found; pick the first CPU device
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.CPU)
        if devices:
            device = devices[0]
            context = cl.Context([device])
            queue = cl.CommandQueue(context)
            info_str = f"Using CPU: {device.name}"
            return context, queue, info_str

    # If we reach here, we failed to find either GPU or CPU
    raise RuntimeError("No suitable GPU or CPU devices found!")

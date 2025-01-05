# Path: LatteLab/utils.py


# External dependencies
import numpy as np
import pyopencl as cl


def flatten(grid_vector):
    """
    Flatten a 3D field into a 1D array.

    Parameters:
    grid_vector (numpy.ndarray): A 3D numpy array to be flattened.

    Returns:
    numpy.ndarray: A 1D numpy array.
    """
    return grid_vector.reshape(-1)

def unflatten(flat_vector, grid_shape):
    """
    Unflatten a 1D array into a 3D field.

    Parameters:
    flat_vector (numpy.ndarray): The 1D array to be reshaped.
    grid_shape (tuple): The shape of the desired 3D field.

    Returns:
    numpy.ndarray: The reshaped 3D field.
    """
    return flat_vector.reshape(grid_shape)

def xyz(n, grid_shape):
    """Convert a 1D index to 3D indices."""
    return np.unravel_index(n, grid_shape)



def create_context_and_queue():
    """
    Creates an OpenCL context and command queue.
    This function attempts to create an OpenCL context and command queue by first 
    searching for available GPU devices. If no GPU devices are found, it will then 
    search for CPU devices. If neither GPU nor CPU devices are found, it raises a 
    RuntimeError.
    Returns:
        tuple: A tuple containing the following:
            - context (cl.Context): The created OpenCL context.
            - queue (cl.CommandQueue): The created OpenCL command queue.
            - info_str (str): A string containing information about the selected device.
    Raises:
        RuntimeError: If no suitable GPU or CPU devices are found.
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

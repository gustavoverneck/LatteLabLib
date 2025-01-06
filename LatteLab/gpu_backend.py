import pyopencl as cl
import numpy as np

class GPUBackend:
    """
    A class to manage GPU operations using PyOpenCL.
    Provides methods to load kernels, create buffers, and execute OpenCL programs.
    """

    def __init__(self):
        # Initialize OpenCL context and command queue
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        self.program = None

    def load_kernel(self, kernel_path):
        """
        Load and build an OpenCL kernel from a file.

        Parameters:
            kernel_path (str): Path to the OpenCL kernel file.

        Raises:
            FileNotFoundError: If the kernel file is not found.
            cl.RuntimeError: If there is an error during kernel compilation.
        """
        try:
            with open(kernel_path, 'r') as kernel_file:
                kernel_source = kernel_file.read()
            self.program = cl.Program(self.context, kernel_source).build()
        except FileNotFoundError:
            raise FileNotFoundError(f"Kernel file not found at {kernel_path}")
        except cl.RuntimeError as e:
            print("OpenCL Kernel Compilation Error:\n", e.program.build_log)
            raise

    def create_buffer(self, data, flags=cl.mem_flags.READ_WRITE):
        """
        Create an OpenCL buffer.

        Parameters:
            data (numpy.ndarray): The data to initialize the buffer with.
            flags (int): OpenCL memory flags (default: cl.mem_flags.READ_WRITE).

        Returns:
            cl.Buffer: The created OpenCL buffer.
        """
        try:
            buffer = cl.Buffer(self.context, flags | cl.mem_flags.COPY_HOST_PTR, hostbuf=data)
            return buffer
        except cl.MemoryError:
            raise MemoryError("Failed to allocate GPU memory for the buffer.")

    def execute_kernel(self, kernel_name, global_size, local_size, *args):
        """
        Execute a kernel on the GPU.

        Parameters:
            kernel_name (str): The name of the kernel to execute.
            global_size (tuple): Global work size.
            local_size (tuple or None): Local work size.
            *args: Arguments to pass to the kernel.

        Raises:
            AttributeError: If the kernel is not found in the loaded program.
            cl.RuntimeError: If there is an error during kernel execution.
        """
        if not self.program:
            raise AttributeError("No OpenCL program loaded. Call `load_kernel` first.")
        try:
            kernel = getattr(self.program, kernel_name)
            kernel.set_args(*args)
            cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
        except AttributeError:
            raise AttributeError(f"Kernel '{kernel_name}' not found in the loaded program.")
        except cl.RuntimeError as e:
            print("Kernel Execution Error:", str(e))
            raise

    def get_buffer(self, buffer, shape=None, dtype=np.float32):
        """
        Retrieve data from a GPU buffer.
        
        Parameters:
            buffer (cl.Buffer): The OpenCL buffer to read from.
            shape (tuple or None): The shape of the returned numpy array.
            dtype (numpy.dtype): The data type of the returned array.
        
        Returns:
            numpy.ndarray: The data from the buffer.
        """
        if not isinstance(buffer, cl.Buffer):
            raise TypeError(f"Expected cl.Buffer, got {type(buffer)}")
        
        result = np.empty(shape, dtype=dtype) if shape else np.empty(0, dtype=dtype)
        cl.enqueue_copy(self.queue, result, buffer).wait()
        return result

    def release_resources(self):
        """
        Release OpenCL resources, including buffers and command queue.
        """
        self.queue.finish()
        self.queue = None
        self.context = None
        self.program = None

# Example usage:
# backend = GPUBackend()
# backend.load_kernel("path/to/kernel.cl")
# buffer = backend.create_buffer(np.array([1, 2, 3], dtype=np.float32))
# backend.execute_kernel("kernel_name", (global_size,), (local_size,), buffer)
# result = backend.get_buffer(buffer, shape=(3,), dtype=np.float32)

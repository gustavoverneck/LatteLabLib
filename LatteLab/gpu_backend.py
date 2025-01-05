# lbm/gpu_backend.py

# External dependencies
import pyopencl as cl

# Internal dependencies
from LatteLab.utils import create_context_and_queue

class GPUBackend:
    def __init__(self):
        # Attempt to create context/queue
        self.context, self.queue, self.device_info = create_context_and_queue()
        print(self.device_info)  # e.g., "Using GPU: GeForce GTX 1080" or "Using CPU: Intel(R) Core(TM)..."

    def load_kernel(self, kernel_file):
        """Compile OpenCL kernel."""
        with open(kernel_file, 'r') as f:
            kernel_code = f.read()
        self.program = cl.Program(self.context, kernel_code).build()

    def create_buffer(self, data, flags=cl.mem_flags.READ_WRITE):
        """Create an OpenCL buffer."""
        return cl.Buffer(
            self.context,
            flags | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=data
        )

    def execute_kernel(self, kernel_name, global_size, local_size, *args):
        """Run the kernel."""
        kernel = getattr(self.program, kernel_name)
        kernel(self.queue, global_size, local_size, *args)
        # Optionally: self.queue.finish() for blocking completion

import pyopencl as cl
import numpy as np

# Context setup
context = cl.create_some_context()
queue = cl.CommandQueue(context)

# Kernel code
kernel_code = """
__kernel void test(__global float* data) {
    int i = get_global_id(0);
    data[i] = i;
}
"""
program = cl.Program(context, kernel_code).build()

# Test buffer
data = np.zeros(10, dtype=np.float32)
buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, data.nbytes)

# Execute kernel
program.test(queue, (10,), None, buffer)

# Read buffer
cl.enqueue_copy(queue, data, buffer)
print("Output:", data)


try:
    program = cl.Program(context, kernel_code).build()
except cl.RuntimeError as e:
    print(e.program.build_log)

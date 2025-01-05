# path: LatteLab/solvers/d2q9.py

# Internal dependencies
from LatteLab.base import LBM
from LatteLab.gpu_backend import GPUBackend

# External dependencies
import numpy as np

class D2Q9_GPU_Solver(LBM):
    def __init__(self, grid_shape, relaxation_time, velocity=None, data_type=np.float32):
        super().__init__(grid_shape, relaxation_time)
        self.backend = GPUBackend()
        self.backend.load_kernel("LatteLab/kernels/d2q9_kernel.cl")

    def streaming(self):
        """Streaming step executed on the GPU."""
        grid_size = np.prod(self.grid_shape)
        global_size = (grid_size,)
        local_size = None
        self.backend.execute_kernel(
            "streaming", global_size, local_size,
            self.backend.create_buffer(self.distributions),
            np.int32(self.grid_shape[0]),
            np.int32(self.grid_shape[1])
        )
    
    def collision(self):
        """Collision step executed on the GPU."""
        grid_size = np.prod(self.grid_shape)
        global_size = (grid_size,)
        local_size = None
        self.backend.execute_kernel(
            "collision", global_size, local_size,
            self.backend.create_buffer(self.distributions),
            self.backend.create_buffer(self.velocity),
            self.backend.create_buffer(self.density),
            np.float32(self.relaxation_time)
        )
    
    def export(self, positions=True, density=True, velocity=True):
        """Export the density field from the GPU to the CPU."""
        self.distributions = self.backend.get_buffer(self.distributions)
        self.velocity = self.backend.get_buffer(self.velocity)
        self.density = self.backend.get_buffer(self.density)



class D2Q9_Solver(LBM):
    def __init__(self, grid_shape, relaxation_time, velocity=None, simulation_type='FLUID', data_type=np.float32):
        super().__init__(grid_shape, relaxation_time)

    def streaming(self):
        """Streaming step executed on the CPU."""
        pass
    
    def collision(self):
        """Collision step executed on the CPU."""
        pass

    def export(self, positions=True, density=True, velocity=True):
        """Export the density field from the CPU to the GPU."""
        pass
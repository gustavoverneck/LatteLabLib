# path: LatteLab/solvers/d2q9.py

# Internal dependencies
from LatteLab.base import LBM
from LatteLab.gpu_backend import GPUBackend

# External dependencies
import numpy as np
from copy import deepcopy

class D2Q9_GPU_Solver(LBM):
    def __init__(self, grid_shape, viscosity, velocities_model="D2Q9", data_type=np.float32):
        super().__init__(grid_shape, viscosity, velocities_model=velocities_model, data_type=data_type)
        self.backend = GPUBackend()
        self.backend.load_kernel("LatteLab/kernels/d2q9_kernel.cl")

        self.modelMessage()

        # Initialize buffers with correct dimensions
        self.f_buffer = self.backend.create_buffer(self.f.flatten())
        self.f_temp_buffer = self.backend.create_buffer(self.f.flatten().deepcopy())
        self.u_buffer = self.backend.create_buffer(self.u.flatten())
        self.rho_buffer = self.backend.create_buffer(self.rho.flatten())
    
    def collision(self):
        """Execute collision step on the GPU."""
        grid_size = np.prod(self.grid_shape)
        global_size = (grid_size,)
        local_size = None
        self.backend.execute_kernel(
            "collision", global_size, local_size,
            self.f_buffer,  # Pass GPU buffers
            self.f_temp_buffer,
            self.u_buffer,
            self.rho_buffer,
            np.float32(self.tau),
            np.int32(self.Nx),
            np.int32(self.Ny)
        )

    def streaming(self):
        """Execute streaming step on the GPU."""
        grid_size = np.prod(self.grid_shape)
        global_size = (grid_size,)
        local_size = None
        self.backend.execute_kernel(
            "streaming", global_size, local_size,
            self.f_buffer,  # Pass GPU buffers
            self.f_temp_buffer,
            self.u_buffer,
            self.rho_buffer,
            np.float32(self.tau),
            np.int32(self.Nx),
            np.int32(self.Ny)
        )
    
    def stream_collide(self):
        self.collision()
        self.streaming()

    def export(self, density=True, velocity=True):
        """Export fields from GPU to CPU."""
        if density:
            self.rho = self.backend.get_buffer(self.rho_buffer)
        if velocity:
            self.u = self.backend.get_buffer(self.u_buffer).reshape(self.grid_shape + (2,))

    def modelMessage(self):
        message = "\nVelocity set: D2Q9\n"
        message += "Grid shape: {}\n".format(self.grid_shape)
        message += "Viscosity: {}\n".format(self.nu)
        message += "Relaxation time: {}\n".format(self.tau)
        print(message)
        pass

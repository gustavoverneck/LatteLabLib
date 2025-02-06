# lbm.py

# External imports
import jax
import jax.numpy as jnp
from jax import jit

import numpy as np

import vispy.scene
import vispy.app
from vispy.scene import visuals

from time import perf_counter

# Internal imports
from .graphics import Graphics
from .config import available_color_schemes

# -----------------------------------

dx = 1.0        # Lattice unit
dt = 1.0        # Time step unit

class LBM:
    def __init__(self, config):
        self.initialize(config)

    def initialize(self, config):
        # Set configuration
        self.config = config
        self.initial_time = perf_counter()
        self.velocities_set = config['velocities_set']
        self.simtype = config['simtype']
        self.use_temperature = config['use_temperature']
        self.use_graphics = config['use_graphics']
        self.grid_size = config['grid_size']
        self.viscosity = config['viscosity']
        self.total_timesteps = config['total_timesteps']
        self.colorscheme = config['cmap']
        self.windowDimensions = config['window_dimensions']

        self.D = int(self.velocities_set[1])    # Number of dimensions
        self.Q = int(self.velocities_set[3])    # Number of velocities

        self.omega = 1.0 / self.viscosity   # Relaxation Frequency

        # Set grid size and number of nodes
        self.Nx, self.Ny = self.grid_size[:2]
        self.Nz = self.grid_size[2] if self.D == 3 else 1
        self.N = self.Nx * self.Ny * self.Nz

        self.timestep = 0
        self.timer_interval = 0.0   # Timer interval for automatic updates

        # Define velocity sets
        velocity_data = {
            'D2Q9': (
                [[0,0], [1,0], [-1,0], [0,1], [0,-1], [1,1], [-1,-1], [1,-1], [-1,1]],
                [4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.],
                [0, 3, 4, 1, 2, 7, 8, 5, 6]
            ),
            'D3Q7': (
                [[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]],
                [1./4.] + [1./8.]*6,
                [0, 4, 5, 6, 1, 2, 3]
            ),
            'D3Q15': (
                [[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                 [1,1,1], [-1,-1,-1], [1,1,-1], [-1,-1,1], [1,-1,1], [-1,1,-1],
                 [-1,1,1], [1,-1,-1]],
                [[2./9.] + [1./9.]*6 + [1./72.]*8],
                [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
            ),
            'D3Q13': (
                [[0,0,0], [1,1,0], [-1,-1,0], [1,0,1], [-1,0,-1], [-1,1,1], [-1,-1,-1], [1,1,-1], [1,0,-1], [-1,0,1], [0,1,-1], [0,-1,1]],
                [[1./2.], [1./24.]*12],
                [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11]
            ),
            'D3Q19': (
                [[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                 [1,1,0], [-1,-1,0], [1,0,1], [-1,0,-1], [0,1,1], [0,-1,-1],
                 [1,-1,0], [-1,1,0], [1,0,-1], [-1,0,1], [0,1,-1], [0,-1,1]],
                [[1./3.] + [1./18.]*6 + [1./36.]*12],
                [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
            ),
            'D3Q27': (
                [[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                 [1,1,0], [-1,-1,0], [1,0,1], [-1,0,-1], [0,1,1], [0,-1,-1],
                 [1,-1,0], [-1,1,0], [1,0,-1], [-1,0,1], [0,1,-1], [0,-1,1],
                 [1,1,1], [-1,-1,-1], [1,1,-1], [-1,-1,1], [1,-1,1], [-1,1,-1],
                 [-1,1,1], [1,-1,-1]],
                [[8./27.] + [2./27.]*6 + [1./54.]*12 + [1./216.]*8],
                [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25]
            )
        }
        
        if self.velocities_set in velocity_data:
            c_vals, w_vals, opposite_vals = velocity_data[self.velocities_set]
            self.c = jnp.array(c_vals, dtype=jnp.int32)
            self.w = jnp.array(w_vals, dtype=jnp.float32)
            self.opposite = jnp.array(opposite_vals, dtype=jnp.int32)
        else:
            raise ValueError(f"Unknown velocity set: {self.velocities_set}")

        # Essential arrays
        self.f = jnp.ones((self.Nx, self.Ny, self.Nz, self.Q), dtype=jnp.float32)
        self.rho = jnp.ones((self.Nx, self.Ny, self.Nz), dtype=jnp.float32)
        self.u = jnp.zeros((self.Nx, self.Ny, self.Nz, self.D), dtype=jnp.float32)
        self.flags = jnp.zeros((self.Nx, self.Ny, self.Nz), dtype=jnp.int32)
        
        if self.use_temperature:
            self.T = jnp.ones((self.Nx, self.Ny, self.Nz), dtype=jnp.float32)
   
    def setInitialConditions(self, func, target=None):
        if target is None:
            raise ValueError("Target array must be provided")

        if target == 'rho':
            self.rho = jax.vmap(lambda x, y, z: func(x, y, z))(jnp.arange(self.Nx)[:, None, None], jnp.arange(self.Ny)[None, :, None], jnp.arange(self.Nz)[None, None, :])
        elif target == 'u':
            self.u = jax.vmap(lambda x, y, z: func(x, y, z))(jnp.arange(self.Nx)[:, None, None], jnp.arange(self.Ny)[None, :, None], jnp.arange(self.Nz)[None, None, :])
        elif target == 'T' and self.use_temperature:
            self.T = jax.vmap(lambda x, y, z: func(x, y, z))(jnp.arange(self.Nx)[:, None, None], jnp.arange(self.Ny)[None, :, None], jnp.arange(self.Nz)[None, None, :])
        elif target == 'flags':
            self.flags = jax.vmap(lambda x, y, z: func(x, y, z))(jnp.arange(self.Nx)[:, None, None], jnp.arange(self.Ny)[None, :, None], jnp.arange(self.Nz)[None, None, :])
        else:
            raise ValueError("Unknown target array")

    def getResults(self):
        """Return LBM simulation results"""
        return self.rho, self.u

    def compute_feq(self):
        """Computes the equilibrium distribution function f_eq."""
        cu = jnp.einsum("dq, xyzd -> xyzq", self.c.T, self.u)  # c . u
        usqr = jnp.einsum("xyzd, xyzd -> xyz", self.u, self.u)  # u . u
        feq = self.w[None, None, None, :] * self.rho[..., None]
        feq *= 1 + 3 * cu + (9 / 2) * cu**2 - (3 / 2) * usqr[..., None]
        return feq

    def bounce_back(self, q, f_new):
        opposite_q = self.opposite[q]
        f_new = f_new.at[self.flags == 1, q].set(self.f[self.flags == 1, opposite_q])   # Bounce-back if solid
        return f_new

    def collide_and_stream(self):
        """Performs LBM collision and streaming step."""
        feq = self.compute_feq()
        self.f = self.f + self.omega * (feq - self.f)  # Collision

        # Streaming step
        for q in range(self.Q):
            self.f = self.f.at[..., q].set(jnp.roll(self.f[..., q], shift=self.c[q], axis=(0, 1, 2)))

        # **Apply Periodic Boundary if No Solids in Borders**
        if not jnp.any(self.flags[0, :, :]):  # Left boundary
            self.f = self.f.at[0, :, :].set(self.f[-2, :, :])  # Copy from right
        if not jnp.any(self.flags[-1, :, :]):  # Right boundary
            self.f = self.f.at[-1, :, :].set(self.f[1, :, :])  # Copy from left
        if not jnp.any(self.flags[:, 0, :]):  # Bottom boundary
            self.f = self.f.at[:, 0, :].set(self.f[:, -2, :])   # Copy from top
        if not jnp.any(self.flags[:, -1, :]):  # Top boundary
            self.f = self.f.at[:, -1, :].set(self.f[:, 1, :])   # Copy from bottom

        if self.D == 3:
            if not jnp.any(self.flags[:, :, 0]):  # Front boundary
                self.f = self.f.at[:, :, 0].set(self.f[:, :, -2])  # Copy from back
            if not jnp.any(self.flags[:, :, -1]):  # Back boundary
                self.f = self.f.at[:, :, -1].set(self.f[:, :, 1])  # Copy from front

    def runNoGraphics(self):
        for t in range(self.total_timesteps):
            self.collide_and_stream()
        print(self.calculateElapsedTime())
    
    def run(self):
        self.graphics = Graphics(self.config)
        self.graphics.initialize()
        self.graphics.setupCamera()
        self.graphics.setInitialFlags()
        self.graphics.startTimer(func=self.update_simulation)

    def update_simulation(self, event):
        self.timestep += 1
        if self.timestep <= self.total_timesteps and self.graphics.running:
            # Connect the key press event to the canvas
            self.graphics.canvas.events.key_press.connect(self.graphics.on_key_press)
            
            # on_resize event
            self.graphics.canvas.events.resize.connect(self.graphics.on_resize)

            # Update timestamp
            self.graphics.timestep += 1   
            if self.graphics.show_timestep:
                self.graphics.updateTimestep(self.timestep)

            # Display help menu if requested
            if self.graphics.show_help_menu:
                self.graphics.display_help_menu()

            # Run LBM simulation
            self.graphics.updateData(rho=self.rho, u=self.u, T=self.T)  
            self.graphics.canvas.update()
        else:
            self.graphics.timer.stop()
            self.graphics.running = False
            print(f"Elapsed time: {self.elapsed_time()}")

    def elapsed_time(self):
        self.final_time = perf_counter()
        elapsed = self.final_time - self.initial_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
# lbm.py

# External imports
import jax
import jax.numpy as jnp
from jax import jit

dx = 1.0
dt = 1.0

class LBM:
    def __init__(self, config):
        self.initialize(config)

    def initialize(self, config):
        # Set configuration
        self.config = config
        self.velocities_set = config['velocities_set']
        self.simtype = config['simtype']
        self.use_temperature = config['use_temperature']
        self.grid_size = config['grid_size']
        self.viscosity = config['viscosity']
       
        self.D = int(self.velocities_set[1])
        self.Q = int(self.velocities_set[3])
       
        self.omega = 1.0 / self.viscosity   # Relaxation Frequency

       # Set grid size and number of nodes
        if self.D == 2:
            self.Nx = self.grid_size[0]
            self.Ny = self.grid_size[1]
            self.Nz = 1
            self.N = self.Nx * self.Ny
            if self.grid_size[2] != 0:
                print("Warning: 3D grid size provided for 2D simulation. Setting Nz to 1")
        elif self.D == 3:
            self.Nx = self.grid_size[0]
            self.Ny = self.grid_size[1]
            self.Nz = self.grid_size[2]
            self.N = self.Nx * self.Ny * self.Nz

        # Set velocities, weights and opposite velocities
            # c : velocity directions
            # w : weights
            # opposite : opposite velocities indices
        if self.velocities_set == 'D2Q9':
            self.c = jnp.array([[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]], dtype=jnp.int32)
            self.w = jnp.array([4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.], dtype=jnp.float32)
            self.opposite = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=jnp.int32)
        elif self.velocities_set == 'D3Q15':
            self.c = jnp.array([], dtype=jnp.int32)
            self.w = jnp.array([], dtype=jnp.float32)
            self.opposite = jnp.array([], dtype=jnp.int32)
        elif self.velocities_set == 'D3Q27':
            self.c = jnp.array([], dtype=jnp.int32)
            self.w = jnp.array([], dtype=jnp.float32)
            self.opposite = jnp.array([], dtype=jnp.int32)
        elif self.simtype == 'D3Q19':
            self.c = jnp.array([[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1], [1,1,0], [-1,-1,0], [1,0,1], [-1,0,-1], [0,1,1], [0,-1,-1], [1,-1,0], [-1,1,0], [1,0,-1], [-1,0,1], [0,1,-1], [0,-1,1]], dtype=jnp.int32)
            self.w = jnp.array([1./3., 1./18., 1./18., 1./18., 1./18., 1./18., 1./18., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36., 1./36.,1./36., 1./36.], dtype=jnp.float32)
            self.opposite = jnp.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])

        # Inicialização das variáveis essenciais
        self.f = jnp.ones((self.Nx, self.Ny, self.Nz, self.Q), dtype=jnp.float32)       # Distribution functions
        self.rho = jnp.ones((self.Nx, self.Ny, self.Nz), dtype=jnp.float32)             # Density
        self.u = jnp.zeros((self.Nx, self.Ny, self.Nz, self.D), dtype=jnp.float32)      # Velocity of the node
        self.flags = jnp.zeros((self.Nx, self.Ny, self.Nz), dtype=jnp.int32)            # Flags -> 0: fluid, 1: solid, 2: equilibrium
        if self.use_temperature:
            self.T = jnp.ones((self.Nx, self.Ny, self.Nz), dtype=jnp.float32)               # Temperature
      
    def setInitialConditions(self, rho, u, T=None):
        # Need to add validation for rho, u and T
        self.u = u
        self.rho = rho
        if self.use_temperature:
            self.T = T
    
    def run(self, timesteps):
        for _ in range(timesteps):
            self.collide_and_stream()
        
    def getResults(self):
        """Return LBM simulation results"""
        return self.rho, self.u

    @jit
    def compute_feq(self):
        cu = jnp.einsum("qd,xyzq->xyz", self.c, self.u)         # c_i . u
        usqr = jnp.einsum("xyzq,xyzq->xyz", self.u, self.u)     # u . u
        feq = self.w[None, None, None, :] * self.rho[..., None]  # rho * w_i
        feq *= 1 + 3 * cu + 9/2 * cu**2 - 3/2 * usqr[..., None]
        return feq

    def stream(self, q, f_new):
        """Process streaming: Return the f_new array with the f values shifted in the direction of c[q]."""
        return f_new.at[..., q].set(jnp.roll(f_new[..., q], shift=self.c[q], axis=(0, 1, 2)))

    def bounce_back(self, q, f_new):
        opposite_q = self.opposite[q]
        f_new = f_new.at[self.flags == 1, q].set(self.f[self.flags == 1, opposite_q])   # Bounce-back if solid
        return f_new

    @jit
    def collide_and_stream(self):
        feq = self.compute_feq()
        self.f = self.f + self.omega * (feq - self.f)  # BGK collision

        # Streaming step (shifts f in the direction of c_i)
        self.f = jax.lax.fori_loop(0, self.Q, self.stream, self.f)

        # Bounce-back boundary condition
        self.f = jax.lax.fori_loop(0, self.Q, self.bounce_back, self.f)

        # **Apply Periodic Boundary if No Solids in Borders**
        if not jnp.any(self.flags[0, :, :]):  # Left boundary
            self.f = self.f.at[0, :, :].set(self.f[-2, :, :])  # Copy from right
        if not jnp.any(self.flags[-1, :, :]):  # Right boundary
            self.f = self.f.at[-1, :, :].set(self.f[1, :, :])  # Copy from left
        if not jnp.any(self.flags[:, 0, :]):  # Bottom boundary
            self.f = self.f.at[:, 0, :].set(self.f[:, -2, :])
        if not jnp.any(self.flags[:, -1, :]):  # Top boundary
            self.f = self.f.at[:, -1, :].set(self.f[:, 1, :])

import numpy as np

class LBM:
    def __init__(self, grid_shape, viscosity, model='D2Q9', velocity=None, simulation_type='FLUID', data_type=np.float32):
        self.simulation_type = simulation_type                                              # simulation type (FLUID, PLASMA)
        self.data_type = np.float32                                                         # data type (precision) of the simulation

        self.grid_shape = grid_shape                                                        # (Nx, Ny, *Nz)
        
        if len(grid_shape) == 2:
            self.Nx, self.Ny = grid_shape
            self.Nz = 0
        elif len(grid_shape) == 3:
            self.Nx, self.Ny, self.Nz = grid_shape

        self.N = np.prod(grid_shape)                                                        # total number of grid cells
        self.nu = viscosity                                                                 # kinematic viscosity nu
        self.tau = 1.0 / (3.0 * viscosity + 0.5)                                            # relaxation time tau
        self.v = velocity or np.zeros(grid_shape + (2,), dtype=data_type)           # velocity field
        self.density = np.ones(grid_shape, dtype=data_type)                                # density field
        self.distributions = np.zeros(grid_shape + (9,), dtype=data_type)  # 9 for D2Q9    # distribution functions

    def initialize(self, density=None, velocity=None):
        """Initialize LBM variables."""
        # Initialize the distribution functions
        self.distributions = self.equilibrium(self.density, self.velocity)

    def equilibrium(self, density, velocity):
        """Calculate the equilibrium distribution function."""
        # Example for D2Q9 model
        weights = [4/9] + [1/9]*4 + [1/36]*4
        c = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                      [1, 1], [-1, 1], [-1, -1], [1, -1]])
        csq = 1.0 / 3.0

        f_eq = np.zeros_like(self.distributions)
        for i, wi in enumerate(weights):
            cu = np.dot(velocity, c[i])
            f_eq[..., i] = wi * density * (1 + 3*cu + 4.5*cu**2 - 1.5*np.sum(velocity**2, axis=-1))
        return f_eq

    def collision(self):
        """Perform the collision step."""
        omega = 1.0 / self.tau
        eq = self.equilibrium(self.density, self.velocity)
        self.distributions += omega * (eq - self.distributions)

    def streaming(self):
        """Perform the streaming step."""
        # Implement the streaming step based on lattice connectivity
        pass

    def update_density(self):
        """Update density field."""
        self.density = np.sum(self.distributions, axis=-1) # sum along velocity directions
        pass

    def flatten(self, grid_vector):
        """Flatten a 3D field into a 1D array."""
        return grid_vector.reshape(-1)

    def unflatten(self, flat_vector):
        """Unflatten a 1D array into a 3D field."""
        return flat_vector.reshape(self.grid_shape)
    
    def xyz(self, n):
        """Convert a 1D index to 3D indices."""
        return np.unravel_index(n, self.grid_shape)
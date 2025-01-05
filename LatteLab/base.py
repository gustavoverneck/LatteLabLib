# Path: LatteLab/base.py

# External dependencies
import numpy as np

# Internal dependencies
from LatteLab.utils import flatten, unflatten, xyz


class LBM:
    """
    Lattice Boltzmann Method (LBM) simulation class.
    Attributes:
        data_type (numpy.dtype): Data type (precision) of the simulation.
        grid_shape (tuple): Shape of the simulation grid (Nx, Ny, *Nz).
        Nx (int): Number of grid cells in the x-dimension.
        Ny (int): Number of grid cells in the y-dimension.
        Nz (int): Number of grid cells in the z-dimension (if applicable).
        N (int): Total number of grid cells.
        nu (float): Kinematic viscosity.
        tau (float): Relaxation time.
        omega (float): Relaxation frequency.
        v (numpy.ndarray): Velocity field.
        rho (numpy.ndarray): Density field.
        f (numpy.ndarray): Distribution functions.
        cs (float): Speed of sound.
        csq (float): Speed of sound squared.
        Q (int): Number of discrete velocity directions.
        c (numpy.ndarray): Discrete velocity vectors.
        w (numpy.ndarray): Weights for the discrete velocity directions.
        distributions (numpy.ndarray): Distribution functions for the simulation.
    Methods:
        __init__(velocities_model, grid_shape, viscosity, velocity=None, data_type=np.float32):
            Initialize the LBM simulation with the given parameters.
        initialize(density=None, velocity=None):
            Initialize LBM variables.
        setVelocitiesModel(velocities_model):
            Set the velocities model (D2Q9 or D3Q19).
        equilibrium(density, velocity):
            Calculate the equilibrium distribution function.
        collision():
            Perform the collision step.
        streaming():
            Perform the streaming step.
        update_density():
            Update the density field.
    """

    def __init__(self, velocities_model, grid_shape, viscosity, velocity=None, data_type=np.float32):
        self.data_type = np.float32                                                         # data type (precision) of the simulation
        self.grid_shape = grid_shape                                                        # (Nx, Ny, *Nz)
        
        # Get the number of grid cells per dimension
        if len(grid_shape) == 2:
            self.Nx, self.Ny = grid_shape
            self.Nz = 0
        elif len(grid_shape) == 3:
            self.Nx, self.Ny, self.Nz = grid_shape

        self.setVelocitiesModel(velocities_model)                                           # set the velocities model

        self.N = np.prod(grid_shape)                                                        # total number of grid cells
        self.nu = viscosity                                                                 # kinematic viscosity nu
        self.tau = 1.0 / (3.0 * viscosity + 0.5)                                            # relaxation time tau
        self.omega = 1.0 / self.tau                                                         # relaxation frequency omega
        self.v = velocity or np.array(self.N, dtype=data_type)                              # velocity field
        self.rho = np.ones(self.N, dtype=data_type)                                         # density field
        self.f = np.zeros((self.N, self.Q), dtype=data_type)                                # distribution functions
        
        self.cs = 1.0 / np.sqrt(3.0)                                                        # speed of sound
        self.csq = 1.0 / 3.0                                                                # speed of sound squared

    def initialize(self, density=None, velocity=None):
        """Initialize LBM variables."""
        # Initialize the distribution functions
        self.distributions = self.equilibrium(self.density, self.velocity)
    
    def setVelocitiesModel(self, velocities_model):
        """
        Set the velocities model for the lattice Boltzmann method.

        Parameters:
        velocities_model (str): The type of velocities model to use. 
                                Supported values are 'D2Q9' and 'D3Q19'.

        - 'D2Q9': Sets the model to 2 dimensions with 9 discrete velocities.
        - 'D3Q19': Sets the model to 3 dimensions with 19 discrete velocities.

        Attributes Set:
        - self.Q (int): Number of discrete velocities.
        - self.c (np.ndarray): Array of discrete velocity vectors.
        - self.w (np.ndarray): Array of weights for each discrete velocity.
        """
        if velocities_model == 'D2Q9':
            self.Q = 9
            self.c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
            self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        elif velocities_model == 'D3Q19':
            self.Q = 19
            self.c = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                               [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, 1], [1, 0, -1],
                               [-1, 0, -1], [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]])
            self.w = np.array([1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36,
                               1/36, 1/36, 1/36, 1/36, 1/36])
        pass

    def equilibrium(self, density, velocity):
        """
        Calculate the equilibrium distribution function.
        Parameters:
        density (numpy.ndarray): The density of the fluid at each lattice point.
        velocity (numpy.ndarray): The velocity of the fluid at each lattice point.
        Returns:
        numpy.ndarray: The equilibrium distribution function for each lattice point and direction.
        """
        f_eq = np.zeros_like(self.distributions)
        for i, w_i in enumerate(self.w):
            cu = np.dot(velocity, self.c[i])
            f_eq[..., i] = w_i * density * (1 + 3*cu + 4.5*cu**2 - 1.5*np.sum(velocity**2, axis=-1))
        return f_eq

    def collision(self):
        """
        Perform the collision step in the lattice Boltzmann method.
        This function updates the particle distribution functions by relaxing them
        towards their equilibrium state. The relaxation is controlled by the 
        parameter omega.
        The collision step is a key part of the lattice Boltzmann method, which is 
        used to simulate fluid dynamics.
        """
        self.distributions += self.omega * (self.equilibrium(self.density, self.velocity) - self.distributions)
        pass

    def streaming(self):
        """
        Perform the streaming step in the lattice Boltzmann method.
        This method updates the distribution functions by propagating them
        along the lattice links to neighboring nodes based on the lattice
        connectivity.
        Note:
            This method should be implemented based on the specific lattice
            structure and connectivity used in the simulation.
        """
        
        # Implement the streaming step based on lattice connectivity
        pass

    def update_density(self):
        """
        Update the density attribute by summing the distributions along the velocity directions.
        This method calculates the density by summing the values in the `distributions` array along the last axis 
        (which represents different velocity directions) and assigns the result to the `density` attribute.
        """
        
        self.density = np.sum(self.distributions, axis=-1) # sum along velocity directions
        pass
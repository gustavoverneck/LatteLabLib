# Path: examples/d2q9_channel_flow.py

# This example demonstrates how to simulate a 2D channel flow using the D2Q9 lattice Boltzmann method.

# External dependencies
import os
import sys
import numpy as np

# Internal dependencies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LatteLab.solvers.d2q9 import D2Q9_GPU_Solver

def main():
    # Simulation parameters
    grid_shape = (128, 64)  # Grid size (Nx, Ny)
    relaxation_time = 0.6   # Relaxation time (tau)
    num_steps = 1000        # Number of simulation steps

    # Initialize solver0
    solver = D2Q9_GPU_Solver(grid_shape=grid_shape, relaxation_time=relaxation_time)

    # Initial density (uniform)
    initial_density = np.ones(np.prod(grid_shape), dtype=np.float32)

    # Initial velocity (zero everywhere except inlet)
    initial_velocity = np.zeros((np.prod(grid_shape), 2), dtype=np.float32)
    for i in range(grid_shape[0]):  # Set inlet velocity profile
        initial_velocity[i, 0] = 0.1  # Constant velocity in x-direction

    # Initialize solver with density and velocity
    solver.initialize(density=initial_density, velocity=initial_velocity, flags=None)

    # Simulation loop
    for step in range(num_steps):
        solver.stream_collide()

    print("Simulation completed!")

if __name__ == "__main__":
    main()

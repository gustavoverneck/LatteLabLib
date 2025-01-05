# examples/d2q9_channel_flow.py

import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LatteLab.solvers.d2q9 import D2Q9_GPU_Solver

def main():
    solver = D2Q9_GPU_Solver(grid_shape=(128, 128), relaxation_time=0.6)
    #solver.initialize()
    
    # Time stepping
    for step in range(1000):
        pass
        # solver.collision()
        # solver.streaming()
        # if step % 100 == 0:
        #     print(f"Step {step}: Density sum = {np.sum(solver.density)}")

if __name__ == "__main__":
    main()
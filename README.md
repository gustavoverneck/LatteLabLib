# LatteLab
## A Lattice Boltzmann library with Python + PyOpenCL for GPU acceleration.
```
lattice_boltzmann_lib/
├── lbm/
│   ├── __init__.py              # Initialize library as a Python package
│   ├── base.py                  # Base LBM class and utilities
│   ├── gpu_backend.py           # GPU backend (PyOpenCL integration)
│   ├── solvers/
│   │   ├── __init__.py          # Initialize solvers as a subpackage
│   │   ├── d2q9.py              # D2Q9 solver (2D fluid dynamics example)
│   │   ├── d3q19.py             # D3Q19 solver (3D fluid dynamics example)
│   ├── utils/
│   │   ├── __init__.py          # Initialize utilities as a subpackage
│   │   ├── config.py            # Configuration utilities
│   │   ├── visualization.py     # Visualization utilities (e.g., Matplotlib/Paraview)
│   │   ├── boundary_conditions.py # Common boundary conditions
│   ├── kernels/
│       ├── d2q9_kernel.cl       # OpenCL kernel for D2Q9 model
│       ├── d3q19_kernel.cl      # OpenCL kernel for D3Q19 model
├── examples/
│   ├── d2q9_channel_flow.py     # Example: 2D channel flow
│   ├── d3q19_cavity_flow.py     # Example: 3D cavity flow
├── tests/
│   ├── test_d2q9.py             # Unit tests for D2Q9 solver
│   ├── test_gpu_backend.py      # Unit tests for PyOpenCL backend
├── Makefile                     # For building/testing the project
├── README.md                    # Documentation and project overview
├── setup.py                     # Installation script for the library
```

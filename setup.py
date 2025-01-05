# setup.py
from setuptools import setup, find_packages

setup(
    name="LatteLab",                   # Name of your library
    version="0.0.1",                                # Library version
    description="A Lattice Boltzmann library with Python + PyOpenCL for GPU acceleration.",
    author="Gustavo Arruda Verneck",
    author_email="gustavoverneck@gmail.com",
    url="https://github.com/gustavoverneck/LatteLabLib",  # Repository or project URL
    license="MIT",
    packages=find_packages(),  # Automatically find all Python packages in the project
    install_requires=[
        "numpy",
        "pyopencl",
    ],
    extras_require={
        "dev": [
            "pytest",     # For testing
            "mypy",       # Optional static type checking
        ],
    },
    python_requires=">=3.8",    # Minimum Python version requirement
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # If you have command-line scripts, include them here:
    # entry_points={
    #     "console_scripts": [
    #         "lbm-run=lbm.cli:main",  # Example: a CLI entry point
    #     ],
    # },
)

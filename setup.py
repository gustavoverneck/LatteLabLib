from setuptools import setup, find_packages

setup(
    name="LatteLab",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "jaxlib==0.5.0",
        "jax==0.5.0",
        "numpy==1.26.4",
        "vispy==0.14.3"
    ],
    author="Gustavo A. Verneck",
    author_email="gustavoverneck@gmail.com",
    description="A simple Python package for running LBM based CFD simulations.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gustavoverneck/LatteLabLib",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12.0',
)
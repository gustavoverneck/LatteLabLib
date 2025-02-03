from setuptools import setup, find_packages

setup(
    name="LatteLabLib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
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
    python_requires='>=3.10',
)
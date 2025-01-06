# Path: utils/constants.py


# ---------------------------------------------------------------------------------------------------------------
# Physical Constants
PI = 3.14159265358979323846
SQRT2 = 1.41421356237309504880
SQRT3 = 1.73205080756887729352
SQRT5 = 2.23606797749978969640


# ---------------------------------------------------------------------------------------------------------------
# Flags
FLUID = 0
SOLID = 1
EQ = 2
PERIODIC = 3


# ---------------------------------------------------------------------------------------------------------------
# Utilities Functions

def Re_from_nu(velocity, characteristic_length, kinematic_viscosity):
    """
    Calculate the Reynolds number in Lattice Boltzmann Method (LBM).

    Parameters:
    velocity (float): The flow velocity (in lattice units).
    characteristic_length (float): The characteristic length (e.g., diameter of a pipe) (in lattice units).
    kinematic_viscosity (float): The kinematic viscosity (in lattice units).

    Returns:
    float: The Reynolds number.
    """
    if kinematic_viscosity == 0:
        raise ValueError("Kinematic viscosity cannot be zero.")
    return (velocity * characteristic_length) / kinematic_viscosity


def nu_from_Re(velocity, characteristic_length, Reynolds_number):
    """
    Calculate the kinematic viscosity in Lattice Boltzmann Method (LBM).

    Parameters:
    velocity (float): The flow velocity (in lattice units).
    characteristic_length (float): The characteristic length (e.g., diameter of a pipe) (in lattice units).
    Reynolds_number (float): The Reynolds number.

    Returns:
    float: The kinematic viscosity.
    """
    if Reynolds_number == 0:
        raise ValueError("Reynolds number cannot be zero.")
    return (velocity * characteristic_length) / Reynolds_number



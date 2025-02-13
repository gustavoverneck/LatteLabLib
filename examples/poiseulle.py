from LatteLab import *
import matplotlib.pyplot as plt

def set_poiseuille_initial_conditions(lbm):
    # Set solid walls at top and bottom (y=0 and y=Ny-1)
    lbm.setInitialConditions(
        func=lambda x, y, z: 1 if y == 0 or y == lbm.Ny-1 else 0,
        target="flags"
    )

    # Initialize density to a constant (e.g., 1.0)
    lbm.setInitialConditions(
        func=lambda x, y, z: 1.0,
        target="rho"
    )

    # Initialize velocity with a parabolic profile (2D)
    def velocity_profile(x, y, z):
        Ly = lbm.Ny - 2  # Channel height excluding walls
        y_center = y - 0.5  # Shift to centerline
        u_max = 0.1  # Maximum velocity (adjust as needed)
        ux = u_max * (1 - (y_center / (Ly/2))**2)  # Parabolic profile
        return [ux, 0.0]  # (ux, uy) for 2D

    lbm.setInitialConditions(
        func=velocity_profile,
        target="u"
    )

# Example configuration for 2D Poiseuille flow
config = {
    "grid_size": [128, 64, 1],  # [Nx, Ny, 1] for 2D
    "velocities_set": "D2Q9",
    "viscosity": 0.1,
    "total_timesteps": 10000,
    "use_temperature": False,
}


def main():
    config = Config(velocities_set="D2Q9", total_timesteps=1000, grid_size=(64,64,1), use_graphics=False).get()
    lbm = LBM(config)
    lbm.runNoGraphics()
    rho, u = lbm.getMacroscopicData()
    lbm.exportMacroscopicData()
if __name__ == '__main__':   
    main()
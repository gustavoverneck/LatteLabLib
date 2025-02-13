# lbm.py

# External imports
import numpy as np
import cupy as cp
import vispy.scene
import vispy.app
from vispy.scene import visuals
import tqdm

from time import perf_counter

# Internal imports
from .graphics import Graphics
from .config import velocities_sets
from .kernels import stream_kernel

# -----------------------------------

dx = 1.0        # Lattice unit
dt = 1.0        # Time step unit

class LBM:
    def __init__(self, config):
        # Set configuration
        self.printDevice()  # Print device that's being used
        self.config = config
        self.initial_time = perf_counter()
        self.velocities_set = config['velocities_set']
        self.simtype = config['simtype']
        self.use_temperature = config['use_temperature']
        self.use_graphics = config['use_graphics']
        self.grid_size = config['grid_size']
        self.viscosity = config['viscosity']
        self.total_timesteps = config['total_timesteps']
        self.colorscheme = config['cmap']
        self.windowDimensions = config['window_dimensions']
        self.selectDataType(config["dtype"])    # Define dtype from config file

        self.D = int(self.velocities_set[1])    # Number of dimensions
        self.Q = int(self.velocities_set[3])    # Number of velocities

        self.cs = 1.0/np.sqrt(3)    # Sound speed
        self.cs2 = 1.0/3.0  # Sound speed squared
        self.tau = 3.0 * self.viscosity + 0.5   # Relaxation time
        self.omega = 1.0 / self.tau   # Relaxation Frequency

        # Set grid size and number of nodes
        self.Nx, self.Ny = self.grid_size[:2]
        self.Nz = self.grid_size[2] if self.D == 3 else 1
        self.N = self.Nx * self.Ny * self.Nz

        self.timestep = 0
        self.timer_interval = 0.0   # Timer interval for automatic updates

        # Set velocity set variables c, w and opposite
        if self.velocities_set in velocities_sets:  # config.velocity_sets
            self.c, self.w = velocities_sets[self.velocities_set]["c"], velocities_sets[self.velocities_set]["w"]
            self.c = cp.array(self.c, dtype=cp.int32)   # convert it to a cupy array
            self.w = cp.array(self.w, dtype=cp.float32) # convert it to a cupy array
        else:
            raise ValueError(f"Unknown velocity set: {self.velocities_set}")

        # Essential arrays
        self.f = cp.ones((self.Nx, self.Ny, self.Nz, self.Q), dtype=self.dtype)     # Distribution function: f = (Nx, Ny, Nz, Q) (cp.float32)
        self.rho = cp.ones((self.Nx, self.Ny, self.Nz), dtype=self.dtype)           # Density: rho = (Nx, Ny, Nz) (cp.float32)
        self.u = cp.zeros((self.Nx, self.Ny, self.Nz, self.D), dtype=self.dtype)    # Velocity: u = (Nx, Ny, Nz, D) (cp.float32)
        self.flags = cp.zeros((self.Nx, self.Ny, self.Nz), dtype=cp.int32)           # Flags: flags = (Nx, Ny, Nz) (uint)
        
        if self.use_temperature:
            self.T = cp.ones((self.Nx, self.Ny, self.Nz), dtype=cp.float32)         # Temperature: T = (Nx, Ny, Nz) ()

        # Compute opposite directions for bounce-back
        c_cpu = cp.asnumpy(self.c)  # Convert to NumPy for easy handling
        self.opp = cp.zeros(self.Q, dtype=cp.int32)
        for i in range(self.Q):
            for j in range(self.Q):
                if np.all(c_cpu[i] == -c_cpu[j]):
                    self.opp[i] = j
                    break


    def selectDataType(self, dtype):
        if dtype == "float32":
            self.dtype = cp.float32
        else: 
            raise ValueError("dtype must be in `available_dtypes`")

    def setInitialConditions(self, func, target=None):
        if target is None:
            raise ValueError("Target array must be provided")

        if target == 'rho':
            self.rho = jax.vmap(lambda x, y, z: func(x, y, z))(cp.arange(self.Nx)[:, None, None], cp.arange(self.Ny)[None, :, None], cp.arange(self.Nz)[None, None, :])
        elif target == 'u':
            self.u = jax.vmap(lambda x, y, z: func(x, y, z))(cp.arange(self.Nx)[:, None, None], cp.arange(self.Ny)[None, :, None], cp.arange(self.Nz)[None, None, :])
        elif target == 'T' and self.use_temperature:
            self.T = jax.vmap(lambda x, y, z: func(x, y, z))(cp.arange(self.Nx)[:, None, None], cp.arange(self.Ny)[None, :, None], cp.arange(self.Nz)[None, None, :])
        elif target == 'flags':
            self.flags = jax.vmap(lambda x, y, z: func(x, y, z))(cp.arange(self.Nx)[:, None, None], cp.arange(self.Ny)[None, :, None], cp.arange(self.Nz)[None, None, :])
        else:
            raise ValueError("Unknown target array")

    def collide_and_stream(self):
        self.collide()  # Compute post-collision distributions
        self.stream()   # Move distributions to adjacent cells

    def stream(self):
        # Flatten arrays for kernel input
        f_flat = self.f.reshape(-1)
        flags_flat = self.flags.reshape(-1)
        f_new_flat = cp.empty_like(f_flat)

        # Execute kernel
        stream_kernel(
            f_flat,
            self.c,
            flags_flat,
            self.opp,
            self.Q,
            self.D,
            self.Nx,
            self.Ny,
            self.Nz,
            f_new_flat
        )
        # Reshape output back to original dimensions
        self.f = f_new_flat.reshape(self.f.shape)

    def collide(self):
        # Compute macroscopic density
        self.rho = cp.sum(self.f, axis=-1)  # Shape: (Nx, Ny, Nz)

        # Compute macroscopic velocity
        for d in range(self.D):
            # Remove [..., None] from rho to match shapes
            self.u[..., d] = cp.sum(self.f * self.c[:, d], axis=-1) / self.rho
        
        # Compute equilibrium distribution (feq)
        feq = cp.zeros_like(self.f)
        for i in range(self.Q):
            e_i = self.c[i]  # Velocity vector for direction i
            cu = 3.0 * cp.sum(self.u * e_i, axis=-1)  # Dot product
            usqr = 1.5 * cp.sum(self.u**2, axis=-1)   # Squared velocity magnitude
            feq[..., i] = self.rho * self.w[i] * (1 + cu + 0.5 * cu**2 - usqr)
        
        # Collision step: Update f
        self.f += (feq - self.f) * self.omega

    def runNoGraphics(self):
        for t in tqdm.tqdm(range(self.total_timesteps)):
            self.collide_and_stream()
        print(self.elapsed_time())
    
    def run(self):
        if self.use_graphics:
            print("Running simulation with graphical interface.")
            self.graphics = Graphics(self.config)
            self.graphics.initialize()
            self.graphics.setupCamera()
            self.graphics.setInitialFlags()
            self.graphics.startTimer(func=self.update_simulation)
        else:
            print("Running simulation without graphical interface.")
            self.runNoGraphics() # Run noNoGraphics setup

    def update_simulation(self, event):
        self.timestep += 1
        if self.timestep <= self.total_timesteps and self.graphics.running:
            # Connect the key press event to the canvas
            self.graphics.canvas.events.key_press.connect(self.graphics.on_key_press)
            
            # on_resize event
            self.graphics.canvas.events.resize.connect(self.graphics.on_resize)

            # Update timestamp
            self.graphics.timestep += 1   
            if self.graphics.show_timestep:
                self.graphics.updateTimestep(self.timestep)

            # Display help menu if requested
            if self.graphics.show_help_menu:
                self.graphics.display_help_menu()

            # Run LBM simulation
            self.collide_and_stream()
            self.graphics.updateData(rho=self.rho, u=self.u, T=self.T)  
            self.graphics.canvas.update()
        else:
            self.graphics.timer.stop()
            self.graphics.running = False
            print(f"Elapsed time: {self.elapsed_time()}")

    def elapsed_time(self):
        self.final_time = perf_counter()
        elapsed = self.final_time - self.initial_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    
    def selectKernels(self):
        pass
    
    def getMacroscopicData(self, key=None):
        """Return LBM simulation results"""
        if key == None:
            return cp.asnumpy(self.rho), cp.asnumpy(self.u)
        elif key == 'rho':
            return cp.asnumpy(self.rho)
        elif key == 'u':
            cp.asnumpy(self.u)
        else:
            print("Value error in `getMacroscopicData` `key` argument.")
    
    def exportMacroscopicData(self, filename="output.dat"):
        # Convert GPU arrays to CPU
        rho = cp.asnumpy(self.rho)
        u = cp.asnumpy(self.u)
        T = cp.asnumpy(self.T) if self.use_temperature else None

        with open(filename, "w") as f:
            # Write header based on dimensionality and temperature
            if self.use_temperature:
                header = "x\ty\tz\trho\tux\tuy"
                if self.D == 3:
                    header += "\tuz"
                header += "\tT\n"
                f.write(header)
            else:
                header = "x\ty\tz\trho\tux\tuy"
                if self.D == 3:
                    header += "\tuz"
                header += "\n"
                f.write(header)

            # Write data
            for x in range(self.Nx):
                for y in range(self.Ny):
                    for z in range(self.Nz):
                        line = f"{x}\t{y}\t{z}\t{rho[x, y, z]}\t"
                        line += f"{u[x, y, z, 0]}\t{u[x, y, z, 1]}"
                        if self.D == 3:  # Include uz only for 3D
                            line += f"\t{u[x, y, z, 2]}"
                        if self.use_temperature:
                            line += f"\t{T[x, y, z]}"
                        line += "\n"
                        f.write(line)
    
    def printDevice(self):
        device = cp.cuda.Device()  # Get the current device
        device_id = device.id
        gpu_name = cp.cuda.runtime.getDeviceProperties(device_id)["name"]
        print(f"Using GPU: {gpu_name} (Device ID: {device_id})")
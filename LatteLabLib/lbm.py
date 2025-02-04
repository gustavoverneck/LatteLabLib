# lbm.py

# External imports
import jax
import jax.numpy as jnp
from jax import jit

import numpy as np

import vispy.scene
import vispy.app
from vispy.scene import visuals

from time import perf_counter

# Internal imports
from .config import available_color_schemes

# -----------------------------------

dx = 1.0        # Lattice unit
dt = 1.0        # Time step unit

class LBM:
    def __init__(self, config):
        self.initialize(config)

    def initialize(self, config):
        # Set configuration
        self.initial_time = perf_counter()
        self.config = config
        self.velocities_set = config['velocities_set']
        self.simtype = config['simtype']
        self.use_temperature = config['use_temperature']
        self.use_graphics = config['use_graphics']
        self.grid_size = config['grid_size']
        self.viscosity = config['viscosity']
        self.total_timesteps = config['total_timesteps']
        self.colorscheme = config['cmap']
        self.windowDimensions = config['window_dimensions']

        self.D = int(self.velocities_set[1])    # Number of dimensions
        self.Q = int(self.velocities_set[3])    # Number of velocities

        self.omega = 1.0 / self.viscosity   # Relaxation Frequency

        # Set grid size and number of nodes
        self.Nx, self.Ny = self.grid_size[:2]
        self.Nz = self.grid_size[2] if self.D == 3 else 1
        self.N = self.Nx * self.Ny * self.Nz

        self.timestep = 0
        self.timer_interval = 0.0   # Timer interval for automatic updates

        # Define velocity sets
        velocity_data = {
            'D2Q9': (
                [[0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]],
                [4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.],
                [0, 3, 4, 1, 2, 7, 8, 5, 6]
            ),
            'D3Q15': (
                [[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                 [1,1,1], [-1,-1,-1], [1,-1,-1], [-1,1,1], [-1,1,-1], [1,-1,1],
                 [1,1,-1], [-1,-1,1]],
                [[2./9.] + [1./9.]*6 + [1./72.]*8],
                [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
            ),
            'D3Q19': (
                [[0,0,0], [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1],
                 [1,1,0], [-1,-1,0], [1,0,1], [-1,0,-1], [0,1,1], [0,-1,-1],
                 [1,-1,0], [-1,1,0], [1,0,-1], [-1,0,1], [0,1,-1], [0,-1,1]],
                [[1./3.] + [1./18.]*6 + [1./36.]*12],
                [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]
            )
        }
        
        if self.velocities_set in velocity_data:
            c_vals, w_vals, opposite_vals = velocity_data[self.velocities_set]
            self.c = jnp.array(c_vals, dtype=jnp.int32)
            self.w = jnp.array(w_vals, dtype=jnp.float32)
            self.opposite = jnp.array(opposite_vals, dtype=jnp.int32)
        else:
            raise ValueError(f"Unknown velocity set: {self.velocities_set}")

        # Essential arrays
        self.f = jnp.ones((self.Nx, self.Ny, self.Nz, self.Q), dtype=jnp.float32)
        self.rho = jnp.ones((self.Nx, self.Ny, self.Nz), dtype=jnp.float32)
        self.u = jnp.zeros((self.Nx, self.Ny, self.Nz, self.D), dtype=jnp.float32)
        self.flags = jnp.zeros((self.Nx, self.Ny, self.Nz), dtype=jnp.int32)

        self.data = np.random.rand(self.Nx, self.Ny, self.Nz)
        
        if self.use_temperature:
            self.T = jnp.ones((self.Nx, self.Ny, self.Nz), dtype=jnp.float32)
        
        if self.use_graphics:
            self.initializeGraphics()

    def initializeGraphics(self):
        vispy.use('pyqt5')
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        self.canvas.size = self.windowDimensions
        self.view = self.canvas.central_widget.add_view()
        
        # Camera settings
        if self.D == 2:
            self.view.camera = vispy.scene.cameras.TurntableCamera(
                            center=(self.Nx/2, self.Ny/2, self.Nx/2),  # Center of the volume
                            elevation=0,  # Initial viewing angle (adjustable)
                            azimuth=90,  # Initial rotation (adjustable)
                            distance=1.5 * max(self.grid_size),  # Adjust distance automatically
                            fov=35
                        )

            
        elif self.D == 3:
            self.view.camera = vispy.scene.cameras.TurntableCamera(
                center=(self.Nx/2, self.Ny/2, self.Nz/2),  # Volume center
                elevation=0,  # Viewing angle (adjustable)
                azimuth=90,  # Initial rotation (adjustable)
                distance=1.5 * max(self.grid_size)  # Adjust distance automatically
            )
           
        self.volume = visuals.Volume(self.data, parent=self.view.scene, clim=(0, 1), cmap=self.colorscheme)
        self.viewmode = "density"

        # Set up flags for key press events
        self.show_help_menu = False
        self.show_timestep = False

        self.canvas.title = f"LatteLab"
        # Initialize timer for automatic updates
        self.timer = vispy.app.Timer(interval=self.timer_interval, connect=self.update_simulation, start=True)
        vispy.app.run()


    def setGraphicsColorScheme(self, colorscheme):
        """Change color scheme dynamically"""
        if colorscheme not in available_color_schemes:
            print(f"Color scheme {colorscheme} not available. Defaulting to 'inferno'")
            self.colorscheme = "inferno"
        else:
            self.colorscheme = colorscheme
        self.volume.cmap = self.colorscheme  # Apply new color scheme
    
    def updateGraphicsTimestamp(self):
        # Add text to display timestep in the upper right corner
        for visual in self.canvas.scene.children:
            if isinstance(visual, visuals.Text):
                visual.parent = None  # Remove previous text
        t1 = visuals.Text(f'Timestep: {self.timestep}', parent=self.canvas.scene, color='white')
        t1.font_size = 18
        t1.pos = self.canvas.size[0] - 100, self.canvas.size[1] - 50

    def setGraphicsWindowDimensions(self, dimensions):
        if not isinstance(dimensions, tuple):
            raise ValueError("Window dimensions must be a tuple")
        if len(dimensions) != 2:
            raise ValueError("Window dimensions tuple must have exactly 2 elements")
        
        self.canvas.size = dimensions

    def setInitialConditions(self, rho, u, T=None):
        # Need to add validation for rho, u and T
        self.u = u
        self.rho = rho
        if self.use_temperature:
            self.T = T
    
    def run(self):
        for t in range(self.total_timesteps):
            #self.collide_and_stream()
            #self.test()
            if self.use_graphics:
                self.updateGraphicsTimestamp()
        
        print("Simulation completed.")
            
    def getResults(self):
        """Return LBM simulation results"""
        return self.rho, self.u

    def compute_feq(self):
        """Computes the equilibrium distribution function f_eq."""
        cu = jnp.einsum("dq, xyzd -> xyzq", self.c.T, self.u)  # c . u
        usqr = jnp.einsum("xyzd, xyzd -> xyz", self.u, self.u)  # u . u
        feq = self.w[None, None, None, :] * self.rho[..., None]
        feq *= 1 + 3 * cu + (9 / 2) * cu**2 - (3 / 2) * usqr[..., None]
        return feq

    def bounce_back(self, q, f_new):
        opposite_q = self.opposite[q]
        f_new = f_new.at[self.flags == 1, q].set(self.f[self.flags == 1, opposite_q])   # Bounce-back if solid
        return f_new

    def collide_and_stream(self):
        """Performs LBM collision and streaming step."""
        feq = self.compute_feq()
        self.f = self.f + self.omega * (feq - self.f)  # Collision

        # Streaming step
        for q in range(self.Q):
            self.f = self.f.at[..., q].set(jnp.roll(self.f[..., q], shift=self.c[q], axis=(0, 1, 2)))

        # **Apply Periodic Boundary if No Solids in Borders**
        if not jnp.any(self.flags[0, :, :]):  # Left boundary
            self.f = self.f.at[0, :, :].set(self.f[-2, :, :])  # Copy from right
        if not jnp.any(self.flags[-1, :, :]):  # Right boundary
            self.f = self.f.at[-1, :, :].set(self.f[1, :, :])  # Copy from left
        if not jnp.any(self.flags[:, 0, :]):  # Bottom boundary
            self.f = self.f.at[:, 0, :].set(self.f[:, -2, :])
        if not jnp.any(self.flags[:, -1, :]):  # Top boundary
            self.f = self.f.at[:, -1, :].set(self.f[:, 1, :])

    # Key press event handler
    def on_key_press(self, event):
        if event.key == 'Space':
            if self.timer.running:
                self.timer.stop()
                print("Simulation paused.")
            else:
                self.timer.start()
                print("Simulation resumed.")

        if event.key == 'h':
            self.show_help_menu = not self.show_help_menu
            if self.show_help_menu:
                self.display_help_menu()
            else:
                self.hide_help_menu()

        if event.key == 'q':
            self.timer.stop()
            print("Simulation stopped.")
            self.canvas.close()
        
        if event.key == 'c':
            print("Change color scheme")
            self.current_colorscheme_index = available_color_schemes.index(self.colorscheme)
            self.colorscheme = available_color_schemes[(self.current_colorscheme_index + 1) % len(available_color_schemes)]
            self.setGraphicsColorScheme(self.colorscheme)

        if event.key == 't':
            if self.viewmode == "density":
                self.viewmode = "temperature"
                self.volume.set_data(self.T)
            else:
                self.viewmode = "density"
                self.volume.set_data(self.data)
            print(f"Toggled view mode to {self.viewmode}")
        
        if event.key == 'i':
            self.show_timestep = not self.show_timestep
            if self.show_timestep:
                self.display_time_step()
            else:
                self.hide_time_step()

        self.canvas.update()

    def display_time_step(self):
        # Display timestep text
        timestep_text = f"Timestep: {self.timestep}"
        self.timestep_visual = visuals.Text(timestep_text, parent=self.canvas.scene, color='white')
        self.timestep_visual.font_size = 10
        self.timestep_visual.pos = 180, self.canvas.size[1] - 300

    def hide_time_step(self):
        self.timestep_visual.parent = None

    def display_help_menu(self):
        # Display help text
        help_text = """space: Pause/Resume simulation.\nh: Display help menu.\nq: Quit simulation.\nc: Change color scheme: {colorscheme}\nt: Toggle variable: {viewmode}""".format(colorscheme=self.colorscheme, viewmode=self.viewmode)
        self.help_visual = visuals.Text(help_text, parent=self.canvas.scene, color='white')
        self.help_visual.font_size = 10
        self.help_visual.pos = 180, self.canvas.size[1] - 400
    
    def hide_help_menu(self):
        self.help_visual.parent = None

    def update_simulation(self, event):
        self.timestep += 1

        if self.timestep <=  self.total_timesteps:
            # Connect the key press event to the canvas
            self.canvas.events.key_press.connect(self.on_key_press)
            
            if self.show_timestep:
                self.updateGraphicsTimestamp()
            
            if self.show_help_menu:
                self.display_help_menu()

            self.data = np.random.rand(self.Nx, self.Ny, self.Nz)
            self.volume.set_data(self.data)        
            self.canvas.update()
        else:
            self.timer.stop()
            self.final_time = perf_counter()
            print(f"Simulation completed in {self.elapsed_time()}.")

        
    def elapsed_time(self):
        elapsed = self.final_time - self.initial_time
        hours, rem = divmod(elapsed, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
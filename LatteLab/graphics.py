# graphics.py


# External Imports
import numpy as np
import cupy as cp
import vispy.scene
import vispy.app
from vispy.scene import visuals

# Internal Imports
from .config import available_color_schemes
import os
from vispy import app
from PyQt5 import QtGui


class Graphics:
    def __init__(self, config):
        self.velocities_set = config['velocities_set']
        self.simtype = config['simtype']
        self.use_temperature = config['use_temperature']
        self.use_graphics = config['use_graphics']
        self.grid_size = config['grid_size']
        self.viscosity = config['viscosity']
        self.total_timesteps = config['total_timesteps']
        self.colorscheme = config['cmap']
        self.windowDimensions = config['window_dimensions']
    
    def initialize(self):
        vispy.use('pyqt5')
        self.running = True
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        self.canvas.size = self.windowDimensions    # Window size
        self.canvas.title = f"LatteLab"             # Window title
        

        # Get the path to the static folder
        static_folder_path = os.path.join(os.path.dirname(__file__), 'static')
        logo_path = os.path.join(static_folder_path, 'logo.png')
        # Set window logo and taskbar icon
        app_icon = QtGui.QIcon(logo_path)
        self.canvas.native.setWindowIcon(app_icon)
        
        self.view = self.canvas.central_widget.add_view()

        self.timestep = 0
        self.timer_interval = 0.0   # Timer interval for automatic updates

        # Vlocity set configuration
        self.D = int(self.velocities_set[1])    # Number of dimensions
        self.Q = int(self.velocities_set[3])    # Number of velocities

        self.rho = np.zeros(self.grid_size)
        self.u = np.zeros((*self.grid_size, self.D))
        
        if self.use_temperature:
            self.T = np.zeros(self.grid_size)

        # Set grid size and number of nodes
        self.Nx, self.Ny = self.grid_size[:2]
        self.Nz = self.grid_size[2] if self.D == 3 else 1
        self.N = self.Nx * self.Ny * self.Nz



    def setupCamera(self):
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
           
        self.volume = visuals.Volume(self.rho, parent=self.view.scene, clim=(0, 1), cmap=self.colorscheme)
        self.viewmode = "density"
    
    def setInitialFlags(self):
        # Set up flags for key press events
        self.viewmode = "density"
        self.show_help_menu = False
        self.show_timestep = False

    def startTimer(self, func):
        # Start the timer
        self.timer = vispy.app.Timer(interval=self.timer_interval, connect=func, start=True)
        vispy.app.run()

    def setCmap(self, colorscheme):
        """Change color scheme dynamically"""
        if colorscheme not in available_color_schemes:
            print(f"Color scheme {colorscheme} not available. Defaulting to 'inferno'")
            self.colorscheme = "inferno"
        else:
            self.colorscheme = colorscheme
        self.volume.cmap = self.colorscheme  # Apply new color scheme
    
    def updateGraphicsTimestamp(self):
        # Update or create text to display timestep in the upper right corner
        if not hasattr(self, 'timestep_text'):
            self.timestep_text = visuals.Text('', parent=self.canvas.scene, color='white', font_size=18)
            self.timestep_text.pos = self.canvas.size[0] - 100, self.canvas.size[1] - 50
        self.timestep_text.text = f'Timestep: {self.timestep}'

    def setGraphicsWindowDimensions(self, dimensions):
        if not isinstance(dimensions, tuple):
            raise ValueError("Window dimensions must be a tuple")
        if len(dimensions) != 2:
            raise ValueError("Window dimensions tuple must have exactly 2 elements")
        
        self.canvas.size = dimensions
    
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
            self.toggle_help_menu()

        if event.key == 'q':
            self.timer.stop()
            print("Simulation forced to stopped.")
            self.canvas.close()
        
        if event.key == 'c':
            print("Change color scheme to", self.colorscheme)
            self.current_colorscheme_index = available_color_schemes.index(self.colorscheme)
            self.colorscheme = available_color_schemes[(self.current_colorscheme_index + 1) % len(available_color_schemes)]
            self.setCmap(self.colorscheme)
        
        if self.use_temperature:
            self.available_data = ["density", "velocity", "temperature"]
        else: 
            self.available_data = ["density", "velocity"]
        
        if event.key == 't':
            current_index = self.available_data.index(self.viewmode)
            self.viewmode = self.available_data[(current_index + 1) % len(self.available_data)]
            if self.viewmode == "density":
                self.volume.set_data(self.rho)
            elif self.viewmode == "velocity":
                self.volume.set_data(np.linalg.norm(self.u, axis=-1))
            elif self.viewmode == "temperature" and self.use_temperature:
                self.volume.set_data(self.T)
            print(f"Toggled view mode to {self.viewmode}")
        
        if event.key == 'i':
            self.toggle_time_step()

        self.canvas.update()
    
    def display_help_menu(self):
        help_text = (
            "space: Pause/Resume simulation.\n"
            "h: Display help menu.\n"
            "q: Quit simulation.\n"
            "c: Change color scheme: {colorscheme}\n"
            "t: Toggle variable: {viewmode}"
        ).format(colorscheme=self.colorscheme, viewmode=self.viewmode)
        
        if not hasattr(self, 'help_visual'):
            self.help_visual = visuals.Text(help_text, parent=self.canvas.scene, color='white', font_size=10)
        else:
            self.help_visual.text = help_text
            self.help_visual.parent = self.canvas.scene
        
        self.help_visual.pos = 180, self.canvas.size[1] - 400
        self.show_help_menu = True
    
    def display_time_step(self):
        if not hasattr(self, 'timestep_text'):
            self.timestep_text = visuals.Text('', parent=self.canvas.scene, color='white', font_size=10)
            self.timestep_text.pos = self.canvas.size[0] - 100, self.canvas.size[1] - 20
        else:
            self.timestep_text.parent = self.canvas.scene
        self.timestep_text.text = f'Timestep: {self.timestep}'
        self.show_timestep = True

    def updateTimestep(self, timestep):
        self.timestep = timestep
        if self.show_timestep:
            self.display_time_step()

    def hide_help_menu(self):
        if hasattr(self, 'help_visual'):
            self.help_visual.parent = None
            self.show_help_menu = False

    def hide_time_step(self):
        if hasattr(self, 'timestep_text'):
            self.timestep_text.parent = None
            self.show_timestep = False

    def toggle_help_menu(self):
        if self.show_help_menu:
            self.hide_help_menu()
        else:
            self.display_help_menu()

    def toggle_time_step(self):
        if self.show_timestep:
            self.hide_time_step()
        else:
            self.display_time_step()

    def updateData(self, rho, u, T=None):
        self.rho = rho
        self.u = np
        if T is not None:
            self.T = T
    
    def on_resize(self, event):
        self.canvas.size = event.size
        if hasattr(self, 'timestep_text'):
            self.timestep_text.pos = self.canvas.size[0] - 100, self.canvas.size[1] - 20
        if hasattr(self, 'help_visual'):
            self.help_visual.pos = 180, self.canvas.size[1] - 400
        self.canvas.update()
# visualisation.py

# External imports
import vispy.scene
import vispy.app
from vispy.scene import visuals

available_color_schemes = ["grays", "hot", "cool", "viridis", "inferno", "plasma", "magma", "cividis", "jet", "turbo", "RdYlBu", "blues"]

class Graphics:
    def __init__(self):
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'turntable'
        self.timestep = 0
        self.volume = None
        self.timestep = 0
    
    def setColorScheme(self, colorscheme):
        if colorscheme not in available_color_schemes:
            print(f"Color scheme {colorscheme} not available. Defaulting to 'inferno'")
            self.colorscheme = "inferno"
        else :
            self.colorscheme = colorscheme

    def render(self):
        self.timer = vispy.app.Timer(interval=1.0, connect=self.on_timer)
        self.timer.start()
        vispy.app.run()

    def updateData(self, data):
        self.volume = visuals.Volume(data, parent=self.view.scene, clim=(0, 1), cmap=self.colorscheme)

    def updateTimestamp(self, amount=1):
        self.timestep += amount
        self.canvas.title = f"Timestamp: {self.timestep}"

    def on_timer(self, event):
        import numpy as np
        # Update data here
        new_data = np.random.rand(64, 64, 64)
        self.updateData(new_data)
        self.updateTimestamp()
        self.canvas.update()


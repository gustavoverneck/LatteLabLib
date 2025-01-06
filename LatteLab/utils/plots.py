# Path: utils/plots.py

# External dependencies
from matplotlib import pyplot as plt

def streamLines(x, y, u, v, density, title="Streamlines", cmap='inferno'):
    """
    Plot streamlines of a 2D flow field.
    
    Parameters:
        x (numpy.ndarray): X-coordinates of the grid.
        y (numpy.ndarray): Y-coordinates of the grid.
        u (numpy.ndarray): X-velocity component.
        v (numpy.ndarray): Y-velocity component.
        density (numpy.ndarray): Density field.
        title (str): Plot title (default: "Streamlines").
        cmap (str): plot color map.
    """
    plt.figure(figsize=(8, 6))
    plt.streamplot(x, y, u, v, color=density, linewidth=1, cmap=cmap)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Density')
    plt.legend()
    plt.show()


def quiverPlot(x, y, u, v, title="Quiver Plot", cmap='inferno'):
    """
    Plot a quiver plot of a 2D flow field.
    
    Parameters:
        x (numpy.ndarray): X-coordinates of the grid.
        y (numpy.ndarray): Y-coordinates of the grid.
        u (numpy.ndarray): X-velocity component.
        v (numpy.ndarray): Y-velocity component.
        title (str): Plot title (default: "Quiver Plot").
        cmap (str): plot color map.
    """
    plt.figure(figsize=(8, 6))
    plt.quiver(x, y, u, v, color='b', cmap=cmap)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
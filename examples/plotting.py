import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data file
df = pd.read_csv("output.dat", sep="\t")

# For 2D simulations, select z=0 slice (if 3D, adjust z_slice)
z_slice = 0  
df_slice = df[df["z"] == z_slice]

# Reshape data into grid format
x = df_slice["x"].unique()
y = df_slice["y"].unique()
X, Y = np.meshgrid(x, y)

# Create density and velocity arrays
rho = df_slice["rho"].values.reshape(len(y), len(x))
ux = df_slice["ux"].values.reshape(len(y), len(x))
uy = df_slice["uy"].values.reshape(len(y), len(x))

# Plot density field
plt.figure(figsize=(10, 6))
plt.pcolormesh(X, Y, rho, shading="auto", cmap="viridis")
plt.colorbar(label="Density")

# Plot velocity vectors (downsample for clarity)
skip = 4  # Plot every 4th vector
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
           ux[::skip, ::skip], uy[::skip, ::skip], 
           color="white", scale=20, width=0.003)

plt.title("Density and Velocity Field")
plt.xlabel("X")
plt.ylabel("Y")
plt.gca().set_aspect("equal")
plt.show()
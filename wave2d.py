import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML  # For displaying animations in Jupyter environment
import warnings
from matplotlib import MatplotlibDeprecationWarning  # Import the warning type

# Ignore MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# Ignore QStandardPaths warning (permission-related warning)
warnings.filterwarnings("ignore", message="QStandardPaths: wrong permissions on runtime directory")

# Parameter setup
Lx = 10.0     # Space size in the x direction (0 to Lx)
Ly = 10.0     # Space size in the y direction (0 to Ly)
T = 2.0       # Total simulation time (set to 5.0)
c = 1.0       # Wave speed
Nx = 100      # Number of divisions in the x direction
Ny = 100      # Number of divisions in the y direction
Nt = 30       # Number of time divisions (set to 30)

# Calculating spatial and temporal intervals
dx = Lx / Nx
dy = Ly / Ny
dt = T / Nt
r = (c * dt / dx) ** 2  # CFL condition (Courant–Friedrichs–Lewy condition)

# Check CFL condition
if r > 0.5:
    print(f"Warning: CFL condition is violated. r = {r}, consider reducing dt or increasing Nx/Ny.")

# Create spatial and time arrays
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
t = np.linspace(0, T, Nt)

# Initial condition setup (e.g., wave starts from the center)
u = np.zeros((Nt, Nx, Ny))
u0 = np.sin(np.pi * x[:, None] / Lx) * np.sin(np.pi * y[None, :] / Ly)  # 2D initial wave

u[0, :, :] = u0  # Initial state at t=0

# Initial velocity is 0 (first derivative with respect to time is 0)
u[1, 1:-1, 1:-1] = u0[1:-1, 1:-1] + r * (
    u0[2:, 1:-1] - 2 * u0[1:-1, 1:-1] + u0[:-2, 1:-1] + u0[1:-1, 2:] - 2 * u0[1:-1, 1:-1] + u0[1:-1, :-2])

# Start the time loop
for n in range(1, Nt-1):
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[n+1, i, j] = 2 * (1 - 2 * r) * u[n, i, j] - u[n-1, i, j] + r * (
                u[n, i+1, j] + u[n, i-1, j] + u[n, i, j+1] + u[n, i, j-1])

# 3D animation setup
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Set up the grid and initial surface
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u[0, :, :], cmap='viridis', edgecolor='none')

ax.set_title('3D Wave Equation Solution (Surface)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Amplitude')

# Fix the Amplitude axis (set zlim to fix the range)
ax.set_zlim(-1.0, 1.0)  # Set the range from -1.0 to 1.0 for illustration

# Animation update function
def update(frame):
    ax.cla()  # Clear the previous surface
    ax.set_title(f'3D Wave Equation Solution (Surface) - t={t[frame]:.2f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('Amplitude')
    ax.set_zlim(-1.0, 1.0)  # Fix the Amplitude axis
    surf = ax.plot_surface(X, Y, u[frame, :, :], cmap='viridis', edgecolor='none')
    return [surf]

# Create the animation (repeat=False to prevent looping)
ani = FuncAnimation(fig, update, frames=Nt, interval=50, blit=False, repeat=False)

# Display the animation in Jupyter Notebook
#HTML(ani.to_jshtml())

# Use plt.show() to display the animation in a regular Python environment
plt.show()

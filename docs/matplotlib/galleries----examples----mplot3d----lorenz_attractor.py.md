# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\lorenz_attractor.py`

```py
"""
================
Lorenz attractor
================

This is an example of plotting Edward Lorenz's 1963 `"Deterministic Nonperiodic
Flow"`_ in a 3-dimensional space using mplot3d.

.. _"Deterministic Nonperiodic Flow":
   https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml

.. note::
   Because this is a simple non-linear ODE, it would be more easily done using
   SciPy's ODE solver, but this approach depends only upon NumPy.
"""

import matplotlib.pyplot as plt
import numpy as np


def lorenz(xyz, *, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    # Calculate the derivative dx/dt
    x_dot = s*(y - x)
    # Calculate the derivative dy/dt
    y_dot = r*x - y - x*z
    # Calculate the derivative dz/dt
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


dt = 0.01
num_steps = 10000

# Initialize an array to store xyz coordinates over time
xyzs = np.empty((num_steps + 1, 3))  # Need one more for the initial values
xyzs[0] = (0., 1., 1.05)  # Set initial values

# Step through "time", calculating the next xyz coordinates using Euler's method
for i in range(num_steps):
    # Calculate the next xyz coordinates using the current xyz coordinates and their derivatives
    xyzs[i + 1] = xyzs[i] + lorenz(xyzs[i]) * dt

# Plot the Lorenz attractor in 3D
ax = plt.figure().add_subplot(projection='3d')

# Plot the xyz coordinates over time
ax.plot(*xyzs.T, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()
```
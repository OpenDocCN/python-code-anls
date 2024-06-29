# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\image_zcoord.py`

```py
"""
==================================
Modifying the coordinate formatter
==================================

Modify the coordinate formatter to report the image "z" value of the nearest
pixel given x and y.  This functionality is built in by default; this example
just showcases how to customize the `~.axes.Axes.format_coord` function.
"""

import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

# Generate a 5x3 matrix of random numbers scaled by 10
X = 10*np.random.rand(5, 3)

# Create a figure and axis
fig, ax = plt.subplots()
# Display the matrix as an image
ax.imshow(X)

# Define a function to format the coordinate display
def format_coord(x, y):
    # Round the coordinates to get the nearest integer indices
    col = round(x)
    row = round(y)
    # Get the dimensions of the matrix
    nrows, ncols = X.shape
    # Check if the rounded indices are within bounds
    if 0 <= col < ncols and 0 <= row < nrows:
        # Get the value from matrix X at the rounded indices
        z = X[row, col]
        # Return formatted string with coordinates and matrix value
        return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}'
    else:
        # Return formatted string with coordinates only if out of bounds
        return f'x={x:1.4f}, y={y:1.4f}'

# Assign the custom format_coord function to the axes object
ax.format_coord = format_coord

# Display the plot
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.format_coord`
#    - `matplotlib.axes.Axes.imshow`
```
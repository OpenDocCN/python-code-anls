# `D:\src\scipysrc\sympy\examples\beginner\plot_examples.py`

```
#! /usr/bin/env python
# Check the plot docstring

from sympy import Symbol, exp, sin, cos
from sympy.plotting import (plot, plot_parametric,
                            plot3d_parametric_surface, plot3d_parametric_line,
                            plot3d)

# Create a list of integers from 0 to 4
lx = range(5)
# Create a list comprehension to square each element of lx
ly = [i**2 for i in lx]

# Define symbolic variables
x = Symbol('x')
y = Symbol('y')
u = Symbol('u')
v = Symbol('v')
# Define a symbolic expression
expr = x**2 - 1

# Plot expr from x=2 to x=4, store the plot object in b (cartesian plot)
b = plot(expr, (x, 2, 4), show=False)
# Plot exp(-x) from x=0 to x=4, store the plot object in e (cartesian plot with coloring)
e = plot(exp(-x), (x, 0, 4), show=False)
# Plot a 3D parametric line sin(x), cos(x), x from x=0 to x=10, store in f
f = plot3d_parametric_line(sin(x), cos(x), x, (x, 0, 10), show=False)
# Plot 3D surface sin(x)*cos(y) over ranges, store in g
g = plot3d(sin(x)*cos(y), (x, -5, 5), (y, -10, 10), show=False)
# Plot 3D parametric surface cos(u)*v, sin(u)*v, u from u=0 to u=10, v=-2 to 2, store in h
h = plot3d_parametric_surface(cos(u)*v, sin(u)*v, u, (u, 0, 10), (v, -2, 2), show=False)

# Some aesthetics adjustments
# Set line_color in plot e based on x value normalized to 0-1
e[0].line_color = lambda x: x / 4
# Set line_color in plot f based on x, y, z values normalized to 0-1
f[0].line_color = lambda x, y, z: z / 10
# Set surface_color in plot g based on x, y values
g[0].surface_color = lambda x, y: sin(x)

# More aesthetics adjustments - color based on parameters or coordinates
# Plot 2D parametric lines, adjust line_color based on parameter u
param_line_2d = plot_parametric((x*cos(x), x*sin(x), (x, 0, 15)),
                                (1.1*x*cos(x), 1.1*x*sin(x), (x, 0, 15)), show=False)
param_line_2d[0].line_color = lambda u: sin(u)  # parametric
param_line_2d[1].line_color = lambda u, v: u**2 + v**2  # coordinates
param_line_2d.title = 'The inner one is colored by parameter and the outer one by coordinates'

# Plot 3D parametric lines, adjust line_color based on parameters u, v, w
param_line_3d = plot3d_parametric_line((x*cos(x), x*sin(x), x, (x, 0, 15)),
                                       (1.5*x*cos(x), 1.5*x*sin(x), x, (x, 0, 15)),
                                       (2*x*cos(x), 2*x*sin(x), x, (x, 0, 15)), show=False)
param_line_3d[0].line_color = lambda u: u  # parametric
param_line_3d[1].line_color = lambda u, v: u*v  # first and second coordinates
param_line_3d[2].line_color = lambda u, v, w: u*v*w  # all coordinates

# Show plots if this script is executed directly
if __name__ == '__main__':
    for p in [b, e, f, g, h, param_line_2d, param_line_3d]:
        p.show()
```
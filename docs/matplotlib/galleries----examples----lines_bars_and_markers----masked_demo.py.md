# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\masked_demo.py`

```py
"""
==============================
Plotting masked and NaN values
==============================

Sometimes you need to plot data with missing values.

One possibility is to simply remove undesired data points. The line plotted
through the remaining data will be continuous, and not indicate where the
missing data is located.

If it is useful to have gaps in the line where the data is missing, then the
undesired points can be indicated using a `masked array`_ or by setting their
values to NaN. No marker will be drawn where either x or y are masked and, if
plotting with a line, it will be broken there.

.. _masked array:
   https://numpy.org/doc/stable/reference/maskedarray.generic.html

The following example illustrates the three cases:

1) Removing points.
2) Masking points.
3) Setting to NaN.
"""

import matplotlib.pyplot as plt
import numpy as np

# Generate a linearly spaced array of x values from -π/2 to π/2 with 31 points
x = np.linspace(-np.pi/2, np.pi/2, 31)

# Compute y values as cosine of x cubed
y = np.cos(x)**3

# 1) Remove points where y > 0.7
x2 = x[y <= 0.7]  # Filtered x values where y is less than or equal to 0.7
y2 = y[y <= 0.7]  # Corresponding filtered y values

# 2) Mask points where y > 0.7 using masked array
y3 = np.ma.masked_where(y > 0.7, y)  # Mask y values where y is greater than 0.7

# 3) Set to NaN where y > 0.7 in a copy of y
y4 = y.copy()  # Create a copy of y
y4[y3 > 0.7] = np.nan  # Set y values to NaN where y3 (masked values) is greater than 0.7

# Plotting the data
plt.plot(x*0.1, y, 'o-', color='lightgrey', label='No mask')  # Plot original data without any masking
plt.plot(x2*0.4, y2, 'o-', label='Points removed')  # Plot data with points removed
plt.plot(x*0.7, y3, 'o-', label='Masked values')  # Plot data with masked values
plt.plot(x*1.0, y4, 'o-', label='NaN values')  # Plot data with NaN values
plt.legend()  # Add legend to the plot
plt.title('Masked and NaN data')  # Add title to the plot
plt.show()  # Display the plot
```
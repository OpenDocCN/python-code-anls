# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\errorbar_limits.py`

```
"""
==============================================
Including upper and lower limits in error bars
==============================================

In matplotlib, errors bars can have "limits". Applying limits to the
error bars essentially makes the error unidirectional. Because of that,
upper and lower limits can be applied in both the y- and x-directions
via the ``uplims``, ``lolims``, ``xuplims``, and ``xlolims`` parameters,
respectively. These parameters can be scalar or boolean arrays.

For example, if ``xlolims`` is ``True``, the x-error bars will only
extend from the data towards increasing values. If ``uplims`` is an
array filled with ``False`` except for the 4th and 7th values, all of the
y-error bars will be bidirectional, except the 4th and 7th bars, which
will extend from the data towards decreasing y-values.
"""

import matplotlib.pyplot as plt
import numpy as np

# example data
x = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
y = np.exp(-x)
xerr = 0.1
yerr = 0.2

# lower & upper limits of the error
lolims = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0], dtype=bool)  # Defines lower limits for each error bar
uplims = np.array([0, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=bool)  # Defines upper limits for each error bar
ls = 'dotted'  # Line style for error bars

fig, ax = plt.subplots(figsize=(7, 4))

# standard error bars
ax.errorbar(x, y, xerr=xerr, yerr=yerr, linestyle=ls)  # Plot standard error bars

# including upper limits
ax.errorbar(x, y + 0.5, xerr=xerr, yerr=yerr, uplims=uplims, linestyle=ls)  # Plot with upper limits

# including lower limits
ax.errorbar(x, y + 1.0, xerr=xerr, yerr=yerr, lolims=lolims, linestyle=ls)  # Plot with lower limits

# including upper and lower limits
ax.errorbar(x, y + 1.5, xerr=xerr, yerr=yerr, lolims=lolims, uplims=uplims,
            marker='o', markersize=8, linestyle=ls)  # Plot with both upper and lower limits

# Plot a series with lower and upper limits in both x & y
# constant x-error with varying y-error
xerr = 0.2
yerr = np.full_like(x, 0.2)
yerr[[3, 6]] = 0.3

# mock up some limits by modifying previous data
xlolims = lolims  # Lower limits in x-direction
xuplims = uplims  # Upper limits in x-direction
lolims = np.zeros_like(x)  # Reset lower limits in y-direction
uplims = np.zeros_like(x)  # Reset upper limits in y-direction
lolims[[6]] = True  # Set specific index for lower limit in y-direction
uplims[[3]] = True  # Set specific index for upper limit in y-direction

# do the plotting
ax.errorbar(x, y + 2.1, xerr=xerr, yerr=yerr,
            xlolims=xlolims, xuplims=xuplims,
            uplims=uplims, lolims=lolims,
            marker='o', markersize=8,
            linestyle='none')  # Plot with mixed x and y limits

# tidy up the figure
ax.set_xlim((0, 5.5))  # Set x-axis limits
ax.set_title('Errorbar upper and lower limits')  # Set plot title
plt.show()  # Display the plot
```
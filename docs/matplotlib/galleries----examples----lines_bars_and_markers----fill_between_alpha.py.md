# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\fill_between_alpha.py`

```py
"""
Fill Between and Alpha
======================

The `~matplotlib.axes.Axes.fill_between` function generates a shaded
region between a min and max boundary that is useful for illustrating ranges.
It has a very handy ``where`` argument to combine filling with logical ranges,
e.g., to just fill in a curve over some threshold value.

At its most basic level, ``fill_between`` can be used to enhance a graph's
visual appearance. Let's compare two graphs of financial data with a simple
line plot on the left and a filled line on the right.
"""

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.cbook as cbook

# load up some sample financial data
r = cbook.get_sample_data('goog.npz')['price_data']
# create two subplots with the shared x and y axes
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

# Find the minimum closing price from the financial data
pricemin = r["close"].min()

# Plot the closing prices on the left subplot
ax1.plot(r["date"], r["close"], lw=2)
# Fill the area between the minimum price and closing prices with transparency
ax2.fill_between(r["date"], pricemin, r["close"], alpha=0.7)

# Set grid lines on for both subplots and exclude labeling for inner axes
for ax in ax1, ax2:
    ax.grid(True)
    ax.label_outer()

ax1.set_ylabel('price')

# Add the main title to the figure and rotate x-axis labels for better fit
fig.suptitle('Google (GOOG) daily closing price')
fig.autofmt_xdate()

# %%
# The alpha channel is not necessary here, but it can be used to soften
# colors for more visually appealing plots.  In other examples, as we'll
# see below, the alpha channel is functionally useful as the shaded
# regions can overlap and alpha allows you to see both.  Note that the
# postscript format does not support alpha (this is a postscript
# limitation, not a matplotlib limitation), so when using alpha save
# your figures in PNG, PDF or SVG.
#
# Our next example computes two populations of random walkers with a
# different mean and standard deviation of the normal distributions from
# which the steps are drawn.  We use filled regions to plot +/- one
# standard deviation of the mean position of the population.  Here the
# alpha channel is useful, not just aesthetic.

# Fixing random state for reproducibility
np.random.seed(19680801)

Nsteps, Nwalkers = 100, 250
t = np.arange(Nsteps)

# Generate random walk steps for two populations
S1 = 0.004 + 0.02*np.random.randn(Nsteps, Nwalkers)
S2 = 0.002 + 0.01*np.random.randn(Nsteps, Nwalkers)

# Calculate positions for the random walkers
X1 = S1.cumsum(axis=0)
X2 = S2.cumsum(axis=0)

# Calculate empirical means and standard deviations over time for both populations
mu1 = X1.mean(axis=1)
sigma1 = X1.std(axis=1)
mu2 = X2.mean(axis=1)
sigma2 = X2.std(axis=1)

# plot it!
fig, ax = plt.subplots(1)
# Plot mean positions for both populations
ax.plot(t, mu1, lw=2, label='mean population 1')
ax.plot(t, mu2, lw=2, label='mean population 2')
# Fill between the upper and lower bounds of +/- one standard deviation for both populations
ax.fill_between(t, mu1+sigma1, mu1-sigma1, facecolor='C0', alpha=0.4)
ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='C1', alpha=0.4)
ax.set_title(r'random walkers empirical $\mu$ and $\pm \sigma$ interval')
ax.legend(loc='upper left')
ax.set_xlabel('num steps')
ax.set_ylabel('position')
ax.grid()

# %%
# The ``where`` keyword argument is very handy for highlighting certain
# regions of the graph.  ``where`` takes a boolean mask the same length
# as the x, ymin and ymax arguments, and only fills in the region where
# the boolean mask is True.  In the example below, we simulate a single
# random walker and compute the analytic mean and standard deviation of
# the population positions.  The population mean is shown as the dashed
# line, and the plus/minus one sigma deviation from the mean is shown
# as the filled region.  We use the where mask ``X > upper_bound`` to
# find the region where the walker is outside the one sigma boundary,
# and shade that region red.

# Fixing random state for reproducibility
np.random.seed(1)

# Number of steps in the random walk
Nsteps = 500
# Time steps
t = np.arange(Nsteps)

# Mean and standard deviation for the random walk
mu = 0.002
sigma = 0.01

# Generate random steps and calculate positions
S = mu + sigma*np.random.randn(Nsteps)
X = S.cumsum()

# Calculate the analytic upper and lower bounds for the population positions
lower_bound = mu*t - sigma*np.sqrt(t)
upper_bound = mu*t + sigma*np.sqrt(t)

# Create a figure and axis for plotting
fig, ax = plt.subplots(1)

# Plot the walker's position as a function of time
ax.plot(t, X, lw=2, label='walker position')

# Plot the population mean as a dashed line
ax.plot(t, mu*t, lw=1, label='population mean', color='C0', ls='--')

# Fill the region between the lower and upper bounds with a light blue color
ax.fill_between(t, lower_bound, upper_bound, facecolor='C0', alpha=0.4,
                label='1 sigma range')

# Add legend to the plot
ax.legend(loc='upper left')

# here we use the where argument to only fill the region where the
# walker is above the population 1 sigma boundary
# Fill the region where the walker's position is above the upper bound with red color
ax.fill_between(t, upper_bound, X, where=X > upper_bound, fc='red', alpha=0.4)
# Fill the region where the walker's position is below the lower bound with red color
ax.fill_between(t, lower_bound, X, where=X < lower_bound, fc='red', alpha=0.4)

# Set labels for x and y axes
ax.set_xlabel('num steps')
ax.set_ylabel('position')

# Add grid to the plot
ax.grid()

# %%
# Another handy use of filled regions is to highlight horizontal or vertical
# spans of an Axes -- for that Matplotlib has the helper functions
# `~matplotlib.axes.Axes.axhspan` and `~matplotlib.axes.Axes.axvspan`.  See
# :doc:`/gallery/subplots_axes_and_figures/axhspan_demo`.

# Display the plot
plt.show()
```
# `D:\src\scipysrc\matplotlib\galleries\examples\statistics\customized_violin.py`

```py
"""
=========================
Violin plot customization
=========================

This example demonstrates how to fully customize violin plots. The first plot
shows the default style by providing only the data. The second plot first
limits what Matplotlib draws with additional keyword arguments. Then a
simplified representation of a box plot is drawn on top. Lastly, the styles of
the artists of the violins are modified.

For more information on violin plots, the scikit-learn docs have a great
section: https://scikit-learn.org/stable/modules/density.html
"""

import matplotlib.pyplot as plt
import numpy as np


def adjacent_values(vals, q1, q3):
    # Calculate upper adjacent value for whiskers
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    # Calculate lower adjacent value for whiskers
    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    # Set ticks and labels for x-axis
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


# create test data
np.random.seed(19680801)
data = [sorted(np.random.normal(0, std, 100)) for std in range(1, 5)]

# Create subplots with shared y-axis
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)

# Plot for the default violin plot
ax1.set_title('Default violin plot')
ax1.set_ylabel('Observed values')
ax1.violinplot(data)

# Plot for the customized violin plot
ax2.set_title('Customized violin plot')
# Customize the violin plot parts
parts = ax2.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)

for pc in parts['bodies']:
    # Customize violin bodies' appearance
    pc.set_facecolor('#D43F3A')
    pc.set_edgecolor('black')
    pc.set_alpha(1)

# Calculate quartiles and whiskers
quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

inds = np.arange(1, len(medians) + 1)
# Add markers for medians
ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
# Add vertical lines for quartiles
ax2.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# Add vertical lines for whiskers
ax2.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# Set style for the axes
labels = ['A', 'B', 'C', 'D']
for ax in [ax1, ax2]:
    set_axis_style(ax, labels)

# Adjust subplot spacing
plt.subplots_adjust(bottom=0.15, wspace=0.05)
# Display the plot
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.violinplot` / `matplotlib.pyplot.violinplot`
#    - `matplotlib.axes.Axes.vlines` / `matplotlib.pyplot.vlines`
```
# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\bar_label_demo.py`

```py
"""
==============
Bar Label Demo
==============

This example shows how to use the `~.Axes.bar_label` helper function
to create bar chart labels.

See also the :doc:`grouped bar
</gallery/lines_bars_and_markers/barchart>`,
:doc:`stacked bar
</gallery/lines_bars_and_markers/bar_stacked>` and
:doc:`horizontal bar chart
</gallery/lines_bars_and_markers/barh>` examples.
"""

import matplotlib.pyplot as plt
import numpy as np

# %%
# data from https://allisonhorst.github.io/palmerpenguins/

# Penguin species and their corresponding counts by sex
species = ('Adelie', 'Chinstrap', 'Gentoo')
sex_counts = {
    'Male': np.array([73, 34, 61]),
    'Female': np.array([73, 34, 58]),
}
width = 0.6  # the width of the bars: can also be len(x) sequence

# Create a new figure and axis for plotting
fig, ax = plt.subplots()
bottom = np.zeros(3)

# Iterate over each sex and plot stacked bars for penguin counts
for sex, sex_count in sex_counts.items():
    p = ax.bar(species, sex_count, width, label=sex, bottom=bottom)
    bottom += sex_count

    # Add labels to the center of each bar
    ax.bar_label(p, label_type='center')

# Set plot title, add legend, and display the plot
ax.set_title('Number of penguins by sex')
ax.legend()
plt.show()

# %%
# Horizontal bar chart

# Fixing random state for reproducibility
np.random.seed(19680801)

# Example data for people and their performance
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

# Create a new figure and axis for horizontal bar chart
fig, ax = plt.subplots()

# Plot horizontal bars with performance data and error bars
hbars = ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

# Label bars with formatted floats
ax.bar_label(hbars, fmt='%.2f')
ax.set_xlim(right=15)  # adjust xlim to fit labels

plt.show()

# %%
# Some of the more advanced things that one can do with bar labels

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot horizontal bars with performance data and error bars
hbars = ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

# Label bars with given captions, custom padding, color, and font size
ax.bar_label(hbars, labels=[f'±{e:.2f}' for e in error],
             padding=8, color='b', fontsize=14)
ax.set_xlim(right=16)

plt.show()

# %%
# Bar labels using {}-style format string

# Names and counts of different fruit flavors
fruit_names = ['Coffee', 'Salted Caramel', 'Pistachio']
fruit_counts = [4000, 2000, 7000]

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot vertical bars with fruit sales data
bar_container = ax.bar(fruit_names, fruit_counts)
ax.set(ylabel='pints sold', title='Gelato sales by flavor', ylim=(0, 8000))

# Label bars with formatted integers
ax.bar_label(bar_container, fmt='{:,.0f}')

# %%
# Bar labels using a callable

# Names and speeds of different animals
animal_names = ['Lion', 'Gazelle', 'Cheetah']
mph_speed = [50, 60, 75]

# Create a new figure and axis
fig, ax = plt.subplots()

# Plot vertical bars with animal speed data
bar_container = ax.bar(animal_names, mph_speed)
ax.set(ylabel='speed in MPH', title='Running speeds', ylim=(0, 80))

# Label bars with speeds converted to km/h using a lambda function
ax.bar_label(bar_container, fmt=lambda x: f'{x * 1.61:.1f} km/h')

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
#    - `matplotlib.axes.Axes.barh` / `matplotlib.pyplot.barh`
#    - `matplotlib.axes.Axes.bar_label` / `matplotlib.pyplot.bar_label`
```
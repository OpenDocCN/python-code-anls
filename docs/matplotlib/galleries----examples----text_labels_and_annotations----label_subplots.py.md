# `D:\src\scipysrc\matplotlib\galleries\examples\text_labels_and_annotations\label_subplots.py`

```py
"""
==================
Labelling subplots
==================

Labelling subplots is relatively straightforward, and varies, so Matplotlib
does not have a general method for doing this.

We showcase two methods to position text at a given physical offset (in
fontsize units or in points) away from a corner of the Axes: one using
`~.Axes.annotate`, and one using `.ScaledTranslation`.

For convenience, this example uses `.pyplot.subplot_mosaic` and subplot
labels as keys for the subplots.  However, the approach also works with
`.pyplot.subplots` or keys that are different from what you want to label the
subplot with.
"""

import matplotlib.pyplot as plt

from matplotlib.transforms import ScaledTranslation

# Create a figure and subplots grid using `subplot_mosaic`
fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              layout='constrained')

# Iterate over each subplot and its corresponding label
for label, ax in axs.items():
    # Use Axes.annotate to annotate the subplot with the label
    # - Place the annotation at the top left corner of the subplot (axes fraction (0, 1))
    # - Offset the annotation to be half-a-fontsize right and half-a-fontsize down
    #   (offset fontsize units (+0.5, -0.5)), positioning it just inside the axes
    ax.annotate(
        label,
        xy=(0, 1), xycoords='axes fraction',
        xytext=(+0.5, -0.5), textcoords='offset points',
        fontsize='medium', verticalalignment='top', fontfamily='serif',
        bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))

# Create another figure and subplots grid with `subplot_mosaic`
fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              layout='constrained')

# Iterate over each subplot and its corresponding label
for label, ax in axs.items():
    # Use ax.text with ScaledTranslation to place text at a physical offset
    # - Place the text at the top left corner of the subplot (axes fraction (0, 1))
    # - Offset the text by -20 points left and +7 points up (offset points (-20, +7)),
    #   positioning it just outside the axes
    ax.text(
        0.0, 1.0, label, transform=(
            ax.transAxes + ScaledTranslation(-20/72, +7/72, fig.dpi_scale_trans)),
        fontsize='medium', va='bottom', fontfamily='serif')

# Create another figure and subplots grid with `subplot_mosaic`
fig, axs = plt.subplot_mosaic([['a)', 'c)'], ['b)', 'c)'], ['d)', 'd)']],
                              layout='constrained')

# Iterate over each subplot
for label, ax in axs.items():
    # Set a normal title for the subplot
    ax.set_title('Normal Title', fontstyle='italic')
    # Set a custom title label for the subplot aligned to the left
    ax.set_title(label, fontfamily='serif', loc='left', fontsize='medium')

plt.show()

# References section describing the used functions, methods, classes, and modules
"""
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.figure.Figure.subplot_mosaic` /
#      `matplotlib.pyplot.subplot_mosaic`
#    - `matplotlib.axes.Axes.set_title`
#    - `matplotlib.axes.Axes.annotate`
#
"""
```
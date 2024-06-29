# `D:\src\scipysrc\matplotlib\galleries\examples\subplots_axes_and_figures\gridspec_and_subplots.py`

```
"""
==================================================
Combining two subplots using subplots and GridSpec
==================================================

Sometimes we want to combine two subplots in an Axes layout created with
`~.Figure.subplots`.  We can get the `~.gridspec.GridSpec` from the Axes
and then remove the covered Axes and fill the gap with a new bigger Axes.
Here we create a layout with the bottom two Axes in the last column combined.

To start with this layout (rather than removing the overlapping Axes) use
`~.pyplot.subplot_mosaic`.

See also :ref:`arranging_axes`.
"""

# 导入 matplotlib.pyplot 库
import matplotlib.pyplot as plt

# 创建一个包含3列3行子图的图形对象 fig 和子图数组 axs
fig, axs = plt.subplots(ncols=3, nrows=3)

# 获取第1行第2列子图的 GridSpec 对象
gs = axs[1, 2].get_gridspec()

# 移除最后一列除第一行以外的所有子图
for ax in axs[1:, -1]:
    ax.remove()

# 在 GridSpec 的第1行到最后一行，最后一列创建一个新的大子图
axbig = fig.add_subplot(gs[1:, -1])

# 在新大子图上添加注释
axbig.annotate('Big Axes \nGridSpec[1:, -1]', (0.1, 0.5),
               xycoords='axes fraction', va='center')

# 调整子图布局使其紧凑显示
fig.tight_layout()

# 显示图形
plt.show()
```
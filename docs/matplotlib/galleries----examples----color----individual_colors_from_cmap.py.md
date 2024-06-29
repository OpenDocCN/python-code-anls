# `D:\src\scipysrc\matplotlib\galleries\examples\color\individual_colors_from_cmap.py`

```py
"""
===========================================
Selecting individual colors from a colormap
===========================================

Sometimes we want to use more colors or a different set of colors than the default color
cycle provides. Selecting individual colors from one of the provided colormaps can be a
convenient way to do this.

We can retrieve colors from any `.Colormap` by calling it with a float or a list of
floats in the range [0, 1]; e.g. ``cmap(0.5)`` will give the middle color. See also
`.Colormap.__call__`.

Extracting colors from a continuous colormap
--------------------------------------------
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 模块，用于数值计算

import matplotlib as mpl  # 导入 matplotlib 库

n_lines = 21  # 设定绘制线条的数量为 21 条
cmap = mpl.colormaps['plasma']  # 从 matplotlib 的 colormaps 中选择 'plasma' 颜色映射

# 在 colormap 中取出均匀间隔的颜色。
colors = cmap(np.linspace(0, 1, n_lines))  # 从 colormap 中按照均匀间隔取出 n_lines 条颜色

fig, ax = plt.subplots(layout='constrained')  # 创建一个具有约束布局的图形和坐标轴对象

for i, color in enumerate(colors):  # 遍历颜色列表
    ax.plot([0, i], color=color)  # 在坐标轴上绘制线条，每条线使用对应的颜色

plt.show()  # 显示绘制的图形

# %%
#
# Extracting colors from a discrete colormap
# ------------------------------------------
# The list of all colors in a `.ListedColormap` is available as the ``colors``
# attribute.

colors = mpl.colormaps['Dark2'].colors  # 从 matplotlib 的 colormaps 中选择 'Dark2' 的颜色列表

fig, ax = plt.subplots(layout='constrained')  # 创建一个具有约束布局的图形和坐标轴对象

for i, color in enumerate(colors):  # 遍历颜色列表
    ax.plot([0, i], color=color)  # 在坐标轴上绘制线条，每条线使用对应的颜色

plt.show()  # 显示绘制的图形

# %%
# See Also
# --------
#
# For more details about manipulating colormaps, see :ref:`colormap-manipulation`.  To
# change the default color cycle, see :ref:`color_cycle`.
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.colors.Colormap`
#    - `matplotlib.colors.Colormap.resampled`
```
# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\line_collection.py`

```
"""
=============================================
Plotting multiple lines with a LineCollection
=============================================

Matplotlib can efficiently draw multiple lines at once using a `~.LineCollection`.
"""

# 导入 matplotlib 库的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，用于数值计算
import numpy as np

# 从 matplotlib 的 collections 模块中导入 LineCollection 类
from matplotlib.collections import LineCollection

# 定义颜色列表，每个颜色对应一个半圆
colors = ["indigo", "blue", "green", "yellow", "orange", "red"]

# 创建半圆的列表，每个半圆半径不同
theta = np.linspace(0, np.pi, 36)
radii = np.linspace(4, 5, num=len(colors))
arcs = [np.column_stack([r * np.cos(theta), r * np.sin(theta)]) for r in radii]

# 创建图形和子图对象
fig, ax = plt.subplots(figsize=(6.4, 3.2))
# 手动设置坐标轴的范围，因为 LineCollection 不参与自动缩放
ax.set_xlim(-6, 6)
ax.set_ylim(0, 6)
ax.set_aspect("equal")  # 使半圆看起来是圆形的

# 创建 LineCollection 对象，包含多个半圆
# 可以通过传递序列（这里用于 *colors*）为每条线设置属性，或者通过传递标量（这里用于 *linewidths*）为所有线设置属性
line_collection = LineCollection(arcs, colors=colors, linewidths=4)
ax.add_collection(line_collection)

plt.show()

# %%
# Instead of passing a list of colors (``colors=colors``), we can alternatively use
# colormapping. The lines are then color-coded based on an additional array of values
# passed to the *array* parameter. In the below example, we color the lines based on
# their radius by passing ``array=radii``.

# 定义半圆数量
num_arcs = 15
theta = np.linspace(0, np.pi, 36)
radii = np.linspace(4, 5.5, num=num_arcs)
arcs = [np.column_stack([r * np.cos(theta), r * np.sin(theta)]) for r in radii]

# 创建图形和子图对象
fig, ax = plt.subplots(figsize=(6.4, 3))
# 手动设置坐标轴的范围，因为 LineCollection 不参与自动缩放
ax.set_xlim(-6, 6)
ax.set_ylim(0, 6)
ax.set_aspect("equal")  # 使半圆看起来是圆形的

# 创建 LineCollection 对象，包含多个半圆，并进行颜色映射
line_collection = LineCollection(arcs, array=radii, cmap="rainbow")
ax.add_collection(line_collection)

# 添加颜色条
fig.colorbar(line_collection, label="Radius")
ax.set_title("Line Collection with mapped colors")

plt.show()
# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.collections.LineCollection`
#    - `matplotlib.collections.Collection.set_array`
#    - `matplotlib.axes.Axes.add_collection`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
```
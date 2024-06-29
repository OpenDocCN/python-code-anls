# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\hatch_style_reference.py`

```
"""
=====================
Hatch style reference
=====================

Hatches can be added to most polygons in Matplotlib, including `~.Axes.bar`,
`~.Axes.fill_between`, `~.Axes.contourf`, and children of `~.patches.Polygon`.
They are currently supported in the PS, PDF, SVG, macosx, and Agg backends. The WX
and Cairo backends do not currently support hatching.

See also :doc:`/gallery/images_contours_and_fields/contourf_hatching` for
an example using `~.Axes.contourf`, and
:doc:`/gallery/shapes_and_collections/hatch_demo` for more usage examples.

"""
# 导入 Matplotlib 库
import matplotlib.pyplot as plt

# 从 Matplotlib 中导入 Rectangle 类
from matplotlib.patches import Rectangle

# 创建一个包含 2 行 5 列子图的图形对象，并设置布局为 constrained，尺寸为 6.4x3.2 英寸
fig, axs = plt.subplots(2, 5, layout='constrained', figsize=(6.4, 3.2))

# 定义不同的填充样式
hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

# 定义一个函数，用于在子图上添加矩形框和文本，并设置填充样式
def hatches_plot(ax, h):
    ax.add_patch(Rectangle((0, 0), 2, 2, fill=False, hatch=h))  # 在指定的坐标位置添加一个无填充的矩形框，并设置填充样式
    ax.text(1, -0.5, f"' {h} '", size=15, ha="center")  # 在指定位置添加文本，显示填充样式
    ax.axis('equal')  # 设置坐标轴比例为相等
    ax.axis('off')  # 关闭坐标轴显示

# 遍历子图数组和填充样式数组，将每种填充样式应用到相应的子图上
for ax, h in zip(axs.flat, hatches):
    hatches_plot(ax, h)

# %%
# Hatching patterns can be repeated to increase the density.

# 创建一个包含 2 行 5 列子图的图形对象，并设置布局为 constrained，尺寸为 6.4x3.2 英寸
fig, axs = plt.subplots(2, 5, layout='constrained', figsize=(6.4, 3.2))

# 定义不同的填充样式，这些样式将用于显示填充重复增加的效果
hatches = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

# 遍历子图数组和填充样式数组，将每种填充样式应用到相应的子图上
for ax, h in zip(axs.flat, hatches):
    hatches_plot(ax, h)

# %%
# Hatching patterns can be combined to create additional patterns.

# 创建一个包含 2 行 5 列子图的图形对象，并设置布局为 constrained，尺寸为 6.4x3.2 英寸
fig, axs = plt.subplots(2, 5, layout='constrained', figsize=(6.4, 3.2))

# 定义不同的填充样式，这些样式将用于显示填充组合创建的额外效果
hatches = ['/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']

# 遍历子图数组和填充样式数组，将每种填充样式应用到相应的子图上
for ax, h in zip(axs.flat, hatches):
    hatches_plot(ax, h)

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.Rectangle`
#    - `matplotlib.axes.Axes.add_patch`
#    - `matplotlib.axes.Axes.text`
```
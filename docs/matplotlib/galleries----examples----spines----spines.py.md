# `D:\src\scipysrc\matplotlib\galleries\examples\spines\spines.py`

```py
"""
======
Spines
======

This demo compares:

- normal Axes, with spines on all four sides;
- an Axes with spines only on the left and bottom;
- an Axes using custom bounds to limit the extent of the spine.

Each `.axes.Axes` has a list of `.Spine` objects, accessible
via the container ``ax.spines``.

.. redirect-from:: /gallery/spines/spines_bounds

"""
# 导入 matplotlib 库
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
x = np.linspace(0, 2 * np.pi, 100)
y = 2 * np.sin(x)

# 创建包含三个子图的图形窗口，使用 constrained layout 避免标签重叠
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, layout='constrained')

# 在第一个子图上绘制数据
ax0.plot(x, y)
ax0.set_title('normal spines')  # 设置子图标题

# 在第二个子图上绘制数据
ax1.plot(x, y)
ax1.set_title('bottom-left spines')  # 设置子图标题

# 隐藏右侧和顶部的坐标轴边框
ax1.spines.right.set_visible(False)
ax1.spines.top.set_visible(False)

# 在第三个子图上绘制数据
ax2.plot(x, y)
ax2.set_title('spines with bounds limited to data range')  # 设置子图标题

# 设置底部和左侧坐标轴边框的范围限制为数据范围
ax2.spines.bottom.set_bounds(x.min(), x.max())
ax2.spines.left.set_bounds(y.min(), y.max())
# 隐藏右侧和顶部的坐标轴边框
ax2.spines.right.set_visible(False)
ax2.spines.top.set_visible(False)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.artist.Artist.set_visible`
#    - `matplotlib.spines.Spine.set_bounds`
```
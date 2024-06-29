# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\contour_corner_mask.py`

```
"""
===================
Contour Corner Mask
===================

Illustrate the difference between ``corner_mask=False`` and
``corner_mask=True`` for masked contour plots.  The default is controlled by
:rc:`contour.corner_mask`.
"""

# 导入 Matplotlib 库
import matplotlib.pyplot as plt
import numpy as np

# Data to plot.
# 创建网格数据
x, y = np.meshgrid(np.arange(7), np.arange(10))
# 计算 z 值
z = np.sin(0.5 * x) * np.cos(0.52 * y)

# Mask various z values.
# 创建布尔掩码数组，用于掩盖特定的 z 值
mask = np.zeros_like(z, dtype=bool)
mask[2, 3:5] = True
mask[3:5, 4] = True
mask[7, 2] = True
mask[5, 0] = True
mask[0, 6] = True
# 使用掩码创建掩盖后的 z 数组
z = np.ma.array(z, mask=mask)

# Define corner_mask values to illustrate
# 定义 corner_mask 参数的取值以进行对比展示
corner_masks = [False, True]

# 创建包含两个子图的图形窗口
fig, axs = plt.subplots(ncols=2)

# 遍历每个子图和对应的 corner_mask 参数取值
for ax, corner_mask in zip(axs, corner_masks):
    # 绘制填充的等高线图
    cs = ax.contourf(x, y, z, corner_mask=corner_mask)
    # 绘制等高线
    ax.contour(cs, colors='k')
    # 设置子图标题，显示当前的 corner_mask 参数取值
    ax.set_title(f'{corner_mask=}')

    # Plot grid.
    # 绘制网格线
    ax.grid(c='k', ls='-', alpha=0.3)

    # Indicate masked points with red circles.
    # 使用红色圆圈标记掩盖的点
    ax.plot(np.ma.array(x, mask=~mask), y, 'ro')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
```
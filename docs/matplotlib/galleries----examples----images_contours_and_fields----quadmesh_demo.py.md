# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\quadmesh_demo.py`

```
"""
=============
QuadMesh Demo
=============

`~.axes.Axes.pcolormesh` uses a `~matplotlib.collections.QuadMesh`,
a faster generalization of `~.axes.Axes.pcolor`, but with some restrictions.

This demo illustrates a bug in quadmesh with masked data.
"""

import numpy as np  # 导入 NumPy 库，用于数值计算

from matplotlib import pyplot as plt  # 导入 matplotlib 的 pyplot 模块，并使用 plt 别名

n = 12  # 设置数据点数量
x = np.linspace(-1.5, 1.5, n)  # 在指定范围内生成 n 个均匀间隔的数据点作为 x 坐标
y = np.linspace(-1.5, 1.5, n * 2)  # 在指定范围内生成 n*2 个均匀间隔的数据点作为 y 坐标
X, Y = np.meshgrid(x, y)  # 生成网格数据 X 和 Y

Qx = np.cos(Y) - np.cos(X)  # 计算 Qx 数据
Qz = np.sin(Y) + np.sin(X)  # 计算 Qz 数据
Z = np.sqrt(X**2 + Y**2) / 5  # 计算 Z 数据并归一化到 [0, 1] 范围
Z = (Z - Z.min()) / (Z.max() - Z.min())  # 对 Z 数据进行归一化处理

# The color array can include masked values.
Zm = np.ma.masked_where(np.abs(Qz) < 0.5 * np.max(Qz), Z)  # 根据条件创建一个掩码数组 Zm

fig, axs = plt.subplots(nrows=1, ncols=3)  # 创建包含 1 行 3 列子图的图形对象 fig 和坐标轴对象 axs

axs[0].pcolormesh(Qx, Qz, Z, shading='gouraud')  # 在第一个子图中绘制二维伪彩色图，不使用掩码值
axs[0].set_title('Without masked values')  # 设置第一个子图标题

# You can control the color of the masked region.
cmap = plt.colormaps[plt.rcParams['image.cmap']].with_extremes(bad='y')  # 定义一个带掩码值的颜色映射 cmap
axs[1].pcolormesh(Qx, Qz, Zm, shading='gouraud', cmap=cmap)  # 在第二个子图中绘制二维伪彩色图，使用掩码值和自定义颜色映射
axs[1].set_title('With masked values')  # 设置第二个子图标题

# Or use the default, which is transparent.
axs[2].pcolormesh(Qx, Qz, Zm, shading='gouraud')  # 在第三个子图中绘制二维伪彩色图，使用掩码值和默认透明度
axs[2].set_title('With masked values')  # 设置第三个子图标题

fig.tight_layout()  # 调整子图布局，使之紧凑显示
plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.pcolormesh` / `matplotlib.pyplot.pcolormesh`
```
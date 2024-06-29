# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\contour_demo.py`

```py
"""
============
Contour Demo
============

Illustrate simple contour plotting, contours on an image with
a colorbar for the contours, and labelled contours.

See also the :doc:`contour image example
</gallery/images_contours_and_fields/contour_image>`.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

import matplotlib.cm as cm  # 导入 matplotlib.cm 库，用于颜色映射

# 定义步长和坐标轴范围
delta = 0.025
x = np.arange(-3.0, 3.0, delta)  # 生成 x 轴数据，范围从 -3.0 到 3.0
y = np.arange(-2.0, 2.0, delta)  # 生成 y 轴数据，范围从 -2.0 到 2.0
X, Y = np.meshgrid(x, y)  # 生成网格数据 X 和 Y

# 计算函数 Z
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

# %%
# 创建一个简单的等高线图，并带有标签，使用默认颜色。
# clabel 中的 inline 参数控制是否在等高线线段上绘制标签，移除标签下方的线段。

fig, ax = plt.subplots()  # 创建图形和坐标轴对象
CS = ax.contour(X, Y, Z)  # 绘制等高线图
ax.clabel(CS, inline=True, fontsize=10)  # 添加等高线标签
ax.set_title('Simplest default with labels')  # 设置图表标题

# %%
# 可以通过提供数据坐标位置列表手动放置等高线标签。

fig, ax = plt.subplots()  # 创建图形和坐标轴对象
CS = ax.contour(X, Y, Z)  # 绘制等高线图
manual_locations = [
    (-1, -1.4), (-0.62, -0.7), (-2, 0.5), (1.7, 1.2), (2.0, 1.4), (2.4, 1.7)
]
ax.clabel(CS, inline=True, fontsize=10, manual=manual_locations)  # 手动指定标签位置
ax.set_title('labels at selected locations')  # 设置图表标题

# %%
# 可以强制所有的等高线颜色相同。

fig, ax = plt.subplots()  # 创建图形和坐标轴对象
CS = ax.contour(X, Y, Z, 6, colors='k')  # 绘制等高线图，指定颜色为黑色
ax.clabel(CS, fontsize=9, inline=True)  # 添加等高线标签
ax.set_title('Single color - negative contours dashed')  # 设置图表标题

# %%
# 可以将负等高线线型设置为实线而不是虚线：

plt.rcParams['contour.negative_linestyle'] = 'solid'  # 设置负等高线线型为实线
fig, ax = plt.subplots()  # 创建图形和坐标轴对象
CS = ax.contour(X, Y, Z, 6, colors='k')  # 绘制等高线图，指定颜色为黑色
ax.clabel(CS, fontsize=9, inline=True)  # 添加等高线标签
ax.set_title('Single color - negative contours solid')  # 设置图表标题

# %%
# 还可以手动指定等高线的颜色。

fig, ax = plt.subplots()  # 创建图形和坐标轴对象
CS = ax.contour(X, Y, Z, 6,
                linewidths=np.arange(.5, 4, .5),
                colors=('r', 'green', 'blue', (1, 1, 0), '#afeeee', '0.5'),
                )  # 绘制等高线图，指定不同的颜色和线宽
ax.clabel(CS, fontsize=9, inline=True)  # 添加等高线标签
ax.set_title('Crazy lines')  # 设置图表标题

# %%
# 或者可以使用 colormap 指定等高线的颜色；默认的 colormap 将用于等高线线条。

fig, ax = plt.subplots()  # 创建图形和坐标轴对象
im = ax.imshow(Z, interpolation='bilinear', origin='lower',
               cmap=cm.gray, extent=(-3, 3, -2, 2))  # 绘制灰度图像
levels = np.arange(-1.2, 1.6, 0.2)  # 定义等高线的层级
CS = ax.contour(Z, levels, origin='lower', cmap='flag', extend='both',
                linewidths=2, extent=(-3, 3, -2, 2))  # 绘制等高线图

# 加粗零等高线。
lws = np.resize(CS.get_linewidth(), len(levels))
lws[6] = 4
CS.set_linewidth(lws)

ax.clabel(CS, levels[1::2],  # 每隔一个层级标签一次
          inline=True, fmt='%1.1f', fontsize=14)  # 添加等高线标签

# 为等高线线条添加颜色条
CB = fig.colorbar(CS, shrink=0.8)

ax.set_title('Lines with colorbar')  # 设置图表标题
# 为图像添加一个颜色条，水平方向，收缩系数为0.8
CBI = fig.colorbar(im, orientation='horizontal', shrink=0.8)

# 调整原始颜色条的位置，使其看起来更加协调
# 获取当前轴的位置和大小信息
l, b, w, h = ax.get_position().bounds
# 获取颜色条轴的位置和大小信息
ll, bb, ww, hh = CB.ax.get_position().bounds
# 设置颜色条轴的新位置，使其相对于当前轴向下移动0.1倍当前轴高度，并保持宽度和高度的比例
CB.ax.set_position([ll, b + 0.1*h, ww, h*0.8])

# 显示绘图结果
plt.show()



# %%
#
# .. admonition:: References
#
#    本示例展示了以下函数、方法、类和模块的使用:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.axes.Axes.get_position`
#    - `matplotlib.axes.Axes.set_position`
```
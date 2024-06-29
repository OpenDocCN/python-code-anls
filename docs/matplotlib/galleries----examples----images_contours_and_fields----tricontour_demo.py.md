# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\tricontour_demo.py`

```
"""
===============
Tricontour Demo
===============

Contour plots of unstructured triangular grids.
"""
# 导入matplotlib库中的pyplot模块，并简称为plt
import matplotlib.pyplot as plt
# 导入numpy库，并简称为np
import numpy as np

# 导入matplotlib库中的tri模块，用于处理三角网格
import matplotlib.tri as tri

# %%
# 创建一个未指定三角形的三角网格（Triangulation），将自动进行点的Delaunay三角化

# 首先创建点的x和y坐标
n_angles = 48
n_radii = 8
min_radius = 0.25
radii = np.linspace(min_radius, 0.95, n_radii)

angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += np.pi / n_angles

x = (radii * np.cos(angles)).flatten()
y = (radii * np.sin(angles)).flatten()
z = (np.cos(radii) * np.cos(3 * angles)).flatten()

# 创建三角网格Triangulation对象，未指定三角形，因此将自动进行Delaunay三角化
triang = tri.Triangulation(x, y)

# 屏蔽不需要的三角形
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                         y[triang.triangles].mean(axis=1))
                < min_radius)

# %%
# 绘制填充色的等高线图

# 创建一个新的图形和坐标轴
fig1, ax1 = plt.subplots()
# 设置坐标轴比例为相等
ax1.set_aspect('equal')
# 绘制填充色的三角网格等高线图
tcf = ax1.tricontourf(triang, z)
# 添加颜色条
fig1.colorbar(tcf)
# 绘制三角网格的等高线
ax1.tricontour(triang, z, colors='k')
# 设置图标题
ax1.set_title('Contour plot of Delaunay triangulation')


# %%
# 除了指定不同的颜色映射，还可以指定不同的填充样式（hatching patterns）。

# 创建一个新的图形和坐标轴
fig2, ax2 = plt.subplots()
# 设置坐标轴比例为相等
ax2.set_aspect("equal")
# 绘制填充色的三角网格等高线图，指定填充样式和颜色映射
tcf = ax2.tricontourf(
    triang,
    z,
    hatches=["*", "-", "/", "//", "\\", None],
    cmap="cividis"
)
# 添加颜色条
fig2.colorbar(tcf)
# 绘制三角网格的等高线，指定线条样式和颜色
ax2.tricontour(triang, z, linestyles="solid", colors="k", linewidths=2.0)
# 设置图标题
ax2.set_title("Hatched Contour plot of Delaunay triangulation")

# %%
# 还可以生成带标签的填充样式，但是不带颜色。

# 创建一个新的图形和坐标轴
fig3, ax3 = plt.subplots()
# 指定等高线的级别数
n_levels = 7
# 绘制填充样式为点状、斜线、反斜线等的等高线图
tcf = ax3.tricontourf(
    triang,
    z,
    n_levels,
    colors="none",
    hatches=[".", "/", "\\", None, "\\\\", "*"],
)
# 绘制黑色的等高线
ax3.tricontour(triang, z, n_levels, colors="black", linestyles="-")

# 为等高线集合创建图例
artists, labels = tcf.legend_elements(str_format="{:2.1f}".format)
# 添加图例到坐标轴
ax3.legend(artists, labels, handleheight=2, framealpha=1)

# %%
# 也可以手动指定三角化，而不是自动进行点的Delaunay三角化，每个三角形由三个点的索引组成

# 创建一个新的点坐标数组
xy = np.asarray([
    [-0.101, 0.872], [-0.080, 0.883], [-0.069, 0.888], [-0.054, 0.890],
    [-0.045, 0.897], [-0.057, 0.895], [-0.073, 0.900], [-0.087, 0.898],
    [-0.090, 0.904], [-0.069, 0.907], [-0.069, 0.921], [-0.080, 0.919],
    [-0.073, 0.928], [-0.052, 0.930], [-0.048, 0.942], [-0.062, 0.949],
    [-0.054, 0.958], [-0.069, 0.954], [-0.087, 0.952], [-0.087, 0.959],
    [-0.080, 0.966], [-0.085, 0.973], [-0.087, 0.965], [-0.097, 0.965],
    [-0.097, 0.975], [-0.092, 0.984], [-0.101, 0.980], [-0.108, 0.980],
    [-0.104, 0.987], [-0.102, 0.993], [-0.115, 1.001], [-0.099, 0.996],
    # 定义一个包含多个二元组的列表，每个二元组表示一个坐标点的 (x, y) 值
    points = [
        [-0.101, 1.007], [-0.090, 1.010], [-0.087, 1.021], [-0.069, 1.021],
        [-0.052, 1.022], [-0.052, 1.017], [-0.069, 1.010], [-0.064, 1.005],
        [-0.048, 1.005], [-0.031, 1.005], [-0.031, 0.996], [-0.040, 0.987],
        [-0.045, 0.980], [-0.052, 0.975], [-0.040, 0.973], [-0.026, 0.968],
        [-0.020, 0.954], [-0.006, 0.947], [ 0.003, 0.935], [ 0.006, 0.926],
        [ 0.005, 0.921], [ 0.022, 0.923], [ 0.033, 0.912], [ 0.029, 0.905],
        [ 0.017, 0.900], [ 0.012, 0.895], [ 0.027, 0.893], [ 0.019, 0.886],
        [ 0.001, 0.883], [-0.012, 0.884], [-0.029, 0.883], [-0.038, 0.879],
        [-0.057, 0.881], [-0.062, 0.876], [-0.078, 0.876], [-0.087, 0.872],
        [-0.030, 0.907], [-0.007, 0.905], [-0.057, 0.916], [-0.025, 0.933],
        [-0.077, 0.990], [-0.059, 0.993]
    ]
x = np.degrees(xy[:, 0])
# 将数组 xy 中所有行的第一列元素转换为角度制，并赋值给 x

y = np.degrees(xy[:, 1])
# 将数组 xy 中所有行的第二列元素转换为角度制，并赋值给 y

x0 = -5
# 设置 x0 为 -5

y0 = 52
# 设置 y0 为 52

z = np.exp(-0.01 * ((x - x0) ** 2 + (y - y0) ** 2))
# 计算高斯分布的函数值，其中 x 和 y 是之前转换后的经度和纬度数组

triangles = np.asarray([
    [67, 66,  1], [65,  2, 66], [ 1, 66,  2], [64,  2, 65], [63,  3, 64],
    [60, 59, 57], [ 2, 64,  3], [ 3, 63,  4], [ 0, 67,  1], [62,  4, 63],
    [57, 59, 56], [59, 58, 56], [61, 60, 69], [57, 69, 60], [ 4, 62, 68],
    [ 6,  5,  9], [61, 68, 62], [69, 68, 61], [ 9,  5, 70], [ 6,  8,  7],
    [ 4, 70,  5], [ 8,  6,  9], [56, 69, 57], [69, 56, 52], [70, 10,  9],
    [54, 53, 55], [56, 55, 53], [68, 70,  4], [52, 56, 53], [11, 10, 12],
    [69, 71, 68], [68, 13, 70], [10, 70, 13], [51, 50, 52], [13, 68, 71],
    [52, 71, 69], [12, 10, 13], [71, 52, 50], [71, 14, 13], [50, 49, 71],
    [49, 48, 71], [14, 16, 15], [14, 71, 48], [17, 19, 18], [17, 20, 19],
    [48, 16, 14], [48, 47, 16], [47, 46, 16], [16, 46, 45], [23, 22, 24],
    [21, 24, 22], [17, 16, 45], [20, 17, 45], [21, 25, 24], [27, 26, 28],
    [20, 72, 21], [25, 21, 72], [45, 72, 20], [25, 28, 26], [44, 73, 45],
    [72, 45, 73], [28, 25, 29], [29, 25, 31], [43, 73, 44], [73, 43, 40],
    [72, 73, 39], [72, 31, 25], [42, 40, 43], [31, 30, 29], [39, 73, 40],
    [42, 41, 40], [72, 33, 31], [32, 31, 33], [39, 38, 72], [33, 72, 38],
    [33, 38, 34], [37, 35, 38], [34, 38, 35], [35, 37, 36]])
# 定义一个包含所有三角形顶点索引的数组，用于绘制三角形图形

# %%
# 不需创建 Triangulation 对象，直接将 x、y 和 triangles 数组传递给 tripcolor 函数即可。
# 如果同一 triangulation 被多次使用，使用 Triangulation 对象能够节省重复计算。

fig4, ax4 = plt.subplots()
# 创建一个新的图形和轴对象

ax4.set_aspect('equal')
# 设置坐标轴纵横比为相等，以避免图形被拉伸

tcf = ax4.tricontourf(x, y, triangles, z)
# 在坐标轴上绘制三角形区域的填充等高线图，使用 x、y、triangles 和 z 数组

fig4.colorbar(tcf)
# 在图形上添加颜色条

ax4.set_title('Contour plot of user-specified triangulation')
# 设置图形的标题

ax4.set_xlabel('Longitude (degrees)')
# 设置 x 轴标签

ax4.set_ylabel('Latitude (degrees)')
# 设置 y 轴标签

plt.show()
# 显示图形

# %%
#
# .. admonition:: References
#
#    本示例展示了以下函数、方法、类和模块的使用:
#
#    - `matplotlib.axes.Axes.tricontourf` / `matplotlib.pyplot.tricontourf`
#    - `matplotlib.tri.Triangulation`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.axes.Axes.legend` / `matplotlib.pyplot.legend`
#    - `matplotlib.contour.ContourSet.legend_elements`
```
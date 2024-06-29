# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\contourf_demo.py`

```py
"""
=============
Contourf demo
=============

How to use the `.axes.Axes.contourf` method to create filled contour plots.
"""
# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块并命名为 plt
import numpy as np  # 导入 numpy 库并命名为 np

# 设置步长
delta = 0.025

# 生成一维数组 x 和 y，范围为 -3.0 到 3.0，步长为 delta
x = y = np.arange(-3.0, 3.01, delta)

# 生成二维数组 X 和 Y，网格化 x 和 y
X, Y = np.meshgrid(x, y)

# 计算高斯分布函数生成 Z1 和 Z2，然后计算 Z
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

# 获取 Z 的行数和列数
nr, nc = Z.shape

# 在 Z 的一个角落放置 NaN（非数字）
Z[-nr // 6:, -nc // 6:] = np.nan
# contourf 方法会将这些值转换为掩码（masked）

# 将 Z 转换为掩码数组
Z = np.ma.array(Z)

# 在另一个角落掩盖一部分数据
Z[:nr // 6, :nc // 6] = np.ma.masked

# 在中间画一个圆形并掩盖其内部数据
interior = np.sqrt(X**2 + Y**2) < 0.5
Z[interior] = np.ma.masked

# %%
# Automatic contour levels
# ------------------------
# 使用自动选择的等高线水平；通常不是一个好主意，因为它们不会落在良好的边界上，但在这里仅用于演示目的。

# 创建一个新的图形和轴对象
fig1, ax2 = plt.subplots(layout='constrained')

# 绘制填充等高线图，并返回 ContourSet 对象 CS
CS = ax2.contourf(X, Y, Z, 10, cmap=plt.cm.bone)

# 创建另一组等高线，使用 CS 的一部分等高线水平和红色线条
CS2 = ax2.contour(CS, levels=CS.levels[::2], colors='r')

# 设置图标题和轴标签
ax2.set_title('Nonsense (3 masked regions)')
ax2.set_xlabel('word length anomaly')
ax2.set_ylabel('sentence length anomaly')

# 创建色标，并设置其标签
cbar = fig1.colorbar(CS)
cbar.ax.set_ylabel('verbosity coefficient')
# 将等高线水平添加到色标上
cbar.add_lines(CS2)

# %%
# Explicit contour levels
# -----------------------
# 使用指定的等高线水平和自动生成的颜色映射绘制等高线图。

# 创建一个新的图形和轴对象
fig2, ax2 = plt.subplots(layout='constrained')

# 指定等高线的水平和颜色，同时设置颜色映射的扩展性
levels = [-1.5, -1, -0.5, 0, 0.5, 1]
CS3 = ax2.contourf(X, Y, Z, levels, colors=('r', 'g', 'b'), extend='both')
# 将数据范围扩展到等高线水平之外的颜色设置为黄色和青色
CS3.cmap.set_under('yellow')
CS3.cmap.set_over('cyan')

# 绘制另一组等高线并设置其颜色和线宽
CS4 = ax2.contour(X, Y, Z, levels, colors=('k',), linewidths=(3,))
ax2.set_title('Listed colors (3 masked regions)')
ax2.clabel(CS4, fmt='%2.1f', colors='w', fontsize=14)

# 色标从 ContourSet 对象 CS3 获取所需的信息
fig2.colorbar(CS3)

# %%
# Extension settings
# ------------------
# 展示所有 4 种可能的 "extend" 设置：

# 扩展颜色映射并创建新的图形和轴对象
extends = ["neither", "both", "min", "max"]
cmap = plt.colormaps["winter"].with_extremes(under="magenta", over="yellow")
# 注意：等高线方法会简单地排除掩码或 NaN 区域，所以不会使用 "bad" 颜色映射值，而是在这些区域中不绘制任何内容。
# 因此，下面的操作将没有效果：
# cmap.set_bad("red")

fig, axs = plt.subplots(2, 2, layout="constrained")
# 使用 zip 函数同时迭代 axs.flat 和 extends 中的元素，ax 表示当前子图对象，extend 表示当前的 extend 参数值
for ax, extend in zip(axs.flat, extends):
    # 在当前子图上绘制填充的等高线图，使用给定的 levels、cmap 和 extend 参数
    cs = ax.contourf(X, Y, Z, levels, cmap=cmap, extend=extend)
    # 在当前子图上添加颜色条，与当前子图关联，设置颜色条的大小为 0.9 倍
    fig.colorbar(cs, ax=ax, shrink=0.9)
    # 设置当前子图的标题，展示当前的 extend 参数值
    ax.set_title("extend = %s" % extend)
    # 设置当前子图的坐标轴定位参数，每个轴上的刻度数为 4
    ax.locator_params(nbins=4)

# 显示绘制好的所有子图
plt.show()

# %%
# 使用 origin 关键字来定向等高线图的数据
# ---------------------------------------------
# 这段代码演示了如何使用 "origin" 关键字来定向等高线图的数据

# 创建数据
x = np.arange(1, 10)
y = x.reshape(-1, 1)
h = x * y

# 创建包含两个子图的图形窗口
fig, (ax1, ax2) = plt.subplots(ncols=2)

# 设置第一个子图的标题，指定 origin='upper'
ax1.set_title("origin='upper'")
# 在第一个子图上绘制填充的等高线图，指定 levels 和 extend 参数，使用 origin='upper'
ax1.contourf(h, levels=np.arange(5, 70, 5), extend='both', origin="upper")

# 设置第二个子图的标题，指定 origin='lower'
ax2.set_title("origin='lower'")
# 在第二个子图上绘制填充的等高线图，指定 levels 和 extend 参数，使用 origin='lower'
ax2.contourf(h, levels=np.arange(5, 70, 5), extend='both', origin="lower")

# 显示绘制好的图形窗口
plt.show()

# %%
#
# .. admonition:: References
#
#    本示例展示了以下函数、方法、类和模块的使用:
#
#    - `matplotlib.axes.Axes.contour` / `matplotlib.pyplot.contour`
#    - `matplotlib.axes.Axes.contourf` / `matplotlib.pyplot.contourf`
#    - `matplotlib.axes.Axes.clabel` / `matplotlib.pyplot.clabel`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors.Colormap`
#    - `matplotlib.colors.Colormap.set_bad`
#    - `matplotlib.colors.Colormap.set_under`
#    - `matplotlib.colors.Colormap.set_over`
```
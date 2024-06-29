# `D:\src\scipysrc\matplotlib\galleries\examples\showcase\anatomy.py`

```
"""
===================
Anatomy of a figure
===================

This figure shows the name of several matplotlib elements composing a figure
"""


import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import numpy as np  # 导入NumPy库

from matplotlib.patches import Circle  # 导入圆形图形类
from matplotlib.patheffects import withStroke  # 导入路径效果类
from matplotlib.ticker import AutoMinorLocator, MultipleLocator  # 导入刻度定位器类


royal_blue = [0, 20/256, 82/256]  # 定义皇家蓝的RGB颜色值


# make the figure

np.random.seed(19680801)  # 设定随机种子以确保结果可重复

X = np.linspace(0.5, 3.5, 100)  # 生成从0.5到3.5的100个等间隔数值
Y1 = 3 + np.cos(X)  # 计算Y1值
Y2 = 1 + np.cos(1 + X / 0.75) / 2  # 计算Y2值
Y3 = np.random.uniform(Y1, Y2, len(X))  # 生成Y1和Y2之间均匀分布的随机数

fig = plt.figure(figsize=(7.5, 7.5))  # 创建画布，并设定大小
ax = fig.add_axes([0.2, 0.17, 0.68, 0.7], aspect=1)  # 在画布上添加坐标轴，设定位置和纵横比

ax.xaxis.set_major_locator(MultipleLocator(1.000))  # 设置x轴主刻度定位器为1的倍数
ax.xaxis.set_minor_locator(AutoMinorLocator(4))  # 设置x轴次刻度定位器为4个次刻度
ax.yaxis.set_major_locator(MultipleLocator(1.000))  # 设置y轴主刻度定位器为1的倍数
ax.yaxis.set_minor_locator(AutoMinorLocator(4))  # 设置y轴次刻度定位器为4个次刻度
ax.xaxis.set_minor_formatter("{x:.2f}")  # 设置x轴次刻度格式化字符串为两位小数

ax.set_xlim(0, 4)  # 设置x轴显示范围
ax.set_ylim(0, 4)  # 设置y轴显示范围

ax.tick_params(which='major', width=1.0, length=10, labelsize=14)  # 设置主刻度的刻度线参数和标签大小
ax.tick_params(which='minor', width=1.0, length=5, labelsize=10,  # 设置次刻度的刻度线参数、标签大小和颜色
               labelcolor='0.25')

ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)  # 绘制网格线，设定线型、线宽、颜色和绘制顺序

ax.plot(X, Y1, c='C0', lw=2.5, label="Blue signal", zorder=10)  # 绘制曲线图，设定颜色、线宽和标签，绘制顺序为10
ax.plot(X, Y2, c='C1', lw=2.5, label="Orange signal")  # 绘制曲线图，设定颜色和线宽
ax.plot(X[::3], Y3[::3], linewidth=0, markersize=9,  # 绘制散点图，设定标记类型、大小和边框颜色
        marker='s', markerfacecolor='none', markeredgecolor='C4',
        markeredgewidth=2.5)

ax.set_title("Anatomy of a figure", fontsize=20, verticalalignment='bottom')  # 设置标题和字体大小
ax.set_xlabel("x Axis label", fontsize=14)  # 设置x轴标签和字体大小
ax.set_ylabel("y Axis label", fontsize=14)  # 设置y轴标签和字体大小
ax.legend(loc="upper right", fontsize=14)  # 添加图例，设定位置和字体大小


# Annotate the figure

def annotate(x, y, text, code):
    # Circle marker
    c = Circle((x, y), radius=0.15, clip_on=False, zorder=10, linewidth=2.5,  # 创建圆形标记，设定位置、半径、绘制顺序和线宽
               edgecolor=royal_blue + [0.6], facecolor='none',
               path_effects=[withStroke(linewidth=7, foreground='white')])
    ax.add_artist(c)  # 将圆形标记添加到图上

    # use path_effects as a background for the texts
    # draw the path_effects and the colored text separately so that the
    # path_effects cannot clip other texts
    for path_effects in [[withStroke(linewidth=7, foreground='white')], []]:
        color = 'white' if path_effects else royal_blue  # 根据是否有路径效果设定文字颜色
        ax.text(x, y - 0.2, text, zorder=100,  # 添加文本注释，设定位置、绘制顺序、水平对齐方式和字体样式
                ha='center', va='top', weight='bold', color=color,
                style='italic', fontfamily='monospace',
                path_effects=path_effects)

        color = 'white' if path_effects else 'black'  # 根据是否有路径效果设定文字颜色
        ax.text(x, y - 0.33, code, zorder=100,  # 添加文本注释，设定位置、绘制顺序、水平对齐方式和字体样式
                ha='center', va='top', weight='normal', color=color,
                fontfamily='monospace', fontsize='medium',
                path_effects=path_effects)


annotate(3.5, -0.13, "Minor tick label", "ax.xaxis.set_minor_formatter")  # 添加注释，设定位置、文本内容和相关代码
annotate(-0.03, 1.0, "Major tick", "ax.yaxis.set_major_locator")  # 添加注释，设定位置、文本内容和相关代码
annotate(0.00, 3.75, "Minor tick", "ax.yaxis.set_minor_locator")  # 添加注释，设定位置、文本内容和相关代码
annotate(-0.15, 3.00, "Major tick label", "ax.yaxis.set_major_formatter")  # 添加注释，设定位置、文本内容和相关代码
# 调用 annotate 函数，在图形上添加 x 轴标签
annotate(1.68, -0.39, "xlabel", "ax.set_xlabel")

# 调用 annotate 函数，在图形上添加 y 轴标签
annotate(-0.38, 1.67, "ylabel", "ax.set_ylabel")

# 调用 annotate 函数，在图形上添加标题
annotate(1.52, 4.15, "Title", "ax.set_title")

# 调用 annotate 函数，在图形上添加线条绘制
annotate(1.75, 2.80, "Line", "ax.plot")

# 调用 annotate 函数，在图形上添加散点图标记
annotate(2.25, 1.54, "Markers", "ax.scatter")

# 调用 annotate 函数，在图形上添加网格
annotate(3.00, 3.00, "Grid", "ax.grid")

# 调用 annotate 函数，在图形上添加图例
annotate(3.60, 3.58, "Legend", "ax.legend")

# 调用 annotate 函数，在图形上添加子图
annotate(2.5, 0.55, "Axes", "fig.subplots")

# 调用 annotate 函数，在图形上添加新的图形
annotate(4, 4.5, "Figure", "plt.figure")

# 调用 annotate 函数，在图形的 x 轴上做标记
annotate(0.65, 0.01, "x Axis", "ax.xaxis")

# 调用 annotate 函数，在图形的 y 轴上做标记
annotate(0, 0.36, "y Axis", "ax.yaxis")

# 调用 annotate 函数，在图形的脊柱上添加标记
annotate(4.0, 0.7, "Spine", "ax.spines")

# 设置图形的边框样式和颜色
fig.patch.set(linewidth=4, edgecolor='0.5')

# 显示图形
plt.show()
```
# `D:\src\scipysrc\matplotlib\galleries\examples\showcase\integral.py`

```py
"""
==================================
Integral as the area under a curve
==================================

Although this is a simple example, it demonstrates some important tweaks:

* A simple line plot with custom color and line width.
* A shaded region created using a Polygon patch.
* A text label with mathtext rendering.
* figtext calls to label the x- and y-axes.
* Use of axis spines to hide the top and right spines.
* Custom tick placement and labels.
"""
import matplotlib.pyplot as plt  # 导入 matplotlib 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from matplotlib.patches import Polygon  # 从 matplotlib.patches 导入 Polygon 类


def func(x):
    return (x - 3) * (x - 5) * (x - 7) + 85  # 定义函数 func(x)，计算给定 x 的函数值


a, b = 2, 9  # 设置积分的上下限
x = np.linspace(0, 10)  # 在 [0, 10] 区间均匀采样点，用于绘图
y = func(x)  # 计算函数在采样点上的取值

fig, ax = plt.subplots()  # 创建一个图形窗口和一个坐标轴

ax.plot(x, y, 'r', linewidth=2)  # 在坐标轴上绘制函数图像，红色线，线宽为 2
ax.set_ylim(bottom=0)  # 设置 y 轴的下限为 0

# 创建阴影区域
ix = np.linspace(a, b)  # 在积分区间 [a, b] 上均匀采样点
iy = func(ix)  # 计算积分区间上函数的取值
verts = [(a, 0), *zip(ix, iy), (b, 0)]  # 构造多边形的顶点坐标
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')  # 创建 Polygon 对象，设置填充色和边框色
ax.add_patch(poly)  # 将多边形添加到坐标轴上

ax.text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$",
        horizontalalignment='center', fontsize=20)  # 在坐标轴上添加数学公式文本

fig.text(0.9, 0.05, '$x$')  # 在图形窗口底部中心添加 x 轴标签
fig.text(0.1, 0.9, '$y$')  # 在图形窗口左侧中心添加 y 轴标签

ax.spines[['top', 'right']].set_visible(False)  # 隐藏坐标轴右侧和顶部的边框线
ax.set_xticks([a, b], labels=['$a$', '$b$'])  # 设置 x 轴刻度位置和标签
ax.set_yticks([])  # 清空 y 轴刻度

plt.show()  # 显示图形
```
# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\mixed_subplots.py`

```py
"""
=============================
2D and 3D Axes in same figure
=============================

This example shows a how to plot a 2D and a 3D plot on the same figure.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块并简写为 plt
import numpy as np  # 导入 numpy 并简写为 np


def f(t):
    """
    定义一个函数 f(t)，计算并返回 cos(2*pi*t) * exp(-t)
    """
    return np.cos(2*np.pi*t) * np.exp(-t)


# 设置一个图形，高度是宽度的两倍
fig = plt.figure(figsize=plt.figaspect(2.))
fig.suptitle('A tale of 2 subplots')  # 设置图形的标题

# 第一个子图
ax = fig.add_subplot(2, 1, 1)  # 添加一个2行1列的子图，当前是第1个子图

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
t3 = np.arange(0.0, 2.0, 0.01)

# 在第一个子图上画图，分别用蓝色圆点和黑色虚线表示
ax.plot(t1, f(t1), 'bo',
        t2, f(t2), 'k--', markerfacecolor='green')
ax.grid(True)  # 打开网格
ax.set_ylabel('Damped oscillation')  # 设置y轴标签

# 第二个子图
ax = fig.add_subplot(2, 1, 2, projection='3d')  # 添加一个2行1列的子图，当前是第2个子图，并设置为3D投影

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# 在第二个子图上绘制三维表面图
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1, 1)  # 设置z轴的范围

plt.show()  # 显示图形
```
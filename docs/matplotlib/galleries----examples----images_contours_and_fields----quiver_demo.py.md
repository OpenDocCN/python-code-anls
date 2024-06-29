# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\quiver_demo.py`

```
"""
=======================================
Advanced quiver and quiverkey functions
=======================================

Demonstrates some more advanced options for `~.axes.Axes.quiver`.  For a simple
example refer to :doc:`/gallery/images_contours_and_fields/quiver_simple_demo`.

Note: The plot autoscaling does not take into account the arrows, so
those on the boundaries may reach out of the picture.  This is not an easy
problem to solve in a perfectly general way.  The recommended workaround is to
manually set the Axes limits in such a case.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块并命名为 plt
import numpy as np  # 导入 numpy 库并命名为 np

# 创建网格数据
X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
U = np.cos(X)  # 根据 X 计算 cos(X) 并赋值给 U
V = np.sin(Y)  # 根据 Y 计算 sin(Y) 并赋值给 V

# %%

# 创建第一个子图
fig1, ax1 = plt.subplots()  # 创建一个图形窗口和一个子图，分别赋值给 fig1 和 ax1
ax1.set_title('Arrows scale with plot width, not view')  # 设置子图 ax1 的标题
Q = ax1.quiver(X, Y, U, V, units='width')  # 在 ax1 上绘制矢量场，使用 'width' 单位
qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')  # 在 ax1 上添加矢量场的键值

# %%

# 创建第二个子图
fig2, ax2 = plt.subplots()  # 创建一个新的图形窗口和一个子图，分别赋值给 fig2 和 ax2
ax2.set_title("pivot='mid'; every third arrow; units='inches'")  # 设置子图 ax2 的标题
Q = ax2.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
               pivot='mid', units='inches')  # 在 ax2 上绘制矢量场，设置部分参数
qk = ax2.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')  # 在 ax2 上添加矢量场的键值
ax2.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)  # 在 ax2 上绘制散点图

# %%

# 创建第三个子图
# sphinx_gallery_thumbnail_number = 3
fig3, ax3 = plt.subplots()  # 创建一个新的图形窗口和一个子图，分别赋值给 fig3 和 ax3
ax3.set_title("pivot='tip'; scales with x view")  # 设置子图 ax3 的标题
M = np.hypot(U, V)  # 计算 U 和 V 的模长并赋值给 M
Q = ax3.quiver(X, Y, U, V, M, units='x', pivot='tip', width=0.022,
               scale=1 / 0.15)  # 在 ax3 上绘制矢量场，设置部分参数
qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')  # 在 ax3 上添加矢量场的键值
ax3.scatter(X, Y, color='0.5', s=1)  # 在 ax3 上绘制散点图

plt.show()  # 显示所有图形窗口中的子图

# %%

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.quiver` / `matplotlib.pyplot.quiver`
#    - `matplotlib.axes.Axes.quiverkey` / `matplotlib.pyplot.quiverkey`
```
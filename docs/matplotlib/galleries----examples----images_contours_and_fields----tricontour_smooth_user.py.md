# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\tricontour_smooth_user.py`

```py
"""
======================
Tricontour Smooth User
======================

Demonstrates high-resolution tricontouring on user-defined triangular grids
with `matplotlib.tri.UniformTriRefiner`.
"""

# 导入 matplotlib 库中的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np，用于数值计算
import numpy as np
# 导入 matplotlib 库中的 tri 模块，用于处理三角网格
import matplotlib.tri as tri


# ----------------------------------------------------------------------------
# Analytical test function
# ----------------------------------------------------------------------------
# 定义一个分析测试函数，根据输入的 x 和 y 计算出 z 值
def function_z(x, y):
    # 计算两个圆的距离和角度
    r1 = np.sqrt((0.5 - x)**2 + (0.5 - y)**2)
    theta1 = np.arctan2(0.5 - x, 0.5 - y)
    r2 = np.sqrt((-x - 0.2)**2 + (-y - 0.2)**2)
    theta2 = np.arctan2(-x - 0.2, -y - 0.2)
    # 计算 z 值，组合两个圆的效果和椭圆的效果
    z = -(2 * (np.exp((r1 / 10)**2) - 1) * 30. * np.cos(7. * theta1) +
          (np.exp((r2 / 10)**2) - 1) * 30. * np.cos(11. * theta2) +
          0.7 * (x**2 + y**2))
    # 返回 z 值，进行归一化处理
    return (np.max(z) - z) / (np.max(z) - np.min(z))

# ----------------------------------------------------------------------------
# Creating a Triangulation
# ----------------------------------------------------------------------------
# 设置点的数量和范围
n_angles = 20
n_radii = 10
min_radius = 0.15
# 创建半径数组
radii = np.linspace(min_radius, 0.95, n_radii)
# 创建角度数组
angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += np.pi / n_angles

# 计算 x, y, z 坐标
x = (radii * np.cos(angles)).flatten()
y = (radii * np.sin(angles)).flatten()
z = function_z(x, y)

# 创建 Triangulation 对象
triang = tri.Triangulation(x, y)

# Mask off unwanted triangles.
# 设置一个条件，用于过滤不需要的三角形
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                         y[triang.triangles].mean(axis=1))
                < min_radius)

# ----------------------------------------------------------------------------
# Refine data
# ----------------------------------------------------------------------------
# 创建 TriRefiner 对象，用于细化三角形网格
refiner = tri.UniformTriRefiner(triang)
# 对 z 数据进行细化处理
tri_refi, z_test_refi = refiner.refine_field(z, subdiv=3)

# ----------------------------------------------------------------------------
# Plot the triangulation and the high-res iso-contours
# ----------------------------------------------------------------------------
# 创建图形和轴对象
fig, ax = plt.subplots()
ax.set_aspect('equal')
# 绘制原始的三角形轮廓
ax.triplot(triang, lw=0.5, color='white')

# 设置等高线的级别
levels = np.arange(0., 1., 0.025)
# 绘制填充的等高线图
ax.tricontourf(tri_refi, z_test_refi, levels=levels, cmap='terrain')
# 绘制等高线
ax.tricontour(tri_refi, z_test_refi, levels=levels,
              colors=['0.25', '0.5', '0.5', '0.5', '0.5'],
              linewidths=[1.0, 0.5, 0.5, 0.5, 0.5])

# 设置图形标题
ax.set_title("High-resolution tricontouring")

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.tricontour` / `matplotlib.pyplot.tricontour`
# 导入matplotlib库中的`Axes`类和`tricontourf`函数，用于绘制三角形网格上的填充等高线图
# 导入matplotlib库中的`pyplot`模块，也包含了`tricontourf`函数的定义
# 导入matplotlib库中的`tri`模块，提供了处理三角形网格的工具和类
# 导入`Triangulation`类，用于创建三角形网格的对象
# 导入`UniformTriRefiner`类，用于在给定的三角形网格上执行均匀细化操作
```
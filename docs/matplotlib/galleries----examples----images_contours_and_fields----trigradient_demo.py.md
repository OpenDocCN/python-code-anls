# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\trigradient_demo.py`

```py
"""
================
Trigradient Demo
================

Demonstrates computation of gradient with
`matplotlib.tri.CubicTriInterpolator`.
"""

# 导入必要的库
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from matplotlib.tri import (CubicTriInterpolator, Triangulation,
                            UniformTriRefiner)  # 导入 matplotlib 的 triangulation 相关模块

# ----------------------------------------------------------------------------
# Electrical potential of a dipole
# ----------------------------------------------------------------------------
def dipole_potential(x, y):
    """The electric dipole potential V, at position *x*, *y*."""
    r_sq = x**2 + y**2  # 计算位置 (x, y) 到原点的距离平方
    theta = np.arctan2(y, x)  # 计算位置 (x, y) 的极角
    z = np.cos(theta)/r_sq  # 计算电偶极子电势
    return (np.max(z) - z) / (np.max(z) - np.min(z))  # 返回标准化后的电势值

# ----------------------------------------------------------------------------
# Creating a Triangulation
# ----------------------------------------------------------------------------
# 定义用于创建三角剖分的点坐标
n_angles = 30
n_radii = 10
min_radius = 0.2
radii = np.linspace(min_radius, 0.95, n_radii)

angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
angles[:, 1::2] += np.pi / n_angles

x = (radii*np.cos(angles)).flatten()  # x 坐标
y = (radii*np.sin(angles)).flatten()  # y 坐标
V = dipole_potential(x, y)  # 计算每个点的电势值

# Create the Triangulation; no triangles specified so Delaunay triangulation
# created.
triang = Triangulation(x, y)  # 创建三角剖分对象

# Mask off unwanted triangles.
triang.set_mask(np.hypot(x[triang.triangles].mean(axis=1),
                         y[triang.triangles].mean(axis=1))
                < min_radius)  # 根据距离原点的平均距离掩盖不需要的三角形

# ----------------------------------------------------------------------------
# Refine data - interpolates the electrical potential V
# ----------------------------------------------------------------------------
refiner = UniformTriRefiner(triang)  # 创建三角剖分的均匀细化器
tri_refi, z_test_refi = refiner.refine_field(V, subdiv=3)  # 对电势值 V 进行细化处理

# ----------------------------------------------------------------------------
# Computes the electrical field (Ex, Ey) as gradient of electrical potential
# ----------------------------------------------------------------------------
tci = CubicTriInterpolator(triang, -V)  # 创建三角剖分的三次插值器，用于电势 V 的插值
# Gradient requested here at the mesh nodes but could be anywhere else:
(Ex, Ey) = tci.gradient(triang.x, triang.y)  # 计算电场的梯度分量 (Ex, Ey)
E_norm = np.sqrt(Ex**2 + Ey**2)  # 计算电场的模长

# ----------------------------------------------------------------------------
# Plot the triangulation, the potential iso-contours and the vector field
# ----------------------------------------------------------------------------
fig, ax = plt.subplots()  # 创建图形和轴对象
ax.set_aspect('equal')  # 设置图形纵横比
ax.use_sticky_edges = False  # 取消粘滞边缘
ax.margins(0.07)  # 设置边距

ax.triplot(triang, color='0.8')  # 绘制三角剖分图

levels = np.arange(0., 1., 0.01)  # 设定等势线的电势值范围
ax.tricontour(tri_refi, z_test_refi, levels=levels, cmap='hot',
              linewidths=[2.0, 1.0, 1.0, 1.0])  # 绘制等势线

# Plots direction of the electrical vector field
# 在三角形网格上绘制电场矢量场，箭头的起点坐标为(triang.x, triang.y)，矢量方向为(Ex/E_norm, Ey/E_norm)
# units='xy' 表示使用数据坐标系作为单位，scale=10. 表示箭头长度的缩放比例，zorder=3 表示绘图层级，color='blue' 表示箭头颜色
# width=0.007 表示箭头的宽度，headwidth=3. 表示箭头头部的宽度，headlength=4. 表示箭头头部的长度
ax.quiver(triang.x, triang.y, Ex/E_norm, Ey/E_norm,
          units='xy', scale=10., zorder=3, color='blue',
          width=0.007, headwidth=3., headlength=4.)

# 设置图表标题为'Gradient plot: an electrical dipole'
ax.set_title('Gradient plot: an electrical dipole')
# 显示图表
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.tricontour` / `matplotlib.pyplot.tricontour`
#    - `matplotlib.axes.Axes.triplot` / `matplotlib.pyplot.triplot`
#    - `matplotlib.tri`
#    - `matplotlib.tri.Triangulation`
#    - `matplotlib.tri.CubicTriInterpolator`
#    - `matplotlib.tri.CubicTriInterpolator.gradient`
#    - `matplotlib.tri.UniformTriRefiner`
#    - `matplotlib.axes.Axes.quiver` / `matplotlib.pyplot.quiver`
```
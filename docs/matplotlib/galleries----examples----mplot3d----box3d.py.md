# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\box3d.py`

```py
"""
===================
3D box surface plot
===================

给定三维网格体积数据 ``X``, ``Y``, ``Z``，此示例绘制了体积表面上的数据值。

策略是选择每个表面的数据，并使用 `.axes3d.Axes3D.contourf` 单独绘制轮廓，使用合适的参数 *zdir* 和 *offset*。
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库
import numpy as np  # 导入 numpy 库

# 定义维度
Nx, Ny, Nz = 100, 300, 500  # 定义 X, Y, Z 的维度
X, Y, Z = np.meshgrid(np.arange(Nx), np.arange(Ny), -np.arange(Nz))  # 创建网格数据 X, Y, Z

# 创建虚假数据
data = (((X+100)**2 + (Y-20)**2 + 2*Z)/1000+1)

kw = {
    'vmin': data.min(),  # 设置最小值
    'vmax': data.max(),  # 设置最大值
    'levels': np.linspace(data.min(), data.max(), 10),  # 设置轮廓级数
}

# 创建一个带有 3D 坐标轴的图形
fig = plt.figure(figsize=(5, 4))  # 创建图形对象
ax = fig.add_subplot(111, projection='3d')  # 添加 3D 坐标轴子图

# 绘制轮廓表面
_ = ax.contourf(
    X[:, :, 0], Y[:, :, 0], data[:, :, 0],  # 在 z = 0 平面上绘制轮廓
    zdir='z', offset=0, **kw
)
_ = ax.contourf(
    X[0, :, :], data[0, :, :], Z[0, :, :],  # 在 y = 0 平面上绘制轮廓
    zdir='y', offset=0, **kw
)
C = ax.contourf(
    data[:, -1, :], Y[:, -1, :], Z[:, -1, :],  # 在 x = X.max() 平面上绘制轮廓
    zdir='x', offset=X.max(), **kw
)

# 设置图形的坐标范围
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

# 绘制边缘线
edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)  # 绘制 x 边缘线
ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)  # 绘制 y 边缘线
ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)  # 绘制 z 边缘线

# 设置标签和 z 轴刻度
ax.set(
    xlabel='X [km]',  # 设置 x 轴标签
    ylabel='Y [km]',  # 设置 y 轴标签
    zlabel='Z [m]',   # 设置 z 轴标签
    zticks=[0, -150, -300, -450],  # 设置 z 轴刻度
)

# 设置缩放和视角
ax.view_init(40, -30, 0)  # 设置视角
ax.set_box_aspect(None, zoom=0.9)  # 设置盒子长宽比

# 添加颜色条
fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label='Name [units]')  # 添加颜色条

# 显示图形
plt.show()
```
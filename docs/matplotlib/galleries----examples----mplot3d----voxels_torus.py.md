# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\voxels_torus.py`

```py
"""
=======================================================
3D voxel / volumetric plot with cylindrical coordinates
=======================================================

Demonstrates using the *x*, *y*, *z* parameters of `.Axes3D.voxels`.
"""

# 导入需要的库
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors

# 定义函数，用于计算坐标轴中间点
def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

# 准备一些坐标，并为每个坐标附加 RGB 值
r, theta, z = np.mgrid[0:1:11j, 0:np.pi*2:25j, -0.5:0.5:11j]
x = r*np.cos(theta)
y = r*np.sin(theta)

# 计算中间点的坐标
rc, thetac, zc = midpoints(r), midpoints(theta), midpoints(z)

# 定义一个不规则的环面团，位于 [0.7, *, 0] 附近
sphere = (rc - 0.7)**2 + (zc + 0.2*np.cos(thetac*2))**2 < 0.2**2

# 合并颜色组件
hsv = np.zeros(sphere.shape + (3,))
hsv[..., 0] = thetac / (np.pi*2)
hsv[..., 1] = rc
hsv[..., 2] = zc + 0.5
colors = matplotlib.colors.hsv_to_rgb(hsv)

# 创建 3D 图形并添加子图
ax = plt.figure().add_subplot(projection='3d')

# 使用 voxels 方法绘制体素图
ax.voxels(x, y, z, sphere,
          facecolors=colors,  # 设置面颜色
          edgecolors=np.clip(2*colors - 0.5, 0, 1),  # 设置边颜色，使之更亮
          linewidth=0.5)  # 设置边宽

# 显示图形
plt.show()
```
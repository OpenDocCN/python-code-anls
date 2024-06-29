# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\trisurf3d_simple.py`

```py
"""
=====================
plot_trisurf(x, y, z)
=====================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf`.
"""
# 导入 matplotlib.pyplot 库并命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库并命名为 np
import numpy as np

# 从 matplotlib 库中导入 colormap 模块
from matplotlib import cm

# 使用 '_mpl-gallery' 风格
plt.style.use('_mpl-gallery')

# 设置半径和角度数量
n_radii = 8
n_angles = 36

# 生成半径和角度的序列
radii = np.linspace(0.125, 1.0, n_radii)
angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)[..., np.newaxis]

# 将极坐标 (半径, 角度) 转换为笛卡尔坐标 (x, y)
x = np.append(0, (radii*np.cos(angles)).flatten())
y = np.append(0, (radii*np.sin(angles)).flatten())
z = np.sin(-x*y)

# 创建 3D 图形和轴
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
# 绘制三角面图
ax.plot_trisurf(x, y, z, vmin=z.min() * 2, cmap=cm.Blues)

# 设置坐标轴标签不显示
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

# 显示图形
plt.show()
```
# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\surface3d_simple.py`

```
"""
=====================
plot_surface(X, Y, Z)
=====================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.plot_surface`.
"""

# 导入 matplotlib 的 pyplot 模块，并使用 plt 作为别名
import matplotlib.pyplot as plt
# 导入 numpy 库，并使用 np 作为别名
import numpy as np
# 从 matplotlib 中导入 cm 模块，用于颜色映射
from matplotlib import cm

# 使用 '_mpl-gallery' 样式
plt.style.use('_mpl-gallery')

# 创建数据
# 生成 X 轴的数据，范围从 -5 到 5，步长为 0.25
X = np.arange(-5, 5, 0.25)
# 生成 Y 轴的数据，范围从 -5 到 5，步长为 0.25
Y = np.arange(-5, 5, 0.25)
# 将 X 和 Y 转换为二维网格
X, Y = np.meshgrid(X, Y)
# 计算 R，即到原点的距离
R = np.sqrt(X**2 + Y**2)
# 计算 Z，使用 sin 函数生成高度数据
Z = np.sin(R)

# 绘制三维曲面图
# 创建图形和坐标轴对象
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# 绘制曲面，设置最小值为 Z 值的最小值乘以 2，使用 Blues 颜色映射
ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)

# 设置坐标轴标签不显示
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

# 显示图形
plt.show()
```
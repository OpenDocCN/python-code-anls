# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\subplot3d.py`

```py
"""
====================
3D plots as subplots
====================

Demonstrate including 3D plots as subplots.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 数学库

from matplotlib import cm  # 导入 colormap
from mpl_toolkits.mplot3d.axes3d import get_test_data  # 导入测试数据生成函数

# 设置一个宽度是高度两倍的图形
fig = plt.figure(figsize=plt.figaspect(0.5))

# =============
# First subplot
# =============
# 添加第一个子图的坐标轴，使用 3D 投影
ax = fig.add_subplot(1, 2, 1, projection='3d')

# 绘制一个三维表面，类似于 mplot3d/surface3d_demo 中的示例
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_zlim(-1.01, 1.01)  # 设置 Z 轴的范围
fig.colorbar(surf, shrink=0.5, aspect=10)  # 添加颜色条

# ==============
# Second subplot
# ==============
# 添加第二个子图的坐标轴，使用 3D 投影
ax = fig.add_subplot(1, 2, 2, projection='3d')

# 绘制一个三维线框图，类似于 mplot3d/wire3d_demo 中的示例
X, Y, Z = get_test_data(0.05)
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()  # 显示图形
```
# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\fillbetween3d.py`

```
"""
=====================
Fill between 3D lines
=====================

Demonstrate how to fill the space between 3D lines with surfaces. Here we
create a sort of "lampshade" shape.
"""

# 导入 matplotlib.pyplot 库，并重命名为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并重命名为 np
import numpy as np

# 设置数据点数量
N = 50
# 在 0 到 2π 之间生成均匀分布的角度值
theta = np.linspace(0, 2*np.pi, N)

# 第一条曲线的 x, y, z 值
x1 = np.cos(theta)
y1 = np.sin(theta)
z1 = 0.1 * np.sin(6 * theta)

# 第二条曲线的 x, y, z 值
x2 = 0.6 * np.cos(theta)
y2 = 0.6 * np.sin(theta)
z2 = 2  # 注意，这里的 z2 是一个标量，可以用于曲线之间的填充

# 创建一个新的图形窗口
fig = plt.figure()
# 添加一个 3D 坐标系子图
ax = fig.add_subplot(projection='3d')
# 在两条曲线之间填充曲面，设置透明度为 0.5，边缘颜色为黑色
ax.fill_between(x1, y1, z1, x2, y2, z2, alpha=0.5, edgecolor='k')

# 显示图形
plt.show()
```
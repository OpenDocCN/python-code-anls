# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\quiver3d_simple.py`

```py
"""
========================
quiver(X, Y, Z, U, V, W)
========================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.quiver`.
"""

# 导入 matplotlib.pyplot 库，简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，简写为 np
import numpy as np

# 使用 '_mpl-gallery' 风格样式
plt.style.use('_mpl-gallery')

# 定义数据
n = 4
# 在 [-1, 1] 区间内生成包含 n 个元素的均匀间隔的数组
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
z = np.linspace(-1, 1, n)
# 生成网格
X, Y, Z = np.meshgrid(x, y, z)
# 定义向量场的三个分量
U = (X + Y) / 5
V = (Y - X) / 5
W = Z * 0

# 创建一个包含单个子图的图形对象和子图对象
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# 绘制三维向量场
ax.quiver(X, Y, Z, U, V, W)

# 设置坐标轴标签不显示
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

# 显示图形
plt.show()
```
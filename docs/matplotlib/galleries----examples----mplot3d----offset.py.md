# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\offset.py`

```
"""
=========================
Automatic text offsetting
=========================

This example demonstrates mplot3d's offset text display.
As one rotates the 3D figure, the offsets should remain oriented the
same way as the axis label, and should also be located "away"
from the center of the plot.

This demo triggers the display of the offset text for the x- and
y-axis by adding 1e5 to X and Y. Anything less would not
automatically trigger it.
"""

# 导入 matplotlib 库中的 pyplot 模块，并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简写为 np
import numpy as np

# 创建一个 3D 图形的子图
ax = plt.figure().add_subplot(projection='3d')

# 使用 np.mgrid 创建 X 和 Y 的网格数据，范围分别为 0 到 6π 和 0 到 4π，步长为 0.25
X, Y = np.mgrid[0:6*np.pi:0.25, 0:4*np.pi:0.25]
# 计算 Z 值，根据 X 和 Y 的余弦值计算平方根
Z = np.sqrt(np.abs(np.cos(X) + np.cos(Y)))

# 在 3D 图形上绘制曲面，X 和 Y 各加上 1e5，使用 'autumn' 颜色映射，cstride 和 rstride 分别为 2
ax.plot_surface(X + 1e5, Y + 1e5, Z, cmap='autumn', cstride=2, rstride=2)

# 设置 X 轴标签
ax.set_xlabel("X label")
# 设置 Y 轴标签
ax.set_ylabel("Y label")
# 设置 Z 轴标签
ax.set_zlabel("Z label")
# 设置 Z 轴的显示范围为 0 到 2
ax.set_zlim(0, 2)

# 显示绘制的图形
plt.show()
```
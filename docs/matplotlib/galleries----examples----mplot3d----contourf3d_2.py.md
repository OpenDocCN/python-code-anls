# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\contourf3d_2.py`

```py
"""
===================================
将填充轮廓投影到图形上
===================================
演示如何在显示3D表面的同时，将填充轮廓“投影”到图形的“墙壁”上。
参见：doc:`contour3d_3`，以查看未填充版本。
"""

# 导入 matplotlib 的 pyplot 模块
import matplotlib.pyplot as plt

# 从 mpl_toolkits.mplot3d 模块导入 axes3d
from mpl_toolkits.mplot3d import axes3d

# 创建一个带有3D投影的子图
ax = plt.figure().add_subplot(projection='3d')

# 获取测试数据 X, Y, Z
X, Y, Z = axes3d.get_test_data(0.05)

# 绘制3D表面
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)

# 在每个维度上绘制轮廓的投影。
# 通过选择与相应轴限制匹配的偏移量，投影的轮廓将位于图形的“墙壁”上。
ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

# 设置坐标轴的限制和标签
ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
       xlabel='X', ylabel='Y', zlabel='Z')

# 显示图形
plt.show()
```
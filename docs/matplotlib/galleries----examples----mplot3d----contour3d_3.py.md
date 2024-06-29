# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\contour3d_3.py`

```
"""
=====================================
Project contour profiles onto a graph
=====================================
Demonstrates displaying a 3D surface while also projecting contour 'profiles'
onto the 'walls' of the graph.
See :doc:`contourf3d_2` for the filled version.
"""

# 导入matplotlib.pyplot库，用于绘图
import matplotlib.pyplot as plt

# 导入axes3d模块，用于3D图形的绘制
from mpl_toolkits.mplot3d import axes3d

# 创建一个带有3D投影的子图
ax = plt.figure().add_subplot(projection='3d')

# 获取测试数据，X、Y、Z是用于绘制的数据网格
X, Y, Z = axes3d.get_test_data(0.05)

# 绘制3D表面
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)

# 在每个维度上投影轮廓。通过选择与相应轴限制匹配的偏移量，
# 投影的轮廓将位于图形的‘墙壁’上。
ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

# 设置坐标轴的范围和标签
ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
       xlabel='X', ylabel='Y', zlabel='Z')

# 显示图形
plt.show()
```
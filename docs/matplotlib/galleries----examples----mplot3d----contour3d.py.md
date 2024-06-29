# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\contour3d.py`

```py
"""
=================================
Plot contour (level) curves in 3D
=================================

This is like a contour plot in 2D except that the ``f(x, y)=c`` curve is
plotted on the plane ``z=c``.
"""

# 导入需要的绘图库
import matplotlib.pyplot as plt

# 导入颜色映射和三维绘图工具
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

# 创建一个带有三维投影的图形子图
ax = plt.figure().add_subplot(projection='3d')

# 获取测试数据，包括 X, Y, Z 数据
X, Y, Z = axes3d.get_test_data(0.05)

# 绘制三维等高线曲线，使用 coolwarm 颜色映射
ax.contour(X, Y, Z, cmap=cm.coolwarm)  # 绘制等高线曲线

# 显示图形
plt.show()
```
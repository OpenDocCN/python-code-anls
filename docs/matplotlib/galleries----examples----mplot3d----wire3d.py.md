# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\wire3d.py`

```
"""
=================
3D wireframe plot
=================

A very basic demonstration of a wireframe plot.
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 从 mpl_toolkits.mplot3d 中导入 axes3d 模块，用于创建 3D 图形
from mpl_toolkits.mplot3d import axes3d

# 创建一个新的图形窗口
fig = plt.figure()

# 在图形窗口中添加一个 3D 子图
ax = fig.add_subplot(projection='3d')

# 使用 axes3d 模块提供的函数获取一些测试数据 X, Y, Z
X, Y, Z = axes3d.get_test_data(0.05)

# 绘制一个基本的线框图
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

# 显示图形
plt.show()
```
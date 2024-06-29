# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\3d_bars.py`

```
"""
=====================
Demo of 3D bar charts
=====================

A basic demo of how to plot 3D bars with and without shading.
"""

# 导入 matplotlib.pyplot 作为 plt
import matplotlib.pyplot as plt
# 导入 numpy 并重命名为 np
import numpy as np

# 设置图形和子图
fig = plt.figure(figsize=(8, 3))
# 添加第一个子图，投影类型为 3D
ax1 = fig.add_subplot(121, projection='3d')
# 添加第二个子图，投影类型为 3D
ax2 = fig.add_subplot(122, projection='3d')

# 虚拟数据
_x = np.arange(4)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

# 定义柱状图的顶部、底部、宽度和深度
top = x + y
bottom = np.zeros_like(top)
width = depth = 1

# 在第一个子图中绘制带阴影的 3D 柱状图
ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
# 设置第一个子图的标题
ax1.set_title('Shaded')

# 在第二个子图中绘制不带阴影的 3D 柱状图
ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
# 设置第二个子图的标题
ax2.set_title('Not Shaded')

# 显示图形
plt.show()
```
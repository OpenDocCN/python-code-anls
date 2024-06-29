# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\text3d.py`

```py
"""
======================
Text annotations in 3D
======================

Demonstrates the placement of text annotations on a 3D plot.

Functionality shown:

- Using the `~.Axes3D.text` function with three types of *zdir* values: None,
  an axis name (ex. 'x'), or a direction tuple (ex. (1, 1, 0)).
- Using the `~.Axes3D.text` function with the color keyword.
- Using the `.text2D` function to place text on a fixed position on the ax
  object.
"""

import matplotlib.pyplot as plt

# 创建一个带有3D投影的子图
ax = plt.figure().add_subplot(projection='3d')

# Demo 1: zdir
# 定义展示用的zdir参数和对应的坐标
zdirs = (None, 'x', 'y', 'z', (1, 1, 0), (1, 1, 1))
xs = (1, 4, 4, 9, 4, 1)
ys = (2, 5, 8, 10, 1, 2)
zs = (10, 3, 8, 9, 1, 8)

# 遍历各个zdir参数，并在相应的位置添加文本标签
for zdir, x, y, z in zip(zdirs, xs, ys, zs):
    label = '(%d, %d, %d), dir=%s' % (x, y, z, zdir)
    ax.text(x, y, z, label, zdir)

# Demo 2: color
# 在指定位置添加红色文本标签
ax.text(9, 0, 0, "red", color='red')

# Demo 3: text2D
# 在ax对象上固定位置添加文本标签
# 0.05, 0.95是相对于ax对象大小的位置坐标
ax.text2D(0.05, 0.95, "2D Text", transform=ax.transAxes)

# 调整显示区域和标签
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形
plt.show()
```
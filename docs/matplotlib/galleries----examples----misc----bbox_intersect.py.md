# `D:\src\scipysrc\matplotlib\galleries\examples\misc\bbox_intersect.py`

```
"""
===========================================
Changing colors of lines intersecting a box
===========================================

The lines intersecting the rectangle are colored in red, while the others
are left as blue lines. This example showcases the `.intersects_bbox` function.

"""

# 导入需要的库
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path  # 导入 Path 类
from matplotlib.transforms import Bbox  # 导入 Bbox 类

# 设置随机种子以便结果可重现
np.random.seed(19680801)

# 定义矩形区域的左下角坐标和宽高
left, bottom, width, height = (-1, -1, 2, 2)

# 创建矩形对象，设置其样式为黑色半透明
rect = plt.Rectangle((left, bottom), width, height,
                     facecolor="black", alpha=0.1)

# 创建图形和轴对象
fig, ax = plt.subplots()
ax.add_patch(rect)  # 将矩形对象添加到轴上显示

# 根据定义的左下角坐标和宽高创建一个 Bbox 对象
bbox = Bbox.from_bounds(left, bottom, width, height)

# 循环生成12条随机线段
for i in range(12):
    # 生成随机的两个点作为线段的顶点坐标
    vertices = (np.random.random((2, 2)) - 0.5) * 6.0
    # 根据顶点坐标创建 Path 对象表示一条线段
    path = Path(vertices)
    # 判断当前线段是否与矩形框相交
    if path.intersects_bbox(bbox):
        color = 'r'  # 如果相交，将线段颜色设为红色
    else:
        color = 'b'  # 如果不相交，将线段颜色设为蓝色
    ax.plot(vertices[:, 0], vertices[:, 1], color=color)  # 绘制线段，并设置颜色

plt.show()  # 显示图形
```
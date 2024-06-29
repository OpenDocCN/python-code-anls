# `D:\src\scipysrc\matplotlib\galleries\examples\units\ellipse_with_units.py`

```py
"""
==================
Ellipse with units
==================

Compare the ellipse generated with arcs versus a polygonal approximation.

.. only:: builder_html

   This example requires :download:`basic_units.py <basic_units.py>`
"""

# 导入厘米单位
from basic_units import cm

# 导入绘图库
import matplotlib.pyplot as plt
import numpy as np

# 导入绘图中的图形补丁模块
from matplotlib import patches

# 定义椭圆的中心坐标和长短轴长度，以及旋转角度
xcenter, ycenter = 0.38 * cm, 0.52 * cm
width, height = 1e-1 * cm, 3e-1 * cm
angle = -30

# 创建角度范围数组
theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
x = 0.5 * width * np.cos(theta)   # 计算椭圆边界点的 x 坐标
y = 0.5 * height * np.sin(theta)  # 计算椭圆边界点的 y 坐标

# 计算旋转矩阵
rtheta = np.radians(angle)
R = np.array([
    [np.cos(rtheta), -np.sin(rtheta)],
    [np.sin(rtheta), np.cos(rtheta)],
])

# 应用旋转矩阵到椭圆边界点
x, y = np.dot(R, [x, y])
x += xcenter  # 平移 x 坐标
y += ycenter  # 平移 y 坐标

# 创建第一个图形
fig = plt.figure()
ax = fig.add_subplot(211, aspect='auto')  # 添加子图，设置宽高比自动调整
ax.fill(x, y, alpha=0.2, facecolor='yellow',
        edgecolor='yellow', linewidth=1, zorder=1)  # 填充椭圆区域

# 添加第一个椭圆到子图
e1 = patches.Ellipse((xcenter, ycenter), width, height,
                     angle=angle, linewidth=2, fill=False, zorder=2)
ax.add_patch(e1)

# 创建第二个图形
ax = fig.add_subplot(212, aspect='equal')  # 添加子图，设置宽高比相等
ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)  # 填充椭圆区域

# 添加第二个椭圆到子图
e2 = patches.Ellipse((xcenter, ycenter), width, height,
                     angle=angle, linewidth=2, fill=False, zorder=2)
ax.add_patch(e2)

# 将第一个图形保存为图片
fig.savefig('ellipse_compare')

# 创建新的图形
fig = plt.figure()
ax = fig.add_subplot(211, aspect='auto')  # 添加子图，设置宽高比自动调整
ax.fill(x, y, alpha=0.2, facecolor='yellow',
        edgecolor='yellow', linewidth=1, zorder=1)  # 填充椭圆区域

# 添加第一个弧形到子图
e1 = patches.Arc((xcenter, ycenter), width, height,
                 angle=angle, linewidth=2, fill=False, zorder=2)
ax.add_patch(e1)

# 创建第二个图形
ax = fig.add_subplot(212, aspect='equal')  # 添加子图，设置宽高比相等
ax.fill(x, y, alpha=0.2, facecolor='green', edgecolor='green', zorder=1)  # 填充椭圆区域

# 添加第二个弧形到子图
e2 = patches.Arc((xcenter, ycenter), width, height,
                 angle=angle, linewidth=2, fill=False, zorder=2)
ax.add_patch(e2)

# 将第二个图形保存为图片
fig.savefig('arc_compare')

# 显示所有图形
plt.show()
```
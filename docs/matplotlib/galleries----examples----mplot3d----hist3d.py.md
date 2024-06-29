# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\hist3d.py`

```py
"""
==============================
Create 3D histogram of 2D data
==============================

Demo of a histogram for 2D data as a bar graph in 3D.
"""

# 导入 matplotlib.pyplot 作为 plt，导入 numpy 库并重命名为 np
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子以确保结果可复现性
np.random.seed(19680801)

# 创建一个新的图形窗口
fig = plt.figure()

# 在图形窗口中添加一个 3D 子图
ax = fig.add_subplot(projection='3d')

# 生成两个随机数组 x 和 y，每个数组包含 100 个元素，取值范围在 [0, 4) 内
x, y = np.random.rand(2, 100) * 4

# 使用 np.histogram2d 函数计算二维直方图，返回直方图数组 hist 和两个边界数组 xedges, yedges
hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])

# 构造数组，用于存放 16 个柱子的锚定位置
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

# 构造数组，分别存放 16 个柱子的尺寸
dx = dy = 0.5 * np.ones_like(zpos)
dz = hist.ravel()

# 使用 ax.bar3d 方法绘制 3D 柱状图，设置柱子的位置、尺寸和高度数据
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')

# 显示图形
plt.show()
```
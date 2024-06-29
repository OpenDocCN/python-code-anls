# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\voxels.py`

```
"""
==========================
3D voxel / volumetric plot
==========================

Demonstrates plotting 3D volumetric objects with `.Axes3D.voxels`.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 库并使用别名 np

# 准备一些坐标
x, y, z = np.indices((8, 8, 8))

# 在左上角和右下角以及它们之间绘制立方体
cube1 = (x < 3) & (y < 3) & (z < 3)  # 第一个立方体的布尔数组
cube2 = (x >= 5) & (y >= 5) & (z >= 5)  # 第二个立方体的布尔数组
link = abs(x - y) + abs(y - z) + abs(z - x) <= 2  # 连接线的布尔数组

# 将这些对象组合成一个单一的布尔数组
voxelarray = cube1 | cube2 | link

# 设置每个对象的颜色
colors = np.empty(voxelarray.shape, dtype=object)
colors[link] = 'red'  # 设置连接线的颜色为红色
colors[cube1] = 'blue'  # 设置第一个立方体的颜色为蓝色
colors[cube2] = 'green'  # 设置第二个立方体的颜色为绿色

# 绘制所有内容
ax = plt.figure().add_subplot(projection='3d')  # 创建一个带有 3D 投影的子图
ax.voxels(voxelarray, facecolors=colors, edgecolor='k')  # 使用布尔数组和颜色数组绘制体素图

plt.show()  # 显示绘制的图形
```
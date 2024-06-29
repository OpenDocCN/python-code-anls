# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\voxels_simple.py`

```py
"""
=========================
voxels([x, y, z], filled)
=========================

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.voxels`.
"""

# 导入需要的库
import matplotlib.pyplot as plt
import numpy as np

# 使用自定义的 matplotlib 样式
plt.style.use('_mpl-gallery')

# 准备坐标
x, y, z = np.indices((8, 8, 8))

# 定义两个立方体的布尔数组：一个在左上角，一个在右下角
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)

# 将两个立方体组合成一个布尔数组
voxelarray = cube1 | cube2

# 创建图形和 3D 坐标轴
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# 绘制体素图
ax.voxels(voxelarray, edgecolor='k')

# 设置坐标轴标签为空，以便隐藏坐标轴刻度
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

# 显示图形
plt.show()
```
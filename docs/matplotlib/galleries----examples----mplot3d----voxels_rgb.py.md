# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\voxels_rgb.py`

```py
"""
==========================================
3D voxel / volumetric plot with RGB colors
==========================================

Demonstrates using `.Axes3D.voxels` to visualize parts of a color space.
"""

# 导入所需的库
import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库并简写为 plt
import numpy as np  # 导入 numpy 库并简写为 np


def midpoints(x):
    sl = ()  # 初始化一个空的元组 sl
    for _ in range(x.ndim):
        # 计算每个维度上的中点值
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]  # 更新切片以处理下一个维度
    return x  # 返回计算得到的中点数组

# 准备一些坐标，并为每个坐标附加 RGB 值
r, g, b = np.indices((17, 17, 17)) / 16.0  # 创建一个包含 RGB 值的网格坐标
rc = midpoints(r)  # 计算红色通道的中点值
gc = midpoints(g)  # 计算绿色通道的中点值
bc = midpoints(b)  # 计算蓝色通道的中点值

# 定义一个球体，球心在 [0.5, 0.5, 0.5] 处
sphere = (rc - 0.5)**2 + (gc - 0.5)**2 + (bc - 0.5)**2 < 0.5**2  # 根据球体方程定义球体

# 合并颜色组件
colors = np.zeros(sphere.shape + (3,))
colors[..., 0] = rc  # 将红色通道赋给颜色数组
colors[..., 1] = gc  # 将绿色通道赋给颜色数组
colors[..., 2] = bc  # 将蓝色通道赋给颜色数组

# 绘制所有内容
ax = plt.figure().add_subplot(projection='3d')  # 创建一个带有 3D 投影的子图
ax.voxels(r, g, b, sphere,  # 使用 voxels 方法绘制体素图
          facecolors=colors,  # 设置每个体素的颜色
          edgecolors=np.clip(2*colors - 0.5, 0, 1),  # 设置边缘颜色为更亮的版本
          linewidth=0.5)  # 设置边缘线的宽度

ax.set(xlabel='r', ylabel='g', zlabel='b')  # 设置坐标轴标签
ax.set_aspect('equal')  # 设置坐标轴比例相等，确保体素是正方体

plt.show()  # 显示绘制的图形
```
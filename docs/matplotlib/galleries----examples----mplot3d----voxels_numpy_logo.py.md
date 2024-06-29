# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\voxels_numpy_logo.py`

```py
"""
===============================
3D voxel plot of the NumPy logo
===============================

Demonstrates using `.Axes3D.voxels` with uneven coordinates.
"""

# 导入必要的库：matplotlib.pyplot用于绘图，numpy用于数值计算
import matplotlib.pyplot as plt
import numpy as np

# 定义一个函数，用于扩展数据的尺寸
def explode(data):
    # 计算新的数组尺寸是原来的两倍
    size = np.array(data.shape) * 2
    # 创建一个全零数组，数据类型与原数组相同
    data_e = np.zeros(size - 1, dtype=data.dtype)
    # 将原始数据填充到新数组的偶数索引位置
    data_e[::2, ::2, ::2] = data
    return data_e

# 构建表示 NumPy logo 的三维体素图像数据
n_voxels = np.zeros((4, 3, 4), dtype=bool)
n_voxels[0, 0, :] = True
n_voxels[-1, 0, :] = True
n_voxels[1, 0, 2] = True
n_voxels[2, 0, 1] = True

# 定义用于填充体素的颜色和边缘的颜色
facecolors = np.where(n_voxels, '#FFD65DC0', '#7A88CCC0')
edgecolors = np.where(n_voxels, '#BFAB6E', '#7D84A6')
filled = np.ones(n_voxels.shape)

# 对上述体素图像进行放大，留下间隙
filled_2 = explode(filled)
fcolors_2 = explode(facecolors)
ecolors_2 = explode(edgecolors)

# 缩小间隙
x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
x[0::2, :, :] += 0.05
y[:, 0::2, :] += 0.05
z[:, :, 0::2] += 0.05
x[1::2, :, :] += 0.95
y[:, 1::2, :] += 0.95
z[:, :, 1::2] += 0.95

# 创建一个带有三维投影的图形子图
ax = plt.figure().add_subplot(projection='3d')
# 使用voxels方法绘制体素图像
ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
# 设置坐标轴的比例为等比例
ax.set_aspect('equal')

# 显示绘制的图形
plt.show()
```
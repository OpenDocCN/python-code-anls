# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\errorbar3d.py`

```
"""
============
3D errorbars
============

An example of using errorbars with upper and lower limits in mplot3d.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 库的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

ax = plt.figure().add_subplot(projection='3d')  # 创建一个带有 3D 投影的子图

# setting up a parametric curve
t = np.arange(0, 2*np.pi+.1, 0.01)  # 创建一个参数曲线的参数 t，范围是 [0, 2π]
x, y, z = np.sin(t), np.cos(3*t), np.sin(5*t)  # 计算参数曲线的 x, y, z 值

estep = 15  # 设置步长
i = np.arange(t.size)  # 创建一个 t 大小的索引数组
zuplims = (i % estep == 0) & (i // estep % 3 == 0)  # 创建上限的布尔数组
zlolims = (i % estep == 0) & (i // estep % 3 == 2)  # 创建下限的布尔数组

# 在 3D 图形中绘制带有误差棒的曲线
ax.errorbar(x, y, z, 0.2, zuplims=zuplims, zlolims=zlolims, errorevery=estep)

ax.set_xlabel("X label")  # 设置 X 轴标签
ax.set_ylabel("Y label")  # 设置 Y 轴标签
ax.set_zlabel("Z label")  # 设置 Z 轴标签

plt.show()  # 显示绘制的图形
```
# `D:\src\scipysrc\matplotlib\galleries\plot_types\3D\stem3d.py`

```py
"""
=============
stem(x, y, z)
=============

See `~mpl_toolkits.mplot3d.axes3d.Axes3D.stem`.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import numpy as np  # 导入NumPy数学计算库

plt.style.use('_mpl-gallery')  # 使用指定的样式表 '_mpl-gallery'

# 创建数据
n = 20
x = np.sin(np.linspace(0, 2*np.pi, n))  # 生成一个包含 n 个元素的正弦值数组
y = np.cos(np.linspace(0, 2*np.pi, n))  # 生成一个包含 n 个元素的余弦值数组
z = np.linspace(0, 1, n)  # 生成一个包含 n 个元素的从0到1均匀分布的数组

# 绘图
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})  # 创建一个包含3D坐标轴的图形和坐标系
ax.stem(x, y, z)  # 在3D坐标系上绘制(x, y, z)坐标点的stem图形

# 设置坐标轴标签为空列表，即不显示坐标轴标签
ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])

plt.show()  # 显示绘制的图形
```
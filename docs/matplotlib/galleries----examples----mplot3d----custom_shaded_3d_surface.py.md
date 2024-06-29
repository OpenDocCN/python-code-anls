# `D:\src\scipysrc\matplotlib\galleries\examples\mplot3d\custom_shaded_3d_surface.py`

```py
"""
=======================================
Custom hillshading in a 3D surface plot
=======================================

Demonstrates using custom hillshading in a 3D surface plot.
"""

import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import numpy as np  # 导入NumPy库

from matplotlib import cbook, cm  # 导入matplotlib中的辅助函数和颜色映射模块
from matplotlib.colors import LightSource  # 导入LightSource类，用于定制光照效果

# Load and format data
dem = cbook.get_sample_data('jacksboro_fault_dem.npz')  # 加载示例数据文件
z = dem['elevation']  # 从示例数据中获取高程数据
nrows, ncols = z.shape  # 获取高程数据的行数和列数
x = np.linspace(dem['xmin'], dem['xmax'], ncols)  # 在xmin到xmax之间创建ncols个均匀间隔的数组作为x坐标
y = np.linspace(dem['ymin'], dem['ymax'], nrows)  # 在ymin到ymax之间创建nrows个均匀间隔的数组作为y坐标
x, y = np.meshgrid(x, y)  # 创建网格坐标矩阵x和y

region = np.s_[5:50, 5:50]  # 定义感兴趣区域的切片，这里是5到49行和列
x, y, z = x[region], y[region], z[region]  # 应用感兴趣区域的切片，限制数据集合

# Set up plot
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))  # 创建一个包含3D子图的Figure和Axes对象

ls = LightSource(270, 45)  # 创建LightSource对象，指定光源的方位角和高度角
# To use a custom hillshading mode, override the built-in shading and pass
# in the rgb colors of the shaded surface calculated from "shade".
rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
# 计算rgb颜色，用于绘制阴影效果，使用地球色彩映射，垂直强化系数为0.1，软混合模式

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
# 绘制三维表面图，使用rgb颜色，设置行和列的步幅为1，设置边缘线宽为0，关闭抗锯齿，关闭阴影效果

plt.show()  # 显示绘制的3D图形
```
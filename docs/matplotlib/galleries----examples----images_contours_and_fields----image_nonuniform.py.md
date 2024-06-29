# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\image_nonuniform.py`

```py
"""
================
Image nonuniform
================

This illustrates the NonUniformImage class.  It is not
available via an Axes method, but it is easily added to an
Axes instance as shown here.
"""

# 导入需要的库
import matplotlib.pyplot as plt  # 导入matplotlib的绘图模块
import numpy as np  # 导入NumPy库，用于数组操作

from matplotlib import cm  # 导入颜色映射模块
from matplotlib.image import NonUniformImage  # 导入非均匀图像类

interp = 'nearest'  # 设置插值方法为最近邻插值

# Linear x array for cell centers:
x = np.linspace(-4, 4, 9)  # 在[-4, 4]之间生成均匀间隔的9个点作为x轴坐标

# Highly nonlinear x array:
x2 = x**3  # 通过将x数组元素求立方得到高度非线性的x2数组

y = np.linspace(-4, 4, 9)  # 在[-4, 4]之间生成均匀间隔的9个点作为y轴坐标

z = np.sqrt(x[np.newaxis, :]**2 + y[:, np.newaxis]**2)  # 计算以x和y坐标为中心的网格点到原点的距离，生成距离矩阵z

fig, axs = plt.subplots(nrows=2, ncols=2, layout='constrained')  # 创建一个包含2x2子图的图形对象
fig.suptitle('NonUniformImage class', fontsize='large')  # 设置图形对象的总标题

ax = axs[0, 0]  # 选择第一行第一列的子图
im = NonUniformImage(ax, interpolation=interp, extent=(-4, 4, -4, 4),
                     cmap=cm.Purples)  # 创建一个非均匀图像对象，指定插值方法、范围和颜色映射
im.set_data(x, y, z)  # 设置图像数据，传入x、y坐标和距离矩阵z
ax.add_image(im)  # 在子图上添加非均匀图像对象
ax.set_xlim(-4, 4)  # 设置子图x轴显示范围
ax.set_ylim(-4, 4)  # 设置子图y轴显示范围
ax.set_title(interp)  # 设置子图标题为当前插值方法

ax = axs[0, 1]  # 选择第一行第二列的子图
im = NonUniformImage(ax, interpolation=interp, extent=(-64, 64, -4, 4),
                     cmap=cm.Purples)  # 创建一个非均匀图像对象，指定插值方法、范围和颜色映射
im.set_data(x2, y, z)  # 设置图像数据，传入x2、y坐标和距离矩阵z
ax.add_image(im)  # 在子图上添加非均匀图像对象
ax.set_xlim(-64, 64)  # 设置子图x轴显示范围
ax.set_ylim(-4, 4)  # 设置子图y轴显示范围
ax.set_title(interp)  # 设置子图标题为当前插值方法

interp = 'bilinear'  # 设置插值方法为双线性插值

ax = axs[1, 0]  # 选择第二行第一列的子图
im = NonUniformImage(ax, interpolation=interp, extent=(-4, 4, -4, 4),
                     cmap=cm.Purples)  # 创建一个非均匀图像对象，指定插值方法、范围和颜色映射
im.set_data(x, y, z)  # 设置图像数据，传入x、y坐标和距离矩阵z
ax.add_image(im)  # 在子图上添加非均匀图像对象
ax.set_xlim(-4, 4)  # 设置子图x轴显示范围
ax.set_ylim(-4, 4)  # 设置子图y轴显示范围
ax.set_title(interp)  # 设置子图标题为当前插值方法

ax = axs[1, 1]  # 选择第二行第二列的子图
im = NonUniformImage(ax, interpolation=interp, extent=(-64, 64, -4, 4),
                     cmap=cm.Purples)  # 创建一个非均匀图像对象，指定插值方法、范围和颜色映射
im.set_data(x2, y, z)  # 设置图像数据，传入x2、y坐标和距离矩阵z
ax.add_image(im)  # 在子图上添加非均匀图像对象
ax.set_xlim(-64, 64)  # 设置子图x轴显示范围
ax.set_ylim(-4, 4)  # 设置子图y轴显示范围
ax.set_title(interp)  # 设置子图标题为当前插值方法

plt.show()  # 显示图形
```
# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\layer_images.py`

```
"""
============
Layer Images
============

Layer images above one another using alpha blending
"""
# 导入 matplotlib.pyplot 作为 plt
import matplotlib.pyplot as plt
# 导入 numpy 并简写为 np
import numpy as np


def func3(x, y):
    # 定义一个函数，返回给定 x, y 的函数值
    return (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2 + y**2))


# 将 dx 和 dy 设置为较小值，以增加图像的分辨率
dx, dy = 0.05, 0.05

# 创建 x 和 y 数组，定义其范围和步长
x = np.arange(-3.0, 3.0, dx)
y = np.arange(-3.0, 3.0, dy)
# 创建网格 X 和 Y，用于绘制 2D 数据
X, Y = np.meshgrid(x, y)

# 当叠加多个图像时，图像需要具有相同的范围 (extent)。
# 这并不意味着它们需要具有相同的形状，但它们需要渲染到由
# xmin、xmax、ymin、ymax 确定的相同坐标系中。
# 注意，如果对图像使用不同的插值方法，由于插值边缘效应，它们的视觉范围可能会有所不同。

# 计算 extent 参数，用于指定图像的范围
extent = np.min(x), np.max(x), np.min(y), np.max(y)
# 创建一个不带边框的图形窗口
fig = plt.figure(frameon=False)

# 创建第一个图像 Z1，一个棋盘格
Z1 = np.add.outer(range(8), range(8)) % 2  # chessboard
im1 = plt.imshow(Z1, cmap=plt.cm.gray, interpolation='nearest',
                 extent=extent)

# 创建第二个图像 Z2，使用 func3 函数生成的数据
Z2 = func3(X, Y)

# 在同一窗口中叠加第二个图像，并设置透明度和插值方法
im2 = plt.imshow(Z2, cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear',
                 extent=extent)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
```
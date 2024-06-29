# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\ellipse_collection.py`

```py
"""
==================
Ellipse Collection
==================

Drawing a collection of ellipses. While this would equally be possible using
a `~.collections.EllipseCollection` or `~.collections.PathCollection`, the use
of an `~.collections.EllipseCollection` allows for much shorter code.
"""

# 导入 matplotlib 库中的 pyplot 模块作为 plt，并导入 numpy 库作为 np
import matplotlib.pyplot as plt
import numpy as np

# 从 matplotlib 的 collections 模块中导入 EllipseCollection 类
from matplotlib.collections import EllipseCollection

# 创建一个包含整数的 NumPy 数组 x 和 y
x = np.arange(10)
y = np.arange(15)

# 创建 X 和 Y 矩阵网格
X, Y = np.meshgrid(x, y)

# 将 X 和 Y 矩阵展平并沿第二轴进行堆叠，得到二维坐标数组 XY
XY = np.column_stack((X.ravel(), Y.ravel()))

# 计算每个椭圆的宽度 ww，高度 hh 和倾斜角 aa
ww = X / 10.0
hh = Y / 15.0
aa = X * 9

# 创建一个新的图形 fig 和坐标系 ax
fig, ax = plt.subplots()

# 使用 EllipseCollection 创建椭圆集合 ec，传入椭圆的宽度 ww、高度 hh、倾斜角 aa
# 设置单位为 'x'，偏移为 XY，并使用 ax.transData 进行坐标变换
ec = EllipseCollection(ww, hh, aa, units='x', offsets=XY,
                       offset_transform=ax.transData)

# 将数据数组 (X + Y) 展平后设置为 ec 的数组属性
ec.set_array((X + Y).ravel())

# 将椭圆集合 ec 添加到坐标系 ax 中
ax.add_collection(ec)

# 自动调整坐标轴视图
ax.autoscale_view()

# 设置 x 轴和 y 轴标签
ax.set_xlabel('X')
ax.set_ylabel('y')

# 添加颜色条 cbar，并设置标签为 'X+Y'
cbar = plt.colorbar(ec)
cbar.set_label('X+Y')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.collections`
#    - `matplotlib.collections.EllipseCollection`
#    - `matplotlib.axes.Axes.add_collection`
#    - `matplotlib.axes.Axes.autoscale_view`
#    - `matplotlib.cm.ScalarMappable.set_array`
```
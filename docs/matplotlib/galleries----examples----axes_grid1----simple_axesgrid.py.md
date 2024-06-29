# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\simple_axesgrid.py`

```py
"""
================
Simple ImageGrid
================

Align multiple images using `~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`.
"""

# 导入 matplotlib.pyplot 库，并简称为 plt
import matplotlib.pyplot as plt
# 导入 numpy 库，并简称为 np
import numpy as np
# 从 mpl_toolkits.axes_grid1 库中导入 ImageGrid 类
from mpl_toolkits.axes_grid1 import ImageGrid

# 创建一个 10x10 的数组，元素值从 0 到 99，然后重塑为 10x10 的二维数组
im1 = np.arange(100).reshape((10, 10))
# 将 im1 进行转置，得到其转置矩阵
im2 = im1.T
# 将 im1 上下翻转，得到翻转后的新矩阵
im3 = np.flipud(im1)
# 将 im2 左右翻转，得到翻转后的新矩阵
im4 = np.fliplr(im2)

# 创建一个新的图形对象，大小为 4x4 英寸
fig = plt.figure(figsize=(4., 4.))
# 在图形上创建一个 ImageGrid，类似于 subplot(111)，但使用 ImageGrid 对象
grid = ImageGrid(fig, 111,
                 nrows_ncols=(2, 2),  # 创建一个 2x2 的 Axes 网格
                 axes_pad=0.1,  # 设置每个 Axes 之间的间距为 0.1 英寸
                 )

# 遍历网格中的每个 Axes 和对应的图像，依次显示图像
for ax, im in zip(grid, [im1, im2, im3, im4]):
    # 在当前的 Axes 上显示图像 im
    ax.imshow(im)

# 显示图形
plt.show()
```
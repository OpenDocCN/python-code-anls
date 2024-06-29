# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\simple_axesgrid2.py`

```py
"""
==================
Simple ImageGrid 2
==================

Align multiple images of different sizes using
`~mpl_toolkits.axes_grid1.axes_grid.ImageGrid`.
"""

# 导入 matplotlib 库中的 pyplot 模块
import matplotlib.pyplot as plt

# 从 matplotlib 中导入 cbook 模块
from matplotlib import cbook

# 从 mpl_toolkits.axes_grid1 库中导入 ImageGrid 类
from mpl_toolkits.axes_grid1 import ImageGrid

# 创建一个新的图形对象，设置其大小为 5.5x3.5 英寸
fig = plt.figure(figsize=(5.5, 3.5))

# 在图形对象中创建一个 ImageGrid 对象，使用 1x3 的布局，位于第一个位置（subplot(111) 类似）
grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 3),   # 设置行数和列数
                 axes_pad=0.1,         # 设置子图之间的间距
                 label_mode="L",       # 设置标签模式为 L
                 )

# 从示例数据中获取一个演示图像的数据
Z = cbook.get_sample_data("axes_grid/bivariate_normal.npy")

# 将示例图像分成三部分
im1 = Z
im2 = Z[:, :10]
im3 = Z[:, 10:]

# 获取图像数据的最小值和最大值
vmin, vmax = Z.min(), Z.max()

# 遍历 ImageGrid 中的每个子图对象和对应的图像数据，并显示图像
for ax, im in zip(grid, [im1, im2, im3]):
    ax.imshow(im, origin="lower", vmin=vmin, vmax=vmax)

# 显示图形对象
plt.show()
```
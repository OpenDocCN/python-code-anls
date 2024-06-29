# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\demo_bboximage.py`

```
"""
==============
BboxImage Demo
==============

A `~matplotlib.image.BboxImage` can be used to position an image according to
a bounding box. This demo shows how to show an image inside a `.text.Text`'s
bounding box as well as how to manually create a bounding box for the image.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块
import numpy as np  # 导入 NumPy 数学库

from matplotlib.image import BboxImage  # 导入 BboxImage 类
from matplotlib.transforms import Bbox, TransformedBbox  # 导入 Bbox 和 TransformedBbox 类

fig, (ax1, ax2) = plt.subplots(ncols=2)  # 创建包含两个子图的图形对象

# ----------------------------
# Create a BboxImage with Text
# ----------------------------

txt = ax1.text(0.5, 0.5, "test", size=30, ha="center", color="w")  # 在 ax1 中创建文本对象
ax1.add_artist(  # 将 BboxImage 对象添加到 ax1 中
    BboxImage(txt.get_window_extent, data=np.arange(256).reshape((1, -1))))

# ------------------------------------
# Create a BboxImage for each colormap
# ------------------------------------
# 获取所有 colormap 的名称列表，跳过反转的 colormap
cmap_names = sorted(m for m in plt.colormaps if not m.endswith("_r"))

ncol = 2  # 列数
nrow = len(cmap_names) // ncol + 1  # 行数

xpad_fraction = 0.3  # 横向填充比例
dx = 1 / (ncol + xpad_fraction * (ncol - 1))  # 图片宽度计算

ypad_fraction = 0.3  # 纵向填充比例
dy = 1 / (nrow + ypad_fraction * (nrow - 1))  # 图片高度计算

for i, cmap_name in enumerate(cmap_names):  # 遍历 colormap 名称列表
    ix, iy = divmod(i, ncol)  # 计算当前 colormap 在网格中的位置
    bbox0 = Bbox.from_bounds(ix*dx*(1+xpad_fraction),  # 创建原始 Bbox 对象
                             1 - iy*dy*(1+ypad_fraction) - dy,
                             dx, dy)
    bbox = TransformedBbox(bbox0, ax2.transAxes)  # 将原始 Bbox 转换为 ax2 坐标系中的 Bbox
    ax2.add_artist(  # 将 BboxImage 对象添加到 ax2 中
        BboxImage(bbox, cmap=cmap_name, data=np.arange(256).reshape((1, -1))))

plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.image.BboxImage`
#    - `matplotlib.transforms.Bbox`
#    - `matplotlib.transforms.TransformedBbox`
#    - `matplotlib.text.Text`
```
# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\barcode_demo.py`

```
"""
=======
Barcode
=======
This demo shows how to produce a bar code.

The figure size is calculated so that the width in pixels is a multiple of the
number of data points to prevent interpolation artifacts. Additionally, the
``Axes`` is defined to span the whole figure and all ``Axis`` are turned off.

The data itself is rendered with `~.Axes.imshow` using

- ``code.reshape(1, -1)`` to turn the data into a 2D array with one row.
- ``imshow(..., aspect='auto')`` to allow for non-square pixels.
- ``imshow(..., interpolation='nearest')`` to prevent blurred edges. This
  should not happen anyway because we fine-tuned the figure width in pixels,
  but just to be safe.
"""

import matplotlib.pyplot as plt
import numpy as np

# 生成条形码所需的数据，用 0 和 1 表示黑白条的序列
code = np.array([
    1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1,
    0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
    1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1,
    1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1])

# 每个条的像素数
pixel_per_bar = 4
# 图像分辨率
dpi = 100

# 创建图像对象，设置尺寸和分辨率
fig = plt.figure(figsize=(len(code) * pixel_per_bar / dpi, 2), dpi=dpi)
# 添加坐标轴，跨越整个图像
ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
ax.set_axis_off()  # 关闭坐标轴显示
# 显示条形码图像，使用二值黑白色彩映射，自动调整像素宽度和最近邻插值
ax.imshow(code.reshape(1, -1), cmap='binary', aspect='auto',
          interpolation='nearest')
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.add_axes`
```
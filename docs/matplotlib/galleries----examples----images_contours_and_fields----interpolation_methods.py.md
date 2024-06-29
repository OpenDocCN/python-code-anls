# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\interpolation_methods.py`

```
"""
=========================
Interpolations for imshow
=========================

This example displays the difference between interpolation methods for
`~.axes.Axes.imshow`.

If *interpolation* is None, it defaults to the :rc:`image.interpolation`.
If the interpolation is ``'none'``, then no interpolation is performed for the
Agg, ps and pdf backends. Other backends will default to ``'antialiased'``.

For the Agg, ps and pdf backends, ``interpolation='none'`` works well when a
big image is scaled down, while ``interpolation='nearest'`` works well when
a small image is scaled up.

See :doc:`/gallery/images_contours_and_fields/image_antialiasing` for a
discussion on the default ``interpolation='antialiased'`` option.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 库，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

# 定义插值方法列表
methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

# 设置随机数种子以保证结果可重复性
np.random.seed(19680801)

# 生成一个 4x4 的随机数组成的网格
grid = np.random.rand(4, 4)

# 创建一个包含 3 行 6 列子图的图形对象，并设置图形大小及子图的坐标轴设置
fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

# 遍历每个子图及对应的插值方法
for ax, interp_method in zip(axs.flat, methods):
    # 在当前子图上显示网格数据，使用给定的插值方法及颜色映射
    ax.imshow(grid, interpolation=interp_method, cmap='viridis')
    # 设置当前子图的标题为插值方法的字符串表示
    ax.set_title(str(interp_method))

# 调整子图布局以更好地适应图形区域
plt.tight_layout()
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
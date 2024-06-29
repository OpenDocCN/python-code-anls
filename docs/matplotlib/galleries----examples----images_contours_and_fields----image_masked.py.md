# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\image_masked.py`

```
"""
============
Image Masked
============

imshow with masked array input and out-of-range colors.

The second subplot illustrates the use of BoundaryNorm to
get a filled contour effect.
"""

# 导入 matplotlib 的 pyplot 模块并简写为 plt
import matplotlib.pyplot as plt
# 导入 numpy 并简写为 np
import numpy as np
# 导入 matplotlib 的 colors 模块
import matplotlib.colors as colors

# 计算一些有趣的数据
x0, x1 = -5, 5
y0, y1 = -3, 3
# 在指定范围内生成均匀间隔的数组
x = np.linspace(x0, x1, 500)
y = np.linspace(y0, y1, 500)
# 生成格点矩阵
X, Y = np.meshgrid(x, y)
# 计算 Z1 和 Z2
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
# 计算 Z
Z = (Z1 - Z2) * 2

# 设置一个调色板
palette = plt.cm.gray.with_extremes(over='r', under='g', bad='b')
# 或者可以使用下面的方式使得 bad 区域透明
# palette.set_bad(alpha = 0.0)
# 如果注释掉所有的 palette.set* 行，将会看到默认效果；
# under 和 over 会使用调色板中的第一个和最后一个颜色。
Zm = np.ma.masked_where(Z > 1.2, Z)

# 通过在 norm 中设置 vmin 和 vmax，建立正常调色板颜色标度的应用范围。
# 超出该范围的任何值将根据 palette.set_over 等进行颜色处理。

# 设置 Axes 对象
fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(6, 5.4))

# 使用 'continuous' colormap 进行绘图
im = ax1.imshow(Zm, interpolation='bilinear',
                cmap=palette,
                norm=colors.Normalize(vmin=-1.0, vmax=1.0),
                aspect='auto',
                origin='lower',
                extent=[x0, x1, y0, y1])
ax1.set_title('Green=low, Red=high, Blue=masked')
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax1)
cbar.set_label('uniform')
ax1.tick_params(axis='x', labelbottom=False)

# 使用少量颜色进行绘图，并使用不均匀间隔的边界
im = ax2.imshow(Zm, interpolation='nearest',
                cmap=palette,
                norm=colors.BoundaryNorm([-1, -0.5, -0.2, 0, 0.2, 0.5, 1],
                                         ncolors=palette.N),
                aspect='auto',
                origin='lower',
                extent=[x0, x1, y0, y1])
ax2.set_title('With BoundaryNorm')
cbar = fig.colorbar(im, extend='both', spacing='proportional',
                    shrink=0.9, ax=ax2)
cbar.set_label('proportional')

fig.suptitle('imshow, with out-of-range and masked data')
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors.BoundaryNorm`
#    - `matplotlib.colorbar.Colorbar.set_label`
```
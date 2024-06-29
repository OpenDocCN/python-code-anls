# `D:\src\scipysrc\matplotlib\galleries\examples\images_contours_and_fields\image_transparency_blend.py`

```
"""
==========================================
Blend transparency with color in 2D images
==========================================

Blend transparency with color to highlight parts of data with imshow.

A common use for `matplotlib.pyplot.imshow` is to plot a 2D statistical
map. The function makes it easy to visualize a 2D matrix as an image and add
transparency to the output. For example, one can plot a statistic (such as a
t-statistic) and color the transparency of each pixel according to its p-value.
This example demonstrates how you can achieve this effect.

First we will generate some data, in this case, we'll create two 2D "blobs"
in a 2D grid. One blob will be positive, and the other negative.
"""

import matplotlib.pyplot as plt
# sphinx_gallery_thumbnail_number = 3
import numpy as np

from matplotlib.colors import Normalize


def normal_pdf(x, mean, var):
    return np.exp(-(x - mean)**2 / (2*var))


# Generate the space in which the blobs will live
xmin, xmax, ymin, ymax = (0, 100, 0, 100)
n_bins = 100
xx = np.linspace(xmin, xmax, n_bins)
yy = np.linspace(ymin, ymax, n_bins)

# Generate the blobs. The range of the values is roughly -.0002 to .0002
means_high = [20, 50]
means_low = [50, 60]
var = [150, 200]

# Compute the Gaussian distributions for the blobs in the x and y directions
gauss_x_high = normal_pdf(xx, means_high[0], var[0])
gauss_y_high = normal_pdf(yy, means_high[1], var[0])

gauss_x_low = normal_pdf(xx, means_low[0], var[1])
gauss_y_low = normal_pdf(yy, means_low[1], var[1])

# Create the weight matrix by subtracting the low Gaussian from the high Gaussian
weights = (np.outer(gauss_y_high, gauss_x_high)
           - np.outer(gauss_y_low, gauss_x_low))

# We'll also create a grey background into which the pixels will fade
greys = np.full((*weights.shape, 3), 70, dtype=np.uint8)

# First we'll plot these blobs using ``imshow`` without transparency.
vmax = np.abs(weights).max()
imshow_kwargs = {
    'vmax': vmax,                    # 设置颜色映射的最大值
    'vmin': -vmax,                   # 设置颜色映射的最小值
    'cmap': 'RdYlBu',                # 使用红-黄-蓝色谱进行颜色映射
    'extent': (xmin, xmax, ymin, ymax),  # 设置图像的数据范围
}

fig, ax = plt.subplots()
ax.imshow(greys)                     # 在图像上叠加灰色背景
ax.imshow(weights, **imshow_kwargs)   # 在图像上叠加权重数据，应用imshow_kwargs中的参数
ax.set_axis_off()                    # 关闭坐标轴显示

# %%
# Blending in transparency
# ========================
#
# The simplest way to include transparency when plotting data with
# `matplotlib.pyplot.imshow` is to pass an array matching the shape of
# the data to the ``alpha`` argument. For example, we'll create a gradient
# moving from left to right below.

# Create an alpha channel of linearly increasing values moving to the right.
alphas = np.ones(weights.shape)
alphas[:, 30:] = np.linspace(1, 0, 70)

# Create the figure and image
# Note that the absolute values may be slightly different
fig, ax = plt.subplots()
ax.imshow(greys)                     # 在图像上叠加灰色背景
ax.imshow(weights, alpha=alphas, **imshow_kwargs)  # 叠加具有alpha通道的权重数据，应用imshow_kwargs中的参数
ax.set_axis_off()                    # 关闭坐标轴显示

# %%
# Using transparency to highlight values with high amplitude
# ==========================================================
#
# Finally, we'll recreate the same plot, but this time we'll use transparency
# to highlight the extreme values in the data. This is often used to highlight
# Create an alpha channel based on weight values
# Any value whose absolute value is > .0001 will have zero transparency
# 使用 Normalize(0, .3, clip=True) 函数将权重值归一化到 [0, 0.3] 的范围，并且裁剪超出范围的值
alphas = Normalize(0, .3, clip=True)(np.abs(weights))
# 将透明度值限制在 [0.4, 1] 的范围内
alphas = np.clip(alphas, .4, 1)  # alpha 值被裁剪到最低为 0.4

# Create the figure and image
# Note that the absolute values may be slightly different
# 创建绘图窗口和图像
fig, ax = plt.subplots()
# 在 ax 上显示灰度图像 greys
ax.imshow(greys)
# 使用带有 alpha 通道的权重图像 weights 来覆盖之前的图像，alpha 值为 alphas
ax.imshow(weights, alpha=alphas, **imshow_kwargs)

# Add contour lines to further highlight different levels.
# 添加等高线以进一步突出不同的级别
# 绘制权重图像 weights 的等高线，水平翻转图像 weights[::-1]，并指定等高线水平为 [-0.1, 0.1]，颜色为黑色，线型为实线
ax.contour(weights[::-1], levels=[-.1, .1], colors='k', linestyles='-')
# 设置图像的坐标轴关闭
ax.set_axis_off()
# 显示图像
plt.show()

# 绘制权重图像 weights 的等高线，水平翻转图像 weights[::-1]，并指定等高线水平为 [-0.0001, 0.0001]，颜色为黑色，线型为实线
ax.contour(weights[::-1], levels=[-.0001, .0001], colors='k', linestyles='-')
# 设置图像的坐标轴关闭
ax.set_axis_off()
# 显示图像
plt.show()
```
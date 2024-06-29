# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\gradient_bar.py`

```
"""
========================
Bar chart with gradients
========================

Matplotlib does not natively support gradients. However, we can emulate a
gradient-filled rectangle by an `.AxesImage` of the right size and coloring.

In particular, we use a colormap to generate the actual colors. It is then
sufficient to define the underlying values on the corners of the image and
let bicubic interpolation fill out the area. We define the gradient direction
by a unit vector *v*. The values at the corners are then obtained by the
lengths of the projections of the corner vectors on *v*.

A similar approach can be used to create a gradient background for an Axes.
In that case, it is helpful to use Axes coordinates (``extent=(0, 1, 0, 1),
transform=ax.transAxes``) to be independent of the data coordinates.
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)

# 定义一个函数，用于在给定的 Axes 上绘制基于色彩映射的渐变图像
def gradient_image(ax, direction=0.3, cmap_range=(0, 1), **kwargs):
    """
    Draw a gradient image based on a colormap.

    Parameters
    ----------
    ax : Axes
        The Axes to draw on.
    direction : float
        The direction of the gradient. This is a number in
        range 0 (=vertical) to 1 (=horizontal).
    cmap_range : float, float
        The fraction (cmin, cmax) of the colormap that should be
        used for the gradient, where the complete colormap is (0, 1).
    **kwargs
        Other parameters are passed on to `.Axes.imshow()`.
        In particular, *cmap*, *extent*, and *transform* may be useful.
    """
    # 根据给定的方向计算单位向量 v 的角度
    phi = direction * np.pi / 2
    v = np.array([np.cos(phi), np.sin(phi)])
    # 定义渐变图像的角落点的数值
    X = np.array([[v @ [1, 0], v @ [1, 1]],
                  [v @ [0, 0], v @ [0, 1]]])
    # 根据色彩映射范围调整 X 的值，以确保在 colormap 范围内
    a, b = cmap_range
    X = a + (b - a) / X.max() * X
    # 使用双三次插值绘制渐变图像
    im = ax.imshow(X, interpolation='bicubic', clim=(0, 1),
                   aspect='auto', **kwargs)
    return im


# 定义一个函数，用于在给定的 Axes 上绘制带有渐变背景的条形图
def gradient_bar(ax, x, y, width=0.5, bottom=0):
    """
    Draw a bar plot with gradient-filled rectangles.

    Parameters
    ----------
    ax : Axes
        The Axes to draw on.
    x : array-like
        The x coordinates of the bars.
    y : array-like
        The y coordinates of the bars.
    width : float, optional
        The width of the bars.
    bottom : float, optional
        The y-coordinate(s) of the bars bases.

    Notes
    -----
    This function draws a set of bars with a gradient-filled background
    using the `gradient_image` function.
    """
    # 遍历每个条形图的左右边界，调用 gradient_image 函数绘制渐变背景
    for left, top in zip(x, y):
        right = left + width
        gradient_image(ax, extent=(left, right, bottom, top),
                       cmap=plt.cm.Blues_r, cmap_range=(0, 0.8))


# 创建一个图形窗口和一个 Axes 对象
fig, ax = plt.subplots()
ax.set(xlim=(0, 10), ylim=(0, 1))

# 绘制背景渐变图像
gradient_image(ax, direction=1, extent=(0, 1, 0, 1), transform=ax.transAxes,
               cmap=plt.cm.RdYlGn, cmap_range=(0.2, 0.8), alpha=0.5)

# 生成数据并绘制渐变背景的条形图
N = 10
x = np.arange(N) + 0.15
y = np.random.rand(N)
gradient_bar(ax, x, y, width=0.7)

# 显示图形
plt.show()
```
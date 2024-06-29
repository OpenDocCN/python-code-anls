# `D:\src\scipysrc\matplotlib\galleries\examples\color\custom_cmap.py`

```
"""
=========================================
Creating a colormap from a list of colors
=========================================

For more detail on creating and manipulating colormaps see
:ref:`colormap-manipulation`.

Creating a :ref:`colormap <colormaps>` from a list of colors
can be done with the `.LinearSegmentedColormap.from_list` method.  You must
pass a list of RGB tuples that define the mixture of colors from 0 to 1.


Creating custom colormaps
=========================
It is also possible to create a custom mapping for a colormap. This is
accomplished by creating dictionary that specifies how the RGB channels
change from one end of the cmap to the other.

Example: suppose you want red to increase from 0 to 1 over the bottom
half, green to do the same over the middle half, and blue over the top
half.  Then you would use::

    cdict = {
        'red': (
            (0.0,  0.0, 0.0),   # At position 0, red is 0
            (0.5,  1.0, 1.0),   # At position 0.5, red is 1.0
            (1.0,  1.0, 1.0),   # At position 1.0, red is 1.0
        ),
        'green': (
            (0.0,  0.0, 0.0),   # At position 0, green is 0
            (0.25, 0.0, 0.0),   # At position 0.25, green is 0.0
            (0.75, 1.0, 1.0),   # At position 0.75, green is 1.0
            (1.0,  1.0, 1.0),   # At position 1.0, green is 1.0
        ),
        'blue': (
            (0.0,  0.0, 0.0),   # At position 0, blue is 0
            (0.5,  0.0, 0.0),   # At position 0.5, blue is 0.0
            (1.0,  1.0, 1.0),   # At position 1.0, blue is 1.0
        )
    }

If, as in this example, there are no discontinuities in the r, g, and b
components, then it is quite simple: the second and third element of
each tuple, above, is the same -- call it "``y``".  The first element ("``x``")
defines interpolation intervals over the full range of 0 to 1, and it
must span that whole range.  In other words, the values of ``x`` divide the
0-to-1 range into a set of segments, and ``y`` gives the end-point color
values for each segment.

Now consider the green, ``cdict['green']`` is saying that for:

- 0 <= ``x`` <= 0.25, ``y`` is zero; no green.
- 0.25 < ``x`` <= 0.75, ``y`` varies linearly from 0 to 1.
- 0.75 < ``x`` <= 1, ``y`` remains at 1, full green.

If there are discontinuities, then it is a little more complicated. Label the 3
elements in each row in the ``cdict`` entry for a given color as ``(x, y0,
y1)``. Then for values of ``x`` between ``x[i]`` and ``x[i+1]`` the color value
is interpolated between ``y1[i]`` and ``y0[i+1]``.

Going back to a cookbook example::

    cdict = {
        'red': (
            (0.0,  0.0, 0.0),   # At position 0, red is 0
            (0.5,  1.0, 0.7),   # At position 0.5, red is 1.0, and at 1.0, red is 0.7
            (1.0,  1.0, 1.0),   # At position 1.0, red is 1.0
        ),
        'green': (
            (0.0,  0.0, 0.0),   # At position 0, green is 0
            (0.5,  1.0, 0.0),   # At position 0.5, green is 1.0, and at 1.0, green is 0.0
            (1.0,  1.0, 1.0),   # At position 1.0, green is 1.0
        ),
        'blue': (
            (0.0,  0.0, 0.0),   # At position 0, blue is 0
            (0.5,  0.0, 0.0),   # At position 0.5, blue is 0.0
            (1.0,  1.0, 1.0),   # At position 1.0, blue is 1.0
        )
    }

and look at ``cdict['red'][1]``; because ``y0 != y1``, it is saying that for
``x`` from 0 to 0.5, red increases from 0 to 1, but then it jumps down, so that
for ``x`` from 0.5 to 1, red increases from 0.7 to 1.  Green ramps from 0 to 1
as ``x`` goes from 0 to 0.5, then jumps back to 0, and ramps back to 1 as ``x``
"""

注释：
"""
import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

import matplotlib as mpl  # 导入 matplotlib 库的整体接口
from matplotlib.colors import LinearSegmentedColormap  # 导入 LinearSegmentedColormap 类

# Make some illustrative fake data:

x = np.arange(0, np.pi, 0.1)  # 创建一个从 0 到 π （不含）的数组，步长为 0.1
y = np.arange(0, 2 * np.pi, 0.1)  # 创建一个从 0 到 2π （不含）的数组，步长为 0.1
X, Y = np.meshgrid(x, y)  # 创建网格坐标 X 和 Y

Z = np.cos(X) * np.sin(Y) * 10  # 根据 X 和 Y 计算 Z 值，用于生成二维数据

# %%
# Colormaps from a list
# ---------------------

colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # 定义颜色列表，顺序为红、绿、蓝
n_bins = [3, 6, 10, 100]  # 插值离散成的分组数
cmap_name = 'my_list'  # 自定义 colormap 的名称
fig, axs = plt.subplots(2, 2, figsize=(6, 9))  # 创建 2x2 的子图布局，尺寸为 6x9
fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)  # 调整子图布局参数
for n_bin, ax in zip(n_bins, axs.flat):  # 遍历分组数和子图
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)  # 根据颜色列表创建线性分段 colormap
    # Fewer bins will result in "coarser" colomap interpolation
    im = ax.imshow(Z, origin='lower', cmap=cmap)  # 在当前子图上绘制 Z 数据的热图
    ax.set_title("N bins: %s" % n_bin)  # 设置子图标题
    fig.colorbar(im, ax=ax)  # 添加子图的颜色条

# %%
# Custom colormaps
# ----------------

cdict1 = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.1),
        (1.0, 1.0, 1.0),
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    'blue': (
        (0.0, 0.0, 1.0),
        (0.5, 0.1, 0.0),
        (1.0, 0.0, 0.0),
    )
}

cdict2 = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 1.0),
        (1.0, 0.1, 1.0),
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    'blue': (
        (0.0, 0.0, 0.1),
        (0.5, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    )
}

cdict3 = {
    'red': (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.8, 1.0),
        (0.75, 1.0, 1.0),
        (1.0, 0.4, 1.0),
    ),
    'green': (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.5, 0.9, 0.9),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    ),
    'blue': (
        (0.0, 0.0, 0.4),
        (0.25, 1.0, 1.0),
        (0.5, 1.0, 0.8),
        (0.75, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
}

# Make a modified version of cdict3 with some transparency
# in the middle of the range.
cdict4 = {
    **cdict3,
    'alpha': (
        (0.0, 1.0, 1.0),
        # (0.25, 1.0, 1.0),
        (0.5, 0.3, 0.3),
        # (0.75, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ),
}

# %%
# Now we will use this example to illustrate 2 ways of
# handling custom colormaps.
# First, the most direct and explicit:

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)  # 使用 cdict1 创建自定义 colormap

# %%
# Second, create the map explicitly and register it.
# Like the first method, this method works with any kind
# of Colormap, not just
# a LinearSegmentedColormap:

mpl.colormaps.register(LinearSegmentedColormap('BlueRed2', cdict2))  # 注册自定义 colormap
# 注册自定义的颜色映射 'BlueRed3' 到 Matplotlib 的颜色映射注册表中
mpl.colormaps.register(LinearSegmentedColormap('BlueRed3', cdict3))
# 注册自定义的颜色映射 'BlueRedAlpha' 到 Matplotlib 的颜色映射注册表中
mpl.colormaps.register(LinearSegmentedColormap('BlueRedAlpha', cdict4))

# %%
# 创建一个包含 4 个子图的图形:

# 创建一个 2x2 的子图布局，指定图形大小为 (6, 9)，调整子图之间的间距和位置
fig, axs = plt.subplots(2, 2, figsize=(6, 9))
fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)

# 在 axs[0, 0] 上绘制 Z 的热图，使用 'blue_red1' 颜色映射
im1 = axs[0, 0].imshow(Z, cmap=blue_red1)
# 在 axs[0, 0] 上创建颜色条
fig.colorbar(im1, ax=axs[0, 0])

# 在 axs[1, 0] 上绘制 Z 的热图，使用内置的 'BlueRed2' 颜色映射
im2 = axs[1, 0].imshow(Z, cmap='BlueRed2')
# 在 axs[1, 0] 上创建颜色条
fig.colorbar(im2, ax=axs[1, 0])

# 设置 'BlueRed3' 为默认的颜色映射，此时所有后续的 imshow 使用此颜色映射
plt.rcParams['image.cmap'] = 'BlueRed3'

# 在 axs[0, 1] 上绘制 Z 的热图，使用默认的 'BlueRed3' 颜色映射
im3 = axs[0, 1].imshow(Z)
# 在 axs[0, 1] 上创建颜色条，并设置标题
fig.colorbar(im3, ax=axs[0, 1])
axs[0, 1].set_title("Alpha = 1")

# 在 axs[1, 1] 上绘制 Z 的热图
im4 = axs[1, 1].imshow(Z)
# 在 axs[1, 1] 上创建颜色条
fig.colorbar(im4, ax=axs[1, 1])

# 将当前图像及其颜色条的颜色映射设置为 'BlueRedAlpha'
im4.set_cmap('BlueRedAlpha')
# 设置 axs[1, 1] 的标题
axs[1, 1].set_title("Varying alpha")

# 设置整个图形的总标题
fig.suptitle('Custom Blue-Red colormaps', fontsize=16)
# 调整子图布局，使总标题不重叠子图
fig.subplots_adjust(top=0.9)

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    本示例中展示了以下函数、方法、类和模块的使用:
#
#    - `matplotlib.axes.Axes.imshow` / `matplotlib.pyplot.imshow`
#    - `matplotlib.figure.Figure.colorbar` / `matplotlib.pyplot.colorbar`
#    - `matplotlib.colors`
#    - `matplotlib.colors.LinearSegmentedColormap`
#    - `matplotlib.colors.LinearSegmentedColormap.from_list`
#    - `matplotlib.cm`
#    - `matplotlib.cm.ScalarMappable.set_cmap`
#    - `matplotlib.cm.ColormapRegistry.register`
```
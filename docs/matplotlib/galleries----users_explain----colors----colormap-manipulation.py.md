# `D:\src\scipysrc\matplotlib\galleries\users_explain\colors\colormap-manipulation.py`

```py
"""
.. redirect-from:: /tutorials/colors/colormap-manipulation

.. _colormap-manipulation:

********************************
Creating Colormaps in Matplotlib
********************************

Matplotlib has a number of built-in colormaps accessible via
`.matplotlib.colormaps`.  There are also external libraries like
palettable_ that have many extra colormaps.

.. _palettable: https://jiffyclub.github.io/palettable/

However, we may also want to create or manipulate our own colormaps.
This can be done using the class `.ListedColormap` or
`.LinearSegmentedColormap`.
Both colormap classes map values between 0 and 1 to colors. There are however
differences, as explained below.

Before manually creating or manipulating colormaps, let us first see how we
can obtain colormaps and their colors from existing colormap classes.

Getting colormaps and accessing their values
============================================

First, getting a named colormap, most of which are listed in
:ref:`colormaps`, may be done using `.matplotlib.colormaps`,
which returns a colormap object.  The length of the list of colors used
internally to define the colormap can be adjusted via `.Colormap.resampled`.
Below we use a modest value of 8 so there are not a lot of values to look at.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

import matplotlib as mpl  # 导入 matplotlib 库，用于绘图
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # 从 matplotlib.colors 模块导入 LinearSegmentedColormap 和 ListedColormap 类

# 从 matplotlib 中获取 'viridis' 颜色映射，并将其重新采样为 8 个值的列表
viridis = mpl.colormaps['viridis'].resampled(8)

# %%
# The object ``viridis`` is a callable, that when passed a float between
# 0 and 1 returns an RGBA value from the colormap:

print(viridis(0.56))

# %%
# ListedColormap
# --------------
#
# `.ListedColormap`\s store their color values in a ``.colors`` attribute.
# The list of colors that comprise the colormap can be directly accessed using
# the ``colors`` property,
# or it can be accessed indirectly by calling  ``viridis`` with an array of
# values matching the length of the colormap.  Note that the returned list is
# in the form of an RGBA (N, 4) array, where N is the length of the colormap.

print('viridis.colors', viridis.colors)  # 打印 viridis colormap 的颜色列表
print('viridis(range(8))', viridis(range(8)))  # 打印使用 range(8) 调用 viridis colormap 后的颜色值列表
print('viridis(np.linspace(0, 1, 8))', viridis(np.linspace(0, 1, 8)))  # 打印使用 np.linspace(0, 1, 8) 调用 viridis colormap 后的颜色值列表

# %%
# The colormap is a lookup table, so "oversampling" the colormap returns
# nearest-neighbor interpolation (note the repeated colors in the list below)

print('viridis(np.linspace(0, 1, 12))', viridis(np.linspace(0, 1, 12)))

# %%
# LinearSegmentedColormap
# -----------------------
# `.LinearSegmentedColormap`\s do not have a ``.colors`` attribute.
# However, one may still call the colormap with an integer array, or with a
# float array between 0 and 1.

copper = mpl.colormaps['copper'].resampled(8)

print('copper(range(8))', copper(range(8)))  # 打印使用 range(8) 调用 copper colormap 后的颜色值列表
print('copper(np.linspace(0, 1, 8))', copper(np.linspace(0, 1, 8)))  # 打印使用 np.linspace(0, 1, 8) 调用 copper colormap 后的颜色值列表

# %%
# Creating listed colormaps
# =========================
#
# Creating a colormap is essentially the inverse operation of the above where
# 我们向 `.ListedColormap` 提供一个颜色规范的列表或数组来创建新的颜色映射。
#
# 在继续本教程之前，让我们定义一个辅助函数，它接受一个或多个颜色映射作为输入，
# 创建一些随机数据，并将颜色映射应用于该数据集的图像绘制中。
def plot_examples(colormaps):
    """
    辅助函数，用于使用关联的颜色映射绘制数据。
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)
    n = len(colormaps)
    fig, axs = plt.subplots(1, n, figsize=(n * 2 + 2, 3),
                            constrained_layout=True, squeeze=False)
    for [ax, cmap] in zip(axs.flat, colormaps):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

# %%
# 最简单的情况下，我们可以键入一个颜色名称列表以创建一个由这些颜色构成的颜色映射。
cmap = ListedColormap(["darkorange", "gold", "lawngreen", "lightseagreen"])
plot_examples([cmap])

# %%
# 实际上，该列表可以包含任何有效的
# :ref:`Matplotlib颜色规范 <colors_def>`。
# 创建自定义颜色映射特别有用的是 (N, 4) 形状的数组。
# 因为我们可以对这样的数组进行各种numpy操作，从而从现有的颜色映射中构建新的颜色映射变得非常直接。
#
# 例如，假设我们想要将一个长度为256的“viridis”颜色映射的前25个条目变成粉色：

viridis = mpl.cm.get_cmap('viridis').reversed()
newcolors = viridis(np.linspace(0, 1, 256))
pink = np.array([248/256, 24/256, 148/256, 1])
newcolors[:25, :] = pink
newcmp = ListedColormap(newcolors)

plot_examples([viridis, newcmp])

# %%
# 我们可以减少颜色映射的动态范围；在这里我们选择颜色映射的中间一半。
# 但是请注意，由于viridis是一个列表颜色映射，我们最终会得到128个离散值，而不是原始颜色映射中的256个值。
# 这种方法不会在颜色空间中进行插值以添加新颜色。

viridis_big = mpl.cm.get_cmap('viridis')
newcmp = ListedColormap(viridis_big(np.linspace(0.25, 0.75, 128)))
plot_examples([viridis, newcmp])

# %%
# 我们还可以轻松地连接两个颜色映射：

top = mpl.cm.get_cmap('Oranges_r').reversed()
bottom = mpl.cm.get_cmap('Blues')
newcolors = np.vstack((top(np.linspace(0, 1, 128)),
                       bottom(np.linspace(0, 1, 128))))
newcmp = ListedColormap(newcolors, name='OrangeBlue')
plot_examples([viridis, newcmp])

# %%
# 当然，我们不必从一个命名的颜色映射开始，我们只需创建一个 (N, 4) 的数组传递给 `.ListedColormap`。
# 在这里，我们创建一个从棕色（RGB: 90, 40, 40）到白色（RGB: 255, 255, 255）的颜色映射。

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(90/256, 1, N)
vals[:, 1] = np.linspace(40/256, 1, N)
vals[:, 2] = np.linspace(40/256, 1, N)
newcmp = ListedColormap(vals)
plot_examples([viridis, newcmp])

# %%
# 创建线性分段色图
# ===================================
#
# `.LinearSegmentedColormap` 类通过锚点指定色图，其中 RGB(A) 值在锚点之间插值。
#
# 指定这些色图的格式允许在锚点处出现不连续性。每个锚点在矩阵中以 `[x[i] yleft[i] yright[i]]` 的形式指定，
# 其中 `x[i]` 是锚点，`yleft[i]` 和 `yright[i]` 是锚点两侧的颜色值。
#
# 如果没有不连续性，则 `yleft[i] == yright[i]`：

cdict = {'red':   [[0.0,  0.0, 0.0],
                   [0.5,  1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'green': [[0.0,  0.0, 0.0],
                   [0.25, 0.0, 0.0],
                   [0.75, 1.0, 1.0],
                   [1.0,  1.0, 1.0]],
         'blue':  [[0.0,  0.0, 0.0],
                   [0.5,  0.0, 0.0],
                   [1.0,  1.0, 1.0]]}

def plot_linearmap(cdict):
    # 使用 `LinearSegmentedColormap` 类创建新的色图对象，命名为 'testCmap'，使用给定的分段数据 `cdict`，并设置总的颜色数为 256
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    # 生成从色图中提取的 RGBA 值数组，使用 `np.linspace` 在 [0, 1] 之间均匀分布
    rgba = newcmp(np.linspace(0, 1, 256))
    # 创建包含单个子图的新图形和轴对象，设置图形大小为 (4, 3)，布局为 'constrained'
    fig, ax = plt.subplots(figsize=(4, 3), layout='constrained')
    # 颜色列表
    col = ['r', 'g', 'b']
    # 在图中绘制三条垂直虚线，位置分别为 [0.25, 0.5, 0.75]，颜色为灰色 ('0.7')，线型为虚线 ('--')
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    # 在同一图中绘制每个颜色通道的插值曲线
    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
    # 设置 X 轴标签为 'index'，Y 轴标签为 'RGB'
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    # 显示图形
    plt.show()

plot_linearmap(cdict)

# %%
# 为了在一个锚点处创建不连续性，第三列与第二列不同。每个颜色通道的矩阵如下所示：
#
#   cdict['red'] = [...
#                   [x[i]      yleft[i]     yright[i]],
#                   [x[i+1]    yleft[i+1]   yright[i+1]],
#                  ...]
#
# 对于传递给色图的值在 `x[i]` 和 `x[i+1]` 之间，插值在 `yright[i]` 和 `yleft[i+1]` 之间进行。
#
# 在下面的示例中，红色通道在 0.5 处存在不连续性。在 0 到 0.5 之间的插值从 0.3 到 1，而在 0.5 到 1 之间的插值从 0.9 到 1。
# 注意，`red[0, 1]` 和 `red[2, 2]` 都不影响插值，因为 `red[0, 1]`（即 `yleft[0]`）是 0 左侧的值，而 `red[2, 2]`（即 `yright[2]`）是 1 右侧的值，超出了颜色映射域。

cdict['red'] = [[0.0,  0.0, 0.3],
                [0.5,  1.0, 0.9],
                [1.0,  1.0, 1.0]]
plot_linearmap(cdict)

# %%
# 直接从颜色列表创建分段色图
# --------------------------------------------------
#
# 上述方法非常灵活，但实施起来有点繁琐。对于一些基本情况，使用 `.LinearSegmentedColormap.from_list` 可能更简单。
# 它从提供的颜色列表中创建一个分段色图，颜色之间的间距相等。

colors = ["darkorange", "gold", "lawngreen", "lightseagreen"]
# 创建一个线性分段的颜色映射，名称为"mycmap"，使用给定的颜色列表
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

# %%
# 如果需要，可以通过节点列表来指定颜色映射的分段位置，节点值应在0到1之间。
# 例如，可以使颜色映射中的红色部分占据更多的空间。
nodes = [0.0, 0.4, 0.8, 1.0]
# 使用节点列表和颜色列表创建线性分段的颜色映射
cmap2 = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

# 调用 plot_examples 函数，展示两种颜色映射
plot_examples([cmap1, cmap2])

# %%
# 定义一个名为 my_cmap 的 ListedColormap 对象，使用给定的颜色列表
colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
my_cmap = ListedColormap(colors, name="my_cmap")
# 创建 my_cmap 的反转版本，并命名为 my_cmap_r
my_cmap_r = my_cmap.reversed()

# 调用 plot_examples 函数，展示原始和反转后的两种颜色映射
plot_examples([my_cmap, my_cmap_r])

# %%
# 如果没有提供名称，`.reversed` 方法也会将复制的颜色映射命名为原始颜色映射名称加上 '_r' 后缀。

# %%
# .. _registering-colormap:
#
# 注册颜色映射
# ======================
#
# 可以将颜色映射添加到 `matplotlib.colormaps` 的命名颜色映射列表中，
# 这样可以通过名称在绘图函数中访问这些颜色映射：

# 将 my_cmap 和 my_cmap_r 添加到 matplotlib 的颜色映射列表中
mpl.colormaps.register(cmap=my_cmap)
mpl.colormaps.register(cmap=my_cmap_r)

# 创建一个简单的数据集
data = [[1, 2, 3, 4, 5]]

# 创建包含两个子图的图形对象
fig, (ax1, ax2) = plt.subplots(nrows=2)

# 在第一个子图上使用 'my_cmap' 颜色映射绘制数据
ax1.imshow(data, cmap='my_cmap')
# 在第二个子图上使用 'my_cmap_r' 反转的颜色映射绘制数据
ax2.imshow(data, cmap='my_cmap_r')

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.axes.Axes.pcolormesh`
#    - `matplotlib.figure.Figure.colorbar`
#    - `matplotlib.colors`
#    - `matplotlib.colors.LinearSegmentedColormap`
#    - `matplotlib.colors.ListedColormap`
#    - `matplotlib.cm`
#    - `matplotlib.colormaps`
```
# `D:\src\scipysrc\matplotlib\galleries\users_explain\colors\colormaps.py`

```
"""
.. redirect-from:: /tutorials/colors/colormaps

.. _colormaps:

********************************
Choosing Colormaps in Matplotlib
********************************

Matplotlib has a number of built-in colormaps accessible via
`.matplotlib.colormaps`.  There are also external libraries that
have many extra colormaps, which can be viewed in the
`Third-party colormaps`_ section of the Matplotlib documentation.
Here we briefly discuss how to choose between the many options.  For
help on creating your own colormaps, see
:ref:`colormap-manipulation`.

To get a list of all registered colormaps, you can do::

    from matplotlib import colormaps
    list(colormaps)

Overview
========

The idea behind choosing a good colormap is to find a good representation in 3D
colorspace for your data set. The best colormap for any given data set depends
on many things including:

- Whether representing form or metric data ([Ware]_)

- Your knowledge of the data set (*e.g.*, is there a critical value
  from which the other values deviate?)

- If there is an intuitive color scheme for the parameter you are plotting

- If there is a standard in the field the audience may be expecting

For many applications, a perceptually uniform colormap is the best choice;
i.e. a colormap in which equal steps in data are perceived as equal
steps in the color space. Researchers have found that the human brain
perceives changes in the lightness parameter as changes in the data
much better than, for example, changes in hue. Therefore, colormaps
which have monotonically increasing lightness through the colormap
will be better interpreted by the viewer. Wonderful examples of
perceptually uniform colormaps can be found in the
`Third-party colormaps`_ section as well.

Color can be represented in 3D space in various ways. One way to represent color
is using CIELAB. In CIELAB, color space is represented by lightness,
:math:`L^*`; red-green, :math:`a^*`; and yellow-blue, :math:`b^*`. The lightness
parameter :math:`L^*` can then be used to learn more about how the matplotlib
colormaps will be perceived by viewers.

An excellent starting resource for learning about human perception of colormaps
is from [IBM]_.


.. _color-colormaps_reference:

Classes of colormaps
====================

Colormaps are often split into several categories based on their function (see,
*e.g.*, [Moreland]_):

1. Sequential: change in lightness and often saturation of color
   incrementally, often using a single hue; should be used for
   representing information that has ordering.

2. Diverging: change in lightness and possibly saturation of two
   different colors that meet in the middle at an unsaturated color;
   should be used when the information being plotted has a critical
   middle value, such as topography or when the data deviates around
   zero.

"""
# sphinx_gallery_thumbnail_number = 2

from colorspacious import cspace_converter  # 导入颜色空间转换函数

import matplotlib.pyplot as plt  # 导入matplotlib绘图库
import numpy as np  # 导入NumPy数值计算库

import matplotlib as mpl  # 导入matplotlib库，并使用mpl作为别名

# %%
#
# 首先，展示每个颜色映射的范围。注意某些颜色映射的变化速度比其他的更快。
#

cmaps = {}  # 初始化空字典，用于存储颜色映射列表

gradient = np.linspace(0, 1, 256)  # 创建一个从0到1的256个元素的数组
gradient = np.vstack((gradient, gradient))  # 垂直堆叠数组，生成渐变色图像数据


def plot_color_gradients(category, cmap_list):
    # 创建图形，并根据颜色映射数量调整图形的高度
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
                        left=0.2, right=0.99)
    axs[0].set_title(f'{category} colormaps', fontsize=14)  # 设置子图标题

    for ax, name in zip(axs, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])  # 在子图中绘制颜色映射的渐变图像
        ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
                transform=ax.transAxes)  # 在子图中添加颜色映射的名称标签

    # 关闭所有子图的坐标轴和刻度线
    for ax in axs:
        ax.set_axis_off()

    # 保存颜色映射列表供后续使用
    cmaps[category] = cmap_list


# %%
# Sequential
# ----------
#
# 对于顺序图，亮度值通过颜色映射单调递增。有些颜色映射中的 :math:`L^*` 值范围从0到100（如binary和其他灰度映射），
# 而其他颜色映射从 :math:`L^*=20` 开始。具有较小 :math:`L^*` 范围的颜色映射在感知范围上也相应较小。
# 注意， :math:`L^*` 函数在不同的颜色映射中有所不同：有些几乎是 :math:`L^*` 的线性变化，而其他则更弯曲。

plot_color_gradients('Perceptually Uniform Sequential',
                     ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

# %%

plot_color_gradients('Sequential',
                     ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'])

# %%
# Sequential2
# -----------
#
# 许多 Sequential2 图中的 :math:`L^*` 值单调递增，但一些（如autumn、cool、spring和winter）在 :math:`L^*` 空间中
# 平稳或上下波动。其他一些（如afmhot、copper、gist_heat和hot）在 :math:`L^*` 函数中有拐点。
# 在颜色映射的平稳或拐点区域表示的数据将导致感知上的
# 使用自定义函数 plot_color_gradients 绘制不同类型的颜色映射的示例

plot_color_gradients('Sequential (2)',
                     ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                      'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                      'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'])

# %%
# 分类为 Diverging 类型的颜色映射
# ------------------------------
#
# 对于 Diverging 类型的颜色映射，我们希望在最大值处有单调递增的 :math:`L^*` 值，
# 这个最大值应该接近 :math:`L^*=100`，然后是单调递减的 :math:`L^*` 值。我们希望
# 在颜色映射的两端有大致相等的最小 :math:`L^*` 值。根据这些标准，BrBG 和 RdBu
# 是很好的选择。coolwarm 也是一个不错的选择，但它在 :math:`L^*` 值上没有很广的范围
# （参见下面的灰度部分）。

plot_color_gradients('Diverging',
                     ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                      'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'])

# %%
# 分类为 Cyclic 类型的颜色映射
# ----------------------------
#
# 对于 Cyclic 类型的颜色映射，我们希望从同一颜色开始和结束，并在中间达到对称中心点。
# :math:`L^*` 应该从起始点到中间单调变化，从中间到结束点反向变化。它在增加和减少的
# 一侧应该对称，并且只在色调上有所不同。在端点和中间，:math:`L^*` 值会反向变化，应
# 该在 :math:`L^*` 空间中平滑以减少伪影。更多关于循环映射设计的信息可以参考 [kovesi-colormaps]_。
#
# HSV 色彩映射虽然不对称到中心点，但也包含在这组色彩映射中。此外，HSV 色彩映射的
# :math:`L^*` 值在整个映射中变化范围很大，因此在感知上不适合用作颜色映射。可以在
# [mycarta-jet]_ 上看到对这个想法的延伸。

plot_color_gradients('Cyclic', ['twilight', 'twilight_shifted', 'hsv'])

# %%
# 分类为 Qualitative 类型的颜色映射
# ---------------------------------
#
# Qualitative 类型的颜色映射并不旨在成为感知映射，但观察其亮度参数可以验证这一点。
# :math:`L^*` 值在整个颜色映射中变化很大，并且明显不是单调递增的。因此，这些颜色映射
# 不适合用作感知映射。

plot_color_gradients('Qualitative',
                     ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                      'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                      'tab20c'])

# %%
# 分类为 Miscellaneous 类型的颜色映射
# -----------------------------------
#
# 一些杂项类别的颜色映射具有特定的用途。例如，gist_earth、ocean 和 terrain
# 都似乎是为了同时绘制地形（绿色/棕色）和水深（蓝色）而创建的。因此，我们会
# 预期在这些颜色映射中看到分歧，但多个拐点可能并非理想，比如在...
# gist_earth and terrain. CMRmap was created to convert well to
# grayscale, though it does appear to have some small kinks in
# :math:`L^*`.  cubehelix was created to vary smoothly in both lightness
# and hue, but appears to have a small hump in the green hue area. turbo
# was created to display depth and disparity data.
#
# The often-used jet colormap is included in this set of colormaps. We can see
# that the :math:`L^*` values vary widely throughout the colormap, making it a
# poor choice for representing data for viewers to see perceptually. See an
# extension on this idea at [mycarta-jet]_ and [turbo]_.

plot_color_gradients('Miscellaneous',
                     ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                      'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                      'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                      'turbo', 'nipy_spectral', 'gist_ncar'])

plt.show()

# %%
# Lightness of Matplotlib colormaps
# =================================
#
# Here we examine the lightness values of the matplotlib colormaps.
# Note that some documentation on the colormaps is available
# ([list-colormaps]_).

mpl.rcParams.update({'font.size': 12})

# Number of colormap per subplot for particular cmap categories
_DSUBS = {'Perceptually Uniform Sequential': 5, 'Sequential': 6,
          'Sequential (2)': 6, 'Diverging': 6, 'Cyclic': 3,
          'Qualitative': 4, 'Miscellaneous': 6}

# Spacing between the colormaps of a subplot
_DC = {'Perceptually Uniform Sequential': 1.4, 'Sequential': 0.7,
       'Sequential (2)': 1.4, 'Diverging': 1.4, 'Cyclic': 1.4,
       'Qualitative': 1.4, 'Miscellaneous': 1.4}

# Indices to step through colormap
x = np.linspace(0.0, 1.0, 100)

# Do plot
for cmap_category, cmap_list in cmaps.items():

    # Do subplots so that colormaps have enough space.
    # Default is 6 colormaps per subplot.
    dsub = _DSUBS.get(cmap_category, 6)
    nsubplots = int(np.ceil(len(cmap_list) / dsub))

    # squeeze=False to handle similarly the case of a single subplot
    fig, axs = plt.subplots(nrows=nsubplots, squeeze=False,
                            figsize=(7, 2.6*nsubplots))
    for i, ax in enumerate(axs.flat):
        # 对 axs 数组中的每个子图进行遍历，i 是索引，ax 是当前子图对象

        locs = []  # 用于存储文本标签的位置列表

        for j, cmap in enumerate(cmap_list[i*dsub:(i+1)*dsub]):
            # 对当前子图使用的每个 colormap 进行遍历，j 是索引，cmap 是当前的 colormap 名称

            # 获取 colormap 的 RGB 值，并将其转换为 CAM02-UCS 色彩空间
            rgb = mpl.colormaps[cmap](x)[np.newaxis, :, :3]
            lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

            # 绘制 colormap 的 L 值。根据不同的 cmap_category 分类进行不同的处理
            if cmap_category == 'Sequential':
                # 这些 colormap 都从高亮度开始，但我们希望它们反转以便在图中看起来更好
                y_ = lab[0, ::-1, 0]
                c_ = x[::-1]
            else:
                y_ = lab[0, :, 0]
                c_ = x

            # 获取水平方向上的 colormap 间距
            dc = _DC.get(cmap_category, 1.4)
            ax.scatter(x + j*dc, y_, c=c_, cmap=cmap, s=300, linewidths=0.0)
            
            # 存储 colormap 标签的位置
            if cmap_category in ('Perceptually Uniform Sequential', 'Sequential'):
                locs.append(x[-1] + j*dc)
            elif cmap_category in ('Diverging', 'Qualitative', 'Cyclic', 'Miscellaneous', 'Sequential (2)'):
                locs.append(x[int(x.size/2.)] + j*dc)

        # 设置子图的 x 轴和 y 轴限制
        ax.set_xlim(axs[0, 0].get_xlim())
        ax.set_ylim(0.0, 100.0)

        # 设置 colormap 的标签
        ax.xaxis.set_ticks_position('top')
        ticker = mpl.ticker.FixedLocator(locs)
        ax.xaxis.set_major_locator(ticker)
        formatter = mpl.ticker.FixedFormatter(cmap_list[i*dsub:(i+1)*dsub])
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=50)
        ax.set_ylabel('Lightness $L^*$', fontsize=12)

    # 设置 x 轴标签
    ax.set_xlabel(cmap_category + ' colormaps', fontsize=14)

    # 调整子图布局并显示图像
    fig.tight_layout(h_pad=0.0, pad=1.5)
    plt.show()
# %%
# Grayscale conversion
# ====================
#
# It is important to pay attention to conversion to grayscale for color
# plots, since they may be printed on black and white printers.  If not
# carefully considered, your readers may end up with indecipherable
# plots because the grayscale changes unpredictably through the
# colormap.
#
# Conversion to grayscale is done in many different ways [bw]_. Some of the
# better ones use a linear combination of the rgb values of a pixel, but
# weighted according to how we perceive color intensity. A nonlinear method of
# conversion to grayscale is to use the :math:`L^*` values of the pixels. In
# general, similar principles apply for this question as they do for presenting
# one's information perceptually; that is, if a colormap is chosen that is
# monotonically increasing in :math:`L^*` values, it will print in a reasonable
# manner to grayscale.
#
# With this in mind, we see that the Sequential colormaps have reasonable
# representations in grayscale. Some of the Sequential2 colormaps have decent
# enough grayscale representations, though some (autumn, spring, summer,
# winter) have very little grayscale change. If a colormap like this was used
# in a plot and then the plot was printed to grayscale, a lot of the
# information may map to the same gray values. The Diverging colormaps mostly
# vary from darker gray on the outer edges to white in the middle. Some
# (PuOr and seismic) have noticeably darker gray on one side than the other
# and therefore are not very symmetric. coolwarm has little range of gray scale
# and would print to a more uniform plot, losing a lot of detail. Note that
# overlaid, labeled contours could help differentiate between one side of the
# colormap vs. the other since color cannot be used once a plot is printed to
# grayscale. Many of the Qualitative and Miscellaneous colormaps, such as
# Accent, hsv, jet and turbo, change from darker to lighter and back to darker
# grey throughout the colormap. This would make it impossible for a viewer to
# interpret the information in a plot once it is printed in grayscale.
#
# Update default font size for matplotlib plots to 14.
mpl.rcParams.update({'font.size': 14})

# Indices to step through colormap.
x = np.linspace(0.0, 1.0, 100)

# Create a gradient array from 0 to 1 with 256 values.
gradient = np.linspace(0, 1, 256)
# Stack the gradient array vertically to create a 2D array with two rows of 256 values each.
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(cmap_category, cmap_list):
    # Create a figure with subplots based on the number of colormaps provided.
    fig, axs = plt.subplots(nrows=len(cmap_list), ncols=2)
    # Adjust the layout of the figure for better spacing.
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99,
                        wspace=0.05)
    # Set the main title of the figure based on the colormap category.
    fig.suptitle(cmap_category + ' colormaps', fontsize=14, y=1.0, x=0.6)
    for ax, name in zip(axs, cmap_list):
        # 对每个子图ax和颜色映射名称name进行迭代

        # 获取颜色映射的RGB值。
        rgb = mpl.colormaps[name](x)[np.newaxis, :, :3]

        # 将颜色映射转换到CAM02-UCS颜色空间，我们需要亮度值。
        lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
        L = lab[0, :, 0]
        L = np.float32(np.vstack((L, L, L)))

        # 在第一个子图上显示渐变图像，使用指定的颜色映射。
        ax[0].imshow(gradient, aspect='auto', cmap=mpl.colormaps[name])

        # 在第二个子图上显示L值图像，使用二值化的颜色映射。
        ax[1].imshow(L, aspect='auto', cmap='binary_r', vmin=0., vmax=100.)

        # 获取第一个子图的位置信息并计算标签的位置。
        pos = list(ax[0].get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.

        # 在图形上添加标签，显示颜色映射的名称。
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # 关闭所有子图的坐标轴和脊柱，而不仅仅是具有颜色映射的子图。
    for ax in axs.flat:
        ax.set_axis_off()

    # 显示整个图形。
    plt.show()
# 遍历颜色映射字典 `cmaps`，获取每个类别和其对应的颜色映射列表
for cmap_category, cmap_list in cmaps.items():

    # 调用函数 `plot_color_gradients`，传入当前类别和对应的颜色映射列表进行绘图
    plot_color_gradients(cmap_category, cmap_list)
```
# `D:\src\scipysrc\matplotlib\galleries\examples\color\named_colors.py`

```py
"""
====================
List of named colors
====================

This plots a list of the named colors supported by Matplotlib.
For more information on colors in matplotlib see

* the :ref:`colors_def` tutorial;
* the `matplotlib.colors` API;
* the :doc:`/gallery/color/color_demo`.

----------------------------
Helper Function for Plotting
----------------------------
First we define a helper function for making a table of colors, then we use it
on some common color categories.
"""

import math  # 导入数学库，用于数学计算

import matplotlib.pyplot as plt  # 导入matplotlib.pyplot模块，并简写为plt

import matplotlib.colors as mcolors  # 导入matplotlib.colors模块，并简写为mcolors
from matplotlib.patches import Rectangle  # 从matplotlib.patches模块导入Rectangle类


def plot_colortable(colors, *, ncols=4, sort_colors=True):
    """
    Plot a color table with the given colors dictionary.

    Parameters:
    colors : dict
        Dictionary mapping color names to RGB values.
    ncols : int, optional
        Number of columns for the color table.
    sort_colors : bool, optional
        Whether to sort colors by hue, saturation, value, and name.

    Returns:
    fig : matplotlib.figure.Figure
        The Figure object containing the color table.
    """

    cell_width = 212  # 每个单元格的宽度
    cell_height = 22  # 每个单元格的高度
    swatch_width = 48  # 颜色示例矩形的宽度
    margin = 12  # 边距大小

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)  # 计算行数

    width = cell_width * ncols + 2 * margin  # 图表宽度
    height = cell_height * nrows + 2 * margin  # 图表高度
    dpi = 72  # 图像分辨率

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)  # 创建子图和图形对象
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)  # 调整子图布局
    ax.set_xlim(0, cell_width * ncols)  # 设置x轴的范围
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)  # 设置y轴的范围
    ax.yaxis.set_visible(False)  # 隐藏y轴
    ax.xaxis.set_visible(False)  # 隐藏x轴
    ax.set_axis_off()  # 关闭坐标轴

    for i, name in enumerate(names):
        row = i % nrows  # 当前行号
        col = i // nrows  # 当前列号
        y = row * cell_height  # 计算当前行的y坐标

        swatch_start_x = cell_width * col  # 颜色示例矩形的起始x坐标
        text_pos_x = cell_width * col + swatch_width + 7  # 文本的起始x坐标

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')  # 在图中添加颜色名称的文本标签

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )  # 在图中添加颜色示例矩形

    return fig  # 返回图形对象

# %%
# -----------
# Base colors
# -----------

plot_colortable(mcolors.BASE_COLORS, ncols=3, sort_colors=False)  # 调用plot_colortable函数绘制基础颜色表

# %%
# ---------------
# Tableau Palette
# ---------------

plot_colortable(mcolors.TABLEAU_COLORS, ncols=2, sort_colors=False)  # 调用plot_colortable函数绘制Tableau颜色表

# %%
# ----------
# CSS Colors
# ----------

# sphinx_gallery_thumbnail_number = 3
plot_colortable(mcolors.CSS4_COLORS)  # 调用plot_colortable函数绘制CSS颜色表
plt.show()  # 显示图形

# %%
# -----------
# XKCD Colors
# -----------
# Matplotlib supports colors from the
# `xkcd color survey <https://xkcd.com/color/rgb/>`_, e.g. ``"xkcd:sky blue"``. Since
# this contains almost 1000 colors, a figure of this would be very large and is thus
# omitted here. You can use the following code to generate the overview yourself ::
#
#     xkcd_fig = plot_colortable(mcolors.XKCD_COLORS)
#     xkcd_fig.savefig("XKCD_Colors.png")
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
# 在这个示例中：
#
# - `matplotlib.colors`: 导入 matplotlib 库中的颜色模块
# - `matplotlib.colors.rgb_to_hsv`: 从颜色模块中导入 RGB 到 HSV 转换函数
# - `matplotlib.colors.to_rgba`: 从颜色模块中导入将颜色转换为 RGBA 格式的函数
# - `matplotlib.figure.Figure.get_size_inches`: 获取 matplotlib 图形对象的尺寸（单位为英寸）
# - `matplotlib.figure.Figure.subplots_adjust`: 调整 matplotlib 图形对象的子图布局
# - `matplotlib.axes.Axes.text`: 在 matplotlib 图形对象的坐标轴上添加文本
# - `matplotlib.patches.Rectangle`: 导入 matplotlib 图形对象中的矩形绘制功能
```
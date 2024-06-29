# `D:\src\scipysrc\matplotlib\galleries\examples\axes_grid1\simple_anchored_artists.py`

```
"""
=======================
Simple Anchored Artists
=======================

This example illustrates the use of the anchored helper classes found in
:mod:`matplotlib.offsetbox` and in :mod:`mpl_toolkits.axes_grid1`.
An implementation of a similar figure, but without use of the toolkit,
can be found in :doc:`/gallery/misc/anchored_artists`.
"""

import matplotlib.pyplot as plt  # 导入 matplotlib.pyplot 模块，用于绘图

def draw_text(ax):
    """
    Draw two text-boxes, anchored by different corners to the upper-left
    corner of the figure.
    """
    from matplotlib.offsetbox import AnchoredText  # 从 matplotlib.offsetbox 模块导入 AnchoredText 类
    # 创建第一个文本框 AnchoredText 对象，锚定在图形的左上角
    at = AnchoredText("Figure 1a",
                      loc='upper left', prop=dict(size=8), frameon=True,
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")  # 设置文本框的样式为圆角
    ax.add_artist(at)  # 将文本框添加到坐标轴中

    # 创建第二个文本框 AnchoredText 对象，锚定在图形的左下角
    at2 = AnchoredText("Figure 1(b)",
                       loc='lower left', prop=dict(size=8), frameon=True,
                       bbox_to_anchor=(0., 1.),
                       bbox_transform=ax.transAxes
                       )
    at2.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")  # 设置文本框的样式为圆角
    ax.add_artist(at2)  # 将文本框添加到坐标轴中


def draw_circle(ax):
    """
    Draw a circle in axis coordinates
    """
    from matplotlib.patches import Circle  # 导入 Circle 类
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea  # 导入 AnchoredDrawingArea 类
    # 创建 AnchoredDrawingArea 对象，包含一个圆形，锚定在图形的右上角
    ada = AnchoredDrawingArea(20, 20, 0, 0,
                              loc='upper right', pad=0., frameon=False)
    p = Circle((10, 10), 10)  # 创建一个半径为 10 的圆形，中心坐标为 (10, 10)
    ada.da.add_artist(p)  # 将圆形添加到 AnchoredDrawingArea 中
    ax.add_artist(ada)  # 将 AnchoredDrawingArea 添加到坐标轴中


def draw_sizebar(ax):
    """
    Draw a horizontal bar with length of 0.1 in data coordinates,
    with a fixed label underneath.
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar  # 导入 AnchoredSizeBar 类
    # 创建 AnchoredSizeBar 对象，绘制长度为 0.1 的水平条，标签为 "1$^{\prime}$"，锚定在图形的下中部
    asb = AnchoredSizeBar(ax.transData,
                          0.1,
                          r"1$^{\prime}$",
                          loc='lower center',
                          pad=0.1, borderpad=0.5, sep=5,
                          frameon=False)
    ax.add_artist(asb)  # 将 AnchoredSizeBar 添加到坐标轴中


fig, ax = plt.subplots()  # 创建一个图形和一个坐标轴
ax.set_aspect(1.)  # 设置坐标轴纵横比为 1

draw_text(ax)  # 调用 draw_text 函数，在坐标轴上绘制文本框
draw_circle(ax)  # 调用 draw_circle 函数，在坐标轴上绘制圆形
draw_sizebar(ax)  # 调用 draw_sizebar 函数，在坐标轴上绘制尺寸条

plt.show()  # 显示绘制的图形
```
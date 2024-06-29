# `D:\src\scipysrc\matplotlib\galleries\examples\misc\anchored_artists.py`

```
"""
================
Anchored Artists
================

This example illustrates the use of the anchored objects without the
helper classes found in :mod:`mpl_toolkits.axes_grid1`. This version
of the figure is similar to the one found in
:doc:`/gallery/axes_grid1/simple_anchored_artists`, but it is
implemented using only the matplotlib namespace, without the help
of additional toolkits.

.. redirect-from:: /gallery/userdemo/anchored_box01
.. redirect-from:: /gallery/userdemo/anchored_box02
.. redirect-from:: /gallery/userdemo/anchored_box03
"""

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox,
                                  DrawingArea, TextArea, VPacker)
from matplotlib.patches import Circle, Ellipse

# 创建文本框，锚定在图形的左上角
def draw_text(ax):
    box = AnchoredOffsetbox(child=TextArea("Figure 1a"),
                            loc="upper left", frameon=True)
    box.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(box)

# 在坐标轴中绘制圆圈
def draw_circles(ax):
    area = DrawingArea(width=40, height=20)
    area.add_artist(Circle((10, 10), 10, fc="tab:blue"))
    area.add_artist(Circle((30, 10), 5, fc="tab:red"))
    box = AnchoredOffsetbox(
        child=area, loc="upper right", pad=0, frameon=False)
    ax.add_artist(box)

# 在数据坐标中绘制椭圆
def draw_ellipse(ax):
    aux_tr_box = AuxTransformBox(ax.transData)
    aux_tr_box.add_artist(Ellipse((0, 0), width=0.1, height=0.15))
    box = AnchoredOffsetbox(child=aux_tr_box, loc="lower left", frameon=True)
    ax.add_artist(box)

# 绘制一个水平条，长度为数据坐标中的0.1，并且有一个固定的标签居中显示在其下方
def draw_sizebar(ax):
    size = 0.1
    text = r"1$^{\prime}$"
    sizebar = AuxTransformBox(ax.transData)
    sizebar.add_artist(Line2D([0, size], [0, 0], color="black"))
    text = TextArea(text)
    packer = VPacker(
        children=[sizebar, text], align="center", sep=5)  # 分隔点数
    ax.add_artist(AnchoredOffsetbox(
        child=packer, loc="lower center", frameon=False,
        pad=0.1, borderpad=0.5))  # 相对于图例字体大小的填充

fig, ax = plt.subplots()
ax.set_aspect(1)  # 设置坐标轴的纵横比为1（即正方形）

draw_text(ax)        # 绘制文本框
draw_circles(ax)     # 绘制圆圈
draw_ellipse(ax)     # 绘制椭圆
draw_sizebar(ax)     # 绘制尺寸条

plt.show()  # 显示图形
```
# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\artist_reference.py`

```
"""
.. _artist_reference:

================================
Reference for Matplotlib artists
================================

This example displays several of Matplotlib's graphics primitives (artists).
A full list of artists is documented at :ref:`the artist API <artist-api>`.

See also :doc:`/gallery/shapes_and_collections/patch_collection`, which groups
all artists into a single `.PatchCollection` instead.

Copyright (c) 2010, Bartosz Telenczuk
BSD License
"""

# 导入 matplotlib 库
import matplotlib.pyplot as plt

# 导入 matplotlib 的不同模块和类
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.path as mpath

# 准备用于 PathPatch 的数据
Path = mpath.Path
codes, verts = zip(*[
    (Path.MOVETO, [0.018, -0.11]),
    (Path.CURVE4, [-0.031, -0.051]),
    (Path.CURVE4, [-0.115, 0.073]),
    (Path.CURVE4, [-0.03, 0.073]),
    (Path.LINETO, [-0.011, 0.039]),
    (Path.CURVE4, [0.043, 0.121]),
    (Path.CURVE4, [0.075, -0.005]),
    (Path.CURVE4, [0.035, -0.027]),
    (Path.CLOSEPOLY, [0.018, -0.11])])

# 创建各种图形对象列表
artists = [
    mpatches.Circle((0, 0), 0.1, ec="none"),
    mpatches.Rectangle((-0.025, -0.05), 0.05, 0.1, ec="none"),
    mpatches.Wedge((0, 0), 0.1, 30, 270, ec="none"),
    mpatches.RegularPolygon((0, 0), 5, radius=0.1),
    mpatches.Ellipse((0, 0), 0.2, 0.1),
    mpatches.Arrow(-0.05, -0.05, 0.1, 0.1, width=0.1),
    mpatches.PathPatch(mpath.Path(verts, codes), ec="none"),
    mpatches.FancyBboxPatch((-0.025, -0.05), 0.05, 0.1, ec="none",
                            boxstyle=mpatches.BoxStyle("Round", pad=0.02)),
    mlines.Line2D([-0.06, 0.0, 0.1], [0.05, -0.05, 0.05], lw=5),
]

# 创建包含多个子图的画布
axs = plt.figure(figsize=(6, 6), layout="constrained").subplots(3, 3)
# 遍历每个子图和对应的图形对象，并设置属性
for i, (ax, artist) in enumerate(zip(axs.flat, artists)):
    # 设置图形对象的颜色
    artist.set(color=mpl.colormaps["hsv"](i / len(artists)))
    # 将图形对象添加到子图中
    ax.add_artist(artist)
    # 设置子图的标题、纵横比和坐标范围
    ax.set(title=type(artist).__name__,
           aspect=1, xlim=(-.2, .2), ylim=(-.2, .2))
    # 关闭子图的坐标轴
    ax.set_axis_off()

# 显示图形
plt.show()

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.path`
#    - `matplotlib.path.Path`
#    - `matplotlib.lines`
#    - `matplotlib.lines.Line2D`
#    - `matplotlib.patches`
#    - `matplotlib.patches.Circle`
#    - `matplotlib.patches.Ellipse`
#    - `matplotlib.patches.Wedge`
#    - `matplotlib.patches.Rectangle`
#    - `matplotlib.patches.Arrow`
#    - `matplotlib.patches.PathPatch`
#    - `matplotlib.patches.FancyBboxPatch`
#    - `matplotlib.patches.RegularPolygon`
#    - `matplotlib.axes.Axes.add_artist`
```
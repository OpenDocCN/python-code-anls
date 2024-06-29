# `D:\src\scipysrc\matplotlib\galleries\examples\shapes_and_collections\hatch_demo.py`

```
"""
==========
Hatch demo
==========

Hatches can be added to most polygons in Matplotlib, including `~.Axes.bar`,
`~.Axes.fill_between`, `~.Axes.contourf`, and children of `~.patches.Polygon`.
They are currently supported in the PS, PDF, SVG, macosx, and Agg backends. The WX
and Cairo backends do not currently support hatching.

See also :doc:`/gallery/images_contours_and_fields/contourf_hatching` for
an example using `~.Axes.contourf`, and
:doc:`/gallery/shapes_and_collections/hatch_style_reference` for swatches
of the existing hatches.

"""

import matplotlib.pyplot as plt  # 导入 matplotlib 的 pyplot 模块，用于绘图
import numpy as np  # 导入 numpy 库，用于数值计算

from matplotlib.patches import Ellipse, Polygon  # 从 matplotlib.patches 模块导入 Ellipse 和 Polygon 类

x = np.arange(1, 5)  # 创建一个包含 1 到 4 的数组
y1 = np.arange(1, 5)  # 创建一个包含 1 到 4 的数组
y2 = np.ones(y1.shape) * 4  # 创建一个与 y1 形状相同的数组，所有元素为 4

fig = plt.figure()  # 创建一个新的图形对象
axs = fig.subplot_mosaic([['bar1', 'patches'], ['bar2', 'patches']])  # 使用 subplot_mosaic 创建包含子图的布局

axs['bar1'].bar(x, y1, edgecolor='black', hatch="/")  # 在 axs['bar1'] 上绘制条形图，设置边框颜色为黑色，填充图案为 /
axs['bar1'].bar(x, y2, bottom=y1, edgecolor='black', hatch='//')  # 在 axs['bar1'] 上绘制叠加的条形图，设置边框颜色为黑色，填充图案为 //

axs['bar2'].bar(x, y1, edgecolor='black', hatch=['--', '+', 'x', '\\'])  # 在 axs['bar2'] 上绘制条形图，设置边框颜色为黑色，填充图案为列表中的不同样式
axs['bar2'].bar(x, y2, bottom=y1, edgecolor='black',
                hatch=['*', 'o', 'O', '.'])  # 在 axs['bar2'] 上绘制叠加的条形图，设置边框颜色为黑色，填充图案为列表中的不同样式

x = np.arange(0, 40, 0.2)  # 创建一个从 0 到 40 的数组，步长为 0.2
axs['patches'].fill_between(x, np.sin(x) * 4 + 30, y2=0,
                            hatch='///', zorder=2, fc='c')  # 在 axs['patches'] 上填充区域，设置填充样式为 ///，设置层次为 2，填充颜色为 'c'
axs['patches'].add_patch(Ellipse((4, 50), 10, 10, fill=True,
                                 hatch='*', facecolor='y'))  # 在 axs['patches'] 上添加一个椭圆形对象，设置填充样式为 *，填充颜色为 'y'
axs['patches'].add_patch(Polygon([(10, 20), (30, 50), (50, 10)],
                                 hatch='\\/...', facecolor='g'))  # 在 axs['patches'] 上添加一个多边形对象，设置填充样式为 \\/...，填充颜色为 'g'
axs['patches'].set_xlim([0, 40])  # 设置 axs['patches'] 的 x 轴范围为 [0, 40]
axs['patches'].set_ylim([10, 60])  # 设置 axs['patches'] 的 y 轴范围为 [10, 60]
axs['patches'].set_aspect(1)  # 设置 axs['patches'] 的纵横比为 1
plt.show()  # 显示图形

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - `matplotlib.patches`
#    - `matplotlib.patches.Ellipse`
#    - `matplotlib.patches.Polygon`
#    - `matplotlib.axes.Axes.add_patch`
#    - `matplotlib.patches.Patch.set_hatch`
#    - `matplotlib.axes.Axes.bar` / `matplotlib.pyplot.bar`
```
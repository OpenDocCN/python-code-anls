# `D:\src\scipysrc\matplotlib\galleries\users_explain\colors\colors.py`

```py
"""
.. redirect-from:: /tutorials/colors/colors

.. _colors_def:

*****************
Specifying colors
*****************

Color formats
=============

Matplotlib recognizes the following formats to specify a color.

+--------------------------------------+--------------------------------------+
| Format                               | Example                              |
+======================================+======================================+
| RGB or RGBA (red, green, blue, alpha)| - ``(0.1, 0.2, 0.5)``                |
| tuple of float values in a closed    | - ``(0.1, 0.2, 0.5, 0.3)``           |
| interval [0, 1].                     |                                      |
+--------------------------------------+--------------------------------------+
| Case-insensitive hex RGB or RGBA     | - ``'#0f0f0f'``                      |
| string.                              | - ``'#0f0f0f80'``                    |
+--------------------------------------+--------------------------------------+
| Case-insensitive RGB or RGBA string  | - ``'#abc'`` as ``'#aabbcc'``        |
| equivalent hex shorthand of          | - ``'#fb1'`` as ``'#ffbb11'``        |
| duplicated characters.               |                                      |
+--------------------------------------+--------------------------------------+
| String representation of float value | - ``'0'`` as black                   |
| in closed interval ``[0, 1]`` for    | - ``'1'`` as white                   |
| grayscale values.                    | - ``'0.8'`` as light gray            |
+--------------------------------------+--------------------------------------+
| Single character shorthand notation  | - ``'b'`` as blue                    |
| for some basic colors.               | - ``'g'`` as green                   |
|                                      | - ``'r'`` as red                     |
| .. note::                            | - ``'c'`` as cyan                    |
|    The colors green, cyan, magenta,  | - ``'m'`` as magenta                 |
|    and yellow do not coincide with   | - ``'y'`` as yellow                  |
|    X11/CSS4 colors. Their particular | - ``'k'`` as black                   |
|    shades were chosen for better     | - ``'w'`` as white                   |
|    visibility of colored lines       |                                      |
|    against typical backgrounds.      |                                      |
+--------------------------------------+--------------------------------------+
| Case-insensitive X11/CSS4 color name | - ``'aquamarine'``                   |
| with no spaces.                      | - ``'mediumseagreen'``               |
+--------------------------------------+--------------------------------------+
| Case-insensitive color name from     | - ``'xkcd:sky blue'``                |
| `xkcd color survey`_ with ``'xkcd:'``| - ``'xkcd:eggshell'``                |
"""

注释：
# Case-insensitive Tableau颜色，从'T10'分类调色板中选择
# - 'tab:blue'
# - 'tab:orange'
# - 'tab:green'
# - 'tab:red'
# - 'tab:purple'
# - 'tab:brown'
# - 'tab:pink'
# - 'tab:gray'
# - 'tab:olive'
# - 'tab:cyan'

# 默认颜色循环的注意事项
# 这是默认的颜色循环。

# "CN"颜色规范，其中'C'之前是一个作为默认属性循环索引的数字
# - 'C0'
# - 'C1'
# 在绘图时，Matplotlib根据需要索引颜色，并且如果循环不包含该颜色，则默认为黑色。
# :rc:`axes.prop_cycle`

# 一个包含上述颜色格式之一和一个alpha浮点数的元组。
# - ('green', 0.3)
# - ('#f00', 0.9)
# .. versionadded:: 3.8

# xkcd颜色调查: https://xkcd.com/color/rgb/

# 参见：
# 下面的链接提供了有关Matplotlib中颜色的更多信息。
# * :doc:`/gallery/color/color_demo` 示例
# * `matplotlib.colors` API
# * :doc:`/gallery/color/named_colors` 示例

# "Red", "Green" 和 "Blue" 是这些颜色的强度。组合起来，它们表示色彩空间。

# 透明度
# ============

# 颜色的*alpha*值指定其透明度，其中0表示完全透明，1表示完全不透明。当颜色是半透明时，背景颜色将透过显示。

# *alpha*值通过以下公式确定最终的颜色，根据背景颜色和前景颜色的混合程度计算：

# .. math::

#    RGB_{result} = RGB_{background} * (1 - \\alpha) + RGB_{foreground} * \\alpha
# 导入绘图库和数学库
import matplotlib.pyplot as plt
import numpy as np

# 导入相关的绘图对象
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.patches as mpatch

# 创建一个新的图形和坐标系
fig = plt.figure(figsize=[9, 5])
ax = fig.add_axes([0, 0, 1, 1])

# 计算重叠的颜色名称
overlap = {name for name in mcolors.CSS4_COLORS
           if f'xkcd:{name}' in mcolors.XKCD_COLORS}

# 计算每行显示的颜色数
n_groups = 3
n_rows = len(overlap) // n_groups + 1

for j, color_name in enumerate(sorted(overlap)):
    # 使用颜色名称从CSS4_COLORS字典中获取对应的RGBA颜色值
    css4 = mcolors.CSS4_COLORS[color_name]
    # 使用"xkcd:颜色名称"从XKCD_COLORS字典中获取对应的RGBA颜色值，并转换为大写
    xkcd = mcolors.XKCD_COLORS[f'xkcd:{color_name}'].upper()

    # 根据颜色的感知亮度选择文本颜色
    # 将两种颜色的RGBA数组合并为一个数组
    rgba = mcolors.to_rgba_array([css4, xkcd])
    # 计算亮度值
    luma = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]
    # 根据亮度值判断CSS4颜色对应的文本颜色
    css4_text_color = 'k' if luma[0] > 0.5 else 'w'
    # 根据亮度值判断XKCD颜色对应的文本颜色
    xkcd_text_color = 'k' if luma[1] > 0.5 else 'w'

    # 计算列偏移量
    col_shift = (j // n_rows) * 3
    # 计算当前行索引
    y_pos = j % n_rows
    # 配置文本参数，设置字体大小和粗体（如果CSS4颜色和XKCD颜色相同则设置粗体）
    text_args = dict(fontsize=10, weight='bold' if css4 == xkcd else None)
    # 在图形上添加CSS4颜色方块
    ax.add_patch(mpatch.Rectangle((0 + col_shift, y_pos), 1, 1, color=css4))
    # 在图形上添加XKCD颜色方块
    ax.add_patch(mpatch.Rectangle((1 + col_shift, y_pos), 1, 1, color=xkcd))
    # 在图形上添加CSS4颜色名称文本
    ax.text(0.5 + col_shift, y_pos + .7, css4,
            color=css4_text_color, ha='center', **text_args)
    # 在图形上添加XKCD颜色名称文本
    ax.text(1.5 + col_shift, y_pos + .7, xkcd,
            color=xkcd_text_color, ha='center', **text_args)
    # 在图形上添加颜色名称的补充文本
    ax.text(2 + col_shift, y_pos + .7, f'  {color_name}', **text_args)
# 对每一个分组进行循环处理，范围是从 0 到 n_groups-1
for g in range(n_groups):
    # 在图形 ax 上绘制水平线，每行的位置为 range(n_rows)，从 3*g 到 3*g + 2.8，颜色为浅灰色，线宽为 1
    ax.hlines(range(n_rows), 3*g, 3*g + 2.8, color='0.7', linewidth=1)
    # 在图形 ax 上添加文本，位于 (0.5 + 3*g, -0.3) 处，内容为 'X11/CSS4'，水平对齐方式为居中
    ax.text(0.5 + 3*g, -0.3, 'X11/CSS4', ha='center')
    # 在图形 ax 上添加文本，位于 (1.5 + 3*g, -0.3) 处，内容为 'xkcd'，水平对齐方式为居中
    ax.text(1.5 + 3*g, -0.3, 'xkcd', ha='center')

# 设置 x 轴的显示范围，从 0 到 3 * n_groups
ax.set_xlim(0, 3 * n_groups)
# 设置 y 轴的显示范围，从 n_rows 到 -1
ax.set_ylim(n_rows, -1)
# 关闭所有轴的显示
ax.axis('off')

# 显示绘制好的图形
plt.show()
```
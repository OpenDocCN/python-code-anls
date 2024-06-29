# `D:\src\scipysrc\matplotlib\galleries\examples\lines_bars_and_markers\marker_reference.py`

```py
"""
================
Marker reference
================

Matplotlib supports multiple categories of markers which are selected using
the ``marker`` parameter of plot commands:

- `Unfilled markers`_
- `Filled markers`_
- `Markers created from TeX symbols`_
- `Markers created from Paths`_

For a list of all markers see also the `matplotlib.markers` documentation.

For example usages see
:doc:`/gallery/lines_bars_and_markers/scatter_star_poly`.

.. redirect-from:: /gallery/shapes_and_collections/marker_path
"""

# 导入 matplotlib.pyplot 库，用于绘图
import matplotlib.pyplot as plt

# 导入 Line2D 类和 MarkerStyle 类以及 Affine2D 类，用于处理线条、标记和仿射变换
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

# 定义文本样式字典
text_style = dict(horizontalalignment='right', verticalalignment='center',
                  fontsize=12, fontfamily='monospace')

# 定义标记样式字典
marker_style = dict(linestyle=':', color='0.8', markersize=10,
                    markerfacecolor="tab:blue", markeredgecolor="tab:blue")


def format_axes(ax):
    """
    格式化绘图轴，设置边距和关闭坐标轴显示
    """
    ax.margins(0.2)
    ax.set_axis_off()
    ax.invert_yaxis()


def split_list(a_list):
    """
    将列表分割成两半
    """
    i_half = len(a_list) // 2
    return a_list[:i_half], a_list[i_half:]


# %%
# Unfilled markers
# ================
# 绘制未填充的标记点，单色显示。

fig, axs = plt.subplots(ncols=2)
fig.suptitle('Un-filled markers', fontsize=14)

# 过滤出未填充的标记点，并排除不起作用的标记及设置
unfilled_markers = [m for m, func in Line2D.markers.items()
                    if func != 'nothing' and m not in Line2D.filled_markers]

# 在子图中绘制未填充的标记点
for ax, markers in zip(axs, split_list(unfilled_markers)):
    for y, marker in enumerate(markers):
        ax.text(-0.5, y, repr(marker), **text_style)  # 在图中标注标记点的名称
        ax.plot([y] * 3, marker=marker, **marker_style)  # 绘制标记点
    format_axes(ax)  # 格式化绘图轴

# %%
# Filled markers
# ==============

fig, axs = plt.subplots(ncols=2)
fig.suptitle('Filled markers', fontsize=14)

# 绘制已填充的标记点
for ax, markers in zip(axs, split_list(Line2D.filled_markers)):
    for y, marker in enumerate(markers):
        ax.text(-0.5, y, repr(marker), **text_style)  # 在图中标注标记点的名称
        ax.plot([y] * 3, marker=marker, **marker_style)  # 绘制标记点
    format_axes(ax)  # 格式化绘图轴

# %%
# .. _marker_fill_styles:
#
# Marker fill styles
# ------------------
# 已填充标记点的填充样式可以分别指定边缘颜色和填充颜色。
# 此外，还可以配置 ``fillstyle`` 为未填充、完全填充或在各个方向上半填充。
# 半填充样式使用 ``markerfacecoloralt`` 作为辅助填充颜色。

fig, ax = plt.subplots()
fig.suptitle('Marker fillstyle', fontsize=14)
fig.subplots_adjust(left=0.4)

# 定义已填充标记点的样式字典
filled_marker_style = dict(marker='o', linestyle=':', markersize=15,
                           color='darkgrey',
                           markerfacecolor='tab:blue',
                           markerfacecoloralt='lightsteelblue',
                           markeredgecolor='brown')

# 在图中绘制不同填充样式的标记点
for y, fill_style in enumerate(Line2D.fillStyles):
    ax.text(-0.5, y, repr(fill_style), **text_style)  # 在图中标注填充样式名称
    ax.plot([y] * 3, fillstyle=fill_style, **filled_marker_style)  # 绘制标记点
# %%
# Markers created from TeX symbols
# ================================
#
# Use :ref:`MathText <mathtext>` to define custom marker symbols using TeX
# syntax. This allows symbols like "$\u266B$" (musical flat). Refer to the
# `STIX font table <http://www.stixfonts.org/allGlyphs.html>`_ for an overview
# of available symbols. For practical examples, see the
# :doc:`/gallery/text_labels_and_annotations/stix_fonts_demo`.

import matplotlib.pyplot as plt

# Create a new figure and axis
fig, ax = plt.subplots()
fig.suptitle('Mathtext markers', fontsize=14)
fig.subplots_adjust(left=0.4)

# Update marker style properties
marker_style.update(markeredgecolor="none", markersize=15)

# Define a list of markers using TeX symbols
markers = ["$1$", r"$\frac{1}{2}$", "$f$", "$\u266B$", r"$\mathcal{A}$"]

# Iterate through markers and plot them on the axis
for y, marker in enumerate(markers):
    # Display the marker symbol as plain text using ax.text
    ax.text(-0.5, y, repr(marker).replace("$", r"\$"), **text_style)
    # Plot the marker symbol at position [y, y, y] on the axis
    ax.plot([y] * 3, marker=marker, **marker_style)

# Apply formatting to the axis
format_axes(ax)

# %%
# Markers created from Paths
# ==========================
#
# Define custom markers using paths created with `matplotlib.path.Path`. This
# example demonstrates how to create markers using geometric paths like stars
# and circles, as well as combining them to form more complex shapes.

import numpy as np
import matplotlib.path as mpath

# Generate paths for star, circle, and a cut-out star in a circle
star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
cut_star = mpath.Path(
    vertices=np.concatenate([circle.vertices, star.vertices[::-1, ...]]),
    codes=np.concatenate([circle.codes, star.codes]))

# Create a new figure and axis
fig, ax = plt.subplots()
fig.suptitle('Path markers', fontsize=14)
fig.subplots_adjust(left=0.4)

# Dictionary of marker names and their corresponding paths
markers = {'star': star, 'circle': circle, 'cut_star': cut_star}

# Iterate through markers, display their names, and plot them on the axis
for y, (name, marker) in enumerate(markers.items()):
    ax.text(-0.5, y, name, **text_style)
    ax.plot([y] * 3, marker=marker, **marker_style)

# Apply formatting to the axis
format_axes(ax)

# %%
# Advanced marker modifications with transform
# ============================================
#
# Customize markers using transformations with the `MarkerStyle` constructor.
# This example demonstrates applying rotations to various marker shapes.

from matplotlib.transforms import Affine2D

# Common marker style properties excluding 'marker'
common_style = {k: v for k, v in filled_marker_style.items() if k != 'marker'}

# List of rotation angles in degrees
angles = [0, 10, 20, 30, 45, 60, 90]

# Create a new figure and axis
fig, ax = plt.subplots()
fig.suptitle('Rotated markers', fontsize=14)

# Plot filled markers with different rotation angles
ax.text(-0.5, 0, 'Filled marker', **text_style)
for x, theta in enumerate(angles):
    t = Affine2D().rotate_deg(theta)
    ax.plot(x, 0, marker=MarkerStyle('o', 'left', t), **common_style)

# Plot unfilled markers with different rotation angles
ax.text(-0.5, 1, 'Un-filled marker', **text_style)
for x, theta in enumerate(angles):
    t = Affine2D().rotate_deg(theta)
    ax.plot(x, 1, marker=MarkerStyle('1', 'left', t), **common_style)

# Plot equation markers with different rotation angles
ax.text(-0.5, 2, 'Equation marker', **text_style)
for x, theta in enumerate(angles):
    t = Affine2D().rotate_deg(theta)
    eq = r'$\frac{1}{x}$'
    ax.plot(x, 2, marker=MarkerStyle(eq, 'left', t), **common_style)

# Display rotation angles below the equation markers
for x, theta in enumerate(angles):
    ax.text(x, 2.5, f"{theta}°", horizontalalignment="center")

# Apply formatting to the axis
format_axes(ax)

# Adjust layout for better presentation
fig.tight_layout()

# %%
# 设置标记的端点风格和连接风格
# =======================================
#
# 标记默认具有默认的端点和连接风格，但在创建 MarkerStyle 时可以进行定制。

from matplotlib.markers import CapStyle, JoinStyle  # 导入标记的端点和连接风格

# 内部标记样式设置
marker_inner = dict(markersize=35,
                    markerfacecolor='tab:blue',
                    markerfacecoloralt='lightsteelblue',
                    markeredgecolor='brown',
                    markeredgewidth=8,
                    )

# 外部标记样式设置
marker_outer = dict(markersize=35,
                    markerfacecolor='tab:blue',
                    markerfacecoloralt='lightsteelblue',
                    markeredgecolor='white',
                    markeredgewidth=1,
                    )

# 创建包含两个子图的图像和坐标轴对象
fig, ax = plt.subplots()
fig.suptitle('Marker CapStyle', fontsize=14)  # 设置主标题
fig.subplots_adjust(left=0.1)  # 调整子图的左边距

# 遍历端点风格枚举，并在子图中展示每种风格的效果
for y, cap_style in enumerate(CapStyle):
    ax.text(-0.5, y, cap_style.name, **text_style)  # 在指定位置添加文本标签，显示端点风格的名称
    for x, theta in enumerate(angles):
        t = Affine2D().rotate_deg(theta)  # 创建一个旋转变换对象
        m = MarkerStyle('1', transform=t, capstyle=cap_style)  # 创建一个带有指定端点风格的标记样式对象
        ax.plot(x, y, marker=m, **marker_inner)  # 在子图中添加内部标记
        ax.plot(x, y, marker=m, **marker_outer)  # 在子图中添加外部标记
        ax.text(x, len(CapStyle) - .5, f'{theta}°', ha='center')  # 在子图中添加角度标签
format_axes(ax)  # 调用自定义函数，格式化坐标轴样式

# %%
# 修改连接风格：

fig, ax = plt.subplots()
fig.suptitle('Marker JoinStyle', fontsize=14)  # 设置主标题
fig.subplots_adjust(left=0.05)  # 调整子图的左边距

# 遍历连接风格枚举，并在子图中展示每种风格的效果
for y, join_style in enumerate(JoinStyle):
    ax.text(-0.5, y, join_style.name, **text_style)  # 在指定位置添加文本标签，显示连接风格的名称
    for x, theta in enumerate(angles):
        t = Affine2D().rotate_deg(theta)  # 创建一个旋转变换对象
        m = MarkerStyle('*', transform=t, joinstyle=join_style)  # 创建一个带有指定连接风格的标记样式对象
        ax.plot(x, y, marker=m, **marker_inner)  # 在子图中添加内部标记
        ax.text(x, len(JoinStyle) - .5, f'{theta}°', ha='center')  # 在子图中添加角度标签
format_axes(ax)  # 调用自定义函数，格式化坐标轴样式

plt.show()  # 显示图形
```
# `D:\src\scipysrc\matplotlib\galleries\users_explain\colors\colorbar_only.py`

```py
"""
.. redirect-from:: /tutorials/colors/colorbar_only

=============================
Customized Colorbars Tutorial
=============================

This tutorial shows how to build and customize standalone colorbars, i.e.
without an attached plot.

A `~.Figure.colorbar` needs a "mappable" (`matplotlib.cm.ScalarMappable`)
object (typically, an image) which indicates the colormap and the norm to be
used.  In order to create a colorbar without an attached image, one can instead
use a `.ScalarMappable` with no associated data.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# %%
# Basic continuous colorbar
# -------------------------
# Here, we create a basic continuous colorbar with ticks and labels.
#
# The arguments to the `~.Figure.colorbar` call are the `.ScalarMappable`
# (constructed using the *norm* and *cmap* arguments), the axes where the
# colorbar should be drawn, and the colorbar's orientation.
#
# For more information see the `~matplotlib.colorbar` API.

fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')

cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=5, vmax=10)

# Create a colorbar using ScalarMappable with specified colormap and normalization
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal', label='Some Units')

# %%
# Colorbar attached next to a pre-existing axes
# ---------------------------------------------
# All examples in this tutorial (except this one) show a standalone colorbar on
# its own figure, but it is possible to display the colorbar *next* to a
# pre-existing Axes *ax* by passing ``ax=ax`` to the colorbar() call (meaning
# "draw the colorbar next to *ax*") rather than ``cax=ax`` (meaning "draw the
# colorbar on *ax*").

fig, ax = plt.subplots(layout='constrained')

# Create a colorbar attached to the existing axes using ScalarMappable and colormap 'magma'
fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap='magma'),
             ax=ax, orientation='vertical', label='a colorbar label')

# %%
# Discrete and extended colorbar with continuous colorscale
# ---------------------------------------------------------
# The following example shows how to make a discrete colorbar based on a
# continuous cmap.  We use `matplotlib.colors.BoundaryNorm` to describe the
# interval boundaries (which must be in increasing order), and further pass the
# *extend* argument to it to further display "over" and "under" colors (which
# are used for data outside of the norm range).

fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')

cmap = mpl.cm.viridis
bounds = [-1, 2, 5, 7, 12, 15]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

# Create a colorbar with discrete intervals based on BoundaryNorm and viridis colormap
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax, orientation='horizontal',
             label="Discrete intervals with extend='both' keyword")

# %%
# Colorbar with arbitrary colors
# ------------------------------
# The following example still uses a `.BoundaryNorm` to describe discrete
# interval boundaries, but now uses a `matplotlib.colors.ListedColormap` to
# 创建一个新的图形对象和轴对象，指定图形大小和布局约束
fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')

# 定义一个离散的颜色映射，包含颜色列表，使用.with_extremes()方法设置溢出范围的颜色
cmap = (mpl.colors.ListedColormap(['red', 'green', 'blue', 'cyan'])
        .with_extremes(under='yellow', over='magenta'))

# 指定离散颜色映射的边界值列表
bounds = [1, 2, 4, 7, 8]

# 创建边界规范对象，用于归一化颜色映射
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# 向图形添加颜色条，使用ScalarMappable对象表示颜色映射，并指定其参数
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),  # 指定颜色映射和规范
    cax=ax,  # 将颜色条放置在指定的轴对象上
    orientation='horizontal',  # 水平方向显示颜色条
    extend='both',  # 在颜色条两端显示溢出值
    spacing='proportional',  # 颜色条段的长度与对应的间隔比例成正比
    label='Discrete intervals, some other units',  # 颜色条的标签
)

# %%
# Colorbar with custom extension lengths
# --------------------------------------
# 创建一个自定义长度扩展的颜色条，用于离散间隔的颜色显示。
# 通过设置``extendfrac='auto'``，使每个扩展的长度与内部颜色段的长度相同。

fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')

# 定义另一个离散的颜色映射，包含不同的颜色列表，并设置溢出范围的颜色
cmap = (mpl.colors.ListedColormap(['royalblue', 'cyan', 'yellow', 'orange'])
        .with_extremes(over='red', under='blue'))

# 指定新的边界值列表
bounds = [-1.0, -0.5, 0.0, 0.5, 1.0]

# 创建新的边界规范对象
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# 向图形添加颜色条，使用ScalarMappable对象表示颜色映射，并指定其参数
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),  # 指定颜色映射和规范
    cax=ax,  # 将颜色条放置在指定的轴对象上
    orientation='horizontal',  # 水平方向显示颜色条
    extend='both',  # 在颜色条两端显示溢出值
    extendfrac='auto',  # 自动调整扩展的长度与内部颜色段的长度相同
    spacing='uniform',  # 颜色条段的长度保持均匀
    label='Custom extension lengths, some other units',  # 颜色条的标签
)

plt.show()  # 显示图形
```
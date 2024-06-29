# `D:\src\scipysrc\matplotlib\galleries\users_explain\axes\legend_guide.py`

```py
"""
.. redirect-from:: /tutorials/intermediate/legend_guide

.. _legend_guide:

============
Legend guide
============

.. currentmodule:: matplotlib.pyplot

This legend guide extends the `~.Axes.legend` docstring -
please read it before proceeding with this guide.

This guide makes use of some common terms, which are documented here for
clarity:

.. glossary::

    legend entry
        A legend is made up of one or more legend entries. An entry is made up
        of exactly one key and one label.

    legend key
        The colored/patterned marker to the left of each legend label.

    legend label
        The text which describes the handle represented by the key.

    legend handle
        The original object which is used to generate an appropriate entry in
        the legend.


Controlling the legend entries
==============================

Calling :func:`legend` with no arguments automatically fetches the legend
handles and their associated labels. This functionality is equivalent to::

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

The :meth:`~matplotlib.axes.Axes.get_legend_handles_labels` function returns
a list of handles/artists which exist on the Axes which can be used to
generate entries for the resulting legend - it is worth noting however that
not all artists can be added to a legend, at which point a "proxy" will have
to be created (see :ref:`proxy_legend_handles` for further details).

.. note::
    Artists with an empty string as label or with a label starting with an
    underscore, "_", will be ignored.

For full control of what is being added to the legend, it is common to pass
the appropriate handles directly to :func:`legend`::

    fig, ax = plt.subplots()
    line_up, = ax.plot([1, 2, 3], label='Line 2')
    line_down, = ax.plot([3, 2, 1], label='Line 1')
    ax.legend(handles=[line_up, line_down])

Renaming legend entries
-----------------------

When the labels cannot directly be set on the handles, they can be directly passed to
`.Axes.legend`::

    fig, ax = plt.subplots()
    line_up, = ax.plot([1, 2, 3], label='Line 2')
    line_down, = ax.plot([3, 2, 1], label='Line 1')
    ax.legend([line_up, line_down], ['Line Up', 'Line Down'])


If the handles are not directly accessible, for example when using some
`Third-party packages <https://matplotlib.org/mpl-third-party/>`_, they can be accessed
via `.Axes.get_legend_handles_labels`. Here we use a dictionary to rename existing
labels::

    my_map = {'Line Up':'Up', 'Line Down':'Down'}

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [my_map[l] for l in labels])


.. _proxy_legend_handles:

Creating artists specifically for adding to the legend (aka. Proxy artists)
===========================================================================

Not all handles can be turned into legend entries automatically,
so it is often necessary to create an artist which *can*. Legend handles
"""
# 导入 matplotlib 库
import matplotlib.pyplot as plt

# 导入 matplotlib 中的 patches 和 lines 模块
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# 创建一个新的图形和一个轴对象
fig, ax = plt.subplots()

# 创建一个红色的图例条目，标签为 'The red data'
red_patch = mpatches.Patch(color='red', label='The red data')

# 将红色图例条目添加到轴的图例中
ax.legend(handles=[red_patch])

# 显示图形
plt.show()

# %%
# 支持多种类型的图例条目。除了创建纯色的图例条目外，
# 还可以创建带有标记的线条：

fig, ax = plt.subplots()

# 创建一个蓝色的线条，带有星形标记，标签为 'Blue stars'
blue_line = mlines.Line2D([], [], color='blue', marker='*',
                          markersize=15, label='Blue stars')

# 将蓝色线条添加到轴的图例中
ax.legend(handles=[blue_line])

# 显示图形
plt.show()

# %%
# 图例位置
# ===============
#
# 可以通过关键字参数 *loc* 指定图例的位置。更多细节请参考 :func:`legend` 中的文档。
#
# 关键字参数 ``bbox_to_anchor`` 可以在手动指定图例位置时提供很大的控制度。
# 例如，如果要将轴的图例放置在图形的右上角而不是轴的角落处，
# 可以简单地指定该位置的坐标系和坐标系统的位置::
#
#     ax.legend(bbox_to_anchor=(1, 1),
#               bbox_transform=fig.transFigure)
#
# 更多自定义图例位置的例子：

fig, ax_dict = plt.subplot_mosaic([['top', 'top'], ['bottom', 'BLANK']],
                                  empty_sentinel="BLANK")

ax_dict['top'].plot([1, 2, 3], label="test1")
ax_dict['top'].plot([3, 2, 1], label="test2")
# 将一个图例放置在子图上方，扩展以充分利用给定的边界框。
ax_dict['top'].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                      ncols=2, mode="expand", borderaxespad=0.)

ax_dict['bottom'].plot([1, 2, 3], label="test1")
ax_dict['bottom'].plot([3, 2, 1], label="test2")
# 将一个图例放置在较小子图的右侧。
ax_dict['bottom'].legend(bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)

# %%
# 图形的图例
# --------------
#
# 有时相对于(子)图而不是单独的轴放置图例更有意义。
# 通过使用 *constrained layout* 并在 *loc* 关键字参数的开始处指定 "outside"，
# 可以将图例放置在(子)图的外部。

fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')

axs['left'].plot([1, 2, 3], label="test1")
axs['left'].plot([3, 2, 1], label="test2")

axs['right'].plot([1, 2, 3], 'C2', label="test3")
axs['right'].plot([3, 2, 1], 'C3', label="test4")
# 将一个图例放置在较小子图的右侧。
fig.legend(loc='outside upper right')

# %%
# 这接受与正常 *loc* 关键字略有不同的语法，
# 其中 "outside right upper" 与 "outside upper right" 是不同的。
#
ucl = ['upper', 'center', 'lower']
lcr = ['left', 'center', 'right']
# 创建一个新的图形和坐标系，指定图形大小为6x4，布局为'constrained'，背景色为浅灰色
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')

# 在坐标系上绘制一条线，数据为([1, 2], [1, 2])，标签为'TEST'
ax.plot([1, 2], [1, 2], label='TEST')

# 在图形的各个位置创建图例，包括左上、上中、右上、左下、下中、右下
for loc in [
        'outside upper left',
        'outside upper center',
        'outside upper right',
        'outside lower left',
        'outside lower center',
        'outside lower right']:
    fig.legend(loc=loc, title=loc)

# 创建另一个新的图形和坐标系，指定图形大小为6x4，布局为'constrained'，背景色为浅灰色
fig, ax = plt.subplots(figsize=(6, 4), layout='constrained', facecolor='0.7')

# 在坐标系上绘制一条线，数据为([1, 2], [1, 2])，标签为'test'
ax.plot([1, 2], [1, 2], label='test')

# 在图形的各个位置创建图例，包括左上、右上、左下、右下
for loc in [
        'outside left upper',
        'outside right upper',
        'outside left lower',
        'outside right lower']:
    fig.legend(loc=loc, title=loc)


# %%
# 同一坐标系上的多个图例
# =======================
#
# 有时，将图例条目分割成多个图例更清晰。
# 虽然直觉上可以多次调用 :func:`legend` 函数来实现这一点，
# 但实际上在同一个坐标系上只能存在一个图例。这样设计是为了可以重复调用
# :func:`legend` 来更新最新的坐标系上的图例。要保留旧的图例实例，必须手动
# 将它们添加到坐标系上：

fig, ax = plt.subplots()
line1, = ax.plot([1, 2, 3], label="Line 1", linestyle='--')
line2, = ax.plot([3, 2, 1], label="Line 2", linewidth=4)

# 为第一条线创建一个图例，放置在右上角
first_legend = ax.legend(handles=[line1], loc='upper right')

# 将第一个图例手动添加到坐标系上
ax.add_artist(first_legend)

# 创建第二条线的另一个图例，放置在右下角
ax.legend(handles=[line2], loc='lower right')

plt.show()

# %%
# 图例处理器
# =========
#
# 为了创建图例条目，需要将句柄作为参数传递给适当的 :class:`~matplotlib.legend_handler.HandlerBase` 子类。
# 使用哪个处理器子类取决于以下规则：
#
# 1. 使用 ``handler_map`` 关键字更新 :func:`~matplotlib.legend.Legend.get_legend_handler_map` 中的值。
# 2. 检查句柄是否在新创建的 ``handler_map`` 中。
# 3. 检查句柄的类型是否在新创建的 ``handler_map`` 中。
# 4. 检查句柄类型的任何mro是否在新创建的 ``handler_map`` 中。
#
# 为了完整起见，这些逻辑主要实现在 :func:`~matplotlib.legend.Legend.get_legend_handler` 中。
#
# 所有这些灵活性意味着我们有必要的钩子来为自己的图例键类型实现自定义处理器。
#
# 使用自定义处理器的最简单示例是实例化一个现有的 `.legend_handler.HandlerBase` 子类。
# 为了简单起见，让我们选择 `.legend_handler.HandlerLine2D`，它接受一个 *numpoints* 参数
# （*numpoints* 也是 :func:`legend` 函数的一个关键字，为了方便）。然后我们可以将实例到处理器的映射
# 作为关键字传递给图例。

from matplotlib.legend_handler import HandlerLine2D
fig, ax = plt.subplots()
# 创建一个新的图形和一个轴对象

line1, = ax.plot([3, 2, 1], marker='o', label='Line 1')
line2, = ax.plot([1, 2, 3], marker='o', label='Line 2')
# 在轴上绘制两条曲线，分别用圆圈标记，设置标签为 'Line 1' 和 'Line 2'，并获取它们的句柄

ax.legend(handler_map={line1: HandlerLine2D(numpoints=4)}, handlelength=4)
# 创建图例，使用自定义的处理器 `HandlerLine2D` 来处理 'Line 1' 的图例，设置图例句柄长度为 4

# %%
# 如你所见，“Line 1” 现在有 4 个标记点，“Line 2” 有 2 个（默认值）。我们还使用了
# `handlelength` 关键字来增加图例条目的长度，以适应更大的图例条目。
# 尝试上面的代码，只需将映射的键从 `line1` 改为 `type(line1)`。注意现在两个 `.Line2D`
# 实例都有 4 个标记点。
#
# 除了处理常见的绘图类型如误差线、柱状图等的处理器外，默认的 `handler_map` 还包括一个
# 特殊的 `tuple` 处理器（`.legend_handler.HandlerTuple`），它简单地将每个给定元组中的
# 句柄叠加在一起。以下示例演示了如何将两个图例键叠加在一起：

from numpy.random import randn

z = randn(10)

fig, ax = plt.subplots()
red_dot, = ax.plot(z, "ro", markersize=15)
# 在部分数据上绘制白色十字。 
white_cross, = ax.plot(z[:5], "w+", markeredgewidth=3, markersize=15)

ax.legend([red_dot, (red_dot, white_cross)], ["Attr A", "Attr A+B"])

# %%
# `.legend_handler.HandlerTuple` 类还可以用于将多个图例键分配给同一条目：

from matplotlib.legend_handler import HandlerLine2D, HandlerTuple

fig, ax = plt.subplots()
p1, = ax.plot([1, 2.5, 3], 'r-d')
p2, = ax.plot([3, 2, 1], 'k-o')

l = ax.legend([(p1, p2)], ['Two keys'], numpoints=1,
              handler_map={tuple: HandlerTuple(ndivide=None)})

# %%
# 实现自定义图例处理器
# ------------------------------------
#
# 可以实现自定义处理器将任何句柄转换为图例键（句柄不一定需要是 matplotlib 艺术家）。
# 处理器必须实现一个 `legend_artist` 方法，该方法返回图例使用的单个艺术家。`legend_artist`
# 的必需签名在 `~.legend_handler.HandlerBase.legend_artist` 中有文档记录。

import matplotlib.patches as mpatches

class AnyObject:
    pass

class AnyObjectHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
                                   edgecolor='black', hatch='xx', lw=3,
                                   transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch

fig, ax = plt.subplots()

ax.legend([AnyObject()], ['My first handler'],
          handler_map={AnyObject: AnyObjectHandler()})

# %%
# 或者，如果我们希望全局接受 `AnyObject` 实例而不需要每次手动设置 `handler_map` 关键字，
# 我们可以使用以下方式注册新的处理器::
#
# 从 matplotlib 中导入图例处理器基类 HandlerPatch
from matplotlib.legend_handler import HandlerPatch

# 定义一个椭圆形图例处理器，继承自 HandlerPatch 类
class HandlerEllipse(HandlerPatch):
    # 创建图例艺术家的方法，用于在图例中绘制椭圆形
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # 计算椭圆的中心位置
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        # 创建椭圆形对象，根据给定的参数
        p = mpatches.Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        # 更新艺术家属性
        self.update_prop(p, orig_handle, legend)
        # 设置变换
        p.set_transform(trans)
        # 返回包含椭圆形对象的列表
        return [p]

# 创建一个圆形对象，设置其属性
c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="green",
                    edgecolor="red", linewidth=3)

# 创建图形和轴对象
fig, ax = plt.subplots()

# 在轴上添加圆形对象
ax.add_patch(c)

# 添加图例到轴上，图例包含圆形对象，并指定处理器映射为 HandlerEllipse() 处理椭圆形
ax.legend([c], ["An ellipse, not a rectangle"],
          handler_map={mpatches.Circle: HandlerEllipse()})
```
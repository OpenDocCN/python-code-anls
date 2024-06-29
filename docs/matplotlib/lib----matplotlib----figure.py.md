# `D:\src\scipysrc\matplotlib\lib\matplotlib\figure.py`

```py
"""
`matplotlib.figure` implements the following classes:

`Figure`
    Top level `~matplotlib.artist.Artist`, which holds all plot elements.
    Many methods are implemented in `FigureBase`.

`SubFigure`
    A logical figure inside a figure, usually added to a figure (or parent
    `SubFigure`) with `Figure.add_subfigure` or `Figure.subfigures` methods
    (provisional API v3.4).

Figures are typically created using pyplot methods `~.pyplot.figure`,
`~.pyplot.subplots`, and `~.pyplot.subplot_mosaic`.

.. plot::
    :include-source:

    fig, ax = plt.subplots(figsize=(2, 2), facecolor='lightskyblue',
                           layout='constrained')
    fig.suptitle('Figure')
    ax.set_title('Axes', loc='left', fontstyle='oblique', fontsize='medium')

Some situations call for directly instantiating a `~.figure.Figure` class,
usually inside an application of some sort (see :ref:`user_interfaces` for a
list of examples) .  More information about Figures can be found at
:ref:`figure-intro`.
"""

# 导入必要的模块和库
from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading

import numpy as np

import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
    Artist, allow_rasterization, _finalize_rasterization)
from matplotlib.backend_bases import (
    DrawEvent, FigureCanvasBase, NonGuiException, MouseButton, _get_renderer)
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage

from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec, SubplotParams
from matplotlib.layout_engine import (
    ConstrainedLayoutEngine, TightLayoutEngine, LayoutEngine,
    PlaceHolderLayoutEngine
)
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
                                   TransformedBbox)

_log = logging.getLogger(__name__)


def _stale_figure_callback(self, val):
    # 如果存在 Figure 对象，则设置其 stale 属性
    if self.figure:
        self.figure.stale = val


class _AxesStack:
    """
    Helper class to track Axes in a figure.

    Axes are tracked both in the order in which they have been added
    (``self._axes`` insertion/iteration order) and in the separate "gca" stack
    (which is the index to which they map in the ``self._axes`` dict).
    """

    def __init__(self):
        # 用于将 Axes 对象映射到 "gca" 顺序的字典
        self._axes = {}  # Mapping of Axes to "gca" order.
        # 计数器，用于保持 Axes 添加的顺序
        self._counter = itertools.count()

    def as_list(self):
        """List the Axes that have been added to the figure."""
        # 返回已添加到 figure 中的 Axes 列表
        return [*self._axes]  # This relies on dict preserving order.

    def remove(self, a):
        """Remove the Axes from the stack."""
        # 从 _axes 字典中移除指定的 Axes 对象
        self._axes.pop(a)
    # 定义一个方法 `bubble`，用于将一个已存在于堆栈中的 Axes 移动到堆栈顶部。
    def bubble(self, a):
        # 如果要移动的 Axes 不在 _axes 字典中，则抛出 ValueError 异常。
        if a not in self._axes:
            raise ValueError("Axes has not been added yet")
        # 将指定的 Axes 在 _axes 字典中的值更新为下一个计数器值。
        self._axes[a] = next(self._counter)

    # 定义一个方法 `add`，用于向堆栈中添加 Axes，如果已经存在则忽略。
    def add(self, a):
        # 如果要添加的 Axes 不在 _axes 字典中，则将其添加，并赋予下一个计数器值。
        if a not in self._axes:
            self._axes[a] = next(self._counter)

    # 定义一个方法 `current`，返回堆栈中值最大的 Axes，如果堆栈为空则返回 None。
    def current(self):
        # 返回 _axes 字典中值最大的键，使用 key=self._axes.__getitem__ 进行比较，如果字典为空则返回默认值 None。
        return max(self._axes, key=self._axes.__getitem__, default=None)

    # 定义 `__getstate__` 方法，返回对象的序列化状态，包括所有实例变量和 _axes 字典中值的最大计数器值。
    def __getstate__(self):
        return {
            **vars(self),
            "_counter": max(self._axes.values(), default=0)
        }

    # 定义 `__setstate__` 方法，用于反序列化对象的状态，更新实例变量和计数器的下一个值。
    def __setstate__(self, state):
        next_counter = state.pop('_counter')  # 取出 '_counter' 字段的值
        vars(self).update(state)  # 更新实例的所有变量
        self._counter = itertools.count(next_counter)  # 从指定的下一个计数器值开始计数
class FigureBase(Artist):
    """
    Base class for `.Figure` and `.SubFigure` containing the methods that add
    artists to the figure or subfigure, create Axes, etc.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # 删除非图形艺术家的 _axes 属性，
        # 因为图形不可能在 Axes 中
        # 这在艺术家基类中的属性方法中使用，
        # 在这个类中被覆盖
        del self._axes

        self._suptitle = None
        self._supxlabel = None
        self._supylabel = None

        # 用于跟踪要对齐的 x、y 标签和标题的分组器。
        # 参见 self.align_xlabels, self.align_ylabels,
        # self.align_titles 和 axis._get_tick_boxes_siblings
        self._align_label_groups = {
            "x": cbook.Grouper(),
            "y": cbook.Grouper(),
            "title": cbook.Grouper()
        }

        self._localaxes = []  # 跟踪所有的 Axes
        self.artists = []  # 跟踪所有的艺术家对象
        self.lines = []  # 跟踪所有的线对象
        self.patches = []  # 跟踪所有的补丁对象
        self.texts = []  # 跟踪所有的文本对象
        self.images = []  # 跟踪所有的图像对象
        self.legends = []  # 跟踪所有的图例对象
        self.subfigs = []  # 跟踪所有的子图对象
        self.stale = True  # 图形是否过时的标志
        self.suppressComposite = None  # 是否禁止组合的标志
        self.set(**kwargs)  # 设置传入的关键字参数

    def _get_draw_artists(self, renderer):
        """Also runs apply_aspect"""
        # 获取所有子元素艺术家对象
        artists = self.get_children()

        # 移除图形的背景补丁对象
        artists.remove(self.patch)

        # 排序所有非动画艺术家对象，按照绘制顺序
        artists = sorted(
            (artist for artist in artists if not artist.get_animated()),
            key=lambda artist: artist.get_zorder())

        # 对所有本地的 Axes 进行适应方面比例的处理
        for ax in self._localaxes:
            locator = ax.get_axes_locator()
            ax.apply_aspect(locator(ax, renderer) if locator else None)

            # 对每个 Axes 的子元素进行适应方面比例的处理
            for child in ax.get_children():
                if hasattr(child, 'apply_aspect'):
                    locator = child.get_axes_locator()
                    child.apply_aspect(
                        locator(child, renderer) if locator else None)

        return artists
    def autofmt_xdate(
            self, bottom=0.2, rotation=30, ha='right', which='major'):
        """
        Date ticklabels often overlap, so it is useful to rotate them
        and right align them.  Also, a common use case is a number of
        subplots with shared x-axis where the x-axis is date data.  The
        ticklabels are often long, and it helps to rotate them on the
        bottom subplot and turn them off on other subplots, as well as
        turn off xlabels.

        Parameters
        ----------
        bottom : float, default: 0.2
            The bottom of the subplots for `subplots_adjust`.
        rotation : float, default: 30 degrees
            The rotation angle of the xtick labels in degrees.
        ha : {'left', 'center', 'right'}, default: 'right'
            The horizontal alignment of the xticklabels.
        which : {'major', 'minor', 'both'}, default: 'major'
            Selects which ticklabels to rotate.
        """
        # 检查 `which` 参数是否在有效取值范围内
        _api.check_in_list(['major', 'minor', 'both'], which=which)
        # 检查是否所有子图都存在子图规范
        allsubplots = all(ax.get_subplotspec() for ax in self.axes)
        # 如果只有一个子图
        if len(self.axes) == 1:
            # 对第一个子图的指定类型的 xticklabels 进行对齐和旋转操作
            for label in self.axes[0].get_xticklabels(which=which):
                label.set_ha(ha)  # 设置水平对齐方式
                label.set_rotation(rotation)  # 设置旋转角度
        else:
            # 如果所有子图都存在子图规范
            if allsubplots:
                # 遍历所有子图
                for ax in self.get_axes():
                    # 如果子图位于最后一行
                    if ax.get_subplotspec().is_last_row():
                        # 对当前子图的指定类型的 xticklabels 进行对齐和旋转操作
                        for label in ax.get_xticklabels(which=which):
                            label.set_ha(ha)  # 设置水平对齐方式
                            label.set_rotation(rotation)  # 设置旋转角度
                    else:
                        # 对非最后一行的子图，隐藏 xticklabels 并清空 x轴标签
                        for label in ax.get_xticklabels(which=which):
                            label.set_visible(False)  # 隐藏标签
                        ax.set_xlabel('')  # 清空 x轴标签内容

        # 如果所有子图都存在子图规范，则调整子图的底部空白区域
        if allsubplots:
            self.subplots_adjust(bottom=bottom)
        self.stale = True  # 标记 Figure 对象为过时状态

    def get_children(self):
        """Get a list of artists contained in the figure."""
        # 返回 Figure 对象中所有艺术家（artists）的列表
        return [self.patch,
                *self.artists,
                *self._localaxes,
                *self.lines,
                *self.patches,
                *self.texts,
                *self.images,
                *self.legends,
                *self.subfigs]

    def contains(self, mouseevent):
        """
        Test whether the mouse event occurred on the figure.

        Returns
        -------
            bool, {}
        """
        # 检查鼠标事件是否发生在 Figure 对象内部
        if self._different_canvas(mouseevent):
            return False, {}  # 如果位于不同的画布上，则返回 False 和空字典
        inside = self.bbox.contains(mouseevent.x, mouseevent.y)  # 检查鼠标位置是否在 Figure 边界框内
        return inside, {}  # 返回是否在内部以及空字典

    def get_window_extent(self, renderer=None):
        # docstring inherited
        # 返回 Figure 对象的边界框
        return self.bbox
    # 定义一个方法 _suplabels，用于向图中添加一个居中显示的 %(name)s 文本。

    """
    Add a centered %(name)s to the figure.

    Parameters
    ----------
    t : str
        The %(name)s text.
        文本参数 t，表示 %(name)s 的文本内容。
    x : float, default: %(x0)s
        The x location of the text in figure coordinates.
        文本的 x 坐标位置，以图形坐标系为单位，默认为 %(x0)s。
    y : float, default: %(y0)s
        The y location of the text in figure coordinates.
        文本的 y 坐标位置，以图形坐标系为单位，默认为 %(y0)s。
    horizontalalignment, ha : {'center', 'left', 'right'}, default: %(ha)s
        The horizontal alignment of the text relative to (*x*, *y*).
        文本在 (x, y) 位置的水平对齐方式，默认为 %(ha)s，可选值有 {'center', 'left', 'right'}。
    verticalalignment, va : {'top', 'center', 'bottom', 'baseline'}, \
        ```

    # 文本在 (x, y) 位置的垂直对齐方式，默认为 %(va)s，可选值有 {'top', 'center', 'bottom', 'baseline'}。
def _supxlabel(self, t, **kwargs):
    # 从 _suplabels 中继承的文档字符串...
    info = {'name': '_supxlabel', 'x0': 0.5, 'y0': 0.01,
            'ha': 'center', 'va': 'bottom', 'rotation': 0,
            'size': 'figure.labelsize', 'weight': 'figure.labelweight'}
    return self._suplabels(t, info, **kwargs)


这段代码定义了 `_supxlabel` 方法，用于添加位于底部的上方 x 轴标签。它继承自 `_suplabels` 方法，根据给定的参数和关键字参数来设置标签的位置和样式。

1. **info 字典**: 包含了与标签相关的信息，如名称 (`name`)、水平对齐方式 (`ha`)、垂直对齐方式 (`va`)、初始位置 (`x0` 和 `y0`)、旋转角度 (`rotation`)、字体大小 (`size`) 和字体粗细 (`weight`)。这些信息用于在 `_suplabels` 方法中设置标签的属性。

2. **self._suplabels(t, info, **kwargs)**: 调用父类的 `_suplabels` 方法，将标签文本 `t`、信息字典 `info` 和其他关键字参数 `kwargs` 传递给它。

3. **返回值**: 返回 `_suplabels` 方法的返回值，通常是 `.Text` 实例，表示创建或更新的文本对象。

这段代码主要用于在 matplotlib 图形中添加自定义的 x 轴标签，其位置和样式可以通过 `info` 字典和传入的关键字参数 `kwargs` 进行控制和设置。
    # 返回包含 t 文本内容的上部标签，通过 _suplabels 处理，使用默认设置信息 info 和额外的关键字参数 kwargs
    def supxlabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supxlabel', 'x0': 0.5, 'y0': 0.01,
                'ha': 'center', 'va': 'bottom', 'rotation': 0,
                'size': 'figure.labelsize', 'weight': 'figure.labelweight'}
        return self._suplabels(t, info, **kwargs)

    # 返回上部 x 轴标签的文本字符串，如果未设置则返回空字符串
    def get_supxlabel(self):
        """Return the supxlabel as string or an empty string if not set."""
        text_obj = self._supxlabel
        return "" if text_obj is None else text_obj.get_text()

    # 使用默认设置信息 info 和额外的关键字参数 kwargs，返回包含 t 文本内容的左侧标签，通过 _suplabels 处理
    @_docstring.Substitution(x0=0.02, y0=0.5, name='supylabel', ha='left',
                             va='center', rc='label')
    @_docstring.copy(_suplabels)
    def supylabel(self, t, **kwargs):
        # docstring from _suplabels...
        info = {'name': '_supylabel', 'x0': 0.02, 'y0': 0.5,
                'ha': 'left', 'va': 'center', 'rotation': 'vertical',
                'rotation_mode': 'anchor', 'size': 'figure.labelsize',
                'weight': 'figure.labelweight'}
        return self._suplabels(t, info, **kwargs)

    # 返回上部 y 轴标签的文本字符串，如果未设置则返回空字符串
    def get_supylabel(self):
        """Return the supylabel as string or an empty string if not set."""
        text_obj = self._supylabel
        return "" if text_obj is None else text_obj.get_text()

    # 获取图形矩形边框的边缘颜色
    def get_edgecolor(self):
        """Get the edge color of the Figure rectangle."""
        return self.patch.get_edgecolor()

    # 获取图形矩形边框的填充颜色
    def get_facecolor(self):
        """Get the face color of the Figure rectangle."""
        return self.patch.get_facecolor()

    # 返回图形背景补丁是否可见，即图形背景是否将被绘制
    def get_frameon(self):
        """
        Return the figure's background patch visibility, i.e.
        whether the figure background will be drawn. Equivalent to
        ``Figure.patch.get_visible()``.
        """
        return self.patch.get_visible()

    # 设置图形矩形边框的线宽
    def set_linewidth(self, linewidth):
        """
        Set the line width of the Figure rectangle.

        Parameters
        ----------
        linewidth : number
        """
        self.patch.set_linewidth(linewidth)

    # 获取图形矩形边框的线宽
    def get_linewidth(self):
        """
        Get the line width of the Figure rectangle.
        """
        return self.patch.get_linewidth()

    # 设置图形矩形边框的边缘颜色
    def set_edgecolor(self, color):
        """
        Set the edge color of the Figure rectangle.

        Parameters
        ----------
        color : :mpltype:`color`
        """
        self.patch.set_edgecolor(color)

    # 设置图形矩形边框的填充颜色
    def set_facecolor(self, color):
        """
        Set the face color of the Figure rectangle.

        Parameters
        ----------
        color : :mpltype:`color`
        """
        self.patch.set_facecolor(color)

    # 设置图形背景补丁的可见性，即图形背景是否将被绘制
    def set_frameon(self, b):
        """
        Set the figure's background patch visibility, i.e.
        whether the figure background will be drawn. Equivalent to
        ``Figure.patch.set_visible()``.

        Parameters
        ----------
        b : bool
        """
        self.patch.set_visible(b)
        self.stale = True
    # 定义一个属性 `frameon`，其 getter 和 setter 方法分别为 `get_frameon` 和 `set_frameon`
    frameon = property(get_frameon, set_frameon)

    def add_artist(self, artist, clip=False):
        """
        将一个 `.Artist` 添加到图形中。

        通常情况下，使用 `.Axes.add_artist` 方法将艺术家添加到 `~.axes.Axes` 对象中；
        只有在需要直接向图形中添加艺术家的罕见情况下才使用此方法。

        Parameters
        ----------
        artist : `~matplotlib.artist.Artist`
            要添加到图形中的艺术家。如果添加的艺术家之前未设置过变换，则其变换将被设置为
            ``figure.transSubfigure``。
        clip : bool, 默认为 False
            是否应该通过图形补丁来剪裁添加的艺术家。

        Returns
        -------
        `~matplotlib.artist.Artist`
            添加的艺术家。
        """
        artist.set_figure(self)  # 将艺术家的图形设置为当前图形对象
        self.artists.append(artist)  # 将艺术家添加到图形的艺术家列表中
        artist._remove_method = self.artists.remove  # 设置艺术家的移除方法

        if not artist.is_transform_set():
            artist.set_transform(self.transSubfigure)  # 如果艺术家的变换未设置，则设置为 figure.transSubfigure

        if clip and artist.get_clip_path() is None:
            artist.set_clip_path(self.patch)  # 如果 clip 为 True 并且艺术家没有剪裁路径，则设置剪裁路径为 figure.patch

        self.stale = True  # 设置图形为过时状态，需要重新绘制
        return artist  # 返回添加的艺术家对象

    @_docstring.dedent_interpd
    def add_axes(self, *args, **kwargs):
        """
        添加一个 `~.axes.Axes` 到图形中。

        调用签名::

            add_axes(rect, projection=None, polar=False, **kwargs)
            add_axes(ax)

        Parameters
        ----------
        rect : tuple (left, bottom, width, height)
            新 `~.axes.Axes` 的尺寸 (left, bottom, width, height)。
            所有量都是相对于图形宽度和高度的分数。

        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
            ...

        Returns
        -------
        `~matplotlib.axes.Axes`
            添加的轴对象。
        """
    def add_subplot(self, *args, **kwargs):
        """
        Add an `~.axes.Axes` to the figure as part of a subplot arrangement.

        Call signatures::

           add_subplot(nrows, ncols, index, **kwargs)
           add_subplot(pos, **kwargs)
           add_subplot(ax)
           add_subplot()

        Parameters
        ----------
        *args : int, (int, int, *index*), or `.SubplotSpec`, default: (1, 1, 1)
            The position of the subplot described by one of

            - Three integers (*nrows*, *ncols*, *index*). The subplot will
              take the *index* position on a grid with *nrows* rows and
              *ncols* columns. *index* starts at 1 in the upper left corner
              and increases to the right.  *index* can also be a two-tuple
              specifying the (*first*, *last*) indices (1-based, and including
              *last*) of the subplot, e.g., ``fig.add_subplot(3, 1, (1, 2))``
              makes a subplot that spans the upper 2/3 of the figure.
            - A 3-digit integer. The digits are interpreted as if given
              separately as three single-digit integers, i.e.
              ``fig.add_subplot(235)`` is the same as
              ``fig.add_subplot(2, 3, 5)``. Note that this can only be used
              if there are no more than 9 subplots.
            - A `.SubplotSpec`.

            In rare circumstances, `.add_subplot` may be called with a single
            argument, a subplot Axes instance already created in the
            present figure but not in the figure's list of Axes.

        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
    """
        # 根据给定的参数，将一个 `~.axes.Axes` 添加到图形中作为子图的一部分
        # 处理不同的调用签名，根据参数的类型和数量来确定子图的位置和特性
        def _add_axes_internal(self, ax, key):
            """Private helper for `add_axes` and `add_subplot`."""
            # 将新创建的 Axes 实例添加到图形的 Axes 栈中
            self._axstack.add(ax)
            # 如果 Axes 实例不在本地 Axes 列表中，则将其添加进去
            if ax not in self._localaxes:
                self._localaxes.append(ax)
            # 设置当前坐标轴为 ax
            self.sca(ax)
            # 设置 ax 的移除方法为 self.delaxes
            ax._remove_method = self.delaxes
            # 支持 plt.subplot 的重新选择逻辑
            ax._projection_init = key
            # 标记图形为需要更新状态
            self.stale = True
            # 设置 ax 的陈旧回调函数为 _stale_figure_callback
            ax.stale_callback = _stale_figure_callback
            # 返回添加的 Axes 实例 ax
            return ax

        def delaxes(self, ax):
            """
            Remove the `~.axes.Axes` *ax* from the figure; update the current Axes.
            """
            # 从图形中移除指定的 `~.axes.Axes` 实例 ax，并更新当前的 Axes
            self._remove_axes(ax, owners=[self._axstack, self._localaxes])
    def _remove_axes(self, ax, owners):
        """
        Common helper for removal of standard Axes (via delaxes) and of child Axes.

        Parameters
        ----------
        ax : `~.AxesBase`
            The Axes to remove.
        owners
            List of objects (list or _AxesStack) "owning" the Axes, from which the Axes
            will be remove()d.
        """
        # 从每个拥有者对象中移除指定的 Axes 对象
        for owner in owners:
            owner.remove(ax)

        # 触发 Axes 改变事件通知所有注册的观察者
        self._axobservers.process("_axes_change_event", self)
        # 标记 Figure 对象为过时状态，需要重新绘制
        self.stale = True
        # 释放 Axes 对象关联的鼠标资源
        self.canvas.release_mouse(ax)

        # 针对所有轴名称进行循环，断开共享 Axes 之间的链接
        for name in ax._axis_names:
            # 获取与当前轴共享的轴组
            grouper = ax._shared_axes[name]
            # 获取与当前轴共享的所有兄弟轴，排除当前轴本身
            siblings = [other for other in grouper.get_siblings(ax) if other is not ax]
            if not siblings:  # 如果当前轴在此轴上没有共享兄弟轴，直接跳过
                continue
            # 从共享组中移除当前轴
            grouper.remove(ax)
            # 更新格式化器和定位器，使它们指向仍然存在的轴
            remaining_axis = siblings[0]._axis_map[name]
            remaining_axis.get_major_formatter().set_axis(remaining_axis)
            remaining_axis.get_major_locator().set_axis(remaining_axis)
            remaining_axis.get_minor_formatter().set_axis(remaining_axis)
            remaining_axis.get_minor_locator().set_axis(remaining_axis)

        # 移除与当前轴相关联的所有双生轴链接
        ax._twinned_axes.remove(ax)

    def clear(self, keep_observers=False):
        """
        Clear the figure.

        Parameters
        ----------
        keep_observers : bool, default: False
            Set *keep_observers* to True if, for example,
            a gui widget is tracking the Axes in the figure.
        """
        self.suppressComposite = None

        # 首先清除所有子图中的 Axes 对象
        for subfig in self.subfigs:
            subfig.clear(keep_observers=keep_observers)
        self.subfigs = []

        # 遍历 Figure 中的每个 Axes 的副本并清除
        for ax in tuple(self.axes):  # Iterate over the copy.
            ax.clear()
            self.delaxes(ax)  # 从 Figure 的轴堆栈中移除该 Axes 对象

        # 清空 Figure 中的各种图形元素列表
        self.artists = []
        self.lines = []
        self.patches = []
        self.texts = []
        self.images = []
        self.legends = []

        # 如果不保留观察者，重置 Figure 的观察者注册表
        if not keep_observers:
            self._axobservers = cbook.CallbackRegistry()

        # 重置 Figure 的超级标题和坐标轴标签
        self._suptitle = None
        self._supxlabel = None
        self._supylabel = None

        # 将 Figure 标记为过时状态，需要重新绘制
        self.stale = True

    # `clear` 的同义词。
    # 定义方法 clf，作为 clear() 方法的别名，不建议使用 clf()
    def clf(self, keep_observers=False):
        """
        [*Discouraged*] Alias for the `clear()` method.

        .. admonition:: Discouraged

            The use of ``clf()`` is discouraged. Use ``clear()`` instead.

        Parameters
        ----------
        keep_observers : bool, default: False
            Set *keep_observers* to True if, for example,
            a gui widget is tracking the Axes in the figure.
        
        Returns
        -------
        Same as `clear()`

        """
        # 调用 clear() 方法并返回其结果
        return self.clear(keep_observers=keep_observers)

    # 下面的文档字符串经过修改，适用于 pyplot 版本的函数
    # 函数签名将 " legend(" 替换为 " figlegend("，代码示例将 "fig.legend(" 替换为 "plt.figlegend"
    # 保持使用 pyplot 中的一致性，将 "ax.plot" 替换为 "plt.plot"
    @_docstring.dedent_interpd
    @_docstring.dedent_interpd
    def text(self, x, y, s, fontdict=None, **kwargs):
        """
        Add text to figure.

        Parameters
        ----------
        x, y : float
            The position to place the text. By default, this is in figure
            coordinates, floats in [0, 1]. The coordinate system can be changed
            using the *transform* keyword.

        s : str
            The text string.

        fontdict : dict, optional
            A dictionary to override the default text properties. If not given,
            the defaults are determined by :rc:`font.*`. Properties passed as
            *kwargs* override the corresponding ones given in *fontdict*.

        Returns
        -------
        `~.text.Text`
            A matplotlib Text instance representing the added text.

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.text.Text` properties
            Other miscellaneous text parameters.

            %(Text:kwdoc)s

        See Also
        --------
        .Axes.text
        .pyplot.text
        """
        # 效果参数，包括使用 self.transSubfigure 变换、fontdict 和 kwargs 中的属性
        effective_kwargs = {
            'transform': self.transSubfigure,
            **(fontdict if fontdict is not None else {}),
            **kwargs,
        }
        # 创建 Text 对象，添加到当前 figure 中的 texts 列表中
        text = Text(x=x, y=y, text=s, **effective_kwargs)
        text.set_figure(self)
        text.stale_callback = _stale_figure_callback

        # 将新创建的 Text 对象添加到 texts 列表中，并设置其移除方法
        self.texts.append(text)
        text._remove_method = self.texts.remove
        # 将 figure 的 stale 状态设置为 True，表示需要更新
        self.stale = True
        # 返回新创建的 Text 对象
        return text

    @_docstring.dedent_interpd
    # 调整子图的布局参数。

    # 检查当前图形的布局引擎是否存在，并且确保其兼容性。
    if (self.get_layout_engine() is not None and
            not self.get_layout_engine().adjust_compatible):
        # 如果当前图形使用的布局引擎与subplots_adjust或tight_layout不兼容，
        # 发出外部警告，并且不执行subplots_adjust操作。
        _api.warn_external(
            "This figure was using a layout engine that is "
            "incompatible with subplots_adjust and/or tight_layout; "
            "not calling subplots_adjust.")
        return

    # 更新子图参数，根据传入的left, bottom, right, top, wspace, hspace参数
    self.subplotpars.update(left, bottom, right, top, wspace, hspace)

    # 遍历所有子图(ax对象)，并根据其子图规格更新位置信息。
    for ax in self.axes:
        if ax.get_subplotspec() is not None:
            ax._set_position(ax.get_subplotspec().get_position(self))

    # 标记图形为需要重新绘制的状态。
    self.stale = True
    def align_xlabels(self, axs=None):
        """
        Align the xlabels of subplots in the same subplot row if label
        alignment is being done automatically (i.e. the label position is
        not manually set).

        Alignment persists for draw events after this is called.

        If a label is on the bottom, it is aligned with labels on Axes that
        also have their label on the bottom and that have the same
        bottom-most subplot row.  If the label is on the top,
        it is aligned with labels on Axes with the same top-most row.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list of (or `~numpy.ndarray`) `~matplotlib.axes.Axes`
            to align the xlabels.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_ylabels
        matplotlib.figure.Figure.align_titles
        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that ``axs`` are from the same `.GridSpec`, so that
        their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with rotated xtick labels::

            fig, axs = plt.subplots(1, 2)
            for tick in axs[0].get_xticklabels():
                tick.set_rotation(55)
            axs[0].set_xlabel('XLabel 0')
            axs[1].set_xlabel('XLabel 1')
            fig.align_xlabels()
        """
        # 如果未提供 axs 参数，则默认使用 self.axes
        if axs is None:
            axs = self.axes
        # 将 axs 中所有有效的 Axes 对象展开成列表
        axs = [ax for ax in np.ravel(axs) if ax.get_subplotspec() is not None]
        # 遍历每个有效的 Axes 对象
        for ax in axs:
            # 调试信息，记录当前处理的 Axes 对象的 xlabel
            _log.debug(' Working on: %s', ax.get_xlabel())
            # 获取当前 Axes 对象的子图规格的 rowspan 属性
            rowspan = ax.get_subplotspec().rowspan
            # 获取当前 Axes 对象 x 轴标签的位置（顶部或底部）
            pos = ax.xaxis.get_label_position()  # top or bottom
            # 遍历其他 Axes 对象，查找与当前对象具有相同标签位置和相同行号的对象
            # 将它们添加到与每个兄弟 Axes 相关联的分组器中
            # 这个列表在 `axis.draw` 中由 `axis._update_label_position` 检查
            for axc in axs:
                if axc.xaxis.get_label_position() == pos:
                    rowspanc = axc.get_subplotspec().rowspan
                    # 如果当前标签在顶部且行号开始位置相同，或者在底部且行号结束位置相同
                    if (pos == 'top' and rowspan.start == rowspanc.start or
                            pos == 'bottom' and rowspan.stop == rowspanc.stop):
                        # 为了对齐的 x 轴标签组，将当前 Axes 对象与 axc 加入分组器中
                        self._align_label_groups['x'].join(ax, axc)
    def align_ylabels(self, axs=None):
        """
        Align the ylabels of subplots in the same subplot column if label
        alignment is being done automatically (i.e. the label position is
        not manually set).

        Alignment persists for draw events after this is called.

        If a label is on the left, it is aligned with labels on Axes that
        also have their label on the left and that have the same
        left-most subplot column.  If the label is on the right,
        it is aligned with labels on Axes with the same right-most column.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list (or `~numpy.ndarray`) of `~matplotlib.axes.Axes`
            to align the ylabels.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels
        matplotlib.figure.Figure.align_titles
        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that ``axs`` are from the same `.GridSpec`, so that
        their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with large yticks labels::

            fig, axs = plt.subplots(2, 1)
            axs[0].plot(np.arange(0, 1000, 50))
            axs[0].set_ylabel('YLabel 0')
            axs[1].set_ylabel('YLabel 1')
            fig.align_ylabels()
        """
        # 如果未提供具体的 Axes 对象列表，则默认使用 self.axes
        if axs is None:
            axs = self.axes
        # 将 axs 转换为包含有效子图规范的 Axes 对象列表
        axs = [ax for ax in np.ravel(axs) if ax.get_subplotspec() is not None]
        # 遍历每个有效的 Axes 对象
        for ax in axs:
            # 记录调试信息，显示当前处理的 Axes 的 ylabel
            _log.debug(' Working on: %s', ax.get_ylabel())
            # 获取当前 Axes 对象的子图规范的列宽
            colspan = ax.get_subplotspec().colspan
            # 获取当前 Axes 对象的 ylabel 位置（左侧或右侧）
            pos = ax.yaxis.get_label_position()  # left or right
            # 遍历其他 Axes 对象，查找具有相同 ylabel 位置和相同列数的对象
            # 将这些对象添加到与每个 Axes 关联的兄弟列表中
            # 这些列表在 `axis.draw` 中由 `axis._update_label_position` 检查
            for axc in axs:
                if axc.yaxis.get_label_position() == pos:
                    colspanc = axc.get_subplotspec().colspan
                    # 如果当前 label 在左侧且列起始位置相同，或者在右侧且列终止位置相同，
                    # 则将当前 Axes 对象与 axc 对象添加到 y 标签对齐组中
                    if (pos == 'left' and colspan.start == colspanc.start or
                            pos == 'right' and colspan.stop == colspanc.stop):
                        self._align_label_groups['y'].join(ax, axc)
    def align_titles(self, axs=None):
        """
        Align the titles of subplots in the same subplot row if title
        alignment is being done automatically (i.e. the title position is
        not manually set).

        Alignment persists for draw events after this is called.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list of (or ndarray) `~matplotlib.axes.Axes`
            to align the titles.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels
        matplotlib.figure.Figure.align_ylabels
        matplotlib.figure.Figure.align_labels

        Notes
        -----
        This assumes that ``axs`` are from the same `.GridSpec`, so that
        their `.SubplotSpec` positions correspond to figure positions.

        Examples
        --------
        Example with titles::

            fig, axs = plt.subplots(1, 2)
            axs[0].set_aspect('equal')
            axs[0].set_title('Title 0')
            axs[1].set_title('Title 1')
            fig.align_titles()
        """
        # 如果未指定 axs 参数，则默认为当前对象的 axes 属性
        if axs is None:
            axs = self.axes
        # 筛选出有效的 axes，即其子图规格不为空的 axes
        axs = [ax for ax in np.ravel(axs) if ax.get_subplotspec() is not None]
        # 遍历每个有效的 axes 进行标题对齐操作
        for ax in axs:
            # 记录调试信息，显示当前处理的 axes 的标题
            _log.debug(' Working on: %s', ax.get_title())
            # 获取当前 axes 的子图规格的 rowspan
            rowspan = ax.get_subplotspec().rowspan
            # 遍历所有有效的 axes 进行比较，找到同一行的 axes 进行标题对齐
            for axc in axs:
                # 获取比较 axes 的子图规格的 rowspan
                rowspanc = axc.get_subplotspec().rowspan
                # 如果两个 axes 的起始行相同，则执行标题对齐操作
                if (rowspan.start == rowspanc.start):
                    self._align_label_groups['title'].join(ax, axc)

    def align_labels(self, axs=None):
        """
        Align the xlabels and ylabels of subplots with the same subplots
        row or column (respectively) if label alignment is being
        done automatically (i.e. the label position is not manually set).

        Alignment persists for draw events after this is called.

        Parameters
        ----------
        axs : list of `~matplotlib.axes.Axes`
            Optional list (or `~numpy.ndarray`) of `~matplotlib.axes.Axes`
            to align the labels.
            Default is to align all Axes on the figure.

        See Also
        --------
        matplotlib.figure.Figure.align_xlabels
        matplotlib.figure.Figure.align_ylabels
        matplotlib.figure.Figure.align_titles
        """
        # 调用 align_xlabels 和 align_ylabels 方法来执行 x 和 y 标签的对齐操作
        self.align_xlabels(axs=axs)
        self.align_ylabels(axs=axs)
    def add_gridspec(self, nrows=1, ncols=1, **kwargs):
        """
        低级API，用于创建一个以当前图形为父级的`.GridSpec`对象。

        这是一个低级API，允许您创建一个 gridspec，并且可以基于该 gridspec 添加子图。大多数用户不需要这种自由度，
        应该使用更高级的方法如 `~.Figure.subplots` 或 `~.Figure.subplot_mosaic`。

        Parameters
        ----------
        nrows : int, 默认为 1
            网格中的行数。

        ncols : int, 默认为 1
            网格中的列数。

        Returns
        -------
        `.GridSpec` 对象

        Other Parameters
        ----------------
        **kwargs
            关键字参数将传递给 `.GridSpec`。

        See Also
        --------
        matplotlib.pyplot.subplots

        Examples
        --------
        添加一个跨越两行的子图::

            fig = plt.figure()
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            # 跨越两行：
            ax3 = fig.add_subplot(gs[:, 1])

        """

        _ = kwargs.pop('figure', None)  # 如果用户已经添加了这个，弹出掉...
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=self, **kwargs)
        return gs

    def add_subfigure(self, subplotspec, **kwargs):
        """
        将一个 `.SubFigure` 添加到图形中作为子图布局的一部分。

        Parameters
        ----------
        subplotspec : `.gridspec.SubplotSpec`
            定义了父 gridspec 中子图将被放置的区域。

        Returns
        -------
        `.SubFigure` 对象

        Other Parameters
        ----------------
        **kwargs
            传递给 `.SubFigure` 对象的其他参数。

        See Also
        --------
        .Figure.subfigures
        """
        sf = SubFigure(self, subplotspec, **kwargs)
        self.subfigs += [sf]
        sf._remove_method = self.subfigs.remove
        sf.stale_callback = _stale_figure_callback
        self.stale = True
        return sf

    def sca(self, a):
        """将当前 Axes 设置为 *a* 并返回 *a*。"""
        self._axstack.bubble(a)
        self._axobservers.process("_axes_change_event", self)
        return a

    def gca(self):
        """
        获取当前的 Axes。

        如果当前图形上没有 Axes，则使用 `.Figure.add_subplot` 创建一个新的 Axes。
        （要检查图形上是否有 Axes，检查 `figure.axes` 是否为空。要检查 pyplot 图形栈上是否有图形，
        检查 `.pyplot.get_fignums()` 是否为空。）
        """
        ax = self._axstack.current()
        return ax if ax is not None else self.add_subplot()
    # Helper function for `~matplotlib.pyplot.gci`. Do not use elsewhere.
    """
    Get the current colorable artist.

    Specifically, returns the current `.ScalarMappable` instance (`.Image`
    created by `imshow` or `figimage`, `.Collection` created by `pcolor` or
    `scatter`, etc.), or *None* if no such instance has been defined.

    The current image is an attribute of the current Axes, or the nearest
    earlier Axes in the current figure that contains an image.

    Notes
    -----
    Historically, the only colorable artists were images; hence the name
    ``gci`` (get current image).
    """
    # Get the current Axes instance from the stack of Axes.
    ax = self._axstack.current()
    if ax is None:
        return None
    # Retrieve the current image associated with the current Axes.
    im = ax._gci()
    if im is not None:
        return im
    # If no image found in the current Axes, search backwards through
    # previously created Axes instances to find an image.
    for ax in reversed(self.axes):
        im = ax._gci()
        if im is not None:
            return im
    # Return None if no current or previously created image is found.
    return None

def _process_projection_requirements(self, *, axes_class=None, polar=False,
                                     projection=None, **kwargs):
    """
    Handle the args/kwargs to add_axes/add_subplot/gca, returning::

        (axes_proj_class, proj_class_kwargs)

    which can be used for new Axes initialization/identification.
    """
    if axes_class is not None:
        # If axes_class is specified, set projection_class to axes_class directly.
        if polar or projection is not None:
            raise ValueError(
                "Cannot combine 'axes_class' and 'projection' or 'polar'")
        projection_class = axes_class
    else:
        # Determine projection_class based on polar and projection arguments.
        if polar:
            if projection is not None and projection != 'polar':
                raise ValueError(
                    f"polar={polar}, yet projection={projection!r}. "
                    "Only one of these arguments should be supplied."
                )
            projection = 'polar'

        if isinstance(projection, str) or projection is None:
            # Get the projection class based on the projection string or None.
            projection_class = projections.get_projection_class(projection)
        elif hasattr(projection, '_as_mpl_axes'):
            # If projection has _as_mpl_axes method, use it to get projection_class.
            projection_class, extra_kwargs = projection._as_mpl_axes()
            kwargs.update(**extra_kwargs)
        else:
            # Raise TypeError if projection doesn't match expected types.
            raise TypeError(
                f"projection must be a string, None or implement a "
                f"_as_mpl_axes method, not {projection!r}")
    # Return the determined projection_class and any additional kwargs.
    return projection_class, kwargs
    def get_default_bbox_extra_artists(self):
        """
        Return a list of Artists typically used in `.Figure.get_tightbbox`.
        """
        # 从当前 Figure 对象的子元素中筛选可见且处于布局中的 Artists，形成列表
        bbox_artists = [artist for artist in self.get_children()
                        if (artist.get_visible() and artist.get_in_layout())]
        
        # 遍历每个子图（axes），将其默认的 bbox_extra_artists 添加到 bbox_artists 列表中
        for ax in self.axes:
            if ax.get_visible():
                bbox_artists.extend(ax.get_default_bbox_extra_artists())
        
        # 返回包含所有符合条件的 Artists 的列表
        return bbox_artists

    # 从 Python 3.8 开始，使用 make_keyword_only 将此方法标记为关键字参数（keyword-only argument），参数名为 bbox_extra_artists
    # 返回图形的（紧凑）边界框，单位为英寸。

    # 如果未提供 renderer 参数，则使用图形的默认渲染器
    if renderer is None:
        renderer = self.figure._get_renderer()

    # 初始化一个空列表 bb，用于存储所有子元素的边界框
    bb = []

    # 如果 bbox_extra_artists 参数为 None，则将所有子元素加入边界框计算中
    if bbox_extra_artists is None:
        artists = [artist for artist in self.get_children()
                   if (artist not in self.axes and artist.get_visible()
                       and artist.get_in_layout())]
    else:
        # 否则，使用提供的 bbox_extra_artists 列表
        artists = bbox_extra_artists

    # 遍历所有艺术家对象，并计算它们的紧凑边界框
    for a in artists:
        bbox = a.get_tightbbox(renderer)
        if bbox is not None:
            bb.append(bbox)

    # 遍历所有子图，计算它们的紧凑边界框
    for ax in self.axes:
        if ax.get_visible():
            # 一些 Axes 对象不接受 bbox_extra_artists 参数，所以需要条件判断处理
            try:
                bbox = ax.get_tightbbox(
                    renderer, bbox_extra_artists=bbox_extra_artists)
            except TypeError:
                bbox = ax.get_tightbbox(renderer)
            bb.append(bbox)

    # 过滤掉无效的边界框（宽度或高度为非有限数或者宽高都为零）
    bb = [b for b in bb
          if (np.isfinite(b.width) and np.isfinite(b.height)
              and (b.width != 0 or b.height != 0))]

    # 检查是否是 Figure 对象，若是，则进行单位转换（从像素到英寸）
    isfigure = hasattr(self, 'bbox_inches')
    if len(bb) == 0:
        # 如果 bb 列表为空，根据对象类型返回相应的边界框
        if isfigure:
            return self.bbox_inches
        else:
            # 子图没有 bbox_inches，但有一个 bbox
            bb = [self.bbox]

    # 使用 Bbox.union 方法合并所有有效的边界框
    _bbox = Bbox.union(bb)

    # 如果是 Figure 对象，进一步进行单位转换（从像素到英寸）
    if isfigure:
        _bbox = TransformedBbox(_bbox, self.dpi_scale_trans.inverted())

    # 返回最终的紧凑边界框对象
    return _bbox
    # 对给定的 per_subplot_kw 字典进行规范化处理，展开包含元组的键
    def _norm_per_subplot_kw(per_subplot_kw):
        expanded = {}
        # 遍历 per_subplot_kw 字典的键值对
        for k, v in per_subplot_kw.items():
            # 如果键 k 是元组类型
            if isinstance(k, tuple):
                # 遍历元组中的每个子键
                for sub_key in k:
                    # 如果子键已经在 expanded 中存在，则抛出 ValueError 异常
                    if sub_key in expanded:
                        raise ValueError(f'The key {sub_key!r} appears multiple times.')
                    # 将子键添加到 expanded 中，其对应的值为 v
                    expanded[sub_key] = v
            else:
                # 如果键 k 不是元组类型
                # 如果键 k 已经在 expanded 中存在，则抛出 ValueError 异常
                if k in expanded:
                    raise ValueError(f'The key {k!r} appears multiple times.')
                # 将键 k 添加到 expanded 中，其对应的值为 v
                expanded[k] = v
        # 返回处理后的 expanded 字典
        return expanded

    # 规范化处理布局字符串，将其转换为二维列表
    @staticmethod
    def _normalize_grid_string(layout):
        if '\n' not in layout:
            # 如果布局字符串不包含换行符，则为单行字符串，使用分号分隔
            return [list(ln) for ln in layout.split(';')]
        else:
            # 如果布局字符串包含换行符，则为多行字符串，使用 inspect.cleandoc 进行清理处理
            layout = inspect.cleandoc(layout)
            # 返回去除换行符后按行分割成列表的结果
            return [list(ln) for ln in layout.strip('\n').split('\n')]

    # 设置图形对象的属性
    def _set_artist_props(self, a):
        # 如果图形对象 a 不是 self 对象本身
        if a != self:
            # 将图形对象 a 的 figure 属性设置为当前 self 对象
            a.set_figure(self)
        # 设置图形对象 a 的 stale_callback 属性为 _stale_figure_callback 函数
        a.stale_callback = _stale_figure_callback
        # 设置图形对象 a 的 transform 属性为 self.transSubfigure
        a.set_transform(self.transSubfigure)
# 定义 SubFigure 类，继承自 FigureBase 类，表示可以放置在另一个图形内部的逻辑图形
@_docstring.interpd
class SubFigure(FigureBase):
    """
    Logical figure that can be placed inside a figure.

    See :ref:`figure-api-subfigure` for an index of methods on this class.
    Typically instantiated using `.Figure.add_subfigure` or
    `.SubFigure.add_subfigure`, or `.SubFigure.subfigures`.  A subfigure has
    the same methods as a figure except for those particularly tied to the size
    or dpi of the figure, and is confined to a prescribed region of the figure.
    For example the following puts two subfigures side-by-side::

        fig = plt.figure()
        sfigs = fig.subfigures(1, 2)  # 创建一个包含两个子图的 subfigure 集合
        axsL = sfigs[0].subplots(1, 2)  # 在第一个子图中创建1行2列的子图
        axsR = sfigs[1].subplots(2, 1)  # 在第二个子图中创建2行1列的子图

    See :doc:`/gallery/subplots_axes_and_figures/subfigures`

    .. note::
        The *subfigure* concept is new in v3.4, and the API is still provisional.
    """
    def __init__(self, parent, subplotspec, *,
                 facecolor=None,
                 edgecolor=None,
                 linewidth=0.0,
                 frameon=None,
                 **kwargs):
        """
        Parameters
        ----------
        parent : `.Figure` or `.SubFigure`
            父图或子图，包含当前子图的图形或子图。子图可以嵌套使用。

        subplotspec : `.gridspec.SubplotSpec`
            定义子图将放置在父图网格规范中的区域。

        facecolor : 默认: ``"none"``
            图形背景颜色；默认为透明。

        edgecolor : 默认: :rc:`figure.edgecolor`
            图形边缘颜色。

        linewidth : float
            边框线宽度（即图形背景的边缘线宽度）。

        frameon : bool, 默认: :rc:`figure.frameon`
            如果为 ``False``，则不绘制图形背景的补丁。

        Other Parameters
        ----------------
        **kwargs : `.SubFigure` 属性，可选

            %(SubFigure:kwdoc)s
        """
        super().__init__(**kwargs)
        
        # 如果未指定 facecolor，则设为 "none"
        if facecolor is None:
            facecolor = "none"
        # 如果未指定 edgecolor，则使用默认的 rc 参数 figure.edgecolor
        if edgecolor is None:
            edgecolor = mpl.rcParams['figure.edgecolor']
        # 如果未指定 frameon，则使用默认的 rc 参数 figure.frameon
        if frameon is None:
            frameon = mpl.rcParams['figure.frameon']

        # 设置子图的父图和子图规范
        self._parent = parent
        self._subplotspec = subplotspec
        self.figure = parent.figure

        # 子图使用父图的 axstack
        self._axstack = parent._axstack
        self.subplotpars = parent.subplotpars
        self.dpi_scale_trans = parent.dpi_scale_trans
        self._axobservers = parent._axobservers
        self.transFigure = parent.transFigure
        
        # 设置相对图框的边界框并重新计算相对图框的转换
        self.bbox_relative = Bbox.null()
        self._redo_transform_rel_fig()
        self.figbbox = self._parent.figbbox
        
        # 使用相对图框和父图的 transSubfigure 创建边界框转换
        self.bbox = TransformedBbox(self.bbox_relative,
                                    self._parent.transSubfigure)
        self.transSubfigure = BboxTransformTo(self.bbox)

        # 创建子图的补丁对象（矩形），用于表示子图的背景和边框
        self.patch = Rectangle(
            xy=(0, 0), width=1, height=1, visible=frameon,
            facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
            # 不要让图形补丁影响边界框计算
            in_layout=False, transform=self.transSubfigure)
        self._set_artist_props(self.patch)
        self.patch.set_antialiased(False)

    @property
    def canvas(self):
        # 返回父图的画布
        return self._parent.canvas

    @property
    def dpi(self):
        # 返回父图的 DPI
        return self._parent.dpi

    @dpi.setter
    def dpi(self, value):
        # 设置父图的 DPI
        self._parent.dpi = value

    def get_dpi(self):
        """
        返回父图的 DPI，以每英寸点数表示的分辨率。
        """
        return self._parent.dpi
    # 设置父图的分辨率，以每英寸点数（DPI）表示。
    def set_dpi(self, val):
        self._parent.dpi = val
        # 设置 stale 标志为 True，表示需要更新图形对象
        self.stale = True

    # 获取渲染器对象，用于渲染图形。
    def _get_renderer(self):
        return self._parent._get_renderer()

    # 重新计算 transSubfigure 边界框相对于整个图形的变换。
    def _redo_transform_rel_fig(self, bbox=None):
        """
        Make the transSubfigure bbox relative to Figure transform.

        Parameters
        ----------
        bbox : bbox or None
            If not None, then the bbox is used for relative bounding box.
            Otherwise, it is calculated from the subplotspec.
        """
        if bbox is not None:
            # 如果提供了 bbox 参数，则使用其边界框更新相对边界框的起始和结束点。
            self.bbox_relative.p0 = bbox.p0
            self.bbox_relative.p1 = bbox.p1
            return
        # 计算子图规格相对于整个图形的位置和大小
        gs = self._subplotspec.get_gridspec()
        wr = np.asarray(gs.get_width_ratios())
        hr = np.asarray(gs.get_height_ratios())
        dx = wr[self._subplotspec.colspan].sum() / wr.sum()
        dy = hr[self._subplotspec.rowspan].sum() / hr.sum()
        x0 = wr[:self._subplotspec.colspan.start].sum() / wr.sum()
        y0 = 1 - hr[:self._subplotspec.rowspan.stop].sum() / hr.sum()
        # 更新相对边界框的起始和结束点坐标
        self.bbox_relative.p0 = (x0, y0)
        self.bbox_relative.p1 = (x0 + dx, y0 + dy)

    # 获取是否正在使用 constrained layout。
    def get_constrained_layout(self):
        """
        Return whether constrained layout is being used.

        See :ref:`constrainedlayout_guide`.
        """
        return self._parent.get_constrained_layout()

    # 获取 constrained layout 的填充信息。
    def get_constrained_layout_pads(self, relative=False):
        """
        Get padding for ``constrained_layout``.

        Returns a list of ``w_pad, h_pad`` in inches and
        ``wspace`` and ``hspace`` as fractions of the subplot.

        See :ref:`constrainedlayout_guide`.

        Parameters
        ----------
        relative : bool
            If `True`, then convert from inches to figure relative.
        """
        return self._parent.get_constrained_layout_pads(relative=relative)

    # 获取布局引擎对象。
    def get_layout_engine(self):
        return self._parent.get_layout_engine()

    # 返回子图中的所有 Axes 对象的列表。
    @property
    def axes(self):
        """
        List of Axes in the SubFigure.  You can access and modify the Axes
        in the SubFigure through this list.

        Modifying this list has no effect. Instead, use `~.SubFigure.add_axes`,
        `~.SubFigure.add_subplot` or `~.SubFigure.delaxes` to add or remove an
        Axes.

        Note: The `.SubFigure.axes` property and `~.SubFigure.get_axes` method
        are equivalent.
        """
        return self._localaxes[:]
    
    # axes 属性的别名，用于返回子图中的所有 Axes 对象的列表。
    get_axes = axes.fget
    def draw(self, renderer):
        """
        Draw method for the Figure class.

        This method is responsible for rendering the figure's contents onto
        the specified renderer.

        Parameters:
        - renderer: The renderer object onto which the figure is drawn.

        Returns:
        - None

        Notes:
        - If the figure is not visible (get_visible() returns False), the method
          returns early without drawing anything.
        - Various artists are drawn onto the renderer by calling _get_draw_artists().
        - The figure's patch (background) is drawn using the renderer.
        - Images in the draw list are composited onto the renderer, considering
          the suppressComposite setting of the figure.
        - The drawing operations are encapsulated within a 'subfigure' group in
          the renderer, identified by a group id (gid) from the figure.
        - Finally, after drawing, the stale flag is reset to False.
        """
        # docstring inherited

        # draw the figure bounding box, perhaps none for white figure
        if not self.get_visible():
            return

        artists = self._get_draw_artists(renderer)

        try:
            # Open a 'subfigure' group in the renderer with the figure's gid
            renderer.open_group('subfigure', gid=self.get_gid())
            
            # Draw the figure's patch (background)
            self.patch.draw(renderer)
            
            # Draw the list of artists, considering composite image suppression
            mimage._draw_list_compositing_images(
                renderer, self, artists, self.figure.suppressComposite)
            
            # Close the 'subfigure' group in the renderer
            renderer.close_group('subfigure')

        finally:
            # Reset the stale flag after drawing
            self.stale = False
# 使用 _docstring.interpd 装饰器对 Figure 类进行修饰，用于处理文档字符串的内插
class Figure(FigureBase):
    """
    The top level container for all the plot elements.

    See `matplotlib.figure` for an index of class methods.

    Attributes
    ----------
    patch
        The `.Rectangle` instance representing the figure background patch.

    suppressComposite
        For multiple images, the figure will make composite images
        depending on the renderer option_image_nocomposite function.  If
        *suppressComposite* is a boolean, this will override the renderer.
    """

    # 设置一个线程锁，用于在绘图开始时共享缓存的字体和数学文本，以避免在创建多个图形时
    # 在 Windows 上出现文件句柄过多的问题，同时提高数学文本解析的性能。然而，这些全局缓存
    # 不具备线程安全性。解决方案是在绘图开始时让 Figure 获取共享锁，并在完成时释放它。
    # 这样可以让多个渲染器共享缓存的字体和解析文本，但每次只有一个图形可以进行绘制，
    # 因此字体缓存和数学文本缓存一次只被一个渲染器使用。

    _render_lock = threading.RLock()

    def __str__(self):
        return "Figure(%gx%g)" % tuple(self.bbox.size)

    def __repr__(self):
        return "<{clsname} size {h:g}x{w:g} with {naxes} Axes>".format(
            clsname=self.__class__.__name__,
            h=self.bbox.size[0], w=self.bbox.size[1],
            naxes=len(self.axes),
        )
    def __init__(self,
                 figsize=None,  # 设置图形的尺寸，单位为英寸，可以为None，默认使用rc参数`figure.figsize`
                 dpi=None,  # 设置图形的分辨率，单位为每英寸点数，默认使用rc参数`figure.dpi`
                 *,
                 facecolor=None,  # 设置图形的背景色，默认使用rc参数`figure.facecolor`
                 edgecolor=None,  # 设置图形的边框色，默认使用rc参数`figure.edgecolor`
                 linewidth=0.0,  # 设置图形边框线的宽度
                 frameon=None,  # 控制是否绘制图形的背景，默认使用rc参数`figure.frameon`
                 subplotpars=None,  # 设置子图的参数，若未提供则使用rc参数`figure.subplot.*`
                 tight_layout=None,  # 控制是否使用紧凑布局机制，默认使用rc参数`figure.autolayout`
                                    # 使用此参数不推荐，请使用`layout='tight'`替代`tight_layout=True`，否则使用`.set_tight_layout`
                 constrained_layout=None,  # 控制是否使用约束布局，默认使用rc参数`figure.constrained_layout.use`
                                           # 使用此参数不推荐，请使用`layout='constrained'`
                 layout=None,  # 设置布局引擎的类型，如'constrained'，'compressed'，'tight'，'none'或`.LayoutEngine`
                 **kwargs
                 ):
        """
        Parameters
        ----------
        figsize : 2-tuple of floats, default: :rc:`figure.figsize`
            Figure dimension ``(width, height)`` in inches.

        dpi : float, default: :rc:`figure.dpi`
            Dots per inch.

        facecolor : default: :rc:`figure.facecolor`
            The figure patch facecolor.

        edgecolor : default: :rc:`figure.edgecolor`
            The figure patch edge color.

        linewidth : float
            The linewidth of the frame (i.e. the edge linewidth of the figure
            patch).

        frameon : bool, default: :rc:`figure.frameon`
            If ``False``, suppress drawing the figure background patch.

        subplotpars : `~matplotlib.gridspec.SubplotParams`
            Subplot parameters. If not given, the default subplot
            parameters :rc:`figure.subplot.*` are used.

        tight_layout : bool or dict, default: :rc:`figure.autolayout`
            Whether to use the tight layout mechanism. See `.set_tight_layout`.

            .. admonition:: Discouraged

                The use of this parameter is discouraged. Please use
                ``layout='tight'`` instead for the common case of
                ``tight_layout=True`` and use `.set_tight_layout` otherwise.

        constrained_layout : bool, default: :rc:`figure.constrained_layout.use`
            This is equal to ``layout='constrained'``.

            .. admonition:: Discouraged

                The use of this parameter is discouraged. Please use
                ``layout='constrained'`` instead.

        layout : {'constrained', 'compressed', 'tight', 'none', `.LayoutEngine`, \
    def pick(self, mouseevent):
        if not self.canvas.widgetlock.locked():
            super().pick(mouseevent)

    def _check_layout_engines_compat(self, old, new):
        """
        Helper for set_layout engine

        If the figure has used the old engine and added a colorbar then the
        value of colorbar_gridspec must be the same on the new engine.
        """
        if old is None or new is None:
            return True
        if old.colorbar_gridspec == new.colorbar_gridspec:
            return True
        # colorbar layout different, so check if any colorbars are on the
        # figure...
        for ax in self.axes:
            if hasattr(ax, '_colorbar'):
                # colorbars list themselves as a colorbar.
                return False
        return True
    # 设置此图形的布局引擎。

    Parameters
    ----------
    layout : {'constrained', 'compressed', 'tight', 'none', `.LayoutEngine`, None}
        - 'constrained' 将使用 `~.ConstrainedLayoutEngine`
        - 'compressed' 也会使用 `~.ConstrainedLayoutEngine`，但会进行修正以尝试为固定比例的 Axes 创建良好的布局
        - 'tight' 使用 `~.TightLayoutEngine`
        - 'none' 移除布局引擎

        如果是 `.LayoutEngine` 的实例，则将使用该实例。

        如果为 `None`，行为由 :rc:`figure.autolayout` 控制（如果为 `True`，则表现为像传入 'tight' 一样）和
        :rc:`figure.constrained_layout.use`（如果为 `True`，则表现为像传入 'constrained' 一样）。如果两者都为 `True`，
        :rc:`figure.autolayout` 优先级较高。

        用户和库可以定义自己的布局引擎，并直接传递实例。

    **kwargs
        关键字参数传递给布局引擎，用于设置填充和边距大小。仅在 *layout* 是字符串时使用。

    """
    if layout is None:
        if mpl.rcParams['figure.autolayout']:
            layout = 'tight'
        elif mpl.rcParams['figure.constrained_layout.use']:
            layout = 'constrained'
        else:
            self._layout_engine = None
            return
    if layout == 'tight':
        # 使用 TightLayoutEngine 创建新的布局引擎实例，并传递关键字参数
        new_layout_engine = TightLayoutEngine(**kwargs)
    elif layout == 'constrained':
        # 使用 ConstrainedLayoutEngine 创建新的布局引擎实例，并传递关键字参数
        new_layout_engine = ConstrainedLayoutEngine(**kwargs)
    elif layout == 'compressed':
        # 使用 ConstrainedLayoutEngine 创建新的布局引擎实例，启用压缩模式，并传递关键字参数
        new_layout_engine = ConstrainedLayoutEngine(compress=True, **kwargs)
    elif layout == 'none':
        if self._layout_engine is not None:
            # 如果当前存在布局引擎，创建一个 PlaceHolderLayoutEngine 实例，传递相关参数
            new_layout_engine = PlaceHolderLayoutEngine(
                self._layout_engine.adjust_compatible,
                self._layout_engine.colorbar_gridspec
            )
        else:
            new_layout_engine = None
    elif isinstance(layout, LayoutEngine):
        # 如果 layout 是 LayoutEngine 的实例，直接使用该实例
        new_layout_engine = layout
    else:
        # 如果 layout 不是预期的值，抛出 ValueError
        raise ValueError(f"Invalid value for 'layout': {layout!r}")

    # 检查新旧布局引擎是否兼容，如果兼容则设置新的布局引擎，否则抛出 RuntimeError
    if self._check_layout_engines_compat(self._layout_engine, new_layout_engine):
        self._layout_engine = new_layout_engine
    else:
        raise RuntimeError('Colorbar layout of new layout engine not '
                           'compatible with old engine, and a colorbar '
                           'has been created.  Engine not changed.')
    # TODO: I'd like to dynamically add the _repr_html_ method
    # to the figure in the right context, but then IPython doesn't
    # use it, for some reason.

    # 定义 _repr_html_ 方法，用于在特定上下文中将图形显示为 HTML
    def _repr_html_(self):
        # 不能使用 isinstance 判断类型，因为这样会导致无条件导入 webagg
        if 'WebAgg' in type(self.canvas).__name__:
            # 在需要时导入 webagg 模块
            from matplotlib.backends import backend_webagg
            # 调用 webagg.ipython_inline_display 方法显示图形
            return backend_webagg.ipython_inline_display(self)

    # 显示图形的方法，如果使用 GUI 后端，则显示图形窗口
    def show(self, warn=True):
        """
        If using a GUI backend with pyplot, display the figure window.

        If the figure was not created using `~.pyplot.figure`, it will lack
        a `~.backend_bases.FigureManagerBase`, and this method will raise an
        AttributeError.

        .. warning::

            This does not manage an GUI event loop. Consequently, the figure
            may only be shown briefly or not shown at all if you or your
            environment are not managing an event loop.

            Use cases for `.Figure.show` include running this from a GUI
            application (where there is persistently an event loop running) or
            from a shell, like IPython, that install an input hook to allow the
            interactive shell to accept input while the figure is also being
            shown and interactive.  Some, but not all, GUI toolkits will
            register an input hook on import.  See :ref:`cp_integration` for
            more details.

            If you're in a shell without input hook integration or executing a
            python script, you should use `matplotlib.pyplot.show` with
            ``block=True`` instead, which takes care of starting and running
            the event loop for you.

        Parameters
        ----------
        warn : bool, default: True
            If ``True`` and we are not running headless (i.e. on Linux with an
            unset DISPLAY), issue warning when called on a non-GUI backend.

        """
        # 如果图形的 canvas 管理器为 None，则抛出 AttributeError
        if self.canvas.manager is None:
            raise AttributeError(
                "Figure.show works only for figures managed by pyplot, "
                "normally created by pyplot.figure()")
        try:
            # 调用 canvas 的 manager 的 show 方法显示图形
            self.canvas.manager.show()
        except NonGuiException as exc:
            # 如果 warn 为 True，并且不是无头模式（例如 Linux 上未设置 DISPLAY），则发出警告
            if warn:
                _api.warn_external(str(exc))

    # 返回图形中的所有 axes 列表的属性方法
    @property
    def axes(self):
        """
        List of Axes in the Figure. You can access and modify the Axes in the
        Figure through this list.

        Do not modify the list itself. Instead, use `~Figure.add_axes`,
        `~.Figure.add_subplot` or `~.Figure.delaxes` to add or remove an Axes.

        Note: The `.Figure.axes` property and `~.Figure.get_axes` method are
        equivalent.
        """
        return self._axstack.as_list()

    # get_axes 方法等同于 axes 方法的 fget 属性
    get_axes = axes.fget
    # 获取渲染器对象，用于绘制图形
    def _get_renderer(self):
        if hasattr(self.canvas, 'get_renderer'):
            return self.canvas.get_renderer()
        else:
            return _get_renderer(self)

    # 获取 DPI 属性的方法
    def _get_dpi(self):
        return self._dpi

    # 设置 DPI 属性的方法，同时调整图形大小
    def _set_dpi(self, dpi, forward=True):
        """
        Parameters
        ----------
        dpi : float
            设置 DPI 值

        forward : bool
            传递给 `~.Figure.set_size_inches`
        """
        if dpi == self._dpi:
            # 如果 DPI 值未改变，则不触发后端事件
            return
        self._dpi = dpi
        # 清空并重新缩放 DPI 转换矩阵
        self.dpi_scale_trans.clear().scale(dpi)
        # 获取当前图形大小，并设置新的图形大小
        w, h = self.get_size_inches()
        self.set_size_inches(w, h, forward=forward)

    # 使用 property 装饰器将 _get_dpi 和 _set_dpi 方法封装为 dpi 属性，并提供文档字符串
    dpi = property(_get_dpi, _set_dpi, doc="The resolution in dots per inch.")

    # 返回是否在绘制时调用 `.Figure.tight_layout` 方法
    def get_tight_layout(self):
        """Return whether `.Figure.tight_layout` is called when drawing."""
        return isinstance(self.get_layout_engine(), TightLayoutEngine)

    # 设置 tight_layout 方法的过时提示信息
    @_api.deprecated("3.6", alternative="set_layout_engine",
                     pending=True)
    def set_tight_layout(self, tight):
        """
        设置是否以及如何在绘制时调用 `.Figure.tight_layout` 方法。

        Parameters
        ----------
        tight : bool or dict with keys "pad", "w_pad", "h_pad", "rect" or None
            如果是布尔值，设置是否在绘制时调用 `.Figure.tight_layout` 方法。
            如果为 ``None``，使用 :rc:`figure.autolayout` 替代。
            如果是字典，作为关键字参数传递给 `.Figure.tight_layout` 方法，覆盖默认的填充值。
        """
        if tight is None:
            tight = mpl.rcParams['figure.autolayout']
        _tight = 'tight' if bool(tight) else 'none'
        _tight_parameters = tight if isinstance(tight, dict) else {}
        # 设置布局引擎为 tight_layout 或 none，并根据参数调整
        self.set_layout_engine(_tight, **_tight_parameters)
        self.stale = True

    # 返回是否正在使用约束布局引擎
    def get_constrained_layout(self):
        """
        返回是否正在使用约束布局引擎。

        See :ref:`constrainedlayout_guide`.
        """
        return isinstance(self.get_layout_engine(), ConstrainedLayoutEngine)

    # 设置 constrained_layout 方法的过时提示信息
    @_api.deprecated("3.6", alternative="set_layout_engine('constrained')",
                     pending=True)
    @_api.deprecated(
         "3.6", alternative="figure.get_layout_engine().set()",
         pending=True)
    def set_constrained_layout_pads(self, **kwargs):
        """
        Set padding for ``constrained_layout``.

        Tip: The parameters can be passed from a dictionary by using
        ``fig.set_constrained_layout(**pad_dict)``.

        See :ref:`constrainedlayout_guide`.

        Parameters
        ----------
        w_pad : float, default: :rc:`figure.constrained_layout.w_pad`
            Width padding in inches.  This is the pad around Axes
            and is meant to make sure there is enough room for fonts to
            look good.  Defaults to 3 pts = 0.04167 inches

        h_pad : float, default: :rc:`figure.constrained_layout.h_pad`
            Height padding in inches. Defaults to 3 pts.

        wspace : float, default: :rc:`figure.constrained_layout.wspace`
            Width padding between subplots, expressed as a fraction of the
            subplot width.  The total padding ends up being w_pad + wspace.

        hspace : float, default: :rc:`figure.constrained_layout.hspace`
            Height padding between subplots, expressed as a fraction of the
            subplot width. The total padding ends up being h_pad + hspace.

        """
        # 检查当前图形的布局引擎是否是 ConstrainedLayoutEngine 类型
        if isinstance(self.get_layout_engine(), ConstrainedLayoutEngine):
            # 如果是，调用其 set 方法设置新的布局参数
            self.get_layout_engine().set(**kwargs)
    # 获取用于“constrained_layout”的填充值和间距

    # 如果布局引擎不是ConstrainedLayoutEngine类型，则返回None
    if not isinstance(self.get_layout_engine(), ConstrainedLayoutEngine):
        return None, None, None, None

    # 从布局引擎获取布局信息
    info = self.get_layout_engine().get()

    # 获取宽度填充值和高度填充值（单位为英寸）
    w_pad = info['w_pad']
    h_pad = info['h_pad']

    # 获取宽度间距和高度间距（作为子图的比例分数）
    wspace = info['wspace']
    hspace = info['hspace']

    # 如果需要相对值并且存在填充值，则进行单位转换为相对于图形的尺寸
    if relative and (w_pad is not None or h_pad is not None):
        renderer = self._get_renderer()
        dpi = renderer.dpi
        w_pad = w_pad * dpi / renderer.width
        h_pad = h_pad * dpi / renderer.height

    # 返回宽度填充值、高度填充值、宽度间距和高度间距
    return w_pad, h_pad, wspace, hspace
    def figimage(self, X, xo=0, yo=0, alpha=None, norm=None, cmap=None,
                 vmin=None, vmax=None, origin=None, resize=False, **kwargs):
        """
        Add a non-resampled image to the figure.

        The image is attached to the lower or upper left corner depending on
        *origin*.

        Parameters
        ----------
        X
            The image data. This is an array of one of the following shapes:

            - (M, N): an image with scalar data.  Color-mapping is controlled
              by *cmap*, *norm*, *vmin*, and *vmax*.
            - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).
            - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),
              i.e. including transparency.

        xo, yo : int
            The *x*/*y* image offset in pixels.

        alpha : None or float
            The alpha blending value.

        %(cmap_doc)s
            Documentation placeholder for `cmap` parameter.

            This parameter is ignored if *X* is RGB(A).

        %(norm_doc)s
            Documentation placeholder for `norm` parameter.

            This parameter is ignored if *X* is RGB(A).

        %(vmin_vmax_doc)s
            Documentation placeholder for `vmin` and `vmax` parameters.

            This parameter is ignored if *X* is RGB(A).

        origin : {'upper', 'lower'}, default: :rc:`image.origin`
            Indicates where the [0, 0] index of the array is in the upper left
            or lower left corner of the Axes.

        resize : bool
            If *True*, resize the figure to match the given image size.

        Returns
        -------
        `matplotlib.image.FigureImage`
            The created FigureImage instance.

        Other Parameters
        ----------------
        **kwargs
            Additional kwargs are `.Artist` kwargs passed on to `.FigureImage`.

        Notes
        -----
        figimage complements the Axes image (`~matplotlib.axes.Axes.imshow`)
        which will be resampled to fit the current Axes.  If you want
        a resampled image to fill the entire figure, you can define an
        `~matplotlib.axes.Axes` with extent [0, 0, 1, 1].

        Examples
        --------
        ::
            f = plt.figure()
            nx = int(f.get_figwidth() * f.dpi)
            ny = int(f.get_figheight() * f.dpi)
            data = np.random.random((ny, nx))
            f.figimage(data)
            plt.show()
        """
        # 如果设置了 resize 参数，根据图像大小调整图形尺寸
        if resize:
            dpi = self.get_dpi()  # 获取当前图形的 DPI 值
            figsize = [x / dpi for x in (X.shape[1], X.shape[0])]  # 计算需要设置的图形尺寸
            self.set_size_inches(figsize, forward=True)  # 设置图形尺寸

        # 创建 FigureImage 对象，设置颜色映射、标准化、偏移量、起始位置等参数
        im = mimage.FigureImage(self, cmap=cmap, norm=norm,
                                offsetx=xo, offsety=yo,
                                origin=origin, **kwargs)
        im.stale_callback = _stale_figure_callback  # 设置图像失效回调函数

        im.set_array(X)  # 设置图像数据
        im.set_alpha(alpha)  # 设置透明度
        if norm is None:
            im.set_clim(vmin, vmax)  # 设置颜色映射范围
        self.images.append(im)  # 将图像对象添加到图形对象的图像列表中
        im._remove_method = self.images.remove  # 设置图像对象的移除方法
        self.stale = True  # 标记图形对象为失效状态
        return im  # 返回创建的图像对象实例
    def set_size_inches(self, w, h=None, forward=True):
        """
        Set the figure size in inches.

        Call signatures::

             fig.set_size_inches(w, h)  # OR
             fig.set_size_inches((w, h))

        Parameters
        ----------
        w : (float, float) or float
            Width and height in inches (if height not specified as a separate
            argument) or width.
        h : float
            Height in inches.
        forward : bool, default: True
            If ``True``, the canvas size is automatically updated, e.g.,
            you can resize the figure window from the shell.

        See Also
        --------
        matplotlib.figure.Figure.get_size_inches
        matplotlib.figure.Figure.set_figwidth
        matplotlib.figure.Figure.set_figheight

        Notes
        -----
        To transform from pixels to inches divide by `Figure.dpi`.
        """
        if h is None:  # Got called with a single pair as argument.
            w, h = w
        size = np.array([w, h])
        if not np.isfinite(size).all() or (size < 0).any():
            raise ValueError(f'figure size must be positive finite not {size}')
        self.bbox_inches.p1 = size  # 设置图形尺寸，单位为英寸
        if forward:
            manager = self.canvas.manager
            if manager is not None:
                manager.resize(*(size * self.dpi).astype(int))  # 调整画布尺寸以匹配新的图形尺寸
        self.stale = True  # 标记图形为需要更新状态

    def get_size_inches(self):
        """
        Return the current size of the figure in inches.

        Returns
        -------
        ndarray
           The size (width, height) of the figure in inches.

        See Also
        --------
        matplotlib.figure.Figure.set_size_inches
        matplotlib.figure.Figure.get_figwidth
        matplotlib.figure.Figure.get_figheight

        Notes
        -----
        The size in pixels can be obtained by multiplying with `Figure.dpi`.
        """
        return np.array(self.bbox_inches.p1)  # 返回当前图形的尺寸，单位为英寸

    def get_figwidth(self):
        """Return the figure width in inches."""
        return self.bbox_inches.width  # 返回图形的宽度，单位为英寸

    def get_figheight(self):
        """Return the figure height in inches."""
        return self.bbox_inches.height  # 返回图形的高度，单位为英寸

    def get_dpi(self):
        """Return the resolution in dots per inch as a float."""
        return self.dpi  # 返回图形的分辨率，单位为每英寸的点数（浮点数）

    def set_dpi(self, val):
        """
        Set the resolution of the figure in dots-per-inch.

        Parameters
        ----------
        val : float
            新的分辨率值，单位为每英寸的点数。
        """
        self.dpi = val  # 设置图形的分辨率
        self.stale = True  # 标记图形为需要更新状态

    def set_figwidth(self, val, forward=True):
        """
        Set the width of the figure in inches.

        Parameters
        ----------
        val : float
            新的图形宽度，单位为英寸。
        forward : bool
            See `set_size_inches`.

        See Also
        --------
        matplotlib.figure.Figure.set_figheight
        matplotlib.figure.Figure.set_size_inches
        """
        self.set_size_inches(val, self.get_figheight(), forward=forward)  # 设置图形的宽度，保持高度不变，并更新画布尺寸
    def set_figheight(self, val, forward=True):
        """
        Set the height of the figure in inches.

        Parameters
        ----------
        val : float
            Desired height of the figure in inches.
        forward : bool
            If True, forward this setting to `set_size_inches`.

        See Also
        --------
        matplotlib.figure.Figure.set_figwidth
        matplotlib.figure.Figure.set_size_inches
        """
        # 调用父类方法设置图形的高度
        self.set_size_inches(self.get_figwidth(), val, forward=forward)

    def clear(self, keep_observers=False):
        """
        Clear the figure.

        Parameters
        ----------
        keep_observers : bool, optional
            If True, keep the observers attached.

        Notes
        -----
        This method clears the figure and updates the toolbar if present.
        """
        # 调用父类方法清除图形，并更新工具栏（如果有）
        super().clear(keep_observers=keep_observers)
        # FigureBase.clear 方法不清除工具栏，因为只有 Figure 才有工具栏
        toolbar = self.canvas.toolbar
        if toolbar is not None:
            toolbar.update()

    @_finalize_rasterization
    @allow_rasterization
    def draw(self, renderer):
        """
        Render the figure using the given renderer.

        Parameters
        ----------
        renderer : RendererBase subclass
            The renderer to use for drawing.

        Notes
        -----
        This method manages the rendering process of the figure,
        including layout, drawing artists, and managing events.
        """
        # 如果图形不可见，则直接返回
        if not self.get_visible():
            return

        with self._render_lock:
            # 获取需要绘制的艺术家对象
            artists = self._get_draw_artists(renderer)
            try:
                # 开始一个新的渲染组
                renderer.open_group('figure', gid=self.get_gid())
                # 如果有坐标轴并且存在布局引擎，则尝试执行布局引擎
                if self.axes and self.get_layout_engine() is not None:
                    try:
                        self.get_layout_engine().execute(self)
                    except ValueError:
                        pass
                        # 在调整窗口大小时可能会出现 ValueError

                # 绘制图形的背景补丁
                self.patch.draw(renderer)
                # 绘制图形上的艺术家对象，并处理合成效果
                mimage._draw_list_compositing_images(
                    renderer, self, artists, self.suppressComposite)

                # 关闭渲染组
                renderer.close_group('figure')
            finally:
                # 设置图形不再过时
                self.stale = False

            # 触发绘制事件
            DrawEvent("draw_event", self.canvas, renderer)._process()

    def draw_without_rendering(self):
        """
        Draw the figure without rendering anything.

        Notes
        -----
        This method is useful to calculate the final size of artists
        that require a draw operation before their size is known.
        """
        # 获取渲染器并禁用绘制
        renderer = _get_renderer(self)
        with renderer._draw_disabled():
            # 调用 draw 方法进行绘制
            self.draw(renderer)

    def draw_artist(self, a):
        """
        Draw a specific `.Artist` object.

        Parameters
        ----------
        a : `.Artist`
            The artist object to draw.
        """
        # 调用特定艺术家对象的 draw 方法
        a.draw(self.canvas.get_renderer())

    def __getstate__(self):
        """
        Return the state of the figure for pickling.

        Returns
        -------
        state : dict
            State dictionary containing information about the figure.
        """
        # 调用父类的 __getstate__ 方法获取初始状态
        state = super().__getstate__()

        # 由于画布不能被 pickle，这允许图形可以从一个画布分离并重新连接到另一个
        state.pop("canvas")

        # 如果有因像素比例变化而导致的 dpi 变化，则丢弃这些改变
        state["_dpi"] = state.get('_original_dpi', state['_dpi'])

        # 添加版本信息到状态中
        state['__mpl_version__'] = mpl.__version__

        # 检查图形管理器是否已在 pyplot 中注册
        from matplotlib import _pylab_helpers
        if self.canvas.manager in _pylab_helpers.Gcf.figs.values():
            state['_restore_to_pylab'] = True
        return state
    def __setstate__(self, state):
        # 从状态中弹出保存的 matplotlib 版本号
        version = state.pop('__mpl_version__')
        # 恢复到 pylab 状态的标志，默认为 False
        restore_to_pylab = state.pop('_restore_to_pylab', False)

        # 如果加载时的 matplotlib 版本号与保存时的版本号不一致，发出警告
        if version != mpl.__version__:
            _api.warn_external(
                f"This figure was saved with matplotlib version {version} and "
                f"loaded with {mpl.__version__} so may not function correctly."
            )
        
        # 将当前对象的状态更新为加载的状态
        self.__dict__ = state

        # 重新初始化部分未存储的状态信息，设置 self.canvas
        FigureCanvasBase(self)

        # 如果需要恢复到 pylab 状态
        if restore_to_pylab:
            # 延迟导入以避免循环依赖
            import matplotlib.pyplot as plt
            import matplotlib._pylab_helpers as pylab_helpers
            # 获取当前所有图形的编号
            allnums = plt.get_fignums()
            # 计算新的图形编号
            num = max(allnums) + 1 if allnums else 1
            # 获取当前后端的模块
            backend = plt._get_backend_mod()
            # 创建新的图形管理器
            mgr = backend.new_figure_manager_given_figure(num, self)
            # 将新的管理器设置为活动管理器
            pylab_helpers.Gcf._set_new_active_manager(mgr)
            # 如果交互式绘图，则绘制
            plt.draw_if_interactive()

        # 标记当前对象为过时
        self.stale = True

    def add_axobserver(self, func):
        """每当 Axes 状态发生变化时，调用 func(self)。"""
        # 连接一个包装 lambda 而不是 func 本身，以避免 func 被弱引用收集
        self._axobservers.connect("_axes_change_event", lambda arg: func(arg))

    def waitforbuttonpress(self, timeout=-1):
        """
        阻塞调用以与图形交互。

        等待用户输入并返回 True（如果按下键盘按键）、False（如果按下鼠标按钮）、
        或 None（如果在指定的超时时间内未收到输入）。负值的 timeout 禁用超时。
        """
        event = None

        def handler(ev):
            nonlocal event
            event = ev
            self.canvas.stop_event_loop()

        # 调用 blocking_input_loop 处理用户输入
        _blocking_input.blocking_input_loop(
            self, ["button_press_event", "key_press_event"], timeout, handler)

        # 返回用户输入的事件类型或 None（如果没有输入）
        return None if event is None else event.name == "key_press_event"
    def tight_layout(self, *, pad=1.08, h_pad=None, w_pad=None, rect=None):
        """
        Adjust the padding between and around subplots.

        To exclude an artist on the Axes from the bounding box calculation
        that determines the subplot parameters (i.e. legend, or annotation),
        set ``a.set_in_layout(False)`` for that artist.

        Parameters
        ----------
        pad : float, default: 1.08
            Padding between the figure edge and the edges of subplots,
            as a fraction of the font size.
        h_pad, w_pad : float, default: *pad*
            Padding (height/width) between edges of adjacent subplots,
            as a fraction of the font size.
        rect : tuple (left, bottom, right, top), default: (0, 0, 1, 1)
            A rectangle in normalized figure coordinates into which the whole
            subplots area (including labels) will fit.

        See Also
        --------
        .Figure.set_layout_engine
        .pyplot.tight_layout
        """
        # note that here we do not permanently set the figures engine to
        # tight_layout but rather just perform the layout in place and remove
        # any previous engines.
        # 创建一个 TightLayoutEngine 的实例，用指定的参数来调整布局
        engine = TightLayoutEngine(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
        # 尝试获取当前的布局引擎
        try:
            previous_engine = self.get_layout_engine()
            # 将当前图的布局引擎设置为新创建的 engine
            self.set_layout_engine(engine)
            # 执行紧凑布局操作
            engine.execute(self)
            # 如果之前有其他布局引擎，并且不是 TightLayoutEngine 或 PlaceHolderLayoutEngine 类型，发出警告
            if previous_engine is not None and not isinstance(
                previous_engine, (TightLayoutEngine, PlaceHolderLayoutEngine)
            ):
                _api.warn_external('The figure layout has changed to tight')
        finally:
            # 最终将图的布局引擎恢复为 'none'
            self.set_layout_engine('none')
"""
Calculate the width and height for a figure with a specified aspect ratio.

While the height is taken from :rc:`figure.figsize`, the width is
adjusted to match the desired aspect ratio. Additionally, it is ensured
that the width is in the range [4., 16.] and the height is in the range
[2., 16.]. If necessary, the default height is adjusted to ensure this.

Parameters
----------
arg : float or 2D array
    If a float, this defines the aspect ratio (i.e. the ratio height /
    width).
    In case of an array the aspect ratio is number of rows / number of
    columns, so that the array could be fitted in the figure undistorted.

Returns
-------
width, height : float
    The figure size in inches.

Notes
-----
If you want to create an Axes within the figure, that still preserves the
aspect ratio, be sure to create it with equal width and height. See
examples below.

Thanks to Fernando Perez for this function.

Examples
--------
Make a figure twice as tall as it is wide::

    w, h = figaspect(2.)
    fig = Figure(figsize=(w, h))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.imshow(A, **kwargs)

Make a figure with the proper aspect for an array::

    A = rand(5, 3)
    w, h = figaspect(A)
    fig = Figure(figsize=(w, h))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.imshow(A, **kwargs)
"""

isarray = hasattr(arg, 'shape') and not np.isscalar(arg)

# min/max sizes to respect when autoscaling.  If John likes the idea, they
# could become rc parameters, for now they're hardwired.
figsize_min = np.array((4.0, 2.0))  # min length for width/height
figsize_max = np.array((16.0, 16.0))  # max length for width/height

# Extract the aspect ratio of the array
if isarray:
    nr, nc = arg.shape[:2]
    arr_ratio = nr / nc
else:
    arr_ratio = arg

# Height of user figure defaults
fig_height = mpl.rcParams['figure.figsize'][1]

# New size for the figure, keeping the aspect ratio of the caller
newsize = np.array((fig_height / arr_ratio, fig_height))

# Sanity checks, don't drop either dimension below figsize_min
newsize /= min(1.0, *(newsize / figsize_min))

# Avoid humongous windows as well
newsize /= max(1.0, *(newsize / figsize_max))

# Finally, if we have a really funky aspect ratio, break it but respect
# the min/max dimensions (we don't want figures 10 feet tall!)
newsize = np.clip(newsize, figsize_min, figsize_max)
return newsize
```
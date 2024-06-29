# `D:\src\scipysrc\matplotlib\lib\matplotlib\layout_engine.py`

```
"""
Classes to layout elements in a `.Figure`.

Figures have a ``layout_engine`` property that holds a subclass of
`~.LayoutEngine` defined here (or *None* for no layout).  At draw time
``figure.get_layout_engine().execute()`` is called, the goal of which is
usually to rearrange Axes on the figure to produce a pleasing layout. This is
like a ``draw`` callback but with two differences.  First, when printing we
disable the layout engine for the final draw. Second, it is useful to know the
layout engine while the figure is being created.  In particular, colorbars are
made differently with different layout engines (for historical reasons).

Matplotlib supplies two layout engines, `.TightLayoutEngine` and
`.ConstrainedLayoutEngine`.  Third parties can create their own layout engine
by subclassing `.LayoutEngine`.
"""

from contextlib import nullcontext

import matplotlib as mpl

from matplotlib._constrained_layout import do_constrained_layout
from matplotlib._tight_layout import (get_subplotspec_list,
                                      get_tight_layout_figure)


class LayoutEngine:
    """
    Base class for Matplotlib layout engines.

    A layout engine can be passed to a figure at instantiation or at any time
    with `~.figure.Figure.set_layout_engine`.  Once attached to a figure, the
    layout engine ``execute`` function is called at draw time by
    `~.figure.Figure.draw`, providing a special draw-time hook.

    .. note::

       However, note that layout engines affect the creation of colorbars, so
       `~.figure.Figure.set_layout_engine` should be called before any
       colorbars are created.

    Currently, there are two properties of `LayoutEngine` classes that are
    consulted while manipulating the figure:

    - ``engine.colorbar_gridspec`` tells `.Figure.colorbar` whether to make the
       axes using the gridspec method (see `.colorbar.make_axes_gridspec`) or
       not (see `.colorbar.make_axes`);
    - ``engine.adjust_compatible`` stops `.Figure.subplots_adjust` from being
        run if it is not compatible with the layout engine.

    To implement a custom `LayoutEngine`:

    1. override ``_adjust_compatible`` and ``_colorbar_gridspec``
    2. override `LayoutEngine.set` to update *self._params*
    3. override `LayoutEngine.execute` with your implementation

    """
    # override these in subclass
    _adjust_compatible = None  # 控制与布局引擎兼容性相关的属性，默认为 None
    _colorbar_gridspec = None   # 控制 colorbar 是否使用 gridspec 方法创建的属性，默认为 None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._params = {}  # 初始化参数字典

    def set(self, **kwargs):
        """
        Set the parameters for the layout engine.
        """
        raise NotImplementedError  # 设置布局引擎参数的方法，需要在子类中实现

    @property
    def colorbar_gridspec(self):
        """
        Return a boolean if the layout engine creates colorbars using a
        gridspec.
        """
        if self._colorbar_gridspec is None:
            raise NotImplementedError
        return self._colorbar_gridspec
    def adjust_compatible(self):
        """
        返回一个布尔值，指示布局引擎是否与 `~.Figure.subplots_adjust` 兼容。
        """
        if self._adjust_compatible is None:
            # 如果 _adjust_compatible 属性为 None，则抛出未实现错误
            raise NotImplementedError
        # 返回 _adjust_compatible 属性的值
        return self._adjust_compatible

    def get(self):
        """
        返回布局引擎参数的副本。
        """
        # 返回 _params 属性的字典形式副本
        return dict(self._params)

    def execute(self, fig):
        """
        在给定的 *fig* 上执行布局。
        """
        # 子类必须实现这个方法，否则抛出未实现错误
        raise NotImplementedError
class PlaceHolderLayoutEngine(LayoutEngine):
    """
    This layout engine does not adjust the figure layout at all.

    The purpose of this `.LayoutEngine` is to act as a placeholder when the user removes
    a layout engine to ensure an incompatible `.LayoutEngine` cannot be set later.

    Parameters
    ----------
    adjust_compatible, colorbar_gridspec : bool
        Allow the PlaceHolderLayoutEngine to mirror the behavior of whatever
        layout engine it is replacing.

    """
    def __init__(self, adjust_compatible, colorbar_gridspec, **kwargs):
        # 初始化占位布局引擎，设置是否允许与其替换的布局引擎保持兼容性
        self._adjust_compatible = adjust_compatible
        # 设置是否允许调整颜色条的网格规范
        self._colorbar_gridspec = colorbar_gridspec
        super().__init__(**kwargs)

    def execute(self, fig):
        """
        Do nothing.
        """
        # 占位操作，什么也不做
        return


class TightLayoutEngine(LayoutEngine):
    """
    Implements the ``tight_layout`` geometry management.  See
    :ref:`tight_layout_guide` for details.
    """
    _adjust_compatible = True
    _colorbar_gridspec = True

    def __init__(self, *, pad=1.08, h_pad=None, w_pad=None,
                 rect=(0, 0, 1, 1), **kwargs):
        """
        Initialize tight_layout engine.

        Parameters
        ----------
        pad : float, default: 1.08
            Padding between the figure edge and the edges of subplots, as a
            fraction of the font size.
        h_pad, w_pad : float
            Padding (height/width) between edges of adjacent subplots.
            Defaults to *pad*.
        rect : tuple (left, bottom, right, top), default: (0, 0, 1, 1).
            rectangle in normalized figure coordinates that the subplots
            (including labels) will fit into.
        """
        super().__init__(**kwargs)
        # 初始化紧凑布局引擎，设置默认参数并初始化为None以便后续赋值
        for td in ['pad', 'h_pad', 'w_pad', 'rect']:
            self._params[td] = None  # 初始化参数为None，以防上面传入None
        # 设置紧凑布局引擎的参数
        self.set(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)

    def execute(self, fig):
        """
        Execute tight_layout.

        This decides the subplot parameters given the padding that
        will allow the Axes labels to not be covered by other labels
        and Axes.

        Parameters
        ----------
        fig : `.Figure` to perform layout on.

        See Also
        --------
        .figure.Figure.tight_layout
        .pyplot.tight_layout
        """
        # 获取参数信息
        info = self._params
        # 获取渲染器
        renderer = fig._get_renderer()
        # 使用上下文管理器，在renderer对象上禁用绘制操作
        with getattr(renderer, "_draw_disabled", nullcontext)():
            # 获取紧凑布局需要的参数
            kwargs = get_tight_layout_figure(
                fig, fig.axes, get_subplotspec_list(fig.axes), renderer,
                pad=info['pad'], h_pad=info['h_pad'], w_pad=info['w_pad'],
                rect=info['rect'])
        # 如果有参数返回，则调整子图布局
        if kwargs:
            fig.subplots_adjust(**kwargs)
    def set(self, *, pad=None, w_pad=None, h_pad=None, rect=None):
        """
        Set the pads for tight_layout.

        Parameters
        ----------
        pad : float
            Padding between the figure edge and the edges of subplots, as a
            fraction of the font size.
        w_pad, h_pad : float
            Padding (width/height) between edges of adjacent subplots.
            Defaults to *pad*.
        rect : tuple (left, bottom, right, top)
            rectangle in normalized figure coordinates that the subplots
            (including labels) will fit into.
        """
        # 遍历关键字参数的默认值（'pad', 'w_pad', 'h_pad', 'rect'）
        for td in self.set.__kwdefaults__:
            # 如果当前参数（如pad、w_pad等）有传入值，则更新self._params中对应的值
            if locals()[td] is not None:
                self._params[td] = locals()[td]
class ConstrainedLayoutEngine(LayoutEngine):
    """
    Implements the ``constrained_layout`` geometry management.  See
    :ref:`constrainedlayout_guide` for details.
    """

    _adjust_compatible = False  # 禁用兼容调整
    _colorbar_gridspec = False  # 禁用颜色条的网格规范

    def __init__(self, *, h_pad=None, w_pad=None,
                 hspace=None, wspace=None, rect=(0, 0, 1, 1),
                 compress=False, **kwargs):
        """
        Initialize ``constrained_layout`` settings.

        Parameters
        ----------
        h_pad, w_pad : float
            Padding around the Axes elements in inches.
            Default to :rc:`figure.constrained_layout.h_pad` and
            :rc:`figure.constrained_layout.w_pad`.
        hspace, wspace : float
            Fraction of the figure to dedicate to space between the
            axes.  These are evenly spread between the gaps between the Axes.
            A value of 0.2 for a three-column layout would have a space
            of 0.1 of the figure width between each column.
            If h/wspace < h/w_pad, then the pads are used instead.
            Default to :rc:`figure.constrained_layout.hspace` and
            :rc:`figure.constrained_layout.wspace`.
        rect : tuple of 4 floats
            Rectangle in figure coordinates to perform constrained layout in
            (left, bottom, width, height), each from 0-1.
        compress : bool
            Whether to shift Axes so that white space in between them is
            removed. This is useful for simple grids of fixed-aspect Axes (e.g.
            a grid of images).  See :ref:`compressed_layout`.
        """
        super().__init__(**kwargs)
        # set the defaults:
        self.set(w_pad=mpl.rcParams['figure.constrained_layout.w_pad'],  # 设置宽度填充默认值
                 h_pad=mpl.rcParams['figure.constrained_layout.h_pad'],  # 设置高度填充默认值
                 wspace=mpl.rcParams['figure.constrained_layout.wspace'],  # 设置宽度空间默认值
                 hspace=mpl.rcParams['figure.constrained_layout.hspace'],  # 设置高度空间默认值
                 rect=(0, 0, 1, 1))  # 设置矩形区域默认值
        # set anything that was passed in (None will be ignored):
        self.set(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace,
                 rect=rect)  # 设置传入的参数，忽略为 None 的项
        self._compress = compress  # 设置是否压缩布局标志

    def execute(self, fig):
        """
        Perform constrained_layout and move and resize Axes accordingly.

        Parameters
        ----------
        fig : `.Figure` to perform layout on.
        """
        width, height = fig.get_size_inches()
        # pads are relative to the current state of the figure...
        w_pad = self._params['w_pad'] / width  # 计算宽度填充相对于当前图形大小的比例
        h_pad = self._params['h_pad'] / height  # 计算高度填充相对于当前图形大小的比例

        return do_constrained_layout(fig, w_pad=w_pad, h_pad=h_pad,
                                     wspace=self._params['wspace'],
                                     hspace=self._params['hspace'],
                                     rect=self._params['rect'],
                                     compress=self._compress)
    # 定义一个方法 `set`，用于设置 constrained_layout 的参数
    def set(self, *, h_pad=None, w_pad=None,
            hspace=None, wspace=None, rect=None):
        """
        Set the pads for constrained_layout.

        Parameters
        ----------
        h_pad, w_pad : float
            Padding around the Axes elements in inches.
            Default to :rc:`figure.constrained_layout.h_pad` and
            :rc:`figure.constrained_layout.w_pad`.
        hspace, wspace : float
            Fraction of the figure to dedicate to space between the
            axes.  These are evenly spread between the gaps between the Axes.
            A value of 0.2 for a three-column layout would have a space
            of 0.1 of the figure width between each column.
            If h/wspace < h/w_pad, then the pads are used instead.
            Default to :rc:`figure.constrained_layout.hspace` and
            :rc:`figure.constrained_layout.wspace`.
        rect : tuple of 4 floats
            Rectangle in figure coordinates to perform constrained layout in
            (left, bottom, width, height), each from 0-1.
        """
        # 遍历方法 `set` 的关键字参数的默认值
        for td in self.set.__kwdefaults__:
            # 如果传入的关键字参数对应的本地变量不为 None，则更新实例对象的 `_params` 字典
            if locals()[td] is not None:
                self._params[td] = locals()[td]
```
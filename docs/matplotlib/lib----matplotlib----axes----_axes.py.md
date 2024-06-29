# `D:\src\scipysrc\matplotlib\lib\matplotlib\axes\_axes.py`

```py
import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real

import re  # 导入正则表达式模块
import numpy as np  # 导入 NumPy 库并用 np 别名表示
from numpy import ma  # 导入 NumPy 中的 ma 模块

import matplotlib as mpl  # 导入 Matplotlib 库并用 mpl 别名表示
import matplotlib.category  # 导入 Matplotlib 中的 category 模块，注册为副作用
import matplotlib.cbook as cbook  # 导入 Matplotlib 中的 cbook 模块
import matplotlib.collections as mcoll  # 导入 Matplotlib 中的 collections 模块
import matplotlib.colors as mcolors  # 导入 Matplotlib 中的 colors 模块
import matplotlib.contour as mcontour  # 导入 Matplotlib 中的 contour 模块
import matplotlib.dates  # noqa # 导入 Matplotlib 中的 dates 模块，注册为副作用
import matplotlib.image as mimage  # 导入 Matplotlib 中的 image 模块
import matplotlib.legend as mlegend  # 导入 Matplotlib 中的 legend 模块
import matplotlib.lines as mlines  # 导入 Matplotlib 中的 lines 模块
import matplotlib.markers as mmarkers  # 导入 Matplotlib 中的 markers 模块
import matplotlib.mlab as mlab  # 导入 Matplotlib 中的 mlab 模块
import matplotlib.patches as mpatches  # 导入 Matplotlib 中的 patches 模块
import matplotlib.path as mpath  # 导入 Matplotlib 中的 path 模块
import matplotlib.quiver as mquiver  # 导入 Matplotlib 中的 quiver 模块
import matplotlib.stackplot as mstack  # 导入 Matplotlib 中的 stackplot 模块
import matplotlib.streamplot as mstream  # 导入 Matplotlib 中的 streamplot 模块
import matplotlib.table as mtable  # 导入 Matplotlib 中的 table 模块
import matplotlib.text as mtext  # 导入 Matplotlib 中的 text 模块
import matplotlib.ticker as mticker  # 导入 Matplotlib 中的 ticker 模块
import matplotlib.transforms as mtransforms  # 导入 Matplotlib 中的 transforms 模块
import matplotlib.tri as mtri  # 导入 Matplotlib 中的 tri 模块
import matplotlib.units as munits  # 导入 Matplotlib 中的 units 模块
from matplotlib import _api, _docstring, _preprocess_data  # 导入 Matplotlib 中的 _api, _docstring, _preprocess_data
from matplotlib.axes._base import (  # 导入 Matplotlib 中 axes 模块的基类和函数
    _AxesBase, _TransformedBoundsLocator, _process_plot_format
)
from matplotlib.axes._secondary_axes import SecondaryAxis  # 导入 Matplotlib 中的 SecondaryAxis 类
from matplotlib.container import (  # 导入 Matplotlib 中 container 模块的各种容器类
    BarContainer, ErrorbarContainer, StemContainer
)

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


# The axes module contains all the wrappers to plotting functions.
# All the other methods should go in the _AxesBase class.


def _make_axes_method(func):
    """
    Patch the qualname for functions that are directly added to Axes.

    Some Axes functionality is defined in functions in other submodules.
    These are simply added as attributes to Axes. As a result, their
    ``__qualname__`` is e.g. only "table" and not "Axes.table". This
    function fixes that.

    Note that the function itself is patched, so that
    ``matplotlib.table.table.__qualname__`` will also show "Axes.table".
    However, since these functions are not intended to be standalone,
    this is bearable.
    """
    func.__qualname__ = f"Axes.{func.__name__}"  # 修正直接添加到 Axes 的函数的 __qualname__ 属性
    return func


@_docstring.interpd
class Axes(_AxesBase):
    """
    An Axes object encapsulates all the elements of an individual (sub-)plot in
    a figure.

    It contains most of the (sub-)plot elements: `~.axis.Axis`,
    `~.axis.Tick`, `~.lines.Line2D`, `~.text.Text`, `~.patches.Polygon`, etc.,
    and sets the coordinate system.

    Like all visible elements in a figure, Axes is an `.Artist` subclass.

    The `Axes` instance supports callbacks through a callbacks attribute which
    is a `~.cbook.CallbackRegistry` instance.  The events you can connect to
    are 'xlim_changed' and 'ylim_changed' and the callback will be called with
    func(*ax*) where *ax* is the `Axes` instance.
    """
    """
    .. note::

        As a user, you do not instantiate Axes directly, but use Axes creation
        methods instead; e.g. from `.pyplot` or `.Figure`:
        `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` or `.Figure.add_axes`.

    Attributes
    ----------
    dataLim : `.Bbox`
        The bounding box enclosing all data displayed in the Axes.
    viewLim : `.Bbox`
        The view limits in data coordinates.

    """
    ### Labelling, legend and texts

    # 定义一个方法用于获取 Axes 的标题
    def get_title(self, loc="center"):
        """
        Get an Axes title.

        Get one of the three available Axes titles. The available titles
        are positioned above the Axes in the center, flush with the left
        edge, and flush with the right edge.

        Parameters
        ----------
        loc : {'center', 'left', 'right'}, str, default: 'center'
            Which title to return.

        Returns
        -------
        str
            The title text string.

        """
        # 根据 loc 参数返回对应位置的标题文本
        titles = {'left': self._left_title,
                  'center': self.title,
                  'right': self._right_title}
        title = _api.check_getitem(titles, loc=loc.lower())
        return title.get_text()

    # 获取图例的句柄和标签文本
    def get_legend_handles_labels(self, legend_handler_map=None):
        """
        Return handles and labels for legend

        ``ax.legend()`` is equivalent to ::

          h, l = ax.get_legend_handles_labels()
          ax.legend(h, l)
        """
        # 通过 mlegend 模块的函数获取图例句柄和标签文本
        handles, labels = mlegend._get_legend_handles_labels(
            [self], legend_handler_map)
        return handles, labels

    # 删除图例
    @_docstring.dedent_interpd
    def _remove_legend(self, legend):
        self.legend_ = None

    # 添加一个子插入 Axes 到当前 Axes 中
    def inset_axes(self, bounds, *, transform=None, zorder=5, **kwargs):
        """
        Add a child inset Axes to this existing Axes.


        Parameters
        ----------
        bounds : [x0, y0, width, height]
            Lower-left corner of inset Axes, and its width and height.

        transform : `.Transform`
            Defaults to `ax.transAxes`, i.e. the units of *rect* are in
            Axes-relative coordinates.

        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
# 如果未提供 transform 参数，则默认使用 self.transAxes 进行坐标变换
if transform is None:
    transform = self.transAxes

# 设置 'label' 关键字参数为 'inset_axes'，如果未显式提供则使用默认值
kwargs.setdefault('label', 'inset_axes')

# 创建一个 _TransformedBoundsLocator 对象，用于根据 bounds 和 transform 来定位 inset 的位置
inset_locator = _TransformedBoundsLocator(bounds, transform)

# 调用 inset_locator 的 __call__ 方法计算边界，并将计算得到的 bounds 作为新的边界
bounds = inset_locator(self, None).bounds

# 根据 kwargs 处理投影类和参数，返回投影类和处理后的关键字参数字典 pkw
projection_class, pkw = self.figure._process_projection_requirements(**kwargs)

# 使用 projection_class 创建一个新的 inset_axes，并传入 figure、bounds、zorder 和 pkw
inset_ax = projection_class(self.figure, bounds, zorder=zorder, **pkw)

# 将 inset_ax 的 axes_locator 设置为 inset_locator，用于在 apply_aspect() 中调整坐标轴位置
inset_ax.set_axes_locator(inset_locator)

# 将创建的 inset_ax 添加为当前 axes 的子 axes
self.add_child_axes(inset_ax)

# 返回创建的 inset_ax 实例作为结果
return inset_ax
    def indicate_inset_zoom(self, inset_ax, **kwargs):
        """
        在主 Axes 上基于 inset_ax 的坐标轴限制添加一个插图指示器矩形，并在 inset_ax 和矩形之间绘制连接线。

        Warnings
        --------
        该方法在版本 3.0 时为实验性方法，API 可能会更改。

        Parameters
        ----------
        inset_ax : `.Axes`
            要连接的插图 Axes。绘制两条连接线，连接指示框与选择的角落不重叠的 inset Axes。

        **kwargs
            其他关键字参数传递给 `.Axes.indicate_inset`

        Returns
        -------
        rectangle_patch : `.patches.Rectangle`
            矩形图形对象。

        connector_lines : 4-tuple of `.patches.ConnectionPatch`
            从在此轴上绘制的矩形出发的四个连接线，顺序为左下、左上、右下、右上。
            其中两条的可见性设置为 False，但用户可以将其可见性设置为 True，如果自动选择不正确的话。
        """

        # 获取 inset_ax 的 x 和 y 轴限制
        xlim = inset_ax.get_xlim()
        ylim = inset_ax.get_ylim()

        # 计算矩形的位置和大小
        rect = (xlim[0], ylim[0], xlim[1] - xlim[0], ylim[1] - ylim[0])

        # 调用 self.indicate_inset 方法来在主 Axes 上绘制指示器
        return self.indicate_inset(rect, inset_ax, **kwargs)
    # 定义了一个方法用于添加第二个 x 轴到当前的 Axes 对象中
    def secondary_xaxis(self, location, functions=None, *, transform=None, **kwargs):
        """
        Add a second x-axis to this `~.axes.Axes`.

        将第二个 x 轴添加到当前的 `~.axes.Axes` 中。

        For example if we want to have a second scale for the data plotted on
        the xaxis.

        例如，如果我们希望在 x 轴上绘制数据时有第二个比例尺。

        %(_secax_docstring)s

        Examples
        --------
        The main axis shows frequency, and the secondary axis shows period.

        主要的坐标轴显示频率，第二个坐标轴显示周期。

        .. plot::

            fig, ax = plt.subplots()
            ax.loglog(range(1, 360, 5), range(1, 360, 5))
            ax.set_xlabel('frequency [Hz]')

            def invert(x):
                # 1/x with special treatment of x == 0
                # 对于 x == 0，特殊处理为无穷大
                x = np.array(x).astype(float)
                near_zero = np.isclose(x, 0)
                x[near_zero] = np.inf
                x[~near_zero] = 1 / x[~near_zero]
                return x

            # the inverse of 1/x is itself
            # 1/x 的反函数是其本身
            secax = ax.secondary_xaxis('top', functions=(invert, invert))
            secax.set_xlabel('Period [s]')
            plt.show()

        To add a secondary axis relative to your data, you can pass a transform
        to the new axis.

        要相对于您的数据添加第二个轴，可以通过传递一个 transform 到新轴来实现。

        .. plot::

            fig, ax = plt.subplots()
            ax.plot(range(0, 5), range(-1, 4))

            # Pass 'ax.transData' as a transform to place the axis
            # relative to your data at y=0
            # 通过将 'ax.transData' 作为 transform 传递，将轴相对于数据放置在 y=0 处
            secax = ax.secondary_xaxis(0, transform=ax.transData)
        """
        # 检查位置参数是否有效，必须是 'top'、'bottom' 或实数
        if not (location in ['top', 'bottom'] or isinstance(location, Real)):
            raise ValueError('secondary_xaxis location must be either '
                             'a float or "top"/"bottom"')

        # 创建一个 SecondaryAxis 对象，用于管理第二个 x 轴的相关设置
        secondary_ax = SecondaryAxis(self, 'x', location, functions,
                                     transform, **kwargs)
        # 将新创建的 SecondaryAxis 对象添加为当前 Axes 对象的子 Axes
        self.add_child_axes(secondary_ax)
        # 返回添加的 SecondaryAxis 对象，以便进一步操作或设置
        return secondary_ax

    # 从注释的文字中去除空格和换行符，使得文档字符串能够被正确格式化和解析
    @_docstring.dedent_interpd
    # 定义方法，用于在当前 Axes 对象中添加第二个 y 轴

    """
    给这个 `~.axes.Axes` 添加第二个 y 轴。

    例如，如果我们希望对 y 轴上的数据进行第二个比例尺的绘制。

    %(_secax_docstring)s

    Examples
    --------
    添加一个将角度转换为弧度的辅助 Axes

    .. plot::

        fig, ax = plt.subplots()
        ax.plot(range(1, 360, 5), range(1, 360, 5))
        ax.set_ylabel('degrees')
        secax = ax.secondary_yaxis('right', functions=(np.deg2rad,
                                                       np.rad2deg))
        secax.set_ylabel('radians')

    要添加一个相对于数据的辅助轴，可以将一个 transform 传递给新轴。

    .. plot::

        fig, ax = plt.subplots()
        ax.plot(range(0, 5), range(-1, 4))

        # 传递 'ax.transData' 作为一个 transform，以在 x=3 处放置轴
        secax = ax.secondary_yaxis(3, transform=ax.transData)
    """

    # 检查 location 参数是否为 'left'、'right' 或实数类型，否则抛出值错误异常
    if not (location in ['left', 'right'] or isinstance(location, Real)):
        raise ValueError('secondary_yaxis location must be either '
                         'a float or "left"/"right"')

    # 创建 SecondaryAxis 对象，代表辅助 y 轴
    secondary_ax = SecondaryAxis(self, 'y', location, functions,
                                 transform, **kwargs)

    # 将创建的辅助轴对象添加到当前 Axes 对象的子对象列表中
    self.add_child_axes(secondary_ax)

    # 返回添加的 SecondaryAxis 对象，以供进一步操作或设置
    return secondary_ax
    # 在 Axes 中添加文本

    # 设置文本的默认水平对齐为左对齐，垂直对齐为基线
    effective_kwargs = {
        'verticalalignment': 'baseline',
        'horizontalalignment': 'left',
        'transform': self.transData,  # 使用数据坐标系作为默认坐标系
        'clip_on': False,  # 禁用文本剪裁
        **(fontdict if fontdict is not None else {}),  # 使用字典 *fontdict* 覆盖默认文本属性
        **kwargs,  # 使用额外传入的关键字参数覆盖默认文本属性
    }

    # 创建文本对象，传入指定的位置坐标和文本内容以及所有的关键字参数
    t = mtext.Text(x, y, text=s, **effective_kwargs)

    # 如果文本对象没有设置剪裁路径，则使用 Axes 的路径作为剪裁路径
    if t.get_clip_path() is None:
        t.set_clip_path(self.patch)

    # 将文本对象添加到 Axes 中
    self._add_text(t)

    # 返回创建的文本对象
    return t
    # 定义一个方法 annotate，用于在图表上添加注释
    def annotate(self, text, xy, xytext=None, xycoords='data', textcoords=None,
                 arrowprops=None, annotation_clip=None, **kwargs):
        # 创建一个 Annotation 对象，用给定的参数初始化
        a = mtext.Annotation(text, xy, xytext=xytext, xycoords=xycoords,
                             textcoords=textcoords, arrowprops=arrowprops,
                             annotation_clip=annotation_clip, **kwargs)
        # 设置 Annotation 对象的变换为 IdentityTransform，即无变换
        a.set_transform(mtransforms.IdentityTransform())
        # 如果 kwargs 中设置了 'clip_on' 为 True，并且 Annotation 对象没有设置剪切路径，则将其设置为图表的补丁路径（self.patch）
        if kwargs.get('clip_on', False) and a.get_clip_path() is None:
            a.set_clip_path(self.patch)
        # 将 Annotation 对象添加到图表中
        self._add_text(a)
        # 返回创建的 Annotation 对象
        return a
    # 将 annotate 方法的文档字符串设置为 mtext.Annotation.__init__ 方法的文档字符串
    annotate.__doc__ = mtext.Annotation.__init__.__doc__
    #### Lines and spans

    # 使用 _docstring.dedent_interpd 对象进行处理
    def axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        """
        Add a horizontal line across the Axes.

        Parameters
        ----------
        y : float, default: 0
            y position in data coordinates of the horizontal line.

        xmin : float, default: 0
            Should be between 0 and 1, 0 being the far left of the plot, 1 the
            far right of the plot.

        xmax : float, default: 1
            Should be between 0 and 1, 0 being the far left of the plot, 1 the
            far right of the plot.

        Returns
        -------
        `~matplotlib.lines.Line2D`

        Other Parameters
        ----------------
        **kwargs
            Valid keyword arguments are `.Line2D` properties, except for
            'transform':

            %(Line2D:kwdoc)s

        See Also
        --------
        hlines : Add horizontal lines in data coordinates.
        axhspan : Add a horizontal span (rectangle) across the axis.
        axline : Add a line with an arbitrary slope.

        Examples
        --------
        * draw a thick red hline at 'y' = 0 that spans the xrange::

            >>> axhline(linewidth=4, color='r')

        * draw a default hline at 'y' = 1 that spans the xrange::

            >>> axhline(y=1)

        * draw a default hline at 'y' = .5 that spans the middle half of
          the xrange::

            >>> axhline(y=.5, xmin=0.25, xmax=0.75)
        """
        # 检查是否存在单位，对xmin和xmax进行单位处理
        self._check_no_units([xmin, xmax], ['xmin', 'xmax'])
        # 如果kwargs中包含'transform'，抛出异常，因为'axhline'会自动生成其transform
        if "transform" in kwargs:
            raise ValueError("'transform' is not allowed as a keyword "
                             "argument; axhline generates its own transform.")
        # 获取当前y轴的边界范围
        ymin, ymax = self.get_ybound()

        # 去除y的单位，并根据kwargs处理单位信息
        yy, = self._process_unit_info([("y", y)], kwargs)
        # 检查是否需要自动缩放y轴
        scaley = (yy < ymin) or (yy > ymax)

        # 获取y轴的变换方式，用于绘制网格线
        trans = self.get_yaxis_transform(which='grid')
        # 创建Line2D对象，表示水平线段，并添加到当前Axes中
        l = mlines.Line2D([xmin, xmax], [y, y], transform=trans, **kwargs)
        self.add_line(l)
        # 设置网格线的插值步数
        l.get_path()._interpolation_steps = mpl.axis.GRIDLINE_INTERPOLATION_STEPS
        # 如果需要缩放y轴，则请求自动缩放视图
        if scaley:
            self._request_autoscale_view("y")
        # 返回创建的Line2D对象
        return l

    @_docstring.dedent_interpd
    def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        """
        Add a vertical line across the Axes.

        Parameters
        ----------
        x : float, default: 0
            x position in data coordinates of the vertical line.

        ymin : float, default: 0
            Should be between 0 and 1, 0 being the bottom of the plot, 1 the
            top of the plot.

        ymax : float, default: 1
            Should be between 0 and 1, 0 being the bottom of the plot, 1 the
            top of the plot.

        Returns
        -------
        `~matplotlib.lines.Line2D`

        Other Parameters
        ----------------
        **kwargs
            Valid keyword arguments are `.Line2D` properties, except for
            'transform':

            %(Line2D:kwdoc)s

        See Also
        --------
        vlines : Add vertical lines in data coordinates.
        axvspan : Add a vertical span (rectangle) across the axis.
        axline : Add a line with an arbitrary slope.

        Examples
        --------
        * draw a thick red vline at *x* = 0 that spans the yrange::

            >>> axvline(linewidth=4, color='r')

        * draw a default vline at *x* = 1 that spans the yrange::

            >>> axvline(x=1)

        * draw a default vline at *x* = .5 that spans the middle half of
          the yrange::

            >>> axvline(x=.5, ymin=0.25, ymax=0.75)
        """
        # 检查 'ymin' 和 'ymax' 参数是否含有单位，若含有则引发错误
        self._check_no_units([ymin, ymax], ['ymin', 'ymax'])

        # 如果用户传入了 'transform' 参数，则抛出错误，因为 'axvline' 会生成自己的 transform
        if "transform" in kwargs:
            raise ValueError("'transform' is not allowed as a keyword "
                             "argument; axvline generates its own transform.")

        # 获取当前图表的 x 轴数据范围
        xmin, xmax = self.get_xbound()

        # 处理 'x' 参数，去除可能存在的单位，用于与非单位化的边界进行比较
        xx, = self._process_unit_info([("x", x)], kwargs)
        scalex = (xx < xmin) or (xx > xmax)

        # 获取 x 轴的 transform
        trans = self.get_xaxis_transform(which='grid')

        # 创建一个 Line2D 对象表示垂直线段，并添加到图表中
        l = mlines.Line2D([x, x], [ymin, ymax], transform=trans, **kwargs)
        self.add_line(l)

        # 设置线段路径的插值步数
        l.get_path()._interpolation_steps = mpl.axis.GRIDLINE_INTERPOLATION_STEPS

        # 如果 x 值超出当前的数据范围，则请求自动调整 x 轴的视图范围
        if scalex:
            self._request_autoscale_view("x")

        # 返回创建的 Line2D 对象
        return l
    def axline(self, xy1, xy2=None, *, slope=None, **kwargs):
        """
        Add an infinitely long straight line.

        The line can be defined either by two points *xy1* and *xy2*, or
        by one point *xy1* and a *slope*.

        This draws a straight line "on the screen", regardless of the x and y
        scales, and is thus also suitable for drawing exponential decays in
        semilog plots, power laws in loglog plots, etc. However, *slope*
        should only be used with linear scales; It has no clear meaning for
        all other scales, and thus the behavior is undefined. Please specify
        the line using the points *xy1*, *xy2* for non-linear scales.

        The *transform* keyword argument only applies to the points *xy1*,
        *xy2*. The *slope* (if given) is always in data coordinates. This can
        be used e.g. with ``ax.transAxes`` for drawing grid lines with a fixed
        slope.

        Parameters
        ----------
        xy1, xy2 : (float, float)
            Points for the line to pass through.
            Either *xy2* or *slope* has to be given.
        slope : float, optional
            The slope of the line. Either *xy2* or *slope* has to be given.

        Returns
        -------
        `.AxLine`

        Other Parameters
        ----------------
        **kwargs
            Valid kwargs are `.Line2D` properties

            %(Line2D:kwdoc)s

        See Also
        --------
        axhline : for horizontal lines
        axvline : for vertical lines

        Examples
        --------
        Draw a thick red line passing through (0, 0) and (1, 1)::

            >>> axline((0, 0), (1, 1), linewidth=4, color='r')
        """
        # 检查是否使用了非线性刻度且同时指定了斜率
        if slope is not None and (self.get_xscale() != 'linear' or
                                  self.get_yscale() != 'linear'):
            raise TypeError("'slope' cannot be used with non-linear scales")

        # 根据传入的参数确定线段的数据范围
        datalim = [xy1] if xy2 is None else [xy1, xy2]
        # 如果传入了变换参数，则线段不在数据空间，无需调整数据范围
        if "transform" in kwargs:
            datalim = []

        # 创建一个AxLine对象表示线段
        line = mlines.AxLine(xy1, xy2, slope, **kwargs)
        # 设置线段的属性
        self._set_artist_props(line)
        # 如果未指定裁剪路径，则使用图形的裁剪路径
        if line.get_clip_path() is None:
            line.set_clip_path(self.patch)
        # 如果未设置标签，则设置一个默认标签
        if not line.get_label():
            line.set_label(f"_child{len(self._children)}")
        # 将线段添加到子元素列表中
        self._children.append(line)
        # 设置线段的移除方法
        line._remove_method = self._children.remove
        # 更新数据范围
        self.update_datalim(datalim)

        # 请求自动调整视图
        self._request_autoscale_view()
        # 返回创建的线段对象
        return line

    @_docstring.dedent_interpd
    def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kwargs):
        """
        Add a horizontal span (rectangle) across the Axes.

        The rectangle spans from *ymin* to *ymax* vertically, and, by default,
        the whole x-axis horizontally.  The x-span can be set using *xmin*
        (default: 0) and *xmax* (default: 1) which are in axis units; e.g.
        ``xmin = 0.5`` always refers to the middle of the x-axis regardless of
        the limits set by `~.Axes.set_xlim`.

        Parameters
        ----------
        ymin : float
            Lower y-coordinate of the span, in data units.
        ymax : float
            Upper y-coordinate of the span, in data units.
        xmin : float, default: 0
            Lower x-coordinate of the span, in x-axis (0-1) units.
        xmax : float, default: 1
            Upper x-coordinate of the span, in x-axis (0-1) units.

        Returns
        -------
        `~matplotlib.patches.Rectangle`
            Horizontal span (rectangle) from (xmin, ymin) to (xmax, ymax).

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Rectangle` properties

        %(Rectangle:kwdoc)s

        See Also
        --------
        axvspan : Add a vertical span across the Axes.
        """
        # Strip units away.
        # 移除 xmin 和 xmax 的单位信息，确保它们是纯数值
        self._check_no_units([xmin, xmax], ['xmin', 'xmax'])

        # 处理 ymin 和 ymax 的单位信息，确保它们符合当前坐标轴的单位要求
        (ymin, ymax), = self._process_unit_info([("y", [ymin, ymax])], kwargs)

        # 创建一个矩形对象 p，其左下角坐标为 (xmin, ymin)，宽度为 (xmax - xmin)，高度为 (ymax - ymin)，其它属性由 kwargs 指定
        p = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)

        # 设置矩形对象 p 的坐标变换，使其按 y 轴的网格线变换
        p.set_transform(self.get_yaxis_transform(which="grid"))

        # 修复由于矩形和非可分离的变换而导致的 add_patch 行为异常，避免不必要的 x 轴限制更新
        ix = self.dataLim.intervalx
        mx = self.dataLim.minposx
        self.add_patch(p)
        self.dataLim.intervalx = ix
        self.dataLim.minposx = mx

        # 设置矩形对象 p 的路径的插值步数，使用 matplotlib 中定义的网格线插值步数
        p.get_path()._interpolation_steps = mpl.axis.GRIDLINE_INTERPOLATION_STEPS

        # 请求自动调整视图，使 y 轴自适应
        self._request_autoscale_view("y")

        # 返回创建的矩形对象 p
        return p
    def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        """
        Add a vertical span (rectangle) across the Axes.

        The rectangle spans from *xmin* to *xmax* horizontally, and, by
        default, the whole y-axis vertically.  The y-span can be set using
        *ymin* (default: 0) and *ymax* (default: 1) which are in axis units;
        e.g. ``ymin = 0.5`` always refers to the middle of the y-axis
        regardless of the limits set by `~.Axes.set_ylim`.

        Parameters
        ----------
        xmin : float
            Lower x-coordinate of the span, in data units.
        xmax : float
            Upper x-coordinate of the span, in data units.
        ymin : float, default: 0
            Lower y-coordinate of the span, in y-axis units (0-1).
        ymax : float, default: 1
            Upper y-coordinate of the span, in y-axis units (0-1).

        Returns
        -------
        `~matplotlib.patches.Rectangle`
            Vertical span (rectangle) from (xmin, ymin) to (xmax, ymax).

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Rectangle` properties

        %(Rectangle:kwdoc)s

        See Also
        --------
        axhspan : Add a horizontal span across the Axes.

        Examples
        --------
        Draw a vertical, green, translucent rectangle from x = 1.25 to
        x = 1.55 that spans the yrange of the Axes.

        >>> axvspan(1.25, 1.55, facecolor='g', alpha=0.5)

        """
        # Strip units away.
        self._check_no_units([ymin, ymax], ['ymin', 'ymax'])
        (xmin, xmax), = self._process_unit_info([("x", [xmin, xmax])], kwargs)

        # 创建一个矩形对象，表示垂直跨度区域，使用给定的参数
        p = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)
        # 设置矩形的坐标变换方式为 x 轴的网格线坐标变换方式
        p.set_transform(self.get_xaxis_transform(which="grid"))

        # 由于矩形和非分离变换可能导致 add_patch 函数存在 bug，
        # 即使对于 xaxis_transformed 的 patch 也会更新 y 轴限制，因此需要撤销该更新
        iy = self.dataLim.intervaly.copy()
        my = self.dataLim.minposy
        self.add_patch(p)
        self.dataLim.intervaly = iy
        self.dataLim.minposy = my

        # 设置矩形路径的插值步骤为 matplotlib 的网格线插值步骤数
        p.get_path()._interpolation_steps = mpl.axis.GRIDLINE_INTERPOLATION_STEPS

        # 请求自动调整 x 轴视图
        self._request_autoscale_view("x")
        return p

    @_api.make_keyword_only("3.9", "label")
    @_preprocess_data(replace_names=["y", "xmin", "xmax", "colors"],
                      label_namer="y")
    @_api.make_keyword_only("3.9", "label")
    @_preprocess_data(replace_names=["x", "ymin", "ymax", "colors"],
                      label_namer="x")
    @_api.make_keyword_only("3.9", "orientation")
    @_preprocess_data(replace_names=["positions", "lineoffsets",
                                     "linelengths", "linewidths",
                                     "colors", "linestyles"])
    @_docstring.dedent_interpd
    # 使用一个自定义的数据-kwarg处理的实现方式
    #### Basic plotting
    # 使用装饰器 `_docstring.dedent_interpd` 对函数的文档字符串进行缩进处理和内插
    # 使用装饰器 `_api.deprecated` 标记此函数已在版本 3.9 被废弃，推荐使用 `plot` 替代
    # 使用装饰器 `_preprocess_data` 对数据进行预处理，替换变量名为 ["x", "y"]，设置标签名生成器为 "y"
    # 使用装饰器 `_docstring.dedent_interpd` 对函数的文档字符串再次进行缩进处理和内插
    def plot_date(self, x, y, fmt='o', tz=None, xdate=True, ydate=False,
                  **kwargs):
        """
        Plot coercing the axis to treat floats as dates.

        .. deprecated:: 3.9

            This method exists for historic reasons and will be removed in version 3.11.

            - ``datetime``-like data should directly be plotted using
              `~.Axes.plot`.
            -  If you need to plot plain numeric data as :ref:`date-format` or
               need to set a timezone, call ``ax.xaxis.axis_date`` /
               ``ax.yaxis.axis_date`` before `~.Axes.plot`. See
               `.Axis.axis_date`.

        Similar to `.plot`, this plots *y* vs. *x* as lines or markers.
        However, the axis labels are formatted as dates depending on *xdate*
        and *ydate*.  Note that `.plot` will work with `datetime` and
        `numpy.datetime64` objects without resorting to this method.

        Parameters
        ----------
        x, y : array-like
            The coordinates of the data points. If *xdate* or *ydate* is
            *True*, the respective values *x* or *y* are interpreted as
            :ref:`Matplotlib dates <date-format>`.

        fmt : str, optional
            The plot format string. For details, see the corresponding
            parameter in `.plot`.

        tz : timezone string or `datetime.tzinfo`, default: :rc:`timezone`
            The time zone to use in labeling dates.

        xdate : bool, default: True
            If *True*, the *x*-axis will be interpreted as Matplotlib dates.

        ydate : bool, default: False
            If *True*, the *y*-axis will be interpreted as Matplotlib dates.

        Returns
        -------
        list of `.Line2D`
            Objects representing the plotted data.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        See Also
        --------
        matplotlib.dates : Helper functions on dates.
        matplotlib.dates.date2num : Convert dates to num.
        matplotlib.dates.num2date : Convert num to dates.
        matplotlib.dates.drange : Create an equally spaced sequence of dates.

        Notes
        -----
        If you are using custom date tickers and formatters, it may be
        necessary to set the formatters/locators after the call to
        `.plot_date`. `.plot_date` will set the default tick locator to
        `.AutoDateLocator` (if the tick locator is not already set to a
        `.DateLocator` instance) and the default tick formatter to
        `.AutoDateFormatter` (if the tick formatter is not already set to a
        `.DateFormatter` instance).
        """
        # 如果 xdate 参数为 True，则调用 self.xaxis_date 方法设置 x 轴为日期格式，可以传入时区信息 tz
        if xdate:
            self.xaxis_date(tz)
        # 如果 ydate 参数为 True，则调用 self.yaxis_date 方法设置 y 轴为日期格式，可以传入时区信息 tz
        if ydate:
            self.yaxis_date(tz)
        # 调用 self.plot 方法进行数据的绘制，返回绘制的线或标记对象列表
        return self.plot(x, y, fmt, **kwargs)
    # 装饰器：使用 _docstring.dedent_interpd 处理函数文档字符串的缩进和插值
    @_docstring.dedent_interpd
    # 定义 loglog 方法，实现在 x 和 y 轴上同时使用对数标尺绘图
    def loglog(self, *args, **kwargs):
        """
        Make a plot with log scaling on both the x- and y-axis.

        Call signatures::

            loglog([x], y, [fmt], data=None, **kwargs)
            loglog([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)

        This is just a thin wrapper around `.plot` which additionally changes
        both the x-axis and the y-axis to log scaling. All the concepts and
        parameters of plot can be used here as well.

        The additional parameters *base*, *subs* and *nonpositive* control the
        x/y-axis properties. They are just forwarded to `.Axes.set_xscale` and
        `.Axes.set_yscale`. To use different properties on the x-axis and the
        y-axis, use e.g.
        ``ax.set_xscale("log", base=10); ax.set_yscale("log", base=2)``.

        Parameters
        ----------
        base : float, default: 10
            Base of the logarithm.

        subs : sequence, optional
            The location of the minor ticks. If *None*, reasonable locations
            are automatically chosen depending on the number of decades in the
            plot. See `.Axes.set_xscale`/`.Axes.set_yscale` for details.

        nonpositive : {'mask', 'clip'}, default: 'clip'
            Non-positive values can be masked as invalid, or clipped to a very
            small positive number.

        **kwargs
            All parameters supported by `.plot`.

        Returns
        -------
        list of `.Line2D`
            Objects representing the plotted data.
        """
        # 从 kwargs 中提取与 x 轴相关的参数，传递给 set_xscale 方法设置对数标尺
        dx = {k: v for k, v in kwargs.items()
              if k in ['base', 'subs', 'nonpositive',
                       'basex', 'subsx', 'nonposx']}
        self.set_xscale('log', **dx)
        # 从 kwargs 中提取与 y 轴相关的参数，传递给 set_yscale 方法设置对数标尺
        dy = {k: v for k, v in kwargs.items()
              if k in ['base', 'subs', 'nonpositive',
                       'basey', 'subsy', 'nonposy']}
        self.set_yscale('log', **dy)
        # 调用 plot 方法进行绘图，传递除了与对数标尺相关的参数之外的所有 kwargs
        return self.plot(
            *args, **{k: v for k, v in kwargs.items() if k not in {*dx, *dy}})
    # 定义一个方法用于绘制以半对数坐标（对数横坐标）为 x 轴的图形

    # 将所有参数解析成一个字典 d，只保留与 x 轴属性相关的参数：'base', 'subs', 'nonpositive',
    # 'basex', 'subsx', 'nonposx'，并将其传递给 set_xscale 方法
    d = {k: v for k, v in kwargs.items()
         if k in ['base', 'subs', 'nonpositive',
                  'basex', 'subsx', 'nonposx']}

    # 调用父类的方法设置 x 轴为对数坐标轴，使用上面解析得到的参数
    self.set_xscale('log', **d)

    # 调用父类的 plot 方法绘制图形，传递除了上面解析过的参数之外的所有参数
    return self.plot(
        *args, **{k: v for k, v in kwargs.items() if k not in d})
    def semilogy(self, *args, **kwargs):
        """
        Make a plot with log scaling on the y-axis.

        Call signatures::

            semilogy([x], y, [fmt], data=None, **kwargs)
            semilogy([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)

        This is just a thin wrapper around `.plot` which additionally changes
        the y-axis to log scaling. All the concepts and parameters of plot can
        be used here as well.

        The additional parameters *base*, *subs*, and *nonpositive* control the
        y-axis properties. They are just forwarded to `.Axes.set_yscale`.

        Parameters
        ----------
        base : float, default: 10
            Base of the y logarithm.

        subs : array-like, optional
            The location of the minor yticks. If *None*, reasonable locations
            are automatically chosen depending on the number of decades in the
            plot. See `.Axes.set_yscale` for details.

        nonpositive : {'mask', 'clip'}, default: 'clip'
            Non-positive values in y can be masked as invalid, or clipped to a
            very small positive number.

        **kwargs
            All parameters supported by `.plot`.

        Returns
        -------
        list of `.Line2D`
            Objects representing the plotted data.
        """
        # 从 kwargs 中提取出 base, subs, nonpositive 等参数，用于设置 y 轴的对数尺度
        d = {k: v for k, v in kwargs.items()
             if k in ['base', 'subs', 'nonpositive',
                      'basey', 'subsy', 'nonposy']}
        # 调用当前对象的 set_yscale 方法，将 y 轴的尺度设置为对数尺度，并传入参数 d
        self.set_yscale('log', **d)
        # 调用当前对象的 plot 方法，绘制图形，并传入除了 d 之外的其他 kwargs 参数
        return self.plot(
            *args, **{k: v for k, v in kwargs.items() if k not in d})

    @_preprocess_data(replace_names=["x"], label_namer="x")
    def acorr(self, x, **kwargs):
        """
        Plot the autocorrelation of *x*.

        Parameters
        ----------
        x : array-like
            Not run through Matplotlib's unit conversion, so this should
            be a unit-less array.

        detrend : callable, default: `.mlab.detrend_none` (no detrending)
            A detrending function applied to *x*.  It must have the
            signature ::

                detrend(x: np.ndarray) -> np.ndarray

        normed : bool, default: True
            If ``True``, input vectors are normalised to unit length.

        usevlines : bool, default: True
            Determines the plot style.

            If ``True``, vertical lines are plotted from 0 to the acorr value
            using `.Axes.vlines`. Additionally, a horizontal line is plotted
            at y=0 using `.Axes.axhline`.

            If ``False``, markers are plotted at the acorr values using
            `.Axes.plot`.

        maxlags : int, default: 10
            Number of lags to show. If ``None``, will return all
            ``2 * len(x) - 1`` lags.

        Returns
        -------
        lags : array (length ``2*maxlags+1``)
            The lag vector.
        c : array  (length ``2*maxlags+1``)
            The auto correlation vector.
        line : `.LineCollection` or `.Line2D`
            `.Artist` added to the Axes of the correlation:

            - `.LineCollection` if *usevlines* is True.
            - `.Line2D` if *usevlines* is False.
        b : `~matplotlib.lines.Line2D` or None
            Horizontal line at 0 if *usevlines* is True
            None *usevlines* is False.

        Other Parameters
        ----------------
        linestyle : `~matplotlib.lines.Line2D` property, optional
            The linestyle for plotting the data points.
            Only used if *usevlines* is ``False``.

        marker : str, default: 'o'
            The marker for plotting the data points.
            Only used if *usevlines* is ``False``.

        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Additional parameters are passed to `.Axes.vlines` and
            `.Axes.axhline` if *usevlines* is ``True``; otherwise they are
            passed to `.Axes.plot`.

        Notes
        -----
        The cross correlation is performed with `numpy.correlate` with
        ``mode = "full"``.
        """
        # 调用 xcorr 方法计算自相关，并传递所有额外的关键字参数
        return self.xcorr(x, x, **kwargs)

    @_api.make_keyword_only("3.9", "normed")
    @_preprocess_data(replace_names=["x", "y"], label_namer="y")
    # 特定的绘图设置
    #### Specialized plotting

    # @_preprocess_data() # let 'plot' do the unpacking..
    def step(self, x, y, *args, where='pre', data=None, **kwargs):
        """
        Make a step plot.

        Call signatures::

            step(x, y, [fmt], *, data=None, where='pre', **kwargs)
            step(x, y, [fmt], x2, y2, [fmt2], ..., *, where='pre', **kwargs)

        This is just a thin wrapper around `.plot` which changes some
        formatting options. Most of the concepts and parameters of plot can be
        used here as well.

        .. note::

            This method uses a standard plot with a step drawstyle: The *x*
            values are the reference positions and steps extend left/right/both
            directions depending on *where*.

            For the common case where you know the values and edges of the
            steps, use `~.Axes.stairs` instead.

        Parameters
        ----------
        x : array-like
            1D sequence of x positions. It is assumed, but not checked, that
            it is uniformly increasing.

        y : array-like
            1D sequence of y levels.

        fmt : str, optional
            A format string, e.g. 'g' for a green line. See `.plot` for a more
            detailed description.

            Note: While full format strings are accepted, it is recommended to
            only specify the color. Line styles are currently ignored (use
            the keyword argument *linestyle* instead). Markers are accepted
            and plotted on the given positions, however, this is a rarely
            needed feature for step plots.

        where : {'pre', 'post', 'mid'}, default: 'pre'
            Define where the steps should be placed:

            - 'pre': The y value is continued constantly to the left from
              every *x* position, i.e. the interval ``(x[i-1], x[i]]`` has the
              value ``y[i]``.
            - 'post': The y value is continued constantly to the right from
              every *x* position, i.e. the interval ``[x[i], x[i+1])`` has the
              value ``y[i]``.
            - 'mid': Steps occur half-way between the *x* positions.

        data : indexable object, optional
            An object with labelled data. If given, provide the label names to
            plot in *x* and *y*.

        **kwargs
            Additional parameters are the same as those for `.plot`.

        Returns
        -------
        list of `.Line2D`
            Objects representing the plotted data.
        """
        # 检查 'where' 参数是否有效
        _api.check_in_list(('pre', 'post', 'mid'), where=where)
        # 设置绘图参数，使用步阶图的绘制风格
        kwargs['drawstyle'] = 'steps-' + where
        # 调用实例的 plot 方法进行绘图，并返回绘制的对象列表
        return self.plot(x, y, *args, data=data, **kwargs)
    def _convert_dx(dx, x0, xconv, convert):
        """
        Small helper to do logic of width conversion flexibly.

        *dx* and *x0* have units, but *xconv* has already been converted
        to unitless (and is an ndarray).  This allows the *dx* to have units
        that are different from *x0*, but are still accepted by the
        ``__add__`` operator of *x0*.
        """

        # x should be an array...
        assert type(xconv) is np.ndarray

        if xconv.size == 0:
            # If xconv is empty, return dx converted to the desired units.
            return convert(dx)

        try:
            # Attempt to add the width to x0; this works for
            # datetime+timedelta, for instance

            # Ensure x0 is a single value, handling cases where x0 is a list or array.
            try:
                x0 = cbook._safe_first_finite(x0)
            except (TypeError, IndexError, KeyError):
                pass

            # Ensure x is a single value, handling cases where xconv is a list or array.
            try:
                x = cbook._safe_first_finite(xconv)
            except (TypeError, IndexError, KeyError):
                x = xconv

            delist = False
            if not np.iterable(dx):
                # If dx is not iterable, convert it to a list for processing.
                dx = [dx]
                delist = True
            # Calculate the difference between converted x0 + ddx and x for each ddx in dx.
            dx = [convert(x0 + ddx) - x for ddx in dx]
            if delist:
                # If dx was originally not iterable, return it back to a single value.
                dx = dx[0]
        except (ValueError, TypeError, AttributeError):
            # If adding fails for any reason, convert dx by itself.
            dx = convert(dx)
        # Return the adjusted dx value.
        return dx

    @_preprocess_data()
    @_docstring.dedent_interpd
    # @_preprocess_data() # let 'bar' do the unpacking..
    @_docstring.dedent_interpd
    @_preprocess_data()
    @_docstring.dedent_interpd
    def broken_barh(self, xranges, yrange, **kwargs):
        """
        Plot a horizontal sequence of rectangles.

        A rectangle is drawn for each element of *xranges*. All rectangles
        have the same vertical position and size defined by *yrange*.

        Parameters
        ----------
        xranges : sequence of tuples (*xmin*, *xwidth*)
            The x-positions and extents of the rectangles. For each tuple
            (*xmin*, *xwidth*) a rectangle is drawn from *xmin* to *xmin* +
            *xwidth*.
        yrange : (*ymin*, *yheight*)
            The y-position and extent for all the rectangles.

        Returns
        -------
        `~.collections.PolyCollection`

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER
        **kwargs : `.PolyCollection` properties

            Each *kwarg* can be either a single argument applying to all
            rectangles, e.g.::

                facecolors='black'

            or a sequence of arguments over which is cycled, e.g.::

                facecolors=('black', 'blue')

            would create interleaving black and blue rectangles.

            Supported keywords:

            %(PolyCollection:kwdoc)s
        """
        # 处理单位信息，获取 x 和 y 的数据
        xdata = cbook._safe_first_finite(xranges) if len(xranges) else None
        ydata = cbook._safe_first_finite(yrange) if len(yrange) else None
        self._process_unit_info(
            [("x", xdata), ("y", ydata)], kwargs, convert=False)

        # 初始化顶点列表
        vertices = []
        # 解析 yrange 参数，获取起始位置和高度
        y0, dy = yrange
        # 将起始位置和结束位置转换为正确的 y 坐标
        y0, y1 = self.convert_yunits((y0, y0 + dy))
        # 遍历 xranges 参数，生成每个矩形的顶点坐标
        for xr in xranges:  # 转换绝对值，而非 x 和 dx
            try:
                x0, dx = xr
            except Exception:
                # 如果 xranges 中的元素不符合预期，抛出 ValueError
                raise ValueError(
                    "each range in xrange must be a sequence with two "
                    "elements (i.e. xrange must be an (N, 2) array)") from None
            # 将 x0 和 x1 转换为正确的 x 坐标
            x0, x1 = self.convert_xunits((x0, x0 + dx))
            # 将矩形的四个顶点坐标添加到顶点列表中
            vertices.append([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])

        # 创建 PolyCollection 对象，使用顶点列表和其他参数
        col = mcoll.PolyCollection(np.array(vertices), **kwargs)
        # 将 PolyCollection 对象添加到图形中，自动更新视图范围
        self.add_collection(col, autolim=True)
        # 请求自动调整视图范围
        self._request_autoscale_view()

        # 返回创建的 PolyCollection 对象
        return col

    @_preprocess_data()
    @_api.make_keyword_only("3.9", "explode")
    @_preprocess_data(replace_names=["x", "explode", "labels", "colors"])
    @staticmethod
    # 定义一个静态方法，用于将 errorbar 的 errorevery 参数规范化为数据 x 的布尔掩码
    def _errorevery_to_mask(x, errorevery):
        """
        Normalize `errorbar`'s *errorevery* to be a boolean mask for data *x*.

        This function is split out to be usable both by 2D and 3D errorbars.
        """
        # 如果 errorevery 是整数，转换为元组 (0, errorevery)
        if isinstance(errorevery, Integral):
            errorevery = (0, errorevery)
        # 如果 errorevery 是元组
        if isinstance(errorevery, tuple):
            # 如果元组长度为 2，且两个元素都是整数，则转换为切片对象
            if (len(errorevery) == 2 and
                    isinstance(errorevery[0], Integral) and
                    isinstance(errorevery[1], Integral)):
                errorevery = slice(errorevery[0], None, errorevery[1])
            else:
                # 抛出值错误，提示错误信息
                raise ValueError(
                    f'{errorevery=!r} is a not a tuple of two integers')
        # 如果 errorevery 是切片对象，则直接通过
        elif isinstance(errorevery, slice):
            pass
        # 如果 errorevery 既不是字符串也不是 NumPy 迭代对象，但是是可迭代的
        elif not isinstance(errorevery, str) and np.iterable(errorevery):
            try:
                x[errorevery]  # 使用 fancy indexing
            except (ValueError, IndexError) as err:
                raise ValueError(
                    f"{errorevery=!r} is iterable but not a valid NumPy fancy "
                    "index to match 'xerr'/'yerr'") from err
        else:
            # 抛出值错误，提示错误信息
            raise ValueError(f"{errorevery=!r} is not a recognized value")
        # 创建一个长度与 x 相同的布尔数组，初始化为 False
        everymask = np.zeros(len(x), bool)
        # 将 errorevery 对应的位置设置为 True
        everymask[errorevery] = True
        # 返回布尔掩码数组
        return everymask
    def arrow(self, x, y, dx, dy, **kwargs):
        """
        Add an arrow to the Axes.

        This draws an arrow from ``(x, y)`` to ``(x+dx, y+dy)``.

        Parameters
        ----------
        %(FancyArrow)s

        Returns
        -------
        `.FancyArrow`
            The created `.FancyArrow` object.

        Notes
        -----
        The resulting arrow is affected by the Axes aspect ratio and limits.
        This may produce an arrow whose head is not square with its stem. To
        create an arrow whose head is square with its stem,
        use :meth:`annotate` for example:

        >>> ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),
        ...             arrowprops=dict(arrowstyle="->"))

        """
        # Strip away units for the underlying patch since units
        # do not make sense to most patch-like code
        x = self.convert_xunits(x)  # Convert x coordinate to the appropriate unit
        y = self.convert_yunits(y)  # Convert y coordinate to the appropriate unit
        dx = self.convert_xunits(dx)  # Convert dx (change in x) to the appropriate unit
        dy = self.convert_yunits(dy)  # Convert dy (change in y) to the appropriate unit

        a = mpatches.FancyArrow(x, y, dx, dy, **kwargs)  # Create a FancyArrow object with given coordinates and options
        self.add_patch(a)  # Add the arrow to the Axes
        self._request_autoscale_view()  # Request autoscaling of the view
        return a  # Return the created FancyArrow object

    @_docstring.copy(mquiver.QuiverKey.__init__)
    def quiverkey(self, Q, X, Y, U, label, **kwargs):
        qk = mquiver.QuiverKey(Q, X, Y, U, label, **kwargs)  # Create a QuiverKey object
        self.add_artist(qk)  # Add the QuiverKey object to the Axes
        return qk  # Return the created QuiverKey object

    # Handle units for x and y, if they've been passed
    def _quiver_units(self, args, kwargs):
        if len(args) > 3:
            x, y = args[0:2]
            x, y = self._process_unit_info([("x", x), ("y", y)], kwargs)  # Process unit information for x and y
            return (x, y) + args[2:]
        return args

    # args can be a combination of X, Y, U, V, C and all should be replaced
    @_preprocess_data()
    @_docstring.dedent_interpd
    def quiver(self, *args, **kwargs):
        """%(quiver_doc)s"""
        # Make sure units are handled for x and y values
        args = self._quiver_units(args, kwargs)  # Process unit information for arguments
        q = mquiver.Quiver(self, *args, **kwargs)  # Create a Quiver object
        self.add_collection(q, autolim=True)  # Add the Quiver object to the Axes, with automatic limits
        self._request_autoscale_view()  # Request autoscaling of the view
        return q  # Return the created Quiver object

    # args can be some combination of X, Y, U, V, C and all should be replaced
    @_preprocess_data()
    @_docstring.dedent_interpd
    def barbs(self, *args, **kwargs):
        """%(barbs_doc)s"""
        # Make sure units are handled for x and y values
        args = self._quiver_units(args, kwargs)  # Process unit information for arguments
        b = mquiver.Barbs(self, *args, **kwargs)  # Create a Barbs object
        self.add_collection(b, autolim=True)  # Add the Barbs object to the Axes, with automatic limits
        self._request_autoscale_view()  # Request autoscaling of the view
        return b  # Return the created Barbs object

    # Uses a custom implementation of data-kwarg handling in
    # _process_plot_var_args.
    def fill(self, *args, data=None, **kwargs):
        """
        Plot filled polygons.

        Parameters
        ----------
        *args : sequence of x, y, [color]
            Each polygon is defined by the lists of *x* and *y* positions of
            its nodes, optionally followed by a *color* specifier. See
            :mod:`matplotlib.colors` for supported color specifiers. The
            standard color cycle is used for polygons without a color
            specifier.

            You can plot multiple polygons by providing multiple *x*, *y*,
            *[color]* groups.

            For example, each of the following is legal::

                ax.fill(x, y)                    # a polygon with default color
                ax.fill(x, y, "b")               # a blue polygon
                ax.fill(x, y, x2, y2)            # two polygons
                ax.fill(x, y, "b", x2, y2, "r")  # a blue and a red polygon

        data : indexable object, optional
            An object with labelled data. If given, provide the label names to
            plot in *x* and *y*, e.g.::

                ax.fill("time", "signal",
                        data={"time": [0, 1, 2], "signal": [0, 1, 0]})

        Returns
        -------
        list of `~matplotlib.patches.Polygon`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.patches.Polygon` properties

        Notes
        -----
        Use :meth:`fill_between` if you would like to fill the region between
        two curves.
        """
        # For compatibility(!), get aliases from Line2D rather than Patch.
        # 从 Line2D 而不是 Patch 中获取别名，以保持兼容性
        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        # _get_patches_for_fill returns a generator, convert it to a list.
        # _get_patches_for_fill 返回一个生成器，将其转换为列表
        patches = [*self._get_patches_for_fill(self, *args, data=data, **kwargs)]
        for poly in patches:
            # 将每个多边形添加到当前图形中
            self.add_patch(poly)
        # 请求自动调整视图
        self._request_autoscale_view()
        # 返回生成的多边形列表
        return patches

    def fill_between(self, x, y1, y2=0, where=None, interpolate=False,
                     step=None, **kwargs):
        """
        Fill the area between two curves.

        Parameters
        ----------
        x : array-like
            The x coordinates of the nodes defining the curves.
        y1 : array-like
            The y coordinates of the nodes defining the first curve.
        y2 : array-like or float, optional
            The y coordinates of the nodes defining the second curve.
            If not provided, defaults to 0.
        where : array-like, optional
            Define where to exclude some x, y pairs from filling.
        interpolate : bool, default: False
            Whether to interpolate between the curves.
        step : {'pre', 'post', 'mid'}, optional
            Specify where in relation to x to interpolate.
        **kwargs : `~matplotlib.patches.Polygon` properties

        Returns
        -------
        `~matplotlib.collections.PatchCollection`

        Notes
        -----
        For best results, use with continuous data where `x`, `y1`, and `y2`
        are arrays of the same length.
        """
        return self._fill_between_x_or_y(
            "x", x, y1, y2,
            where=where, interpolate=interpolate, step=step, **kwargs)

    if _fill_between_x_or_y.__doc__:
        # 如果 _fill_between_x_or_y 有文档字符串，则格式化填充文档字符串
        fill_between.__doc__ = _fill_between_x_or_y.__doc__.format(
            dir="horizontal", ind="x", dep="y"
        )

    def fill_betweenx(self, y, x1, x2=0, where=None,
                      step=None, interpolate=False, **kwargs):
        """
        Fill the area between two horizontal curves.

        Parameters
        ----------
        y : array-like
            The y coordinates of the nodes defining the horizontal curves.
        x1 : array-like
            The x coordinates of the nodes defining the first curve.
        x2 : array-like or float, optional
            The x coordinates of the nodes defining the second curve.
            If not provided, defaults to 0.
        where : array-like, optional
            Define where to exclude some y, x pairs from filling.
        step : {'pre', 'post', 'mid'}, optional
            Specify where in relation to y to interpolate.
        interpolate : bool, default: False
            Whether to interpolate between the curves.
        **kwargs : `~matplotlib.patches.Polygon` properties

        Returns
        -------
        `~matplotlib.collections.PatchCollection`

        Notes
        -----
        For best results, use with continuous data where `y`, `x1`, and `x2`
        are arrays of the same length.
        """
        return self._fill_between_x_or_y(
            "y", y, x1, x2,
            where=where, interpolate=interpolate, step=step, **kwargs)

    if _fill_between_x_or_y.__doc__:
        # 如果 _fill_between_x_or_y 有文档字符串，则格式化填充文档字符串
        fill_betweenx.__doc__ = _fill_between_x_or_y.__doc__.format(
            dir="vertical", ind="y", dep="x"
        )
    fill_betweenx = _preprocess_data(
        _docstring.dedent_interpd(fill_betweenx),
        replace_names=["y", "x1", "x2", "where"])


# 预处理数据，使用插值后的填充函数说明文档，替换名称为["y", "x1", "x2", "where"]
fill_betweenx = _preprocess_data(
    _docstring.dedent_interpd(fill_betweenx),
    replace_names=["y", "x1", "x2", "where"])



    #### plotting z(x, y): imshow, pcolor and relatives, contour


# 绘制 z(x, y) 的图像：imshow, pcolor 以及相关函数，包括 contour



    @_preprocess_data()
    @_docstring.interpd
    @_preprocess_data()
    @_docstring.dedent_interpd
    @_preprocess_data()
    @_docstring.dedent_interpd
    @_preprocess_data()
    @_docstring.dedent_interpd
    @_preprocess_data()
    @_docstring.dedent_interpd


# 一系列预处理数据的装饰器，插值和去除缩进后的函数说明文档



    def contour(self, *args, **kwargs):
        """
        Plot contour lines.

        Call signature::

            contour([X, Y,] Z, /, [levels], **kwargs)

        The arguments *X*, *Y*, *Z* are positional-only.
        %(contour_doc)s
        """
        kwargs['filled'] = False
        contours = mcontour.QuadContourSet(self, *args, **kwargs)
        self._request_autoscale_view()
        return contours


# 绘制等高线。
# 使用 QuadContourSet 类在当前对象上绘制等高线，设置 filled 参数为 False，表示绘制线条而非填充区域。
# 调用 _request_autoscale_view() 方法请求自动缩放视图。



    def contourf(self, *args, **kwargs):
        """
        Plot filled contours.

        Call signature::

            contourf([X, Y,] Z, /, [levels], **kwargs)

        The arguments *X*, *Y*, *Z* are positional-only.
        %(contour_doc)s
        """
        kwargs['filled'] = True
        contours = mcontour.QuadContourSet(self, *args, **kwargs)
        self._request_autoscale_view()
        return contours


# 绘制填充的等高线。
# 使用 QuadContourSet 类在当前对象上绘制填充的等高线，设置 filled 参数为 True，表示绘制填充区域。
# 调用 _request_autoscale_view() 方法请求自动缩放视图。



    def clabel(self, CS, levels=None, **kwargs):
        """
        Label a contour plot.

        Adds labels to line contours in given `.ContourSet`.

        Parameters
        ----------
        CS : `.ContourSet` instance
            Line contours to label.

        levels : array-like, optional
            A list of level values, that should be labeled. The list must be
            a subset of ``CS.levels``. If not given, all levels are labeled.

        **kwargs
            All other parameters are documented in `~.ContourLabeler.clabel`.
        """
        return CS.clabel(levels, **kwargs)


# 为等高线图添加标签。
# 将标签添加到给定的 `.ContourSet` 实例中的线条等高线上。
# CS 参数是 `.ContourSet` 实例，表示要添加标签的线条等高线。
# levels 是一个可选的数组，包含应该添加标签的等级值。如果未提供，则对所有等级添加标签。
# **kwargs 包含 `~.ContourLabeler.clabel` 中的所有其他参数文档。



    @_api.make_keyword_only("3.9", "range")
    @_preprocess_data(replace_names=["x", 'weights'], label_namer="x")
    @_preprocess_data()
    @_api.make_keyword_only("3.9", "range")
    @_preprocess_data(replace_names=["x", "y", "weights"])
    @_docstring.dedent_interpd
    @_preprocess_data(replace_names=["x", "weights"], label_namer="x")
    @_docstring.dedent_interpd
    @_api.make_keyword_only("3.9", "NFFT")
    @_preprocess_data(replace_names=["x"])
    @_docstring.dedent_interpd
    @_api.make_keyword_only("3.9", "NFFT")
    @_preprocess_data(replace_names=["x", "y"], label_namer="y")
    @_docstring.dedent_interpd
    @_api.make_keyword_only("3.9", "Fs")
    @_preprocess_data(replace_names=["x"])
    @_docstring.dedent_interpd


# 一系列用于数据分析的装饰器，包括对参数进行处理、替换名称、添加文档说明
    # 定义一个方法用于绘制频谱的幅度谱
    def magnitude_spectrum(self, x, Fs=None, Fc=None, window=None,
                           pad_to=None, sides=None, scale=None,
                           **kwargs):
        """
        绘制幅度谱。

        计算 *x* 的幅度谱。数据会填充到 *pad_to* 指定的长度，并且会对信号应用窗口函数 *window*。

        Parameters
        ----------
        x : 1-D array or sequence
            包含数据的数组或序列。

        %(Spectral)s

        %(Single_Spectrum)s

        scale : {'default', 'linear', 'dB'}
            *spec* 中数值的缩放方式。'linear' 表示无缩放，'dB' 表示返回以分贝为单位的幅度（20 * log10）。'default' 是 'linear'。

        Fc : int, 默认值: 0
            *x* 的中心频率，用于调整频率范围以反映信号采集后进行滤波和降采样到基带时使用的频率范围。

        Returns
        -------
        spectrum : 1-D array
            在缩放之前的幅度谱数值（实数）。

        freqs : 1-D array
            对应于 *spectrum* 元素的频率。

        line : `~matplotlib.lines.Line2D`
            由此函数创建的线条对象。

        Other Parameters
        ----------------
        data : 可索引对象, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            关键字参数控制 `.Line2D` 属性：

            %(Line2D:kwdoc)s

        See Also
        --------
        psd
            绘制功率谱密度图。
        angle_spectrum
            绘制对应频率的角度。
        phase_spectrum
            绘制对应频率的相位（解包裹角度）。
        specgram
            可以绘制信号段的幅度谱，以色彩图方式展示。
        """
        # 如果 Fc 为 None，则设置为 0
        if Fc is None:
            Fc = 0

        # 调用 mlab.magnitude_spectrum 计算幅度谱和对应的频率
        spec, freqs = mlab.magnitude_spectrum(x=x, Fs=Fs, window=window,
                                              pad_to=pad_to, sides=sides)
        # 将频率偏移 Fc
        freqs += Fc

        # 检查并获取 y 轴单位，根据 scale 参数决定返回能量或者分贝单位的幅度谱
        yunits = _api.check_getitem(
            {None: 'energy', 'default': 'energy', 'linear': 'energy',
             'dB': 'dB'},
            scale=scale)
        if yunits == 'energy':
            Z = spec
        else:  # yunits == 'dB'
            Z = 20. * np.log10(spec)

        # 调用 self.plot 方法绘制频谱线条，并传入其他关键字参数
        line, = self.plot(freqs, Z, **kwargs)
        # 设置 x 轴标签为 'Frequency'，y 轴标签为 'Magnitude (%s)' % yunits
        self.set_xlabel('Frequency')
        self.set_ylabel('Magnitude (%s)' % yunits)

        # 返回计算得到的幅度谱、频率数组和绘制的线条对象
        return spec, freqs, line

    # 将 "Fs" 参数标记为仅限关键字参数，要求 Python 版本为 3.9+
    @_api.make_keyword_only("3.9", "Fs")
    # 应用预处理数据的装饰器，替换数据参数为 "x"
    @_preprocess_data(replace_names=["x"])
    # 对文档字符串进行缩进处理和插值
    @_docstring.dedent_interpd
    def angle_spectrum(self, x, Fs=None, Fc=None, window=None,
                       pad_to=None, sides=None, **kwargs):
        """
        Plot the angle spectrum.

        Compute the angle spectrum (wrapped phase spectrum) of *x*.
        Data is padded to a length of *pad_to* and the windowing function
        *window* is applied to the signal.

        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data.

        %(Spectral)s
            Additional spectral parameters passed to `mlab.angle_spectrum`.

        %(Single_Spectrum)s
            Additional single spectrum parameters passed to `mlab.angle_spectrum`.

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        Returns
        -------
        spectrum : 1-D array
            The values for the angle spectrum in radians (real valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *spectrum*.

        line : `~matplotlib.lines.Line2D`
            The line created by this function.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s

        See Also
        --------
        magnitude_spectrum
            Plots the magnitudes of the corresponding frequencies.
        phase_spectrum
            Plots the unwrapped version of this function.
        specgram
            Can plot the angle spectrum of segments within the signal in a
            colormap.
        """
        # 如果未指定 Fc，则设为默认值 0
        if Fc is None:
            Fc = 0

        # 调用 mlab 库的 angle_spectrum 函数计算角谱和对应频率
        spec, freqs = mlab.angle_spectrum(x=x, Fs=Fs, window=window,
                                          pad_to=pad_to, sides=sides)

        # 将频率偏移 Fc，以反映信号获取、滤波和下采样到基带时使用的频率范围
        freqs += Fc

        # 调用当前对象的 plot 方法绘制频率和角谱的关系图，使用传入的 kwargs 控制线条属性
        lines = self.plot(freqs, spec, **kwargs)

        # 设置 x 轴标签为 'Frequency'，y 轴标签为 'Angle (radians)'
        self.set_xlabel('Frequency')
        self.set_ylabel('Angle (radians)')

        # 返回计算得到的角谱、频率数组以及绘制的线条对象的第一个元素（一般是 matplotlib 的 Line2D 对象）
        return spec, freqs, lines[0]

    @_api.make_keyword_only("3.9", "Fs")
    @_preprocess_data(replace_names=["x"])
    @_docstring.dedent_interpd
    # 定义一个方法用于绘制相位谱
    def phase_spectrum(self, x, Fs=None, Fc=None, window=None,
                       pad_to=None, sides=None, **kwargs):
        """
        Plot the phase spectrum.

        Compute the phase spectrum (unwrapped angle spectrum) of *x*.
        Data is padded to a length of *pad_to* and the windowing function
        *window* is applied to the signal.

        Parameters
        ----------
        x : 1-D array or sequence
            Array or sequence containing the data

        %(Spectral)s
            Spectral相关参数的说明，这里会从外部文档插入具体内容。

        %(Single_Spectrum)s
            Single_Spectrum相关参数的说明，这里会从外部文档插入具体内容。

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        Returns
        -------
        spectrum : 1-D array
            The values for the phase spectrum in radians (real valued).

        freqs : 1-D array
            The frequencies corresponding to the elements in *spectrum*.

        line : `~matplotlib.lines.Line2D`
            The line created by this function.

        Other Parameters
        ----------------
        data : indexable object, optional
            Placeholder for data parameter.

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s
                Line2D的关键字参数说明，这里会从外部文档插入具体内容。

        See Also
        --------
        magnitude_spectrum
            Plots the magnitudes of the corresponding frequencies.
        angle_spectrum
            Plots the wrapped version of this function.
        specgram
            Can plot the phase spectrum of segments within the signal in a
            colormap.
        """
        # 如果未指定 Fc，则默认为 0
        if Fc is None:
            Fc = 0

        # 调用 mlab 库中的 phase_spectrum 函数计算相位谱和对应的频率
        spec, freqs = mlab.phase_spectrum(x=x, Fs=Fs, window=window,
                                          pad_to=pad_to, sides=sides)
        # 将频率加上中心频率 Fc，用于绘制时考虑信号获取、滤波和下采样后的频率范围
        freqs += Fc

        # 调用当前对象的 plot 方法绘制相位谱图，并传入额外的参数 kwargs
        lines = self.plot(freqs, spec, **kwargs)
        # 设置 x 轴的标签为 'Frequency'
        self.set_xlabel('Frequency')
        # 设置 y 轴的标签为 'Phase (radians)'
        self.set_ylabel('Phase (radians)')

        # 返回计算得到的相位谱 spec、频率 freqs 和绘制的线条对象的第一个元素
        return spec, freqs, lines[0]
    
    # 将下面的装饰器应用于 phase_spectrum 方法，将使得 NFFT 参数成为关键字参数
    @_api.make_keyword_only("3.9", "NFFT")
    # 将下面的装饰器应用于 phase_spectrum 方法，预处理数据（替换名称为 "x" 和 "y"）
    @_preprocess_data(replace_names=["x", "y"])
    # 将下面的装饰器应用于 phase_spectrum 方法，用于格式化和插入文档字符串
    @_docstring.dedent_interpd
    def cohere(self, x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none,
               window=mlab.window_hanning, noverlap=0, pad_to=None,
               sides='default', scale_by_freq=None, **kwargs):
        r"""
        Plot the coherence between *x* and *y*.

        Coherence is the normalized cross spectral density:

        .. math::

          C_{xy} = \frac{|P_{xy}|^2}{P_{xx}P_{yy}}

        Parameters
        ----------
        %(Spectral)s
            Spectral parameters (see documentation).

        %(PSD)s
            PSD-related parameters (see documentation).

        noverlap : int, default: 0 (no overlap)
            The number of points of overlap between blocks.

        Fc : int, default: 0
            The center frequency of *x*, which offsets the x extents of the
            plot to reflect the frequency range used when a signal is acquired
            and then filtered and downsampled to baseband.

        Returns
        -------
        Cxy : 1-D array
            The coherence vector.

        freqs : 1-D array
            The frequencies for the elements in *Cxy*.

        Other Parameters
        ----------------
        data : indexable object, optional
            DATA_PARAMETER_PLACEHOLDER

        **kwargs
            Keyword arguments control the `.Line2D` properties:

            %(Line2D:kwdoc)s
                Documentation for keyword arguments for Line2D.

        References
        ----------
        Bendat & Piersol -- Random Data: Analysis and Measurement Procedures,
        John Wiley & Sons (1986)

        """
        # 计算信号 x 和 y 的 coherence 和对应的频率
        cxy, freqs = mlab.cohere(x=x, y=y, NFFT=NFFT, Fs=Fs, detrend=detrend,
                                 window=window, noverlap=noverlap,
                                 scale_by_freq=scale_by_freq, sides=sides,
                                 pad_to=pad_to)
        # 将频率向量加上中心频率 Fc
        freqs += Fc

        # 绘制 coherence 对频率的图像，传入额外的参数 kwargs 控制线条属性
        self.plot(freqs, cxy, **kwargs)
        # 设置 x 轴标签为 'Frequency'
        self.set_xlabel('Frequency')
        # 设置 y 轴标签为 'Coherence'
        self.set_ylabel('Coherence')
        # 打开网格线
        self.grid(True)

        # 返回 coherence 向量和频率向量
        return cxy, freqs

    @_api.make_keyword_only("3.9", "NFFT")
    @_preprocess_data(replace_names=["x"])
    @_docstring.dedent_interpd
    @_api.make_keyword_only("3.9", "precision")
    @_docstring.dedent_interpd
    def matshow(self, Z, **kwargs):
        """
        Plot the values of a 2D matrix or array as color-coded image.

        The matrix will be shown the way it would be printed, with the first
        row at the top.  Row and column numbering is zero-based.

        Parameters
        ----------
        Z : (M, N) array-like
            The matrix to be displayed.

        Returns
        -------
        `~matplotlib.image.AxesImage`

        Other Parameters
        ----------------
        **kwargs : `~matplotlib.axes.Axes.imshow` arguments

        See Also
        --------
        imshow : More general function to plot data on a 2D regular raster.

        Notes
        -----
        This is just a convenience function wrapping `.imshow` to set useful
        defaults for displaying a matrix. In particular:

        - Set ``origin='upper'``.
        - Set ``interpolation='nearest'``.
        - Set ``aspect='equal'``.
        - Ticks are placed to the left and above.
        - Ticks are formatted to show integer indices.

        """
        # 将输入的Z转换为NumPy数组（如果不是的话）
        Z = np.asanyarray(Z)
        # 设置绘图参数，包括默认参数和传入的kwargs
        kw = {'origin': 'upper',
              'interpolation': 'nearest',
              'aspect': 'equal',          # （已经是imshow的默认值）
              **kwargs}
        # 调用imshow方法绘制图像，并返回AxesImage对象
        im = self.imshow(Z, **kw)
        # 设置标题位置为图像的上方
        self.title.set_y(1.05)
        # 设置X轴的刻度在顶部显示
        self.xaxis.tick_top()
        # 设置X轴刻度和标签都显示
        self.xaxis.set_ticks_position('both')
        # 设置X轴主刻度定位器，显示整数刻度
        self.xaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        # 设置Y轴主刻度定位器，显示整数刻度
        self.yaxis.set_major_locator(
            mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        # 返回绘制的图像对象
        return im

    @_api.make_keyword_only("3.9", "vert")
    @_preprocess_data(replace_names=["dataset"])
    @_api.make_keyword_only("3.9", "vert")
    # 以下是完全由其他模块实现的方法。

    # 使用_make_axes_method方法创建table方法
    table = _make_axes_method(mtable.table)

    # 使用_preprocess_data和_make_axes_method方法创建stackplot方法
    # args可以是Y或者y1, y2, ...，并且应该全部替换
    stackplot = _preprocess_data()(_make_axes_method(mstack.stackplot))

    # 使用_preprocess_data和_make_axes_method方法创建streamplot方法
    # replace_names参数用于替换x, y, u, v, start_points
    streamplot = _preprocess_data(
            replace_names=["x", "y", "u", "v", "start_points"])(
        _make_axes_method(mstream.streamplot))

    # 使用_make_axes_method方法创建tricontour方法
    tricontour = _make_axes_method(mtri.tricontour)

    # 使用_make_axes_method方法创建tricontourf方法
    tricontourf = _make_axes_method(mtri.tricontourf)

    # 使用_make_axes_method方法创建tripcolor方法
    tripcolor = _make_axes_method(mtri.tripcolor)

    # 使用_make_axes_method方法创建triplot方法
    triplot = _make_axes_method(mtri.triplot)

    def _get_aspect_ratio(self):
        """
        Convenience method to calculate the aspect ratio of the Axes in
        the display coordinate system.
        """
        # 获取当前Axes对象所在的Figure对象的大小（单位为英寸）
        figure_size = self.get_figure().get_size_inches()
        # 获取当前Axes对象的位置（单位为图像的相对坐标）
        ll, ur = self.get_position() * figure_size
        # 计算Axes对象的宽度和高度
        width, height = ur - ll
        # 返回Axes对象的高宽比例，除以数据比例
        return height / (width * self.get_data_ratio())
```
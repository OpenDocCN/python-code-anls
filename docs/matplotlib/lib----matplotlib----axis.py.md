# `D:\src\scipysrc\matplotlib\lib\matplotlib\axis.py`

```
"""
Classes for the ticks and x- and y-axis.
"""

import datetime
import functools
import logging
from numbers import Real
import warnings

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.scale as mscale
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.units as munits

_log = logging.getLogger(__name__)

GRIDLINE_INTERPOLATION_STEPS = 180

# This list is being used for compatibility with Axes.grid, which
# allows all Line2D kwargs.
_line_inspector = martist.ArtistInspector(mlines.Line2D)
_line_param_names = _line_inspector.get_setters()
_line_param_aliases = [list(d)[0] for d in _line_inspector.aliasd.values()]
_gridline_param_names = ['grid_' + name
                         for name in _line_param_names + _line_param_aliases]

_MARKER_DICT = {
    'out': (mlines.TICKDOWN, mlines.TICKUP),
    'in': (mlines.TICKUP, mlines.TICKDOWN),
    'inout': ('|', '|'),
}


class Tick(martist.Artist):
    """
    Abstract base class for the axis ticks, grid lines and labels.

    Ticks mark a position on an Axis. They contain two lines as markers and
    two labels; one each for the bottom and top positions (in case of an
    `.XAxis`) or for the left and right positions (in case of a `.YAxis`).

    Attributes
    ----------
    tick1line : `~matplotlib.lines.Line2D`
        The left/bottom tick marker.
    tick2line : `~matplotlib.lines.Line2D`
        The right/top tick marker.
    gridline : `~matplotlib.lines.Line2D`
        The grid line associated with the label position.
    label1 : `~matplotlib.text.Text`
        The left/bottom tick label.
    label2 : `~matplotlib.text.Text`
        The right/top tick label.

    """
    def __init__(
        self, axes, loc, *,
        size=None,  # points
        width=None,
        color=None,
        tickdir=None,
        pad=None,
        labelsize=None,
        labelcolor=None,
        labelfontfamily=None,
        zorder=None,
        gridOn=None,  # defaults to axes.grid depending on axes.grid.which
        tick1On=True,
        tick2On=True,
        label1On=True,
        label2On=False,
        major=True,
        labelrotation=0,
        grid_color=None,
        grid_linestyle=None,
        grid_linewidth=None,
        grid_alpha=None,
        **kwargs,  # Other Line2D kwargs applied to gridlines.
    ):
        """
        Initialize a Tick object for an axis.

        Parameters
        ----------
        axes : `matplotlib.axes.Axes`
            The axes that this tick belongs to.
        loc : {'top', 'bottom', 'left', 'right'}
            Location of the tick on the axis.
        size : float, optional
            Size of the tick markers in points.
        width : float, optional
            Width of the tick markers in points.
        color : color, optional
            Color of the ticks.
        tickdir : {'in', 'out', 'inout'}, optional
            Direction of the ticks.
        pad : float, optional
            Padding between tick and label.
        labelsize : float, optional
            Font size of the tick labels.
        labelcolor : color, optional
            Color of the tick labels.
        labelfontfamily : str, optional
            Font family of the tick labels.
        zorder : float, optional
            Drawing order for the ticks.
        gridOn : bool, optional
            Whether to show grid lines associated with this tick.
        tick1On : bool, optional
            Whether to show the first tick marker.
        tick2On : bool, optional
            Whether to show the second tick marker.
        label1On : bool, optional
            Whether to show the first tick label.
        label2On : bool, optional
            Whether to show the second tick label.
        major : bool, optional
            Whether the tick is major or minor.
        labelrotation : float or {'auto', 'default'}, optional
            Rotation angle of the tick labels.
        grid_color : color, optional
            Color of the grid lines.
        grid_linestyle : str, optional
            Line style of the grid lines.
        grid_linewidth : float, optional
            Line width of the grid lines.
        grid_alpha : float, optional
            Transparency of the grid lines.
        **kwargs
            Other parameters passed to `matplotlib.lines.Line2D` for grid lines.
        """
        # Call the constructor of the superclass (martist.Artist)
        super().__init__()

        # Initialize attributes with provided or default values
        self.axes = axes
        self.loc = loc
        self.size = size
        self.width = width
        self.color = color
        self.tickdir = tickdir
        self.pad = pad
        self.labelsize = labelsize
        self.labelcolor = labelcolor
        self.labelfontfamily = labelfontfamily
        self.zorder = zorder
        self.gridOn = gridOn
        self.tick1On = tick1On
        self.tick2On = tick2On
        self.label1On = label1On
        self.label2On = label2On
        self.major = major
        self.labelrotation = labelrotation
        self.grid_color = grid_color
        self.grid_linestyle = grid_linestyle
        self.grid_linewidth = grid_linewidth
        self.grid_alpha = grid_alpha

        # Set label rotation mode and angle
        self._set_labelrotation(labelrotation)

        # Process additional Line2D kwargs for grid lines
        self.gridline_kw = kwargs

    def _set_labelrotation(self, labelrotation):
        """
        Set the label rotation mode and angle.

        Parameters
        ----------
        labelrotation : float or {'auto', 'default'}
            Rotation angle of the tick labels.
        """
        if isinstance(labelrotation, str):
            mode = labelrotation
            angle = 0
        elif isinstance(labelrotation, (tuple, list)):
            mode, angle = labelrotation
        else:
            mode = 'default'
            angle = labelrotation
        
        # Check validity of label rotation mode
        _api.check_in_list(['auto', 'default'], labelrotation=mode)
        
        # Store label rotation parameters
        self._labelrotation = (mode, angle)
    def _apply_tickdir(self, tickdir):
        """
        设置刻度方向。有效值为 'out', 'in', 'inout'。
        """
        # 此方法负责更新 `_pad`，在子类中还负责设置刻度线标记。
        # 从用户角度看，应始终通过 `_apply_params` 调用此方法，后者使用新的 pad 更新刻度标签位置。
        
        # 如果 tickdir 为 None，则使用默认的配置参数中的方向设置
        if tickdir is None:
            tickdir = mpl.rcParams[f'{self.__name__}.direction']
        else:
            # 检查 tickdir 是否在允许的列表中
            _api.check_in_list(['in', 'out', 'inout'], tickdir=tickdir)
        
        # 设置当前对象的刻度方向
        self._tickdir = tickdir
        
        # 计算并设置刻度外的填充长度 `_pad`
        self._pad = self._base_pad + self.get_tick_padding()

    def get_tickdir(self):
        """
        返回刻度方向。
        """
        return self._tickdir

    def get_tick_padding(self):
        """
        获取刻度在轴外部的长度。
        """
        # 定义不同方向刻度的外部长度比例
        padding = {
            'in': 0.0,
            'inout': 0.5,
            'out': 1.0
        }
        # 返回当前刻度方向对应的外部长度
        return self._size * padding[self._tickdir]

    def get_children(self):
        """
        返回此对象的子对象列表，包括刻度线、网格线和标签等。
        """
        children = [self.tick1line, self.tick2line,
                    self.gridline, self.label1, self.label2]
        return children

    @_api.rename_parameter("3.8", "clippath", "path")
    def set_clip_path(self, path, transform=None):
        """
        设置裁剪路径。

        Parameters
        ----------
        path : Path
            裁剪路径对象。
        transform : Transform, optional
            可选的变换对象，默认为 None。
        """
        # 调用父类方法设置裁剪路径
        super().set_clip_path(path, transform)
        
        # 设置网格线对象的裁剪路径
        self.gridline.set_clip_path(path, transform)
        
        # 将对象标记为过时的
        self.stale = True

    def contains(self, mouseevent):
        """
        测试鼠标事件是否发生在刻度标记上。

        返回值
        -------
        bool
            始终返回 False。
        dict
            空字典。
        """
        return False, {}

    def set_pad(self, val):
        """
        设置刻度标签的填充值（单位：点）。

        Parameters
        ----------
        val : float
            填充值。
        """
        # 调用 _apply_params 方法设置填充值
        self._apply_params(pad=val)
        
        # 将对象标记为过时的
        self.stale = True

    def get_pad(self):
        """
        获取刻度标签的填充值（单位：点）。
        """
        return self._base_pad

    def get_loc(self):
        """
        返回刻度位置（数据坐标）的标量值。
        """
        return self._loc

    @martist.allow_rasterization
    def draw(self, renderer):
        """
        绘制刻度对象及其子对象。

        Parameters
        ----------
        renderer : RendererBase
            渲染器对象。
        """
        if not self.get_visible():
            self.stale = False
            return
        
        # 开始绘制组
        renderer.open_group(self.__name__, gid=self.get_gid())
        
        # 依次绘制每个子对象
        for artist in [self.gridline, self.tick1line, self.tick2line,
                       self.label1, self.label2]:
            artist.draw(renderer)
        
        # 关闭绘制组
        renderer.close_group(self.__name__)
        
        # 将对象标记为不过时的
        self.stale = False

    @_api.deprecated("3.8")
    def set_label1(self, s):
        """
        设置 label1 的文本内容。

        Parameters
        ----------
        s : str
            文本内容。
        """
        # 设置 label1 的文本内容
        self.label1.set_text(s)
        
        # 将对象标记为过时的
        self.stale = True

    set_label = set_label1

    @_api.deprecated("3.8")
    # 设置 label2 的文本内容
    def set_label2(self, s):
        """
        Set the label2 text.

        Parameters
        ----------
        s : str
            要设置的文本内容
        """
        # 调用 label2 对象的方法设置文本内容
        self.label2.set_text(s)
        # 将 stale 属性设置为 True，表示数据已过时
        self.stale = True

    # 设置 label1 和 label2 的 URL
    def set_url(self, url):
        """
        Set the url of label1 and label2.

        Parameters
        ----------
        url : str
            要设置的 URL 地址
        """
        # 调用父类的 set_url 方法设置 URL
        super().set_url(url)
        # 设置 label1 的 URL
        self.label1.set_url(url)
        # 设置 label2 的 URL
        self.label2.set_url(url)
        # 将 stale 属性设置为 True，表示数据已过时
        self.stale = True

    # 设置 Artist 对象的 figure 属性
    def _set_artist_props(self, a):
        # 设置 Artist 对象的 figure 属性为当前对象的 figure
        a.set_figure(self.figure)

    # 获取当前轴的视图区间的限制 (min, max)
    def get_view_interval(self):
        """
        Return the view limits ``(min, max)`` of the axis the tick belongs to.
        """
        # 抛出 NotImplementedError，要求派生类必须重写该方法
        raise NotImplementedError('Derived must override')
    # 使用关键字参数更新图形元素的显示状态，如网格线和刻度线
    def _apply_params(self, **kwargs):
        # 遍历需要更新的属性及其对应的目标对象
        for name, target in [("gridOn", self.gridline),
                             ("tick1On", self.tick1line),
                             ("tick2On", self.tick2line),
                             ("label1On", self.label1),
                             ("label2On", self.label2)]:
            # 如果关键字存在于kwargs中，则设置目标对象的可见性，并将其从kwargs中移除
            if name in kwargs:
                target.set_visible(kwargs.pop(name))
        
        # 如果kwargs中包含'size', 'width', 'pad', 'tickdir'中的任何一个关键字
        if any(k in kwargs for k in ['size', 'width', 'pad', 'tickdir']):
            # 设置对象内部属性为kwargs中对应的值，若不存在则使用默认值
            self._size = kwargs.pop('size', self._size)
            self._width = kwargs.pop('width', self._width)
            self._base_pad = kwargs.pop('pad', self._base_pad)
            
            # 根据'tickdir'设置对象的方向属性，并更新刻度线的标记
            self._apply_tickdir(kwargs.pop('tickdir', self._tickdir))
            for line in (self.tick1line, self.tick2line):
                line.set_markersize(self._size)
                line.set_markeredgewidth(self._width)
            
            # 根据_apply_tickdir返回的结果设置标签的变换属性
            trans = self._get_text1_transform()[0]
            self.label1.set_transform(trans)
            trans = self._get_text2_transform()[0]
            self.label2.set_transform(trans)
        
        # 从kwargs中筛选出关于刻度线颜色和顺序的关键字参数
        tick_kw = {k: v for k, v in kwargs.items() if k in ['color', 'zorder']}
        if 'color' in kwargs:
            tick_kw['markeredgecolor'] = kwargs['color']
        
        # 更新tick1line和tick2line的属性
        self.tick1line.set(**tick_kw)
        self.tick2line.set(**tick_kw)
        
        # 将相关属性设置到self对象中
        for k, v in tick_kw.items():
            setattr(self, '_' + k, v)
        
        # 如果kwargs中包含'labelrotation'关键字，则设置标签的旋转角度
        if 'labelrotation' in kwargs:
            self._set_labelrotation(kwargs.pop('labelrotation'))
            self.label1.set(rotation=self._labelrotation[1])
            self.label2.set(rotation=self._labelrotation[1])
        
        # 从kwargs中筛选出与标签相关的关键字参数，并更新label1和label2的属性
        label_kw = {k[5:]: v for k, v in kwargs.items()
                    if k in ['labelsize', 'labelcolor', 'labelfontfamily']}
        self.label1.set(**label_kw)
        self.label2.set(**label_kw)
        
        # 从kwargs中筛选出与网格线相关的关键字参数，并更新gridline的属性
        grid_kw = {k[5:]: v for k, v in kwargs.items()
                   if k in _gridline_param_names}
        self.gridline.set(**grid_kw)

    # 更新tick的数据坐标位置
    def update_position(self, loc):
        """Set the location of tick in data coords with scalar *loc*."""
        raise NotImplementedError('Derived must override')

    # 返回第一个标签的变换属性
    def _get_text1_transform(self):
        raise NotImplementedError('Derived must override')

    # 返回第二个标签的变换属性
    def _get_text2_transform(self):
        raise NotImplementedError('Derived must override')
class XTick(Tick):
    """
    Contains all the Artists needed to make an x tick - the tick line,
    the label text and the grid line
    """
    __name__ = 'xtick'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x in data coords, y in axes coords
        ax = self.axes
        # 设置 tick1line 的数据和坐标系变换
        self.tick1line.set(
            data=([0], [0]), transform=ax.get_xaxis_transform("tick1"))
        # 设置 tick2line 的数据和坐标系变换
        self.tick2line.set(
            data=([0], [1]), transform=ax.get_xaxis_transform("tick2"))
        # 设置 gridline 的数据和坐标系变换
        self.gridline.set(
            data=([0, 0], [0, 1]), transform=ax.get_xaxis_transform("grid"))
        # 根据 y 轴最小值位置下移3个点的位置确定 label1 的文本位置和坐标系变换
        trans, va, ha = self._get_text1_transform()
        self.label1.set(
            x=0, y=0,
            verticalalignment=va, horizontalalignment=ha, transform=trans,
        )
        # 根据 y 轴最小值位置下移3个点的位置确定 label2 的文本位置和坐标系变换
        trans, va, ha = self._get_text2_transform()
        self.label2.set(
            x=0, y=1,
            verticalalignment=va, horizontalalignment=ha, transform=trans,
        )

    def _get_text1_transform(self):
        # 获取第一个文本位置的坐标系变换
        return self.axes.get_xaxis_text1_transform(self._pad)

    def _get_text2_transform(self):
        # 获取第二个文本位置的坐标系变换
        return self.axes.get_xaxis_text2_transform(self._pad)

    def _apply_tickdir(self, tickdir):
        # docstring inherited
        super()._apply_tickdir(tickdir)
        # 根据 tick 方向应用对应的标记点到 tick1line 和 tick2line
        mark1, mark2 = _MARKER_DICT[self._tickdir]
        self.tick1line.set_marker(mark1)
        self.tick2line.set_marker(mark2)

    def update_position(self, loc):
        """Set the location of tick in data coords with scalar *loc*."""
        # 更新 tick 在数据坐标中的位置
        self.tick1line.set_xdata((loc,))
        self.tick2line.set_xdata((loc,))
        self.gridline.set_xdata((loc,))
        self.label1.set_x(loc)
        self.label2.set_x(loc)
        self._loc = loc
        self.stale = True

    def get_view_interval(self):
        # docstring inherited
        # 获取视图间隔的 x 轴区间
        return self.axes.viewLim.intervalx


class YTick(Tick):
    """
    Contains all the Artists needed to make a Y tick - the tick line,
    the label text and the grid line
    """
    __name__ = 'ytick'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x in axes coords, y in data coords
        ax = self.axes
        # 设置 tick1line 的数据和坐标系变换
        self.tick1line.set(
            data=([0], [0]), transform=ax.get_yaxis_transform("tick1"))
        # 设置 tick2line 的数据和坐标系变换
        self.tick2line.set(
            data=([1], [0]), transform=ax.get_yaxis_transform("tick2"))
        # 设置 gridline 的数据和坐标系变换
        self.gridline.set(
            data=([0, 1], [0, 0]), transform=ax.get_yaxis_transform("grid"))
        # 根据 y 轴最小值位置下移3个点的位置确定 label1 的文本位置和坐标系变换
        trans, va, ha = self._get_text1_transform()
        self.label1.set(
            x=0, y=0,
            verticalalignment=va, horizontalalignment=ha, transform=trans,
        )
        # 根据 y 轴最小值位置下移3个点的位置确定 label2 的文本位置和坐标系变换
        trans, va, ha = self._get_text2_transform()
        self.label2.set(
            x=1, y=0,
            verticalalignment=va, horizontalalignment=ha, transform=trans,
        )
    # 获取文本1的变换对象，用于在y轴上放置文本
    def _get_text1_transform(self):
        return self.axes.get_yaxis_text1_transform(self._pad)

    # 获取文本2的变换对象，用于在y轴上放置文本
    def _get_text2_transform(self):
        return self.axes.get_yaxis_text2_transform(self._pad)

    # 应用刻度方向的变化
    def _apply_tickdir(self, tickdir):
        # 调用父类的方法来应用刻度方向的设置
        super()._apply_tickdir(tickdir)
        # 根据刻度方向设置第一和第二刻度线的标记
        mark1, mark2 = {
            'out': (mlines.TICKLEFT, mlines.TICKRIGHT),
            'in': (mlines.TICKRIGHT, mlines.TICKLEFT),
            'inout': ('_', '_'),
        }[self._tickdir]
        self.tick1line.set_marker(mark1)
        self.tick2line.set_marker(mark2)

    # 更新刻度的位置
    def update_position(self, loc):
        """Set the location of tick in data coords with scalar *loc*."""
        # 设置第一和第二刻度线的y数据为loc
        self.tick1line.set_ydata((loc,))
        self.tick2line.set_ydata((loc,))
        # 设置网格线的y数据为loc
        self.gridline.set_ydata((loc,))
        # 设置第一和第二刻度标签的y坐标为loc
        self.label1.set_y(loc)
        self.label2.set_y(loc)
        # 更新对象的_loc属性为loc
        self._loc = loc
        # 设置对象为过时状态，需要重新绘制
        self.stale = True

    # 获取视图间隔（y轴方向）
    def get_view_interval(self):
        # 返回当前坐标轴视图限制对象的y轴间隔
        return self.axes.viewLim.intervaly
class Ticker:
    """
    A container for the objects defining tick position and format.

    Attributes
    ----------
    locator : `~matplotlib.ticker.Locator` subclass
        Determines the positions of the ticks.
    formatter : `~matplotlib.ticker.Formatter` subclass
        Determines the format of the tick labels.
    """

    def __init__(self):
        # 初始化 Ticker 对象的属性
        self._locator = None
        self._formatter = None
        self._locator_is_default = True
        self._formatter_is_default = True

    @property
    def locator(self):
        # 返回 locator 属性的值
        return self._locator

    @locator.setter
    def locator(self, locator):
        # 设置 locator 属性，确保传入的 locator 是 matplotlib.ticker.Locator 的子类
        if not isinstance(locator, mticker.Locator):
            raise TypeError('locator must be a subclass of '
                            'matplotlib.ticker.Locator')
        self._locator = locator

    @property
    def formatter(self):
        # 返回 formatter 属性的值
        return self._formatter

    @formatter.setter
    def formatter(self, formatter):
        # 设置 formatter 属性，确保传入的 formatter 是 matplotlib.ticker.Formatter 的子类
        if not isinstance(formatter, mticker.Formatter):
            raise TypeError('formatter must be a subclass of '
                            'matplotlib.ticker.Formatter')
        self._formatter = formatter


class _LazyTickList:
    """
    A descriptor for lazy instantiation of tick lists.

    See comment above definition of the ``majorTicks`` and ``minorTicks``
    attributes.
    """

    def __init__(self, major):
        # 初始化 _LazyTickList 对象，major 参数指示是主刻度还是次刻度
        self._major = major

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            # 在实例化时通过延迟加载生成刻度列表，以避免不必要的计算开销
            if self._major:
                # 如果是主刻度，则先清空主刻度列表，然后生成刻度并添加到列表中
                instance.majorTicks = []
                tick = instance._get_tick(major=True)
                instance.majorTicks.append(tick)
                return instance.majorTicks
            else:
                # 如果是次刻度，则先清空次刻度列表，然后生成刻度并添加到列表中
                instance.minorTicks = []
                tick = instance._get_tick(major=False)
                instance.minorTicks.append(tick)
                return instance.minorTicks


class Axis(martist.Artist):
    """
    Base class for `.XAxis` and `.YAxis`.

    Attributes
    ----------
    isDefault_label : bool

    axes : `~matplotlib.axes.Axes`
        The `~.axes.Axes` to which the Axis belongs.
    major : `~matplotlib.axis.Ticker`
        Determines the major tick positions and their label format.
    minor : `~matplotlib.axis.Ticker`
        Determines the minor tick positions and their label format.
    callbacks : `~matplotlib.cbook.CallbackRegistry`

    label : `~matplotlib.text.Text`
        The axis label.
    labelpad : float
        The distance between the axis label and the tick labels.
        Defaults to :rc:`axes.labelpad` = 4.
    """
    offsetText : `~matplotlib.text.Text`
        # 一个 `.Text` 对象，包含刻度的数据偏移量（如果有的话）。

    pickradius : float
        # 用于包含测试的接受半径。参见 `.Axis.contains`。

    majorTicks : list of `.Tick`
        # 主刻度列表。
        # 
        # .. warning::
        # 
        #     刻度不能保证持久存在。各种操作可能会创建、删除和修改刻度实例。存在
        #     单个刻度变更不会在进一步操作图形时保持（包括在显示的图形上移动/缩放）的
        #     立即风险。
        # 
        #     操作单个刻度是最后的手段。如果可能，请使用 `.set_tick_params` 替代。

    minorTicks : list of `.Tick`
        # 次要刻度列表。
    """
    OFFSETTEXTPAD = 3
    # `_get_tick()` 中用于创建刻度实例的类。必须在子类中进行覆盖，
    # 或者子类必须重新实现 `_get_tick()`。

    def __str__(self):
        # 返回当前对象的字符串表示形式，包括类名和其在 Axes 中的位置。
        return "{}({},{})".format(
            type(self).__name__, *self.axes.transAxes.transform((0, 0)))

    def __init__(self, axes, *, pickradius=15, clear=True):
        """
        Parameters
        ----------
        axes : `~matplotlib.axes.Axes`
            创建的 Axis 所属的 `~.axes.Axes`。
        pickradius : float
            用于包含测试的接受半径。参见 `.Axis.contains`。
        clear : bool, default: True
            是否在创建时清除 Axis。例如，在作为 Axes 的一部分创建 Axis 时，
            不需要清除，因为 `Axes.clear` 将调用 `Axis.clear`。
            .. versionadded:: 3.8
        """
        super().__init__()

        self._remove_overlapping_locs = True

        self.set_figure(axes.figure)

        self.isDefault_label = True

        self.axes = axes
        self.major = Ticker()
        self.minor = Ticker()
        self.callbacks = cbook.CallbackRegistry(signals=["units"])

        self._autolabelpos = True

        self.label = mtext.Text(
            np.nan, np.nan,
            fontsize=mpl.rcParams['axes.labelsize'],
            fontweight=mpl.rcParams['axes.labelweight'],
            color=mpl.rcParams['axes.labelcolor'],
        )
        self._set_artist_props(self.label)

        self.offsetText = mtext.Text(np.nan, np.nan)
        self._set_artist_props(self.offsetText)

        self.labelpad = mpl.rcParams['axes.labelpad']

        self.pickradius = pickradius

        # 为测试目的在此初始化；稍后添加 API
        self._major_tick_kw = dict()
        self._minor_tick_kw = dict()

        if clear:
            self.clear()
        else:
            self.converter = None
            self.units = None

        self._autoscale_on = True

    @property
    def isDefault_majloc(self):
        # 检查是否使用默认的主刻度定位器。
        return self.major._locator_is_default

    @isDefault_majloc.setter
    # 设置是否使用默认的主要定位器（locator）
    def isDefault_majloc(self, value):
        self.major._locator_is_default = value

    # 获取是否使用默认的主要格式化器（formatter）
    @property
    def isDefault_majfmt(self):
        return self.major._formatter_is_default

    # 设置是否使用默认的主要格式化器（formatter）
    @isDefault_majfmt.setter
    def isDefault_majfmt(self, value):
        self.major._formatter_is_default = value

    # 获取是否使用默认的次要定位器（locator）
    @property
    def isDefault_minloc(self):
        return self.minor._locator_is_default

    # 设置是否使用默认的次要定位器（locator）
    @isDefault_minloc.setter
    def isDefault_minloc(self, value):
        self.minor._locator_is_default = value

    # 获取是否使用默认的次要格式化器（formatter）
    @property
    def isDefault_minfmt(self):
        return self.minor._formatter_is_default

    # 设置是否使用默认的次要格式化器（formatter）
    @isDefault_minfmt.setter
    def isDefault_minfmt(self, value):
        self.minor._formatter_is_default = value

    # 返回当前轴的共享轴（shared Axes）的Grouper
    def _get_shared_axes(self):
        """Return Grouper of shared Axes for current axis."""
        return self.axes._shared_axes[
            self._get_axis_name()].get_siblings(self.axes)

    # 返回当前轴的共享轴（shared Axes）的列表
    def _get_shared_axis(self):
        """Return list of shared axis for current axis."""
        name = self._get_axis_name()
        return [ax._axis_map[name] for ax in self._get_shared_axes()]

    # 返回当前轴的名称
    def _get_axis_name(self):
        """Return the axis name."""
        return [name for name, axis in self.axes._axis_map.items()
                if axis is self][0]

    # 在初始化过程中，轴对象经常创建后来未使用的刻度，这是一个非常慢的步骤。
    # 使用自定义描述符使刻度列表变为惰性加载，根据需要实例化它们，以提升性能。
    majorTicks = _LazyTickList(major=True)
    minorTicks = _LazyTickList(major=False)

    # 获取要移除的重叠位置
    def get_remove_overlapping_locs(self):
        return self._remove_overlapping_locs

    # 设置是否移除重叠位置
    def set_remove_overlapping_locs(self, val):
        self._remove_overlapping_locs = bool(val)

    # 属性：设置是否移除重叠位置，文档说明详细了解此属性的用途
    remove_overlapping_locs = property(
        get_remove_overlapping_locs, set_remove_overlapping_locs,
        doc=('If minor ticker locations that overlap with major '
             'ticker locations should be trimmed.'))

    # 设置标签的坐标位置和坐标系
    def set_label_coords(self, x, y, transform=None):
        """
        Set the coordinates of the label.

        By default, the x coordinate of the y label and the y coordinate of the
        x label are determined by the tick label bounding boxes, but this can
        lead to poor alignment of multiple labels if there are multiple Axes.

        You can also specify the coordinate system of the label with the
        transform.  If None, the default coordinate system will be the axes
        coordinate system: (0, 0) is bottom left, (0.5, 0.5) is center, etc.
        """
        self._autolabelpos = False
        if transform is None:
            transform = self.axes.transAxes

        self.label.set_transform(transform)
        self.label.set_position((x, y))
        self.stale = True

    # 获取轴的变换
    def get_transform(self):
        """Return the transform used in the Axis' scale"""
        return self._scale.get_transform()
    # 返回此轴的比例尺（作为字符串）
    def get_scale(self):
        return self._scale.name

    # 设置此轴的比例尺
    def _set_scale(self, value, **kwargs):
        if not isinstance(value, mscale.ScaleBase):
            # 如果值不是 ScaleBase 类型，则使用 scale_factory 创建对应的比例尺
            self._scale = mscale.scale_factory(value, self, **kwargs)
        else:
            self._scale = value
        # 设置默认的定位器和格式化程序
        self._scale.set_default_locators_and_formatters(self)

        # 标记默认的主要定位器、次要定位器、主要格式化程序和次要格式化程序为真
        self.isDefault_majloc = True
        self.isDefault_minloc = True
        self.isDefault_majfmt = True
        self.isDefault_minfmt = True

    # 此方法被 Axes.set_{x,y}scale 直接包装调用
    def _set_axes_scale(self, value, **kwargs):
        """
        设置此轴的比例尺。

        Parameters
        ----------
        value : {"linear", "log", "symlog", "logit", ...} or `.ScaleBase`
            应用的轴比例尺类型。

        **kwargs
            接受不同的关键字参数，取决于比例尺类型。
            参见各个比例尺类的关键字参数：

            - `matplotlib.scale.LinearScale`
            - `matplotlib.scale.LogScale`
            - `matplotlib.scale.SymmetricalLogScale`
            - `matplotlib.scale.LogitScale`
            - `matplotlib.scale.FuncScale`
            - `matplotlib.scale.AsinhScale`

        Notes
        -----
        默认情况下，Matplotlib 支持上述提到的比例尺。
        此外，可以使用 `matplotlib.scale.register_scale` 注册自定义比例尺。
        这些比例尺也可以在这里使用。
        """
        name = self._get_axis_name()
        # 获取旧的默认限制范围
        old_default_lims = (self.get_major_locator()
                            .nonsingular(-np.inf, np.inf))
        for ax in self._get_shared_axes():
            # 设置共享轴的比例尺
            ax._axis_map[name]._set_scale(value, **kwargs)
            ax._update_transScale()
            ax.stale = True
        # 获取新的默认限制范围
        new_default_lims = (self.get_major_locator()
                            .nonsingular(-np.inf, np.inf))
        # 如果旧的默认限制范围与新的不同，强制现在进行自动缩放
        if old_default_lims != new_default_lims:
            self.axes.autoscale_view(
                **{f"scale{k}": k == name for k in self.axes._axis_names})

    # 限制比例尺范围
    def limit_range_for_scale(self, vmin, vmax):
        return self._scale.limit_range_for_scale(vmin, vmax, self.get_minpos())

    # 返回此轴是否自动缩放
    def _get_autoscale_on(self):
        return self._autoscale_on

    # 设置此轴是否在绘制时或通过 `.Axes.autoscale_view` 自动缩放
    def _set_autoscale_on(self, b):
        """
        如果 b 不是 None，则设置此轴在绘制时或通过 `.Axes.autoscale_view` 自动缩放。

        Parameters
        ----------
        b : bool
        """
        if b is not None:
            self._autoscale_on = b

    # 获取此轴的子元素
    def get_children(self):
        return [self.label, self.offsetText,
                *self.get_major_ticks(), *self.get_minor_ticks()]
    def _reset_major_tick_kw(self):
        # 清空主要刻度的关键字参数字典
        self._major_tick_kw.clear()
        # 根据当前绘图参数设置主要刻度的网格显示属性
        self._major_tick_kw['gridOn'] = (
                mpl.rcParams['axes.grid'] and
                mpl.rcParams['axes.grid.which'] in ('both', 'major'))

    def _reset_minor_tick_kw(self):
        # 清空次要刻度的关键字参数字典
        self._minor_tick_kw.clear()
        # 根据当前绘图参数设置次要刻度的网格显示属性
        self._minor_tick_kw['gridOn'] = (
                mpl.rcParams['axes.grid'] and
                mpl.rcParams['axes.grid.which'] in ('both', 'minor'))

    def clear(self):
        """
        Clear the axis.

        This resets axis properties to their default values:

        - the label
        - the scale
        - locators, formatters and ticks
        - major and minor grid
        - units
        - registered callbacks
        """
        # 重置标签的视觉默认设置
        self.label._reset_visual_defaults()
        # 使用文本绘图参数重新设置标签的格式
        # 然后使用坐标轴绘图参数更新格式化
        self.label.set_color(mpl.rcParams['axes.labelcolor'])
        self.label.set_fontsize(mpl.rcParams['axes.labelsize'])
        self.label.set_fontweight(mpl.rcParams['axes.labelweight'])
        # 重置偏移文本的视觉默认设置
        self.offsetText._reset_visual_defaults()
        # 使用绘图参数设置标签间隔
        self.labelpad = mpl.rcParams['axes.labelpad']

        # 初始化轴
        self._init()

        # 设置比例为线性
        self._set_scale('linear')

        # 清除此轴的回调注册表，防止"泄漏"
        self.callbacks = cbook.CallbackRegistry(signals=["units"])

        # 设置主要刻度的网格显示属性
        self._major_tick_kw['gridOn'] = (
                mpl.rcParams['axes.grid'] and
                mpl.rcParams['axes.grid.which'] in ('both', 'major'))
        # 设置次要刻度的网格显示属性
        self._minor_tick_kw['gridOn'] = (
                mpl.rcParams['axes.grid'] and
                mpl.rcParams['axes.grid.which'] in ('both', 'minor'))
        # 重置刻度
        self.reset_ticks()

        # 清空单位转换器和单位
        self.converter = None
        self.units = None
        # 设置为需要刷新
        self.stale = True

    def reset_ticks(self):
        """
        Re-initialize the major and minor Tick lists.

        Each list starts with a single fresh Tick.
        """
        # 恢复主要刻度和次要刻度的延迟刻度列表
        try:
            del self.majorTicks
        except AttributeError:
            pass
        try:
            del self.minorTicks
        except AttributeError:
            pass
        # 尝试设置裁剪路径为坐标轴的补丁路径
        try:
            self.set_clip_path(self.axes.patch)
        except AttributeError:
            pass
    def minorticks_on(self):
        """
        在轴上显示默认的次要刻度，取决于轴的比例尺 (`~.axis.Axis.get_scale`).

        各种比例尺使用特定的次要定位器:

        - log: `~.LogLocator`
        - symlog: `~.SymmetricalLogLocator`
        - asinh: `~.AsinhLocator`
        - logit: `~.LogitLocator`
        - 默认: `~.AutoMinorLocator`

        显示次要刻度可能会降低性能；如果绘图速度有问题，可以使用 `minorticks_off()` 关闭它们。
        """
        # 获取当前轴的比例尺类型
        scale = self.get_scale()
        # 根据不同的比例尺类型设置相应的次要定位器
        if scale == 'log':
            s = self._scale
            self.set_minor_locator(mticker.LogLocator(s.base, s.subs))
        elif scale == 'symlog':
            s = self._scale
            self.set_minor_locator(
                mticker.SymmetricalLogLocator(s._transform, s.subs))
        elif scale == 'asinh':
            s = self._scale
            self.set_minor_locator(
                    mticker.AsinhLocator(s.linear_width, base=s._base,
                                         subs=s._subs))
        elif scale == 'logit':
            # 对于logit比例尺，设置LogitLocator来显示次要刻度
            self.set_minor_locator(mticker.LogitLocator(minor=True))
        else:
            # 默认情况下使用AutoMinorLocator来显示次要刻度
            self.set_minor_locator(mticker.AutoMinorLocator())

    def minorticks_off(self):
        """从轴上移除次要刻度."""
        # 使用NullLocator来移除轴上的次要刻度
        self.set_minor_locator(mticker.NullLocator())
    # 检查参数 `which` 是否在 ['major', 'minor', 'both'] 中，确保有效性
    _api.check_in_list(['major', 'minor', 'both'], which=which)
    
    # 将 `kwargs` 中的参数转换为适用于 tick 参数的格式，并存储在 `kwtrans` 中
    kwtrans = self._translate_tick_params(kwargs)

    # 如果 reset 参数为 True，则重置指定类型（'major', 'minor', 或 'both'）的 tick 参数
    if reset:
        if which in ['major', 'both']:
            # 重置主刻度的参数，并更新为新的转换后的参数
            self._reset_major_tick_kw()
            self._major_tick_kw.update(kwtrans)
        if which in ['minor', 'both']:
            # 重置次刻度的参数，并更新为新的转换后的参数
            self._reset_minor_tick_kw()
            self._minor_tick_kw.update(kwtrans)
        # 重置所有刻度
        self.reset_ticks()
    else:
        # 如果 reset 参数为 False，则更新指定类型（'major', 'minor', 或 'both'）的 tick 参数
        if which in ['major', 'both']:
            # 更新主刻度的参数为新的转换后的参数，并应用到每一个主刻度上
            self._major_tick_kw.update(kwtrans)
            for tick in self.majorTicks:
                tick._apply_params(**kwtrans)
        if which in ['minor', 'both']:
            # 更新次刻度的参数为新的转换后的参数，并应用到每一个次刻度上
            self._minor_tick_kw.update(kwtrans)
            for tick in self.minorTicks:
                tick._apply_params(**kwtrans)
        
        # 如果 `kwargs` 中包含 'label1On' 或 'label2On'，则设置偏移文本的可见性
        if 'label1On' in kwtrans or 'label2On' in kwtrans:
            self.offsetText.set_visible(
                self._major_tick_kw.get('label1On', False)
                or self._major_tick_kw.get('label2On', False))
        
        # 如果 `kwargs` 中包含 'labelcolor'，则设置偏移文本的颜色
        if 'labelcolor' in kwtrans:
            self.offsetText.set_color(kwtrans['labelcolor'])

    # 设置对象为过时状态，需要重新绘制
    self.stale = True
    def get_tick_params(self, which='major'):
        """
        Get appearance parameters for ticks, ticklabels, and gridlines.

        .. versionadded:: 3.7

        Parameters
        ----------
        which : {'major', 'minor'}, default: 'major'
            The group of ticks for which the parameters are retrieved.

        Returns
        -------
        dict
            Properties for styling tick elements added to the axis.

        Notes
        -----
        This method returns the appearance parameters for styling *new*
        elements added to this axis and may be different from the values
        on current elements if they were modified directly by the user
        (e.g., via ``set_*`` methods on individual tick objects).

        Examples
        --------
        ::

            >>> ax.yaxis.set_tick_params(labelsize=30, labelcolor='red',
            ...                          direction='out', which='major')
            >>> ax.yaxis.get_tick_params(which='major')
            {'direction': 'out',
            'left': True,
            'right': False,
            'labelleft': True,
            'labelright': False,
            'gridOn': False,
            'labelsize': 30,
            'labelcolor': 'red'}
            >>> ax.yaxis.get_tick_params(which='minor')
            {'left': True,
            'right': False,
            'labelleft': True,
            'labelright': False,
            'gridOn': False}

        """
        # 检查输入的 which 参数是否在 'major' 或 'minor' 中
        _api.check_in_list(['major', 'minor'], which=which)
        
        # 如果 which 参数为 'major'，返回经过反转的主要刻度参数字典
        if which == 'major':
            return self._translate_tick_params(
                self._major_tick_kw, reverse=True
            )
        
        # 如果 which 参数为 'minor'，返回经过反转的次要刻度参数字典
        return self._translate_tick_params(self._minor_tick_kw, reverse=True)
    def _translate_tick_params(kw, reverse=False):
        """
        Translate the kwargs supported by `.Axis.set_tick_params` to kwargs
        supported by `.Tick._apply_params`.

        In particular, this maps axis specific names like 'top', 'left'
        to the generic tick1, tick2 logic of the axis. Additionally, there
        are some other name translations.

        Returns a new dict of translated kwargs.

        Note: Use reverse=True to translate from those supported by
        `.Tick._apply_params` back to those supported by
        `.Axis.set_tick_params`.
        """
        # 复制输入的参数字典，以便进行修改而不影响原始输入
        kw_ = {**kw}

        # 定义允许的关键字列表，这些关键字将被允许进行翻译和使用
        allowed_keys = [
            'size', 'width', 'color', 'tickdir', 'pad',
            'labelsize', 'labelcolor', 'labelfontfamily', 'zorder', 'gridOn',
            'tick1On', 'tick2On', 'label1On', 'label2On',
            'length', 'direction', 'left', 'bottom', 'right', 'top',
            'labelleft', 'labelbottom', 'labelright', 'labeltop',
            'labelrotation',
            *_gridline_param_names]

        # 定义关键字映射，用于将一个关键字翻译成另一个
        keymap = {
            # tick_params的关键字 -> axis的关键字
            'length': 'size',
            'direction': 'tickdir',
            'rotation': 'labelrotation',
            'left': 'tick1On',
            'bottom': 'tick1On',
            'right': 'tick2On',
            'top': 'tick2On',
            'labelleft': 'label1On',
            'labelbottom': 'label1On',
            'labelright': 'label2On',
            'labeltop': 'label2On',
        }

        # 根据reverse参数决定翻译方向，将关键字从一种格式转换为另一种格式
        if reverse:
            kwtrans = {
                oldkey: kw_.pop(newkey)
                for oldkey, newkey in keymap.items() if newkey in kw_
            }
        else:
            kwtrans = {
                newkey: kw_.pop(oldkey)
                for oldkey, newkey in keymap.items() if oldkey in kw_
            }

        # 处理'colors'关键字，将其转换为'color'和'labelcolor'两个关键字
        if 'colors' in kw_:
            c = kw_.pop('colors')
            kwtrans['color'] = c
            kwtrans['labelcolor'] = c

        # 检查剩余的关键字是否在允许的关键字列表中，若不在则抛出异常
        for key in kw_:
            if key not in allowed_keys:
                raise ValueError(
                    "keyword %s is not recognized; valid keywords are %s"
                    % (key, allowed_keys))

        # 将修改后的关键字字典与剩余的未被翻译的关键字合并，返回最终的翻译后的参数字典
        kwtrans.update(kw_)
        return kwtrans

    @_api.rename_parameter("3.8", "clippath", "path")
    def set_clip_path(self, path, transform=None):
        # 调用父类的set_clip_path方法，设置裁剪路径和变换
        super().set_clip_path(path, transform)
        # 对每一个主刻度和次刻度，都设置裁剪路径和变换
        for child in self.majorTicks + self.minorTicks:
            child.set_clip_path(path, transform)
        # 标记当前对象为已过时的状态
        self.stale = True

    def get_view_interval(self):
        """Return the ``(min, max)`` view limits of this axis."""
        # 抛出未实现错误，提示需要派生类重写此方法
        raise NotImplementedError('Derived must override')
    # 抛出未实现错误，要求派生类必须覆盖此方法
    def set_view_interval(self, vmin, vmax, ignore=False):
        """
        Set the axis view limits.  This method is for internal use; Matplotlib
        users should typically use e.g. `~.Axes.set_xlim` or `~.Axes.set_ylim`.

        If *ignore* is False (the default), this method will never reduce the
        preexisting view limits, only expand them if *vmin* or *vmax* are not
        within them.  Moreover, the order of *vmin* and *vmax* does not matter;
        the orientation of the axis will not change.

        If *ignore* is True, the view limits will be set exactly to ``(vmin,
        vmax)`` in that order.
        """
        raise NotImplementedError('Derived must override')

    # 返回此轴的数据限制的 ``(min, max)``
    def get_data_interval(self):
        """Return the ``(min, max)`` data limits of this axis."""
        raise NotImplementedError('Derived must override')

    # 设置轴的数据限制。此方法供内部使用。
    def set_data_interval(self, vmin, vmax, ignore=False):
        """
        Set the axis data limits.  This method is for internal use.

        If *ignore* is False (the default), this method will never reduce the
        preexisting data limits, only expand them if *vmin* or *vmax* are not
        within them.  Moreover, the order of *vmin* and *vmax* does not matter;
        the orientation of the axis will not change.

        If *ignore* is True, the data limits will be set exactly to ``(vmin,
        vmax)`` in that order.
        """
        raise NotImplementedError('Derived must override')

    # 返回此轴是否以“反向”方向定向
    def get_inverted(self):
        """
        Return whether this Axis is oriented in the "inverse" direction.

        The "normal" direction is increasing to the right for the x-axis and to
        the top for the y-axis; the "inverse" direction is increasing to the
        left for the x-axis and to the bottom for the y-axis.
        """
        low, high = self.get_view_interval()
        return high < low

    # 设置此轴是否以“反向”方向定向
    def set_inverted(self, inverted):
        """
        Set whether this Axis is oriented in the "inverse" direction.

        The "normal" direction is increasing to the right for the x-axis and to
        the top for the y-axis; the "inverse" direction is increasing to the
        left for the x-axis and to the bottom for the y-axis.
        """
        a, b = self.get_view_interval()
        # 将 a 和 b 排序后传递给 _set_lim，根据 inverted 的布尔值决定是否反转排序
        # 避免在 Python 3.8 和 np.bool_ 之间出现不良交互，将 inverted 强制转换为布尔值
        self._set_lim(*sorted((a, b), reverse=bool(inverted)), auto=None)
    # 设置默认的坐标轴数据和视图间隔限制，如果它们尚未被修改过。
    def set_default_intervals(self):
        """
        Set the default limits for the axis data and view interval if they
        have not been not mutated yet.
        """
        # 主要支持自定义对象的绘图。例如，如果有人传入一个日期时间对象，
        # 我们无法自动设置数据和视图限制的默认最小/最大值。AxisInfo 接口提供了
        # 自定义类型通过 AxisInfo.default_limits 属性注册默认限制的钩子，
        # 下面的派生代码将检查它并在可用时使用它（否则默认使用 0..1）。

    # 设置艺术家对象（artist）的属性。
    def _set_artist_props(self, a):
        if a is None:
            return
        a.set_figure(self.figure)
    def _update_ticks(self):
        """
        Update ticks (position and labels) using the current data interval of
        the axes.  Return the list of ticks that will be drawn.
        """
        # 获取主刻度的位置
        major_locs = self.get_majorticklocs()
        # 使用主刻度的位置来格式化主刻度的标签
        major_labels = self.major.formatter.format_ticks(major_locs)
        # 获取主刻度对象列表
        major_ticks = self.get_major_ticks(len(major_locs))
        # 遍历主刻度对象列表，更新位置和标签
        for tick, loc, label in zip(major_ticks, major_locs, major_labels):
            tick.update_position(loc)  # 更新刻度的位置
            tick.label1.set_text(label)  # 设置刻度标签文本
            tick.label2.set_text(label)  # 设置刻度次标签文本（如果有）

        # 获取次刻度的位置
        minor_locs = self.get_minorticklocs()
        # 使用次刻度的位置来格式化次刻度的标签
        minor_labels = self.minor.formatter.format_ticks(minor_locs)
        # 获取次刻度对象列表
        minor_ticks = self.get_minor_ticks(len(minor_locs))
        # 遍历次刻度对象列表，更新位置和标签
        for tick, loc, label in zip(minor_ticks, minor_locs, minor_labels):
            tick.update_position(loc)  # 更新刻度的位置
            tick.label1.set_text(label)  # 设置刻度标签文本
            tick.label2.set_text(label)  # 设置刻度次标签文本（如果有）

        # 将主刻度和次刻度对象合并成一个列表
        ticks = [*major_ticks, *minor_ticks]

        # 获取视图区间的低点和高点
        view_low, view_high = self.get_view_interval()
        # 确保视图区间的低点小于高点
        if view_low > view_high:
            view_low, view_high = view_high, view_low

        # 如果当前是3D图且开启了自动边距
        if (hasattr(self, "axes") and self.axes.name == '3d'
                and mpl.rcParams['axes3d.automargin']):
            # 根据最新的自动边距行为调整边距
            # 详细调整说明见注释内部的文档说明
            margin = 0.019965277777777776
            delta = view_high - view_low
            view_high = view_high - delta * margin
            view_low = view_low + delta * margin

        # 将视图区间转换为相应的坐标变换后的值
        interval_t = self.get_transform().transform([view_low, view_high])

        # 筛选出需要绘制的刻度对象
        ticks_to_draw = []
        for tick in ticks:
            try:
                loc_t = self.get_transform().transform(tick.get_loc())
            except AssertionError:
                # 处理可能出现的异常情况
                # transforms.transform 不允许掩码值，但某些刻度可能会生成掩码值，因此需要此 try/except 块
                pass
            else:
                # 如果坐标在视图区间内，则添加到绘制列表中
                if mtransforms._interval_contains_close(interval_t, loc_t):
                    ticks_to_draw.append(tick)

        return ticks_to_draw

    def _get_ticklabel_bboxes(self, ticks, renderer=None):
        """Return lists of bboxes for ticks' label1's and label2's."""
        # 如果未提供渲染器，则使用图表对象的渲染器
        if renderer is None:
            renderer = self.figure._get_renderer()
        # 返回刻度标签1和标签2的边界框列表
        return ([tick.label1.get_window_extent(renderer)
                 for tick in ticks if tick.label1.get_visible()],
                [tick.label2.get_window_extent(renderer)
                 for tick in ticks if tick.label2.get_visible()])
    # 返回一个边界框，用于包围当前轴上的标签、轴标签和偏移文本
    def get_tightbbox(self, renderer=None, *, for_layout_only=False):
        """
        Return a bounding box that encloses the axis. It only accounts
        tick labels, axis label, and offsetText.

        If *for_layout_only* is True, then the width of the label (if this
        is an x-axis) or the height of the label (if this is a y-axis) is
        collapsed to near zero.  This allows tight/constrained_layout to ignore
        too-long labels when doing their layout.
        """
        if not self.get_visible():  # 如果轴不可见，则返回
            return
        if renderer is None:  # 如果没有指定渲染器，则使用图形对象的渲染器
            renderer = self.figure._get_renderer()

        # 更新要绘制的刻度
        ticks_to_draw = self._update_ticks()

        # 更新轴标签的位置
        self._update_label_position(renderer)

        # 获取刻度标签的边界框
        tlb1, tlb2 = self._get_ticklabel_bboxes(ticks_to_draw, renderer)

        # 更新偏移文本的位置，并设置其文本内容
        self._update_offset_text_position(tlb1, tlb2)
        self.offsetText.set_text(self.major.formatter.get_offset())

        # 收集所有边界框，包括偏移文本和刻度标签的边界框
        bboxes = [
            *(a.get_window_extent(renderer)
              for a in [self.offsetText]
              if a.get_visible()),  # 如果偏移文本可见，则获取其边界框
            *tlb1, *tlb2,  # 获取主次刻度标签的边界框
        ]

        # 处理轴标签的边界框
        if self.label.get_visible():  # 如果轴标签可见
            bb = self.label.get_window_extent(renderer)  # 获取轴标签的边界框
            # 对于紧凑布局，希望忽略轴标签的宽度/高度，因为其调整不能改善
            # 以下代码将轴标签的相关方向收缩到接近零
            if for_layout_only:
                if self.axis_name == "x" and bb.width > 0:  # 如果是 x 轴且宽度大于零
                    bb.x0 = (bb.x0 + bb.x1) / 2 - 0.5
                    bb.x1 = bb.x0 + 1.0
                if self.axis_name == "y" and bb.height > 0:  # 如果是 y 轴且高度大于零
                    bb.y0 = (bb.y0 + bb.y1) / 2 - 0.5
                    bb.y1 = bb.y0 + 1.0
            bboxes.append(bb)  # 将轴标签的边界框添加到 bboxes 中

        # 过滤掉宽度或高度为非正无穷的边界框，并保证至少宽度和高度大于零
        bboxes = [b for b in bboxes
                  if 0 < b.width < np.inf and 0 < b.height < np.inf]

        # 如果存在有效的边界框，则返回它们的联合边界框，否则返回 None
        if bboxes:
            return mtransforms.Bbox.union(bboxes)
        else:
            return None

    # 获取主刻度和次刻度的最大刻度间距
    def get_tick_padding(self):
        values = []
        if len(self.majorTicks):
            values.append(self.majorTicks[0].get_tick_padding())
        if len(self.minorTicks):
            values.append(self.minorTicks[0].get_tick_padding())
        return max(values, default=0)

    # 允许用于光栅化的装饰器
    @martist.allow_rasterization
    # 绘制该轴的内容，使用给定的渲染器
    def draw(self, renderer):
        # 继承的文档字符串，说明该方法继承自父类
        if not self.get_visible():
            return
        # 在渲染器中开启一个分组，标识为当前轴对象的标识符
        renderer.open_group(__name__, gid=self.get_gid())

        # 更新需要绘制的刻度
        ticks_to_draw = self._update_ticks()
        # 获取需要绘制刻度标签的边界框
        tlb1, tlb2 = self._get_ticklabel_bboxes(ticks_to_draw, renderer)

        # 遍历需要绘制的刻度，并依次绘制
        for tick in ticks_to_draw:
            tick.draw(renderer)

        # 调整标签位置，以避免与刻度标签重叠
        self._update_label_position(renderer)
        # 绘制轴标签
        self.label.draw(renderer)

        # 更新偏移文本的位置
        self._update_offset_text_position(tlb1, tlb2)
        # 设置偏移文本的内容为主刻度格式化器的偏移值，并绘制
        self.offsetText.set_text(self.major.formatter.get_offset())
        self.offsetText.draw(renderer)

        # 在渲染器中关闭之前开启的轴分组
        renderer.close_group(__name__)
        # 标记该对象不需要更新
        self.stale = False

    # 返回该轴的网格线列表，每条网格线表示为一个 `.Line2D` 对象
    def get_gridlines(self):
        # 获取主刻度
        ticks = self.get_major_ticks()
        # 返回所有主刻度的网格线，以列表形式存储
        return cbook.silent_list('Line2D gridline',
                                 [tick.gridline for tick in ticks])

    # 返回轴标签文本的实例，作为一个 `Text` 对象
    def get_label(self):
        """Return the axis label as a Text instance."""
        return self.label

    # 返回偏移文本的实例，作为一个 `Text` 对象
    def get_offset_text(self):
        """Return the axis offsetText as a Text instance."""
        return self.offsetText

    # 返回用于拾取操作的轴深度
    def get_pickradius(self):
        """Return the depth of the axis used by the picker."""
        return self._pickradius

    # 返回该轴的主刻度标签列表，每个标签表示为一个 `~.text.Text` 对象
    def get_majorticklabels(self):
        # 更新刻度
        self._update_ticks()
        # 获取主刻度
        ticks = self.get_major_ticks()
        # 获取所有可见的主刻度标签
        labels1 = [tick.label1 for tick in ticks if tick.label1.get_visible()]
        labels2 = [tick.label2 for tick in ticks if tick.label2.get_visible()]
        return labels1 + labels2

    # 返回该轴的次刻度标签列表，每个标签表示为一个 `~.text.Text` 对象
    def get_minorticklabels(self):
        # 更新刻度
        self._update_ticks()
        # 获取次刻度
        ticks = self.get_minor_ticks()
        # 获取所有可见的次刻度标签
        labels1 = [tick.label1 for tick in ticks if tick.label1.get_visible()]
        labels2 = [tick.label2 for tick in ticks if tick.label2.get_visible()]
        return labels1 + labels2
    def get_ticklabels(self, minor=False, which=None):
        """
        Get this Axis' tick labels.

        Parameters
        ----------
        minor : bool
           Whether to return the minor or the major ticklabels.

        which : None, ('minor', 'major', 'both')
           Overrides *minor*.

           Selects which ticklabels to return

        Returns
        -------
        list of `~matplotlib.text.Text`
        """
        # 如果指定了 which 参数
        if which is not None:
            # 根据 which 参数返回相应类型的 tick labels
            if which == 'minor':
                return self.get_minorticklabels()
            elif which == 'major':
                return self.get_majorticklabels()
            elif which == 'both':
                return self.get_majorticklabels() + self.get_minorticklabels()
            else:
                # 检查 which 是否在指定的列表中，否则抛出异常
                _api.check_in_list(['major', 'minor', 'both'], which=which)
        # 如果 minor 参数为 True，则返回次要 tick labels
        if minor:
            return self.get_minorticklabels()
        # 默认返回主要 tick labels
        return self.get_majorticklabels()

    def get_majorticklines(self):
        r"""Return this Axis' major tick lines as a list of `.Line2D`\s."""
        # 初始化空列表来存储主要刻度线
        lines = []
        # 获取主要刻度线对象列表
        ticks = self.get_major_ticks()
        # 遍历主要刻度线对象，将其刻度线添加到 lines 列表中
        for tick in ticks:
            lines.append(tick.tick1line)
            lines.append(tick.tick2line)
        # 返回主要刻度线的列表
        return cbook.silent_list('Line2D ticklines', lines)

    def get_minorticklines(self):
        r"""Return this Axis' minor tick lines as a list of `.Line2D`\s."""
        # 初始化空列表来存储次要刻度线
        lines = []
        # 获取次要刻度线对象列表
        ticks = self.get_minor_ticks()
        # 遍历次要刻度线对象，将其刻度线添加到 lines 列表中
        for tick in ticks:
            lines.append(tick.tick1line)
            lines.append(tick.tick2line)
        # 返回次要刻度线的列表
        return cbook.silent_list('Line2D ticklines', lines)

    def get_ticklines(self, minor=False):
        r"""Return this Axis' tick lines as a list of `.Line2D`\s."""
        # 如果 minor 参数为 True，则返回次要刻度线列表
        if minor:
            return self.get_minorticklines()
        # 否则返回主要刻度线列表
        return self.get_majorticklines()

    def get_majorticklocs(self):
        """Return this Axis' major tick locations in data coordinates."""
        # 调用主要刻度定位器返回主要刻度的位置
        return self.major.locator()

    def get_minorticklocs(self):
        """Return this Axis' minor tick locations in data coordinates."""
        # 移除重复的次要刻度，与主要刻度位置重叠的次要刻度位置
        minor_locs = np.asarray(self.minor.locator())
        if self.remove_overlapping_locs:
            # 获取主要刻度的位置
            major_locs = self.major.locator()
            # 获取坐标变换对象并转换次要刻度位置和主要刻度位置
            transform = self._scale.get_transform()
            tr_minor_locs = transform.transform(minor_locs)
            tr_major_locs = transform.transform(major_locs)
            # 获取视图间隔的转换后的极限值
            lo, hi = sorted(transform.transform(self.get_view_interval()))
            # 使用转换后的视图限制作为比例尺，设置公差，移除重叠的次要刻度位置
            tol = (hi - lo) * 1e-5
            mask = np.isclose(tr_minor_locs[:, None], tr_major_locs[None, :], atol=tol, rtol=0).any(axis=1)
            minor_locs = minor_locs[~mask]
        # 返回处理后的次要刻度位置数组
        return minor_locs
    def get_ticklocs(self, *, minor=False):
        """
        Return this Axis' tick locations in data coordinates.

        The locations are not clipped to the current axis limits and hence
        may contain locations that are not visible in the output.

        Parameters
        ----------
        minor : bool, default: False
            True to return the minor tick locations,
            False to return the major tick locations.

        Returns
        -------
        array of tick locations
        """
        # 根据参数决定返回主要或次要刻度的位置
        return self.get_minorticklocs() if minor else self.get_majorticklocs()

    def get_ticks_direction(self, minor=False):
        """
        Return an array of this Axis' tick directions.

        Parameters
        ----------
        minor : bool, default: False
            True to return the minor tick directions,
            False to return the major tick directions.

        Returns
        -------
        array of tick directions
        """
        if minor:
            # 返回次要刻度的方向数组
            return np.array(
                [tick._tickdir for tick in self.get_minor_ticks()])
        else:
            # 返回主要刻度的方向数组
            return np.array(
                [tick._tickdir for tick in self.get_major_ticks()])

    def _get_tick(self, major):
        """Return the default tick instance."""
        if self._tick_class is None:
            # 如果未定义 _tick_class，抛出未实现错误
            raise NotImplementedError(
                f"The Axis subclass {self.__class__.__name__} must define "
                "_tick_class or reimplement _get_tick()")
        tick_kw = self._major_tick_kw if major else self._minor_tick_kw
        # 根据参数创建并返回刻度实例
        return self._tick_class(self.axes, 0, major=major, **tick_kw)

    def _get_tick_label_size(self, axis_name):
        """
        Return the text size of tick labels for this Axis.

        This is a convenience function to avoid having to create a `Tick` in
        `.get_tick_space`, since it is expensive.
        """
        tick_kw = self._major_tick_kw
        # 获取刻度标签的文本大小
        size = tick_kw.get('labelsize',
                           mpl.rcParams[f'{axis_name}tick.labelsize'])
        return mtext.FontProperties(size=size).get_size_in_points()

    def _copy_tick_props(self, src, dest):
        """Copy the properties from *src* tick to *dest* tick."""
        if src is None or dest is None:
            return
        # 复制源刻度到目标刻度的属性
        dest.label1.update_from(src.label1)
        dest.label2.update_from(src.label2)
        dest.tick1line.update_from(src.tick1line)
        dest.tick2line.update_from(src.tick2line)
        dest.gridline.update_from(src.gridline)

    def get_label_text(self):
        """Get the text of the label."""
        # 获取轴标签的文本内容
        return self.label.get_text()

    def get_major_locator(self):
        """Get the locator of the major ticker."""
        # 获取主要刻度的定位器
        return self.major.locator

    def get_minor_locator(self):
        """Get the locator of the minor ticker."""
        # 获取次要刻度的定位器
        return self.minor.locator

    def get_major_formatter(self):
        """Get the formatter of the major ticker."""
        # 获取主要刻度的格式化器
        return self.major.formatter
    # 获取次要刻度的格式化器
    def get_minor_formatter(self):
        """Get the formatter of the minor ticker."""
        # 返回次要刻度的格式化器对象
        return self.minor.formatter

    # 获取主要刻度的列表
    def get_major_ticks(self, numticks=None):
        r"""
        Return the list of major `.Tick`\s.

        .. warning::

            Ticks are not guaranteed to be persistent. Various operations
            can create, delete and modify the Tick instances. There is an
            imminent risk that changes to individual ticks will not
            survive if you work on the figure further (including also
            panning/zooming on a displayed figure).

            Working on the individual ticks is a method of last resort.
            Use `.set_tick_params` instead if possible.
        """
        # 如果未指定刻度数，则默认使用主要刻度的总数
        if numticks is None:
            numticks = len(self.get_majorticklocs())

        # 确保主要刻度列表中至少有 numticks 个刻度
        while len(self.majorTicks) < numticks:
            # 创建新的刻度并添加到主要刻度列表中
            tick = self._get_tick(major=True)
            self.majorTicks.append(tick)
            # 复制第一个刻度的属性到新创建的刻度上
            self._copy_tick_props(self.majorTicks[0], tick)

        # 返回前 numticks 个主要刻度对象的列表
        return self.majorTicks[:numticks]

    # 获取次要刻度的列表
    def get_minor_ticks(self, numticks=None):
        r"""
        Return the list of minor `.Tick`\s.

        .. warning::

            Ticks are not guaranteed to be persistent. Various operations
            can create, delete and modify the Tick instances. There is an
            imminent risk that changes to individual ticks will not
            survive if you work on the figure further (including also
            panning/zooming on a displayed figure).

            Working on the individual ticks is a method of last resort.
            Use `.set_tick_params` instead if possible.
        """
        # 如果未指定刻度数，则默认使用次要刻度的总数
        if numticks is None:
            numticks = len(self.get_minorticklocs())

        # 确保次要刻度列表中至少有 numticks 个刻度
        while len(self.minorTicks) < numticks:
            # 创建新的刻度并添加到次要刻度列表中
            tick = self._get_tick(major=False)
            self.minorTicks.append(tick)
            # 复制第一个刻度的属性到新创建的刻度上
            self._copy_tick_props(self.minorTicks[0], tick)

        # 返回前 numticks 个次要刻度对象的列表
        return self.minorTicks[:numticks]
    def grid(self, visible=None, which='major', **kwargs):
        """
        Configure the grid lines.

        Parameters
        ----------
        visible : bool or None
            Whether to show the grid lines.  If any *kwargs* are supplied, it
            is assumed you want the grid on and *visible* will be set to True.

            If *visible* is *None* and there are no *kwargs*, this toggles the
            visibility of the lines.

        which : {'major', 'minor', 'both'}
            The grid lines to apply the changes on.

        **kwargs : `~matplotlib.lines.Line2D` properties
            Define the line properties of the grid, e.g.::

                grid(color='r', linestyle='-', linewidth=2)
        """
        # 如果有传入 kwargs，则假设要显示网格线，并将 visible 设置为 True
        if kwargs:
            if visible is None:
                visible = True
            # 如果 visible 不是 None 且为假值（如 False），则警告用户启用网格线
            elif not visible:  # something false-like but not None
                _api.warn_external('First parameter to grid() is false, '
                                   'but line properties are supplied. The '
                                   'grid will be enabled.')
                visible = True
        
        # 将 which 转换为小写字母形式
        which = which.lower()
        # 检查 which 是否在 ['major', 'minor', 'both'] 中
        _api.check_in_list(['major', 'minor', 'both'], which=which)
        
        # 将 kwargs 中的属性名称前加上 'grid_'，存储在 gridkw 字典中
        gridkw = {f'grid_{name}': value for name, value in kwargs.items()}
        
        # 如果 which 是 'minor' 或 'both'，设置次要刻度的网格状态和属性
        if which in ['minor', 'both']:
            gridkw['gridOn'] = (not self._minor_tick_kw['gridOn']
                                if visible is None else visible)
            self.set_tick_params(which='minor', **gridkw)
        
        # 如果 which 是 'major' 或 'both'，设置主要刻度的网格状态和属性
        if which in ['major', 'both']:
            gridkw['gridOn'] = (not self._major_tick_kw['gridOn']
                                if visible is None else visible)
            self.set_tick_params(which='major', **gridkw)
        
        # 设置为需要刷新状态
        self.stale = True

    def update_units(self, data):
        """
        Introspect *data* for units converter and update the
        ``axis.converter`` instance if necessary. Return *True*
        if *data* is registered for unit conversion.
        """
        # 获取数据对应的单位转换器
        converter = munits.registry.get_converter(data)
        
        # 如果找不到对应的转换器，返回 False
        if converter is None:
            return False
        
        # 检查是否需要更新轴的单位转换器
        neednew = self.converter != converter
        self.converter = converter
        
        # 如果默认单位可用且当前轴没有单位，则设置默认单位
        default = self.converter.default_units(data, self)
        if default is not None and self.units is None:
            self.set_units(default)
        
        # 否则如果需要新的单位转换器，更新轴信息
        elif neednew:
            self._update_axisinfo()
        
        # 设置为需要刷新状态
        self.stale = True
        return True
    def _update_axisinfo(self):
        """
        检查存储单位的轴转换器，看是否需要更新轴信息。
        """
        # 如果没有设置转换器，则直接返回
        if self.converter is None:
            return

        # 获取当前单位对应的轴信息
        info = self.converter.axisinfo(self.units, self)

        # 如果获取的信息为空，则返回
        if info is None:
            return
        
        # 如果主刻度的定位器不是默认的并且与新的主刻度定位器不同，则更新
        if info.majloc is not None and \
           self.major.locator != info.majloc and self.isDefault_majloc:
            self.set_major_locator(info.majloc)
            self.isDefault_majloc = True
        
        # 如果次刻度的定位器不是默认的并且与新的次刻度定位器不同，则更新
        if info.minloc is not None and \
           self.minor.locator != info.minloc and self.isDefault_minloc:
            self.set_minor_locator(info.minloc)
            self.isDefault_minloc = True
        
        # 如果主刻度的格式化器不是默认的并且与新的主刻度格式化器不同，则更新
        if info.majfmt is not None and \
           self.major.formatter != info.majfmt and self.isDefault_majfmt:
            self.set_major_formatter(info.majfmt)
            self.isDefault_majfmt = True
        
        # 如果次刻度的格式化器不是默认的并且与新的次刻度格式化器不同，则更新
        if info.minfmt is not None and \
           self.minor.formatter != info.minfmt and self.isDefault_minfmt:
            self.set_minor_formatter(info.minfmt)
            self.isDefault_minfmt = True
        
        # 如果标签信息不是默认的，则更新标签文本
        if info.label is not None and self.isDefault_label:
            self.set_label_text(info.label)
            self.isDefault_label = True
        
        # 设置默认的间隔
        self.set_default_intervals()

    def have_units(self):
        """
        检查是否存在单位设置。

        Returns
        -------
        bool
            如果存在单位设置，则返回 True，否则返回 False。
        """
        return self.converter is not None or self.units is not None

    def convert_units(self, x):
        """
        将给定的值转换到轴的单位下。

        Parameters
        ----------
        x : scalar or array-like
            需要转换的值。

        Returns
        -------
        scalar or array-like
            转换后的值。

        Raises
        ------
        munits.ConversionError
            如果转换失败，则抛出异常。
        """
        # 如果 x 已经是 Matplotlib 原生支持的类型，则无需转换，直接返回
        if munits._is_natively_supported(x):
            return x

        # 如果尚未设置转换器，则从注册表获取适合 x 的转换器
        if self.converter is None:
            self.converter = munits.registry.get_converter(x)

        # 如果仍然没有转换器，则直接返回 x
        if self.converter is None:
            return x
        
        # 尝试进行转换，如果出现异常，则捕获并抛出 ConversionError 异常
        try:
            ret = self.converter.convert(x, self.units, self)
        except Exception as e:
            raise munits.ConversionError('Failed to convert value(s) to axis '
                                         f'units: {x!r}') from e
        return ret

    def set_units(self, u):
        """
        设置轴的单位。

        Parameters
        ----------
        u : units tag
            要设置的单位。

        Notes
        -----
        任何共享轴的单位也将被更新。
        """
        # 如果单位没有改变，则直接返回
        if u == self.units:
            return
        
        # 更新所有共享轴的单位，并更新轴信息
        for axis in self._get_shared_axis():
            axis.units = u
            axis._update_axisinfo()
            axis.callbacks.process('units')
            axis.stale = True

    def get_units(self):
        """
        返回轴的单位。

        Returns
        -------
        units tag
            轴的单位。
        """
        return self.units
    # 设置轴标签的文本值
    def set_label_text(self, label, fontdict=None, **kwargs):
        """
        Set the text value of the axis label.

        Parameters
        ----------
        label : str
            Text string.
        fontdict : dict
            Text properties.

            .. admonition:: Discouraged

               The use of *fontdict* is discouraged. Parameters should be passed as
               individual keyword arguments or using dictionary-unpacking
               ``set_label_text(..., **fontdict)``.

        **kwargs
            Merged into fontdict.
        """
        # 标记当前轴标签不再使用默认值
        self.isDefault_label = False
        # 设置轴标签的文本内容
        self.label.set_text(label)
        # 如果提供了字典类型的文本属性，更新轴标签的属性
        if fontdict is not None:
            self.label.update(fontdict)
        # 更新轴标签的其他属性
        self.label.update(kwargs)
        # 标记轴为需要更新状态
        self.stale = True
        # 返回更新后的轴标签对象
        return self.label

    # 设置主刻度线的格式化器
    def set_major_formatter(self, formatter):
        """
        Set the formatter of the major ticker.

        In addition to a `~matplotlib.ticker.Formatter` instance,
        this also accepts a ``str`` or function.

        For a ``str`` a `~matplotlib.ticker.StrMethodFormatter` is used.
        The field used for the value must be labeled ``'x'`` and the field used
        for the position must be labeled ``'pos'``.
        See the  `~matplotlib.ticker.StrMethodFormatter` documentation for
        more information.

        For a function, a `~matplotlib.ticker.FuncFormatter` is used.
        The function must take two inputs (a tick value ``x`` and a
        position ``pos``), and return a string containing the corresponding
        tick label.
        See the  `~matplotlib.ticker.FuncFormatter` documentation for
        more information.

        Parameters
        ----------
        formatter : `~matplotlib.ticker.Formatter`, ``str``, or function
        """
        # 调用内部方法设置主刻度线的格式化器
        self._set_formatter(formatter, self.major)

    # 设置次要刻度线的格式化器
    def set_minor_formatter(self, formatter):
        """
        Set the formatter of the minor ticker.

        In addition to a `~matplotlib.ticker.Formatter` instance,
        this also accepts a ``str`` or function.
        See `.Axis.set_major_formatter` for more information.

        Parameters
        ----------
        formatter : `~matplotlib.ticker.Formatter`, ``str``, or function
        """
        # 调用内部方法设置次要刻度线的格式化器
        self._set_formatter(formatter, self.minor)
    # 设置刻度标签的格式化器和级别
    def _set_formatter(self, formatter, level):
        # 如果 formatter 是字符串，则转换为 StrMethodFormatter 对象
        if isinstance(formatter, str):
            formatter = mticker.StrMethodFormatter(formatter)
        # 如果 formatter 是可调用的且不是 TickHelper 类型，则转换为 FuncFormatter 对象
        # 避免错误，如使用 Locator 而不是 Formatter。
        elif (callable(formatter) and
              not isinstance(formatter, mticker.TickHelper)):
            formatter = mticker.FuncFormatter(formatter)
        else:
            # 检查 formatter 是否为 mticker.Formatter 类型
            _api.check_isinstance(mticker.Formatter, formatter=formatter)

        # 如果 formatter 是 FixedFormatter 类型且序列长度大于0，
        # 且 level 的 locator 不是 FixedLocator 类型，则发出警告
        if (isinstance(formatter, mticker.FixedFormatter)
                and len(formatter.seq) > 0
                and not isinstance(level.locator, mticker.FixedLocator)):
            _api.warn_external('FixedFormatter should only be used together '
                               'with FixedLocator')

        # 根据级别设置是否为默认的格式化器
        if level == self.major:
            self.isDefault_majfmt = False
        else:
            self.isDefault_minfmt = False

        # 将 formatter 设置为当前级别的格式化器
        level.formatter = formatter
        # 将 formatter 关联到当前 Axis 对象
        formatter.set_axis(self)
        # 标记状态为过时，需要更新
        self.stale = True

    # 设置主刻度定位器
    def set_major_locator(self, locator):
        """
        Set the locator of the major ticker.

        Parameters
        ----------
        locator : `~matplotlib.ticker.Locator`
        """
        # 检查 locator 是否为 mticker.Locator 类型
        _api.check_isinstance(mticker.Locator, locator=locator)
        # 设置主刻度定位器不为默认
        self.isDefault_majloc = False
        # 将 locator 设置为主刻度定位器
        self.major.locator = locator
        # 如果主刻度的 formatter 存在，则将其定位器设置为 locator
        if self.major.formatter:
            self.major.formatter._set_locator(locator)
        # 将 locator 关联到当前 Axis 对象
        locator.set_axis(self)
        # 标记状态为过时，需要更新
        self.stale = True

    # 设置次刻度定位器
    def set_minor_locator(self, locator):
        """
        Set the locator of the minor ticker.

        Parameters
        ----------
        locator : `~matplotlib.ticker.Locator`
        """
        # 检查 locator 是否为 mticker.Locator 类型
        _api.check_isinstance(mticker.Locator, locator=locator)
        # 设置次刻度定位器不为默认
        self.isDefault_minloc = False
        # 将 locator 设置为次刻度定位器
        self.minor.locator = locator
        # 如果次刻度的 formatter 存在，则将其定位器设置为 locator
        if self.minor.formatter:
            self.minor.formatter._set_locator(locator)
        # 将 locator 关联到当前 Axis 对象
        locator.set_axis(self)
        # 标记状态为过时，需要更新
        self.stale = True

    # 设置选择半径，用于 picker 使用的轴深度
    def set_pickradius(self, pickradius):
        """
        Set the depth of the axis used by the picker.

        Parameters
        ----------
        pickradius : float
            The acceptance radius for containment tests.
            See also `.Axis.contains`.
        """
        # 如果 pickradius 不是实数或小于0，则引发 ValueError
        if not isinstance(pickradius, Real) or pickradius < 0:
            raise ValueError("pick radius should be a distance")
        # 设置选择半径
        self._pickradius = pickradius

    # pickradius 的属性，用于获取和设置选择半径
    pickradius = property(
        get_pickradius, set_pickradius, doc="The acceptance radius for "
        "containment tests. See also `.Axis.contains`.")

    # 用于 set_ticklabels 的辅助函数，定义为静态方法以便可序列化
    @staticmethod
    def _format_with_dict(tickd, x, pos):
        return tickd.get(x, "")
    def _set_tick_locations(self, ticks, *, minor=False):
        # set_tick_locations方法用于设置刻度的位置
        
        # 如果用户更改了单位，这里的信息将丢失
        # 将ticks转换为适当的单位
        ticks = self.convert_units(ticks)
        
        # 创建一个FixedLocator对象来验证ticks的有效性
        locator = mticker.FixedLocator(ticks)
        
        # 如果ticks非空，则更新所有共享的轴的视图间隔
        if len(ticks):
            for axis in self._get_shared_axis():
                # set_view_interval会保持任何预先存在的反转
                axis.set_view_interval(min(ticks), max(ticks))
        
        # 设置Axes对象为stale，表示需要更新
        self.axes.stale = True
        
        # 如果是设置次要刻度，则调用set_minor_locator设置次要定位器，并返回次要刻度的列表
        if minor:
            self.set_minor_locator(locator)
            return self.get_minor_ticks(len(ticks))
        else:
            # 否则，调用set_major_locator设置主要定位器，并返回主要刻度的列表
            self.set_major_locator(locator)
            return self.get_major_ticks(len(ticks))

    def set_ticks(self, ticks, labels=None, *, minor=False, **kwargs):
        """
        Set this Axis' tick locations and optionally tick labels.

        If necessary, the view limits of the Axis are expanded so that all
        given ticks are visible.

        Parameters
        ----------
        ticks : 1D array-like
            Array of tick locations (either floats or in axis units). The axis
            `.Locator` is replaced by a `~.ticker.FixedLocator`.

            Pass an empty list (``set_ticks([])``) to remove all ticks.

            Some tick formatters will not label arbitrary tick positions;
            e.g. log formatters only label decade ticks by default. In
            such a case you can set a formatter explicitly on the axis
            using `.Axis.set_major_formatter` or provide formatted
            *labels* yourself.

        labels : list of str, optional
            Tick labels for each location in *ticks*; must have the same length as
            *ticks*. If set, the labels are used as is, via a `.FixedFormatter`.
            If not set, the labels are generated using the axis tick `.Formatter`.

        minor : bool, default: False
            If ``False``, set only the major ticks; if ``True``, only the minor ticks.

        **kwargs
            `.Text` properties for the labels. Using these is only allowed if
            you pass *labels*. In other cases, please use `~.Axes.tick_params`.

        Notes
        -----
        The mandatory expansion of the view limits is an intentional design
        choice to prevent the surprise of a non-visible tick. If you need
        other limits, you should set the limits explicitly after setting the
        ticks.
        """
        # 如果labels为None但kwargs不为空，则抛出异常
        if labels is None and kwargs:
            first_key = next(iter(kwargs))
            raise ValueError(
                f"Incorrect use of keyword argument {first_key!r}. Keyword arguments "
                "other than 'minor' modify the text labels and can only be used if "
                "'labels' are passed as well.")
        
        # 调用_set_tick_locations方法设置刻度位置，并获取返回值
        result = self._set_tick_locations(ticks, minor=minor)
        
        # 如果labels不为None，则调用set_ticklabels设置刻度标签
        if labels is not None:
            self.set_ticklabels(labels, minor=minor, **kwargs)
        
        # 返回_set_tick_locations方法的返回值
        return result
    def _get_tick_boxes_siblings(self, renderer):
        """
        Get the bounding boxes for this `.axis` and its siblings
        as set by `.Figure.align_xlabels` or  `.Figure.align_ylabels`.

        By default, it just gets bboxes for *self*.
        """
        # 获取当前轴及其兄弟轴的边界框，这些轴由 `.Figure.align_xlabels` 或 `.Figure.align_ylabels` 设置。
        # 默认情况下，仅获取当前轴的边界框。
        name = self._get_axis_name()
        if name not in self.figure._align_label_groups:
            return [], []
        # 获取管理当前轴标签组的 Grouper 对象。
        grouper = self.figure._align_label_groups[name]
        bboxes = []
        bboxes2 = []
        # 如果需要对齐来自其他轴的标签：
        for ax in grouper.get_siblings(self.axes):
            # 获取对应于当前轴的轴对象，并更新其刻度
            axis = ax._axis_map[name]
            ticks_to_draw = axis._update_ticks()
            # 获取刻度标签的边界框
            tlb, tlb2 = axis._get_ticklabel_bboxes(ticks_to_draw, renderer)
            bboxes.extend(tlb)
            bboxes2.extend(tlb2)
        return bboxes, bboxes2

    def _update_label_position(self, renderer):
        """
        Update the label position based on the bounding box enclosing
        all the ticklabels and axis spine.
        """
        # 根据包围所有刻度标签和轴脊柱的边界框更新标签位置。
        raise NotImplementedError('Derived must override')

    def _update_offset_text_position(self, bboxes, bboxes2):
        """
        Update the offset text position based on the sequence of bounding
        boxes of all the ticklabels.
        """
        # 根据所有刻度标签的边界框序列更新偏移文本的位置。
        raise NotImplementedError('Derived must override')

    def axis_date(self, tz=None):
        """
        Set up axis ticks and labels to treat data along this Axis as dates.

        Parameters
        ----------
        tz : str or `datetime.tzinfo`, default: :rc:`timezone`
            The timezone used to create date labels.
        """
        # 通过提供带有所需时区的示例 datetime 实例，
        # 可以选择注册的转换器，并设置 "units" 属性，即时区。
        if isinstance(tz, str):
            import dateutil.tz
            tz = dateutil.tz.gettz(tz)
        self.update_units(datetime.datetime(2009, 1, 1, 0, 0, 0, 0, tz))

    def get_tick_space(self):
        """Return the estimated number of ticks that can fit on the axis."""
        # 必须在子类中重写此方法
        raise NotImplementedError()
    def _get_ticks_position(self):
        """
        Helper for `XAxis.get_ticks_position` and `YAxis.get_ticks_position`.

        Check the visibility of tick1line, label1, tick2line, and label2 on
        the first major and the first minor ticks, and return

        - 1 if only tick1line and label1 are visible (which corresponds to
          "bottom" for the x-axis and "left" for the y-axis);
        - 2 if only tick2line and label2 are visible (which corresponds to
          "top" for the x-axis and "right" for the y-axis);
        - "default" if only tick1line, tick2line and label1 are visible;
        - "unknown" otherwise.
        """
        # 获取第一个主要刻度和第一个次要刻度
        major = self.majorTicks[0]
        minor = self.minorTicks[0]
        
        # 检查第一个主要和次要刻度上的可见性，并根据条件返回相应的位置代码或状态
        if all(tick.tick1line.get_visible()
               and not tick.tick2line.get_visible()
               and tick.label1.get_visible()
               and not tick.label2.get_visible()
               for tick in [major, minor]):
            return 1
        elif all(tick.tick2line.get_visible()
                 and not tick.tick1line.get_visible()
                 and tick.label2.get_visible()
                 and not tick.label1.get_visible()
                 for tick in [major, minor]):
            return 2
        elif all(tick.tick1line.get_visible()
                 and tick.tick2line.get_visible()
                 and tick.label1.get_visible()
                 and not tick.label2.get_visible()
                 for tick in [major, minor]):
            return "default"
        else:
            return "unknown"

    def get_label_position(self):
        """
        Return the label position (top or bottom)
        """
        # 返回当前轴的标签位置
        return self.label_position

    def set_label_position(self, position):
        """
        Set the label position (top or bottom)

        Parameters
        ----------
        position : {'top', 'bottom'}
        """
        # 设置轴的标签位置，但是此方法尚未实现，故抛出未实现错误
        raise NotImplementedError()

    def get_minpos(self):
        # 获取最小位置，此方法尚未实现
        raise NotImplementedError()
def _make_getset_interval(method_name, lim_name, attr_name):
    """
    Helper to generate ``get_{data,view}_interval`` and
    ``set_{data,view}_interval`` implementations.
    """

    def getter(self):
        # 从轴对象中获取特定限制属性的值并返回
        return getattr(getattr(self.axes, lim_name), attr_name)

    def setter(self, vmin, vmax, ignore=False):
        # 设置特定限制属性的值，根据 ignore 参数决定是否忽略原始值
        if ignore:
            setattr(getattr(self.axes, lim_name), attr_name, (vmin, vmax))
        else:
            oldmin, oldmax = getter(self)
            if oldmin < oldmax:
                # 如果原始最小值小于最大值，则按照给定的 vmin 和 vmax 更新限制范围
                setter(self, min(vmin, vmax, oldmin), max(vmin, vmax, oldmax),
                       ignore=True)
            else:
                # 如果原始最小值大于等于最大值，则反向按照 vmin 和 vmax 更新限制范围
                setter(self, max(vmin, vmax, oldmin), min(vmin, vmax, oldmax),
                       ignore=True)
        # 设置 stale 标志为 True，表示需要更新
        self.stale = True

    getter.__name__ = f"get_{method_name}_interval"
    setter.__name__ = f"set_{method_name}_interval"

    return getter, setter


class XAxis(Axis):
    __name__ = 'xaxis'
    axis_name = 'x'  #: Read-only name identifying the axis.
    _tick_class = XTick

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init()

    def _init(self):
        """
        Initialize the label and offsetText instance values and
        `label_position` / `offset_text_position`.
        """
        # 设置标签的初始位置和对齐方式，使用 blended_transform_factory 方法创建变换
        self.label.set(
            x=0.5, y=0,
            verticalalignment='top', horizontalalignment='center',
            transform=mtransforms.blended_transform_factory(
                self.axes.transAxes, mtransforms.IdentityTransform()),
        )
        self.label_position = 'bottom'

        # 根据全局设置确定刻度文本的颜色
        if mpl.rcParams['xtick.labelcolor'] == 'inherit':
            tick_color = mpl.rcParams['xtick.color']
        else:
            tick_color = mpl.rcParams['xtick.labelcolor']

        # 设置偏移文本的初始位置和样式
        self.offsetText.set(
            x=1, y=0,
            verticalalignment='top', horizontalalignment='right',
            transform=mtransforms.blended_transform_factory(
                self.axes.transAxes, mtransforms.IdentityTransform()),
            fontsize=mpl.rcParams['xtick.labelsize'],
            color=tick_color
        )
        self.offset_text_position = 'bottom'

    def contains(self, mouseevent):
        """Test whether the mouse event occurred in the x-axis."""
        # 检测鼠标事件是否发生在 x 轴上
        if self._different_canvas(mouseevent):
            return False, {}
        x, y = mouseevent.x, mouseevent.y
        try:
            # 尝试进行坐标变换，将像素坐标转换为轴坐标
            trans = self.axes.transAxes.inverted()
            xaxes, yaxes = trans.transform((x, y))
        except ValueError:
            return False, {}
        # 获取当前轴在画布上的边界框，检测鼠标事件是否在轴内部
        (l, b), (r, t) = self.axes.transAxes.transform([(0, 0), (1, 1)])
        inaxis = 0 <= xaxes <= 1 and (
            b - self._pickradius < y < b or
            t < y < t + self._pickradius)
        return inaxis, {}
    # 设置标签的位置（顶部或底部）
    def set_label_position(self, position):
        """
        Set the label position (top or bottom)

        Parameters
        ----------
        position : {'top', 'bottom'}
        """
        # 根据位置参数获取对应的垂直对齐方式，baseline（基线）或 top（顶部）
        self.label.set_verticalalignment(_api.check_getitem({
            'top': 'baseline', 'bottom': 'top',
        }, position=position))
        # 记录标签的位置
        self.label_position = position
        # 将图表标记为需要更新状态
        self.stale = True

    # 根据包围所有刻度标签和坐标轴脊柱的边界框更新标签位置
    def _update_label_position(self, renderer):
        """
        Update the label position based on the bounding box enclosing
        all the ticklabels and axis spine
        """
        if not self._autolabelpos:
            return

        # 获取当前轴及其通过 `fig.align_xlabels()` 设置的任何兄弟轴的边界框
        bboxes, bboxes2 = self._get_tick_boxes_siblings(renderer=renderer)
        x, y = self.label.get_position()

        if self.label_position == 'bottom':
            # 如果标签位置在底部，则与底部脊柱的边界框或轴的边界框进行合并
            bbox = mtransforms.Bbox.union([
                *bboxes, self.axes.spines.get("bottom", self.axes).get_window_extent()])
            self.label.set_position((x, bbox.y0 - self.labelpad * self.figure.dpi / 72))
        else:
            # 如果标签位置在顶部，则与顶部脊柱的边界框或轴的边界框进行合并
            bbox = mtransforms.Bbox.union([
                *bboxes2, self.axes.spines.get("top", self.axes).get_window_extent()])
            self.label.set_position((x, bbox.y1 + self.labelpad * self.figure.dpi / 72))

    # 根据所有刻度标签的边界框序列更新偏移文本的位置
    def _update_offset_text_position(self, bboxes, bboxes2):
        """
        Update the offset_text position based on the sequence of bounding
        boxes of all the ticklabels
        """
        x, y = self.offsetText.get_position()
        if not hasattr(self, '_tick_position'):
            self._tick_position = 'bottom'
        if self._tick_position == 'bottom':
            if not len(bboxes):
                bottom = self.axes.bbox.ymin
            else:
                bbox = mtransforms.Bbox.union(bboxes)
                bottom = bbox.y0
            y = bottom - self.OFFSETTEXTPAD * self.figure.dpi / 72
        else:
            if not len(bboxes2):
                top = self.axes.bbox.ymax
            else:
                bbox = mtransforms.Bbox.union(bboxes2)
                top = bbox.y1
            y = top + self.OFFSETTEXTPAD * self.figure.dpi / 72
        self.offsetText.set_position((x, y))
    def set_ticks_position(self, position):
        """
        Set the ticks position.

        Parameters
        ----------
        position : {'top', 'bottom', 'both', 'default', 'none'}
            'both' sets the ticks to appear on both positions, but does not
            change the tick labels.  'default' resets the tick positions to
            the default: ticks on both positions, labels at bottom.  'none'
            can be used if you don't want any ticks. 'none' and 'both'
            affect only the ticks, not the labels.
        """
        # 根据传入的 position 参数设置刻度线的位置和相关属性
        if position == 'top':
            # 设置刻度线在顶部，标签也在顶部
            self.set_tick_params(which='both', top=True, labeltop=True,
                                 bottom=False, labelbottom=False)
            # 记录当前刻度线位置为顶部
            self._tick_position = 'top'
            # 设置偏移文本的垂直对齐方式为底部
            self.offsetText.set_verticalalignment('bottom')
        elif position == 'bottom':
            # 设置刻度线在底部，标签也在底部
            self.set_tick_params(which='both', top=False, labeltop=False,
                                 bottom=True, labelbottom=True)
            # 记录当前刻度线位置为底部
            self._tick_position = 'bottom'
            # 设置偏移文本的垂直对齐方式为顶部
            self.offsetText.set_verticalalignment('top')
        elif position == 'both':
            # 设置刻度线同时在顶部和底部
            self.set_tick_params(which='both', top=True, bottom=True)
        elif position == 'none':
            # 不显示刻度线
            self.set_tick_params(which='both', top=False, bottom=False)
        elif position == 'default':
            # 恢复默认设置，刻度线在顶部，标签在底部
            self.set_tick_params(which='both', top=True, labeltop=False,
                                 bottom=True, labelbottom=True)
            # 记录当前刻度线位置为底部
            self._tick_position = 'bottom'
            # 设置偏移文本的垂直对齐方式为顶部
            self.offsetText.set_verticalalignment('top')
        else:
            # 检查传入的 position 是否在预定义的选项中
            _api.check_in_list(['top', 'bottom', 'both', 'default', 'none'],
                               position=position)
        # 设置标记为脏，表示需要重新绘制
        self.stale = True

    def tick_top(self):
        """
        Move ticks and ticklabels (if present) to the top of the Axes.
        """
        # 检查标签是否存在，并根据情况设置标签显示
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        # 调用 set_ticks_position 方法将刻度线移动到顶部
        self.set_ticks_position('top')
        # 如果在调用本方法之前标签已被关闭，则保持标签关闭状态
        self.set_tick_params(which='both', labeltop=label)

    def tick_bottom(self):
        """
        Move ticks and ticklabels (if present) to the bottom of the Axes.
        """
        # 检查标签是否存在，并根据情况设置标签显示
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        # 调用 set_ticks_position 方法将刻度线移动到底部
        self.set_ticks_position('bottom')
        # 如果在调用本方法之前标签已被关闭，则保持标签关闭状态
        self.set_tick_params(which='both', labelbottom=label)
    # 返回ticks位置信息，根据内部函数返回的结果从预定义的字典中获取
    def get_ticks_position(self):
        """
        Return the ticks position ("top", "bottom", "default", or "unknown").
        """
        return {1: "bottom", 2: "top",
                "default": "default", "unknown": "unknown"}[
                    self._get_ticks_position()]

    # 创建获取和设置视图间隔的函数，并绑定到对应的属性上
    get_view_interval, set_view_interval = _make_getset_interval(
        "view", "viewLim", "intervalx")
    # 创建获取和设置数据间隔的函数，并绑定到对应的属性上
    get_data_interval, set_data_interval = _make_getset_interval(
        "data", "dataLim", "intervalx")

    # 返回轴数据范围的最小正值
    def get_minpos(self):
        return self.axes.dataLim.minposx

    # 设置默认的间隔
    def set_default_intervals(self):
        # 继承的文档字符串
        # 只有在dataLim未改变且用户未更改视图时才更改视图:
        if (not self.axes.dataLim.mutatedx() and
                not self.axes.viewLim.mutatedx()):
            # 如果有转换器，则获取轴信息
            if self.converter is not None:
                info = self.converter.axisinfo(self.units, self)
                # 如果信息有默认限制，则将其转换为轴单位，并设置视图间隔
                if info.default_limits is not None:
                    xmin, xmax = self.convert_units(info.default_limits)
                    self.axes.viewLim.intervalx = xmin, xmax
        # 设置stale属性为True，表示需要更新
        self.stale = True

    # 返回tick空间大小
    def get_tick_space(self):
        # 获取终点位置
        ends = mtransforms.Bbox.unit().transformed(
            self.axes.transAxes - self.figure.dpi_scale_trans)
        length = ends.width * 72
        # 估算tick文本的纵横比不超过3:1
        size = self._get_tick_label_size('x') * 3
        if size > 0:
            # 返回tick空间的整数值
            return int(np.floor(length / size))
        else:
            # 如果size不大于0，则返回一个极大值，表示无限大
            return 2**31 - 1
class YAxis(Axis):
    __name__ = 'yaxis'
    axis_name = 'y'  #: Read-only name identifying the axis.
    _tick_class = YTick  # YTick 类的定义，用于此轴的刻度处理

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init()  # 初始化方法调用

    def _init(self):
        """
        Initialize the label and offsetText instance values and
        `label_position` / `offset_text_position`.
        """
        # 设置标签位置和偏移文本的初始值，使用相应的转换工厂创建适当的变换
        self.label.set(
            x=0, y=0.5,
            verticalalignment='bottom', horizontalalignment='center',
            rotation='vertical', rotation_mode='anchor',
            transform=mtransforms.blended_transform_factory(
                mtransforms.IdentityTransform(), self.axes.transAxes),
        )
        self.label_position = 'left'  # 标签位置初始化为左侧

        if mpl.rcParams['ytick.labelcolor'] == 'inherit':
            tick_color = mpl.rcParams['ytick.color']
        else:
            tick_color = mpl.rcParams['ytick.labelcolor']

        # 设置偏移文本的初始位置和属性
        self.offsetText.set(
            x=0, y=0.5,
            verticalalignment='baseline', horizontalalignment='left',
            transform=mtransforms.blended_transform_factory(
                self.axes.transAxes, mtransforms.IdentityTransform()),
            fontsize=mpl.rcParams['ytick.labelsize'],
            color=tick_color
        )
        self.offset_text_position = 'left'  # 偏移文本位置初始化为左侧

    def contains(self, mouseevent):
        # docstring inherited
        # 检查鼠标事件是否在该轴范围内
        if self._different_canvas(mouseevent):
            return False, {}
        x, y = mouseevent.x, mouseevent.y
        try:
            trans = self.axes.transAxes.inverted()
            xaxes, yaxes = trans.transform((x, y))
        except ValueError:
            return False, {}
        (l, b), (r, t) = self.axes.transAxes.transform([(0, 0), (1, 1)])
        # 检查鼠标位置是否在轴范围内
        inaxis = 0 <= yaxes <= 1 and (
            l - self._pickradius < x < l or
            r < x < r + self._pickradius)
        return inaxis, {}  # 返回是否在轴内以及空字典作为附加信息

    def set_label_position(self, position):
        """
        Set the label position (left or right)

        Parameters
        ----------
        position : {'left', 'right'}
        """
        self.label.set_rotation_mode('anchor')
        # 设置标签的旋转模式和垂直对齐方式
        self.label.set_verticalalignment(_api.check_getitem({
            'left': 'bottom', 'right': 'top',
        }, position=position))
        self.label_position = position  # 更新标签位置
        self.stale = True  # 设置为过期状态，需要重新绘制
    # 更新标签的位置，基于包围所有刻度标签和轴脊柱的边界框
    def _update_label_position(self, renderer):
        """
        Update the label position based on the bounding box enclosing
        all the ticklabels and axis spine
        """
        # 如果不是自动标签位置，则直接返回
        if not self._autolabelpos:
            return

        # 获取当前轴及任何已被 `fig.align_ylabels()` 设置的同级轴的边界框
        bboxes, bboxes2 = self._get_tick_boxes_siblings(renderer=renderer)
        x, y = self.label.get_position()

        # 如果标签位置为 'left'
        if self.label_position == 'left':
            # 与左侧脊柱的范围的并集，如果存在的话；否则与轴的范围的并集
            bbox = mtransforms.Bbox.union([
                *bboxes, self.axes.spines.get("left", self.axes).get_window_extent()])
            self.label.set_position((bbox.x0 - self.labelpad * self.figure.dpi / 72, y))
        else:
            # 与右侧脊柱的范围的并集，如果存在的话；否则与轴的范围的并集
            bbox = mtransforms.Bbox.union([
                *bboxes2, self.axes.spines.get("right", self.axes).get_window_extent()])
            self.label.set_position((bbox.x1 + self.labelpad * self.figure.dpi / 72, y))

    # 更新偏移文本的位置，基于所有刻度标签的边界框序列
    def _update_offset_text_position(self, bboxes, bboxes2):
        """
        Update the offset_text position based on the sequence of bounding
        boxes of all the ticklabels
        """
        x, _ = self.offsetText.get_position()
        # 如果轴上有 'outline' 脊柱，例如对于颜色条的特殊情况
        if 'outline' in self.axes.spines:
            bbox = self.axes.spines['outline'].get_window_extent()
        else:
            bbox = self.axes.bbox
        top = bbox.ymax
        self.offsetText.set_position(
            (x, top + self.OFFSETTEXTPAD * self.figure.dpi / 72)
        )

    # 设置偏移位置
    def set_offset_position(self, position):
        """
        Parameters
        ----------
        position : {'left', 'right'}
        """
        x, y = self.offsetText.get_position()
        # 根据位置参数调整 x 的值
        x = _api.check_getitem({'left': 0, 'right': 1}, position=position)

        self.offsetText.set_ha(position)  # 设置水平对齐方式
        self.offsetText.set_position((x, y))  # 设置新的位置
        self.stale = True  # 设置标记为失效，需要重新绘制
    def set_ticks_position(self, position):
        """
        Set the ticks position.

        Parameters
        ----------
        position : {'left', 'right', 'both', 'default', 'none'}
            'both' sets the ticks to appear on both positions, but does not
            change the tick labels.  'default' resets the tick positions to
            the default: ticks on both positions, labels at left.  'none'
            can be used if you don't want any ticks. 'none' and 'both'
            affect only the ticks, not the labels.
        """
        # 根据不同的位置参数设置刻度的位置和标签的显示
        if position == 'right':
            self.set_tick_params(which='both', right=True, labelright=True,
                                 left=False, labelleft=False)
            # 设置偏移位置
            self.set_offset_position(position)
        elif position == 'left':
            self.set_tick_params(which='both', right=False, labelright=False,
                                 left=True, labelleft=True)
            # 设置偏移位置
            self.set_offset_position(position)
        elif position == 'both':
            self.set_tick_params(which='both', right=True,
                                 left=True)
        elif position == 'none':
            self.set_tick_params(which='both', right=False,
                                 left=False)
        elif position == 'default':
            self.set_tick_params(which='both', right=True, labelright=False,
                                 left=True, labelleft=True)
        else:
            # 检查位置参数是否在预定义的列表中
            _api.check_in_list(['left', 'right', 'both', 'default', 'none'],
                               position=position)
        # 标记对象状态为失效，需要重新绘制
        self.stale = True

    def tick_right(self):
        """
        Move ticks and ticklabels (if present) to the right of the Axes.
        """
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        # 将刻度和刻度标签（如果存在）移到轴的右侧
        self.set_ticks_position('right')
        # 如果在调用此方法之前标签被关闭，则保持关闭状态
        self.set_tick_params(which='both', labelright=label)

    def tick_left(self):
        """
        Move ticks and ticklabels (if present) to the left of the Axes.
        """
        label = True
        if 'label1On' in self._major_tick_kw:
            label = (self._major_tick_kw['label1On']
                     or self._major_tick_kw['label2On'])
        # 将刻度和刻度标签（如果存在）移到轴的左侧
        self.set_ticks_position('left')
        # 如果在调用此方法之前标签被关闭，则保持关闭状态
        self.set_tick_params(which='both', labelleft=label)

    def get_ticks_position(self):
        """
        Return the ticks position ("left", "right", "default", or "unknown").
        """
        # 返回当前刻度的位置信息，可能是"left", "right", "default" 或 "unknown"
        return {1: "left", 2: "right",
                "default": "default", "unknown": "unknown"}[
                    self._get_ticks_position()]

    get_view_interval, set_view_interval = _make_getset_interval(
        "view", "viewLim", "intervaly")
    # 使用 _make_getset_interval 函数创建 get_data_interval 和 set_data_interval 方法
    get_data_interval, set_data_interval = _make_getset_interval(
        "data", "dataLim", "intervaly")

    # 定义获取最小位置的方法
    def get_minpos(self):
        return self.axes.dataLim.minposy

    # 定义设置默认间隔的方法
    def set_default_intervals(self):
        # 继承的文档字符串
        # 如果 dataLim 没有改变且用户没有改变视图，则修改视图:
        if (not self.axes.dataLim.mutatedy() and
                not self.axes.viewLim.mutatedy()):
            # 如果存在转换器，则获取单位信息
            if self.converter is not None:
                info = self.converter.axisinfo(self.units, self)
                # 如果信息中有默认限制，则转换单位并设置视图间隔
                if info.default_limits is not None:
                    ymin, ymax = self.convert_units(info.default_limits)
                    self.axes.viewLim.intervaly = ymin, ymax
        # 设置 stale 属性为 True，表示对象状态需要更新
        self.stale = True

    # 定义获取刻度空间的方法
    def get_tick_space(self):
        # 计算单位转换后的高度
        ends = mtransforms.Bbox.unit().transformed(
            self.axes.transAxes - self.figure.dpi_scale_trans)
        length = ends.height * 72
        # 根据刻度标签的尺寸计算刻度间隔
        size = self._get_tick_label_size('y') * 2
        if size > 0:
            return int(np.floor(length / size))  # 返回刻度间隔的整数值
        else:
            return 2**31 - 1  # 如果尺寸小于等于0，返回一个大整数作为默认值
```
# `D:\src\scipysrc\matplotlib\lib\matplotlib\axes\_secondary_axes.py`

```py
import numbers  # 导入 numbers 模块，用于数值类型的支持

import numpy as np  # 导入 numpy 库，并简写为 np

from matplotlib import _api, _docstring, transforms  # 从 matplotlib 中导入 _api, _docstring, transforms 模块
import matplotlib.ticker as mticker  # 导入 matplotlib 的 ticker 模块，并简写为 mticker
from matplotlib.axes._base import _AxesBase, _TransformedBoundsLocator  # 从 matplotlib.axes._base 中导入 _AxesBase, _TransformedBoundsLocator 类
from matplotlib.axis import Axis  # 从 matplotlib.axis 中导入 Axis 类
from matplotlib.transforms import Transform  # 从 matplotlib.transforms 中导入 Transform 类


class SecondaryAxis(_AxesBase):
    """
    General class to hold a Secondary_X/Yaxis.
    """

    def __init__(self, parent, orientation, location, functions, transform=None,
                 **kwargs):
        """
        See `.secondary_xaxis` and `.secondary_yaxis` for the doc string.
        While there is no need for this to be private, it should really be
        called by those higher level functions.
        """
        _api.check_in_list(["x", "y"], orientation=orientation)  # 检查 orientation 参数是否在列表 ["x", "y"] 中
        self._functions = functions  # 将 functions 参数存储到实例变量 self._functions 中
        self._parent = parent  # 将 parent 参数存储到实例变量 self._parent 中
        self._orientation = orientation  # 将 orientation 参数存储到实例变量 self._orientation 中
        self._ticks_set = False  # 初始化实例变量 self._ticks_set 为 False

        if self._orientation == 'x':
            super().__init__(self._parent.figure, [0, 1., 1, 0.0001], **kwargs)  # 如果 orientation 是 'x'，则调用父类 _AxesBase 的构造函数，设置 x 轴的位置和图形对象
            self._axis = self.xaxis  # 将 x 轴对象存储到实例变量 self._axis 中
            self._locstrings = ['top', 'bottom']  # 设置位置字符串列表为 ['top', 'bottom']
            self._otherstrings = ['left', 'right']  # 设置其它字符串列表为 ['left', 'right']
        else:  # 如果 orientation 是 'y'
            super().__init__(self._parent.figure, [0, 1., 0.0001, 1], **kwargs)  # 调用父类 _AxesBase 的构造函数，设置 y 轴的位置和图形对象
            self._axis = self.yaxis  # 将 y 轴对象存储到实例变量 self._axis 中
            self._locstrings = ['right', 'left']  # 设置位置字符串列表为 ['right', 'left']
            self._otherstrings = ['top', 'bottom']  # 设置其它字符串列表为 ['top', 'bottom']
        self._parentscale = None  # 初始化实例变量 self._parentscale 为 None

        # this gets positioned w/o constrained_layout so exclude:
        # 设置位置和转换
        self.set_location(location, transform)
        # 设置函数
        self.set_functions(functions)

        # styling:
        # 获取另一个轴对象
        otheraxis = self.yaxis if self._orientation == 'x' else self.xaxis
        # 将其主要定位器设置为 NullLocator
        otheraxis.set_major_locator(mticker.NullLocator())
        # 设置 ticks 的位置为 'none'
        otheraxis.set_ticks_position('none')

        # 将其它轴的脊柱设置为不可见
        self.spines[self._otherstrings].set_visible(False)
        # 将位置轴的脊柱设置为可见
        self.spines[self._locstrings].set_visible(True)

        if self._pos < 0.5:
            # 翻转位置字符串列表
            self._locstrings = self._locstrings[::-1]
        # 设置对齐方式
        self.set_alignment(self._locstrings[0])

    def set_alignment(self, align):
        """
        Set if axes spine and labels are drawn at top or bottom (or left/right)
        of the Axes.

        Parameters
        ----------
        align : {'top', 'bottom', 'left', 'right'}
            Either 'top' or 'bottom' for orientation='x' or
            'left' or 'right' for orientation='y' axis.
        """
        _api.check_in_list(self._locstrings, align=align)  # 检查 align 参数是否在位置字符串列表中
        if align == self._locstrings[1]:  # 如果 align 等于第二个位置字符串
            self._locstrings = self._locstrings[::-1]  # 翻转位置字符串列表
        # 设置第一个位置轴的脊柱为可见
        self.spines[self._locstrings[0]].set_visible(True)
        # 设置第二个位置轴的脊柱为不可见
        self.spines[self._locstrings[1]].set_visible(False)
        # 设置轴的 ticks 位置
        self._axis.set_ticks_position(align)
        # 设置轴的标签位置
        self._axis.set_label_position(align)
    def set_location(self, location, transform=None):
        """
        Set the vertical or horizontal location of the axes in
        parent-normalized coordinates.

        Parameters
        ----------
        location : {'top', 'bottom', 'left', 'right'} or float
            The position to put the secondary axis.  Strings can be 'top' or
            'bottom' for orientation='x' and 'right' or 'left' for
            orientation='y'. A float indicates the relative position on the
            parent Axes to put the new Axes, 0.0 being the bottom (or left)
            and 1.0 being the top (or right).

        transform : `.Transform`, optional
            Transform for the location to use. Defaults to
            the parent's ``transAxes``, so locations are normally relative to
            the parent axes.

            .. versionadded:: 3.9
        """

        # 检查 transform 参数的类型是否为 `.Transform` 或 None
        _api.check_isinstance((transforms.Transform, None), transform=transform)

        # 将矩形放置到图形相对坐标系中
        if isinstance(location, str):
            # 检查 location 是否在有效的字符串列表中
            _api.check_in_list(self._locstrings, location=location)
            # 根据位置字符串确定位置，'top' 或 'right' 时为 1.0，否则为 0.0
            self._pos = 1. if location in ('top', 'right') else 0.
        elif isinstance(location, numbers.Real):
            # 如果 location 是实数，则直接设置位置
            self._pos = location
        else:
            # 抛出异常，说明 location 参数不合法
            raise ValueError(
                f"location must be {self._locstrings[0]!r}, "
                f"{self._locstrings[1]!r}, or a float, not {location!r}")

        # 记录设置的位置
        self._loc = location

        # 如果是水平方向 ('x')
        if self._orientation == 'x':
            # 设置边界范围，从 x = 0 到 x = 1，y = pos 到 y = pos + 1e-10，在父坐标系的相对位置
            bounds = [0, self._pos, 1., 1e-10]

            # 如果提供了 transform 参数，则混合使用父坐标系的 transAxes 和 transform
            if transform is not None:
                transform = transforms.blended_transform_factory(
                    self._parent.transAxes, transform)
        else:  # 如果是垂直方向 ('y')
            # 设置边界范围，从 x = pos 到 x = pos + 1e-10，y = 0 到 y = 1，在父坐标系的相对位置
            bounds = [self._pos, 0, 1e-10, 1]
            # 如果提供了 transform 参数，则混合使用 transform 和 父坐标系的 transAxes，用于指定 x 轴
            if transform is not None:
                transform = transforms.blended_transform_factory(
                    transform, self._parent.transAxes)  # 使用提供的 x 轴

        # 如果未提供 transform 参数，则使用父坐标系的 transAxes
        if transform is None:
            transform = self._parent.transAxes

        # 设置轴的定位器，以允许轴在父坐标系中移动
        # 在 ax.apply_aspect() 中被调用
        self.set_axes_locator(_TransformedBoundsLocator(bounds, transform))

    def apply_aspect(self, position=None):
        # 继承的文档字符串
        self._set_lims()  # 设置坐标轴限制
        super().apply_aspect(position)  # 调用父类的 apply_aspect 方法

    @_docstring.copy(Axis.set_ticks)
    # 设置次要坐标轴的刻度位置和标签
    def set_ticks(self, ticks, labels=None, *, minor=False, **kwargs):
        # 调用底层坐标轴对象的设置刻度方法，并返回结果
        ret = self._axis.set_ticks(ticks, labels, minor=minor, **kwargs)
        # 标记次要坐标轴需要重新绘制
        self.stale = True
        # 记录刻度已设置标志为True
        self._ticks_set = True
        # 返回设置刻度的结果
        return ret

    # 设置次要坐标轴的转换函数
    def set_functions(self, functions):
        """
        设置次要坐标轴如何从父坐标轴转换限制。

        Parameters
        ----------
        functions : 2-tuple of func, or `Transform` with an inverse.
            用于父坐标轴值和次要坐标轴值之间的转换。

            如果作为包含两个函数的2元组提供，第一个函数是正向转换函数，第二个是反向转换函数。

            如果提供了一个Transform对象，则该对象必须具有反向转换方法。
        """

        if (isinstance(functions, tuple) and len(functions) == 2 and
                callable(functions[0]) and callable(functions[1])):
            # 如果functions是一个包含两个可调用函数的2元组，则设置为这两个函数
            self._functions = functions
        elif isinstance(functions, Transform):
            # 如果functions是Transform对象，则设置为其正向和反向转换方法
            self._functions = (
                 functions.transform,
                 lambda x: functions.inverted().transform(x)
            )
        elif functions is None:
            # 如果functions为None，则设置为恒等函数
            self._functions = (lambda x: x, lambda x: x)
        else:
            # 如果参数不合法，则抛出数值错误
            raise ValueError('functions argument of secondary Axes '
                             'must be a two-tuple of callable functions '
                             'with the first function being the transform '
                             'and the second being the inverse')
        # 更新坐标轴的比例
        self._set_scale()

    # 绘制次要坐标轴
    def draw(self, renderer):
        """
        绘制次要坐标轴。

        从父坐标轴获取其限制，并使用由`~.axes._secondary_axes.set_functions`指定的转换器转换它们
        （或在初始化Axes时使用*functions*参数指定的转换器）。
        """
        # 设置次要坐标轴的限制
        self._set_lims()
        # 设置比例，以防父坐标轴已设置其比例
        self._set_scale()
        # 调用超类的绘制方法进行绘制
        super().draw(renderer)
    def _set_scale(self):
        """
        Check if parent has set its scale
        """

        # 根据方向确定父对象的刻度尺度
        if self._orientation == 'x':
            pscale = self._parent.xaxis.get_scale()
            set_scale = self.set_xscale
        else:  # 'y'
            pscale = self._parent.yaxis.get_scale()
            set_scale = self.set_yscale
        # 如果父对象的刻度尺度与上次相同，则直接返回
        if pscale == self._parentscale:
            return

        # 如果已设置了刻度，则获取当前刻度位置
        if self._ticks_set:
            ticks = self._axis.get_ticklocs()

        # 根据父对象的刻度尺度设置当前对象的刻度尺度
        # 如果父对象刻度尺度为'log'，则设置为'functionlog'，否则设置为'function'
        set_scale('functionlog' if pscale == 'log' else 'function',
                  functions=self._functions[::-1])

        # 如果已设置了刻度，则保留当前设置的刻度位置
        if self._ticks_set:
            self._axis.set_major_locator(mticker.FixedLocator(ticks))

        # 记录当前父对象的刻度尺度，以便下次比较
        self._parentscale = pscale

    def _set_lims(self):
        """
        Set the limits based on parent limits and the convert method
        between the parent and this secondary Axes.
        """
        # 根据方向确定父对象的限制范围
        if self._orientation == 'x':
            lims = self._parent.get_xlim()
            set_lim = self.set_xlim
        else:  # 'y'
            lims = self._parent.get_ylim()
            set_lim = self.set_ylim
        # 检查限制范围的顺序，确保适当的转换
        order = lims[0] < lims[1]
        lims = self._functions[0](np.array(lims))
        neworder = lims[0] < lims[1]
        # 如果顺序变化了，则反转限制范围，因为转换方法会处理反转
        if neworder != order:
            lims = lims[::-1]
        # 设置当前对象的限制范围
        set_lim(lims)

    def set_aspect(self, *args, **kwargs):
        """
        Secondary Axes cannot set the aspect ratio, so calling this just
        sets a warning.
        """
        # 提示次要轴无法设置纵横比
        _api.warn_external("Secondary Axes can't set the aspect ratio")

    def set_color(self, color):
        """
        Change the color of the secondary Axes and all decorators.

        Parameters
        ----------
        color : :mpltype:`color`
        """
        # 设置次要轴及其所有装饰元素的颜色
        axis = self._axis_map[self._orientation]
        axis.set_tick_params(colors=color)
        for spine in self.spines.values():
            if spine.axis is axis:
                spine.set_color(color)
        axis.label.set_color(color)
# 定义 _secax_docstring 变量，其值是一个包含文档字符串的长字符串
_secax_docstring = '''
Warnings
--------
This method is experimental as of 3.1, and the API may change.

Parameters
----------
location : {'top', 'bottom', 'left', 'right'} or float
    The position to put the secondary axis.  Strings can be 'top' or
    'bottom' for orientation='x' and 'right' or 'left' for
    orientation='y'. A float indicates the relative position on the
    parent Axes to put the new Axes, 0.0 being the bottom (or left)
    and 1.0 being the top (or right).

functions : 2-tuple of func, or Transform with an inverse

    If a 2-tuple of functions, the user specifies the transform
    function and its inverse.  i.e.
    ``functions=(lambda x: 2 / x, lambda x: 2 / x)`` would be an
    reciprocal transform with a factor of 2. Both functions must accept
    numpy arrays as input.

    The user can also directly supply a subclass of
    `.transforms.Transform` so long as it has an inverse.

    See :doc:`/gallery/subplots_axes_and_figures/secondary_axis`
    for examples of making these conversions.

transform : `.Transform`, optional
    If specified, *location* will be
    placed relative to this transform (in the direction of the axis)
    rather than the parent's axis. i.e. a secondary x-axis will
    use the provided y transform and the x transform of the parent.

    .. versionadded:: 3.9

Returns
-------
ax : axes._secondary_axes.SecondaryAxis

Other Parameters
----------------
**kwargs : `~matplotlib.axes.Axes` properties.
    Other miscellaneous Axes parameters.
'''
# 更新文档字符串插值表
_docstring.interpd.update(_secax_docstring=_secax_docstring)
```
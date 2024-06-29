# `D:\src\scipysrc\matplotlib\lib\matplotlib\ticker.py`

```
"""
Tick locating and formatting
============================

This module contains classes for configuring tick locating and formatting.
Generic tick locators and formatters are provided, as well as domain specific
custom ones.

Although the locators know nothing about major or minor ticks, they are used
by the Axis class to support major and minor tick locating and formatting.

.. _tick_locating:
.. _locators:

Tick locating
-------------

The Locator class is the base class for all tick locators. The locators
handle autoscaling of the view limits based on the data limits, and the
choosing of tick locations. A useful semi-automatic tick locator is
`MultipleLocator`. It is initialized with a base, e.g., 10, and it picks
axis limits and ticks that are multiples of that base.

The Locator subclasses defined here are:

======================= =======================================================
`AutoLocator`           `MaxNLocator` with simple defaults. This is the default
                        tick locator for most plotting.
`MaxNLocator`           Finds up to a max number of intervals with ticks at
                        nice locations.
`LinearLocator`         Space ticks evenly from min to max.
`LogLocator`            Space ticks logarithmically from min to max.
`MultipleLocator`       Ticks and range are a multiple of base; either integer
                        or float.
`FixedLocator`          Tick locations are fixed.
`IndexLocator`          Locator for index plots (e.g., where
                        ``x = range(len(y))``).
`NullLocator`           No ticks.
`SymmetricalLogLocator` Locator for use with the symlog norm; works like
                        `LogLocator` for the part outside of the threshold and
                        adds 0 if inside the limits.
`AsinhLocator`          Locator for use with the asinh norm, attempting to
                        space ticks approximately uniformly.
`LogitLocator`          Locator for logit scaling.
`AutoMinorLocator`      Locator for minor ticks when the axis is linear and the
                        major ticks are uniformly spaced. Subdivides the major
                        tick interval into a specified number of minor
                        intervals, defaulting to 4 or 5 depending on the major
                        interval.
======================= =======================================================

There are a number of locators specialized for date locations - see
the :mod:`.dates` module.

You can define your own locator by deriving from Locator. You must
override the ``__call__`` method, which returns a sequence of locations,
and you will probably want to override the autoscale method to set the
view limits from the data limits.

If you want to override the default locator, use one of the above or a custom

"""

# 导入所需的模块
import matplotlib.ticker as ticker

# 定义基类 Locator，用于所有刻度定位器的基础
class Locator:
    """
    Base class for tick locators. Handles autoscaling and choosing tick locations.
    """
    def __call__(self):
        """
        Method that should be overridden to return a sequence of tick locations.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Method that can be overridden to set view limits from data limits.
        """
        raise NotImplementedError("Must override autoscale in subclasses.")

# 定义 MultipleLocator 类，用于生成基于给定基数的刻度
class MultipleLocator(Locator):
    """
    Tick locator that chooses axis limits and ticks that are multiples of a base.
    """
    def __init__(self, base=1.0):
        """
        Initialize with a base value for the multiples.
        """
        self.base = base

    def __call__(self):
        """
        Returns a sequence of tick locations that are multiples of the base.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Override autoscale to adjust view limits based on data limits.
        """
        raise NotImplementedError("Must override autoscale in subclasses.")

# 定义 AutoLocator 类，使用 MaxNLocator 作为默认刻度定位器
class AutoLocator(MaxNLocator):
    """
    Default tick locator used for most plots, inherits from MaxNLocator.
    """
    def __init__(self):
        """
        Initialize with default parameters of MaxNLocator.
        """
        super().__init__()

# 定义 MaxNLocator 类，根据给定的最大间隔数量在合适位置生成刻度
class MaxNLocator(Locator):
    """
    Tick locator that finds up to a max number of intervals with ticks at nice locations.
    """
    def __init__(self, nbins=9, steps=[1, 2, 5, 10]):
        """
        Initialize with number of intervals and steps for tick locations.
        """
        self.nbins = nbins
        self.steps = steps

    def __call__(self):
        """
        Returns a sequence of tick locations with a max number of intervals.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Override autoscale to adjust view limits based on data limits.
        """
        raise NotImplementedError("Must override autoscale in subclasses.")

# 定义 LinearLocator 类，生成在最小和最大值之间均匀分布的刻度
class LinearLocator(Locator):
    """
    Tick locator that spaces ticks evenly from min to max.
    """
    def __init__(self, numticks=5):
        """
        Initialize with number of ticks to be generated.
        """
        self.numticks = numticks

    def __call__(self):
        """
        Returns a sequence of evenly spaced tick locations.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Override autoscale to adjust view limits based on data limits.
        """
        raise NotImplementedError("Must override autoscale in subclasses.")

# 定义 LogLocator 类，生成在最小和最大值之间对数分布的刻度
class LogLocator(Locator):
    """
    Tick locator that spaces ticks logarithmically from min to max.
    """
    def __init__(self, base=10.0):
        """
        Initialize with a base value for the logarithm.
        """
        self.base = base

    def __call__(self):
        """
        Returns a sequence of logarithmically spaced tick locations.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Override autoscale to adjust view limits based on data limits.
        """
        raise NotImplementedError("Must override autoscale in subclasses.")

# 定义 FixedLocator 类，固定位置生成刻度
class FixedLocator(Locator):
    """
    Tick locator that fixes tick locations at specified positions.
    """
    def __init__(self, locs):
        """
        Initialize with a list or array of specific locations.
        """
        self.locs = locs

    def __call__(self):
        """
        Returns the fixed tick locations.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Override autoscale to adjust view limits based on data limits.
        """
        raise NotImplementedError("Must override autoscale in subclasses.")

# 定义 IndexLocator 类，用于索引图中的刻度定位
class IndexLocator(Locator):
    """
    Tick locator that places ticks at specified indices.
    """
    def __init__(self, base, offset=0.0):
        """
        Initialize with a base and an optional offset.
        """
        self.base = base
        self.offset = offset

    def __call__(self):
        """
        Returns a sequence of tick locations based on indices.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Override autoscale to adjust view limits based on data limits.
        """
        raise NotImplementedError("Must override autoscale in subclasses.")

# 定义 NullLocator 类，不生成任何刻度
class NullLocator(Locator):
    """
    Tick locator that does not generate any ticks.
    """
    def __call__(self):
        """
        Returns an empty sequence.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Override autoscale to adjust view limits based on data limits.
        """
        raise NotImplementedError("Must override autoscale in subclasses.")

# 定义 SymmetricalLogLocator 类，用于对称对数刻度定位
class SymmetricalLogLocator(Locator):
    """
    Tick locator for symlog norm, behaves like LogLocator outside the threshold.
    """
    def __init__(self, linthresh):
        """
        Initialize with a linear threshold for the symmetric log scale.
        """
        self.linthresh = linthresh

    def __call__(self):
        """
        Returns a sequence of symmetrical logarithmic tick locations.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Override autoscale to adjust view limits based on data limits.
        """
        raise NotImplementedError("Must override autoscale in subclasses.")

# 定义 AsinhLocator 类，用于 asinh 标准的刻度定位
class AsinhLocator(Locator):
    """
    Tick locator for asinh norm, attempts to space ticks approximately uniformly.
    """
    def __call__(self):
        """
        Returns a sequence of asinh norm tick locations.
        """
        raise NotImplementedError("Must override __call__ in subclasses.")

    def autoscale(self):
        """
        Override autoscale to adjust view limits
# 设置主要刻度的定位器为 xmajor_locator
ax.xaxis.set_major_locator(xmajor_locator)
# 设置次要刻度的定位器为 xminor_locator
ax.xaxis.set_minor_locator(xminor_locator)
# 设置主要刻度的定位器为 ymajor_locator
ax.yaxis.set_major_locator(ymajor_locator)
# 设置次要刻度的定位器为 yminor_locator
ax.yaxis.set_minor_locator(yminor_locator)
# 导入必要的库和模块
import itertools  # 导入 itertools 库，用于生成迭代器的函数
import logging  # 导入 logging 模块，用于记录日志
import locale  # 导入 locale 模块，用于处理地区相关的信息
import math  # 导入 math 模块，提供数学运算函数
from numbers import Integral  # 从 numbers 模块中导入 Integral 类，用于整数类型的基类
import string  # 导入 string 模块，提供字符串相关的函数和常量

import numpy as np  # 导入 NumPy 库，并使用 np 作为别名

import matplotlib as mpl  # 导入 Matplotlib 库，并使用 mpl 作为别名
from matplotlib import _api, cbook  # 从 Matplotlib 中导入 _api 和 cbook 模块
from matplotlib import transforms as mtransforms  # 从 Matplotlib 中导入 transforms 模块，并使用 mtransforms 作为别名

_log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

__all__ = ('TickHelper', 'Formatter', 'FixedFormatter',
           'NullFormatter', 'FuncFormatter', 'FormatStrFormatter',
           'StrMethodFormatter', 'ScalarFormatter', 'LogFormatter',
           'LogFormatterExponent', 'LogFormatterMathtext',
           'LogFormatterSciNotation',
           'LogitFormatter', 'EngFormatter', 'PercentFormatter',
           'Locator', 'IndexLocator', 'FixedLocator', 'NullLocator',
           'LinearLocator', 'LogLocator', 'AutoLocator',
           'MultipleLocator', 'MaxNLocator', 'AutoMinorLocator',
           'SymmetricalLogLocator', 'AsinhLocator', 'LogitLocator')

# 定义一个虚拟的轴对象，用于辅助格式化器类的测试和开发
class _DummyAxis:
    __name__ = "dummy"

    def __init__(self, minpos=0):
        self._data_interval = (0, 1)  # 数据区间，默认为 (0, 1)
        self._view_interval = (0, 1)  # 视图区间，默认为 (0, 1)
        self._minpos = minpos  # 最小正数值，默认为给定的 minpos 参数值

    def get_view_interval(self):
        return self._view_interval  # 返回当前视图区间

    def set_view_interval(self, vmin, vmax):
        self._view_interval = (vmin, vmax)  # 设置视图区间为给定的 (vmin, vmax)

    def get_minpos(self):
        return self._minpos  # 返回最小正数值

    def get_data_interval(self):
        return self._data_interval  # 返回数据区间

    def set_data_interval(self, vmin, vmax):
        self._data_interval = (vmin, vmax)  # 设置数据区间为给定的 (vmin, vmax)

    def get_tick_space(self):
        # 使用经久不衰的默认值 nbins==9 作为刻度空间
        return 9


class TickHelper:
    axis = None  # 初始化 axis 属性为 None

    def set_axis(self, axis):
        self.axis = axis  # 设置 axis 属性为给定的 axis 对象

    def create_dummy_axis(self, **kwargs):
        if self.axis is None:
            self.axis = _DummyAxis(**kwargs)  # 如果 axis 属性为 None，则创建一个新的 _DummyAxis 对象


class Formatter(TickHelper):
    """
    Create a string based on a tick value and location.
    """
    # some classes want to see all the locs to help format
    # individual ones
    locs = []  # 初始化 locs 属性为空列表

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position pos.
        ``pos=None`` indicates an unspecified location.
        """
        raise NotImplementedError('Derived must override')  # 抛出未实现的错误，派生类必须覆盖这个方法

    def format_ticks(self, values):
        """Return the tick labels for all the ticks at once."""
        self.set_locs(values)  # 设置 locs 属性为给定的 values
        return [self(value, i) for i, value in enumerate(values)]  # 返回所有刻度的刻度标签列表

    def format_data(self, value):
        """
        Return the full string representation of the value with the
        position unspecified.
        """
        return self.__call__(value)  # 返回给定值的完整字符串表示形式

    def format_data_short(self, value):
        """
        Return a short string version of the tick value.

        Defaults to the position-independent long value.
        """
        return self.format_data(value)  # 返回给定值的短字符串版本，默认为独立于位置的长值

    def get_offset(self):
        return ''  # 返回空字符串作为偏移量
    # 设置刻度位置的方法，用于存储给定的刻度位置信息
    def set_locs(self, locs):
        """
        Set the locations of the ticks.

        This method is called before computing the tick labels because some
        formatters need to know all tick locations to do so.
        """
        self.locs = locs

    @staticmethod
    # 处理减号字符替换的静态方法，根据配置决定是否替换为 Unicode 的减号符号（U+2212）
    def fix_minus(s):
        """
        Some classes may want to replace a hyphen for minus with the proper
        Unicode symbol (U+2212) for typographical correctness.  This is a
        helper method to perform such a replacement when it is enabled via
        :rc:`axes.unicode_minus`.
        """
        return (s.replace('-', '\N{MINUS SIGN}')
                if mpl.rcParams['axes.unicode_minus']
                else s)

    # 设置刻度定位器的方法，子类可以重写此方法以设置特定的定位器
    def _set_locator(self, locator):
        """Subclasses may want to override this to set a locator."""
        pass
class NullFormatter(Formatter):
    """Always return the empty string."""

    def __call__(self, x, pos=None):
        # docstring inherited
        # 返回空字符串，用于格式化标签
        return ''


class FixedFormatter(Formatter):
    """
    Return fixed strings for tick labels based only on position, not value.

    .. note::
        `.FixedFormatter` should only be used together with `.FixedLocator`.
        Otherwise, the labels may end up in unexpected positions.
    """

    def __init__(self, seq):
        """Set the sequence *seq* of strings that will be used for labels."""
        # 初始化固定的字符串序列，用于标签
        self.seq = seq
        self.offset_string = ''

    def __call__(self, x, pos=None):
        """
        Return the label that matches the position, regardless of the value.

        For positions ``pos < len(seq)``, return ``seq[i]`` regardless of
        *x*. Otherwise return empty string. ``seq`` is the sequence of
        strings that this object was initialized with.
        """
        if pos is None or pos >= len(self.seq):
            # 如果位置为空或超出序列长度，返回空字符串
            return ''
        else:
            # 根据位置返回对应的固定字符串标签
            return self.seq[pos]

    def get_offset(self):
        # 获取偏移量字符串
        return self.offset_string

    def set_offset_string(self, ofs):
        # 设置偏移量字符串
        self.offset_string = ofs


class FuncFormatter(Formatter):
    """
    Use a user-defined function for formatting.

    The function should take in two inputs (a tick value ``x`` and a
    position ``pos``), and return a string containing the corresponding
    tick label.
    """

    def __init__(self, func):
        # 初始化使用者定义的格式化函数
        self.func = func
        self.offset_string = ""

    def __call__(self, x, pos=None):
        """
        Return the value of the user defined function.

        *x* and *pos* are passed through as-is.
        """
        # 调用用户定义的函数来返回标签
        return self.func(x, pos)

    def get_offset(self):
        # 获取偏移量字符串
        return self.offset_string

    def set_offset_string(self, ofs):
        # 设置偏移量字符串
        self.offset_string = ofs


class FormatStrFormatter(Formatter):
    """
    Use an old-style ('%' operator) format string to format the tick.

    The format string should have a single variable format (%) in it.
    It will be applied to the value (not the position) of the tick.

    Negative numeric values (e.g., -1) will use a dash, not a Unicode minus;
    use mathtext to get a Unicode minus by wrapping the format specifier with $
    (e.g. "$%g$").
    """

    def __init__(self, fmt):
        # 初始化旧式格式化字符串来格式化刻度
        self.fmt = fmt

    def __call__(self, x, pos=None):
        """
        Return the formatted label string.

        Only the value *x* is formatted. The position is ignored.
        """
        # 返回格式化后的标签字符串，仅格式化值 *x*，位置被忽略
        return self.fmt % x


class _UnicodeMinusFormat(string.Formatter):
    """
    A specialized string formatter so that `.StrMethodFormatter` respects
    :rc:`axes.unicode_minus`.  This implementation relies on the fact that the
    format string is only ever called with kwargs *x* and *pos*, so it blindly
    replaces dashes by unicode minuses without further checking.
    """
    # 定义一个方法 `format_field`，用于格式化字段
    def format_field(self, value, format_spec):
        # 调用父类的 `format_field` 方法，对值进行格式化处理，并修复负号的显示
        return Formatter.fix_minus(super().format_field(value, format_spec))
# 自定义 Formatter 类，继承自 Formatter 类
class StrMethodFormatter(Formatter):
    """
    Use a new-style format string (as used by `str.format`) to format the tick.

    The field used for the tick value must be labeled *x* and the field used
    for the tick position must be labeled *pos*.

    The formatter will respect :rc:`axes.unicode_minus` when formatting
    negative numeric values.

    It is typically unnecessary to explicitly construct `.StrMethodFormatter`
    objects, as `~.Axis.set_major_formatter` directly accepts the format string
    itself.
    """

    # 初始化方法，接受一个格式化字符串作为参数
    def __init__(self, fmt):
        self.fmt = fmt

    # 调用实例时调用的方法，返回格式化后的标签字符串
    def __call__(self, x, pos=None):
        """
        Return the formatted label string.

        *x* and *pos* are passed to `str.format` as keyword arguments
        with those exact names.
        """
        # 使用 _UnicodeMinusFormat 类来格式化字符串，传入格式化字符串和 x, pos 参数
        return _UnicodeMinusFormat().format(self.fmt, x=x, pos=pos)


# 另一个自定义 Formatter 类，继承自 Formatter 类
class ScalarFormatter(Formatter):
    """
    Format tick values as a number.

    Parameters
    ----------
    useOffset : bool or float, default: :rc:`axes.formatter.useoffset`
        Whether to use offset notation. See `.set_useOffset`.
    useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
        Whether to use fancy math formatting. See `.set_useMathText`.
    useLocale : bool, default: :rc:`axes.formatter.use_locale`.
        Whether to use locale settings for decimal sign and positive sign.
        See `.set_useLocale`.

    Notes
    -----
    In addition to the parameters above, the formatting of scientific vs.
    floating point representation can be configured via `.set_scientific`
    and `.set_powerlimits`).

    **Offset notation and scientific notation**

    Offset notation and scientific notation look quite similar at first sight.
    Both split some information from the formatted tick values and display it
    at the end of the axis.

    - The scientific notation splits up the order of magnitude, i.e. a
      multiplicative scaling factor, e.g. ``1e6``.

    - The offset notation separates an additive constant, e.g. ``+1e6``. The
      offset notation label is always prefixed with a ``+`` or ``-`` sign
      and is thus distinguishable from the order of magnitude label.

    The following plot with x limits ``1_000_000`` to ``1_000_010`` illustrates
    the different formatting. Note the labels at the right edge of the x axis.

    .. plot::

        lim = (1_000_000, 1_000_010)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'hspace': 2})
        ax1.set(title='offset_notation', xlim=lim)
        ax2.set(title='scientific notation', xlim=lim)
        ax2.xaxis.get_major_formatter().set_useOffset(False)
        ax3.set(title='floating point notation', xlim=lim)
        ax3.xaxis.get_major_formatter().set_useOffset(False)
        ax3.xaxis.get_major_formatter().set_scientific(False)
    """
    def __init__(self, useOffset=None, useMathText=None, useLocale=None):
        # 初始化函数，设置对象的初始属性值
        if useOffset is None:
            # 如果未指定 useOffset 参数，则使用默认值 axes.formatter.useoffset
            useOffset = mpl.rcParams['axes.formatter.useoffset']
        # 设置偏移阈值为 axes.formatter.offset_threshold
        self._offset_threshold = \
            mpl.rcParams['axes.formatter.offset_threshold']
        # 设置 useOffset 属性
        self.set_useOffset(useOffset)
        # 设置是否使用 LaTeX 渲染文本的属性
        self._usetex = mpl.rcParams['text.usetex']
        # 设置是否使用数学文本的属性
        self.set_useMathText(useMathText)
        # 设置数量级为 0
        self.orderOfMagnitude = 0
        # 设置格式为空字符串
        self.format = ''
        # 默认启用科学计数法
        self._scientific = True
        # 设置科学计数法的限制
        self._powerlimits = mpl.rcParams['axes.formatter.limits']
        # 设置是否使用地区设置的属性
        self.set_useLocale(useLocale)

    def get_useOffset(self):
        """
        Return whether automatic mode for offset notation is active.

        This returns True if ``set_useOffset(True)``; it returns False if an
        explicit offset was set, e.g. ``set_useOffset(1000)``.

        See Also
        --------
        ScalarFormatter.set_useOffset
        """
        # 返回当前是否使用偏移符号的状态
        return self._useOffset

    def set_useOffset(self, val):
        """
        Set whether to use offset notation.

        When formatting a set numbers whose value is large compared to their
        range, the formatter can separate an additive constant. This can
        shorten the formatted numbers so that they are less likely to overlap
        when drawn on an axis.

        Parameters
        ----------
        val : bool or float
            - If False, do not use offset notation.
            - If True (=automatic mode), use offset notation if it can make
              the residual numbers significantly shorter. The exact behavior
              is controlled by :rc:`axes.formatter.offset_threshold`.
            - If a number, force an offset of the given value.

        Examples
        --------
        With active offset notation, the values

        ``100_000, 100_002, 100_004, 100_006, 100_008``

        will be formatted as ``0, 2, 4, 6, 8`` plus an offset ``+1e5``, which
        is written to the edge of the axis.
        """
        # 设置是否使用偏移符号的属性值
        if val in [True, False]:
            self.offset = 0
            self._useOffset = val
        else:
            self._useOffset = False
            self.offset = val

    useOffset = property(fget=get_useOffset, fset=set_useOffset)

    def get_useLocale(self):
        """
        Return whether locale settings are used for formatting.

        See Also
        --------
        ScalarFormatter.set_useLocale
        """
        # 返回当前是否使用地区设置的状态
        return self._useLocale

    def set_useLocale(self, val):
        """
        Set whether to use locale settings for decimal sign and positive sign.

        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_locale`.
        """
        # 设置是否使用地区设置的属性值
        if val is None:
            self._useLocale = mpl.rcParams['axes.formatter.use_locale']
        else:
            self._useLocale = val

    useLocale = property(fget=get_useLocale, fset=set_useLocale)
    def _format_maybe_minus_and_locale(self, fmt, arg):
        """
        Format *arg* with *fmt*, applying Unicode minus and locale if desired.
        """
        return self.fix_minus(
                # 如果使用数学文本，则需要转义由 locale.format_string 引入的逗号
                # 但是不需要转义 fmt 中最开始就存在的逗号。
                (",".join(locale.format_string(part, (arg,), True).replace(",", "{,}")
                          for part in fmt.split(",")) if self._useMathText
                 else locale.format_string(fmt, (arg,), True))
                if self._useLocale  # 如果启用了本地化处理
                else fmt % arg)  # 否则直接使用格式化字符串 fmt 处理参数 arg

    def get_useMathText(self):
        """
        Return whether to use fancy math formatting.

        See Also
        --------
        ScalarFormatter.set_useMathText
        """
        return self._useMathText

    def set_useMathText(self, val):
        r"""
        Set whether to use fancy math formatting.

        If active, scientific notation is formatted as :math:`1.2 \times 10^3`.

        Parameters
        ----------
        val : bool or None
            *None* resets to :rc:`axes.formatter.use_mathtext`.
        """
        if val is None:
            self._useMathText = mpl.rcParams['axes.formatter.use_mathtext']
            if self._useMathText is False:
                try:
                    from matplotlib import font_manager
                    ufont = font_manager.findfont(
                        font_manager.FontProperties(
                            mpl.rcParams["font.family"]
                        ),
                        fallback_to_default=False,
                    )
                except ValueError:
                    ufont = None

                if ufont == str(cbook._get_data_path("fonts/ttf/cmr10.ttf")):
                    _api.warn_external(
                        "cmr10 font should ideally be used with "
                        "mathtext, set axes.formatter.use_mathtext to True"
                    )
        else:
            self._useMathText = val

    useMathText = property(fget=get_useMathText, fset=set_useMathText)

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x* at position *pos*.
        """
        if len(self.locs) == 0:
            return ''
        else:
            xp = (x - self.offset) / (10. ** self.orderOfMagnitude)
            if abs(xp) < 1e-8:
                xp = 0
            return self._format_maybe_minus_and_locale(self.format, xp)

    def set_scientific(self, b):
        """
        Turn scientific notation on or off.

        See Also
        --------
        ScalarFormatter.set_powerlimits
        """
        self._scientific = bool(b)
    def set_powerlimits(self, lims):
        r"""
        Set size thresholds for scientific notation.

        Parameters
        ----------
        lims : (int, int)
            A tuple *(min_exp, max_exp)* containing the powers of 10 that
            determine the switchover threshold. For a number representable as
            :math:`a \times 10^\mathrm{exp}` with :math:`1 <= |a| < 10`,
            scientific notation will be used if ``exp <= min_exp`` or
            ``exp >= max_exp``.

            The default limits are controlled by :rc:`axes.formatter.limits`.

            In particular numbers with *exp* equal to the thresholds are
            written in scientific notation.

            Typically, *min_exp* will be negative and *max_exp* will be
            positive.

            For example, ``formatter.set_powerlimits((-3, 4))`` will provide
            the following formatting:
            :math:`1 \times 10^{-3}, 9.9 \times 10^{-3}, 0.01,`
            :math:`9999, 1 \times 10^4`.

        See Also
        --------
        ScalarFormatter.set_scientific
        """
        # 检查参数 lims 是否为长度为 2 的序列，否则抛出 ValueError 异常
        if len(lims) != 2:
            raise ValueError("'lims' must be a sequence of length 2")
        # 将参数 lims 赋值给对象的 _powerlimits 属性
        self._powerlimits = lims

    def format_data_short(self, value):
        # docstring inherited
        # 如果 value 是 np.ma.masked，则返回空字符串
        if value is np.ma.masked:
            return ""
        # 如果 value 是整数类型，使用 "%d" 格式化
        if isinstance(value, Integral):
            fmt = "%d"
        else:
            # 如果 value 不是整数
            # 检查 self.axis 是否具有 "__name__" 属性，并且其值是 ["xaxis", "yaxis"] 中的一个
            if getattr(self.axis, "__name__", "") in ["xaxis", "yaxis"]:
                # 如果 self.axis 的名称是 "xaxis"
                if self.axis.__name__ == "xaxis":
                    # 获取 x 轴的转换和反转换对象
                    axis_trf = self.axis.axes.get_xaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    # 将 value 映射到屏幕坐标系上
                    screen_xy = axis_trf.transform((value, 0))
                    # 计算邻近值的数据坐标
                    neighbor_values = axis_inv_trf.transform(
                        screen_xy + [[-1, 0], [+1, 0]])[:, 0]
                else:  # 如果 self.axis 的名称是 "yaxis"
                    # 获取 y 轴的转换和反转换对象
                    axis_trf = self.axis.axes.get_yaxis_transform()
                    axis_inv_trf = axis_trf.inverted()
                    # 将 value 映射到屏幕坐标系上
                    screen_xy = axis_trf.transform((0, value))
                    # 计算邻近值的数据坐标
                    neighbor_values = axis_inv_trf.transform(
                        screen_xy + [[0, -1], [0, +1]])[:, 1]
                # 计算邻近值与 value 的最大差距
                delta = abs(neighbor_values - value).max()
            else:
                # 粗略估计：不超过 1e4 的划分
                a, b = self.axis.get_view_interval()
                delta = (b - a) / 1e4
            # 使用 cbook._g_sig_digits(value, delta) 计算有效数字，并生成格式字符串
            fmt = f"%-#.{cbook._g_sig_digits(value, delta)}g"
        # 返回格式化后的字符串
        return self._format_maybe_minus_and_locale(fmt, value)
    def format_data(self, value):
        # 返回科学计数法的格式化字符串，包括指数
        e = math.floor(math.log10(abs(value)))  # 计算数值的指数部分
        s = round(value / 10**e, 10)  # 计算数值的尾数部分
        significand = self._format_maybe_minus_and_locale(
            "%d" if s % 1 == 0 else "%1.10g", s)  # 格式化尾数，考虑是否有负号和本地化设置
        if e == 0:
            return significand  # 如果指数为0，直接返回尾数部分
        exponent = self._format_maybe_minus_and_locale("%d", e)  # 格式化指数部分
        if self._useMathText or self._usetex:
            exponent = "10^{%s}" % exponent  # 如果使用数学文本或LaTeX，格式化为10的幂形式
            return (exponent if s == 1  # 如果尾数为1，则重新格式化为10的幂形式
                    else rf"{significand} \times {exponent}")  # 否则返回尾数乘以10的幂形式
        else:
            return f"{significand}e{exponent}"  # 否则返回科学计数法表示的字符串形式

    def get_offset(self):
        """
        Return scientific notation, plus offset.
        """
        if len(self.locs) == 0:
            return ''  # 如果locs为空，则返回空字符串
        if self.orderOfMagnitude or self.offset:
            offsetStr = ''  # 初始化偏移量字符串
            sciNotStr = ''  # 初始化科学计数法字符串
            if self.offset:
                offsetStr = self.format_data(self.offset)  # 格式化偏移量
                if self.offset > 0:
                    offsetStr = '+' + offsetStr  # 如果偏移量大于0，添加正号
            if self.orderOfMagnitude:
                if self._usetex or self._useMathText:
                    sciNotStr = self.format_data(10 ** self.orderOfMagnitude)  # 格式化科学计数法
                else:
                    sciNotStr = '1e%d' % self.orderOfMagnitude  # 否则返回普通格式的科学计数法字符串
            if self._useMathText or self._usetex:
                if sciNotStr != '':
                    sciNotStr = r'\times\mathdefault{%s}' % sciNotStr  # 如果使用数学文本或LaTeX，添加乘号
                s = fr'${sciNotStr}\mathdefault{{{offsetStr}}}$'  # 格式化最终字符串
            else:
                s = ''.join((sciNotStr, offsetStr))  # 否则直接连接两个部分
            return self.fix_minus(s)  # 返回修正负号的字符串
        return ''  # 如果不需要显示科学计数法和偏移量，则返回空字符串

    def set_locs(self, locs):
        # 继承的文档字符串
        self.locs = locs  # 设置本地位置
        if len(self.locs) > 0:
            if self._useOffset:
                self._compute_offset()  # 如果使用偏移量，计算偏移量
            self._set_order_of_magnitude()  # 设置数量级
            self._set_format()  # 设置格式化
    def _compute_offset(self):
        # 获取当前轴上可见刻度的范围
        vmin, vmax = sorted(self.axis.get_view_interval())
        
        # 将刻度转换为 NumPy 数组
        locs = np.asarray(self.locs)
        
        # 仅保留在视图范围内的刻度位置
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        
        # 如果没有可见的刻度，则偏移量为 0 并返回
        if not len(locs):
            self.offset = 0
            return
        
        # 获取最小和最大刻度值
        lmin, lmax = locs.min(), locs.max()
        
        # 如果最小值等于最大值，或者最小值小于等于 0 且最大值大于等于 0，则偏移量为 0 并返回
        if lmin == lmax or lmin <= 0 <= lmax:
            self.offset = 0
            return
        
        # 比较最小和最大刻度值的绝对值
        abs_min, abs_max = sorted([abs(float(lmin)), abs(float(lmax))])
        
        # 确定刻度值的符号
        sign = math.copysign(1, lmin)
        
        # 找到使得 abs_min 和 abs_max 在给定精度上相等的最小十的幂次
        oom_max = np.ceil(math.log10(abs_max))
        oom = 1 + next(oom for oom in itertools.count(oom_max, -1)
                       if abs_min // 10 ** oom != abs_max // 10 ** oom)
        
        # 如果 (abs_max - abs_min) / 10 ** oom 小于等于 1e-2，则处理跨越大的十的倍数的情况
        if (abs_max - abs_min) / 10 ** oom <= 1e-2:
            oom = 1 + next(oom for oom in itertools.count(oom_max, -1)
                           if abs_max // 10 ** oom - abs_min // 10 ** oom > 1)
        
        # 仅当偏移量至少能够节省 _offset_threshold 数字时才使用偏移量
        n = self._offset_threshold - 1
        self.offset = (sign * (abs_max // 10 ** oom) * 10 ** oom
                       if abs_max // 10 ** oom >= 10**n
                       else 0)
    def _set_order_of_magnitude(self):
        # 如果需要使用科学计数法，则找到合适的指数
        # 如果使用数值偏移量，则在应用偏移后找到指数
        # 当下限与上限相等且不为零时，使用提供的指数。
        if not self._scientific:
            self.orderOfMagnitude = 0
            return

        if self._powerlimits[0] == self._powerlimits[1] != 0:
            # 当下限与上限相等且不为零时，使用固定的缩放因子
            self.orderOfMagnitude = self._powerlimits[0]
            return

        # 限制在可见刻度范围内
        vmin, vmax = sorted(self.axis.get_view_interval())
        locs = np.asarray(self.locs)
        locs = locs[(vmin <= locs) & (locs <= vmax)]
        locs = np.abs(locs)

        if not len(locs):
            self.orderOfMagnitude = 0
            return

        if self.offset:
            # 计算最大值与最小值之差的对数，向下取整作为指数
            oom = math.floor(math.log10(vmax - vmin))
        else:
            # 找到刻度的最大值，如果最大值为零，则指数为零，否则取其对数向下取整作为指数
            val = locs.max()
            if val == 0:
                oom = 0
            else:
                oom = math.floor(math.log10(val))

        # 根据计算得到的指数与设定的下限和上限比较，确定最终的指数值
        if oom <= self._powerlimits[0]:
            self.orderOfMagnitude = oom
        elif oom >= self._powerlimits[1]:
            self.orderOfMagnitude = oom
        else:
            self.orderOfMagnitude = 0

    def _set_format(self):
        # 设置格式字符串以格式化所有的刻度标签
        if len(self.locs) < 2:
            # 临时将位置数组扩展为包含轴端点
            _locs = [*self.locs, *self.axis.get_view_interval()]
        else:
            _locs = self.locs

        # 根据偏移量和指数对位置进行调整
        locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
        loc_range = np.ptp(locs)

        # 曲线坐标可能会产生两个相同的点
        if loc_range == 0:
            loc_range = np.max(np.abs(locs))

        # 如果两个点都为零，则取默认值为1
        if loc_range == 0:
            loc_range = 1

        if len(self.locs) < 2:
            # 仅需要端点来计算 loc_range
            locs = locs[:-2]

        # 计算 loc_range 的对数，并转换为整数
        loc_range_oom = int(math.floor(math.log10(loc_range)))

        # 第一次估计：确定有效数字的个数
        sigfigs = max(0, 3 - loc_range_oom)

        # 细化估计：通过比较阈值来确定最终的有效数字个数
        thresh = 1e-3 * 10 ** loc_range_oom
        while sigfigs >= 0:
            if np.abs(locs - np.round(locs, decimals=sigfigs)).max() < thresh:
                sigfigs -= 1
            else:
                break

        sigfigs += 1

        # 根据计算得到的有效数字个数设置格式化字符串
        self.format = f'%1.{sigfigs}f'

        # 如果使用了 LaTeX 或 MathText，则将格式化字符串进行进一步的处理
        if self._usetex or self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format
class LogFormatter(Formatter):
    """
    Base class for formatting ticks on a log or symlog scale.

    It may be instantiated directly, or subclassed.

    Parameters
    ----------
    base : float, default: 10.
        Base of the logarithm used in all calculations.

    labelOnlyBase : bool, default: False
        If True, label ticks only at integer powers of base.
        This is normally True for major ticks and False for
        minor ticks.

    minor_thresholds : (subset, all), default: (1, 0.4)
        If labelOnlyBase is False, these two numbers control
        the labeling of ticks that are not at integer powers of
        base; normally these are the minor ticks. The controlling
        parameter is the log of the axis data range.  In the typical
        case where base is 10 it is the number of decades spanned
        by the axis, so we can call it 'numdec'. If ``numdec <= all``,
        all minor ticks will be labeled.  If ``all < numdec <= subset``,
        then only a subset of minor ticks will be labeled, so as to
        avoid crowding. If ``numdec > subset`` then no minor ticks will
        be labeled.

    linthresh : None or float, default: None
        If a symmetric log scale is in use, its ``linthresh``
        parameter must be supplied here.

    Notes
    -----
    The `set_locs` method must be called to enable the subsetting
    logic controlled by the ``minor_thresholds`` parameter.

    In some cases such as the colorbar, there is no distinction between
    major and minor ticks; the tick locations might be set manually,
    or by a locator that puts ticks at integer powers of base and
    at intermediate locations.  For this situation, disable the
    minor_thresholds logic by using ``minor_thresholds=(np.inf, np.inf)``,
    so that all ticks will be labeled.

    To disable labeling of minor ticks when 'labelOnlyBase' is False,
    use ``minor_thresholds=(0, 0)``.  This is the default for the
    "classic" style.

    Examples
    --------
    To label a subset of minor ticks when the view limits span up
    to 2 decades, and all of the ticks when zoomed in to 0.5 decades
    or less, use ``minor_thresholds=(2, 0.5)``.

    To label all minor ticks when the view limits span up to 1.5
    decades, use ``minor_thresholds=(1.5, 1.5)``.
    """

    def __init__(self, base=10.0, labelOnlyBase=False,
                 minor_thresholds=None,
                 linthresh=None):
        # 调用父类的初始化方法，设定基数（base）
        self.set_base(base)
        # 设置是否仅在基数的整数倍位置标记主要刻度线
        self.set_label_minor(labelOnlyBase)
        # 如果未提供 minor_thresholds 参数，则根据 classic_mode 设置默认值
        if minor_thresholds is None:
            if mpl.rcParams['_internal.classic_mode']:
                minor_thresholds = (0, 0)
            else:
                minor_thresholds = (1, 0.4)
        # 设置次要刻度线标记的阈值
        self.minor_thresholds = minor_thresholds
        # 初始化子标签和对数线性阈值
        self._sublabels = None
        self._linthresh = linthresh
    def set_base(self, base):
        """
        Change the *base* for labeling.

        .. warning::
           Should always match the base used for :class:`LogLocator`
        """
        # 设置用于标签的基数，将输入的base转换为浮点数并赋给实例变量self._base
        self._base = float(base)

    def set_label_minor(self, labelOnlyBase):
        """
        Switch minor tick labeling on or off.

        Parameters
        ----------
        labelOnlyBase : bool
            If True, label ticks only at integer powers of base.
        """
        # 设置是否只在基数的整数幂处标记次要刻度线的标签
        self.labelOnlyBase = labelOnlyBase

    def set_locs(self, locs=None):
        """
        Use axis view limits to control which ticks are labeled.

        The *locs* parameter is ignored in the present algorithm.
        """
        if np.isinf(self.minor_thresholds[0]):
            # 如果最小次要阈值是无穷大，则将子标签设置为None并返回
            self._sublabels = None
            return

        # 处理symlog情况：
        linthresh = self._linthresh
        if linthresh is None:
            try:
                linthresh = self.axis.get_transform().linthresh
            except AttributeError:
                pass

        # 获取轴的视图间隔
        vmin, vmax = self.axis.get_view_interval()
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        if linthresh is None and vmin <= 0:
            # 可能是一个带有格式参数的颜色条，设置了一个LogFormatter的方式，但现在不再适用。
            # 在这种情况下，将子标签设置为{1}，即标记基数的幂
            self._sublabels = {1}
            return

        b = self._base
        if linthresh is not None:  # symlog
            # 计算对数部分轴上的十进制数目
            numdec = 0
            if vmin < -linthresh:
                rhs = min(vmax, -linthresh)
                numdec += math.log(vmin / rhs) / math.log(b)
            if vmax > linthresh:
                lhs = max(vmin, linthresh)
                numdec += math.log(vmax / lhs) / math.log(b)
        else:
            # 计算轴上的十进制数目
            vmin = math.log(vmin) / math.log(b)
            vmax = math.log(vmax) / math.log(b)
            numdec = abs(vmax - vmin)

        if numdec > self.minor_thresholds[0]:
            # 仅标记基数的幂
            self._sublabels = {1}
        elif numdec > self.minor_thresholds[1]:
            # 在对数空间系数之间添加标签；
            # 包括基数幂以确保位置包括“主要”和“次要”点，如在颜色条中
            c = np.geomspace(1, b, int(b)//2 + 1)
            self._sublabels = set(np.round(c))
            # 对于基数为10，这会产生(1, 2, 3, 4, 6, 10)的结果。
        else:
            # 标记所有基数的整数倍
            self._sublabels = set(np.arange(1, b + 1))

    def _num_to_string(self, x, vmin, vmax):
        if x > 10000:
            s = '%1.0e' % x
        elif x < 1:
            s = '%1.0e' % x
        else:
            # 调用_pprint_val方法将数值x转换为字符串s，传递的参数是vmax - vmin
            s = self._pprint_val(x, vmax - vmin)
        return s
    def __call__(self, x, pos=None):
        # 如果输入值为0.0，则返回字符串'0'，用于对应Symlog情况
        if x == 0.0:  # Symlog
            return '0'

        x = abs(x)
        b = self._base
        # 计算以指定基数 b 为底时，x 的对数值除以 b 的对数值，用于确定是否为整数的十倍
        fx = math.log(x) / math.log(b)
        is_x_decade = _is_close_to_int(fx)
        exponent = round(fx) if is_x_decade else np.floor(fx)
        coeff = round(b ** (fx - exponent))

        # 如果设置了仅标记基数且 x 不是十倍数，则返回空字符串
        if self.labelOnlyBase and not is_x_decade:
            return ''
        # 如果设置了子标签且 coeff 不在子标签中，则返回空字符串
        if self._sublabels is not None and coeff not in self._sublabels:
            return ''

        # 获取轴的视图间隔 vmin 和 vmax，并确保它们不是奇异值
        vmin, vmax = self.axis.get_view_interval()
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)
        # 将 x 转换为字符串格式化后返回，考虑了数值可能的负号问题
        s = self._num_to_string(x, vmin, vmax)
        return self.fix_minus(s)

    def format_data(self, value):
        # 使用上下文管理器设置 labelOnlyBase 为 False，调用 __call__ 方法处理数据并返回结果
        with cbook._setattr_cm(self, labelOnlyBase=False):
            return cbook.strip_math(self.__call__(value))

    def format_data_short(self, value):
        # docstring inherited
        # 将 value 格式化为长度为 12 的字符串返回，右侧删除空格
        return ('%-12g' % value).rstrip()

    def _pprint_val(self, x, d):
        # 如果数值 x 绝对值小于 1e4 且为整数，则格式化为整数字符串返回
        if abs(x) < 1e4 and x == int(x):
            return '%d' % x
        # 根据数值大小 d 选择适当的格式化字符串 fmt
        fmt = ('%1.3e' if d < 1e-2 else
               '%1.3f' if d <= 1 else
               '%1.2f' if d <= 10 else
               '%1.1f' if d <= 1e5 else
               '%1.1e')
        s = fmt % x
        # 分割科学计数法表示的字符串 s
        tup = s.split('e')
        if len(tup) == 2:
            mantissa = tup[0].rstrip('0').rstrip('.')
            exponent = int(tup[1])
            # 如果指数不为零，则组合成科学计数法格式返回；否则仅返回尾数部分
            if exponent:
                s = '%se%d' % (mantissa, exponent)
            else:
                s = mantissa
        else:
            s = s.rstrip('0').rstrip('.')
        return s
class LogFormatterExponent(LogFormatter):
    """
    Format values for log axis using ``exponent = log_base(value)``.
    """

    def _num_to_string(self, x, vmin, vmax):
        # Calculate the logarithm of x with respect to the specified base
        fx = math.log(x) / math.log(self._base)
        
        # Determine the string representation based on the magnitude of fx
        if abs(fx) > 10000:
            s = '%1.0g' % fx  # Use scientific notation for very large values
        elif abs(fx) < 1:
            s = '%1.0g' % fx  # Use scientific notation for very small values
        else:
            # Calculate the number of decades and format the value accordingly
            fd = math.log(vmax - vmin) / math.log(self._base)
            s = self._pprint_val(fx, fd)
        
        return s


class LogFormatterMathtext(LogFormatter):
    """
    Format values for log axis using ``exponent = log_base(value)``.
    """

    def _non_decade_format(self, sign_string, base, fx, usetex):
        """Return string for non-decade locations."""
        # Format the label for non-decade locations using LaTeX math notation
        return r'$\mathdefault{%s%s^{%.2f}}$' % (sign_string, base, fx)

    def __call__(self, x, pos=None):
        # docstring inherited
        if x == 0:  # Special case for zero
            return r'$\mathdefault{0}$'

        sign_string = '-' if x < 0 else ''  # Determine the sign of x
        x = abs(x)
        b = self._base  # Get the base for logarithm computation

        # Calculate the logarithm of x with respect to the specified base
        fx = math.log(x) / math.log(b)

        # Check if x is a decade and round the exponent accordingly
        is_x_decade = _is_close_to_int(fx)
        exponent = round(fx) if is_x_decade else np.floor(fx)
        
        # Calculate the coefficient based on the base and exponent
        coeff = round(b ** (fx - exponent))

        # Handle special cases based on formatter settings and magnitude of fx
        if self.labelOnlyBase and not is_x_decade:
            return ''  # Return empty string if only labeling bases and not a decade
        if self._sublabels is not None and coeff not in self._sublabels:
            return ''  # Return empty string if coefficient is not in sublabels

        if is_x_decade:
            fx = round(fx)  # Round fx if it represents a decade

        # Format the base as string considering whether it's an integer or not
        if b % 1 == 0.0:
            base = '%d' % b
        else:
            base = '%s' % b

        # Determine the format based on the magnitude of fx and settings
        if abs(fx) < mpl.rcParams['axes.formatter.min_exponent']:
            return r'$\mathdefault{%s%g}$' % (sign_string, x)  # Scientific notation for small fx
        elif not is_x_decade:
            usetex = mpl.rcParams['text.usetex']
            return self._non_decade_format(sign_string, base, fx, usetex)  # Non-decade format
        else:
            return r'$\mathdefault{%s%s^{%d}}$' % (sign_string, base, fx)  # Decade format


class LogFormatterSciNotation(LogFormatterMathtext):
    """
    Format values following scientific notation in a logarithmic axis.
    """

    def _non_decade_format(self, sign_string, base, fx, usetex):
        """Return string for non-decade locations."""
        b = float(base)  # Convert base to float for calculations
        exponent = math.floor(fx)  # Compute the floor of fx for the exponent
        coeff = b ** (fx - exponent)  # Calculate the coefficient

        # Round the coefficient if it's close to an integer
        if _is_close_to_int(coeff):
            coeff = round(coeff)

        # Format the string using LaTeX math notation
        return r'$\mathdefault{%s%g\times%s^{%d}}$' \
            % (sign_string, coeff, base, exponent)


class LogitFormatter(Formatter):
    """
    Probability formatter (using Math text).
    """

    def __init__(
        self,
        *,
        use_overline=False,
        one_half=r"\frac{1}{2}",
        minor=False,
        minor_threshold=25,
        minor_number=6,
    ):
        r"""
        Parameters
        ----------
        use_overline : bool, default: False
            If x > 1/2, with x = 1-v, indicate if x should be displayed as
            $\overline{v}$. The default is to display $1-v$.

        one_half : str, default: r"\frac{1}{2}"
            The string used to represent 1/2.

        minor : bool, default: False
            Indicate if the formatter is formatting minor ticks or not.
            Basically minor ticks are not labelled, except when only few ticks
            are provided, ticks with most space with neighbor ticks are
            labelled. See other parameters to change the default behavior.

        minor_threshold : int, default: 25
            Maximum number of locs for labelling some minor ticks. This
            parameter have no effect if minor is False.

        minor_number : int, default: 6
            Number of ticks which are labelled when the number of ticks is
            below the threshold.
        """
        # 初始化函数，用于设置格式化器的各项参数
        self._use_overline = use_overline
        self._one_half = one_half
        self._minor = minor
        self._labelled = set()  # 初始化一个空集合，用于记录已经标记的刻度
        self._minor_threshold = minor_threshold  # 设置最大次要刻度标记数的阈值
        self._minor_number = minor_number  # 设置当次要刻度较少时需要标记的刻度数目

    def use_overline(self, use_overline):
        r"""
        Switch display mode with overline for labelling p>1/2.

        Parameters
        ----------
        use_overline : bool, default: False
            If x > 1/2, with x = 1-v, indicate if x should be displayed as
            $\overline{v}$. The default is to display $1-v$.
        """
        # 设置是否使用上划线模式显示大于1/2的标签
        self._use_overline = use_overline

    def set_one_half(self, one_half):
        r"""
        Set the way one half is displayed.

        one_half : str, default: r"\frac{1}{2}"
            The string used to represent 1/2.
        """
        # 设置显示1/2的方式的字符串表示
        self._one_half = one_half

    def set_minor_threshold(self, minor_threshold):
        """
        Set the threshold for labelling minors ticks.

        Parameters
        ----------
        minor_threshold : int
            Maximum number of locations for labelling some minor ticks. This
            parameter have no effect if minor is False.
        """
        # 设置标记次要刻度的阈值
        self._minor_threshold = minor_threshold

    def set_minor_number(self, minor_number):
        """
        Set the number of minor ticks to label when some minor ticks are
        labelled.

        Parameters
        ----------
        minor_number : int
            Number of ticks which are labelled when the number of ticks is
            below the threshold.
        """
        # 设置少量次要刻度标记时需要标记的刻度数目
        self._minor_number = minor_number
    # 设置对象的位置信息，将给定的位置数组转换为 NumPy 数组并存储在 self.locs 中
    def set_locs(self, locs):
        self.locs = np.array(locs)
        # 清空已标记的位置信息
        self._labelled.clear()

        # 如果不是次要刻度，则直接返回 None
        if not self._minor:
            return None

        # 检查是否所有位置都是十分之一或者其补数，或者是 2*x 是整数且为 1 的情况
        if all(
            _is_decade(x, rtol=1e-7)
            or _is_decade(1 - x, rtol=1e-7)
            or (_is_close_to_int(2 * x) and
                int(np.round(2 * x)) == 1)
            for x in locs
        ):
            # 次要刻度是从理想值中采样的，因此不需要标签
            return None

        # 如果位置数量小于次要刻度的阈值，则考虑标记其中一些位置
        if len(locs) < self._minor_threshold:
            if len(locs) < self._minor_number:
                # 如果位置数量少于指定的次要刻度数目，则将这些位置添加到已标记集合中
                self._labelled.update(locs)
            else:
                # 如果次要刻度不多，只显示少数几个十年，选择一些（间隔的）次要刻度进行标记。
                # 只有次要刻度是已知的，我们假设选择显示哪些刻度足够。
                # 对于每个刻度，计算其与前一个刻度和后一个刻度的距离。选择具有最小距离的刻度。
                # 在距离相同的情况下，选择总距离最小的刻度。
                diff = np.diff(-np.log(1 / self.locs - 1))
                space_pessimistic = np.minimum(
                    np.concatenate(((np.inf,), diff)),
                    np.concatenate((diff, (np.inf,))),
                )
                space_sum = (
                    np.concatenate(((0,), diff))
                    + np.concatenate((diff, (0,)))
                )
                # 选择最好的次要刻度，即距离最小的刻度
                good_minor = sorted(
                    range(len(self.locs)),
                    key=lambda i: (space_pessimistic[i], space_sum[i]),
                )[-self._minor_number:]
                self._labelled.update(locs[i] for i in good_minor)

    # 格式化数值 x，根据给定的位置信息 locs，支持科学计数法
    def _format_value(self, x, locs, sci_notation=True):
        if sci_notation:
            exponent = math.floor(np.log10(x))
            min_precision = 0
        else:
            exponent = 0
            min_precision = 1

        value = x * 10 ** (-exponent)

        # 如果位置信息数量小于 2，则精度为 min_precision
        if len(locs) < 2:
            precision = min_precision
        else:
            # 计算 x 与 locs 中其他位置的差值，并选择合适的精度
            diff = np.sort(np.abs(locs - x))[1]
            precision = -np.log10(diff) + exponent
            precision = (
                int(np.round(precision))
                if _is_close_to_int(precision)
                else math.ceil(precision)
            )
            if precision < min_precision:
                precision = min_precision

        # 根据计算得到的精度格式化数值的尾数
        mantissa = r"%.*f" % (precision, value)

        if not sci_notation:
            return mantissa

        # 构建科学计数法表示的字符串
        s = r"%s\cdot10^{%d}" % (mantissa, exponent)
        return s

    # 返回字符串 s 的 1 减法格式，支持使用上划线或者直接显示 1-s
    def _one_minus(self, s):
        if self._use_overline:
            return r"\overline{%s}" % s
        else:
            return f"1-{s}"
    # 定义一个调用方法，用于生成刻度标签文本
    def __call__(self, x, pos=None):
        # 如果设置了次要标签并且 x 不在已标记列表中，则返回空字符串
        if self._minor and x not in self._labelled:
            return ""
        # 如果 x 小于等于 0 或者大于等于 1，则返回空字符串
        if x <= 0 or x >= 1:
            return ""
        # 如果 2 * x 接近整数并且其四舍五入值等于 1，则使用 1/2 的文本表示
        if _is_close_to_int(2 * x) and round(2 * x) == 1:
            s = self._one_half
        # 如果 x 小于 0.5 并且 x 近似于一个十的幂次数（相对容差为 1e-7）
        elif x < 0.5 and _is_decade(x, rtol=1e-7):
            # 计算 x 的对数作为指数
            exponent = round(math.log10(x))
            s = "10^{%d}" % exponent
        # 如果 x 大于 0.5 并且 1-x 近似于一个十的幂次数（相对容差为 1e-7）
        elif x > 0.5 and _is_decade(1 - x, rtol=1e-7):
            # 计算 1-x 的对数作为指数，并生成相应的文本表示
            exponent = round(math.log10(1 - x))
            s = self._one_minus("10^{%d}" % exponent)
        # 如果 x 小于 0.1，则使用自定义格式化方法对 x 进行文本表示
        elif x < 0.1:
            s = self._format_value(x, self.locs)
        # 如果 x 大于 0.9，则使用自定义格式化方法对 1-x 进行文本表示，并添加 1- 前缀
        elif x > 0.9:
            s = self._one_minus(self._format_value(1-x, 1-self.locs))
        # 对于其他情况，使用自定义格式化方法对 x 进行文本表示，禁用科学计数法
        else:
            s = self._format_value(x, self.locs, sci_notation=False)
        # 返回 LaTeX 格式的数学默认字体的文本
        return r"$\mathdefault{%s}$" % s

    # 定义一个格式化数据的简短方法，用于生成较短的数据文本表示
    def format_data_short(self, value):
        # 继承的文档字符串描述
        # 当值小于 0.1 时，使用科学计数法表示值
        if value < 0.1:
            return f"{value:e}"
        # 当值小于 0.9 时，使用普通浮点数表示值
        if value < 0.9:
            return f"{value:f}"
        # 其他情况下，使用带有 "1-" 前缀的科学计数法表示 1-value
        return f"1-{1 - value:e}"
class EngFormatter(Formatter):
    """
    Format axis values using engineering prefixes to represent powers
    of 1000, plus a specified unit, e.g., 10 MHz instead of 1e7.
    """

    # The SI engineering prefixes
    ENG_PREFIXES = {
        -30: "q",    # Prefix for 10^-30
        -27: "r",    # Prefix for 10^-27
        -24: "y",    # Prefix for 10^-24
        -21: "z",    # Prefix for 10^-21
        -18: "a",    # Prefix for 10^-18
        -15: "f",    # Prefix for 10^-15
        -12: "p",    # Prefix for 10^-12
         -9: "n",    # Prefix for 10^-9
         -6: "\N{MICRO SIGN}",  # Prefix for 10^-6 (micro sign symbol)
         -3: "m",    # Prefix for 10^-3
          0: "",     # Prefix for 10^0 (no prefix)
          3: "k",    # Prefix for 10^3
          6: "M",    # Prefix for 10^6
          9: "G",    # Prefix for 10^9
         12: "T",    # Prefix for 10^12
         15: "P",    # Prefix for 10^15
         18: "E",    # Prefix for 10^18
         21: "Z",    # Prefix for 10^21
         24: "Y",    # Prefix for 10^24
         27: "R",    # Prefix for 10^27
         30: "Q"     # Prefix for 10^30
    }

    def __init__(self, unit="", places=None, sep=" ", *, usetex=None,
                 useMathText=None):
        r"""
        Parameters
        ----------
        unit : str, default: ""
            Unit symbol to use, suitable for use with single-letter
            representations of powers of 1000. For example, 'Hz' or 'm'.

        places : int, default: None
            Precision with which to display the number, specified in
            digits after the decimal point (there will be between one
            and three digits before the decimal point). If it is None,
            the formatting falls back to the floating point format '%g',
            which displays up to 6 *significant* digits, i.e. the equivalent
            value for *places* varies between 0 and 5 (inclusive).

        sep : str, default: " "
            Separator used between the value and the prefix/unit. For
            example, one get '3.14 mV' if ``sep`` is " " (default) and
            '3.14mV' if ``sep`` is "". Besides the default behavior, some
            other useful options may be:

            * ``sep=""`` to append directly the prefix/unit to the value;
            * ``sep="\N{THIN SPACE}"`` (``U+2009``);
            * ``sep="\N{NARROW NO-BREAK SPACE}"`` (``U+202F``);
            * ``sep="\N{NO-BREAK SPACE}"`` (``U+00A0``).

        usetex : bool, default: :rc:`text.usetex`
            To enable/disable the use of TeX's math mode for rendering the
            numbers in the formatter.

        useMathText : bool, default: :rc:`axes.formatter.use_mathtext`
            To enable/disable the use mathtext for rendering the numbers in
            the formatter.
        """
        self.unit = unit  # 存储单位符号
        self.places = places  # 存储小数点后的精度
        self.sep = sep  # 存储值与前缀/单位之间的分隔符
        self.set_usetex(usetex)  # 设置是否使用TeX的数学模式来渲染数字
        self.set_useMathText(useMathText)  # 设置是否使用mathtext来渲染数字

    def get_usetex(self):
        return self._usetex

    def set_usetex(self, val):
        if val is None:
            self._usetex = mpl.rcParams['text.usetex']  # 如果未指定，使用默认的TeX设置
        else:
            self._usetex = val

    usetex = property(fget=get_usetex, fset=set_usetex)  # 使用property定义usetex属性

    def get_useMathText(self):
        return self._useMathText

    def set_useMathText(self, val):
        if val is None:
            self._useMathText = mpl.rcParams['axes.formatter.use_mathtext']  # 如果未指定，使用默认的mathtext设置
        else:
            self._useMathText = val
    # 定义一个名为 useMathText 的属性，其 getter 方法为 get_useMathText，setter 方法为 set_useMathText
    useMathText = property(fget=get_useMathText, fset=set_useMathText)

    # 定义一个可调用对象，接受参数 x 和 pos=None
    def __call__(self, x, pos=None):
        # 将 x 格式化为工程符号表示，加上单位 self.unit
        s = f"{self.format_eng(x)}{self.unit}"
        # 当不存在前缀和单位时，移除末尾的分隔符 self.sep
        if self.sep and s.endswith(self.sep):
            s = s[:-len(self.sep)]
        # 对结果字符串进行修正处理，返回修正后的结果
        return self.fix_minus(s)

    # 定义一个方法，用于将数字 num 格式化为工程符号表示的字符串
    def format_eng(self, num):
        """
        Format a number in engineering notation, appending a letter
        representing the power of 1000 of the original number.
        Some examples:

        >>> format_eng(0)        # for self.places = 0
        '0'

        >>> format_eng(1000000)  # for self.places = 1
        '1.0 M'

        >>> format_eng(-1e-6)  # for self.places = 2
        '-1.00 \N{MICRO SIGN}'
        """
        sign = 1
        # 如果 self.places 为 None，则格式化方式为 "g"，否则为 ".self.places:d f"
        fmt = "g" if self.places is None else f".{self.places:d}f"

        # 如果 num 小于 0，则记录符号为 -1，并取其绝对值
        if num < 0:
            sign = -1
            num = -num

        # 计算 num 的对数值，用于确定工程符号的幂次
        if num != 0:
            pow10 = int(math.floor(math.log10(num) / 3) * 3)
        else:
            pow10 = 0
            # 强制将 num 设为 0，避免出现如 format_eng(-0) = "0" 和 format_eng(0.0) = "0"
            # 但 format_eng(-0.0) = "-0.0" 的不一致情况
            num = 0.0

        # 将 pow10 限制在 ENG_PREFIXES 的最小和最大值范围内
        pow10 = np.clip(pow10, min(self.ENG_PREFIXES), max(self.ENG_PREFIXES))

        # 计算数值的尾数部分
        mant = sign * num / (10.0 ** pow10)
        
        # 处理四舍五入可能导致的数值接近 1000 而非 1k 的情况，
        # 注意处理 SI 前缀范围外的特殊情况（例如超出 'Y' 的情况）
        if (abs(float(format(mant, fmt))) >= 1000
                and pow10 < max(self.ENG_PREFIXES)):
            mant /= 1000
            pow10 += 3

        # 获取当前工程符号的前缀
        prefix = self.ENG_PREFIXES[int(pow10)]

        # 根据 _usetex 或 _useMathText 的设置，选择是否使用 LaTeX 格式化
        if self._usetex or self._useMathText:
            formatted = f"${mant:{fmt}}${self.sep}{prefix}"
        else:
            formatted = f"{mant:{fmt}}{self.sep}{prefix}"

        # 返回格式化后的字符串
        return formatted
class PercentFormatter(Formatter):
    """
    Format numbers as a percentage.

    Parameters
    ----------
    xmax : float
        Determines how the number is converted into a percentage.
        *xmax* is the data value that corresponds to 100%.
        Percentages are computed as ``x / xmax * 100``. So if the data is
        already scaled to be percentages, *xmax* will be 100. Another common
        situation is where *xmax* is 1.0.

    decimals : None or int
        The number of decimal places to place after the point.
        If *None* (the default), the number will be computed automatically.

    symbol : str or None
        A string that will be appended to the label. It may be
        *None* or empty to indicate that no symbol should be used. LaTeX
        special characters are escaped in *symbol* whenever latex mode is
        enabled, unless *is_latex* is *True*.

    is_latex : bool
        If *False*, reserved LaTeX characters in *symbol* will be escaped.
    """
    def __init__(self, xmax=100, decimals=None, symbol='%', is_latex=False):
        # 初始化 PercentFormatter 类
        # 将 xmax 强制转换为浮点数，并存储在实例变量 self.xmax 中
        self.xmax = xmax + 0.0
        # 存储小数点后的位数要求
        self.decimals = decimals
        # 存储百分号符号或其他自定义符号
        self._symbol = symbol
        # 存储 LaTeX 模式是否启用的标志
        self._is_latex = is_latex

    def __call__(self, x, pos=None):
        """Format the tick as a percentage with the appropriate scaling."""
        # 获取坐标轴的视图间隔
        ax_min, ax_max = self.axis.get_view_interval()
        # 计算显示范围的绝对值
        display_range = abs(ax_max - ax_min)
        # 调用 format_pct 方法格式化百分比并返回结果
        return self.fix_minus(self.format_pct(x, display_range))
    # 将输入的数字 x 转换为百分比形式，基于最大值 self.xmax 进行计算
    def convert_to_pct(self, x):
        return 100.0 * (x / self.xmax)

    # 格式化数字 x 为百分比字符串，根据 self.decimals 的设置确定小数点后的位数，
    # 并在末尾添加符号 self.symbol
    def format_pct(self, x, display_range):
        """
        Format the number as a percentage number with the correct
        number of decimals and adds the percent symbol, if any.

        If ``self.decimals`` is `None`, the number of digits after the
        decimal point is set based on the *display_range* of the axis
        as follows:

        ============= ======== =======================
        display_range decimals sample
        ============= ======== =======================
        >50           0        ``x = 34.5`` => 35%
        >5            1        ``x = 34.5`` => 34.5%
        >0.5          2        ``x = 34.5`` => 34.50%
        ...           ...      ...
        ============= ======== =======================

        This method will not be very good for tiny axis ranges or
        extremely large ones. It assumes that the values on the chart
        are percentages displayed on a reasonable scale.
        """
        # 将 x 转换为百分比形式
        x = self.convert_to_pct(x)
        if self.decimals is None:
            # 如果未指定小数点位数，则根据 display_range 的范围自动设置
            # scaled_range 是基于轴的显示范围的百分比表示
            scaled_range = self.convert_to_pct(display_range)
            if scaled_range <= 0:
                decimals = 0
            else:
                # 使用数学运算确定适当的小数点位数
                decimals = math.ceil(2.0 - math.log10(2.0 * scaled_range))
                if decimals > 5:
                    decimals = 5
                elif decimals < 0:
                    decimals = 0
        else:
            decimals = self.decimals
        # 根据计算得到的小数点位数格式化字符串 s
        s = f'{x:0.{int(decimals)}f}'

        # 将格式化的百分比字符串和符号 self.symbol 连接后返回
        return s + self.symbol

    @property
    # 获取配置的百分比符号字符串
    def symbol(self):
        r"""
        The configured percent symbol as a string.

        If LaTeX is enabled via :rc:`text.usetex`, the special characters
        ``{'#', '$', '%', '&', '~', '_', '^', '\', '{', '}'}`` are
        automatically escaped in the string.
        """
        # 获取符号字符串
        symbol = self._symbol
        if not symbol:
            symbol = ''
        # 如果不是 LaTeX 并且启用了 LaTeX 设置，则自动转义特殊字符
        elif not self._is_latex and mpl.rcParams['text.usetex']:
            # 转义特殊字符以适应 LaTeX 格式
            for spec in r'\#$%&~_^{}':
                symbol = symbol.replace(spec, '\\' + spec)
        return symbol

    @symbol.setter
    # 设置百分比符号字符串
    def symbol(self, symbol):
        self._symbol = symbol
class Locator(TickHelper):
    """
    Determine tick locations.

    Note that the same locator should not be used across multiple
    `~matplotlib.axis.Axis` because the locator stores references to the Axis
    data and view limits.
    """

    # Some automatic tick locators can generate so many ticks they
    # kill the machine when you try and render them.
    # This parameter is set to cause locators to raise an error if too
    # many ticks are generated.
    MAXTICKS = 1000  # 定义最大刻度数为1000，防止生成过多刻度导致内存问题

    def tick_values(self, vmin, vmax):
        """
        Return the values of the located ticks given **vmin** and **vmax**.

        .. note::
            To get tick locations with the vmin and vmax values defined
            automatically for the associated ``axis`` simply call
            the Locator instance::

                >>> print(type(loc))
                <type 'Locator'>
                >>> print(loc())
                [1, 2, 3, 4]

        """
        raise NotImplementedError('Derived must override')  # 子类必须覆盖这个方法

    def set_params(self, **kwargs):
        """
        Do nothing, and raise a warning. Any locator class not supporting the
        set_params() function will call this.
        """
        _api.warn_external(
            "'set_params()' not defined for locator of type " +
            str(type(self)))  # 输出警告，表示此定位器类型不支持 set_params() 方法

    def __call__(self):
        """Return the locations of the ticks."""
        # note: some locators return data limits, other return view limits,
        # hence there is no *one* interface to call self.tick_values.
        raise NotImplementedError('Derived must override')  # 子类必须覆盖这个方法

    def raise_if_exceeds(self, locs):
        """
        Log at WARNING level if *locs* is longer than `Locator.MAXTICKS`.

        This is intended to be called immediately before returning *locs* from
        ``__call__`` to inform users in case their Locator returns a huge
        number of ticks, causing Matplotlib to run out of memory.

        The "strange" name of this method dates back to when it would raise an
        exception instead of emitting a log.
        """
        if len(locs) >= self.MAXTICKS:
            _log.warning(
                "Locator attempting to generate %s ticks ([%s, ..., %s]), "
                "which exceeds Locator.MAXTICKS (%s).",
                len(locs), locs[0], locs[-1], self.MAXTICKS)  # 如果刻度数量超过最大限制，记录警告信息
        return locs
    # 调整范围以避免奇异性的方法
    # 
    # 此方法在自动缩放期间调用，当Axes包含数据时，(v0, v1)设置为数据限制，
    # 否则设置为(-inf, +inf)。
    # 
    # - 如果v0与v1相等（可能在一定浮点数误差范围内），此方法返回围绕该值的扩展区间。
    # - 如果(v0, v1)为(-inf, +inf)，此方法返回适当的默认视图限制。
    # - 否则，返回(v0, v1)而不进行修改。
    def nonsingular(self, v0, v1):
        return mtransforms.nonsingular(v0, v1, expander=.05)

    # 选择从vmin到vmax的范围的刻度
    # 
    # 子类应该重写此方法以更改定位器的行为。
    def view_limits(self, vmin, vmax):
        return mtransforms.nonsingular(vmin, vmax)
class IndexLocator(Locator):
    """
    Place ticks at every nth point plotted.

    IndexLocator assumes index plotting; i.e., that the ticks are placed at integer
    values in the range between 0 and len(data) inclusive.
    """

    def __init__(self, base, offset):
        """Place ticks every *base* data point, starting at *offset*."""
        # 初始化 IndexLocator 对象，设定间隔基数和起始偏移量
        self._base = base
        self.offset = offset

    def set_params(self, base=None, offset=None):
        """Set parameters within this locator"""
        # 根据提供的参数设置基数和偏移量
        if base is not None:
            self._base = base
        if offset is not None:
            self.offset = offset

    def __call__(self):
        """Return the locations of the ticks"""
        # 获取当前轴的数据范围
        dmin, dmax = self.axis.get_data_interval()
        # 返回计算出的刻度位置
        return self.tick_values(dmin, dmax)

    def tick_values(self, vmin, vmax):
        """
        Return the locations of the ticks.

        This function calculates the positions of ticks based on the provided
        vmin and vmax.

        """
        # 计算并返回刻度的位置
        return self.raise_if_exceeds(
            np.arange(vmin + self.offset, vmax + 1, self._base))


class FixedLocator(Locator):
    """
    Place ticks at a set of fixed values.

    If *nbins* is None ticks are placed at all values. Otherwise, the *locs* array of
    possible positions will be subsampled to keep the number of ticks <=
    :math:`nbins* +1`. The subsampling will be done to include the smallest absolute
    value; for example, if zero is included in the array of possibilities, then it of
    the chosen ticks.
    """

    def __init__(self, locs, nbins=None):
        # 初始化 FixedLocator 对象，设定固定位置的刻度值和最大刻度数量
        self.locs = np.asarray(locs)
        _api.check_shape((None,), locs=self.locs)
        self.nbins = max(nbins, 2) if nbins is not None else None

    def set_params(self, nbins=None):
        """Set parameters within this locator."""
        # 根据提供的参数设置刻度的数量
        if nbins is not None:
            self.nbins = nbins

    def __call__(self):
        # 返回刻度位置
        return self.tick_values(None, None)

    def tick_values(self, vmin, vmax):
        """
        Return the locations of the ticks.

        .. note::

            Because the values are fixed, vmin and vmax are not used in this
            method.

        """
        # 如果没有限制刻度数量，直接返回固定位置的刻度值
        if self.nbins is None:
            return self.locs
        # 否则，根据最大刻度数量进行子采样
        step = max(int(np.ceil(len(self.locs) / self.nbins)), 1)
        ticks = self.locs[::step]
        for i in range(1, step):
            ticks1 = self.locs[i::step]
            # 选择包含最小绝对值的刻度值进行返回
            if np.abs(ticks1).min() < np.abs(ticks).min():
                ticks = ticks1
        return self.raise_if_exceeds(ticks)


class NullLocator(Locator):
    """
    No ticks
    """

    def __call__(self):
        # 返回空列表，表示没有刻度
        return self.tick_values(None, None)

    def tick_values(self, vmin, vmax):
        """
        Return the locations of the ticks.

        .. note::

            Because the values are Null, vmin and vmax are not used in this
            method.
        """
        # 返回空列表，表示没有刻度
        return []


class LinearLocator(Locator):
    """
    Place ticks at evenly spaced values.

    The first time this function is called it will try to set the
    number of ticks to make a nice tick partitioning.  Thereafter, the
    """

    # 以下部分为示例结束的内容，已省略
    number of ticks will be fixed so that interactive navigation will
    be nice

    """
    定义一个 Tick 定位器的类，用于在图形中定位坐标轴的刻度线位置。

    Parameters
    ----------
    numticks : int or None, default None
        刻度线的数量。如果为 None，则默认为 11。
    presets : dict or None, default: None
        将 ``(vmin, vmax)`` 映射到位置数组的字典。如果当前 ``(vmin, vmax)`` 存在条目，则覆盖 *numticks*。

    """
    def __init__(self, numticks=None, presets=None):
        """
        初始化方法，用于设置 Tick 定位器的初始参数。

        Parameters
        ----------
        numticks : int or None, default None
            刻度线的数量。如果为 None，则 *numticks* = 11。
        presets : dict or None, default: None
            将 ``(vmin, vmax)`` 映射到位置数组的字典。如果当前 ``(vmin, vmax)`` 存在条目，则覆盖 *numticks*。

        """
        self.numticks = numticks
        # 如果 presets 为 None，则初始化为空字典
        if presets is None:
            self.presets = {}
        else:
            self.presets = presets

    @property
    def numticks(self):
        """
        获取当前的刻度线数量。

        Returns
        -------
        int
            当前的刻度线数量，默认为 11（如果未设置）。

        """
        # 返回保存在 _numticks 属性中的值，如果未设置则返回默认值 11
        return self._numticks if self._numticks is not None else 11

    @numticks.setter
    def numticks(self, numticks):
        """
        设置刻度线数量。

        Parameters
        ----------
        numticks : int or None
            要设置的刻度线数量。

        """
        self._numticks = numticks

    def set_params(self, numticks=None, presets=None):
        """
        设置 Tick 定位器的参数。

        Parameters
        ----------
        numticks : int or None, optional
            要设置的刻度线数量。
        presets : dict or None, optional
            将 ``(vmin, vmax)`` 映射到位置数组的字典。

        """
        # 如果 presets 不为 None，则更新 presets
        if presets is not None:
            self.presets = presets
        # 如果 numticks 不为 None，则更新 numticks
        if numticks is not None:
            self.numticks = numticks

    def __call__(self):
        """
        调用实例时返回刻度线的位置。

        Returns
        -------
        array-like
            刻度线的位置。

        """
        # 获取当前轴的视图区间的上下界
        vmin, vmax = self.axis.get_view_interval()
        # 返回根据当前视图区间计算的刻度线位置
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        """
        计算给定视图区间的刻度线位置。

        Parameters
        ----------
        vmin, vmax : float
            视图区间的上下界。

        Returns
        -------
        array-like
            刻度线的位置数组。

        """
        # 确保视图区间非奇异
        vmin, vmax = mtransforms.nonsingular(vmin, vmax, expander=0.05)

        # 如果 presets 中包含当前视图区间的条目，则返回预设的刻度线位置数组
        if (vmin, vmax) in self.presets:
            return self.presets[(vmin, vmax)]

        # 如果 numticks 为 0，则返回空数组
        if self.numticks == 0:
            return []

        # 否则根据当前视图区间和刻度线数量生成线性分布的刻度线位置
        ticklocs = np.linspace(vmin, vmax, self.numticks)

        return self.raise_if_exceeds(ticklocs)

    def view_limits(self, vmin, vmax):
        """
        智能选择视图限制的方法。

        Parameters
        ----------
        vmin, vmax : float
            视图区间的上下界。

        Returns
        -------
        tuple
            修正后的视图区间的上下界。

        """
        # 如果 vmax 小于 vmin，则交换二者
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # 如果 vmin 等于 vmax，则微调 vmin 和 vmax
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        # 如果配置中指定使用 'round_numbers' 模式，则根据 numticks 调整视图范围
        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            exponent, remainder = divmod(
                math.log10(vmax - vmin), math.log10(max(self.numticks - 1, 1)))
            exponent -= (remainder < .5)
            scale = max(self.numticks - 1, 1) ** (-exponent)
            vmin = math.floor(scale * vmin) / scale
            vmax = math.ceil(scale * vmax) / scale

        # 返回修正后的视图区间
        return mtransforms.nonsingular(vmin, vmax)
class MultipleLocator(Locator):
    """
    Place ticks at every integer multiple of a base plus an offset.
    """

    def __init__(self, base=1.0, offset=0.0):
        """
        Parameters
        ----------
        base : float > 0
            Interval between ticks.
        offset : float
            Value added to each multiple of *base*.

            .. versionadded:: 3.8
        """
        # 初始化函数，设置基础间隔和偏移量
        self._edge = _Edge_integer(base, 0)
        self._offset = offset

    def set_params(self, base=None, offset=None):
        """
        Set parameters within this locator.

        Parameters
        ----------
        base : float > 0
            Interval between ticks.
        offset : float
            Value added to each multiple of *base*.

            .. versionadded:: 3.8
        """
        # 根据传入的参数设置基础间隔和偏移量
        if base is not None:
            self._edge = _Edge_integer(base, 0)
        if offset is not None:
            self._offset = offset

    def __call__(self):
        """Return the locations of the ticks."""
        # 返回刻度的位置
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        # 计算刻度值的函数
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        step = self._edge.step
        vmin -= self._offset
        vmax -= self._offset
        vmin = self._edge.ge(vmin) * step
        n = (vmax - vmin + 0.001 * step) // step
        locs = vmin - step + np.arange(n + 3) * step + self._offset
        return self.raise_if_exceeds(locs)

    def view_limits(self, dmin, dmax):
        """
        Set the view limits to the nearest tick values that contain the data.
        """
        # 根据数据设置视图限制，以包含数据的最近刻度值
        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            vmin = self._edge.le(dmin - self._offset) * self._edge.step + self._offset
            vmax = self._edge.ge(dmax - self._offset) * self._edge.step + self._offset
            if vmin == vmax:
                vmin -= 1
                vmax += 1
        else:
            vmin = dmin
            vmax = dmax

        return mtransforms.nonsingular(vmin, vmax)


def scale_range(vmin, vmax, n=1, threshold=100):
    dv = abs(vmax - vmin)  # > 0 as nonsingular is called before.
    meanv = (vmax + vmin) / 2
    if abs(meanv) / dv < threshold:
        offset = 0
    else:
        offset = math.copysign(10 ** (math.log10(abs(meanv)) // 1), meanv)
    scale = 10 ** (math.log10(dv / n) // 1)
    return scale, offset


class _Edge_integer:
    """
    Helper for `.MaxNLocator`, `.MultipleLocator`, etc.

    Take floating-point precision limitations into account when calculating
    tick locations as integer multiples of a step.
    """
    def __init__(self, step, offset):
        """
        Parameters
        ----------
        step : float > 0
            Interval between ticks.
        offset : float
            Offset subtracted from the data limits prior to calculating tick
            locations.
        """
        # 检查步长参数是否大于零，如果不是则抛出值错误异常
        if step <= 0:
            raise ValueError("'step' must be positive")
        # 将步长和偏移量的绝对值保存为对象属性
        self.step = step
        self._offset = abs(offset)

    def closeto(self, ms, edge):
        # 当偏移量远大于步长时允许更大的误差容忍度
        if self._offset > 0:
            # 计算偏移量与步长比值的对数值
            digits = np.log10(self._offset / self.step)
            # 计算容忍度，确保最小为 1e-10，且最大为 0.4999
            tol = max(1e-10, 10 ** (digits - 12))
            tol = min(0.4999, tol)
        else:
            # 如果偏移量为零，则容忍度设为 1e-10
            tol = 1e-10
        # 返回是否与边缘值足够接近的布尔值
        return abs(ms - edge) < tol

    def le(self, x):
        """Return the largest n: n*step <= x."""
        # 将 x 按步长进行除法，d 是商，m 是余数
        d, m = divmod(x, self.step)
        # 如果余数与步长的比值与 1 接近，则返回商加 1，否则返回商
        if self.closeto(m / self.step, 1):
            return d + 1
        return d

    def ge(self, x):
        """Return the smallest n: n*step >= x."""
        # 将 x 按步长进行除法，d 是商，m 是余数
        d, m = divmod(x, self.step)
        # 如果余数与步长的比值与 0 接近，则返回商，否则返回商加 1
        if self.closeto(m / self.step, 0):
            return d
        return d + 1
# 定义一个名为 MaxNLocator 的类，继承自 Locator 类
class MaxNLocator(Locator):
    """
    Place evenly spaced ticks, with a cap on the total number of ticks.

    Finds nice tick locations with no more than :math:`nbins + 1` ticks being within the
    view limits. Locations beyond the limits are added to support autoscaling.
    """

    # 默认参数字典，包含 nbins、steps、integer、symmetric、prune 和 min_n_ticks
    default_params = dict(nbins=10,
                          steps=None,
                          integer=False,
                          symmetric=False,
                          prune=None,
                          min_n_ticks=2)

    # 初始化方法，接受 nbins 和其他关键字参数
    def __init__(self, nbins=None, **kwargs):
        """
        Parameters
        ----------
        nbins : int or 'auto', default: 10
            Maximum number of intervals; one less than max number of
            ticks.  If the string 'auto', the number of bins will be
            automatically determined based on the length of the axis.

        steps : array-like, optional
            Sequence of acceptable tick multiples, starting with 1 and
            ending with 10. For example, if ``steps=[1, 2, 4, 5, 10]``,
            ``20, 40, 60`` or ``0.4, 0.6, 0.8`` would be possible
            sets of ticks because they are multiples of 2.
            ``30, 60, 90`` would not be generated because 3 does not
            appear in this example list of steps.

        integer : bool, default: False
            If True, ticks will take only integer values, provided at least
            *min_n_ticks* integers are found within the view limits.

        symmetric : bool, default: False
            If True, autoscaling will result in a range symmetric about zero.

        prune : {'lower', 'upper', 'both', None}, default: None
            Remove the 'lower' tick, the 'upper' tick, or ticks on 'both' sides
            *if they fall exactly on an axis' edge* (this typically occurs when
            :rc:`axes.autolimit_mode` is 'round_numbers').  Removing such ticks
            is mostly useful for stacked or ganged plots, where the upper tick
            of an Axes overlaps with the lower tick of the axes above it.

        min_n_ticks : int, default: 2
            Relax *nbins* and *integer* constraints if necessary to obtain
            this minimum number of ticks.
        """
        # 如果传入 nbins 参数，则更新到 kwargs 字典中
        if nbins is not None:
            kwargs['nbins'] = nbins
        # 调用 set_params 方法，设置参数
        self.set_params(**{**self.default_params, **kwargs})

    # 静态方法声明，此处省略具体内容
    @staticmethod
    def _validate_steps(steps):
        # 检查 steps 是否可迭代，如果不是则抛出 ValueError 异常
        if not np.iterable(steps):
            raise ValueError('steps argument must be an increasing sequence '
                             'of numbers between 1 and 10 inclusive')
        # 将 steps 转换为 NumPy 数组
        steps = np.asarray(steps)
        # 检查 steps 是否是严格递增的序列，且最后一个元素不超过 10，第一个元素不小于 1
        if np.any(np.diff(steps) <= 0) or steps[-1] > 10 or steps[0] < 1:
            raise ValueError('steps argument must be an increasing sequence '
                             'of numbers between 1 and 10 inclusive')
        # 如果 steps 的第一个元素不是 1，则将其加入到 steps 的开头
        if steps[0] != 1:
            steps = np.concatenate([[1], steps])
        # 如果 steps 的最后一个元素不是 10，则将其加入到 steps 的末尾
        if steps[-1] != 10:
            steps = np.concatenate([steps, [10]])
        return steps

    @staticmethod
    def _staircase(steps):
        # 创建一个扩展的阶梯数组，用于定位所需步骤。这个数组可能比实际需要的要大很多。
        return np.concatenate([0.1 * steps[:-1], steps, [10 * steps[1]]])

    def set_params(self, **kwargs):
        """
        为该定位器设置参数。

        Parameters
        ----------
        nbins : int or 'auto', optional
            参见 `.MaxNLocator`
        steps : array-like, optional
            参见 `.MaxNLocator`
        integer : bool, optional
            参见 `.MaxNLocator`
        symmetric : bool, optional
            参见 `.MaxNLocator`
        prune : {'lower', 'upper', 'both', None}, optional
            参见 `.MaxNLocator`
        min_n_ticks : int, optional
            参见 `.MaxNLocator`
        """
        # 处理 'nbins' 参数
        if 'nbins' in kwargs:
            self._nbins = kwargs.pop('nbins')
            # 如果 nbins 不是 'auto'，则将其转换为整数
            if self._nbins != 'auto':
                self._nbins = int(self._nbins)
        # 处理 'symmetric' 参数
        if 'symmetric' in kwargs:
            self._symmetric = kwargs.pop('symmetric')
        # 处理 'prune' 参数
        if 'prune' in kwargs:
            prune = kwargs.pop('prune')
            # 检查 'prune' 是否属于指定的列表中
            _api.check_in_list(['upper', 'lower', 'both', None], prune=prune)
            self._prune = prune
        # 处理 'min_n_ticks' 参数
        if 'min_n_ticks' in kwargs:
            self._min_n_ticks = max(1, kwargs.pop('min_n_ticks'))
        # 处理 'steps' 参数
        if 'steps' in kwargs:
            steps = kwargs.pop('steps')
            # 如果 steps 为 None，则使用默认的步骤数组
            if steps is None:
                self._steps = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10])
            else:
                # 否则，使用 _validate_steps 函数验证并设置 steps
                self._steps = self._validate_steps(steps)
            # 根据验证后的 steps，生成扩展的阶梯数组
            self._extended_steps = self._staircase(self._steps)
        # 处理 'integer' 参数
        if 'integer' in kwargs:
            self._integer = kwargs.pop('integer')
        # 如果还有未处理的参数，则引发异常
        if kwargs:
            raise _api.kwarg_error("set_params", kwargs)

    def __call__(self):
        # 获取当前坐标轴的视图间隔，并返回刻度值
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)
    # 根据是否对称性调整最大和最小值
    def tick_values(self, vmin, vmax):
        # 如果设置为对称，重新计算最大和最小值
        if self._symmetric:
            vmax = max(abs(vmin), abs(vmax))  # 取最大值的绝对值作为新的最大值
            vmin = -vmax  # 最小值设为负的最大值

        # 调用 mtransforms.nonsingular 函数确保 vmin 和 vmax 不重合
        vmin, vmax = mtransforms.nonsingular(
            vmin, vmax, expander=1e-13, tiny=1e-14)

        # 获取原始刻度位置
        locs = self._raw_ticks(vmin, vmax)

        # 根据 prune 参数修剪刻度位置
        prune = self._prune
        if prune == 'lower':
            locs = locs[1:]  # 移除最低端的刻度
        elif prune == 'upper':
            locs = locs[:-1]  # 移除最高端的刻度
        elif prune == 'both':
            locs = locs[1:-1]  # 移除两端的刻度

        # 检查刻度是否超出范围并返回结果
        return self.raise_if_exceeds(locs)

    # 根据是否对称性调整数据范围的上下限
    def view_limits(self, dmin, dmax):
        if self._symmetric:
            dmax = max(abs(dmin), abs(dmax))  # 取最大值的绝对值作为新的最大值
            dmin = -dmax  # 最小值设为负的最大值

        # 调用 mtransforms.nonsingular 函数确保 dmin 和 dmax 不重合
        dmin, dmax = mtransforms.nonsingular(
            dmin, dmax, expander=1e-12, tiny=1e-13)

        # 根据全局参数 axes.autolimit_mode 决定返回哪些数据范围
        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            return self._raw_ticks(dmin, dmax)[[0, -1]]  # 返回首尾两个刻度位置
        else:
            return dmin, dmax  # 直接返回给定的数据范围
# 返回 True 如果 *x* 是 *base* 的整数幂
def _is_decade(x, *, base=10, rtol=None):
    # 检查 x 是否有限，如果不是，则返回 False
    if not np.isfinite(x):
        return False
    # 如果 x 等于 0.0，则认为是 10 的 0 次幂，返回 True
    if x == 0.0:
        return True
    # 计算 x 的绝对值的对数除以 base 的对数，判断是否近似为整数
    lx = np.log(abs(x)) / np.log(base)
    # 如果没有给定相对误差 rtol，则使用 np.isclose 判断 lx 是否近似为整数
    if rtol is None:
        return np.isclose(lx, np.round(lx))
    else:
        # 否则使用给定的 rtol 判断 lx 是否近似为整数
        return np.isclose(lx, np.round(lx), rtol=rtol)


def _decade_less_equal(x, base):
    """
    返回不大于 *x* 的最大 *base* 的整数幂。

    如果 *x* 是负数，则指数将是更大的整数。
    """
    return (x if x == 0 else
            -_decade_greater_equal(-x, base) if x < 0 else
            base ** np.floor(np.log(x) / np.log(base)))


def _decade_greater_equal(x, base):
    """
    返回不小于 *x* 的最小 *base* 的整数幂。

    如果 *x* 是负数，则指数将是更小的整数。
    """
    return (x if x == 0 else
            -_decade_less_equal(-x, base) if x < 0 else
            base ** np.ceil(np.log(x) / np.log(base)))


def _decade_less(x, base):
    """
    返回小于 *x* 的最大 *base* 的整数幂。

    如果 *x* 是负数，则指数将是更大的整数。
    """
    if x < 0:
        return -_decade_greater(-x, base)
    less = _decade_less_equal(x, base)
    # 如果 less 等于 x，则除以 base 得到更小的整数幂
    if less == x:
        less /= base
    return less


def _decade_greater(x, base):
    """
    返回大于 *x* 的最小 *base* 的整数幂。

    如果 *x* 是负数，则指数将是更小的整数。
    """
    if x < 0:
        return -_decade_less(-x, base)
    greater = _decade_greater_equal(x, base)
    # 如果 greater 等于 x，则乘以 base 得到更大的整数幂
    if greater == x:
        greater *= base
    return greater


def _is_close_to_int(x):
    # 检查 x 是否接近于整数
    return math.isclose(x, round(x))


class LogLocator(Locator):
    """
    放置对数间隔的刻度。

    在值 ``subs[j] * base**i`` 处放置刻度。
    """

    @_api.delete_parameter("3.8", "numdecs")
    def __init__(self, base=10.0, subs=(1.0,), numdecs=4, numticks=None):
        """
        Parameters
        ----------
        base : float, default: 10.0
            使用的对数的基数，主要刻度位于 ``base**n`` 处，其中 ``n`` 是整数。
        subs : None or {'auto', 'all'} or sequence of float, default: (1.0,)
            在基数的整数幂处放置刻度的倍数。
            默认值 ``(1.0, )`` 仅在基数的整数幂处放置刻度。
            允许的字符串值有 ``'auto'`` 和 ``'all'``。
            - ``'auto'``: 仅在整数幂之间放置刻度。
            - ``'all'``: 在整数幂之间和整数幂处放置刻度。
            - ``None``: 等同于 ``'auto'``。
        numticks : None or int, default: None
            给定轴上允许的最大刻度数。默认值 ``None`` 将尝试智能选择，只要此定位器已经使用
            `~.axis.Axis.get_tick_space` 分配给轴，否则默认为 9。
        """
        if numticks is None:
            # 如果未指定 numticks，则根据 matplotlib 配置决定
            if mpl.rcParams['_internal.classic_mode']:
                numticks = 15
            else:
                numticks = 'auto'
        # 设置对数的基数为浮点数 base
        self._base = float(base)
        # 设置刻度的倍数
        self._set_subs(subs)
        # 设置 numdecs 属性（已弃用）
        self._numdecs = numdecs
        # 设置 numticks 属性
        self.numticks = numticks

    @_api.delete_parameter("3.8", "numdecs")
    def set_params(self, base=None, subs=None, numdecs=None, numticks=None):
        """设置此定位器中的参数。"""
        # 如果 base 不为 None，则设置对数的基数为 base
        if base is not None:
            self._base = float(base)
        # 如果 subs 不为 None，则设置刻度的倍数
        if subs is not None:
            self._set_subs(subs)
        # 如果 numdecs 不为 None，则设置 numdecs 属性（已弃用）
        if numdecs is not None:
            self._numdecs = numdecs
        # 如果 numticks 不为 None，则设置 numticks 属性
        if numticks is not None:
            self.numticks = numticks

    numdecs = _api.deprecate_privatize_attribute(
        "3.8", addendum="此属性已不再生效。")
    def _set_subs(self, subs):
        """
        Set the minor ticks for the log scaling every ``base**i*subs[j]``.
        """
        # 如果 subs 参数为 None，则将 _subs 设置为 'auto'，与之前的糟糕 API 保持一致
        if subs is None:  
            self._subs = 'auto'
        # 如果 subs 参数为字符串类型
        elif isinstance(subs, str):
            # 检查 subs 是否在 ('all', 'auto') 中
            _api.check_in_list(('all', 'auto'), subs=subs)
            self._subs = subs
        # 如果 subs 参数为其他类型
        else:
            try:
                # 尝试将 subs 转换为 float 类型的 NumPy 数组
                self._subs = np.asarray(subs, dtype=float)
            except ValueError as e:
                # 如果转换出错，抛出 ValueError 异常
                raise ValueError("subs must be None, 'all', 'auto' or "
                                 "a sequence of floats, not "
                                 f"{subs}.") from e
            # 检查 _subs 数组的维度是否为 1
            if self._subs.ndim != 1:
                # 如果不是一维数组，抛出 ValueError 异常
                raise ValueError("A sequence passed to subs must be "
                                 "1-dimensional, not "
                                 f"{self._subs.ndim}-dimensional.")

    def __call__(self):
        """Return the locations of the ticks."""
        # 获取当前轴的视图间隔 vmin 和 vmax
        vmin, vmax = self.axis.get_view_interval()
        # 调用 tick_values 方法计算刻度的位置，并返回结果
        return self.tick_values(vmin, vmax)
    # 定义一个方法，计算给定范围内的刻度值
    def tick_values(self, vmin, vmax):
        # 如果需要自动确定刻度数量
        if self.numticks == 'auto':
            # 如果指定了坐标轴，则使用其提供的刻度空间，限定在2到9之间
            if self.axis is not None:
                numticks = np.clip(self.axis.get_tick_space(), 2, 9)
            else:
                numticks = 9
        else:
            numticks = self.numticks

        # 获取底数
        b = self._base
        
        # 如果最小值小于等于0
        if vmin <= 0.0:
            # 如果有指定坐标轴，获取其最小正值
            if self.axis is not None:
                vmin = self.axis.get_minpos()

            # 如果获取到的最小值仍然小于等于0或者不是有限值，抛出异常
            if vmin <= 0.0 or not np.isfinite(vmin):
                raise ValueError(
                    "Data has no positive values, and therefore cannot be log-scaled.")

        # 调试日志记录最小值和最大值
        _log.debug('vmin %s vmax %s', vmin, vmax)

        # 如果最大值小于最小值，交换它们
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        
        # 计算对数刻度下的最小和最大值
        log_vmin = math.log(vmin) / math.log(b)
        log_vmax = math.log(vmax) / math.log(b)

        # 计算刻度之间的数量级
        numdec = math.floor(log_vmax) - math.ceil(log_vmin)

        # 如果刻度值是字符串
        if isinstance(self._subs, str):
            # 如果数量级大于10或者底数小于3
            if numdec > 10 or b < 3:
                # 如果设置为自动，返回空数组表示没有主要或次要刻度
                if self._subs == 'auto':
                    return np.array([])  # no minor or major ticks
                else:
                    subs = np.array([1.0])  # major ticks
            else:
                # 否则根据设置的第一个值和底数生成刻度
                _first = 2.0 if self._subs == 'auto' else 1.0
                subs = np.arange(_first, b)
        else:
            subs = self._subs

        # 获取主要刻度之间的数量级
        stride = (max(math.ceil(numdec / (numticks - 1)), 1)
                  if mpl.rcParams['_internal.classic_mode'] else
                  numdec // numticks + 1)

        # 如果决定的步长大于等于刻度数量级，修正步长到可用范围内
        if stride >= numdec:
            stride = max(1, numdec - 1)

        # 判断subs是否包含除了1以外的内容，用于区分主要和次要定位器
        have_subs = len(subs) > 1 or (len(subs) == 1 and subs[0] != 1.0)

        # 计算刻度之间的数量级范围
        decades = np.arange(math.floor(log_vmin) - stride,
                            math.ceil(log_vmax) + 2 * stride, stride)

        # 如果有次要刻度
        if have_subs:
            # 如果步长为1，计算次要刻度位置
            if stride == 1:
                ticklocs = np.concatenate(
                    [subs * decade_start for decade_start in b ** decades])
            else:
                ticklocs = np.array([])
        else:
            # 否则计算主要刻度位置
            ticklocs = b ** decades

        # 调试日志记录刻度位置
        _log.debug('ticklocs %r', ticklocs)
        
        # 如果subs长度大于1且步长为1，并且刻度位置只有一个在范围内
        if (len(subs) > 1
                and stride == 1
                and ((vmin <= ticklocs) & (ticklocs <= vmax)).sum() <= 1):
            # 如果是期望每个数量级至少有两个刻度的次要定位器，切换到AutoLocator
            return AutoLocator().tick_values(vmin, vmax)
        else:
            # 否则返回刻度位置
            return self.raise_if_exceeds(ticklocs)
    def view_limits(self, vmin, vmax):
        """尝试智能选择视图限制。"""
        b = self._base  # 从对象属性中获取基础值

        vmin, vmax = self.nonsingular(vmin, vmax)  # 调用nonsingular方法处理vmin和vmax，确保它们不同

        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            vmin = _decade_less_equal(vmin, b)  # 如果autolimit_mode是'round_numbers'，则调整vmin使其不超过b的一个数量级
            vmax = _decade_greater_equal(vmax, b)  # 调整vmax使其不低于b的一个数量级

        return vmin, vmax

    def nonsingular(self, vmin, vmax):
        if vmin > vmax:
            vmin, vmax = vmax, vmin  # 如果vmin大于vmax，则交换它们的值
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = 1, 10  # 如果vmin或vmax不是有限数值，则将它们设置为初始范围1到10，表示尚未绘制数据。
        elif vmax <= 0:
            _api.warn_external(
                "Data has no positive values, and therefore cannot be "
                "log-scaled.")
            vmin, vmax = 1, 10  # 如果vmax小于等于0，则发出警告并将vmin和vmax设置为1到10。
        else:
            # 考虑共享的坐标轴
            minpos = min(axis.get_minpos() for axis in self.axis._get_shared_axis())  # 获取所有共享轴的最小正值
            if not np.isfinite(minpos):
                minpos = 1e-300  # 如果没有有限的最小正值，则将其设为一个极小的数值
            if vmin <= 0:
                vmin = minpos  # 如果vmin小于等于0，则将其设为最小正值
            if vmin == vmax:
                vmin = _decade_less(vmin, self._base)  # 如果vmin等于vmax，则调整vmin以小于当前基数的一个数量级
                vmax = _decade_greater(vmax, self._base)  # 调整vmax以大于当前基数的一个数量级
        return vmin, vmax
class SymmetricalLogLocator(Locator):
    """
    Place ticks spaced linearly near zero and spaced logarithmically beyond a threshold.
    """

    def __init__(self, transform=None, subs=None, linthresh=None, base=None):
        """
        Parameters
        ----------
        transform : `~.scale.SymmetricalLogTransform`, optional
            If set, defines the *base* and *linthresh* of the symlog transform.
        base, linthresh : float, optional
            The *base* and *linthresh* of the symlog transform, as documented
            for `.SymmetricalLogScale`.  These parameters are only used if
            *transform* is not set.
        subs : sequence of float, default: [1]
            The multiples of integer powers of the base where ticks are placed,
            i.e., ticks are placed at
            ``[sub * base**i for i in ... for sub in subs]``.

        Notes
        -----
        Either *transform*, or both *base* and *linthresh*, must be given.
        """
        # 如果提供了 transform 参数，则使用其 base 和 linthresh 属性
        if transform is not None:
            self._base = transform.base
            self._linthresh = transform.linthresh
        # 否则，如果同时提供了 linthresh 和 base 参数，则使用它们
        elif linthresh is not None and base is not None:
            self._base = base
            self._linthresh = linthresh
        else:
            # 如果以上参数都未提供，则抛出 ValueError 异常
            raise ValueError("Either transform, or both linthresh "
                             "and base, must be provided.")
        # 如果 subs 未提供，默认设置为 [1.0]
        if subs is None:
            self._subs = [1.0]
        else:
            self._subs = subs
        # 默认设置 numticks 为 15
        self.numticks = 15

    def set_params(self, subs=None, numticks=None):
        """Set parameters within this locator."""
        # 如果提供了 numticks 参数，则更新 numticks 属性
        if numticks is not None:
            self.numticks = numticks
        # 如果提供了 subs 参数，则更新 _subs 属性
        if subs is not None:
            self._subs = subs

    def __call__(self):
        """Return the locations of the ticks."""
        # 注意，这些是未经转换的坐标
        # 获取当前轴的视图区间的最小值和最大值
        vmin, vmax = self.axis.get_view_interval()
        # 返回计算得到的刻度位置数组
        return self.tick_values(vmin, vmax)
    # 返回给定范围内的刻度值列表
    def tick_values(self, vmin, vmax):
        # 获取对象内部的线性阈值
        linthresh = self._linthresh

        # 如果最大值小于最小值，则交换它们
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # 域被分为三个部分，实际可能只有其中一些存在。
        #
        # <======== -t ==0== t ========>
        # aaaaaaaaa    bbbbb   ccccccccc
        #
        # a) 和 c) 将在整数对数位置有刻度。如果刻度数超过 self.numticks，则需要减少刻度数。
        #
        # b) 在 0 点有一个刻度，仅仅是 0（我们假设 t 是一个小数，线性段只是一个实现细节，不是有趣的部分。）
        #
        # 我们也可以在 t 点添加刻度，但那似乎通常不那么有趣。
        #
        # "simple" 模式是当范围完全落在 [-t, t] 内时 -- 它应该只显示 (vmin, 0, vmax)
        if -linthresh <= vmin < vmax <= linthresh:
            # 只有线性范围存在
            return sorted({vmin, 0, vmax})

        # 低对数范围存在
        has_a = (vmin < -linthresh)
        # 高对数范围存在
        has_c = (vmax > linthresh)

        # 检查线性范围是否存在
        has_b = (has_a and vmax > -linthresh) or (has_c and vmin < linthresh)

        # 获取基数值
        base = self._base

        # 定义函数，计算对数范围
        def get_log_range(lo, hi):
            lo = np.floor(np.log(lo) / np.log(base))
            hi = np.ceil(np.log(hi) / np.log(base))
            return lo, hi

        # 计算所有范围，以确定步幅
        a_lo, a_hi = (0, 0)
        if has_a:
            a_upper_lim = min(-linthresh, vmax)
            a_lo, a_hi = get_log_range(abs(a_upper_lim), abs(vmin) + 1)

        c_lo, c_hi = (0, 0)
        if has_c:
            c_lower_lim = max(linthresh, vmin)
            c_lo, c_hi = get_log_range(c_lower_lim, vmax + 1)

        # 计算在 a 和 c 范围内的整数指数总数
        total_ticks = (a_hi - a_lo) + (c_hi - c_lo)
        if has_b:
            total_ticks += 1
        stride = max(total_ticks // (self.numticks - 1), 1)

        decades = []
        if has_a:
            decades.extend(-1 * (base ** (np.arange(a_lo, a_hi,
                                                    stride)[::-1])))

        if has_b:
            decades.append(0.0)

        if has_c:
            decades.extend(base ** (np.arange(c_lo, c_hi, stride)))

        subs = np.asarray(self._subs)

        # 如果替代值的长度大于 1 或者第一个元素不等于 1.0
        if len(subs) > 1 or subs[0] != 1.0:
            ticklocs = []
            for decade in decades:
                if decade == 0:
                    ticklocs.append(decade)
                else:
                    ticklocs.extend(subs * decade)
        else:
            ticklocs = decades

        # 返回刻度位置数组
        return self.raise_if_exceeds(np.array(ticklocs))
    def view_limits(self, vmin, vmax):
        """Try to choose the view limits intelligently."""
        # 从对象中获取基础值
        b = self._base
        # 如果最大值小于最小值，则交换它们，确保vmin <= vmax
        if vmax < vmin:
            vmin, vmax = vmax, vmin

        # 如果 matplotlib 的参数 axes.autolimit_mode 设置为 'round_numbers'
        if mpl.rcParams['axes.autolimit_mode'] == 'round_numbers':
            # 将最小值调整为不大于当前基础值的最大的十进制数
            vmin = _decade_less_equal(vmin, b)
            # 将最大值调整为不小于当前基础值的最小的十进制数
            vmax = _decade_greater_equal(vmax, b)
            # 如果调整后的最小值和最大值相等
            if vmin == vmax:
                # 将最小值调整为小于当前基础值的最大的十进制数
                vmin = _decade_less(vmin, b)
                # 将最大值调整为大于当前基础值的最小的十进制数
                vmax = _decade_greater(vmax, b)

        # 返回经过变换后的非奇异视图限制
        return mtransforms.nonsingular(vmin, vmax)
class AsinhLocator(Locator):
    """
    Place ticks spaced evenly on an inverse-sinh scale.

    Generally used with the `~.scale.AsinhScale` class.

    .. note::

       This API is provisional and may be revised in the future
       based on early user feedback.
    """

    def __init__(self, linear_width, numticks=11, symthresh=0.2,
                 base=10, subs=None):
        """
        Parameters
        ----------
        linear_width : float
            The scale parameter defining the extent
            of the quasi-linear region.
        numticks : int, default: 11
            The approximate number of major ticks that will fit
            along the entire axis
        symthresh : float, default: 0.2
            The fractional threshold beneath which data which covers
            a range that is approximately symmetric about zero
            will have ticks that are exactly symmetric.
        base : int, default: 10
            The number base used for rounding tick locations
            on a logarithmic scale. If this is less than one,
            then rounding is to the nearest integer multiple
            of powers of ten.
        subs : tuple, default: None
            Multiples of the number base, typically used
            for the minor ticks, e.g. (2, 5) when base=10.
        """
        super().__init__()
        # Initialize the AsinhLocator object with provided parameters
        self.linear_width = linear_width
        self.numticks = numticks
        self.symthresh = symthresh
        self.base = base
        self.subs = subs

    def set_params(self, numticks=None, symthresh=None,
                   base=None, subs=None):
        """Set parameters within this locator."""
        # Update parameters if provided, ensuring valid values
        if numticks is not None:
            self.numticks = numticks
        if symthresh is not None:
            self.symthresh = symthresh
        if base is not None:
            self.base = base
        if subs is not None:
            self.subs = subs if len(subs) > 0 else None

    def __call__(self):
        # Get the view interval of the axis
        vmin, vmax = self.axis.get_view_interval()
        # Check if the data range is almost symmetric about zero
        if (vmin * vmax) < 0 and abs(1 + vmax / vmin) < self.symthresh:
            # Data-range appears to be almost symmetric, so round up:
            bound = max(abs(vmin), abs(vmax))
            return self.tick_values(-bound, bound)
        else:
            return self.tick_values(vmin, vmax)
    def tick_values(self, vmin, vmax):
        # 构造一组均匀分布的“屏幕上”位置。

        # 计算线性宽度乘以 arcsinh 处理的最小和最大值
        ymin, ymax = self.linear_width * np.arcsinh(np.array([vmin, vmax])
                                                    / self.linear_width)
        # 在 ymin 和 ymax 之间均匀分布 numticks 个点
        ys = np.linspace(ymin, ymax, self.numticks)
        # 计算零点偏差
        zero_dev = abs(ys / (ymax - ymin))
        if ymin * ymax < 0:
            # 如果坐标轴跨越零点，则确保包含零点刻度
            ys = np.hstack([ys[(zero_dev > 0.5 / self.numticks)], 0.0])

        # 将“屏幕上”网格转换为数据空间：
        xs = self.linear_width * np.sinh(ys / self.linear_width)
        # 标记零点位置
        zero_xs = (ys == 0)

        # 将数据空间值四舍五入为直观的基数-n 数字，分别处理正负值，并且仔细处理零值。
        with np.errstate(divide="ignore"):  # base ** log(0) = base ** -inf = 0.
            if self.base > 1:
                # 计算基数幂
                pows = (np.sign(xs)
                        * self.base ** np.floor(np.log(abs(xs)) / math.log(self.base)))
                # 展开成一维数组，乘以 subs 数组（如果有的话）
                qs = np.outer(pows, self.subs).flatten() if self.subs else pows
            else:  # 当 base <= 1 时，不需要调整 sign(pows)，因为在计算 qs 时会取消掉
                # 处理零点位置，否则取 10 的对数
                pows = np.where(zero_xs, 1, 10**np.floor(np.log10(abs(xs))))
                # 将 xs 除以 pows 后四舍五入
                qs = pows * np.round(xs / pows)
        # 将结果转换为排序后的数组，确保唯一值，并返回
        ticks = np.array(sorted(set(qs)))

        return ticks if len(ticks) >= 2 else np.linspace(vmin, vmax, self.numticks)
class LogitLocator(MaxNLocator):
    """
    Place ticks spaced evenly on a logit scale.
    """

    def __init__(self, minor=False, *, nbins="auto"):
        """
        Parameters
        ----------
        nbins : int or 'auto', optional
            Number of ticks. Only used if minor is False.
        minor : bool, default: False
            Indicate if this locator is for minor ticks or not.
        """
        # 初始化 LogitLocator 对象
        self._minor = minor
        # 调用父类 MaxNLocator 的初始化方法，设置 nbins 和 steps 参数
        super().__init__(nbins=nbins, steps=[1, 2, 5, 10])

    def set_params(self, minor=None, **kwargs):
        """Set parameters within this locator."""
        # 如果 minor 参数不为 None，则设置 self._minor 的值
        if minor is not None:
            self._minor = minor
        # 调用父类 MaxNLocator 的 set_params 方法，设置其他参数
        super().set_params(**kwargs)

    @property
    def minor(self):
        # 返回当前对象的 _minor 属性值
        return self._minor

    @minor.setter
    def minor(self, value):
        # 设置当前对象的 _minor 属性值为给定的 value
        self.set_params(minor=value)

    def nonsingular(self, vmin, vmax):
        standard_minpos = 1e-7
        initial_range = (standard_minpos, 1 - standard_minpos)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        # 如果 vmin 或 vmax 不是有限值，设置它们为 initial_range 的初始值
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = initial_range  # Initial range, no data plotted yet.
        elif vmax <= 0 or vmin >= 1:
            # 当所有值为负数时，vmax <= 0；当所有值大于等于1时，vmin >= 1
            _api.warn_external(
                "Data has no values between 0 and 1, and therefore cannot be "
                "logit-scaled."
            )
            # 设置 vmin 和 vmax 为 initial_range 的初始值
            vmin, vmax = initial_range
        else:
            # 获取最小的正数值（minpos），如果不存在则使用 standard_minpos
            minpos = (
                self.axis.get_minpos()
                if self.axis is not None
                else standard_minpos
            )
            # 如果 minpos 不是有限值，则使用 standard_minpos
            if not np.isfinite(minpos):
                minpos = standard_minpos  # This should never take effect.
            # 如果 vmin 小于等于 0，则将其设置为 minpos
            if vmin <= 0:
                vmin = minpos
            # 如果 vmax 大于等于 1，则将其设置为 1 - minpos
            if vmax >= 1:
                vmax = 1 - minpos
            # 如果 vmin 等于 vmax，则将它们分别调整为 0.1*vmin 和 1-0.1*vmin
            if vmin == vmax:
                vmin, vmax = 0.1 * vmin, 1 - 0.1 * vmin

        return vmin, vmax


class AutoLocator(MaxNLocator):
    """
    Place evenly spaced ticks, with the step size and maximum number of ticks chosen
    automatically.

    This is a subclass of `~matplotlib.ticker.MaxNLocator`, with parameters
    *nbins = 'auto'* and *steps = [1, 2, 2.5, 5, 10]*.
    """
    def __init__(self):
        """
        初始化方法，用于创建对象实例时进行初始化操作。
        """
        # 检查是否启用了经典模式
        if mpl.rcParams['_internal.classic_mode']:
            # 如果启用了经典模式，设置 nbins 为 9
            nbins = 9
            # 设置步长为 [1, 2, 5, 10]
            steps = [1, 2, 5, 10]
        else:
            # 如果未启用经典模式，将 nbins 设置为 'auto'
            nbins = 'auto'
            # 设置步长为 [1, 2, 2.5, 5, 10]
            steps = [1, 2, 2.5, 5, 10]
        # 调用父类的初始化方法，并传入 nbins 和 steps 作为参数
        super().__init__(nbins=nbins, steps=steps)
class AutoMinorLocator(Locator):
    """
    Place evenly spaced minor ticks, with the step size and maximum number of ticks
    chosen automatically.
    
    The Axis scale must be linear with evenly spaced major ticks.
    """

    def __init__(self, n=None):
        """
        *n* is the number of subdivisions of the interval between
        major ticks; e.g., n=2 will place a single minor tick midway
        between major ticks.

        If *n* is omitted or None, the value stored in rcParams will be used.
        In case *n* is set to 'auto', it will be set to 4 or 5. If the distance
        between the major ticks equals 1, 2.5, 5 or 10 it can be perfectly
        divided in 5 equidistant sub-intervals with a length multiple of
        0.05. Otherwise it is divided in 4 sub-intervals.
        """
        self.ndivs = n  # 初始化自动次要定位器对象，设置分隔主要刻度之间的次要刻度数目

    def __call__(self):
        # docstring inherited
        if self.axis.get_scale() == 'log':
            _api.warn_external('AutoMinorLocator does not work on logarithmic scales')
            return []  # 如果坐标轴是对数刻度，则警告并返回空列表

        majorlocs = np.unique(self.axis.get_majorticklocs())  # 获取坐标轴上的所有主要刻度位置
        if len(majorlocs) < 2:
            # Need at least two major ticks to find minor tick locations.
            # TODO: Figure out a way to still be able to display minor ticks with less
            # than two major ticks visible. For now, just display no ticks at all.
            return []  # 如果主要刻度少于两个，则返回空列表，目前无法显示次要刻度

        majorstep = majorlocs[1] - majorlocs[0]  # 计算主要刻度之间的步长

        if self.ndivs is None:
            self.ndivs = mpl.rcParams[
                'ytick.minor.ndivs' if self.axis.axis_name == 'y'
                else 'xtick.minor.ndivs']  # 根据坐标轴类型获取默认的次要刻度数目

        if self.ndivs == 'auto':
            majorstep_mantissa = 10 ** (np.log10(majorstep) % 1)
            ndivs = 5 if np.isclose(majorstep_mantissa, [1, 2.5, 5, 10]).any() else 4  # 根据主要刻度步长设置次要刻度数目
        else:
            ndivs = self.ndivs

        minorstep = majorstep / ndivs  # 计算次要刻度之间的步长

        vmin, vmax = sorted(self.axis.get_view_interval())  # 获取当前视图的最小和最大值
        t0 = majorlocs[0]
        tmin = round((vmin - t0) / minorstep)  # 计算最小次要刻度位置
        tmax = round((vmax - t0) / minorstep) + 1  # 计算最大次要刻度位置
        locs = (np.arange(tmin, tmax) * minorstep) + t0  # 生成次要刻度位置数组

        return self.raise_if_exceeds(locs)  # 返回次要刻度位置数组，超出范围则抛出异常

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            f"Cannot get tick locations for a {type(self).__name__}")  # 当前类不支持获取刻度位置操作，抛出未实现异常
```
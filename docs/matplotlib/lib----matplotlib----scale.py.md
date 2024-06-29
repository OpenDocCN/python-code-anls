# `D:\src\scipysrc\matplotlib\lib\matplotlib\scale.py`

```py
"""
Scales define the distribution of data values on an axis, e.g. a log scaling.
They are defined as subclasses of `ScaleBase`.

See also `.axes.Axes.set_xscale` and the scales examples in the documentation.

See :doc:`/gallery/scales/custom_scale` for a full example of defining a custom
scale.

Matplotlib also supports non-separable transformations that operate on both
`~.axis.Axis` at the same time.  They are known as projections, and defined in
`matplotlib.projections`.
"""

# 导入必要的库和模块
import inspect
import textwrap

import numpy as np

import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
    NullFormatter, ScalarFormatter, LogFormatterSciNotation, LogitFormatter,
    NullLocator, LogLocator, AutoLocator, AutoMinorLocator,
    SymmetricalLogLocator, AsinhLocator, LogitLocator)
from matplotlib.transforms import Transform, IdentityTransform

# 定义一个基类 `ScaleBase`，所有的比例尺都是它的子类
class ScaleBase:
    """
    The base class for all scales.

    Scales are separable transformations, working on a single dimension.

    Subclasses should override

    :attr:`name`
        The scale's name.
    :meth:`get_transform`
        A method returning a `.Transform`, which converts data coordinates to
        scaled coordinates.  This transform should be invertible, so that e.g.
        mouse positions can be converted back to data coordinates.
    :meth:`set_default_locators_and_formatters`
        A method that sets default locators and formatters for an `~.axis.Axis`
        that uses this scale.
    :meth:`limit_range_for_scale`
        An optional method that "fixes" the axis range to acceptable values,
        e.g. restricting log-scaled axes to positive values.
    """

    def __init__(self, axis):
        r"""
        Construct a new scale.

        Notes
        -----
        The following note is for scale implementers.

        For back-compatibility reasons, scales take an `~matplotlib.axis.Axis`
        object as first argument.  However, this argument should not
        be used: a single scale object should be usable by multiple
        `~matplotlib.axis.Axis`\es at the same time.
        """

    def get_transform(self):
        """
        Return the `.Transform` object associated with this scale.
        """
        raise NotImplementedError()

    def set_default_locators_and_formatters(self, axis):
        """
        Set the locators and formatters of *axis* to instances suitable for
        this scale.
        """
        raise NotImplementedError()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Return the range *vmin*, *vmax*, restricted to the
        domain supported by this scale (if any).

        *minpos* should be the minimum positive value in the data.
        This is used by log scales to determine a minimum value.
        """
        return vmin, vmax


class LinearScale(ScaleBase):
    """
    The default linear scale.
    """

    name = 'linear'  # 设置比例尺的名称为 'linear'
    def __init__(self, axis):
        # 这个方法只是为了防止继承基类的构造函数文档字符串，否则会插入到 Axis.set_scale 的文档字符串中
        """
        """  # noqa: D419

    def set_default_locators_and_formatters(self, axis):
        # 继承的文档字符串
        # 设置主要定位器为 AutoLocator
        axis.set_major_locator(AutoLocator())
        # 设置主要格式化器为 ScalarFormatter
        axis.set_major_formatter(ScalarFormatter())
        # 设置次要格式化器为 NullFormatter
        axis.set_minor_formatter(NullFormatter())
        # 根据 rcParams 更新 x 和 y 轴的次要定位器
        if (axis.axis_name == 'x' and mpl.rcParams['xtick.minor.visible'] or
                axis.axis_name == 'y' and mpl.rcParams['ytick.minor.visible']):
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(NullLocator())

    def get_transform(self):
        """
        返回线性缩放的变换，即 `~matplotlib.transforms.IdentityTransform`。
        """
        return IdentityTransform()
class FuncTransform(Transform):
    """
    A simple transform that takes an arbitrary function for the
    forward and inverse transform.
    """

    input_dims = output_dims = 1

    def __init__(self, forward, inverse):
        """
        Parameters
        ----------
        forward : callable
            The forward function for the transform.  This function must have
            an inverse and, for best behavior, be monotonic.
            It must have the signature::

               def forward(values: array-like) -> array-like

        inverse : callable
            The inverse of the forward function.  Signature as ``forward``.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 检查传入的 forward 和 inverse 是否为可调用函数
        if callable(forward) and callable(inverse):
            # 将传入的 forward 和 inverse 函数保存为对象的私有属性
            self._forward = forward
            self._inverse = inverse
        else:
            # 如果不是可调用函数则抛出错误
            raise ValueError('arguments to FuncTransform must be functions')

    def transform_non_affine(self, values):
        # 使用保存的 forward 函数对传入的 values 进行转换
        return self._forward(values)

    def inverted(self):
        # 返回一个新的 FuncTransform 对象，其 forward 和 inverse 函数颠倒顺序
        return FuncTransform(self._inverse, self._forward)


class FuncScale(ScaleBase):
    """
    Provide an arbitrary scale with user-supplied function for the axis.
    """

    name = 'function'

    def __init__(self, axis, functions):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        functions : (callable, callable)
            two-tuple of the forward and inverse functions for the scale.
            The forward function must be monotonic.

            Both functions must have the signature::

               def forward(values: array-like) -> array-like
        """
        # 从 functions 参数中获取 forward 和 inverse 函数
        forward, inverse = functions
        # 创建一个 FuncTransform 对象，用传入的 forward 和 inverse 函数初始化
        transform = FuncTransform(forward, inverse)
        # 将 transform 对象保存为 FuncScale 对象的私有属性
        self._transform = transform

    def get_transform(self):
        """Return the `.FuncTransform` associated with this scale."""
        # 返回保存的 FuncTransform 对象，用于数据转换
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        # docstring inherited
        # 设置主要刻度定位器和格式化程序
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
        # 根据 rcParams 更新 x 和 y 轴的次要刻度定位器
        if (axis.axis_name == 'x' and mpl.rcParams['xtick.minor.visible'] or
                axis.axis_name == 'y' and mpl.rcParams['ytick.minor.visible']):
            axis.set_minor_locator(AutoMinorLocator())
        else:
            axis.set_minor_locator(NullLocator())


class LogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base, nonpositive='clip'):
        # 调用父类的初始化方法
        super().__init__()
        # 检查 base 是否合法
        if base <= 0 or base == 1:
            # 如果 base 不合法则抛出错误
            raise ValueError('The log base cannot be <= 0 or == 1')
        # 将 base 和 nonpositive 参数保存为对象的属性
        self.base = base
        self._clip = _api.check_getitem(
            {"clip": True, "mask": False}, nonpositive=nonpositive)
    # 返回对象的字符串表示，包括类名和初始化参数
    def __str__(self):
        return "{}(base={}, nonpositive={!r})".format(
            type(self).__name__, self.base, "clip" if self._clip else "mask")

    # 在版本3.8中将参数名称'a'重命名为'values'
    def transform_non_affine(self, values):
        # 忽略由于NaN传递给变换函数而产生的无效值
        with np.errstate(divide="ignore", invalid="ignore"):
            # 根据基数选择对数函数，进行对数变换
            log = {np.e: np.log, 2: np.log2, 10: np.log10}.get(self.base)
            if log:  # 如果可行，尽量使用NumPy的单次调用进行变换
                out = log(values)
            else:
                out = np.log(values)
                out /= np.log(self.base)
            if self._clip:
                # SVG规范要求兼容的浏览器必须支持高达3.4e38的值（C浮点数）；
                # 然而实验表明Inkscape（使用cairo进行渲染）会遇到cairo的
                # 24位限制，似乎也适用于Agg。Ghostscript（用于PDF渲染）
                # 似乎会更早地溢出，最大值约为2 ** 15，以使测试通过。
                # 另一方面，在实践中，我们希望将值截断到
                # np.log10(np.nextafter(0, 1)) ~ -323
                # 因此，选择截断值为-1000似乎是安全的。
                out[values <= 0] = -1000
        return out

    # 返回反转的对数变换对象
    def inverted(self):
        return InvertedLogTransform(self.base)
class InvertedLogTransform(Transform):
    input_dims = output_dims = 1

    def __init__(self, base):
        super().__init__()
        self.base = base

    def __str__(self):
        return f"{type(self).__name__}(base={self.base})"

    @_api.rename_parameter("3.8", "a", "values")
    def transform_non_affine(self, values):
        # 使用给定的基数对输入的值进行指数变换
        return np.power(self.base, values)

    def inverted(self):
        # 返回一个基于相同基数的对数变换对象
        return LogTransform(self.base)


class LogScale(ScaleBase):
    """
    A standard logarithmic scale.  Care is taken to only plot positive values.
    """
    name = 'log'

    def __init__(self, axis, *, base=10, subs=None, nonpositive="clip"):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        base : float, default: 10
            The base of the logarithm.
        nonpositive : {'clip', 'mask'}, default: 'clip'
            Determines the behavior for non-positive values. They can either
            be masked as invalid, or clipped to a very small positive number.
        subs : sequence of int, default: None
            Where to place the subticks between each major tick.  For example,
            in a log10 scale, ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place 8
            logarithmically spaced minor ticks between each major tick.
        """
        self._transform = LogTransform(base, nonpositive)
        self.subs = subs

    base = property(lambda self: self._transform.base)

    def set_default_locators_and_formatters(self, axis):
        # docstring inherited
        # 设置主要刻度线定位器和格式化器为对数定位器和对数格式化器
        axis.set_major_locator(LogLocator(self.base))
        axis.set_major_formatter(LogFormatterSciNotation(self.base))
        axis.set_minor_locator(LogLocator(self.base, self.subs))
        axis.set_minor_formatter(
            LogFormatterSciNotation(self.base,
                                    labelOnlyBase=(self.subs is not None)))

    def get_transform(self):
        """Return the `.LogTransform` associated with this scale."""
        # 返回与该比例尺关联的 `.LogTransform` 对象
        return self._transform

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """Limit the domain to positive values."""
        # 将定义域限制为正值
        if not np.isfinite(minpos):
            minpos = 1e-300  # Should rarely (if ever) have a visible effect.

        return (minpos if vmin <= 0 else vmin,
                minpos if vmax <= 0 else vmax)


class FuncScaleLog(LogScale):
    """
    Provide an arbitrary scale with user-supplied function for the axis and
    then put on a logarithmic axes.
    """

    name = 'functionlog'
    def __init__(self, axis, functions, base=10):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            The axis for the scale.
        functions : (callable, callable)
            two-tuple of the forward and inverse functions for the scale.
            The forward function must be monotonic.

            Both functions must have the signature::

                def forward(values: array-like) -> array-like

        base : float, default: 10
            Logarithmic base of the scale.
        """
        forward, inverse = functions  # 解包 functions 元组，得到前向和反向函数
        self.subs = None  # 初始化 self.subs 为 None
        self._transform = FuncTransform(forward, inverse) + LogTransform(base)
        # 创建一个组合变换对象，包括用于前向和反向转换的自定义函数和对数转换的基数 base

    @property
    def base(self):
        return self._transform._b.base  # 返回 LogTransform 的基数 base

    def get_transform(self):
        """Return the `.Transform` associated with this scale."""
        return self._transform  # 返回与该比例尺关联的变换对象
class SymmetricalLogTransform(Transform):
    # 输入和输出维度均为1
    input_dims = output_dims = 1

    def __init__(self, base, linthresh, linscale):
        super().__init__()
        # 如果 base 小于等于1，则引发数值错误
        if base <= 1.0:
            raise ValueError("'base' must be larger than 1")
        # 如果 linthresh 小于等于0，则引发数值错误
        if linthresh <= 0.0:
            raise ValueError("'linthresh' must be positive")
        # 如果 linscale 小于等于0，则引发数值错误
        if linscale <= 0.0:
            raise ValueError("'linscale' must be positive")
        # 初始化基础属性
        self.base = base
        self.linthresh = linthresh
        self.linscale = linscale
        # 计算调整后的线性比例
        self._linscale_adj = (linscale / (1.0 - self.base ** -1))
        # 计算对数的基
        self._log_base = np.log(base)

    # 非仿射变换函数，对输入的数值进行对数变换
    @_api.rename_parameter("3.8", "a", "values")
    def transform_non_affine(self, values):
        # 计算绝对值
        abs_a = np.abs(values)
        # 忽略除法和无效操作的错误
        with np.errstate(divide="ignore", invalid="ignore"):
            # 计算对数变换后的输出
            out = np.sign(values) * self.linthresh * (
                self._linscale_adj +
                np.log(abs_a / self.linthresh) / self._log_base)
            # 确定在线性阈值内的输入
            inside = abs_a <= self.linthresh
        # 在线性阈值内，直接乘以线性比例调整
        out[inside] = values[inside] * self._linscale_adj
        return out

    # 返回反转的对数变换对象
    def inverted(self):
        return InvertedSymmetricalLogTransform(self.base, self.linthresh,
                                               self.linscale)


class InvertedSymmetricalLogTransform(Transform):
    # 输入和输出维度均为1
    input_dims = output_dims = 1

    def __init__(self, base, linthresh, linscale):
        super().__init__()
        # 创建对称对数变换对象
        symlog = SymmetricalLogTransform(base, linthresh, linscale)
        # 初始化基础属性
        self.base = base
        self.linthresh = linthresh
        # 计算反转后的线性阈值
        self.invlinthresh = symlog.transform(linthresh)
        self.linscale = linscale
        # 计算调整后的线性比例
        self._linscale_adj = (linscale / (1.0 - self.base ** -1))

    # 非仿射变换函数，对输入的数值进行反向对数变换
    @_api.rename_parameter("3.8", "a", "values")
    def transform_non_affine(self, values):
        # 计算绝对值
        abs_a = np.abs(values)
        # 忽略除法和无效操作的错误
        with np.errstate(divide="ignore", invalid="ignore"):
            # 计算反向对数变换后的输出
            out = np.sign(values) * self.linthresh * (
                np.power(self.base,
                         abs_a / self.linthresh - self._linscale_adj))
            # 确定在反转线性阈值内的输入
            inside = abs_a <= self.invlinthresh
        # 在反转线性阈值内，直接除以线性比例调整
        out[inside] = values[inside] / self._linscale_adj
        return out

    # 返回对称对数变换对象
    def inverted(self):
        return SymmetricalLogTransform(self.base,
                                       self.linthresh, self.linscale)


class SymmetricalLogScale(ScaleBase):
    """
    The symmetrical logarithmic scale is logarithmic in both the
    positive and negative directions from the origin.

    Since the values close to zero tend toward infinity, there is a
    need to have a range around zero that is linear.  The parameter
    *linthresh* allows the user to specify the size of this range
    (-*linthresh*, *linthresh*).

    Parameters
    ----------
    base : float, default: 10
        The base of the logarithm.

    linthresh : float, default: 2
        Defines the range ``(-x, x)``, within which the plot is linear.
        This avoids having the plot go to infinity around zero.
    """
    subs : sequence of int
        Where to place the subticks between each major tick.
        For example, in a log10 scale: ``[2, 3, 4, 5, 6, 7, 8, 9]`` will place
        8 logarithmically spaced minor ticks between each major tick.

    linscale : float, optional
        This allows the linear range ``(-linthresh, linthresh)`` to be
        stretched relative to the logarithmic range. Its value is the number of
        decades to use for each half of the linear range. For example, when
        *linscale* == 1.0 (the default), the space used for the positive and
        negative halves of the linear range will be equal to one decade in
        the logarithmic range.
    """
    # 设置名称为 'symlog' 的比例尺类型
    name = 'symlog'

    def __init__(self, axis, *, base=10, linthresh=2, subs=None, linscale=1):
        # 使用 SymmetricalLogTransform 对象初始化比例尺
        self._transform = SymmetricalLogTransform(base, linthresh, linscale)
        self.subs = subs

    # 获取 base 属性，返回 SymmetricalLogTransform 对象的基数属性
    base = property(lambda self: self._transform.base)
    # 获取 linthresh 属性，返回 SymmetricalLogTransform 对象的线性阈值属性
    linthresh = property(lambda self: self._transform.linthresh)
    # 获取 linscale 属性，返回 SymmetricalLogTransform 对象的线性比例属性
    linscale = property(lambda self: self._transform.linscale)

    def set_default_locators_and_formatters(self, axis):
        # 使用 SymmetricalLogLocator 对象设置主刻度定位器
        axis.set_major_locator(SymmetricalLogLocator(self.get_transform()))
        # 使用 LogFormatterSciNotation 对象设置主刻度格式化器
        axis.set_major_formatter(LogFormatterSciNotation(self.base))
        # 使用 SymmetricalLogLocator 对象设置次刻度定位器
        axis.set_minor_locator(SymmetricalLogLocator(self.get_transform(),
                                                     self.subs))
        # 使用 NullFormatter 对象设置次刻度格式化器为空

    def get_transform(self):
        """Return the `.SymmetricalLogTransform` associated with this scale."""
        # 返回与此比例尺关联的 SymmetricalLogTransform 对象
        return self._transform
class AsinhTransform(Transform):
    """Inverse hyperbolic-sine transformation used by `.AsinhScale`"""
    # 输入和输出的维度均为1
    input_dims = output_dims = 1

    def __init__(self, linear_width):
        super().__init__()
        # 如果 linear_width 小于等于0，抛出数值错误异常
        if linear_width <= 0.0:
            raise ValueError("Scale parameter 'linear_width' " +
                             "must be strictly positive")
        # 设置线性宽度参数
        self.linear_width = linear_width

    @_api.rename_parameter("3.8", "a", "values")
    # 非仿射变换函数，将值进行反双曲正弦变换
    def transform_non_affine(self, values):
        return self.linear_width * np.arcsinh(values / self.linear_width)

    def inverted(self):
        # 返回反变换对象 InvertedAsinhTransform
        return InvertedAsinhTransform(self.linear_width)


class InvertedAsinhTransform(Transform):
    """Hyperbolic sine transformation used by `.AsinhScale`"""
    # 输入和输出的维度均为1
    input_dims = output_dims = 1

    def __init__(self, linear_width):
        super().__init__()
        # 设置线性宽度参数
        self.linear_width = linear_width

    @_api.rename_parameter("3.8", "a", "values")
    # 非仿射变换函数，将值进行双曲正弦变换
    def transform_non_affine(self, values):
        return self.linear_width * np.sinh(values / self.linear_width)

    def inverted(self):
        # 返回反变换对象 AsinhTransform
        return AsinhTransform(self.linear_width)


class AsinhScale(ScaleBase):
    """
    A quasi-logarithmic scale based on the inverse hyperbolic sine (asinh)

    For values close to zero, this is essentially a linear scale,
    but for large magnitude values (either positive or negative)
    it is asymptotically logarithmic. The transition between these
    linear and logarithmic regimes is smooth, and has no discontinuities
    in the function gradient in contrast to
    the `.SymmetricalLogScale` ("symlog") scale.

    Specifically, the transformation of an axis coordinate :math:`a` is
    :math:`a \\rightarrow a_0 \\sinh^{-1} (a / a_0)` where :math:`a_0`
    is the effective width of the linear region of the transformation.
    In that region, the transformation is
    :math:`a \\rightarrow a + \\mathcal{O}(a^3)`.
    For large values of :math:`a` the transformation behaves as
    :math:`a \\rightarrow a_0 \\, \\mathrm{sgn}(a) \\ln |a| + \\mathcal{O}(1)`.

    .. note::

       This API is provisional and may be revised in the future
       based on early user feedback.
    """

    name = 'asinh'

    # 自动刻度倍增器字典，针对特定基数返回对应的倍增器
    auto_tick_multipliers = {
        3: (2, ),
        4: (2, ),
        5: (2, ),
        8: (2, 4),
        10: (2, 5),
        16: (2, 4, 8),
        64: (4, 16),
        1024: (256, 512)
    }
    def __init__(self, axis, *, linear_width=1.0,
                 base=10, subs='auto', **kwargs):
        """
        Parameters
        ----------
        linear_width : float, default: 1
            The scale parameter (elsewhere referred to as :math:`a_0`)
            defining the extent of the quasi-linear region,
            and the coordinate values beyond which the transformation
            becomes asymptotically logarithmic.
        base : int, default: 10
            The number base used for rounding tick locations
            on a logarithmic scale. If this is less than one,
            then rounding is to the nearest integer multiple
            of powers of ten.
        subs : sequence of int
            Multiples of the number base used for minor ticks.
            If set to 'auto', this will use built-in defaults,
            e.g. (2, 5) for base=10.
        """
        # 调用父类构造函数初始化对象
        super().__init__(axis)
        # 使用 AsinhTransform 类创建一个变换对象，并将 linear_width 参数传递给它
        self._transform = AsinhTransform(linear_width)
        # 将 base 参数转换为整数，并赋值给对象的 _base 属性
        self._base = int(base)
        # 如果 subs 参数为 'auto'，则从预设的自动倍数字典中获取与 base 对应的值，否则直接使用 subs
        if subs == 'auto':
            self._subs = self.auto_tick_multipliers.get(self._base)
        else:
            self._subs = subs

    # 定义 linear_width 属性的只读访问器
    linear_width = property(lambda self: self._transform.linear_width)

    # 返回对象的 _transform 属性，通常用于获取坐标变换对象
    def get_transform(self):
        return self._transform

    # 设置默认的刻度定位器和格式化程序
    def set_default_locators_and_formatters(self, axis):
        # 设置主要刻度定位器为 AsinhLocator，并传递 linear_width 和 base 参数
        axis.set(major_locator=AsinhLocator(self.linear_width,
                                            base=self._base),
                 # 设置次要刻度定位器为 AsinhLocator，并传递 linear_width、base 和 subs 参数
                 minor_locator=AsinhLocator(self.linear_width,
                                            base=self._base,
                                            subs=self._subs),
                 # 设置次要刻度格式化程序为空格式化程序
                 minor_formatter=NullFormatter())
        # 如果 base 大于 1，则设置主要刻度格式化程序为科学计数法格式化程序
        if self._base > 1:
            axis.set_major_formatter(LogFormatterSciNotation(self._base))
        else:
            # 否则，设置主要刻度格式化程序为默认的格式化字符串 '{x:.3g}'
            axis.set_major_formatter('{x:.3g}')
class LogitTransform(Transform):
    input_dims = output_dims = 1  # 设定输入和输出的维度都为1

    def __init__(self, nonpositive='mask'):
        super().__init__()
        _api.check_in_list(['mask', 'clip'], nonpositive=nonpositive)
        self._nonpositive = nonpositive  # 存储非正值处理方式（掩码或剪裁）
        self._clip = {"clip": True, "mask": False}[nonpositive]  # 根据非正值处理方式确定是否剪裁

    @_api.rename_parameter("3.8", "a", "values")
    def transform_non_affine(self, values):
        """logit transform (base 10), masked or clipped"""
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log10(values / (1 - values))  # 计算基于10的logit变换，处理除0和无效值
        if self._clip:  # 根据选择的处理方式进行剪裁
            out[values <= 0] = -1000  # 将小于等于0的值剪裁为-1000
            out[1 <= values] = 1000   # 将大于等于1的值剪裁为1000
        return out  # 返回变换后的值数组

    def inverted(self):
        return LogisticTransform(self._nonpositive)  # 返回逆变换对象

    def __str__(self):
        return f"{type(self).__name__}({self._nonpositive!r})"  # 返回对象的字符串表示


class LogisticTransform(Transform):
    input_dims = output_dims = 1  # 设定输入和输出的维度都为1

    def __init__(self, nonpositive='mask'):
        super().__init__()
        self._nonpositive = nonpositive  # 存储非正值处理方式（掩码或剪裁）

    @_api.rename_parameter("3.8", "a", "values")
    def transform_non_affine(self, values):
        """logistic transform (base 10)"""
        return 1.0 / (1 + 10**(-values))  # 计算基于10的logistic变换

    def inverted(self):
        return LogitTransform(self._nonpositive)  # 返回逆变换对象

    def __str__(self):
        return f"{type(self).__name__}({self._nonpositive!r})"  # 返回对象的字符串表示


class LogitScale(ScaleBase):
    """
    Logit scale for data between zero and one, both excluded.

    This scale is similar to a log scale close to zero and to one, and almost
    linear around 0.5. It maps the interval ]0, 1[ onto ]-infty, +infty[.
    """
    name = 'logit'  # 设置比例尺的名称为'logit'

    def __init__(self, axis, nonpositive='mask', *,
                 one_half=r"\frac{1}{2}", use_overline=False):
        r"""
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            Currently unused.
        nonpositive : {'mask', 'clip'}
            Determines the behavior for values beyond the open interval ]0, 1[.
            They can either be masked as invalid, or clipped to a number very
            close to 0 or 1.
        use_overline : bool, default: False
            Indicate the usage of survival notation (\overline{x}) in place of
            standard notation (1-x) for probability close to one.
        one_half : str, default: r"\frac{1}{2}"
            The string used for ticks formatter to represent 1/2.
        """
        self._transform = LogitTransform(nonpositive)  # 使用给定的非正值处理方式创建LogitTransform对象
        self._use_overline = use_overline  # 存储是否使用上划线表示
        self._one_half = one_half  # 存储表示1/2的字符串用于刻度格式化

    def get_transform(self):
        """Return the `.LogitTransform` associated with this scale."""
        return self._transform  # 返回与此比例尺关联的LogitTransform对象
    # 设置默认的定位器和格式化器
    def set_default_locators_and_formatters(self, axis):
        # 继承的文档字符串
        # 使用 LogitLocator 设置主要刻度线定位器，这些刻度线以对数刻度分布，如 0.01, 0.1, 0.5, 0.9, 0.99 等
        axis.set_major_locator(LogitLocator())
        # 使用 LogitFormatter 设置主要刻度线格式化器，配置了参数 one_half 和 use_overline
        axis.set_major_formatter(
            LogitFormatter(
                one_half=self._one_half,
                use_overline=self._use_overline
            )
        )
        # 使用 LogitLocator 设置次要刻度线定位器，这些刻度线同样以对数刻度分布，但是是次要的
        axis.set_minor_locator(LogitLocator(minor=True))
        # 使用 LogitFormatter 设置次要刻度线格式化器，配置了参数 minor、one_half 和 use_overline
        axis.set_minor_formatter(
            LogitFormatter(
                minor=True,
                one_half=self._one_half,
                use_overline=self._use_overline
            )
        )

    # 为比例尺限定范围
    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        限定域的值在 0 到 1 之间（不包括边界）。
        """
        # 如果最小正数不是有限的，则设定为一个极小值，几乎不会对结果可见
        if not np.isfinite(minpos):
            minpos = 1e-7  # 这个值几乎不会有明显效果
        # 返回限定后的范围，如果 vmin 小于等于 0，则使用 minpos，否则使用 vmin；如果 vmax 大于等于 1，则使用 1 - minpos，否则使用 vmax
        return (minpos if vmin <= 0 else vmin,
                1 - minpos if vmax >= 1 else vmax)
# 定义一个字典，将字符串表示的比例尺名称映射到对应的比例尺类
_scale_mapping = {
    'linear': LinearScale,          # 线性比例尺
    'log':    LogScale,             # 对数比例尺
    'symlog': SymmetricalLogScale,  # 对称对数比例尺
    'asinh':  AsinhScale,           # 反双曲正弦比例尺
    'logit':  LogitScale,           # 逻辑比例尺
    'function': FuncScale,          # 函数比例尺
    'functionlog': FuncScaleLog,    # 函数对数比例尺
}


def get_scale_names():
    """Return the names of the available scales."""
    return sorted(_scale_mapping)   # 返回已定义比例尺名称的排序列表


def scale_factory(scale, axis, **kwargs):
    """
    Return a scale class by name.

    Parameters
    ----------
    scale : {%(names)s}  # 使用可用比例尺名称列表中的某个值
    axis : `~matplotlib.axis.Axis`  # matplotlib 的坐标轴对象
    """
    # 根据比例尺名称获取对应的比例尺类
    scale_cls = _api.check_getitem(_scale_mapping, scale=scale)
    return scale_cls(axis, **kwargs)  # 返回创建的比例尺类实例


if scale_factory.__doc__:
    # 如果有函数文档字符串，则更新其中的参数列表，替换为可用比例尺名称的字符串表示
    scale_factory.__doc__ = scale_factory.__doc__ % {
        "names": ", ".join(map(repr, get_scale_names()))
    }


def register_scale(scale_class):
    """
    Register a new kind of scale.

    Parameters
    ----------
    scale_class : subclass of `ScaleBase`
        The scale to register.
    """
    _scale_mapping[scale_class.name] = scale_class  # 注册新的比例尺类到 `_scale_mapping` 中


def _get_scale_docs():
    """
    Helper function for generating docstrings related to scales.
    """
    docs = []
    for name, scale_class in _scale_mapping.items():
        docstring = inspect.getdoc(scale_class.__init__) or ""
        docs.extend([
            f"    {name!r}",   # 比例尺名称
            "",                # 空行
            textwrap.indent(docstring, " " * 8),  # 缩进的比例尺类构造函数文档字符串
            ""                 # 空行
        ])
    return "\n".join(docs)  # 返回格式化好的文档字符串


# 更新文档字符串中的插值变量，包括可用比例尺名称和相关的比例尺类文档字符串
_docstring.interpd.update(
    scale_type='{%s}' % ', '.join([repr(x) for x in get_scale_names()]),
    scale_docs=_get_scale_docs().rstrip(),
)
```
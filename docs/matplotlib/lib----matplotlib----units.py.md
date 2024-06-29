# `D:\src\scipysrc\matplotlib\lib\matplotlib\units.py`

```py
```python`
"""
这里的类提供了对自定义类与 Matplotlib 的支持，例如那些没有暴露数组接口但知道如何将自己转换为数组的类。
它还支持带单位和单位转换的类。使用案例包括用于自定义对象的转换器，例如包含日期时间对象的列表，
以及具有单位意识的对象。这里不假设任何特定的单位实现；相反，单位实现必须在注册转换器字典中注册，
并提供一个 `ConversionInterface`。例如，这里是一个完整的实现，支持使用原生日期时间对象进行绘图：

    import matplotlib.units as units
    import matplotlib.dates as dates
    import matplotlib.ticker as ticker
    import datetime

    class DateConverter(units.ConversionInterface):

        @staticmethod
        def convert(value, unit, axis):
            "将日期时间值转换为标量或数组。"
            return dates.date2num(value)

        @staticmethod
        def axisinfo(unit, axis):
            "返回主要和次要刻度定位器和格式化程序。"
            if unit != 'date':
                return None
            majloc = dates.AutoDateLocator()
            majfmt = dates.AutoDateFormatter(majloc)
            return units.AxisInfo(majloc=majloc, majfmt=majfmt, label='date')

        @staticmethod
        def default_units(x, axis):
            "返回 x 的默认单位或 None。"
            return 'date'

    # 最后，我们将我们的对象类型注册到 Matplotlib 单位注册表中。
    units.registry[datetime.date] = DateConverter()
"""

from decimal import Decimal
from numbers import Number

import numpy as np
from numpy import ma

from matplotlib import cbook


class ConversionError(TypeError):
    pass


def _is_natively_supported(x):
    """
    判断 *x* 是否是 Matplotlib 原生支持的类型或这些类型对象的数组。
    """
    # Matplotlib 原生支持除了 Decimal 外的所有数值类型。
    if np.iterable(x):
        # 假设列表是同构的，如单位系统中的其他函数。
        for thisx in x:
            if thisx is ma.masked:
                continue
            return isinstance(thisx, Number) and not isinstance(thisx, Decimal)
    else:
        return isinstance(x, Number) and not isinstance(x, Decimal)


class AxisInfo:
    """
    支持默认轴标签、刻度标签和限制的信息类。

    `ConversionInterface.axisinfo` 必须返回此类的实例。
    """
    # 初始化函数，用于创建一个坐标轴对象
    def __init__(self, majloc=None, minloc=None,
                 majfmt=None, minfmt=None, label=None,
                 default_limits=None):
        """
        Parameters
        ----------
        majloc, minloc : Locator, optional
            主刻度和次刻度的刻度定位器对象。
        majfmt, minfmt : Formatter, optional
            主刻度和次刻度的刻度格式化器对象。
        label : str, optional
            坐标轴的默认标签。
        default_limits : optional
            如果没有数据被绘制，则坐标轴的默认最小和最大限制值。

        Notes
        -----
        如果以上任何参数为 ``None``，则坐标轴将使用默认值。
        """
        # 设置对象属性，接收传入的参数
        self.majloc = majloc
        self.minloc = minloc
        self.majfmt = majfmt
        self.minfmt = minfmt
        self.label = label
        self.default_limits = default_limits
class ConversionInterface:
    """
    The minimal interface for a converter to take custom data types (or
    sequences) and convert them to values Matplotlib can use.
    """

    @staticmethod
    def axisinfo(unit, axis):
        """Return an `.AxisInfo` for the axis with the specified units."""
        return None

    @staticmethod
    def default_units(x, axis):
        """Return the default unit for *x* or ``None`` for the given axis."""
        return None

    @staticmethod
    def convert(obj, unit, axis):
        """
        Convert *obj* using *unit* for the specified *axis*.

        If *obj* is a sequence, return the converted sequence.  The output must
        be a sequence of scalars that can be used by the numpy array layer.
        """
        return obj


class DecimalConverter(ConversionInterface):
    """Converter for decimal.Decimal data to float."""

    @staticmethod
    def convert(value, unit, axis):
        """
        Convert Decimals to floats.

        The *unit* and *axis* arguments are not used.

        Parameters
        ----------
        value : decimal.Decimal or iterable
            Decimal or list of Decimal need to be converted
        """
        if isinstance(value, Decimal):
            return float(value)
        # value is Iterable[Decimal]
        elif isinstance(value, ma.MaskedArray):
            return ma.asarray(value, dtype=float)
        else:
            return np.asarray(value, dtype=float)

    # axisinfo and default_units can be inherited as Decimals are Numbers.


class Registry(dict):
    """Register types with conversion interface."""

    def get_converter(self, x):
        """Get the converter interface instance for *x*, or None."""
        # Unpack in case of e.g. Pandas or xarray object
        x = cbook._unpack_to_numpy(x)

        if isinstance(x, np.ndarray):
            # In case x in a masked array, access the underlying data (only its
            # type matters).  If x is a regular ndarray, getdata() just returns
            # the array itself.
            x = np.ma.getdata(x).ravel()
            # If there are no elements in x, infer the units from its dtype
            if not x.size:
                return self.get_converter(np.array([0], dtype=x.dtype))
        for cls in type(x).__mro__:  # Look up in the cache.
            try:
                return self[cls]
            except KeyError:
                pass
        try:  # If cache lookup fails, look up based on first element...
            first = cbook._safe_first_finite(x)
        except (TypeError, StopIteration):
            pass
        else:
            # ... and avoid infinite recursion for pathological iterables for
            # which indexing returns instances of the same iterable class.
            if type(first) is not type(x):
                return self.get_converter(first)
        return None


# Create a Registry instance for type conversions
registry = Registry()

# Register the DecimalConverter to handle conversions involving Decimal objects
registry[Decimal] = DecimalConverter()
```
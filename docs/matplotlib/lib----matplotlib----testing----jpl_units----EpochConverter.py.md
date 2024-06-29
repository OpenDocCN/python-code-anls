# `D:\src\scipysrc\matplotlib\lib\matplotlib\testing\jpl_units\EpochConverter.py`

```py
"""EpochConverter module containing class EpochConverter."""

from matplotlib import cbook, units
import matplotlib.dates as date_ticker

__all__ = ['EpochConverter']


class EpochConverter(units.ConversionInterface):
    """
    Provides Matplotlib conversion functionality for Monte Epoch and Duration
    classes.
    """

    jdRef = 1721425.5  # 定义 Julian Date 参考值

    @staticmethod
    def axisinfo(unit, axis):
        # docstring inherited
        majloc = date_ticker.AutoDateLocator()  # 创建自动日期定位器
        majfmt = date_ticker.AutoDateFormatter(majloc)  # 使用定位器创建自动日期格式化器
        return units.AxisInfo(majloc=majloc, majfmt=majfmt, label=unit)  # 返回轴信息对象

    @staticmethod
    def float2epoch(value, unit):
        """
        Convert a Matplotlib floating-point date into an Epoch of the specified
        units.

        = INPUT VARIABLES
        - value     The Matplotlib floating-point date.
        - unit      The unit system to use for the Epoch.

        = RETURN VALUE
        - Returns the value converted to an Epoch in the specified time system.
        """
        # Delay-load due to circular dependencies.
        import matplotlib.testing.jpl_units as U

        secPastRef = value * 86400.0 * U.UnitDbl(1.0, 'sec')  # 计算距参考日期的秒数
        return U.Epoch(unit, secPastRef, EpochConverter.jdRef)  # 返回 Epoch 对象

    @staticmethod
    def epoch2float(value, unit):
        """
        Convert an Epoch value to a float suitable for plotting as a python
        datetime object.

        = INPUT VARIABLES
        - value    An Epoch or list of Epochs that need to be converted.
        - unit     The units to use for an axis with Epoch data.

        = RETURN VALUE
        - Returns the value parameter converted to floats.
        """
        return value.julianDate(unit) - EpochConverter.jdRef  # 将 Epoch 转换为浮点数表示的日期

    @staticmethod
    def duration2float(value):
        """
        Convert a Duration value to a float suitable for plotting as a python
        datetime object.

        = INPUT VARIABLES
        - value    A Duration or list of Durations that need to be converted.

        = RETURN VALUE
        - Returns the value parameter converted to floats.
        """
        return value.seconds() / 86400.0  # 将 Duration 转换为浮点数表示的日期间隔

    @staticmethod
    def convert(value, unit, axis):
        # docstring inherited

        # Delay-load due to circular dependencies.
        import matplotlib.testing.jpl_units as U

        if not cbook.is_scalar_or_string(value):
            return [EpochConverter.convert(x, unit, axis) for x in value]  # 递归处理列表中的每个元素
        if unit is None:
            unit = EpochConverter.default_units(value, axis)  # 获取默认单位
        if isinstance(value, U.Duration):
            return EpochConverter.duration2float(value)  # 转换 Duration 类型数据
        else:
            return EpochConverter.epoch2float(value, unit)  # 转换 Epoch 类型数据

    @staticmethod
    def default_units(value, axis):
        # docstring inherited
        if cbook.is_scalar_or_string(value):
            return value.frame()  # 返回单个值的参考框架
        else:
            return EpochConverter.default_units(value[0], axis)  # 递归获取列表中第一个元素的默认单位
```
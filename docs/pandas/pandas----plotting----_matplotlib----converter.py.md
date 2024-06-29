# `D:\src\scipysrc\pandas\pandas\plotting\_matplotlib\converter.py`

```
# 导入未来的注解特性，用于类型提示的提前引入
from __future__ import annotations

# 上下文管理工具，用于处理上下文管理相关的操作
import contextlib
# Python 中的日期时间模块，命名为 pydt 方便引用
import datetime as pydt
# 从 datetime 模块中导入 datetime 和 tzinfo 类型
from datetime import (
    datetime,
    tzinfo,
)
# 函数工具，用于创建装饰器
import functools
# 类型提示中的条件引入，用于类型检查时引入特定的模块
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
# 警告相关的模块，用于处理警告信息
import warnings

# matplotlib 主要的命名空间
import matplotlib as mpl
# matplotlib 中用于日期处理的模块
import matplotlib.dates as mdates
# matplotlib 中用于单位处理的模块
import matplotlib.units as munits
# 数值计算库 numpy
import numpy as np

# pandas 的 C 库中导入 lib 模块
from pandas._libs import lib
# pandas 的时间序列相关的类型和函数
from pandas._libs.tslibs import (
    Timestamp,
    to_offset,
)
# pandas 时间序列相关的数据类型
from pandas._libs.tslibs.dtypes import (
    FreqGroup,
    periods_per_day,
)
# pandas 的类型提示相关
from pandas._typing import (
    F,
    npt,
)

# pandas 中的常见数据类型检查函数
from pandas.core.dtypes.common import (
    is_float,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_nested_list_like,
)
# 从 pandas 中导入 Index、Series 和 get_option 函数
from pandas import (
    Index,
    Series,
    get_option,
)
# pandas 中的索引相关模块
import pandas.core.common as com
# pandas 中的日期时间索引相关模块
from pandas.core.indexes.datetimes import date_range
# pandas 中的周期索引相关模块
from pandas.core.indexes.period import (
    Period,
    PeriodIndex,
    period_range,
)
# pandas 中的日期时间工具模块
import pandas.core.tools.datetimes as tools

# 如果类型检查为真，则引入 collections.abc 中的 Generator 类型
if TYPE_CHECKING:
    from collections.abc import Generator
    # 引入 matplotlib.axis 中的 Axis 类型
    from matplotlib.axis import Axis
    # 引入 pandas._libs.tslibs.offsets 中的 BaseOffset 类型
    from pandas._libs.tslibs.offsets import BaseOffset

# 缓存被我们覆盖的单位的字典
_mpl_units = {}

# 返回一个包含类型转换器的元组列表
def get_pairs():
    pairs = [
        (Timestamp, DatetimeConverter),
        (Period, PeriodConverter),
        (pydt.datetime, DatetimeConverter),
        (pydt.date, DatetimeConverter),
        (pydt.time, TimeConverter),
        (np.datetime64, DatetimeConverter),
    ]
    return pairs


# 注册 pandas 转换器的装饰器函数
def register_pandas_matplotlib_converters(func: F) -> F:
    """
    Decorator applying pandas_converters.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with pandas_converters():
            return func(*args, **kwargs)

    return cast(F, wrapper)


# pandas 转换器的上下文管理器，用于在绘图时注册 pandas 的转换器
@contextlib.contextmanager
def pandas_converters() -> Generator[None, None, None]:
    """
    Context manager registering pandas' converters for a plot.

    See Also
    --------
    register_pandas_matplotlib_converters : Decorator that applies this.
    """
    # 获取绘图时是否注册 pandas 转换器的选项值
    value = get_option("plotting.matplotlib.register_converters")

    if value:
        # 如果值为 True 或 "auto"，则注册转换器
        register()
    try:
        yield
    finally:
        if value == "auto":
            # 如果值为 "auto"，则取消注册转换器
            deregister()


# 注册转换器函数，将 pandas 转换器注册到 matplotlib 中
def register() -> None:
    pairs = get_pairs()
    for type_, cls in pairs:
        # 如果类型已经注册过并且不是我们要注册的类，则缓存先前的转换器
        if type_ in munits.registry and not isinstance(munits.registry[type_], cls):
            previous = munits.registry[type_]
            _mpl_units[type_] = previous
        # 替换为 pandas 转换器
        munits.registry[type_] = cls()


# 取消注册转换器函数，从 matplotlib 中移除 pandas 转换器
def deregister() -> None:
    for type_, cls in get_pairs():
        # 根据类型直接移除我们的类的转换器
        if type(munits.registry.get(type_)) is cls:
            munits.registry.pop(type_)

    # 恢复旧的键
    # 遍历 _mpl_units 字典中的每个单位和对应的格式化器
    for unit, formatter in _mpl_units.items():
        # 检查格式化器的类型是否不属于 DatetimeConverter、PeriodConverter、TimeConverter 中的任何一种
        if type(formatter) not in {DatetimeConverter, PeriodConverter, TimeConverter}:
            # 将当前单位的格式化器注册到 munits.registry 中，以确保操作是幂等的（不重复注册）
            munits.registry[unit] = formatter
# 将时间对象转换为秒数（浮点数表示）
def _to_ordinalf(tm: pydt.time) -> float:
    # 计算总秒数，包括小时、分钟、秒和微秒的部分
    tot_sec = tm.hour * 3600 + tm.minute * 60 + tm.second + tm.microsecond / 10**6
    return tot_sec


# 将各种时间表示形式转换为秒数的统一接口
def time2num(d):
    if isinstance(d, str):
        parsed = Timestamp(d)  # 解析时间戳字符串
        return _to_ordinalf(parsed.time())  # 返回时间对象的秒数表示
    if isinstance(d, pydt.time):
        return _to_ordinalf(d)  # 直接返回时间对象的秒数表示
    return d  # 对于其他类型的输入直接返回原始值


# 时间单位转换类，实现了Matplotlib的ConversionInterface接口
class TimeConverter(munits.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        valid_types = (str, pydt.time)
        # 根据值的类型进行不同的转换操作
        if isinstance(value, valid_types) or is_integer(value) or is_float(value):
            return time2num(value)  # 调用time2num将值转换为秒数
        if isinstance(value, Index):
            return value.map(time2num)  # 对索引对象的每个值都调用time2num进行转换
        if isinstance(value, (list, tuple, np.ndarray, Index)):
            return [time2num(x) for x in value]  # 对列表或数组中的每个元素调用time2num进行转换
        return value  # 如果值类型不符合预期，则直接返回原始值


    @staticmethod
    def axisinfo(unit, axis) -> munits.AxisInfo | None:
        if unit != "time":
            return None  # 如果单位不是"time"，则返回空

        majloc = mpl.ticker.AutoLocator()  # 创建自动刻度定位器
        majfmt = TimeFormatter(majloc)  # 使用TimeFormatter格式化刻度标签
        return munits.AxisInfo(majloc=majloc, majfmt=majfmt, label="time")  # 返回包含刻度信息的AxisInfo对象


    @staticmethod
    def default_units(x, axis) -> str:
        return "time"  # 返回默认单位为"time"


# 时间格式化类，继承自Matplotlib的ticker.Formatter类
class TimeFormatter(mpl.ticker.Formatter):
    def __init__(self, locs) -> None:
        self.locs = locs  # 初始化时刻度位置信息

    def __call__(self, x, pos: int | None = 0) -> str:
        """
        Return the time of day as a formatted string.

        Parameters
        ----------
        x : float
            The time of day specified as seconds since 00:00 (midnight),
            with up to microsecond precision.
        pos
            Unused

        Returns
        -------
        str
            A string in HH:MM:SS.mmmuuu format. Microseconds,
            milliseconds and seconds are only displayed if non-zero.
        """
        fmt = "%H:%M:%S.%f"
        s = int(x)
        msus = round((x - s) * 10**6)
        ms = msus // 1000
        us = msus % 1000
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        _, h = divmod(h, 24)
        if us != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)  # 格式化带微秒的时间字符串
        elif ms != 0:
            return pydt.time(h, m, s, msus).strftime(fmt)[:-3]  # 格式化带毫秒的时间字符串
        elif s != 0:
            return pydt.time(h, m, s).strftime("%H:%M:%S")  # 格式化只带秒的时间字符串

        return pydt.time(h, m).strftime("%H:%M")  # 格式化只带小时和分钟的时间字符串


# 期间转换类，继承自Matplotlib的日期转换器DateConverter类
class PeriodConverter(mdates.DateConverter):
    @staticmethod
    def convert(values, units, axis):
        if is_nested_list_like(values):
            values = [PeriodConverter._convert_1d(v, units, axis) for v in values]
        else:
            values = PeriodConverter._convert_1d(values, units, axis)
        return values
    def _convert_1d(values, units, axis):
        # 检查 axis 是否具有 `freq` 属性，若没有则抛出 TypeError 异常
        if not hasattr(axis, "freq"):
            raise TypeError("Axis must have `freq` set to convert to Periods")
        
        # 定义有效的类型列表，用于判断 values 的类型
        valid_types = (str, datetime, Period, pydt.date, pydt.time, np.datetime64)
        
        # 使用警告过滤器忽略特定的 FutureWarning 类别警告
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "Period with BDay freq is deprecated", category=FutureWarning
            )
            warnings.filterwarnings(
                "ignore", r"PeriodDtype\[B\] is deprecated", category=FutureWarning
            )
            
            # 根据 values 的类型进行不同的转换处理
            if (
                isinstance(values, valid_types)
                or is_integer(values)
                or is_float(values)
            ):
                # 如果 values 是有效类型之一，直接调用 get_datevalue 函数转换为日期值
                return get_datevalue(values, axis.freq)
            
            elif isinstance(values, PeriodIndex):
                # 如果 values 是 PeriodIndex 类型，则调整频率后返回其整数表示
                return values.asfreq(axis.freq).asi8
            
            elif isinstance(values, Index):
                # 如果 values 是 Index 类型，则对每个元素调用 get_datevalue 函数并返回结果
                return values.map(lambda x: get_datevalue(x, axis.freq))
            
            elif lib.infer_dtype(values, skipna=False) == "period":
                # 如果 values 的推断类型为 'period'，将 ndarray[period] 转换为 PeriodIndex
                return PeriodIndex(values, freq=axis.freq).asi8
            
            elif isinstance(values, (list, tuple, np.ndarray, Index)):
                # 如果 values 是列表、元组、数组或 Index 类型，则对每个元素调用 get_datevalue 函数并返回结果列表
                return [get_datevalue(x, axis.freq) for x in values]
        
        # 如果以上条件均不匹配，则直接返回 values
        return values
# 定义函数，根据日期和频率返回对应的日期序数值
def get_datevalue(date, freq):
    # 如果输入的日期是 Period 类型，则返回按频率调整后的日期序数值
    if isinstance(date, Period):
        return date.asfreq(freq).ordinal
    # 如果日期是字符串、datetime 对象、pydt.date 对象、pydt.time 对象、np.datetime64 对象中的一种，则创建 Period 对象，并返回其序数值
    elif isinstance(date, (str, datetime, pydt.date, pydt.time, np.datetime64)):
        return Period(date, freq).ordinal
    # 如果日期是整数、浮点数或者是包含单个元素的 np.ndarray 或 Index 对象，则直接返回该日期
    elif (
        is_integer(date)
        or is_float(date)
        or (isinstance(date, (np.ndarray, Index)) and (date.size == 1))
    ):
        return date
    # 如果日期为 None，则返回 None
    elif date is None:
        return None
    # 如果日期类型无法识别，则抛出 ValueError 异常
    raise ValueError(f"Unrecognizable date '{date}'")


# Datetime Conversion
# 继承自 mdates.DateConverter 类的 DatetimeConverter 类
class DatetimeConverter(mdates.DateConverter):
    # 静态方法，将输入的 values 转换为指定单位下的日期数值
    @staticmethod
    def convert(values, unit, axis):
        # 如果 values 是嵌套的列表或类似列表的结构，则对每个元素调用 _convert_1d 方法
        if is_nested_list_like(values):
            values = [DatetimeConverter._convert_1d(v, unit, axis) for v in values]
        else:
            # 否则，直接调用 _convert_1d 方法
            values = DatetimeConverter._convert_1d(values, unit, axis)
        return values

    # 静态方法，将一维的 values 转换为指定单位下的日期数值
    @staticmethod
    def _convert_1d(values, unit, axis):
        # 内部函数，尝试将 values 转换为日期数值，如果失败则返回原值
        def try_parse(values):
            try:
                return mdates.date2num(tools.to_datetime(values))
            except Exception:
                return values

        # 如果 values 是 datetime、pydt.date、np.datetime64 或 pydt.time 类型，则直接转换为日期数值
        if isinstance(values, (datetime, pydt.date, np.datetime64, pydt.time)):
            return mdates.date2num(values)
        # 如果 values 是整数或浮点数，则直接返回
        elif is_integer(values) or is_float(values):
            return values
        # 如果 values 是字符串，则尝试解析为日期数值
        elif isinstance(values, str):
            return try_parse(values)
        # 如果 values 是列表、元组、np.ndarray、Index 或 Series 类型，则处理为日期数值
        elif isinstance(values, (list, tuple, np.ndarray, Index, Series)):
            # 如果 values 是 Series 类型，则转换为 DatetimeIndex 以获取 asi8
            if isinstance(values, Series):
                values = Index(values)
            # 如果 values 是 Index 类型，则获取其值
            if isinstance(values, Index):
                values = values.values
            # 如果 values 不是 np.ndarray 类型，则转换为 np.ndarray
            if not isinstance(values, np.ndarray):
                values = com.asarray_tuplesafe(values)

            # 如果 values 是整数类型或浮点数类型的数组，则直接返回
            if is_integer_dtype(values) or is_float_dtype(values):
                return values

            # 尝试将 values 转换为日期数值
            try:
                values = tools.to_datetime(values)
            except Exception:
                pass

            # 将 values 转换为日期数值
            values = mdates.date2num(values)

        return values

    # 静态方法，返回指定单位下的坐标轴信息
    @staticmethod
    def axisinfo(unit: tzinfo | None, axis) -> munits.AxisInfo:
        """
        Return the :class:`~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a tzinfo instance or None.
        The *axis* argument is required but not used.
        """
        tz = unit

        # 设置主要刻度定位器为 PandasAutoDateLocator 类的实例
        majloc = PandasAutoDateLocator(tz=tz)
        # 设置主要刻度格式化器为 PandasAutoDateFormatter 类的实例
        majfmt = PandasAutoDateFormatter(majloc, tz=tz)
        # 设置默认的日期范围为 2000-01-01 到 2010-01-01
        datemin = pydt.date(2000, 1, 1)
        datemax = pydt.date(2010, 1, 1)

        # 返回 AxisInfo 对象，包含上述设置的主要刻度定位器、主要刻度格式化器、空标签和默认的日期范围
        return munits.AxisInfo(
            majloc=majloc, majfmt=majfmt, label="", default_limits=(datemin, datemax)
        )


# 类 PandasAutoDateFormatter 继承自 mdates.AutoDateFormatter
class PandasAutoDateFormatter(mdates.AutoDateFormatter):
    # 定义构造函数，初始化对象
    def __init__(self, locator, tz=None, defaultfmt: str = "%Y-%m-%d") -> None:
        # 调用父类 mdates.AutoDateFormatter 的构造函数进行初始化
        mdates.AutoDateFormatter.__init__(self, locator, tz, defaultfmt)
class PandasAutoDateLocator(mdates.AutoDateLocator):
    # 自定义日期定位器，继承自matplotlib的AutoDateLocator类

    def get_locator(self, dmin, dmax):
        """Pick the best locator based on a distance."""
        # 根据时间范围选择最佳的定位器

        tot_sec = (dmax - dmin).total_seconds()
        # 计算时间范围的总秒数差

        if abs(tot_sec) < self.minticks:
            # 如果时间范围小于最小刻度，则选择毫秒级定位器

            self._freq = -1
            # 设置频率为-1，表示毫秒级定位器
            locator = MilliSecondLocator(self.tz)
            # 创建毫秒级定位器对象
            locator.set_axis(self.axis)
            # 设置定位器对象的轴

            # error: Item "None" of "Axis | _DummyAxis | _AxisWrapper | None"
            # has no attribute "get_data_interval"
            # 错误："None"对象没有"get_data_interval"属性
            locator.axis.set_view_interval(  # type: ignore[union-attr]
                *self.axis.get_view_interval()  # type: ignore[union-attr]
            )
            # 设置定位器对象的视图间隔
            locator.axis.set_data_interval(  # type: ignore[union-attr]
                *self.axis.get_data_interval()  # type: ignore[union-attr]
            )
            # 设置定位器对象的数据间隔
            return locator
            # 返回毫秒级定位器对象

        return mdates.AutoDateLocator.get_locator(self, dmin, dmax)
        # 否则调用父类的方法选择自动日期定位器

    def _get_unit(self):
        # 获取单位

        return MilliSecondLocator.get_unit_generic(self._freq)
        # 返回毫秒级定位器的通用单位


class MilliSecondLocator(mdates.DateLocator):
    # 毫秒级日期定位器，继承自matplotlib的DateLocator类
    UNIT = 1.0 / (24 * 3600 * 1000)
    # 单位为每天的毫秒数

    def __init__(self, tz) -> None:
        # 初始化方法
        mdates.DateLocator.__init__(self, tz)
        # 调用父类的初始化方法
        self._interval = 1.0
        # 设置间隔为1.0

    def _get_unit(self):
        # 获取单位

        return self.get_unit_generic(-1)
        # 返回通用单位为负1的毫秒级定位器单位

    @staticmethod
    def get_unit_generic(freq):
        # 静态方法：获取通用单位

        unit = mdates.RRuleLocator.get_unit_generic(freq)
        # 调用父类的方法获取通用单位
        if unit < 0:
            return MilliSecondLocator.UNIT
            # 如果单位小于0，返回毫秒级定位器的单位
        return unit
        # 否则返回获取到的单位
    def __call__(self):
        # if no data have been set, this will tank with a ValueError
        try:
            # 将视图限制转换为 datetime 类型的最小值和最大值
            dmin, dmax = self.viewlim_to_dt()
        except ValueError:
            # 如果转换失败，则返回空列表
            return []

        # We need to cap at the endpoints of valid datetime
        # 将有效日期时间的端点进行上限设置
        nmax, nmin = mdates.date2num((dmax, dmin))

        # 计算毫秒级 ticks 的最大数目
        num = (nmax - nmin) * 86400 * 1000
        max_millis_ticks = 6
        for interval in [1, 10, 50, 100, 200, 500]:
            if num <= interval * (max_millis_ticks - 1):
                self._interval = interval
                break
            # We went through the whole loop without breaking, default to 1
            # 如果循环完毕仍未中断，则默认间隔为 1000.0
            self._interval = 1000.0

        # 根据单位和间隔估算 ticks 数量
        estimate = (nmax - nmin) / (self._get_unit() * self._get_interval())

        if estimate > self.MAXTICKS * 2:
            # 如果估算的 ticks 数超过阈值，则引发运行时错误
            raise RuntimeError(
                "MillisecondLocator estimated to generate "
                f"{estimate:d} ticks from {dmin} to {dmax}: exceeds Locator.MAXTICKS"
                f"* 2 ({self.MAXTICKS * 2:d}) "
            )

        interval = self._get_interval()
        freq = f"{interval}ms"
        tz = self.tz.tzname(None)
        st = dmin.replace(tzinfo=None)
        ed = dmax.replace(tzinfo=None)
        # 生成时间范围内的日期序列，并将其转换为对象类型
        all_dates = date_range(start=st, end=ed, freq=freq, tz=tz).astype(object)

        try:
            if len(all_dates) > 0:
                # 如果日期序列不为空，则检查是否超过最大 ticks 数，如果没有则返回定位值
                locs = self.raise_if_exceeds(mdates.date2num(all_dates))
                return locs
        except Exception:  # pragma: no cover
            pass

        # 将日期时间转换为数字格式并返回
        lims = mdates.date2num([dmin, dmax])
        return lims

    def _get_interval(self):
        # 返回当前对象的间隔值
        return self._interval

    def autoscale(self):
        """
        Set the view limits to include the data range.
        """
        # 将数据限制转换为 datetime 类型的最小值和最大值
        dmin, dmax = self.datalim_to_dt()

        # 将最小值和最大值转换为数字格式
        vmin = mdates.date2num(dmin)
        vmax = mdates.date2num(dmax)

        # 对最小值和最大值进行非奇异性处理并返回结果
        return self.nonsingular(vmin, vmax)
# Fixed frequency dynamic tick locators and formatters

# -------------------------------------------------------------------------
# --- Locators ---
# -------------------------------------------------------------------------

def _get_default_annual_spacing(nyears) -> tuple[int, int]:
    """
    Returns a default spacing between consecutive ticks for annual data.
    """
    if nyears < 11:
        (min_spacing, maj_spacing) = (1, 1)
    elif nyears < 20:
        (min_spacing, maj_spacing) = (1, 2)
    elif nyears < 50:
        (min_spacing, maj_spacing) = (1, 5)
    elif nyears < 100:
        (min_spacing, maj_spacing) = (5, 10)
    elif nyears < 200:
        (min_spacing, maj_spacing) = (5, 25)
    elif nyears < 600:
        (min_spacing, maj_spacing) = (10, 50)
    else:
        factor = nyears // 1000 + 1
        (min_spacing, maj_spacing) = (factor * 20, factor * 100)
    return (min_spacing, maj_spacing)

def _period_break(dates: PeriodIndex, period: str) -> npt.NDArray[np.intp]:
    """
    Returns the indices where the given period changes.

    Parameters
    ----------
    dates : PeriodIndex
        Array of intervals to monitor.
    period : str
        Name of the period to monitor.
    """
    # Generate a mask indicating where the period changes
    mask = _period_break_mask(dates, period)
    return np.nonzero(mask)[0]

def _period_break_mask(dates: PeriodIndex, period: str) -> npt.NDArray[np.bool_]:
    # Determine the current and previous periods and create a mask where they differ
    current = getattr(dates, period)
    previous = getattr(dates - 1 * dates.freq, period)
    return current != previous

def has_level_label(label_flags: npt.NDArray[np.intp], vmin: float) -> bool:
    """
    Returns true if the ``label_flags`` indicate there is at least one label
    for this level.

    If the minimum view limit is not an exact integer, then the first tick
    label won't be shown, so we must adjust for that.
    """
    # Check if label_flags is empty or if the first tick label won't be shown due to vmin
    if label_flags.size == 0 or (
        label_flags.size == 1 and label_flags[0] == 0 and vmin % 1 > 0.0
    ):
        return False
    else:
        return True

def _get_periods_per_ymd(freq: BaseOffset) -> tuple[int, int, int]:
    # Obtain the dtype code from the frequency and determine period counts per year, month, and day
    # Error: "BaseOffset" has no attribute "_period_dtype_code"
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]
    freq_group = FreqGroup.from_period_dtype_code(dtype_code)

    ppd = -1  # placeholder for above-day freqs

    if dtype_code >= FreqGroup.FR_HR.value:
        # Error: "BaseOffset" has no attribute "_creso"
        ppd = periods_per_day(freq._creso)  # type: ignore[attr-defined]
        ppm = 28 * ppd
        ppy = 365 * ppd
    elif freq_group == FreqGroup.FR_BUS:
        ppm = 19
        ppy = 261
    elif freq_group == FreqGroup.FR_DAY:
        ppm = 28
        ppy = 365
    elif freq_group == FreqGroup.FR_WK:
        ppm = 3
        ppy = 52
    elif freq_group == FreqGroup.FR_MTH:
        ppm = 1
        ppy = 12
    elif freq_group == FreqGroup.FR_QTR:
        ppm = -1  # placeholder
        ppy = 4
    elif freq_group == FreqGroup.FR_ANN:
        # 如果频率组为年度频率，则设定 ppm 为占位符 -1，ppy 为 1
        ppm = -1  # placeholder
        ppy = 1
    else:
        # 如果频率组不是年度频率，则抛出未实现错误，显示不支持的频率类型
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")

    # 返回结果 ppd, ppm, ppy
    return ppd, ppm, ppy
@functools.cache
# 使用 functools 模块的 cache 装饰器，缓存函数的结果以提高性能
def _daily_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # 从频率对象中获取周期数据类型代码
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]

    # 获取每天、每月和每年的周期数
    periodsperday, periodspermonth, periodsperyear = _get_periods_per_ymd(freq)

    # 保存原始的 vmin 用于后续使用
    vmin_orig = vmin
    (vmin, vmax) = (int(vmin), int(vmax))
    span = vmax - vmin + 1

    with warnings.catch_warnings():
        # 忽略特定的警告信息
        warnings.filterwarnings(
            "ignore", "Period with BDay freq is deprecated", category=FutureWarning
        )
        warnings.filterwarnings(
            "ignore", r"PeriodDtype\[B\] is deprecated", category=FutureWarning
        )
        # 生成日期范围对象，起始和结束使用 vmin 和 vmax
        dates_ = period_range(
            start=Period(ordinal=vmin, freq=freq),
            end=Period(ordinal=vmax, freq=freq),
            freq=freq,
        )

    # 初始化输出信息数组
    info = np.zeros(
        span, dtype=[("val", np.int64), ("maj", bool), ("min", bool), ("fmt", "|S20")]
    )
    info["val"][:] = dates_.asi8  # 将日期的整数表示存入数组
    info["fmt"][:] = ""  # 初始化格式字符串为空字符串
    info["maj"][[0, -1]] = True  # 将第一个和最后一个设为主要刻度
    # .. and set some shortcuts
    info_maj = info["maj"]  # 设置主要刻度的引用
    info_min = info["min"]  # 设置次要刻度的引用
    info_fmt = info["fmt"]  # 设置格式字符串的引用

    def first_label(label_flags):
        # 如果第一个刻度标志为0，并且有多于一个刻度，并且 vmin_orig 为小数，则返回第二个刻度
        if (label_flags[0] == 0) and (label_flags.size > 1) and ((vmin_orig % 1) > 0.0):
            return label_flags[1]
        else:
            return label_flags[0]

    # 情况1：少于一个月
    # 情况2：少于三个月
    elif span <= periodsperyear // 4:
        month_start = _period_break(dates_, "month")  # 找到月份开始的位置
        info_maj[month_start] = True  # 标记月份开始位置为主要刻度
        if dtype_code < FreqGroup.FR_HR.value:
            info["min"] = True  # 如果数据类型代码小于 FreqGroup.FR_HR 的值，则设置所有次要刻度为真
        else:
            day_start = _period_break(dates_, "day")  # 找到天数开始的位置
            info["min"][day_start] = True  # 将天数开始位置的次要刻度设为真
        week_start = _period_break(dates_, "week")  # 找到周开始的位置
        year_start = _period_break(dates_, "year")  # 找到年份开始的位置
        info_fmt[week_start] = "%d"  # 设置周开始位置的格式字符串
        info_fmt[month_start] = "\n\n%b"  # 设置月份开始位置的格式字符串
        info_fmt[year_start] = "\n\n%b\n%Y"  # 设置年份开始位置的格式字符串
        if not has_level_label(year_start, vmin_orig):
            if not has_level_label(month_start, vmin_orig):
                info_fmt[first_label(week_start)] = "\n\n%b\n%Y"  # 根据条件设置周刻度的格式字符串
            else:
                info_fmt[first_label(month_start)] = "\n\n%b\n%Y"  # 根据条件设置月刻度的格式字符串
    # 情况3：少于14个月
    elif span <= 1.15 * periodsperyear:
        year_start = _period_break(dates_, "year")  # 找到年份开始的位置
        month_start = _period_break(dates_, "month")  # 找到月份开始的位置
        week_start = _period_break(dates_, "week")  # 找到周开始的位置
        info_maj[month_start] = True  # 标记月份开始位置为主要刻度
        info_min[week_start] = True  # 标记周开始位置为次要刻度
        info_min[year_start] = False  # 取消年份开始位置的次要刻度
        info_min[month_start] = False  # 取消月份开始位置的次要刻度
        info_fmt[month_start] = "%b"  # 设置月份开始位置的格式字符串
        info_fmt[year_start] = "%b\n%Y"  # 设置年份开始位置的格式字符串
        if not has_level_label(year_start, vmin_orig):
            info_fmt[first_label(month_start)] = "%b\n%Y"  # 根据条件设置月刻度的格式字符串
    # 情况4：少于2.5年
    # Case 4. Less than 4 years .................
    elif span <= 4 * periodsperyear:
        # 找到最近的年份分界点
        year_start = _period_break(dates_, "year")
        # 找到最近的月份分界点
        month_start = _period_break(dates_, "month")
        # 将年份分界点标记为主要分界点
        info_maj[year_start] = True
        # 将月份分界点标记为次要分界点
        info_min[month_start] = True
        # 取消年份分界点的次要标记
        info_min[year_start] = False

        # 获取月份分界点对应的月份
        month_break = dates_[month_start].month
        # 如果月份是1月或者7月，将其标记为主要分界点
        jan_or_jul = month_start[(month_break == 1) | (month_break == 7)]
        info_fmt[jan_or_jul] = "%b"
        # 年份分界点的格式为"月份\n年份"
        info_fmt[year_start] = "%b\n%Y"
    # Case 5. Less than 11 years ................
    elif span <= 11 * periodsperyear:
        # 找到最近的年份分界点
        year_start = _period_break(dates_, "year")
        # 找到最近的季度分界点
        quarter_start = _period_break(dates_, "quarter")
        # 将年份分界点标记为主要分界点
        info_maj[year_start] = True
        # 将季度分界点标记为次要分界点
        info_min[quarter_start] = True
        # 取消年份分界点的次要标记
        info_min[year_start] = False
        # 年份分界点的格式为"年份"
        info_fmt[year_start] = "%Y"
    # Case 6. More than 12 years ................
    else:
        # 找到最近的年份分界点
        year_start = _period_break(dates_, "year")
        # 获取年份分界点所在的年份
        year_break = dates_[year_start].year
        # 计算跨度对应的年数
        nyears = span / periodsperyear
        # 根据年数获取默认的主要和次要年度间隔
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        # 根据主要年度间隔找到对应的主要分界点索引
        major_idx = year_start[(year_break % maj_anndef == 0)]
        info_maj[major_idx] = True
        # 根据次要年度间隔找到对应的次要分界点索引
        minor_idx = year_start[(year_break % min_anndef == 0)]
        info_min[minor_idx] = True
        # 年份分界点的格式为"年份"
        info_fmt[major_idx] = "%Y"

    # 返回包含所有分界点信息的字典
    return info
# 使用 functools.cache 装饰器缓存函数结果，以加快后续调用速度
@functools.cache
def _monthly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    # 获取年、月、日的周期数
    _, _, periodsperyear = _get_periods_per_ymd(freq)

    # 保存原始的 vmin 值
    vmin_orig = vmin
    # 将输入的浮点数 vmin 和 vmax 转换为整数
    (vmin, vmax) = (int(vmin), int(vmax))
    # 计算整数范围的长度
    span = vmax - vmin + 1

    # 初始化输出数组 info，用于存储每个值的信息：值、是否为主要刻度、是否为次要刻度、格式字符串
    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    # 设置 info 数组的值字段为从 vmin 到 vmax 的整数序列
    info["val"] = np.arange(vmin, vmax + 1)
    dates_ = info["val"]
    info["fmt"] = ""
    # 找到每年的起始点索引
    year_start = (dates_ % 12 == 0).nonzero()[0]
    info_maj = info["maj"]
    info_fmt = info["fmt"]

    # 根据范围长度与年周期数的关系，确定日期格式和刻度类型
    if span <= 1.15 * periodsperyear:
        # 小于等于1.15倍年周期数时，设定年为主要刻度，月为次要刻度，并设置日期格式为月份的简称
        info_maj[year_start] = True
        info["min"] = True

        info_fmt[:] = "%b"
        info_fmt[year_start] = "%b\n%Y"

        # 如果未包含原始的 vmin 则修改格式
        if not has_level_label(year_start, vmin_orig):
            if dates_.size > 1:
                idx = 1
            else:
                idx = 0
            info_fmt[idx] = "%b\n%Y"

    elif span <= 2.5 * periodsperyear:
        # 小于等于2.5倍年周期数时，设定年为主要刻度，季度为次要刻度，并设置日期格式
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        # TODO: 检查以下内容：info['fmt'] 是否真的是需要的？
        #  2023-09-15 在 test_finder_monthly 中可以到达此点
        info["fmt"][quarter_start] = True
        info["min"] = True

        info_fmt[quarter_start] = "%b"
        info_fmt[year_start] = "%b\n%Y"

    elif span <= 4 * periodsperyear:
        # 小于等于4倍年周期数时，设定年为主要刻度，并设置日期格式
        info_maj[year_start] = True
        info["min"] = True

        jan_or_jul = (dates_ % 12 == 0) | (dates_ % 12 == 6)
        info_fmt[jan_or_jul] = "%b"
        info_fmt[year_start] = "%b\n%Y"

    elif span <= 11 * periodsperyear:
        # 小于等于11倍年周期数时，设定年为主要刻度，季度为次要刻度，并设置日期格式为年份
        quarter_start = (dates_ % 3 == 0).nonzero()
        info_maj[year_start] = True
        info["min"][quarter_start] = True

        info_fmt[year_start] = "%Y"

    else:
        # 大于11倍年周期数时，根据周期数计算默认的年刻度间隔，并设置年为主要刻度，并设置日期格式为年份
        nyears = span / periodsperyear
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        years = dates_[year_start] // 12 + 1
        major_idx = year_start[(years % maj_anndef == 0)]
        info_maj[major_idx] = True
        info["min"][year_start[(years % min_anndef == 0)]] = True

        info_fmt[major_idx] = "%Y"

    return info


# 使用 functools.cache 装饰器缓存函数结果，以加快后续调用速度
@functools.cache
def _quarterly_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    _, _, periodsperyear = _get_periods_per_ymd(freq)
    # 保存原始的 vmin 值
    vmin_orig = vmin
    # 将输入的浮点数 vmin 和 vmax 转换为整数
    (vmin, vmax) = (int(vmin), int(vmax))
    # 计算整数范围的长度
    span = vmax - vmin + 1

    # 初始化输出数组 info，用于存储每个值的信息：值、是否为主要刻度、是否为次要刻度、格式字符串
    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    # 设置 info 数组的值字段为从 vmin 到 vmax 的整数序列
    info["val"] = np.arange(vmin, vmax + 1)
    info["fmt"] = ""
    dates_ = info["val"]
    info_maj = info["maj"]
    info_fmt = info["fmt"]
    year_start = (dates_ % 4 == 0).nonzero()[0]
    # 如果时间跨度小于等于3.5倍每年的周期数
    if span <= 3.5 * periodsperyear:
        # 在 info_maj 中标记 year_start 处为 True
        info_maj[year_start] = True
        # 在 info 字典中设置 "min" 键为 True

        # 设置 info_fmt 数组的格式为 "Q%q"
        info_fmt[:] = "Q%q"
        # 修改 info_fmt 数组在 year_start 处的格式为 "Q%q\n%F"
        info_fmt[year_start] = "Q%q\n%F"

        # 如果没有等级标签，并且 dates_ 的大小大于1
        if not has_level_label(year_start, vmin_orig):
            if dates_.size > 1:
                # 设置 idx 为 1
                idx = 1
            else:
                # 设置 idx 为 0
                idx = 0
            # 修改 info_fmt 数组在 idx 处的格式为 "Q%q\n%F"

    # 如果时间跨度小于等于11倍每年的周期数但大于3.5倍
    elif span <= 11 * periodsperyear:
        # 在 info_maj 中标记 year_start 处为 True
        info_maj[year_start] = True
        # 在 info 字典中设置 "min" 键为 True
        info["min"] = True
        # 修改 info_fmt 数组在 year_start 处的格式为 "%F"

    # 如果时间跨度大于11倍每年的周期数
    else:
        # 计算年份，假设 1970 年起始
        years = dates_[year_start] // 4 + 1970
        # 计算年份数
        nyears = span / periodsperyear
        # 调用函数计算默认的年度间隔，返回最小和最大的年度间隔
        (min_anndef, maj_anndef) = _get_default_annual_spacing(nyears)
        # 找到主要的年度索引，这些年份是年度间隔的整数倍
        major_idx = year_start[(years % maj_anndef == 0)]
        # 在 info_maj 中标记主要年度索引处为 True
        info_maj[major_idx] = True
        # 在 info 字典中设置 "min" 键，这些年份是最小年度间隔的整数倍，为 True
        info["min"][year_start[(years % min_anndef == 0)]] = True
        # 修改 info_fmt 数组在主要年度索引处的格式为 "%F"

    # 返回 info 字典作为函数的结果
    return info
@functools.cache
def _annual_finder(vmin: float, vmax: float, freq: BaseOffset) -> np.ndarray:
    """
    # 使用 functools.cache 装饰器缓存结果，避免重复计算
    Note: small difference here vs other finders in adding 1 to vmax
    """
    # 将输入参数 vmin 和 vmax 转换为整数，并将 vmax 增加 1
    (vmin, vmax) = (int(vmin), int(vmax + 1))
    # 计算数值范围的长度
    span = vmax - vmin + 1

    # 创建一个包含各种字段的零数组
    info = np.zeros(
        span, dtype=[("val", int), ("maj", bool), ("min", bool), ("fmt", "|S8")]
    )
    # 设置 'val' 字段为从 vmin 到 vmax 的整数序列
    info["val"] = np.arange(vmin, vmax + 1)
    # 初始化 'fmt' 字段为空字符串
    info["fmt"] = ""
    # 将 'val' 字段赋值给 dates_
    dates_ = info["val"]

    # 获取默认的年度主要和次要间隔
    (min_anndef, maj_anndef) = _get_default_annual_spacing(span)
    # 找到主要刻度的索引
    major_idx = dates_ % maj_anndef == 0
    # 找到次要刻度的索引
    minor_idx = dates_ % min_anndef == 0
    # 将对应索引位置的 'maj' 字段设置为 True
    info["maj"][major_idx] = True
    # 将对应索引位置的 'min' 字段设置为 True
    info["min"][minor_idx] = True
    # 将对应索引位置的 'fmt' 字段设置为 "%Y"
    info["fmt"][major_idx] = "%Y"

    # 返回包含信息的数组
    return info


def get_finder(freq: BaseOffset):
    """
    # 获取适合频率的时间定位器函数
    error: "BaseOffset" has no attribute "_period_dtype_code"
    """
    # 获取频率的周期数据类型代码
    dtype_code = freq._period_dtype_code  # type: ignore[attr-defined]
    # 根据周期数据类型代码创建频率组对象
    fgroup = FreqGroup.from_period_dtype_code(dtype_code)

    # 根据频率组选择合适的时间定位器函数
    if fgroup == FreqGroup.FR_ANN:
        return _annual_finder
    elif fgroup == FreqGroup.FR_QTR:
        return _quarterly_finder
    elif fgroup == FreqGroup.FR_MTH:
        return _monthly_finder
    elif (dtype_code >= FreqGroup.FR_BUS.value) or fgroup == FreqGroup.FR_WK:
        return _daily_finder
    else:  # pragma: no cover
        # 抛出未实现的错误，如果频率不受支持
        raise NotImplementedError(f"Unsupported frequency: {dtype_code}")


class TimeSeries_DateLocator(mpl.ticker.Locator):  # pyright: ignore[reportAttributeAccessIssue]
    """
    # 控制 :class:`Series` 轴上的刻度定位器

    Parameters
    ----------
    freq : BaseOffset
        有效的频率指示符.
    minor_locator : {False, True}, optional
        是否为次要刻度定位器 (True) 或不是.
    dynamic_mode : {True, False}, optional
        是否在动态模式下工作.
    base : {int}, optional
    quarter : {int}, optional
    month : {int}, optional
    day : {int}, optional
    """

    axis: Axis

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        base: int = 1,
        quarter: int = 1,
        month: int = 1,
        day: int = 1,
        plot_obj=None,
    ) -> None:
        # 将 freq 转换为偏移量对象，以确保正确性
        freq = to_offset(freq, is_period=True)
        self.freq = freq
        # 初始化基础值，季度、月份和日期
        self.base = base
        (self.quarter, self.month, self.day) = (quarter, month, day)
        # 设置是否为次要刻度定位器和动态模式
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        # 根据频率获取相应的时间定位器函数
        self.finder = get_finder(freq)

    def _get_default_locs(self, vmin, vmax):
        """返回默认刻度位置."""
        # 获取适合范围的时间定位器
        locator = self.finder(vmin, vmax, self.freq)

        # 如果是次要刻度定位器，则返回次要刻度位置
        if self.isminor:
            return np.compress(locator["min"], locator["val"])
        # 否则返回主要刻度位置
        return np.compress(locator["maj"], locator["val"])
    def __call__(self):
        """Return the locations of the ticks."""
        # axis calls Locator.set_axis inside set_m<xxxx>_formatter

        # 获取当前轴的视图间隔
        vi = tuple(self.axis.get_view_interval())
        vmin, vmax = vi
        # 如果最大值小于最小值，则交换二者的位置
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        # 如果是动态的情况下，获取默认的刻度位置
        if self.isdynamic:
            locs = self._get_default_locs(vmin, vmax)
        else:  # pragma: no cover
            # 否则，根据基准(base)计算最小值的下一个基准倍数
            base = self.base
            (d, m) = divmod(vmin, base)
            vmin = (d + 1) * base
            # 使用基准(base)创建一个范围列表，增量为base
            # 错误：无法匹配 "range" 的重载变体，参数类型为 "float", "float", "int"
            locs = list(range(vmin, vmax + 1, base))  # type: ignore[call-overload]
        return locs

    def autoscale(self):
        """
        Sets the view limits to the nearest multiples of base that contain the
        data.
        """
        # 需要 matplotlib >= 0.98.0

        # 获取数据的数据间隔
        (vmin, vmax) = self.axis.get_data_interval()

        # 使用默认的刻度位置
        locs = self._get_default_locs(vmin, vmax)
        # 获取刻度位置的第一个和最后一个
        (vmin, vmax) = locs[[0, -1]]
        # 如果最小值等于最大值，则扩展视图范围
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        # 返回经过处理后的非奇异视图范围
        return mpl.transforms.nonsingular(vmin, vmax)
# -------------------------------------------------------------------------
# --- Formatter ---
# -------------------------------------------------------------------------

# 定义一个自定义的 Matplotlib 刻度格式化器，用于处理时间序列的日期索引

class TimeSeries_DateFormatter(mpl.ticker.Formatter):  # pyright: ignore[reportAttributeAccessIssue]
    """
    Formats the ticks along an axis controlled by a :class:`PeriodIndex`.

    Parameters
    ----------
    freq : BaseOffset
        Valid frequency specifier.
    minor_locator : bool, default False
        Whether the current formatter should apply to minor ticks (True) or
        major ticks (False).
    dynamic_mode : bool, default True
        Whether the formatter works in dynamic mode or not.
    """

    axis: Axis

    def __init__(
        self,
        freq: BaseOffset,
        minor_locator: bool = False,
        dynamic_mode: bool = True,
        plot_obj=None,
    ) -> None:
        # 将频率转换为 BaseOffset 对象，以适应时间周期
        freq = to_offset(freq, is_period=True)
        self.format = None
        self.freq = freq
        self.locs: list[Any] = []  # unused, for matplotlib compat
        self.formatdict: dict[Any, Any] | None = None
        self.isminor = minor_locator
        self.isdynamic = dynamic_mode
        self.offset = 0
        self.plot_obj = plot_obj
        # 根据频率获取相应的日期查找器
        self.finder = get_finder(freq)

    def _set_default_format(self, vmin, vmax):
        """Returns the default ticks spacing."""
        # 使用日期查找器获取默认的刻度格式
        info = self.finder(vmin, vmax, self.freq)

        if self.isminor:
            format = np.compress(info["min"] & np.logical_not(info["maj"]), info)
        else:
            format = np.compress(info["maj"], info)
        self.formatdict = {x: f for (x, _, _, f) in format}
        return self.formatdict

    def set_locs(self, locs) -> None:
        """Sets the locations of the ticks"""
        # 设置刻度的位置，但实际上并不使用这些位置。这只是为了与 matplotlib 兼容。
        self.locs = locs

        (vmin, vmax) = tuple(self.axis.get_view_interval())
        if vmax < vmin:
            (vmin, vmax) = (vmax, vmin)
        # 根据视图间隔设置默认的刻度格式
        self._set_default_format(vmin, vmax)

    def __call__(self, x, pos: int | None = 0) -> str:
        if self.formatdict is None:
            return ""
        else:
            fmt = self.formatdict.pop(x, "")
            if isinstance(fmt, np.bytes_):
                fmt = fmt.decode("utf-8")
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Period with BDay freq is deprecated",
                    category=FutureWarning,
                )
                # 根据给定的 x 值和频率创建 Period 对象
                period = Period(ordinal=int(x), freq=self.freq)
            assert isinstance(period, Period)
            # 使用期间对象格式化日期，并返回格式化后的字符串
            return period.strftime(fmt)


class TimeSeries_TimedeltaFormatter(mpl.ticker.Formatter):  # pyright: ignore[reportAttributeAccessIssue]
    """
    Formats the ticks along an axis controlled by a :class:`TimedeltaIndex`.
    """

    axis: Axis

    @staticmethod
    # 将秒数转换为格式化的时间字符串 'D days HH:MM:SS.F'
    def format_timedelta_ticks(x, pos, n_decimals: int) -> str:
        """
        Convert seconds to 'D days HH:MM:SS.F'
        """
        # 将秒数 x 拆分为整秒部分 s 和纳秒部分 ns
        s, ns = divmod(x, 10**9)  # TODO(non-nano): this looks like it assumes ns
        # 将整秒 s 转换为分钟 m 和剩余秒数 s
        m, s = divmod(s, 60)
        # 将分钟 m 转换为小时 h 和剩余分钟 m
        h, m = divmod(m, 60)
        # 将小时 h 转换为天数 d 和剩余小时 h
        d, h = divmod(h, 24)
        # 计算小数部分的位数 decimals
        decimals = int(ns * 10 ** (n_decimals - 9))
        # 格式化成 HH:MM:SS 格式的时间字符串
        s = f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
        # 如果有小数部分，则添加到时间字符串末尾
        if n_decimals > 0:
            s += f".{decimals:0{n_decimals}d}"
        # 如果存在天数 d，则在时间字符串前加上 'D days '
        if d != 0:
            s = f"{int(d):d} days {s}"
        return s

    # 调用函数，用于格式化时间间隔刻度的显示
    def __call__(self, x, pos: int | None = 0) -> str:
        # 获取当前坐标轴的视图间隔范围
        (vmin, vmax) = tuple(self.axis.get_view_interval())
        # 计算需要显示的小数位数 n_decimals
        n_decimals = min(int(np.ceil(np.log10(100 * 10**9 / abs(vmax - vmin)))), 9)
        # 调用 format_timedelta_ticks 函数，返回格式化后的时间字符串
        return self.format_timedelta_ticks(x, pos, n_decimals)
```
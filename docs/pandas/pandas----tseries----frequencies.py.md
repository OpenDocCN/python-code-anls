# `D:\src\scipysrc\pandas\pandas\tseries\frequencies.py`

```
# 从未来导入注释，确保代码可以在较旧的 Python 版本中运行
from __future__ import annotations

# 导入类型检查模块，用于类型提示
from typing import TYPE_CHECKING

# 导入 NumPy 库
import numpy as np

# 导入 pandas 内部的 C 扩展库
from pandas._libs import lib
from pandas._libs.algos import unique_deltas
from pandas._libs.tslibs import (
    Timestamp,
    get_unit_from_dtype,
    periods_per_day,
    tz_convert_from_utc,
)
# 导入 pandas 内部的 C 扩展日历模块
from pandas._libs.tslibs.ccalendar import (
    DAYS,
    MONTH_ALIASES,
    MONTH_NUMBERS,
    MONTHS,
    int_to_weekday,
)
# 导入 pandas 内部时间序列相关模块
from pandas._libs.tslibs.dtypes import OFFSET_TO_PERIOD_FREQSTR
from pandas._libs.tslibs.fields import (
    build_field_sarray,
    month_position_check,
)
from pandas._libs.tslibs.offsets import (
    DateOffset,
    Day,
    to_offset,
)
from pandas._libs.tslibs.parsing import get_rule_month
# 导入 pandas 内部的缓存只读装饰器
from pandas.util._decorators import cache_readonly

# 导入 pandas 核心模块
from pandas.core.dtypes.common import is_numeric_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    PeriodDtype,
)
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

# 导入 pandas 核心算法模块
from pandas.core.algorithms import unique

# 如果是类型检查阶段，则导入额外的类型提示
if TYPE_CHECKING:
    from pandas._typing import npt
    from pandas import (
        DatetimeIndex,
        Series,
        TimedeltaIndex,
    )
    from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin

# --------------------------------------------------------------------
# Offset related functions

# 需要后缀的偏移量列表
_need_suffix = ["QS", "BQE", "BQS", "YS", "BYE", "BYS"]

# 为每个后缀和月份组合生成偏移量字符串映射
for _prefix in _need_suffix:
    for _m in MONTHS:
        key = f"{_prefix}-{_m}"
        OFFSET_TO_PERIOD_FREQSTR[key] = OFFSET_TO_PERIOD_FREQSTR[_prefix]

# 为年度和季度偏移量创建别名映射
for _prefix in ["Y", "Q"]:
    for _m in MONTHS:
        _alias = f"{_prefix}-{_m}"
        OFFSET_TO_PERIOD_FREQSTR[_alias] = _alias

# 为每周中的每天生成偏移量字符串映射
for _d in DAYS:
    OFFSET_TO_PERIOD_FREQSTR[f"W-{_d}"] = f"W-{_d}"


def get_period_alias(offset_str: str) -> str | None:
    """
    Alias to closest period strings BQ->Q etc.
    返回与偏移量字符串最接近的周期字符串 BQ->Q 等。
    """
    return OFFSET_TO_PERIOD_FREQSTR.get(offset_str, None)


# ---------------------------------------------------------------------
# Period codes


def infer_freq(
    index: DatetimeIndex | TimedeltaIndex | Series | DatetimeLikeArrayMixin,
) -> str | None:
    """
    Infer the most likely frequency given the input index.
    推断输入索引的最可能频率。

    Parameters
    ----------
    index : DatetimeIndex, TimedeltaIndex, Series or array-like
      If passed a Series will use the values of the series (NOT THE INDEX).

    Returns
    -------
    str or None
        None if no discernible frequency.
        如果没有可识别的频率，则返回 None。

    Raises
    ------
    TypeError
        If the index is not datetime-like.
        如果索引不是类似日期时间的类型。
    ValueError
        If there are fewer than three values.
        如果少于三个值。

    Examples
    --------
    >>> idx = pd.date_range(start="2020/12/01", end="2020/12/30", periods=30)
    >>> pd.infer_freq(idx)
    'D'
    """
    from pandas.core.api import DatetimeIndex  # 导入 pandas 核心 API 中的 DatetimeIndex 类
    # 检查 index 是否为 ABCSeries 的实例
    if isinstance(index, ABCSeries):
        # 获取 index 对象的值
        values = index._values
        # 如果值的 dtype 不是日期时间相关的类型或者是 object 类型，则抛出 TypeError 异常
        if not (
            lib.is_np_dtype(values.dtype, "mM")
            or isinstance(values.dtype, DatetimeTZDtype)
            or values.dtype == object
        ):
            raise TypeError(
                "cannot infer freq from a non-convertible dtype "
                f"on a Series of {index.dtype}"
            )
        # 将 index 赋值为其值
        index = values

    # 声明 inferer 变量，用于后续频率推断
    inferer: _FrequencyInferer

    # 如果 index 没有 dtype 属性，则不做任何操作
    if not hasattr(index, "dtype"):
        pass
    # 如果 index 的 dtype 是 PeriodDtype，则抛出 TypeError 异常
    elif isinstance(index.dtype, PeriodDtype):
        raise TypeError(
            "PeriodIndex given. Check the `freq` attribute "
            "instead of using infer_freq."
        )
    # 如果 index 的 dtype 是时间增量（timedelta）相关的类型
    elif lib.is_np_dtype(index.dtype, "m"):
        # 创建 TimedeltaIndex 或 TimedeltaArray 对应的频率推断器
        inferer = _TimedeltaFrequencyInferer(index)
        # 返回推断得到的频率
        return inferer.get_freq()

    # 如果 index 的 dtype 是数值类型，则抛出 TypeError 异常
    elif is_numeric_dtype(index.dtype):
        raise TypeError(
            f"cannot infer freq from a non-convertible index of dtype {index.dtype}"
        )

    # 如果 index 不是 DatetimeIndex 类型，则将其转换为 DatetimeIndex 类型
    if not isinstance(index, DatetimeIndex):
        index = DatetimeIndex(index)

    # 使用 index 创建频率推断器
    inferer = _FrequencyInferer(index)
    # 返回推断得到的频率
    return inferer.get_freq()
class _FrequencyInferer:
    """
    Not sure if I can avoid the state machine here
    """

    def __init__(self, index) -> None:
        self.index = index  # 初始化对象的索引属性

        # 获取索引的 i8values 属性，这是索引的 64 位整数表示
        self.i8values = index.asi8

        # 对于 get_unit_from_dtype 函数，需要传入索引数据的 dtype，对于带时区信息的索引，与 index.dtype 不同
        if isinstance(index, ABCIndex):
            # 错误：Union[ExtensionArray, ndarray[Any, Any]] 的 "ndarray[Any, Any]" 类型的项没有 "_ndarray" 属性
            self._creso = get_unit_from_dtype(
                index._data._ndarray.dtype  # type: ignore[union-attr]
            )
        else:
            # 对于非带时区信息的索引，使用 index._ndarray 的 dtype
            self._creso = get_unit_from_dtype(index._ndarray.dtype)

        # 如果索引具有时区信息，则将 i8values 转换为本地时间
        if hasattr(index, "tz"):
            if index.tz is not None:
                self.i8values = tz_convert_from_utc(
                    self.i8values, index.tz, reso=self._creso
                )

        # 如果索引长度小于 3，则抛出值错误异常
        if len(index) < 3:
            raise ValueError("Need at least 3 dates to infer frequency")

        # 检查索引是否单调递增或单调递减
        self.is_monotonic = (
            self.index._is_monotonic_increasing or self.index._is_monotonic_decreasing
        )

    @cache_readonly
    def deltas(self) -> npt.NDArray[np.int64]:
        # 返回唯一的时间间隔数组
        return unique_deltas(self.i8values)

    @cache_readonly
    def deltas_asi8(self) -> npt.NDArray[np.int64]:
        # 返回索引的 asi8 属性的唯一时间间隔数组
        # 注意：这里不能使用 self.i8values，因为在 __init__ 中可能已经对时区进行了转换
        return unique_deltas(self.index.asi8)

    @cache_readonly
    def is_unique(self) -> bool:
        # 返回索引的时间间隔数组是否只有一个唯一值
        return len(self.deltas) == 1

    @cache_readonly
    def is_unique_asi8(self) -> bool:
        # 返回索引的 asi8 属性的时间间隔数组是否只有一个唯一值
        return len(self.deltas_asi8) == 1
    def get_freq(self) -> str | None:
        """
        Find the appropriate frequency string to describe the inferred
        frequency of self.i8values

        Returns
        -------
        str or None
        """
        # 如果序列不是单调递增或索引不唯一，则返回 None
        if not self.is_monotonic or not self.index._is_unique:
            return None

        # 获取第一个时间间隔值
        delta = self.deltas[0]
        # 计算每天的周期数
        ppd = periods_per_day(self._creso)
        # 如果 delta 不为零且是周期数的整数倍，则推断为每日频率
        if delta and _is_multiple(delta, ppd):
            return self._infer_daily_rule()

        # 如果小时间隔为 [1, 17], [1, 65], [1, 17, 65] 中的一个，则推断为商业小时频率
        if self.hour_deltas in ([1, 17], [1, 65], [1, 17, 65]):
            return "bh"

        # 如果 self.is_unique_asi8 为 False，则返回 None
        if not self.is_unique_asi8:
            return None

        # 获取第一个原始时间戳间隔值
        delta = self.deltas_asi8[0]
        # 计算每小时的周期数、每分钟的周期数、每秒的周期数
        pph = ppd // 24
        ppm = pph // 60
        pps = ppm // 60
        # 根据 delta 是否是周期数的整数倍，推断为小时、分钟、秒、毫秒、微秒或纳秒频率
        if _is_multiple(delta, pph):
            # 小时
            return _maybe_add_count("h", delta / pph)
        elif _is_multiple(delta, ppm):
            # 分钟
            return _maybe_add_count("min", delta / ppm)
        elif _is_multiple(delta, pps):
            # 秒
            return _maybe_add_count("s", delta / pps)
        elif _is_multiple(delta, (pps // 1000)):
            # 毫秒
            return _maybe_add_count("ms", delta / (pps // 1000))
        elif _is_multiple(delta, (pps // 1_000_000)):
            # 微秒
            return _maybe_add_count("us", delta / (pps // 1_000_000))
        else:
            # 纳秒
            return _maybe_add_count("ns", delta)

    @cache_readonly
    def day_deltas(self) -> list[int]:
        # 计算每天的周期数
        ppd = periods_per_day(self._creso)
        # 返回每个时间间隔值除以周期数的列表
        return [x / ppd for x in self.deltas]

    @cache_readonly
    def hour_deltas(self) -> list[int]:
        # 计算每小时的周期数
        pph = periods_per_day(self._creso) // 24
        # 返回每个时间间隔值除以周期数的列表
        return [x / pph for x in self.deltas]

    @cache_readonly
    def fields(self) -> np.ndarray:  # structured array of fields
        # 使用 i8values 构建结构化字段数组，使用指定的分辨率
        return build_field_sarray(self.i8values, reso=self._creso)

    @cache_readonly
    def rep_stamp(self) -> Timestamp:
        # 返回以指定单位为单位的时间戳对象，使用第一个 i8values 作为时间戳值
        return Timestamp(self.i8values[0], unit=self.index.unit)

    def month_position_check(self) -> str | None:
        # 检查月份位置，返回结果字符串或 None
        return month_position_check(self.fields, self.index.dayofweek)

    @cache_readonly
    def mdiffs(self) -> npt.NDArray[np.int64]:
        # 计算年份加月份的唯一时间间隔，并返回其数组
        nmonths = self.fields["Y"] * 12 + self.fields["M"]
        return unique_deltas(nmonths.astype("i8"))

    @cache_readonly
    def ydiffs(self) -> npt.NDArray[np.int64]:
        # 计算年份的唯一时间间隔，并返回其数组
        return unique_deltas(self.fields["Y"].astype("i8"))
    # 推断每日规则的方法，返回一个字符串或者 None
    def _infer_daily_rule(self) -> str | None:
        # 获取年度规则
        annual_rule = self._get_annual_rule()
        if annual_rule:
            # 获取年份差异中的第一个值
            nyears = self.ydiffs[0]
            # 获取当前时间戳对应的月份的别名
            month = MONTH_ALIASES[self.rep_stamp.month]
            # 组合年度规则和月份别名成为别名字符串
            alias = f"{annual_rule}-{month}"
            # 返回添加计数后的别名
            return _maybe_add_count(alias, nyears)

        # 获取季度规则
        quarterly_rule = self._get_quarterly_rule()
        if quarterly_rule:
            # 获取月份差异中的第一个值除以 3 得到季度数
            nquarters = self.mdiffs[0] / 3
            # 根据当前时间戳的月份计算对应的月份别名
            mod_dict = {0: 12, 2: 11, 1: 10}
            month = MONTH_ALIASES[mod_dict[self.rep_stamp.month % 3]]
            # 组合季度规则和月份别名成为别名字符串
            alias = f"{quarterly_rule}-{month}"
            # 返回添加计数后的别名
            return _maybe_add_count(alias, nquarters)

        # 获取月度规则
        monthly_rule = self._get_monthly_rule()
        if monthly_rule:
            # 返回添加计数后的月度规则别名
            return _maybe_add_count(monthly_rule, self.mdiffs[0])

        # 如果是唯一的规则，则获取每日规则
        if self.is_unique:
            return self._get_daily_rule()

        # 如果是工作日规则
        if self._is_business_daily():
            return "B"

        # 获取每周中的第几周规则
        wom_rule = self._get_wom_rule()
        if wom_rule:
            return wom_rule

        # 默认返回 None
        return None

    # 获取每日规则的方法，返回一个字符串或者 None
    def _get_daily_rule(self) -> str | None:
        # 计算每天的周期数
        ppd = periods_per_day(self._creso)
        # 计算天数除以每天的周期数得到天数
        days = self.deltas[0] / ppd
        if days % 7 == 0:
            # 如果天数是 7 的倍数，则返回每周规则
            wd = int_to_weekday[self.rep_stamp.weekday()]
            alias = f"W-{wd}"
            # 返回添加计数后的每周规则别名
            return _maybe_add_count(alias, days / 7)
        else:
            # 否则返回添加计数后的每日规则别名
            return _maybe_add_count("D", days)

    # 获取年度规则的方法，返回一个字符串或者 None
    def _get_annual_rule(self) -> str | None:
        # 如果年份差异的长度大于 1，则返回 None
        if len(self.ydiffs) > 1:
            return None

        # 如果月份字段的唯一值的长度大于 1，则返回 None
        if len(unique(self.fields["M"])) > 1:
            return None

        # 进行月份位置检查
        pos_check = self.month_position_check()

        # 如果位置检查结果为 None，则返回 None，否则返回对应的年度规则
        if pos_check is None:
            return None
        else:
            return {"cs": "YS", "bs": "BYS", "ce": "YE", "be": "BYE"}.get(pos_check)

    # 获取季度规则的方法，返回一个字符串或者 None
    def _get_quarterly_rule(self) -> str | None:
        # 如果月份差异的长度大于 1，则返回 None
        if len(self.mdiffs) > 1:
            return None

        # 如果月份差异的第一个值除以 3 不等于 0，则返回 None
        if not self.mdiffs[0] % 3 == 0:
            return None

        # 进行月份位置检查
        pos_check = self.month_position_check()

        # 如果位置检查结果为 None，则返回 None，否则返回对应的季度规则
        if pos_check is None:
            return None
        else:
            return {"cs": "QS", "bs": "BQS", "ce": "QE", "be": "BQE"}.get(pos_check)

    # 获取月度规则的方法，返回一个字符串或者 None
    def _get_monthly_rule(self) -> str | None:
        # 如果月份差异的长度大于 1，则返回 None
        if len(self.mdiffs) > 1:
            return None
        
        # 进行月份位置检查
        pos_check = self.month_position_check()

        # 如果位置检查结果为 None，则返回 None，否则返回对应的月度规则
        if pos_check is None:
            return None
        else:
            return {"cs": "MS", "bs": "BMS", "ce": "ME", "be": "BME"}.get(pos_check)
    # 判断是否为工作日报告
    def _is_business_daily(self) -> bool:
        # 快速检查：如果日期间隔不是 [1, 3]，则不可能是工作日报告
        if self.day_deltas != [1, 3]:
            return False

        # 可能是工作日报告，但需要进一步确认
        # 获取第一个日期的星期几
        first_weekday = self.index[0].weekday()
        # 计算相邻日期之间的差异
        shifts = np.diff(self.i8values)
        # 根据自定义函数计算每天的时间段数量
        ppd = periods_per_day(self._creso)
        shifts = np.floor_divide(shifts, ppd)
        # 计算每个日期对应的星期几
        weekdays = np.mod(first_weekday + np.cumsum(shifts), 7)

        # 返回是否符合工作日报告的条件
        return bool(
            np.all(
                ((weekdays == 0) & (shifts == 3))
                | ((weekdays > 0) & (weekdays <= 4) & (shifts == 1))
            )
        )

    # 获取每月第几周的规则
    def _get_wom_rule(self) -> str | None:
        # 获取索引日期的星期几列表并去重
        weekdays = unique(self.index.weekday)
        # 如果星期几的种类超过一个，则无法确定规则，返回 None
        if len(weekdays) > 1:
            return None

        # 获取每个日期对应的月中第几周并去重
        week_of_months = unique((self.index.day - 1) // 7)
        # 只尝试推断到第四周。参见 issue #9425
        week_of_months = week_of_months[week_of_months < 4]
        # 如果周数为空或超过一个，则无法确定规则，返回 None
        if len(week_of_months) == 0 or len(week_of_months) > 1:
            return None

        # 获取第一个周数，并转换为星期几的文字表示
        week = week_of_months[0] + 1
        wd = int_to_weekday[weekdays[0]]

        # 返回规则字符串，例如 "WOM-3Wed"
        return f"WOM-{week}{wd}"
class _TimedeltaFrequencyInferer(_FrequencyInferer):
    # 继承自 _FrequencyInferer 类的私有类，用于推断时间增量的频率信息

    def _infer_daily_rule(self):
        # 如果标记为唯一，返回每日规则
        if self.is_unique:
            return self._get_daily_rule()


def _is_multiple(us, mult: int) -> bool:
    # 判断一个整数是否是另一个整数的倍数
    return us % mult == 0


def _maybe_add_count(base: str, count: float) -> str:
    # 可能添加计数到基础字符串中，根据计数的值
    if count != 1:
        assert count == int(count)
        count = int(count)
        return f"{count}{base}"
    else:
        return base


# ----------------------------------------------------------------------
# 频率比较


def is_subperiod(source, target) -> bool:
    """
    如果可以从 source 频率降采样到 target 频率，返回 True

    Parameters
    ----------
    source : str or DateOffset
        转换前的频率
    target : str or DateOffset
        转换后的目标频率

    Returns
    -------
    bool
    """
    if target is None or source is None:
        return False
    source = _maybe_coerce_freq(source)  # 转换 source 到标准频率表示
    target = _maybe_coerce_freq(target)  # 转换 target 到标准频率表示

    if _is_annual(target):
        if _is_quarterly(source):
            return _quarter_months_conform(
                get_rule_month(source), get_rule_month(target)
            )
        return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_quarterly(target):
        return source in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    elif _is_monthly(target):
        return source in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif _is_weekly(target):
        return source in {target, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    elif target == "B":
        return source in {"B", "h", "min", "s", "ms", "us", "ns"}
    elif target == "C":
        return source in {"C", "h", "min", "s", "ms", "us", "ns"}
    elif target == "D":
        return source in {"D", "h", "min", "s", "ms", "us", "ns"}
    elif target == "h":
        return source in {"h", "min", "s", "ms", "us", "ns"}
    elif target == "min":
        return source in {"min", "s", "ms", "us", "ns"}
    elif target == "s":
        return source in {"s", "ms", "us", "ns"}
    elif target == "ms":
        return source in {"ms", "us", "ns"}
    elif target == "us":
        return source in {"us", "ns"}
    elif target == "ns":
        return source in {"ns"}
    else:
        return False


def is_superperiod(source, target) -> bool:
    """
    如果可以从 source 频率升采样到 target 频率，返回 True

    Parameters
    ----------
    source : str or DateOffset
        转换前的频率
    target : str or DateOffset
        转换后的目标频率

    Returns
    -------
    bool
    """
    if target is None or source is None:
        return False
    source = _maybe_coerce_freq(source)  # 转换 source 到标准频率表示
    target = _maybe_coerce_freq(target)  # 转换 target 到标准频率表示
    # 如果源频率为年度
    if _is_annual(source):
        # 如果目标频率也为年度，则比较规则的月份是否相同
        if _is_annual(target):
            return get_rule_month(source) == get_rule_month(target)

        # 如果目标频率为季度，则比较规则的月份是否符合季度的要求
        if _is_quarterly(target):
            smonth = get_rule_month(source)
            tmonth = get_rule_month(target)
            return _quarter_months_conform(smonth, tmonth)
        
        # 如果目标频率为日、复利、双重、月度、小时、分钟、秒、毫秒、微秒、纳秒中的任意一种，则返回 True
        return target in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    
    # 如果源频率为季度，则返回目标频率是否为日、复利、双重、月度、小时、分钟、秒、毫秒、微秒、纳秒中的任意一种
    elif _is_quarterly(source):
        return target in {"D", "C", "B", "M", "h", "min", "s", "ms", "us", "ns"}
    
    # 如果源频率为月度，则返回目标频率是否为日、复利、双重、小时、分钟、秒、毫秒、微秒、纳秒中的任意一种
    elif _is_monthly(source):
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    
    # 如果源频率为每周，则返回目标频率是否为源频率自身或者日、复利、双重、小时、分钟、秒、毫秒、微秒、纳秒中的任意一种
    elif _is_weekly(source):
        return target in {source, "D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    
    # 如果源频率为复利，则返回目标频率是否为日、复利、双重、小时、分钟、秒、毫秒、微秒、纳秒中的任意一种
    elif source == "B":
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    
    # 如果源频率为双重，则返回目标频率是否为日、复利、双重、小时、分钟、秒、毫秒、微秒、纳秒中的任意一种
    elif source == "C":
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    
    # 如果源频率为日，则返回目标频率是否为日、复利、双重、小时、分钟、秒、毫秒、微秒、纳秒中的任意一种
    elif source == "D":
        return target in {"D", "C", "B", "h", "min", "s", "ms", "us", "ns"}
    
    # 如果源频率为小时，则返回目标频率是否为小时、分钟、秒、毫秒、微秒、纳秒中的任意一种
    elif source == "h":
        return target in {"h", "min", "s", "ms", "us", "ns"}
    
    # 如果源频率为分钟，则返回目标频率是否为分钟、秒、毫秒、微秒、纳秒中的任意一种
    elif source == "min":
        return target in {"min", "s", "ms", "us", "ns"}
    
    # 如果源频率为秒，则返回目标频率是否为秒、毫秒、微秒、纳秒中的任意一种
    elif source == "s":
        return target in {"s", "ms", "us", "ns"}
    
    # 如果源频率为毫秒，则返回目标频率是否为毫秒、微秒、纳秒中的任意一种
    elif source == "ms":
        return target in {"ms", "us", "ns"}
    
    # 如果源频率为微秒，则返回目标频率是否为微秒、纳秒中的任意一种
    elif source == "us":
        return target in {"us", "ns"}
    
    # 如果源频率为纳秒，则返回目标频率是否为纳秒
    elif source == "ns":
        return target in {"ns"}
    
    # 如果源频率不在上述任何情况中，则返回 False
    else:
        return False
# 将给定的频率代码转换为规则代码并将其大写化，如果需要的话
def _maybe_coerce_freq(code) -> str:
    """we might need to coerce a code to a rule_code
    and uppercase it

    Parameters
    ----------
    code : str or DateOffset
        要转换的频率代码

    Returns
    -------
    str
        转换后的规则代码
    """
    assert code is not None  # 断言代码不为空
    if isinstance(code, DateOffset):
        # 如果代码是 DateOffset 类型，则将其转换为 PeriodDtype 对象的频率字符串
        code = PeriodDtype(to_offset(code.name))._freqstr
    if code in {"h", "min", "s", "ms", "us", "ns"}:
        # 如果代码是时间单位的简写，则直接返回
        return code
    else:
        # 否则将代码转换为大写形式并返回
        return code.upper()


# 判断两个月份是否在同一个季度内
def _quarter_months_conform(source: str, target: str) -> bool:
    """Check if two months conform to the same quarter.

    Parameters
    ----------
    source : str
        源月份
    target : str
        目标月份

    Returns
    -------
    bool
        是否属于同一个季度
    """
    snum = MONTH_NUMBERS[source]  # 获取源月份的月数
    tnum = MONTH_NUMBERS[target]  # 获取目标月份的月数
    return snum % 3 == tnum % 3  # 判断两个月份是否同属于同一个季度


# 判断给定的规则是否为年度频率
def _is_annual(rule: str) -> bool:
    """Check if the given rule represents an annual frequency.

    Parameters
    ----------
    rule : str
        规则字符串

    Returns
    -------
    bool
        是否为年度频率
    """
    rule = rule.upper()  # 将规则字符串转换为大写
    return rule == "Y" or rule.startswith("Y-")  # 判断是否为年度频率或者以 Y- 开头


# 判断给定的规则是否为季度频率
def _is_quarterly(rule: str) -> bool:
    """Check if the given rule represents a quarterly frequency.

    Parameters
    ----------
    rule : str
        规则字符串

    Returns
    -------
    bool
        是否为季度频率
    """
    rule = rule.upper()  # 将规则字符串转换为大写
    return rule == "Q" or rule.startswith(("Q-", "BQ"))  # 判断是否为季度频率或者以 Q-, BQ 开头


# 判断给定的规则是否为月度频率
def _is_monthly(rule: str) -> bool:
    """Check if the given rule represents a monthly frequency.

    Parameters
    ----------
    rule : str
        规则字符串

    Returns
    -------
    bool
        是否为月度频率
    """
    rule = rule.upper()  # 将规则字符串转换为大写
    return rule in ("M", "BM")  # 判断是否为月度频率或者 BM（月底）


# 判断给定的规则是否为周度频率
def _is_weekly(rule: str) -> bool:
    """Check if the given rule represents a weekly frequency.

    Parameters
    ----------
    rule : str
        规则字符串

    Returns
    -------
    bool
        是否为周度频率
    """
    rule = rule.upper()  # 将规则字符串转换为大写
    return rule == "W" or rule.startswith("W-")  # 判断是否为周度频率或者以 W- 开头


__all__ = [
    "Day",
    "get_period_alias",
    "infer_freq",
    "is_subperiod",
    "is_superperiod",
    "to_offset",
]
```
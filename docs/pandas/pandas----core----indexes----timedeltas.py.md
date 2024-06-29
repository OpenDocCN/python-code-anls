# `D:\src\scipysrc\pandas\pandas\core\indexes\timedeltas.py`

```
# 实现 TimedeltaIndex 类，用于处理 timedelta64 类型的不可变索引

from __future__ import annotations

from typing import TYPE_CHECKING

# 导入 pandas 内部库
from pandas._libs import (
    index as libindex,
    lib,
)
# 导入时间相关库
from pandas._libs.tslibs import (
    Resolution,
    Timedelta,
    to_offset,
)

# 导入 pandas 的数据类型检查相关工具
from pandas.core.dtypes.common import (
    is_scalar,
    pandas_dtype,
)
# 导入 pandas 的通用系列类型
from pandas.core.dtypes.generic import ABCSeries

# 导入 timedelta 类型的数组
from pandas.core.arrays.timedeltas import TimedeltaArray
# 导入 pandas 公共工具
import pandas.core.common as com
# 导入 pandas 的基础索引类及名称提取工具
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
# 导入日期时间相关的混合类
from pandas.core.indexes.datetimelike import DatetimeTimedeltaMixin
# 导入索引扩展相关的工具
from pandas.core.indexes.extension import inherit_names

# 如果是类型检查阶段，则导入额外的类型相关库
if TYPE_CHECKING:
    from pandas._libs import NaTType
    from pandas._typing import DtypeObj

# 继承 TimedeltaArray 的运算方法，以及特定方法的属性名
@inherit_names(
    ["__neg__", "__pos__", "__abs__", "total_seconds", "round", "floor", "ceil"]
    + TimedeltaArray._field_ops,
    TimedeltaArray,
    wrap=True,
)
# 继承 TimedeltaArray 的方法，以及特定方法的属性名
@inherit_names(
    [
        "components",
        "to_pytimedelta",
        "sum",
        "std",
        "median",
    ],
    TimedeltaArray,
)
class TimedeltaIndex(DatetimeTimedeltaMixin):
    """
    Immutable Index of timedelta64 data.

    Represented internally as int64, and scalars returned Timedelta objects.

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional timedelta-like data to construct index with.
    freq : str or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        ``'infer'`` can be passed in order to set the frequency of the index as
        the inferred frequency upon creation.
    dtype : numpy.dtype or str, default None
        Valid ``numpy`` dtypes are ``timedelta64[ns]``, ``timedelta64[us]``,
        ``timedelta64[ms]``, and ``timedelta64[s]``.
    copy : bool
        Make a copy of input array.
    name : object
        Name to be stored in the index.

    Attributes
    ----------
    days
    seconds
    microseconds
    nanoseconds
    components
    inferred_freq

    Methods
    -------
    to_pytimedelta
    to_series
    round
    floor
    ceil
    to_frame
    mean

    See Also
    --------
    Index : The base pandas Index type.
    Timedelta : Represents a duration between two dates or times.
    DatetimeIndex : Index of datetime64 data.
    PeriodIndex : Index of Period data.
    timedelta_range : Create a fixed-frequency TimedeltaIndex.

    Notes
    -----
    To learn more about the frequency strings, please see
    :ref:`this link<timeseries.offset_aliases>`.

    Examples
    --------
    >>> pd.TimedeltaIndex(["0 days", "1 days", "2 days", "3 days", "4 days"])
    TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq=None)

    We can also let pandas infer the frequency when possible.

    >>> pd.TimedeltaIndex(np.arange(5) * 24 * 3600 * 1e9, freq="infer")
    ```
    TimedeltaIndex(['0 days', '1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')
    """
    创建一个TimedeltaIndex对象，包含5个时间增量，表示0到4天，数据类型为'timedelta64[ns]'，频率为每天一次

    _typ = "timedeltaindex"
    将_typ属性设置为字符串"timedeltaindex"

    _data_cls = TimedeltaArray
    将_data_cls属性设置为TimedeltaArray类，用于存储时间增量数据

    @property
    def _engine_type(self) -> type[libindex.TimedeltaEngine]:
        返回libindex.TimedeltaEngine类的类型，作为引擎类型的属性

    _data: TimedeltaArray
    声明_data属性为TimedeltaArray类型，用于存储时间增量数据

    # Use base class method instead of DatetimeTimedeltaMixin._get_string_slice
    _get_string_slice = Index._get_string_slice
    将_get_string_slice属性设置为Index类的_get_string_slice方法，用于获取字符串片段而不是使用DatetimeTimedeltaMixin._get_string_slice方法

    # error: Signature of "_resolution_obj" incompatible with supertype
    # "DatetimeIndexOpsMixin"
    @property
    def _resolution_obj(self) -> Resolution | None:  # type: ignore[override]
        返回_resolution_obj属性，它可以是Resolution类型或者None，用于获取时间增量数组的分辨率对象，忽略覆盖类型的类型检查错误

    # -------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        freq=lib.no_default,
        dtype=None,
        copy: bool = False,
        name=None,
    ):
        构造函数__new__，用于创建TimedeltaIndex对象实例
        name = maybe_extract_name(name, data, cls)
        如果可能，从参数中提取名称并分配给name变量

        if is_scalar(data):
            如果data是标量（单个值），则引发错误

        if dtype is not None:
            如果指定了dtype，则将其转换为pandas的dtype类型

        if (
            isinstance(data, TimedeltaArray)
            and freq is lib.no_default
            and (dtype is None or dtype == data.dtype)
        ):
            如果data是TimedeltaArray类型，并且未指定freq，并且dtype为空或与data的dtype相同
            如果copy为True，则复制数据并返回简单新实例
            否则，返回简单新视图

        if (
            isinstance(data, TimedeltaIndex)
            and freq is lib.no_default
            and name is None
            and (dtype is None or dtype == data.dtype)
        ):
            如果data是TimedeltaIndex类型，并且未指定freq和名称，并且dtype为空或与data的dtype相同
            如果copy为True，则复制数据并返回
            否则，返回视图

        # - Cases checked above all return/raise before reaching here - #

        tdarr = TimedeltaArray._from_sequence_not_strict(
            data, freq=freq, unit=None, dtype=dtype, copy=copy
        )
        从不严格的序列中创建TimedeltaArray对象tdarr
        如果不复制且数据是ABCSeries或Index类型，则保留引用关系

        返回简单新实例，包括名称和引用

    # -------------------------------------------------------------------

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        返回布尔值，指示给定的dtype是否可以与自身的dtype进行比较
        lib.is_np_dtype(dtype, "m")用于检查dtype是否是日期时间数据类型

    # -------------------------------------------------------------------
    # Indexing Methods

    def get_loc(self, key):
        """
        Get integer location for requested label

        Returns
        -------
        loc : int, slice, or ndarray[int]
        """
        获取请求标签的整数位置
        首先检查索引错误
        尝试验证并返回key的标量值，如果验证失败则引发KeyError

        返回Index类的get_loc方法处理后的结果
    # error: Return type "tuple[Timedelta | NaTType, None]" of "_parse_with_reso"
    # incompatible with return type "tuple[datetime, Resolution]" in supertype
    # "DatetimeIndexOpsMixin"
    # 定义一个方法 _parse_with_reso，接受一个字符串参数 label，返回一个元组，包含 Timedelta 或 NaTType 和 None
    def _parse_with_reso(self, label: str) -> tuple[Timedelta | NaTType, None]:  # type: ignore[override]
        # 将字符串 label 转换为 Timedelta 对象
        parsed = Timedelta(label)
        # 返回 Timedelta 对象和 None
        return parsed, None

    # 定义一个方法 _parsed_string_to_bounds，接受两个参数 reso 和 parsed（Timedelta 对象）
    def _parsed_string_to_bounds(self, reso, parsed: Timedelta):
        # reso 未被使用，只是为了与 DTI/PI 的签名匹配
        # 根据 parsed 的分辨率 round 调整左边界 lbound
        lbound = parsed.round(parsed.resolution_string)
        # 计算右边界 rbound，通过 to_offset 方法和减去 1 纳秒
        rbound = lbound + to_offset(parsed.resolution_string) - Timedelta(1, "ns")
        # 返回左右边界的元组
        return lbound, rbound

    # -------------------------------------------------------------------

    @property
    # 定义一个属性 inferred_type，返回字符串 "timedelta64"
    def inferred_type(self) -> str:
        return "timedelta64"
# 返回一个固定频率的 TimedeltaIndex，默认以天为单位
def timedelta_range(
    start=None,                                 # 左边界，可以是字符串或类似 timedelta 的对象，默认为 None
    end=None,                                   # 右边界，可以是字符串或类似 timedelta 的对象，默认为 None
    periods: int | None = None,                 # 要生成的周期数，默认为 None
    freq=None,                                  # 频率字符串，可以是 '5h' 这样的多个单位的字符串，默认为 'D'
    name=None,                                  # 结果 TimedeltaIndex 的名称，默认为 None
    closed=None,                                # 使得区间对于给定频率的左侧、右侧或两侧（None）封闭，默认为 None
    *,
    unit: str | None = None,                    # 指定结果的期望分辨率，默认为 None

) -> TimedeltaIndex:
    """
    Return a fixed frequency TimedeltaIndex with day as the default.

    Parameters
    ----------
    start : str or timedelta-like, default None
        Left bound for generating timedeltas.
    end : str or timedelta-like, default None
        Right bound for generating timedeltas.
    periods : int, default None
        Number of periods to generate.
    freq : str, Timedelta, datetime.timedelta, or DateOffset, default 'D'
        Frequency strings can have multiples, e.g. '5h'.
    name : str, default None
        Name of the resulting TimedeltaIndex.
    closed : str, default None
        Make the interval closed with respect to the given frequency to
        the 'left', 'right', or both sides (None).
    unit : str, default None
        Specify the desired resolution of the result.

        .. versionadded:: 2.0.0

    Returns
    -------
    TimedeltaIndex
        Fixed frequency, with day as the default.

    See Also
    --------
    date_range : Return a fixed frequency DatetimeIndex.
    period_range : Return a fixed frequency PeriodIndex.

    Notes
    -----
    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``TimedeltaIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see
    :ref:`this link<timeseries.offset_aliases>`.

    Examples
    --------
    >>> pd.timedelta_range(start="1 day", periods=4)
    TimedeltaIndex(['1 days', '2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')

    The ``closed`` parameter specifies which endpoint is included.  The default
    behavior is to include both endpoints.

    >>> pd.timedelta_range(start="1 day", periods=4, closed="right")
    TimedeltaIndex(['2 days', '3 days', '4 days'],
                   dtype='timedelta64[ns]', freq='D')

    The ``freq`` parameter specifies the frequency of the TimedeltaIndex.
    Only fixed frequencies can be passed, non-fixed frequencies such as
    'M' (month end) will raise.

    >>> pd.timedelta_range(start="1 day", end="2 days", freq="6h")
    TimedeltaIndex(['1 days 00:00:00', '1 days 06:00:00', '1 days 12:00:00',
                    '1 days 18:00:00', '2 days 00:00:00'],
                   dtype='timedelta64[ns]', freq='6h')

    Specify ``start``, ``end``, and ``periods``; the frequency is generated
    automatically (linearly spaced).

    >>> pd.timedelta_range(start="1 day", end="5 days", periods=4)
    TimedeltaIndex(['1 days 00:00:00', '2 days 08:00:00', '3 days 16:00:00',
                    '5 days 00:00:00'],
                   dtype='timedelta64[ns]', freq=None)

    **Specify a unit**
    """
    # 使用 pd 下的 timedelta_range 函数生成固定频率的 TimedeltaIndex
    return pd.tseries.frequencies.timedelta_range(
        start=start,
        end=end,
        periods=periods,
        freq=freq,
        name=name,
        closed=closed,
        unit=unit,
    )
    # 如果频率（freq）参数为 None，并且 periods、start、end 中至少有一个为 None，则将频率设置为 "D"（天）
    if freq is None and com.any_none(periods, start, end):
        freq = "D"

    # 将频率（freq）转换为偏移量对象
    freq = to_offset(freq)

    # 使用 TimedeltaArray 类的方法生成时间增量数组（tdarr）
    # 根据给定的 start、end、periods、freq 参数生成时间增量数组
    tdarr = TimedeltaArray._generate_range(
        start, end, periods, freq, closed=closed, unit=unit
    )

    # 使用 TimedeltaIndex 类的方法创建一个新的 TimedeltaIndex 对象
    # 通过时间增量数组（tdarr）创建 TimedeltaIndex 对象，并指定名称（name）
    return TimedeltaIndex._simple_new(tdarr, name=name)
```
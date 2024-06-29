# `D:\src\scipysrc\pandas\pandas\io\stata.py`

```
"""
Module contains tools for processing Stata files into DataFrames

The StataReader below was originally written by Joe Presbrey as part of PyDTA.
It has been extended and improved by Skipper Seabold from the Statsmodels
project who also developed the StataWriter and was finally added to pandas in
a once again improved version.

You can find more information on http://presbrey.mit.edu/PyDTA and
https://www.statsmodels.org/devel/
"""

# 引入未来的注解功能，以支持类型检查和类型提示
from __future__ import annotations

# 引入标准库中的一些模块
from collections import abc  # 引入抽象基类模块
from datetime import (  # 从 datetime 模块中引入多个类和函数
    datetime,         # 日期时间类
    timedelta,        # 时间间隔类
)
from io import BytesIO  # 从 io 模块中引入字节流类
import os  # 引入操作系统相关的功能
import struct  # 引入处理结构化数据的模块
import sys  # 引入系统相关的功能
from typing import (  # 从 typing 模块中引入类型提示相关的类和函数
    IO,              # IO 类型
    TYPE_CHECKING,   # 类型检查标志
    AnyStr,          # 任意字符串类型
    Final,           # 最终类型标志
    cast,            # 类型转换函数
)
import warnings  # 引入警告模块

import numpy as np  # 引入 NumPy 库，用于数值计算

# 从 pandas 库中引入相关子模块和类
from pandas._libs import lib  # 引入 pandas 库的内部 C 库
from pandas._libs.lib import infer_dtype  # 引入类型推断函数
from pandas._libs.writers import max_len_string_array  # 引入最大长度字符串数组函数
from pandas.errors import (  # 从 pandas.errors 模块中引入多个错误类
    CategoricalConversionWarning,  # 类别转换警告类
    InvalidColumnName,             # 无效列名类
    PossiblePrecisionLoss,         # 可能精度丢失类
    ValueLabelTypeMismatch,        # 值标签类型不匹配类
)
from pandas.util._decorators import (  # 从 pandas.util._decorators 模块中引入装饰器函数
    Appender,  # 追加装饰器
    doc,       # 文档装饰器
)
from pandas.util._exceptions import find_stack_level  # 引入查找堆栈层级异常函数

# 从 pandas.core.dtypes.base 模块中引入扩展数据类型类
from pandas.core.dtypes.base import ExtensionDtype
# 从 pandas.core.dtypes.common 模块中引入多个数据类型检测函数
from pandas.core.dtypes.common import (
    ensure_object,      # 确保对象类型
    is_numeric_dtype,   # 是否为数值类型
    is_string_dtype,    # 是否为字符串类型
)
# 从 pandas.core.dtypes.dtypes 模块中引入类别数据类型类
from pandas.core.dtypes.dtypes import CategoricalDtype

# 从 pandas 库中引入多个类和函数
from pandas import (
    Categorical,      # 类别数据类
    DatetimeIndex,    # 日期时间索引类
    NaT,              # 不可用的日期时间对象
    Timestamp,        # 时间戳类
    isna,             # 是否为 NA 值
    to_datetime,      # 转换为日期时间函数
)
# 从 pandas.core.frame 模块中引入数据帧类
from pandas.core.frame import DataFrame
# 从 pandas.core.indexes.base 模块中引入索引基类
from pandas.core.indexes.base import Index
# 从 pandas.core.indexes.range 模块中引入范围索引类
from pandas.core.indexes.range import RangeIndex
# 从 pandas.core.series 模块中引入序列类
from pandas.core.series import Series
# 从 pandas.core.shared_docs 模块中引入共享文档字符串
from pandas.core.shared_docs import _shared_docs

# 从 pandas.io.common 模块中引入获取句柄函数
from pandas.io.common import get_handle

# 如果支持类型检查，则导入以下类型相关的类和函数
if TYPE_CHECKING:
    from collections.abc import (  # 从 collections.abc 模块中引入抽象基类
        Callable,   # 可调用对象
        Hashable,   # 可哈希对象
        Sequence,   # 序列对象
    )
    from types import TracebackType  # 引入异常回溯类型
    from typing import Literal  # 引入文字类型

# 版本错误消息模板，描述给定 Stata 文件的版本支持情况
_version_error = (
    "Version of given Stata file is {version}. pandas supports importing "
    "versions 103, 104, 105, 108, 110 (Stata 7), 111 (Stata 7SE), 113 (Stata 8/9), "
    "114 (Stata 10/11), 115 (Stata 12), 117 (Stata 13), 118 (Stata 14/15/16),"
    "and 119 (Stata 15/16, over 32,767 variables)."
)

# 处理 Stata 文件的参数描述，用于说明如何处理 Stata 文件中的数据
_statafile_processing_params1 = """\
convert_dates : bool, default True
    Convert date variables to DataFrame time values.
convert_categoricals : bool, default True
    Read value labels and convert columns to Categorical/Factor variables."""

_statafile_processing_params2 = """\
index_col : str, optional
    Column to set as index.
convert_missing : bool, default False
    Flag indicating whether to convert missing values to their Stata
    representations.  If False, missing values are replaced with nan.
    If True, columns containing missing values are returned with
    object data types and missing values are represented by
    # 导入 StataMissingValue 对象
    from statsmodels.imputation import StataMissingValue
# 是否保留 Stata 数据类型，默认为 True
preserve_dtypes : bool, default True
# 如果为 False，则将数值数据转换为 pandas 默认类型（float64 或 int64）
# 用于外部数据的读取。

columns : list or None
# 要保留的列。返回的列将按给定顺序返回。None 表示返回所有列。

order_categoricals : bool, default True
# 指示转换的分类数据是否有序的标志。

_chunksize_params = """\
chunksize : int, default None
# 返回 StataReader 对象进行迭代，每次返回给定行数的数据块。

_iterator_params = """\
iterator : bool, default False
# 返回 StataReader 对象的标志。

_reader_notes = """\
Notes
-----
迭代读取的分类变量可能具有不同的分类和数据类型。
这种情况发生在 DTA 文件中的变量关联到一个仅标记值的不完整集合时。

_read_stata_doc = f"""
Read Stata file into DataFrame.

Parameters
----------
filepath_or_buffer : str, path object or file-like object
# 接受任何有效的字符串路径。字符串可以是 URL，支持的 URL 方案包括 http、ftp、s3 和 file。
# 对于 file URL，预期会有一个主机名。本地文件可以是："file://localhost/path/to/table.dta"。

# 如果要传递路径对象，pandas 接受任何 os.PathLike。

# 通过类文件对象，我们指的是具有 read() 方法的对象，
# 例如文件句柄（例如通过内置的 open 函数）或 StringIO。

{_statafile_processing_params1}
{_statafile_processing_params2}
{_chunksize_params}
{_iterator_params}
{_shared_docs["decompression_options"] % "filepath_or_buffer"}
{_shared_docs["storage_options"]}

Returns
-------
DataFrame, pandas.api.typing.StataReader
# 如果 iterator 或 chunksize 为 True，则返回 StataReader 对象，否则返回 DataFrame。

See Also
--------
io.stata.StataReader : 用于 Stata 数据文件的低级读取器。
DataFrame.to_stata: 导出 Stata 数据文件。

{_reader_notes}

Examples
--------

Creating a dummy stata for this example

>>> df = pd.DataFrame({'animal': ['falcon', 'parrot', 'falcon', 'parrot'],
...                   'speed': [350, 18, 361, 15]})  # doctest: +SKIP
>>> df.to_stata('animals.dta')  # doctest: +SKIP

Read a Stata dta file:

>>> df = pd.read_stata('animals.dta')  # doctest: +SKIP

Read a Stata dta file in 10,000 line chunks:

>>> values = np.random.randint(0, 10, size=(20_000, 1), dtype="uint8")  # doctest: +SKIP
>>> df = pd.DataFrame(values, columns=["i"])  # doctest: +SKIP
>>> df.to_stata('filename.dta')  # doctest: +SKIP

>>> with pd.read_stata('filename.dta', chunksize=10000) as itr:  # doctest: +SKIP
>>>     for chunk in itr:
...         # 操作单个数据块，例如 chunk.mean()
...         pass  # doctest: +SKIP
"""

_read_method_doc = f"""\
Reads observations from Stata file, converting them into a dataframe

Parameters
----------
nrows : int
    Number of lines to read from data file, if None read whole file.
# 列表，包含了 Stata 数据文件中日期的多种格式
_date_formats = ["%tc", "%tC", "%td", "%d", "%tw", "%tm", "%tq", "%th", "%ty"]

# Stata 时间起点，1960年1月1日
stata_epoch: Final = datetime(1960, 1, 1)
# Unix 时间起点，1970年1月1日
unix_epoch: Final = datetime(1970, 1, 1)


def _stata_elapsed_date_to_datetime_vec(dates: Series, fmt: str) -> Series:
    """
    Convert from Stata Internal Format to datetime. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        需要转换为 datetime 的 Stata 内部格式日期
    fmt : str
        转换的目标格式。可以是 tc, td, tw, tm, tq, th, ty 等

    Returns
    -------
    converted : Series
        转换后的日期序列

    Examples
    --------
    >>> dates = pd.Series([52])
    >>> _stata_elapsed_date_to_datetime_vec(dates, "%tw")
    0   1961-01-01
    dtype: datetime64[s]

    Notes
    -----
    datetime/c - tc
        毫秒数，从 1960年1月1日 00:00:00.000 开始计算，假设一天有 86400 秒
    datetime/C - tC - NOT IMPLEMENTED
        毫秒数，从 1960年1月1日 00:00:00.000 开始计算，考虑闰秒
    date - td
        天数，从 1960年1月1日 开始计算（1960年1月1日 = 0）
    weekly date - tw
        自1960年以来的周数
        假设一年有 52 周，然后加上剩余的周数乘以 7。
        datetime 值是一年中的某一周的开始，而不是 ISO 日历周。
    monthly date - tm
        自1960年以来的月数
    quarterly date - tq
        自1960年以来的季度数
    half-yearly date - th
        自1960年以来的半年数
    date - ty
        自0000年以来的年数
    """

    if fmt.startswith(("%tc", "tc")):
        # 相对于基准的毫秒数
        td = np.timedelta64(stata_epoch - unix_epoch, "ms")
        res = np.array(dates._values, dtype="M8[ms]") + td
        return Series(res, index=dates.index)

    elif fmt.startswith(("%td", "td", "%d", "d")):
        # 相对于基准的天数
        td = np.timedelta64(stata_epoch - unix_epoch, "D")
        res = np.array(dates._values, dtype="M8[D]") + td
        return Series(res, index=dates.index)

    elif fmt.startswith(("%tm", "tm")):
        # 相对于基准的月数
        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 12
        res = np.array(ordinals, dtype="M8[M]").astype("M8[s]")
        return Series(res, index=dates.index)
    elif fmt.startswith(("%tq", "tq")):
        # 如果格式以 "%tq" 或 "tq" 开头，则表示相对于基准的季度差值
        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 4
        # 将 ordinals 转换为 numpy 数组，使用 "M8[3M]" 类型表示三个月的时间间隔，并转换为秒级精度
        res = np.array(ordinals, dtype="M8[3M]").astype("M8[s]")
        return Series(res, index=dates.index)

    elif fmt.startswith(("%th", "th")):
        # 如果格式以 "%th" 或 "th" 开头，则表示相对于基准的半年差值
        ordinals = dates + (stata_epoch.year - unix_epoch.year) * 2
        # 将 ordinals 转换为 numpy 数组，使用 "M8[6M]" 类型表示六个月的时间间隔，并转换为秒级精度
        res = np.array(ordinals, dtype="M8[6M]").astype("M8[s]")
        return Series(res, index=dates.index)

    elif fmt.startswith(("%ty", "ty")):
        # 如果格式以 "%ty" 或 "ty" 开头，则表示年份，而非时间差
        ordinals = dates - 1970
        # 将 ordinals 转换为 numpy 数组，使用 "M8[Y]" 类型表示年的时间间隔，并转换为秒级精度
        res = np.array(ordinals, dtype="M8[Y]").astype("M8[s]")
        return Series(res, index=dates.index)

    bad_locs = np.isnan(dates)
    has_bad_values = False
    if bad_locs.any():
        has_bad_values = True
        dates._values[bad_locs] = 1.0  # 将 NaN 值替换为 NaT

    dates = dates.astype(np.int64)

    if fmt.startswith(("%tC", "tC")):
        # 如果格式以 "%tC" 或 "tC" 开头，则发出警告，表明是 Stata 内部格式，返回对象类型的日期
        warnings.warn(
            "Encountered %tC format. Leaving in Stata Internal Format.",
            stacklevel=find_stack_level(),
        )
        conv_dates = Series(dates, dtype=object)
        if has_bad_values:
            conv_dates[bad_locs] = NaT
        return conv_dates

    # 不计闰日 - 一周有 7 天。
    # 第 52 周可能多于 7 天
    elif fmt.startswith(("%tw", "tw")):
        year = stata_epoch.year + dates // 52
        days = (dates % 52) * 7
        per_y = (year - 1970).array.view("Period[Y]")  # 将年份差转换为年周期
        per_d = per_y.asfreq("D", how="S")  # 转换为天周期
        per_d_shifted = per_d + days._values  # 加上天数差值
        per_s = per_d_shifted.asfreq("s", how="S")  # 转换为秒级周期
        conv_dates_arr = per_s.view("M8[s]")  # 转换为秒级时间戳
        conv_dates = Series(conv_dates_arr, index=dates.index)

    else:
        raise ValueError(f"Date fmt {fmt} not understood")  # 抛出格式错误异常

    if has_bad_values:  # 如果存在 NaN 值，则恢复为 NaT
        conv_dates[bad_locs] = NaT

    return conv_dates
# 根据给定的日期序列 `dates` 和格式字符串 `fmt` 将日期转换为 Stata 内部格式
def _datetime_to_stata_elapsed_vec(dates: Series, fmt: str) -> Series:
    """
    Convert from datetime to SIF. https://www.stata.com/help.cgi?datetime

    Parameters
    ----------
    dates : Series
        包含待转换为 Stata 内部格式的 datetime 或 datetime64[ns] 的序列或数组
    fmt : str
        指定转换的格式。可以是 tc, td, tw, tm, tq, th, ty
    """
    # 保存日期的索引
    index = dates.index
    # 定义每天的纳秒、微秒和毫秒数
    NS_PER_DAY = 24 * 3600 * 1000 * 1000 * 1000
    US_PER_DAY = NS_PER_DAY / 1000
    MS_PER_DAY = NS_PER_DAY / 1_000_000

    # 安全地解析日期的函数
    def parse_dates_safe(
        dates: Series, delta: bool = False, year: bool = False, days: bool = False
    ) -> DataFrame:
        d = {}
        # 如果日期类型是 numpy 的 datetime64[ns]
        if lib.is_np_dtype(dates.dtype, "M"):
            if delta:
                # 计算时间差
                time_delta = dates.dt.as_unit("ms") - Timestamp(stata_epoch).as_unit(
                    "ms"
                )
                d["delta"] = time_delta._values.view(np.int64)
            if days or year:
                # 获取年份和月份
                date_index = DatetimeIndex(dates)
                d["year"] = date_index._data.year
                d["month"] = date_index._data.month
            if days:
                # 计算与年初的天数差
                year_start = np.asarray(dates).astype("M8[Y]").astype(dates.dtype)
                diff = dates - year_start
                d["days"] = np.asarray(diff).astype("m8[D]").view("int64")

        # 如果日期类型是 datetime.datetime
        elif infer_dtype(dates, skipna=False) == "datetime":
            if delta:
                # 计算时间差
                delta = dates._values - stata_epoch

                def f(x: timedelta) -> float:
                    return US_PER_DAY * x.days + 1000000 * x.seconds + x.microseconds

                v = np.vectorize(f)
                d["delta"] = v(delta)
            if year:
                # 提取年份和月份
                year_month = dates.apply(lambda x: 100 * x.year + x.month)
                d["year"] = year_month._values // 100
                d["month"] = year_month._values - d["year"] * 100
            if days:
                # 计算每年中的天数
                def g(x: datetime) -> int:
                    return (x - datetime(x.year, 1, 1)).days

                v = np.vectorize(g)
                d["days"] = v(dates)
        else:
            # 如果日期类型不支持，抛出异常
            raise ValueError(
                "Columns containing dates must contain either "
                "datetime64, datetime or null values."
            )

        return DataFrame(d, index=index)

    # 检查是否存在缺失值
    bad_loc = isna(dates)
    index = dates.index
    if bad_loc.any():
        if lib.is_np_dtype(dates.dtype, "M"):
            # 如果是 numpy datetime64[ns] 类型，将缺失值填充为 Stata 纪元时间
            dates._values[bad_loc] = to_datetime(stata_epoch)
        else:
            # 否则将缺失值填充为 Stata 纪元时间
            dates._values[bad_loc] = stata_epoch

    # 根据指定的 fmt 进行不同的日期转换处理
    if fmt in ["%tc", "tc"]:
        # 调用安全解析日期函数，生成时间差数据
        d = parse_dates_safe(dates, delta=True)
        conv_dates = d.delta
    elif fmt in ["%tC", "tC"]:
        # 如果是 %tC 或 tC 格式，发出警告并保持原始日期
        warnings.warn(
            "Stata Internal Format tC not supported.",
            stacklevel=find_stack_level(),
        )
        conv_dates = dates
    elif fmt in ["%td", "td"]:
        # 如果日期格式为 "%td" 或 "td"，则调用 parse_dates_safe 函数解析日期，支持增量日期
        d = parse_dates_safe(dates, delta=True)
        # 计算日期转换后的值，单位为天数
        conv_dates = d.delta // MS_PER_DAY
    elif fmt in ["%tw", "tw"]:
        # 如果日期格式为 "%tw" 或 "tw"，则调用 parse_dates_safe 函数解析日期，返回年份和天数
        d = parse_dates_safe(dates, year=True, days=True)
        # 计算日期转换后的值，单位为周数
        conv_dates = 52 * (d.year - stata_epoch.year) + d.days // 7
    elif fmt in ["%tm", "tm"]:
        # 如果日期格式为 "%tm" 或 "tm"，则调用 parse_dates_safe 函数解析日期，返回年份和月份
        d = parse_dates_safe(dates, year=True)
        # 计算日期转换后的值，单位为月数
        conv_dates = 12 * (d.year - stata_epoch.year) + d.month - 1
    elif fmt in ["%tq", "tq"]:
        # 如果日期格式为 "%tq" 或 "tq"，则调用 parse_dates_safe 函数解析日期，返回年份和月份
        d = parse_dates_safe(dates, year=True)
        # 计算日期转换后的值，单位为季度数
        conv_dates = 4 * (d.year - stata_epoch.year) + (d.month - 1) // 3
    elif fmt in ["%th", "th"]:
        # 如果日期格式为 "%th" 或 "th"，则调用 parse_dates_safe 函数解析日期，返回年份和月份
        d = parse_dates_safe(dates, year=True)
        # 计算日期转换后的值，单位为半年数
        conv_dates = 2 * (d.year - stata_epoch.year) + (d.month > 6).astype(int)
    elif fmt in ["%ty", "ty"]:
        # 如果日期格式为 "%ty" 或 "ty"，则调用 parse_dates_safe 函数解析日期，返回年份
        d = parse_dates_safe(dates, year=True)
        # 取得日期的年份作为转换后的值
        conv_dates = d.year
    else:
        # 如果日期格式未知，则抛出 ValueError 异常
        raise ValueError(f"Format {fmt} is not a known Stata date format")

    # 使用 numpy 创建一个 Series 对象，存储转换后的日期值，数据类型为 np.float64，不进行拷贝
    conv_dates = Series(conv_dates, dtype=np.float64, copy=False)
    
    # 定义缺失值，对应 Stata 中的缺失值表示
    missing_value = struct.unpack("<d", b"\x00\x00\x00\x00\x00\x00\xe0\x7f")[0]
    
    # 将转换后的日期中可能存在的无效位置（bad_loc）设置为缺失值
    conv_dates[bad_loc] = missing_value
    
    # 返回存储转换后日期值的 Series 对象，同时设置其索引和不进行拷贝
    return Series(conv_dates, index=index, copy=False)
# 错误消息: 固定宽度字符串在 Stata .dta 文件中限制为 244 个字符。列 '{0}' 不符合此限制。使用 'version=117' 参数可以写入更新的格式（Stata 13及更高版本）。
excessive_string_length_error: Final = """
Fixed width strings in Stata .dta files are limited to 244 (or fewer)
characters.  Column '{0}' does not satisfy this restriction. Use the
'version=117' parameter to write the newer (Stata 13 and later) format.
"""

# 精度丢失警告: 列从 {0} 转换为 {1}，并且某些数据超出了无损转换范围。这可能导致保存数据时精度损失。
precision_loss_doc: Final = """
Column converted from {0} to {1}, and some data are outside of the lossless
conversion range. This may result in a loss of precision in the saved data.
"""

# 值标签不匹配警告: Stata 值标签（pandas 类别）必须是字符串。列 {0} 包含非字符串标签，这些标签将被转换为字符串。请检查由于重复标签而导致的 Stata 数据文件是否丢失信息。
value_label_mismatch_doc: Final = """
Stata value labels (pandas categories) must be strings. Column {0} contains
non-string labels which will be converted to strings.  Please check that the
Stata data file created has not lost information due to duplicate labels.
"""

# 无效列名警告: 并非所有 pandas 列名都是有效的 Stata 变量名。以下是已进行的替换：
#
#     {0}
#
# 如果这不是您预期的结果，请确保 DataFrame 中使用了符合 Stata 标准的列名（仅限字符串，最多 32 个字符，仅限字母数字和下划线，不使用 Stata 保留字）。
invalid_name_doc: Final = """
Not all pandas column names were valid Stata variable names.
The following replacements have been made:

    {0}

If this is not what you expect, please make sure you have Stata-compliant
column names in your DataFrame (strings only, max 32 characters, only
alphanumerics and underscores, no Stata reserved words)
"""

# 分类转换警告: 一个或多个带有值标签的系列未完全标记。使用迭代器读取此数据集会导致具有不同类别的分类变量。这是因为在读取完整数据集之前无法知道所有可能的值。为避免此警告，可以选择不使用迭代器读取数据集，或通过将 `convert_categoricals` 设置为 False 并通过读取器的 `value_labels` 方法手动转换分类数据。
categorical_conversion_warning: Final = """
One or more series with value labels are not fully labeled. Reading this
dataset with an iterator results in categorical variable with different
categories. This occurs since it is not possible to know all possible values
until the entire dataset has been read. To avoid this warning, you can either
read dataset without an iterator, or manually convert categorical data by
``convert_categoricals`` to False and then accessing the variable labels
through the value_labels method of the reader.
"""

def _cast_to_stata_types(data: DataFrame) -> DataFrame:
    """
    检查 pandas DataFrame 的列的 dtypes 是否与 Stata 支持的数据类型和范围兼容，并在必要时进行转换。

    Parameters
    ----------
    data : DataFrame
        要检查和转换的 DataFrame

    Notes
    -----
    在 Stata 中，数值列必须是 int8、int16、int32、float32 或 float64 之一，并具有一些额外的值限制。
    int8 和 int16 列会检查违反值限制，并在需要时向上转换。
    int64 数据在 Stata 中无法使用，因此在 int32 范围内的值时会向下转换为 int32，并在超出此范围时向 float64 转换。
    如果 int64 值超出可以完全表示为 float64 值的范围，则会发出警告。
    布尔列被转换为 int8。
    uint 列如果没有精度损失则转换为相同大小的 int 类型，否则会向较大类型转换。
    uint64 目前不支持，因为在 DataFrame 中转换为对象。
    """
    ws = ""
    # original, if small, if large
    conversion_data: tuple[
        tuple[type, type, type],
        tuple[type, type, type],
        tuple[type, type, type],
        tuple[type, type, type],
        tuple[type, type, type],
        # 大小为 5 的元组，每个元组包含 3 个类型
    ] = (
        (np.bool_, np.int8, np.int8),  # 定义一个元组列表，每个元组包含三个 NumPy 数据类型
        (np.uint8, np.int8, np.int16),  # 不同类型的数据类型组合
        (np.uint16, np.int16, np.int32),
        (np.uint32, np.int32, np.int64),
        (np.uint64, np.int64, np.float64),
    )

    float32_max = struct.unpack("<f", b"\xff\xff\xff\x7e")[0]  # 解析字节序列为单精度浮点数的最大值
    float64_max = struct.unpack("<d", b"\xff\xff\xff\xff\xff\xff\xdf\x7f")[0]  # 解析字节序列为双精度浮点数的最大值

    if ws:  # 如果给定了警告消息
        warnings.warn(
            ws,  # 发出警告，内容为 ws
            PossiblePrecisionLoss,  # 使用 PossiblePrecisionLoss 作为警告类别
            stacklevel=find_stack_level(),  # 设置警告的堆栈级别
        )

    return data  # 返回变量 data
class StataValueLabel:
    """
    Parse a categorical column and prepare formatted output

    Parameters
    ----------
    catarray : Series
        Categorical Series to encode
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    """

    def __init__(
        self, catarray: Series, encoding: Literal["latin-1", "utf-8"] = "latin-1"
    ) -> None:
        # 检查编码是否为支持的 latin-1 或 utf-8
        if encoding not in ("latin-1", "utf-8"):
            raise ValueError("Only latin-1 and utf-8 are supported.")
        
        # 设置标签名称为传入 Series 的名称
        self.labname = catarray.name
        self._encoding = encoding
        # 获取分类的唯一值，并用枚举器创建值标签
        categories = catarray.cat.categories
        self.value_labels = enumerate(categories)

        # 调用私有方法准备值标签
        self._prepare_value_labels()

    def _prepare_value_labels(self) -> None:
        """Encode value labels."""
        
        # 初始化变量
        self.text_len = 0
        self.txt: list[bytes] = []
        self.n = 0
        # 偏移量数组，用于存储类别长度（转换为 int32）
        self.off = np.array([], dtype=np.int32)
        # 值数组，用于存储枚举值（转换为 int32）
        self.val = np.array([], dtype=np.int32)
        self.len = 0

        # 计算长度并设置偏移量和标签列表
        offsets: list[int] = []
        values: list[float] = []
        for vl in self.value_labels:
            category: str | bytes = vl[1]
            # 如果类别不是字符串，转换为字符串并发出警告
            if not isinstance(category, str):
                category = str(category)
                warnings.warn(
                    value_label_mismatch_doc.format(self.labname),
                    ValueLabelTypeMismatch,
                    stacklevel=find_stack_level(),
                )
            # 根据设定编码对类别进行编码
            category = category.encode(self._encoding)
            offsets.append(self.text_len)
            # 计算文本长度，包括填充的 +1
            self.text_len += len(category) + 1
            values.append(vl[0])
            self.txt.append(category)
            self.n += 1

        # 检查文本总长度是否超过 32000 字符
        if self.text_len > 32000:
            raise ValueError(
                "Stata value labels for a single variable must "
                "have a combined length less than 32,000 characters."
            )

        # 将偏移量和值数组转换为 int32 类型
        self.off = np.array(offsets, dtype=np.int32)
        self.val = np.array(values, dtype=np.int32)

        # 计算总长度
        self.len = 4 + 4 + 4 * self.n + 4 * self.n + self.text_len
    def generate_value_label(self, byteorder: str) -> bytes:
        """
        Generate the binary representation of the value labels.

        Parameters
        ----------
        byteorder : str
            Byte order of the output

        Returns
        -------
        value_label : bytes
            Bytes containing the formatted value label
        """
        # 获取编码方式
        encoding = self._encoding
        # 创建一个字节流对象
        bio = BytesIO()
        # 空字节，用于填充
        null_byte = b"\x00"

        # 写入长度信息
        bio.write(struct.pack(byteorder + "i", self.len))

        # 处理标签名
        labname = str(self.labname)[:32].encode(encoding)
        # 如果编码不是 utf-8 或 utf8，则标签名长度为 32，否则为 128
        lab_len = 32 if encoding not in ("utf-8", "utf8") else 128
        # 填充标签名字节到指定长度
        labname = _pad_bytes(labname, lab_len + 1)
        bio.write(labname)

        # 填充 - 3 个字节
        for i in range(3):
            bio.write(struct.pack("c", null_byte))

        # 写入 value_label_table 中的 n（int32）
        bio.write(struct.pack(byteorder + "i", self.n))

        # 写入 textlen（int32）
        bio.write(struct.pack(byteorder + "i", self.text_len))

        # 写入 off 数组（每个元素为 int32）
        for offset in self.off:
            bio.write(struct.pack(byteorder + "i", offset))

        # 写入 val 数组（每个元素为 int32）
        for value in self.val:
            bio.write(struct.pack(byteorder + "i", value))

        # 写入 txt 中的文本标签，以空字符结尾
        for text in self.txt:
            bio.write(text + null_byte)

        # 返回生成的字节流对象的值
        return bio.getvalue()
class StataNonCatValueLabel(StataValueLabel):
    """
    准备值标签的格式化版本

    Parameters
    ----------
    labname : str
        值标签的名称
    value_labels: Dictionary
        值到标签的映射
    encoding : {"latin-1", "utf-8"}
        用于值标签的编码方式。
    """

    def __init__(
        self,
        labname: str,
        value_labels: dict[float, str],
        encoding: Literal["latin-1", "utf-8"] = "latin-1",
    ) -> None:
        if encoding not in ("latin-1", "utf-8"):
            raise ValueError("Only latin-1 and utf-8 are supported.")

        self.labname = labname
        self._encoding = encoding
        # 将值标签字典按键值排序
        self.value_labels = sorted(  # type: ignore[assignment]
            value_labels.items(), key=lambda x: x[0]
        )
        self._prepare_value_labels()  # 准备值标签的内部方法


class StataMissingValue:
    """
    观测数据的缺失值表示

    Parameters
    ----------
    value : {int, float}
        Stata 缺失值的代码

    Notes
    -----
    更多信息: <https://www.stata.com/help.cgi?missing>

    整数缺失值对应的 Stata 编码是 '.', '.a', ..., '.z'，范围分别是
    101 ... 127 (int8), 32741 ... 32767 (int16) 和 2147483621 ... 2147483647 (int32)。
    浮点数数据类型的缺失值更复杂，但模式可以从下表中简单推断。

    np.float32 缺失值 (Stata 中的 float)
    0000007f    .
    0008007f    .a
    0010007f    .b
    ...
    00c0007f    .x
    00c8007f    .y
    00d0007f    .z

    np.float64 缺失值 (Stata 中的 double)
    000000000000e07f    .
    000000000001e07f    .a
    000000000002e07f    .b
    ...
    000000000018e07f    .x
    000000000019e07f    .y
    00000000001ae07f    .z
    """

    # 构建缺失值字典
    MISSING_VALUES: dict[float, str] = {}
    bases: Final = (101, 32741, 2147483621)
    for b in bases:
        # 将长整型用于避免在 32 位平台上的哈希问题 #8968
        MISSING_VALUES[b] = "."
        for i in range(1, 27):
            MISSING_VALUES[i + b] = "." + chr(96 + i)

    float32_base: bytes = b"\x00\x00\x00\x7f"
    increment_32: int = struct.unpack("<i", b"\x00\x08\x00\x00")[0]
    for i in range(27):
        key = struct.unpack("<f", float32_base)[0]
        MISSING_VALUES[key] = "."
        if i > 0:
            MISSING_VALUES[key] += chr(96 + i)
        int_value = struct.unpack("<i", struct.pack("<f", key))[0] + increment_32
        float32_base = struct.pack("<i", int_value)

    float64_base: bytes = b"\x00\x00\x00\x00\x00\x00\xe0\x7f"
    increment_64 = struct.unpack("q", b"\x00\x00\x00\x00\x00\x01\x00\x00")[0]
    # 对于范围内的每个索引，生成一个浮点数密钥
    for i in range(27):
        # 从 float64_base 中解包出一个双精度浮点数密钥
        key = struct.unpack("<d", float64_base)[0]
        # 将该密钥映射到 MISSING_VALUES 字典，初始值为 "."
        MISSING_VALUES[key] = "."
        # 如果索引大于 0，则将字符附加到该密钥的映射值
        if i > 0:
            MISSING_VALUES[key] += chr(96 + i)
        # 将密钥打包成双精度浮点数，然后解包并加上增量，最后重新打包以更新 float64_base
        int_value = struct.unpack("q", struct.pack("<d", key))[0] + increment_64
        float64_base = struct.pack("q", int_value)

    # 声明不可变的 BASE_MISSING_VALUES 字典，存储不同数据类型的基本缺失值
    BASE_MISSING_VALUES: Final = {
        "int8": 101,
        "int16": 32741,
        "int32": 2147483621,
        "float32": struct.unpack("<f", float32_base)[0],
        "float64": struct.unpack("<d", float64_base)[0],
    }

    # 类的构造函数，初始化实例的值，并处理 32 位平台上的哈希问题 #8968
    def __init__(self, value: float) -> None:
        # 将传入的浮点数值转换为整数，以避免在 32 位平台上的哈希问题
        self._value = value
        value = int(value) if value < 2147483648 else float(value)
        # 使用 MISSING_VALUES 字典获取与该值对应的字符串表示
        self._str = self.MISSING_VALUES[value]

    @property
    def string(self) -> str:
        """
        缺失值的 Stata 表示形式: '.', '.a'..'.z'

        Returns
        -------
        str
            缺失值的表示形式
        """
        return self._str

    @property
    def value(self) -> float:
        """
        缺失值的二进制表示形式

        Returns
        -------
        {int, float}
            缺失值的二进制表示形式
        """
        return self._value

    def __str__(self) -> str:
        # 返回对象的字符串表示形式
        return self.string

    def __repr__(self) -> str:
        # 返回对象的规范字符串表示形式，包含类型信息
        return f"{type(self)}({self})"

    def __eq__(self, other: object) -> bool:
        # 判断两个对象是否相等
        return (
            isinstance(other, type(self))
            and self.string == other.string
            and self.value == other.value
        )

    @classmethod
    def get_base_missing_value(cls, dtype: np.dtype) -> float:
        # 根据指定的数据类型返回基本的缺失值
        if dtype.type is np.int8:
            value = cls.BASE_MISSING_VALUES["int8"]
        elif dtype.type is np.int16:
            value = cls.BASE_MISSING_VALUES["int16"]
        elif dtype.type is np.int32:
            value = cls.BASE_MISSING_VALUES["int32"]
        elif dtype.type is np.float32:
            value = cls.BASE_MISSING_VALUES["float32"]
        elif dtype.type is np.float64:
            value = cls.BASE_MISSING_VALUES["float64"]
        else:
            raise ValueError("Unsupported dtype")
        return value
class StataParser:
    # 定义 StataParser 类，用于解析 Stata 数据文件
    pass

class StataReader(StataParser, abc.Iterator):
    # StataReader 类继承自 StataParser 类和 abc.Iterator 接口
    __doc__ = _stata_reader_doc  # 设置类文档字符串为 _stata_reader_doc 变量的值

    _path_or_buf: IO[bytes]  # 将由 _open_file 方法赋值的字节流对象路径或缓冲区定义为 IO[bytes]

    def __init__(
        self,
        path_or_buf: FilePath | ReadBuffer[bytes],
        convert_dates: bool = True,
        convert_categoricals: bool = True,
        index_col: str | None = None,
        convert_missing: bool = False,
        preserve_dtypes: bool = True,
        columns: Sequence[str] | None = None,
        order_categoricals: bool = True,
        chunksize: int | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
    ) -> None:
        super().__init__()  # 调用父类 StataParser 的构造函数

        # 设置读取器的参数（在读取调用时可以临时覆盖）
        self._convert_dates = convert_dates
        self._convert_categoricals = convert_categoricals
        self._index_col = index_col
        self._convert_missing = convert_missing
        self._preserve_dtypes = preserve_dtypes
        self._columns = columns
        self._order_categoricals = order_categoricals
        self._original_path_or_buf = path_or_buf  # 原始路径或缓冲区
        self._compression = compression  # 压缩选项
        self._storage_options = storage_options  # 存储选项
        self._encoding = ""  # 编码
        self._chunksize = chunksize  # 块大小
        self._using_iterator = False  # 是否使用迭代器
        self._entered = False  # 是否已进入
        if self._chunksize is None:
            self._chunksize = 1  # 如果未指定块大小，默认为 1
        elif not isinstance(chunksize, int) or chunksize <= 0:
            raise ValueError("chunksize must be a positive integer when set.")  # 如果 chunksize 不是正整数，则抛出 ValueError 异常

        # 文件的状态变量
        self._close_file: Callable[[], None] | None = None  # 关闭文件的函数或 None
        self._column_selector_set = False  # 列选择器是否已设置
        self._value_label_dict: dict[str, dict[int, str]] = {}  # 值标签字典，映射变量名到值标签字典
        self._value_labels_read = False  # 是否已读取值标签
        self._dtype: np.dtype | None = None  # 数据类型或 None
        self._lines_read = 0  # 已读取的行数

        self._native_byteorder = _set_endianness(sys.byteorder)  # 设置本机字节顺序为系统的字节顺序

    def _ensure_open(self) -> None:
        """
        确保文件已打开并读取了其头部数据。
        """
        if not hasattr(self, "_path_or_buf"):
            self._open_file()  # 如果不存在 _path_or_buf 属性，则调用 _open_file 方法打开文件
    def _open_file(self) -> None:
        """
        Open the file (with compression options, etc.), and read header information.
        """
        # 如果未进入上下文管理器，则发出警告
        if not self._entered:
            warnings.warn(
                "StataReader is being used without using a context manager. "
                "Using StataReader as a context manager is the only supported method.",
                ResourceWarning,
                stacklevel=find_stack_level(),
            )
        # 获取文件句柄和处理器，考虑到压缩选项
        handles = get_handle(
            self._original_path_or_buf,
            "rb",
            storage_options=self._storage_options,
            is_text=False,
            compression=self._compression,
        )
        # 如果文件句柄可寻址，则直接使用它
        if hasattr(handles.handle, "seekable") and handles.handle.seekable():
            # 如果句柄直接可寻址，则直接使用它而不进行额外的复制
            self._path_or_buf = handles.handle
            self._close_file = handles.close
        else:
            # 否则，将内容复制到内存中，并确保没有编码
            with handles:
                self._path_or_buf = BytesIO(handles.handle.read())
            self._close_file = self._path_or_buf.close

        # 读取文件头信息
        self._read_header()
        # 设置数据类型
        self._setup_dtype()

    def __enter__(self) -> Self:
        """进入上下文管理器"""
        self._entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # 如果需要关闭文件，则执行关闭操作
        if self._close_file:
            self._close_file()

    def _set_encoding(self) -> None:
        """
        设置依赖于文件版本的字符串编码
        """
        # 如果文件格式版本小于 118，则使用 latin-1 编码，否则使用 utf-8 编码
        if self._format_version < 118:
            self._encoding = "latin-1"
        else:
            self._encoding = "utf-8"

    def _read_int8(self) -> int:
        # 从字节流中读取一个有符号 8 位整数并返回
        return struct.unpack("b", self._path_or_buf.read(1))[0]

    def _read_uint8(self) -> int:
        # 从字节流中读取一个无符号 8 位整数并返回
        return struct.unpack("B", self._path_or_buf.read(1))[0]

    def _read_uint16(self) -> int:
        # 从字节流中读取一个无符号 16 位整数并返回，根据字节顺序进行解析
        return struct.unpack(f"{self._byteorder}H", self._path_or_buf.read(2))[0]

    def _read_uint32(self) -> int:
        # 从字节流中读取一个无符号 32 位整数并返回，根据字节顺序进行解析
        return struct.unpack(f"{self._byteorder}I", self._path_or_buf.read(4))[0]

    def _read_uint64(self) -> int:
        # 从字节流中读取一个无符号 64 位整数并返回，根据字节顺序进行解析
        return struct.unpack(f"{self._byteorder}Q", self._path_or_buf.read(8))[0]

    def _read_int16(self) -> int:
        # 从字节流中读取一个有符号 16 位整数并返回，根据字节顺序进行解析
        return struct.unpack(f"{self._byteorder}h", self._path_or_buf.read(2))[0]

    def _read_int32(self) -> int:
        # 从字节流中读取一个有符号 32 位整数并返回，根据字节顺序进行解析
        return struct.unpack(f"{self._byteorder}i", self._path_or_buf.read(4))[0]

    def _read_int64(self) -> int:
        # 从字节流中读取一个有符号 64 位整数并返回，根据字节顺序进行解析
        return struct.unpack(f"{self._byteorder}q", self._path_or_buf.read(8))[0]

    def _read_char8(self) -> bytes:
        # 从字节流中读取一个字节，并返回其作为 bytes 对象
        return struct.unpack("c", self._path_or_buf.read(1))[0]
    # 从文件或缓冲区中读取指定数量的 16 位有符号整数，并返回为元组
    def _read_int16_count(self, count: int) -> tuple[int, ...]:
        return struct.unpack(
            f"{self._byteorder}{'h' * count}",
            self._path_or_buf.read(2 * count),
        )

    # 读取数据文件的头部信息
    def _read_header(self) -> None:
        first_char = self._read_char8()
        # 如果头部的第一个字符是 '<'，则表示为新版本的头部信息
        if first_char == b"<":
            self._read_new_header()
        else:
            self._read_old_header(first_char)

    # 读取新版本数据文件的头部信息
    def _read_new_header(self) -> None:
        # 读取头部共同部分的固定长度信息（27个字节）
        self._path_or_buf.read(27)  # stata_dta><header><release>
        # 读取数据格式版本号（3个字节）
        self._format_version = int(self._path_or_buf.read(3))
        # 检查版本号是否在支持范围内（117至119），否则抛出数值错误异常
        if self._format_version not in [117, 118, 119]:
            raise ValueError(_version_error.format(version=self._format_version))
        # 设置文件编码方式
        self._set_encoding()
        # 读取部分信息，包括字节顺序信息（21个字节）
        self._path_or_buf.read(21)  # </release><byteorder>
        # 确定字节顺序是大端还是小端
        self._byteorder = ">" if self._path_or_buf.read(3) == b"MSF" else "<"
        # 读取变量数目（2字节或4字节，取决于版本号）
        self._nvar = (
            self._read_uint16() if self._format_version <= 118 else self._read_uint32()
        )
        # 读取观测数目
        self._path_or_buf.read(7)  # </K><N>
        self._nobs = self._get_nobs()
        # 读取数据标签
        self._path_or_buf.read(11)  # </N><label>
        self._data_label = self._get_data_label()
        # 读取时间戳
        self._path_or_buf.read(19)  # </label><timestamp>
        self._time_stamp = self._get_time_stamp()
        # 读取剩余部分的固定长度信息（26个字节）
        self._path_or_buf.read(26)  # </timestamp></header><map>
        self._path_or_buf.read(8)  # 0x0000000000000000
        self._path_or_buf.read(8)  # position of <map>

        # 读取并计算偏移量，获取各种数据区块的位置
        self._seek_vartypes = self._read_int64() + 16
        self._seek_varnames = self._read_int64() + 10
        self._seek_sortlist = self._read_int64() + 10
        self._seek_formats = self._read_int64() + 9
        self._seek_value_label_names = self._read_int64() + 19

        # 版本特定处理，获取变量标签数据的偏移量
        self._seek_variable_labels = self._get_seek_variable_labels()

        # 读取字符集信息（8个字节）
        self._path_or_buf.read(8)  # <characteristics>
        # 读取数据位置信息
        self._data_location = self._read_int64() + 6
        # 读取字符串位置信息
        self._seek_strls = self._read_int64() + 7
        # 读取值标签位置信息
        self._seek_value_labels = self._read_int64() + 14

        # 获取数据类型列表和数据类型描述列表
        self._typlist, self._dtyplist = self._get_dtypes(self._seek_vartypes)

        # 定位到变量名称的位置并获取变量名称列表
        self._path_or_buf.seek(self._seek_varnames)
        self._varlist = self._get_varlist()

        # 定位到排序列表的位置并获取排序列表
        self._path_or_buf.seek(self._seek_sortlist)
        self._srtlist = self._read_int16_count(self._nvar + 1)[:-1]

        # 定位到格式列表的位置并获取格式列表
        self._path_or_buf.seek(self._seek_formats)
        self._fmtlist = self._get_fmtlist()

        # 定位到值标签名称列表的位置并获取值标签名称列表
        self._path_or_buf.seek(self._seek_value_label_names)
        self._lbllist = self._get_lbllist()

        # 定位到变量标签数据的位置并获取变量标签数据
        self._path_or_buf.seek(self._seek_variable_labels)
        self._variable_labels = self._get_variable_labels()

    # 获取数据类型信息，适用于版本号为117至119的数据文件
    def _get_dtypes(
        self, seek_vartypes: int
    # 返回一个元组，包含类型列表和数据类型列表
    def _get_vartypes(self) -> tuple[list[int | str], list[str | np.dtype]]:
        # 将文件指针移到指定位置
        self._path_or_buf.seek(seek_vartypes)
        typlist = []  # 存储变量类型的列表
        dtyplist = []  # 存储数据类型的列表
        for _ in range(self._nvar):
            typ = self._read_uint16()  # 读取一个16位无符号整数
            if typ <= 2045:
                typlist.append(typ)  # 添加变量类型到列表中
                dtyplist.append(str(typ))  # 将变量类型转换为字符串并添加到数据类型列表中
            else:
                try:
                    typlist.append(self.TYPE_MAP_XML[typ])  # 根据类型映射表将类型添加到列表中
                    dtyplist.append(self.DTYPE_MAP_XML[typ])  # 根据数据类型映射表将数据类型添加到列表中
                except KeyError as err:
                    raise ValueError(f"cannot convert stata types [{typ}]") from err

        return typlist, dtyplist  # 返回类型列表和数据类型列表的元组

    def _get_varlist(self) -> list[str]:
        # 根据文件格式版本选择合适的读取长度
        b = 33 if self._format_version < 118 else 129
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    # 返回格式列表
    def _get_fmtlist(self) -> list[str]:
        if self._format_version >= 118:
            b = 57
        elif self._format_version > 113:
            b = 49
        elif self._format_version > 104:
            b = 12
        else:
            b = 7

        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    # 返回标签列表
    def _get_lbllist(self) -> list[str]:
        if self._format_version >= 118:
            b = 129
        elif self._format_version > 108:
            b = 33
        else:
            b = 9
        return [self._decode(self._path_or_buf.read(b)) for _ in range(self._nvar)]

    # 返回变量标签列表
    def _get_variable_labels(self) -> list[str]:
        if self._format_version >= 118:
            vlblist = [
                self._decode(self._path_or_buf.read(321)) for _ in range(self._nvar)
            ]
        elif self._format_version > 105:
            vlblist = [
                self._decode(self._path_or_buf.read(81)) for _ in range(self._nvar)
            ]
        else:
            vlblist = [
                self._decode(self._path_or_buf.read(32)) for _ in range(self._nvar)
            ]
        return vlblist

    # 返回观测值数目
    def _get_nobs(self) -> int:
        if self._format_version >= 118:
            return self._read_uint64()  # 读取64位无符号整数作为观测值数目
        else:
            return self._read_uint32()  # 读取32位无符号整数作为观测值数目

    # 返回数据标签
    def _get_data_label(self) -> str:
        if self._format_version >= 118:
            strlen = self._read_uint16()  # 读取16位无符号整数，表示字符串长度
            return self._decode(self._path_or_buf.read(strlen))  # 解码指定长度的字节流为字符串
        elif self._format_version == 117:
            strlen = self._read_int8()  # 读取8位有符号整数，表示字符串长度
            return self._decode(self._path_or_buf.read(strlen))  # 解码指定长度的字节流为字符串
        elif self._format_version > 105:
            return self._decode(self._path_or_buf.read(81))  # 解码81个字节的数据为字符串
        else:
            return self._decode(self._path_or_buf.read(32))  # 解码32个字节的数据为字符串
    def _get_time_stamp(self) -> str:
        # 如果格式版本大于等于118
        if self._format_version >= 118:
            # 读取一个字节的长度信息
            strlen = self._read_int8()
            # 从路径或缓冲区中读取指定长度的数据，并解码为UTF-8格式的字符串
            return self._path_or_buf.read(strlen).decode("utf-8")
        # 如果格式版本为117
        elif self._format_version == 117:
            # 读取一个字节的长度信息
            strlen = self._read_int8()
            # 从路径或缓冲区中读取指定长度的数据，并调用特定方法解码
            return self._decode(self._path_or_buf.read(strlen))
        # 如果格式版本大于104但小于117
        elif self._format_version > 104:
            # 从路径或缓冲区中读取18个字节的数据，并调用特定方法解码
            return self._decode(self._path_or_buf.read(18))
        else:
            # 其他情况抛出值错误异常
            raise ValueError

    def _get_seek_variable_labels(self) -> int:
        # 如果格式版本为117
        if self._format_version == 117:
            # 读取8个字节的数据，并忽略
            self._path_or_buf.read(8)  # <variable_labels>, throw away
            # 返回一个计算值，用于修正偏移量，具体计算基于变量数目和标签长度等信息
            return self._seek_value_label_names + (33 * self._nvar) + 20 + 17
        # 如果格式版本大于等于118
        elif self._format_version >= 118:
            # 读取一个64位整数，并返回加上一个固定的偏移量
            return self._read_int64() + 17
        else:
            # 其他情况抛出值错误异常
            raise ValueError

    def _setup_dtype(self) -> np.dtype:
        """Map between numpy and state dtypes"""
        # 如果已经存在dtype属性，则直接返回
        if self._dtype is not None:
            return self._dtype

        dtypes = []  # 用于存储转换后的numpy数据类型
        # 遍历类型列表_typlist，转换为numpy数据类型
        for i, typ in enumerate(self._typlist):
            if typ in self.NUMPY_TYPE_MAP:
                typ = cast(str, typ)  # 仅处理NUMPY_TYPE_MAP中的字符串类型
                # 添加到dtypes列表中，使用自定义的格式字符串
                dtypes.append((f"s{i}", f"{self._byteorder}{self.NUMPY_TYPE_MAP[typ]}"))
            else:
                # 对于不在NUMPY_TYPE_MAP中的类型，使用S开头的字符串类型
                dtypes.append((f"s{i}", f"S{typ}"))
        # 将转换后的dtypes列表转换为numpy的dtype对象，并存储在_dtype属性中
        self._dtype = np.dtype(dtypes)

        return self._dtype

    def _decode(self, s: bytes) -> str:
        # 输入参数s为字节类型，需要进行解码操作
        s = s.partition(b"\0")[0]  # 截断首个null字节之后的内容
        try:
            # 尝试使用指定编码解码字节串s，并返回解码后的字符串
            return s.decode(self._encoding)
        except UnicodeDecodeError:
            # 处理Unicode解码错误的情况，针对Stata 117文件转换为118文件时产生的格式问题
            encoding = self._encoding
            msg = f"""
    # 如果在解码数据文件时使用指定的编码（通常是 UTF-8）失败，则发出警告并返回使用 latin-1 编码的结果。
    # 这种情况可能是由于文件被 Stata 或其他软件错误编码而导致的。应验证返回的字符串值是否正确。
    msg = """
One or more strings in the dta file could not be decoded using {encoding}, and
so the fallback encoding of latin-1 is being used.  This can happen when a file
has been incorrectly encoded by Stata or some other software. You should verify
the string values returned are correct."""
    warnings.warn(
        msg,
        UnicodeWarning,
        stacklevel=find_stack_level(),
    )
    return s.decode("latin-1")

def _read_new_value_labels(self) -> None:
    """Reads value labels with variable length strings (108 and later format)"""
    # 根据文件格式版本选择读取值标签的起始位置
    if self._format_version >= 117:
        self._path_or_buf.seek(self._seek_value_labels)
    else:
        assert self._dtype is not None
        offset = self._nobs * self._dtype.itemsize
        self._path_or_buf.seek(self._data_location + offset)

    while True:
        if self._format_version >= 117:
            # 检查是否到达值标签表的结尾
            if self._path_or_buf.read(5) == b"</val":  # <lbl>
                break  # 值标签表结束

        # 读取字符串长度
        slength = self._path_or_buf.read(4)
        if not slength:
            break  # 值标签表结束（格式小于 117），或者文件结束

        # 根据文件格式版本选择合适的字符串长度来读取值标签名称
        if self._format_version == 108:
            labname = self._decode(self._path_or_buf.read(9))
        elif self._format_version <= 117:
            labname = self._decode(self._path_or_buf.read(33))
        else:
            labname = self._decode(self._path_or_buf.read(129))

        self._path_or_buf.read(3)  # 跳过填充部分

        # 读取值标签数量和文本长度
        n = self._read_uint32()
        txtlen = self._read_uint32()

        # 读取偏移和值数组
        off = np.frombuffer(
            self._path_or_buf.read(4 * n), dtype=f"{self._byteorder}i4", count=n
        )
        val = np.frombuffer(
            self._path_or_buf.read(4 * n), dtype=f"{self._byteorder}i4", count=n
        )

        # 根据偏移排序
        ii = np.argsort(off)
        off = off[ii]
        val = val[ii]

        # 读取文本数据
        txt = self._path_or_buf.read(txtlen)

        # 创建值标签字典并填充数据
        self._value_label_dict[labname] = {}
        for i in range(n):
            end = off[i + 1] if i < n - 1 else txtlen
            self._value_label_dict[labname][val[i]] = self._decode(
                txt[off[i] : end]
            )

        if self._format_version >= 117:
            self._path_or_buf.read(6)  # 跳过结束标记 </lbl>
    # 读取固定长度字符串的值标签（适用于格式版本105及更早）
    def _read_old_value_labels(self) -> None:
        # 断言数据类型不为空
        assert self._dtype is not None
        # 计算偏移量，定位到值标签数据的位置
        offset = self._nobs * self._dtype.itemsize
        self._path_or_buf.seek(self._data_location + offset)

        while True:
            # 如果读取的前两个字节为空，则可能已达到文件末尾，因此停止循环
            if not self._path_or_buf.read(2):
                # 文件末尾，停止读取
                break

            # 否则，回退两个字节重新读取，考虑字节顺序
            self._path_or_buf.seek(-2, os.SEEK_CUR)
            # 读取16位无符号整数
            n = self._read_uint16()
            # 解码值标签名称，长度为9个字节
            labname = self._decode(self._path_or_buf.read(9))
            self._path_or_buf.read(1)  # 读取填充字节
            # 从缓冲区读取数据，根据字节顺序生成numpy数组
            codes = np.frombuffer(
                self._path_or_buf.read(2 * n), dtype=f"{self._byteorder}i2", count=n
            )
            # 将值标签及其对应的编码添加到字典中
            self._value_label_dict[labname] = {}
            for i in range(n):
                self._value_label_dict[labname][codes[i]] = self._decode(
                    self._path_or_buf.read(8)
                )

    # 读取值标签
    def _read_value_labels(self) -> None:
        # 确保文件已打开
        self._ensure_open()
        # 如果值标签已经读取过，则直接返回，避免重复读取
        if self._value_labels_read:
            return

        # 根据文件格式版本选择不同的值标签读取方法
        if self._format_version >= 108:
            self._read_new_value_labels()
        else:
            self._read_old_value_labels()
        # 标记值标签已读取
        self._value_labels_read = True

    # 读取字符串长标签（strls）
    def _read_strls(self) -> None:
        # 定位到字符串长标签的起始位置
        self._path_or_buf.seek(self._seek_strls)
        # 创建一个空字典，用于存储字符串长标签的数据
        self.GSO = {"0": ""}
        while True:
            # 如果从缓冲区读取的前3个字节不是 b"GSO"，则退出循环
            if self._path_or_buf.read(3) != b"GSO":
                break

            # 根据格式版本选择不同的读取方法
            if self._format_version == 117:
                v_o = self._read_uint64()
            else:
                buf = self._path_or_buf.read(12)
                # 仅在小端机器上测试过。
                v_size = 2 if self._format_version == 118 else 3
                if self._byteorder == "<":
                    buf = buf[0:v_size] + buf[4 : (12 - v_size)]
                else:
                    buf = buf[4 - v_size : 4] + buf[(4 + v_size) :]
                v_o = struct.unpack(f"{self._byteorder}Q", buf)[0]
            typ = self._read_uint8()
            length = self._read_uint32()
            # 从缓冲区读取长度为length的数据
            va = self._path_or_buf.read(length)
            if typ == 130:
                # 如果类型为130，则解码字符串
                decoded_va = va[0:-1].decode(self._encoding)
            else:
                # 否则，将数据转换为字符串
                decoded_va = str(va)
            # 将值添加到GSO字典中，使用v_o作为键
            self.GSO[str(v_o)] = decoded_va

    # 迭代器的下一个元素
    def __next__(self) -> DataFrame:
        # 标记正在使用迭代器
        self._using_iterator = True
        # 调用read方法读取指定行数的数据帧
        return self.read(nrows=self._chunksize)
    def get_chunk(self, size: int | None = None) -> DataFrame:
        """
        从 Stata 文件中读取行，并将其作为 DataFrame 返回

        Parameters
        ----------
        size : int, defaults to None
            要读取的行数。如果为 None，则读取整个文件。

        Returns
        -------
        DataFrame
        """
        if size is None:
            size = self._chunksize  # 如果 size 为 None，则使用默认的块大小
        return self.read(nrows=size)  # 调用 read 方法读取指定行数的数据

    @Appender(_read_method_doc)
    def read(
        self,
        nrows: int | None = None,
        convert_dates: bool | None = None,
        convert_categoricals: bool | None = None,
        index_col: str | None = None,
        convert_missing: bool | None = None,
        preserve_dtypes: bool | None = None,
        columns: Sequence[str] | None = None,
        order_categoricals: bool | None = None,
    ):
        """
        从 Stata 文件中读取数据，并根据参数进行转换和处理

        Parameters
        ----------
        nrows : int, optional
            要读取的行数，默认为 None，表示读取整个文件。
        convert_dates : bool, optional
            是否转换日期类型，默认为 None。
        convert_categoricals : bool, optional
            是否转换分类变量，默认为 None。
        index_col : str, optional
            指定作为索引的列名，默认为 None。
        convert_missing : bool, optional
            是否转换缺失值，默认为 None。
        preserve_dtypes : bool, optional
            是否保留数据类型，默认为 None。
        columns : Sequence[str], optional
            指定要读取的列名列表，默认为 None。
        order_categoricals : bool, optional
            是否对分类变量排序，默认为 None。

        Returns
        -------
        DataFrame
            包含读取数据的 DataFrame 对象。
        """

    def _do_convert_missing(self, data: DataFrame, convert_missing: bool) -> DataFrame:
        """
        检查数据中的缺失值，并根据需求进行替换

        Parameters
        ----------
        data : DataFrame
            要处理的数据框。
        convert_missing : bool
            是否转换缺失值。

        Returns
        -------
        DataFrame
            处理后的数据框，替换了缺失值。
        """
        # 检查缺失值，并进行替换
        replacements = {}
        for i in range(len(data.columns)):
            fmt = self._typlist[i]
            if fmt not in self.VALID_RANGE:
                continue

            fmt = cast(str, fmt)  # 确保 fmt 在 VALID_RANGE 中是字符串类型
            nmin, nmax = self.VALID_RANGE[fmt]
            series = data.iloc[:, i]

            # 使用 ndarray 而不是 Series 可以更快地进行检查和替换
            svals = series._values
            missing = (svals < nmin) | (svals > nmax)

            if not missing.any():
                continue

            if convert_missing:  # 如果需要转换缺失值，则按 Stata 的约定进行替换
                missing_loc = np.nonzero(np.asarray(missing))[0]
                umissing, umissing_loc = np.unique(series[missing], return_inverse=True)
                replacement = Series(series, dtype=object)
                for j, um in enumerate(umissing):
                    missing_value = StataMissingValue(um)

                    loc = missing_loc[umissing_loc == j]
                    replacement.iloc[loc] = missing_value
            else:  # 否则，所有的缺失值替换为相同的值
                dtype = series.dtype
                if dtype not in (np.float32, np.float64):
                    dtype = np.float64
                replacement = Series(series, dtype=dtype)
                # 注意：在 ._values 上操作比直接操作更快
                # TODO: 能否修复这个问题？
                replacement._values[missing] = np.nan
            replacements[i] = replacement

        if replacements:
            for idx, value in replacements.items():
                data.isetitem(idx, value)  # 使用 isetitem 方法替换数据框中的值

        return data  # 返回替换了缺失值的数据框
    # 插入字符串列表操作，用于处理特定数据帧的某些列
    def _insert_strls(self, data: DataFrame) -> DataFrame:
        # 如果对象没有属性 "GSO" 或者 "GSO" 的长度为 0，则直接返回原始数据帧
        if not hasattr(self, "GSO") or len(self.GSO) == 0:
            return data
        # 遍历数据帧中的列类型列表
        for i, typ in enumerate(self._typlist):
            # 如果列类型不为 "Q"，则跳过当前循环
            if typ != "Q":
                continue
            # 将数据帧第 i 列的每个元素转换为字符串，并用其作为键值从 self.GSO 字典中获取对应的值，更新到数据帧中第 i 列
            data.isetitem(i, [self.GSO[str(k)] for k in data.iloc[:, i]])
        # 返回更新后的数据帧
        return data

    # 执行列选择操作，返回仅包含指定列的数据帧
    def _do_select_columns(self, data: DataFrame, columns: Sequence[str]) -> DataFrame:
        # 如果未设置列选择器，则进行列匹配和信息复制操作
        if not self._column_selector_set:
            # 将列名列表转换为集合，检查是否有重复列名
            column_set = set(columns)
            if len(column_set) != len(columns):
                raise ValueError("columns contains duplicate entries")
            # 找出数据帧中未匹配的列名
            unmatched = column_set.difference(data.columns)
            if unmatched:
                joined = ", ".join(list(unmatched))
                # 抛出异常，指示在 Stata 数据集中未找到指定的列名
                raise ValueError(
                    "The following columns were not "
                    f"found in the Stata data set: {joined}"
                )
            # 复制保留列的信息以备后续处理
            dtyplist = []
            typlist = []
            fmtlist = []
            lbllist = []
            for col in columns:
                # 获取列在数据帧中的位置索引，并根据索引获取相关信息列表的对应元素
                i = data.columns.get_loc(col)  # type: ignore[no-untyped-call]
                dtyplist.append(self._dtyplist[i])
                typlist.append(self._typlist[i])
                fmtlist.append(self._fmtlist[i])
                lbllist.append(self._lbllist[i])

            # 更新对象的数据类型、列类型、格式和标签信息列表
            self._dtyplist = dtyplist
            self._typlist = typlist
            self._fmtlist = fmtlist
            self._lbllist = lbllist
            self._column_selector_set = True

        # 返回仅包含指定列的数据帧
        return data[columns]

    # 执行分类变量转换操作，返回更新后的数据帧
    def _do_convert_categoricals(
        self,
        data: DataFrame,
        value_label_dict: dict[str, dict[int, str]],
        lbllist: Sequence[str],
        order_categoricals: bool,
        ) -> DataFrame:
        """
        Converts categorical columns to Categorical type.
        """
        # 如果值-标签字典为空，则直接返回数据，不进行转换
        if not value_label_dict:
            return data
        # 初始化一个空列表，用于存储转换后的数据
        cat_converted_data = []
        # 遍历数据列和标签列表的并行关系
        for col, label in zip(data, lbllist):
            # 如果标签在值-标签字典中
            if label in value_label_dict:
                # 显式调用有序的情况
                vl = value_label_dict[label]
                # 提取字典键，并转换为数组
                keys = np.array(list(vl.keys()))
                # 获取当前列的数据
                column = data[col]
                # 检查当前列数据是否都在键集合中
                key_matches = column.isin(keys)
                # 如果正在使用迭代器且所有类别都匹配
                if self._using_iterator and key_matches.all():
                    # 初始类别设定为键数组
                    initial_categories: np.ndarray | None = keys
                    # 如果所有类别都匹配且正在迭代，则在所有块中使用相同的键。
                    # 如果某些类别缺少值标签，则会回退到跨块变化的类别。
                else:
                    # 如果正在使用迭代器
                    if self._using_iterator:
                        # 发出警告表明正在使用迭代器
                        warnings.warn(
                            categorical_conversion_warning,
                            CategoricalConversionWarning,
                            stacklevel=find_stack_level(),
                        )
                    # 否则初始类别设为None
                    initial_categories = None
                # 创建分类数据对象，使用初始类别和有序标志
                cat_data = Categorical(
                    column, categories=initial_categories, ordered=order_categoricals
                )
                # 如果初始类别为None，则需要匹配分类数据中的类别
                if initial_categories is None:
                    # 初始化一个空列表，用于存储转换后的类别
                    categories = []
                    # 遍历当前分类数据中的每个类别
                    for category in cat_data.categories:
                        # 如果类别在值-标签字典中
                        if category in vl:
                            # 将其映射到对应的值标签
                            categories.append(vl[category])
                        else:
                            # 否则保持原始类别
                            categories.append(category)
                else:
                    # 如果所有类别都匹配，则直接使用值标签的值作为类别
                    categories = list(vl.values())
                try:
                    # 尝试重新命名类别以避免重复
                    # TODO: 如果可以使用非复制的rename_categories，则使用它
                    cat_data = cat_data.rename_categories(categories)
                except ValueError as err:
                    # 获取重复类别并生成消息
                    vc = Series(categories, copy=False).value_counts()
                    repeated_cats = list(vc.index[vc > 1])
                    repeats = "-" * 80 + "\n" + "\n".join(repeated_cats)
                    # GH 25772
                    msg = f"""
    def data(self, convert_dates: bool = True, convert_categoricals: bool = True) -> DataFrame:
        """
        Return the data in the Stata file as a pandas DataFrame.

        Parameters
        ----------
        convert_dates : bool, default True
            Whether to convert Stata dates to pandas datetime objects.
        convert_categoricals : bool, default True
            Whether to convert Stata categorical variables to pandas categoricals.

        Returns
        -------
        pandas.DataFrame
            The DataFrame containing the data from the Stata file.

        Raises
        ------
        ValueError
            If value labels for any column are not unique.

        Notes
        -----
        Either read the file with `convert_categoricals` set to False or use the
        low level interface in `StataReader` to separately read the values and the
        value_labels.

        Examples
        --------
        >>> df = pd.DataFrame([(1,)], columns=["variable"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> data_label = "This is a data file."
        >>> path = "/My_path/filename.dta"
        >>> df.to_stata(
        ...     path,
        ...     time_stamp=time_stamp,  # doctest: +SKIP
        ...     data_label=data_label,  # doctest: +SKIP
        ...     version=None,
        ... )  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.data())  # doctest: +SKIP
           variable
        0         1
        """
        self._ensure_open()
        data = []
        for col in self._varlist:
            if convert_categoricals and col in self._variable_labels:
                repeats = self._variable_labels[col]
                if len(set(repeats.values())) != len(repeats):
                    msg = (
                        f"Value labels for column {col} are not unique. "
                        "These cannot be converted to pandas categoricals.\n\n"
                        "Either read the file with `convert_categoricals` set to False or use the "
                        "low level interface in `StataReader` to separately read the values and the "
                        "value_labels.\n\n"
                        "The repeated labels are:\n"
                        f"{repeats}"
                    )
                    raise ValueError(msg)  # raise ValueError with detailed message
                cat_data = self.cat(category=col)
                # TODO: is the next line needed above in the data(...) method?
                cat_series = Series(cat_data, index=data.index, copy=False)
                cat_converted_data.append((col, cat_series))
            else:
                cat_converted_data.append((col, data[col]))
        data = DataFrame(dict(cat_converted_data), copy=False)
        return data
    def value_labels(self) -> dict[str, dict[int, str]]:
        """
        Return a nested dict associating each variable name to its value and label.

        Returns
        -------
        dict
            A dictionary where keys are variable names and values are dictionaries
            mapping integer values to their corresponding string labels.

        Examples
        --------
        >>> df = pd.DataFrame([[1, 2], [3, 4]], columns=["col_1", "col_2"])
        >>> time_stamp = pd.Timestamp(2000, 2, 29, 14, 21)
        >>> path = "/My_path/filename.dta"
        >>> value_labels = {"col_1": {3: "x"}}
        >>> df.to_stata(
        ...     path,
        ...     time_stamp=time_stamp,  # doctest: +SKIP
        ...     value_labels=value_labels,
        ...     version=None,
        ... )  # doctest: +SKIP
        >>> with pd.io.stata.StataReader(path) as reader:  # doctest: +SKIP
        ...     print(reader.value_labels())  # doctest: +SKIP
        {'col_1': {3: 'x'}}
        >>> pd.read_stata(path)  # doctest: +SKIP
            index col_1 col_2
        0       0    1    2
        1       1    x    4
        """
        # 如果尚未读取过变量标签，则调用 _read_value_labels 方法读取
        if not self._value_labels_read:
            self._read_value_labels()

        # 返回存储变量标签字典的属性 _value_label_dict
        return self._value_label_dict
@Appender(_read_stata_doc)
# 定义 read_stata 函数，添加到 _read_stata_doc 的注解中
def read_stata(
    filepath_or_buffer: FilePath | ReadBuffer[bytes],
    *,
    convert_dates: bool = True,
    convert_categoricals: bool = True,
    index_col: str | None = None,
    convert_missing: bool = False,
    preserve_dtypes: bool = True,
    columns: Sequence[str] | None = None,
    order_categoricals: bool = True,
    chunksize: int | None = None,
    iterator: bool = False,
    compression: CompressionOptions = "infer",
    storage_options: StorageOptions | None = None,
) -> DataFrame | StataReader:
    # 创建 StataReader 实例，传入参数以配置读取行为
    reader = StataReader(
        filepath_or_buffer,
        convert_dates=convert_dates,
        convert_categoricals=convert_categoricals,
        index_col=index_col,
        convert_missing=convert_missing,
        preserve_dtypes=preserve_dtypes,
        columns=columns,
        order_categoricals=order_categoricals,
        chunksize=chunksize,
        storage_options=storage_options,
        compression=compression,
    )

    # 如果需要迭代或使用分块读取，直接返回 reader 对象
    if iterator or chunksize:
        return reader

    # 否则，在上下文管理器中使用 reader，并返回其读取的结果
    with reader:
        return reader.read()


def _set_endianness(endianness: str) -> str:
    # 根据给定的 endianness 设定数据的字节顺序
    if endianness.lower() in ["<", "little"]:
        return "<"  # 返回小端字节顺序标识
    elif endianness.lower() in [">", "big"]:
        return ">"  # 返回大端字节顺序标识
    else:  # 如果不是已知的字节顺序标识，则引发 ValueError 异常
        raise ValueError(f"Endianness {endianness} not understood")


def _pad_bytes(name: AnyStr, length: int) -> AnyStr:
    """
    Take a char string and pads it with null bytes until it's length chars.
    """
    # 将字符串填充到指定长度，确保字符串长度为 length，使用空字节填充
    if isinstance(name, bytes):
        return name + b"\x00" * (length - len(name))  # 对字节类型的字符串进行填充
    return name + "\x00" * (length - len(name))  # 对非字节类型的字符串进行填充


def _convert_datetime_to_stata_type(fmt: str) -> np.dtype:
    """
    Convert from one of the stata date formats to a type in TYPE_MAP.
    """
    # 将 stata 的日期格式转换为 TYPE_MAP 中的数据类型
    if fmt in [
        "tc",
        "%tc",
        "td",
        "%td",
        "tw",
        "%tw",
        "tm",
        "%tm",
        "tq",
        "%tq",
        "th",
        "%th",
        "ty",
        "%ty",
    ]:
        return np.dtype(np.float64)  # Stata 期望 SIFs 使用双精度浮点数
    else:
        raise NotImplementedError(f"Format {fmt} not implemented")


def _maybe_convert_to_int_keys(convert_dates: dict, varlist: list[Hashable]) -> dict:
    # 将 convert_dates 中的键可能转换为整数键，与 varlist 中的变量名匹配
    new_dict = {}
    for key in convert_dates:
        if not convert_dates[key].startswith("%"):  # 确保日期格式以 '%' 开头
            convert_dates[key] = "%" + convert_dates[key]
        if key in varlist:
            new_dict.update({varlist.index(key): convert_dates[key]})
        else:
            if not isinstance(key, int):
                raise ValueError("convert_dates key must be a column or an integer")
            new_dict.update({key: convert_dates[key]})
    return new_dict


def _dtype_to_stata_type(dtype: np.dtype, column: Series) -> int:
    """
    Convert dtype types to stata types. Returns the byte of the given ordinal.
    See TYPE_MAP and comments for an explanation. This is also explained in
    the dta spec.
    """
    # 将 dtype 类型转换为 stata 类型的字节表示
    # 返回给定序号的字节表示，参见 TYPE_MAP 和相关注释的解释
    # 根据数据类型返回对应的Pandas或Stata数据类型编码
    def pandas_dtype(dtype):
        # TODO: expand to handle datetime to integer conversion
        # 如果数据类型是对象类型（通常用于字符串），则尝试将其强制转换为最大的字符串长度
        if dtype.type is np.object_:  # try to coerce it to the biggest string
            # 内存不高效，是否有其他方法可以改进？
            itemsize = max_len_string_array(ensure_object(column._values))
            return max(itemsize, 1)
        # 如果数据类型是np.float64，则返回255，对应双精度浮点数（Stata）
        elif dtype.type is np.float64:
            return 255
        # 如果数据类型是np.float32，则返回254，对应单精度浮点数（Stata）
        elif dtype.type is np.float32:
            return 254
        # 如果数据类型是np.int32，则返回253，对应32位整数（Stata）
        elif dtype.type is np.int32:
            return 253
        # 如果数据类型是np.int16，则返回252，对应16位整数（Stata）
        elif dtype.type is np.int16:
            return 252
        # 如果数据类型是np.int8，则返回251，对应8位整数（Stata）
        elif dtype.type is np.int8:
            return 251
        else:  # 如果数据类型不受支持，则抛出未实现错误
            raise NotImplementedError(f"Data type {dtype} not supported.")
# 定义一个函数，将 NumPy 的数据类型映射到 Stata 中的默认格式字符串
def _dtype_to_default_stata_fmt(
    dtype: np.dtype, column: Series, dta_version: int = 114, force_strl: bool = False
) -> str:
    """
    Map numpy dtype to stata's default format for this type. Not terribly
    important since users can change this in Stata. Semantics are

    object  -> "%DDs" where DD is the length of the string.  If not a string,
                raise ValueError
    float64 -> "%10.0g"
    float32 -> "%9.0g"
    int64   -> "%9.0g"
    int32   -> "%12.0g"
    int16   -> "%8.0g"
    int8    -> "%8.0g"
    strl    -> "%9s"
    """
    # 根据 Stata 版本设置字符串最大长度限制
    if dta_version < 117:
        max_str_len = 244
    else:
        max_str_len = 2045
        # 如果强制使用 strl 类型，则返回 "%9s" 格式
        if force_strl:
            return "%9s"
    # 如果数据类型是 object (字符串)
    if dtype.type is np.object_:
        # 确保列的对象值并计算最大字符串长度
        itemsize = max_len_string_array(ensure_object(column._values))
        # 如果字符串超过最大长度限制则抛出异常或调整格式
        if itemsize > max_str_len:
            if dta_version >= 117:
                return "%9s"
            else:
                raise ValueError(excessive_string_length_error.format(column.name))
        # 返回格式化字符串，最小长度为1
        return "%" + str(max(itemsize, 1)) + "s"
    elif dtype == np.float64:
        return "%10.0g"
    elif dtype == np.float32:
        return "%9.0g"
    elif dtype == np.int32:
        return "%12.0g"
    elif dtype in (np.int8, np.int16):
        return "%8.0g"
    else:  # 数据类型不支持则引发异常
        raise NotImplementedError(f"Data type {dtype} not supported.")


# 使用文档字符串装饰器配置 StataWriter 类，继承自 StataParser
@doc(
    storage_options=_shared_docs["storage_options"],
    compression_options=_shared_docs["compression_options"] % "fname",
)
class StataWriter(StataParser):
    """
    A class for writing Stata binary dta files

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, pathlib.Path or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
    data : DataFrame
        Input to save
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
    write_index : bool
        Write the index to Stata dataset.
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
    """
    # StataWriter 类用于写入 Stata 二进制数据文件
    pass  # 空白语句，仅用于占位符，不执行任何操作
    {compression_options}
        压缩选项，用于指定如何压缩数据文件。

        .. versionchanged:: 1.4.0 Zstandard 支持。

    {storage_options}
        存储选项，用于指定数据文件的存储设置。

    value_labels : dict of dicts
        包含列名为键、字典为值的字典。每个字典表示列值到标签的映射关系。
        单个变量所有标签的总长度不能超过32,000个字符。

        .. versionadded:: 1.4.0
            版本新增功能说明。

    Returns
    -------
    writer : StataWriter instance
        返回一个 StataWriter 实例，该实例具有 write_file 方法，可以将文件写入指定的 `fname`。

    Raises
    ------
    NotImplementedError
        * 如果日期时间包含时区信息时
    ValueError
        * 列在 convert_dates 中列出但其数据类型既不是 datetime64[ns] 也不是 datetime 时
        * 列数据类型无法在 Stata 中表示时
        * convert_dates 中列出的列不在 DataFrame 中时
        * 分类标签包含超过32,000个字符时

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1]], columns=["a", "b"])
    >>> writer = StataWriter("./data_file.dta", data)
    >>> writer.write_file()

    直接写入一个 ZIP 文件
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    >>> writer = StataWriter("./data_file.zip", data, compression=compression)
    >>> writer.write_file()

    保存一个包含日期的 DataFrame
    >>> from datetime import datetime
    >>> data = pd.DataFrame([[datetime(2000, 1, 1)]], columns=["date"])
    >>> writer = StataWriter("./date_data_file.dta", data, {"date": "tw"})
    >>> writer.write_file()
    ) -> None:
        super().__init__()
        self.data = data
        self._convert_dates = {} if convert_dates is None else convert_dates
        self._write_index = write_index
        self._time_stamp = time_stamp
        self._data_label = data_label
        self._variable_labels = variable_labels
        self._non_cat_value_labels = value_labels
        self._value_labels: list[StataValueLabel] = []
        self._has_value_labels = np.array([], dtype=bool)
        self._compression = compression
        self._output_file: IO[bytes] | None = None
        self._converted_names: dict[Hashable, str] = {}
        # attach nobs, nvars, data, varlist, typlist
        self._prepare_pandas(data)
        self.storage_options = storage_options
        
        # 如果未指定字节序，则使用系统默认字节序
        if byteorder is None:
            byteorder = sys.byteorder
        # 根据给定的字节序设置对象的字节序
        self._byteorder = _set_endianness(byteorder)
        # 将文件名存储在对象中
        self._fname = fname
        # 类型转换器，用于将特定类型的数据转换为 NumPy 类型
        self.type_converters = {253: np.int32, 252: np.int16, 251: np.int8}

    def _write(self, to_write: str) -> None:
        """
        Helper to call encode before writing to file for Python 3 compat.
        """
        # 将字符串编码为字节流后写入文件的辅助方法
        self.handles.handle.write(to_write.encode(self._encoding))

    def _write_bytes(self, value: bytes) -> None:
        """
        Helper to assert file is open before writing.
        """
        # 在写入文件之前确保文件已经打开的辅助方法
        self.handles.handle.write(value)

    def _prepare_non_cat_value_labels(
        self, data: DataFrame
    ) -> list[StataNonCatValueLabel]:
        """
        Check for value labels provided for non-categorical columns. Value
        labels
        """
        # 存储非分类列的值标签的准备方法
        non_cat_value_labels: list[StataNonCatValueLabel] = []
        # 如果未提供非分类列的值标签，则返回空列表
        if self._non_cat_value_labels is None:
            return non_cat_value_labels

        # 遍历每个值标签，为每个非分类列生成 StataNonCatValueLabel 对象
        for labname, labels in self._non_cat_value_labels.items():
            # 如果值标签的名称已经转换过，则使用转换后的列名
            if labname in self._converted_names:
                colname = self._converted_names[labname]
            # 否则，使用原始的值标签名称作为列名
            elif labname in data.columns:
                colname = str(labname)
            else:
                # 如果值标签名称在数据集中找不到，则引发 KeyError 异常
                raise KeyError(
                    f"Can't create value labels for {labname}, it wasn't "
                    "found in the dataset."
                )

            # 检查数据列的类型是否为数值类型，因为值标签仅适用于数值列
            if not is_numeric_dtype(data[colname].dtype):
                # 如果数据列不是数值类型，则引发 ValueError 异常
                raise ValueError(
                    f"Can't create value labels for {labname}, value labels "
                    "can only be applied to numeric columns."
                )
            # 创建 StataNonCatValueLabel 对象并添加到列表中
            svl = StataNonCatValueLabel(colname, labels, self._encoding)
            non_cat_value_labels.append(svl)
        return non_cat_value_labels
    def _prepare_categoricals(self, data: DataFrame) -> DataFrame:
        """
        检查是否有分类列，保留 Stata 文件的分类信息并将分类数据转换为整数
        """
        # 检查每列数据类型是否为 CategoricalDtype
        is_cat = [isinstance(dtype, CategoricalDtype) for dtype in data.dtypes]
        # 如果没有任何分类列，则直接返回原始数据
        if not any(is_cat):
            return data

        # 标记对象是否含有值标签（value labels）
        self._has_value_labels |= np.array(is_cat)

        # 获取 StataMissingValue 类的基础缺失值方法
        get_base_missing_value = StataMissingValue.get_base_missing_value
        data_formatted = []
        # 遍历数据的列，及其对应的是否为分类列的标志
        for col, col_is_cat in zip(data, is_cat):
            if col_is_cat:
                # 创建 StataValueLabel 对象，用于处理值标签
                svl = StataValueLabel(data[col], encoding=self._encoding)
                self._value_labels.append(svl)
                # 获取分类数据的编码后的数据类型
                dtype = data[col].cat.codes.dtype
                # 如果数据类型为 np.int64，则无法导出到 Stata，抛出异常
                if dtype == np.int64:
                    raise ValueError(
                        "It is not possible to export "
                        "int64-based categorical data to Stata."
                    )
                # 复制分类数据的编码值
                values = data[col].cat.codes._values.copy()

                # 如果值的最大值超过了当前数据类型的基础缺失值，则进行类型上的升级
                if values.max() >= get_base_missing_value(dtype):
                    if dtype == np.int8:
                        dtype = np.dtype(np.int16)
                    elif dtype == np.int16:
                        dtype = np.dtype(np.int32)
                    else:
                        dtype = np.dtype(np.float64)
                    values = np.array(values, dtype=dtype)

                # 将缺失值替换为该类型的 Stata 缺失值
                values[values == -1] = get_base_missing_value(dtype)
                data_formatted.append((col, values))
            else:
                # 如果不是分类列，则保持列的原始数据
                data_formatted.append((col, data[col]))
        # 根据处理后的数据字典创建 DataFrame 并返回
        return DataFrame.from_dict(dict(data_formatted))

    def _replace_nans(self, data: DataFrame) -> DataFrame:
        # 替换缺失值为 Stata 标准的缺失值
        """
        检查浮点数数据列中的 NaN 值，并用 Stata 的通用缺失值 (.) 替换这些值
        """
        # 遍历数据的每一列
        for c in data:
            dtype = data[c].dtype
            # 如果列的数据类型为浮点数（np.float32 或 np.float64）
            if dtype in (np.float32, np.float64):
                # 根据数据类型选择适当的 Stata 缺失值进行替换
                if dtype == np.float32:
                    replacement = self.MISSING_VALUES["f"]
                else:
                    replacement = self.MISSING_VALUES["d"]
                # 使用 Stata 缺失值替换列中的 NaN 值
                data[c] = data[c].fillna(replacement)

        # 返回替换后的 DataFrame
        return data

    def _update_strl_names(self) -> None:
        """无操作函数，用于向前兼容"""
    # 定义一个方法用于验证变量名，确保符合 Stata 导出的要求
    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 114 and 117 support ascii characters in a-z, A-Z, 0-9
        and _.
        """
        # 遍历变量名中的每个字符
        for c in name:
            # 检查字符是否在 Stata 允许的字符范围内，否则替换为下划线
            if (
                (c < "A" or c > "Z")
                and (c < "a" or c > "z")
                and (c < "0" or c > "9")
                and c != "_"
            ):
                name = name.replace(c, "_")
        # 返回经过验证处理后的变量名
        return name
    def _check_column_names(self, data: DataFrame) -> DataFrame:
        """
        Checks column names to ensure that they are valid Stata column names.
        This includes checks for:
            * Non-string names
            * Stata keywords
            * Variables that start with numbers
            * Variables with names that are too long

        When an illegal variable name is detected, it is converted, and if
        dates are exported, the variable name is propagated to the date
        conversion dictionary
        """
        # 初始化一个空字典，用于存储转换后的列名
        converted_names: dict[Hashable, str] = {}
        # 获取数据框中的列名列表
        columns = list(data.columns)
        # 创建原始列名的副本
        original_columns = columns[:]

        # 初始化变量名重复计数器
        duplicate_var_id = 0
        # 遍历列名列表
        for j, name in enumerate(columns):
            # 保留原始列名备份
            orig_name = name
            # 如果列名不是字符串，转换为字符串类型
            if not isinstance(name, str):
                name = str(name)

            # 调用方法验证变量名的有效性
            name = self._validate_variable_name(name)

            # 变量名不能是保留字
            if name in self.RESERVED_WORDS:
                name = "_" + name

            # 变量名不能以数字开头
            if "0" <= name[0] <= "9":
                name = "_" + name

            # 变量名最长不超过32个字符
            name = name[: min(len(name), 32)]

            # 如果变量名发生了变化
            if not name == orig_name:
                # 检查是否有重复的变量名
                while columns.count(name) > 0:
                    # 避免重复，添加递增数字前缀
                    name = "_" + str(duplicate_var_id) + name
                    name = name[: min(len(name), 32)]
                    duplicate_var_id += 1
                # 记录原始列名到转换后的列名的映射关系
                converted_names[orig_name] = name

            # 更新列名列表中的当前列名
            columns[j] = name

        # 更新数据框的列名
        data.columns = Index(columns)

        # 如果需要进行日期转换，检查并更新日期转换字典
        if self._convert_dates:
            for c, o in zip(columns, original_columns):
                if c != o:
                    self._convert_dates[c] = self._convert_dates[o]
                    del self._convert_dates[o]

        # 如果有转换过的列名，生成警告消息并抛出警告
        if converted_names:
            conversion_warning = []
            for orig_name, name in converted_names.items():
                msg = f"{orig_name}   ->   {name}"
                conversion_warning.append(msg)

            # 构造警告消息文本
            ws = invalid_name_doc.format("\n    ".join(conversion_warning))
            # 抛出警告
            warnings.warn(
                ws,
                InvalidColumnName,
                stacklevel=find_stack_level(),
            )

        # 记录已转换的列名到对象属性
        self._converted_names = converted_names
        # 更新字符串列表名
        self._update_strl_names()

        # 返回更新后的数据框
        return data

    def _set_formats_and_types(self, dtypes: Series) -> None:
        # 初始化格式列表和类型列表
        self.fmtlist: list[str] = []
        self.typlist: list[int] = []
        # 遍历数据类型序列
        for col, dtype in dtypes.items():
            # 添加格式到格式列表
            self.fmtlist.append(_dtype_to_default_stata_fmt(dtype, self.data[col]))
            # 添加类型到类型列表
            self.typlist.append(_dtype_to_stata_type(dtype, self.data[col]))
    # 准备将 pandas 数据转换为适合写入 Stata 文件的格式
    def _prepare_pandas(self, data: DataFrame) -> None:
        # 注意：可能需要不同的 API 或类来处理 pandas 对象，以便设置不同的语义 - 可通过向 pandas.io 提交 PR 处理这个问题

        # 复制数据，以确保不改变原始数据
        data = data.copy()

        # 如果需要写入索引，则重置数据并更新为新的 DataFrame
        if self._write_index:
            temp = data.reset_index()
            if isinstance(temp, DataFrame):
                data = temp

        # 确保列名为字符串类型
        data = self._check_column_names(data)

        # 检查列的类型是否与 Stata 兼容，必要时进行类型转换
        data = _cast_to_stata_types(data)

        # 将 NaN 替换为 Stata 的缺失值
        data = self._replace_nans(data)

        # 初始化所有列为未标记状态
        self._has_value_labels = np.repeat(False, data.shape[1])

        # 为非分类数据准备值标签
        non_cat_value_labels = self._prepare_non_cat_value_labels(data)

        # 获取非分类数据列的标签名，并检查是否存在值标签
        non_cat_columns = [svl.labname for svl in non_cat_value_labels]
        has_non_cat_val_labels = data.columns.isin(non_cat_columns)
        self._has_value_labels |= has_non_cat_val_labels
        self._value_labels.extend(non_cat_value_labels)

        # 准备分类数据，转换为整数数据并去除标签
        data = self._prepare_categoricals(data)

        # 设置数据集的观测值数和变量数
        self.nobs, self.nvar = data.shape
        self.data = data
        self.varlist = data.columns.tolist()

        # 检查并转换所有日期列
        dtypes = data.dtypes
        for col in data:
            if col in self._convert_dates:
                continue
            if lib.is_np_dtype(data[col].dtype, "M"):
                self._convert_dates[col] = "tc"

        # 将日期转换配置应用到整数键
        self._convert_dates = _maybe_convert_to_int_keys(
            self._convert_dates, self.varlist
        )

        # 将日期列的数据类型转换为 Stata 支持的类型
        for key in self._convert_dates:
            new_type = _convert_datetime_to_stata_type(self._convert_dates[key])
            dtypes.iloc[key] = np.dtype(new_type)

        # 确保对象数组为字符串并转换为字节表示
        self._encode_strings()

        # 设置格式和类型
        self._set_formats_and_types(dtypes)

        # 如果需要，为日期列设置指定的格式
        if self._convert_dates is not None:
            for key in self._convert_dates:
                if isinstance(key, int):
                    self.fmtlist[key] = self._convert_dates[key]
        def _encode_strings(self) -> None:
            """
            Encode strings in dta-specific encoding

            Do not encode columns marked for date conversion or for strL
            conversion. The strL converter independently handles conversion and
            also accepts empty string arrays.
            """
            # 获取日期转换标记列表
            convert_dates = self._convert_dates
            # 获取strL转换标记列表（在dta 114版本中未必可用）
            convert_strl = getattr(self, "_convert_strl", [])
            # 遍历数据的每一列
            for i, col in enumerate(self.data):
                # 如果当前列在日期转换标记列表或strL转换标记列表中，则跳过
                if i in convert_dates or col in convert_strl:
                    continue
                # 获取当前列的数据
                column = self.data[col]
                # 获取当前列的数据类型
                dtype = column.dtype
                # 如果数据类型是numpy的object类型
                if dtype.type is np.object_:
                    # 推断当前列的数据类型
                    inferred_dtype = infer_dtype(column, skipna=True)
                    # 如果推断出的数据类型不是字符串，并且列长度不为0
                    if not ((inferred_dtype == "string") or len(column) == 0):
                        # 获取列名
                        col = column.name
                        # 抛出数值错误异常，提供详细的错误信息
                        raise ValueError(
                            f"""\
    def _close(self) -> None:
        """
        Close the file if it was created by the writer.

        If a buffer or file-like object was passed in, for example a GzipFile,
        then leave this file open for the caller to close.
        """
        # 如果输出文件不为None，则进行写入压缩
        if self._output_file is not None:
            # 确保handles.handle是BytesIO类型
            assert isinstance(self.handles.handle, BytesIO)
            # 将handles.handle赋值给bio，并将self._output_file赋值给self.handles.handle
            bio, self.handles.handle = self.handles.handle, self._output_file
            # 将bio的内容写入self.handles.handle中
            self.handles.handle.write(bio.getvalue())

    def _write_map(self) -> None:
        """No-op, future compatibility"""
        # 空操作，为了未来兼容性而保留

    def _write_file_close_tag(self) -> None:
        """No-op, future compatibility"""
        # 空操作，为了未来兼容性而保留

    def _write_characteristics(self) -> None:
        """No-op, future compatibility"""
        # 空操作，为了未来兼容性而保留

    def _write_strls(self) -> None:
        """No-op, future compatibility"""
        # 空操作，为了未来兼容性而保留

    def _write_expansion_fields(self) -> None:
        """Write 5 zeros for expansion fields"""
        # 写入5个零作为扩展字段
        self._write(_pad_bytes("", 5))

    def _write_value_labels(self) -> None:
        # 遍历_value_labels列表
        for vl in self._value_labels:
            # 将vl对象生成的值标签使用指定的字节顺序写入文件
            self._write_bytes(vl.generate_value_label(self._byteorder))

    def _write_header(
        self,
        data_label: str | None = None,
        time_stamp: datetime | None = None,
        version_major: int = 0,
        version_minor: int = 0,
        variable_display_width: int = 120,
    ) -> None:
        """
        Write the header information to the output file.

        Parameters:
        - data_label: Optional data label to include in the header.
        - time_stamp: Optional timestamp to include in the header.
        - version_major: Major version number.
        - version_minor: Minor version number.
        - variable_display_width: Width for displaying variables.

        Writes various header information to the file based on provided parameters.
        """
        # 获取当前时间戳
        timestamp = time_stamp or datetime.now()
        # 格式化时间戳为指定格式的字符串
        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        # 将数据标签、时间戳和版本号写入文件
        self._write_bytes(f"DATA LABEL: {data_label}\n".encode(self._encoding))
        self._write_bytes(f"TIME STAMP: {formatted_time}\n".encode(self._encoding))
        self._write_bytes(f"VERSION: {version_major}.{version_minor}\n".encode(self._encoding))
        self._write_bytes(f"VARIABLE DISPLAY WIDTH: {variable_display_width}\n".encode(self._encoding))
    ) -> None:
        # 获取当前对象的字节顺序
        byteorder = self._byteorder
        # 写入一个字节，格式为有符号字节，数值为114
        self._write_bytes(struct.pack("b", 114))
        # 根据字节顺序写入一个字节，条件运算符根据条件选择 "\x01" 或 "\x02"
        self._write(byteorder == ">" and "\x01" or "\x02")
        # 写入一个字节，数值为 1，表示文件类型
        self._write("\x01")
        # 写入一个字节，数值为 0，未使用
        self._write("\x00")
        # 写入两个字节，表示变量数量，根据字节顺序打包
        self._write_bytes(struct.pack(byteorder + "h", self.nvar)[:2])
        # 写入四个字节，表示观测数量，根据字节顺序打包
        self._write_bytes(struct.pack(byteorder + "i", self.nobs)[:4])
        # 写入数据标签，81个字节，字符数组，以空字符结尾
        if data_label is None:
            # 若无数据标签，写入空字符数组
            self._write_bytes(self._null_terminate_bytes(_pad_bytes("", 80)))
        else:
            # 若有数据标签，写入经过填充和空字符结尾处理的数据标签
            self._write_bytes(
                self._null_terminate_bytes(_pad_bytes(data_label[:80], 80))
            )
        # 时间戳，18个字节，字符数组，以空字符结尾
        # 格式为 dd Mon yyyy hh:mm
        if time_stamp is None:
            # 若时间戳为空，默认为当前时间
            time_stamp = datetime.now()
        elif not isinstance(time_stamp, datetime):
            # 若时间戳不是 datetime 类型，抛出数值错误
            raise ValueError("time_stamp should be datetime type")
        # 避免特定地区的月份转换，创建月份映射表
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        month_lookup = {i + 1: month for i, month in enumerate(months)}
        # 格式化时间戳为指定格式并写入
        ts = (
            time_stamp.strftime("%d ")
            + month_lookup[time_stamp.month]
            + time_stamp.strftime(" %Y %H:%M")
        )
        self._write_bytes(self._null_terminate_bytes(ts))

    def _write_variable_types(self) -> None:
        # 遍历变量类型列表，写入每个类型对应的字节表示
        for typ in self.typlist:
            self._write_bytes(struct.pack("B", typ))

    def _write_varnames(self) -> None:
        # 变量名列表由 _check_column_names 检查
        # 遍历变量名列表，写入每个变量名并以空字符结尾
        for name in self.varlist:
            name = self._null_terminate_str(name)
            name = _pad_bytes(name[:32], 33)
            self._write(name)

    def _write_sortlist(self) -> None:
        # 排序列表，长度为 2*(nvar+1)，整数数组，按照字节顺序编码
        srtlist = _pad_bytes("", 2 * (self.nvar + 1))
        self._write(srtlist)

    def _write_formats(self) -> None:
        # 格式列表，长度为 49*nvar，字符数组
        for fmt in self.fmtlist:
            self._write(_pad_bytes(fmt, 49))

    def _write_value_label_names(self) -> None:
        # 值标签名列表，长度为 33*nvar，字符数组
        for i in range(self.nvar):
            # 当变量具有值标签时使用变量名
            if self._has_value_labels[i]:
                name = self.varlist[i]
                name = self._null_terminate_str(name)
                name = _pad_bytes(name[:32], 33)
                self._write(name)
            else:  # 默认为空标签
                self._write(_pad_bytes("", 33))
    # 写入变量标签信息到数据文件
    def _write_variable_labels(self) -> None:
        # 缺失标签用80个空格字符加空终止符填充
        blank = _pad_bytes("", 81)

        # 如果没有变量标签信息，则填充空白数据
        if self._variable_labels is None:
            for i in range(self.nvar):
                self._write(blank)
            return

        # 遍历数据列，写入对应的变量标签信息或空白数据
        for col in self.data:
            if col in self._variable_labels:
                label = self._variable_labels[col]
                # 标签长度超过80个字符则抛出异常
                if len(label) > 80:
                    raise ValueError("Variable labels must be 80 characters or fewer")
                # 检查标签是否为Latin-1编码字符
                is_latin1 = all(ord(c) < 256 for c in label)
                if not is_latin1:
                    raise ValueError(
                        "Variable labels must contain only characters that "
                        "can be encoded in Latin-1"
                    )
                # 将标签信息写入数据文件，并用空字符填充到81个字节
                self._write(_pad_bytes(label, 81))
            else:
                # 如果列没有对应的标签信息，则写入空白数据
                self._write(blank)

    # 无操作函数，用于未来兼容性
    def _convert_strls(self, data: DataFrame) -> DataFrame:
        """No-op, future compatibility"""
        return data

    # 准备数据，返回NumPy记录数组
    def _prepare_data(self) -> np.rec.recarray:
        data = self.data
        typlist = self.typlist
        convert_dates = self._convert_dates

        # 1. 转换日期数据
        if self._convert_dates is not None:
            for i, col in enumerate(data):
                if i in convert_dates:
                    # 转换日期数据为Stata的经过的时间向量
                    data[col] = _datetime_to_stata_elapsed_vec(
                        data[col], self.fmtlist[i]
                    )

        # 2. 转换strls数据（无操作）
        data = self._convert_strls(data)

        # 3. 转换不良字符串数据为空字符并填充到正确的长度
        dtypes = {}
        native_byteorder = self._byteorder == _set_endianness(sys.byteorder)
        for i, col in enumerate(data):
            typ = typlist[i]
            if typ <= self._max_string_length:
                # 填充缺失数据并转换为对应的字节数组
                dc = data[col].fillna("")
                data[col] = dc.apply(_pad_bytes, args=(typ,))
                stype = f"S{typ}"
                dtypes[col] = stype
                data[col] = data[col].astype(stype)
            else:
                dtype = data[col].dtype
                if not native_byteorder:
                    # 如果字节顺序不是本机顺序，则调整字节顺序
                    dtype = dtype.newbyteorder(self._byteorder)
                dtypes[col] = dtype

        # 将数据转换为记录数组，不包含索引，列数据类型由dtypes指定
        return data.to_records(index=False, column_dtypes=dtypes)

    # 将记录数组的字节表示写入数据文件
    def _write_data(self, records: np.rec.recarray) -> None:
        self._write_bytes(records.tobytes())

    # 静态方法：给字符串添加空终止符
    @staticmethod
    def _null_terminate_str(s: str) -> str:
        s += "\x00"
        return s

    # 将字符串添加空终止符后转换为字节数组
    def _null_terminate_bytes(self, s: str) -> bytes:
        return self._null_terminate_str(s).encode(self._encoding)
    df: DataFrame,
    columns: Sequence[str],
    version: int = 117,
    byteorder: str | None = None,
    ) -> None:
        # 检查传入的版本号是否在支持的列表中，如果不在则抛出数值错误异常
        if version not in (117, 118, 119):
            raise ValueError("Only dta versions 117, 118 and 119 supported")
        # 将传入的版本号赋值给对象的属性 _dta_ver
        self._dta_ver = version

        # 将传入的 DataFrame 对象赋值给对象的属性 df
        self.df = df
        # 将传入的列名列表赋值给对象的属性 columns
        self.columns = columns
        # 初始化 _gso_table 属性为一个空字典，包含一个空字符串键和 (0, 0) 的元组值
        self._gso_table = {"": (0, 0)}
        # 如果 byteorder 为 None，则设为系统的字节序
        if byteorder is None:
            byteorder = sys.byteorder
        # 调用 _set_endianness 函数设置字节序，并将结果赋值给对象的属性 _byteorder
        self._byteorder = _set_endianness(byteorder)
        # 检查当前选择的字节序是否与系统的字节序匹配，并将结果赋值给对象的属性 _native_byteorder
        self._native_byteorder = self._byteorder == _set_endianness(sys.byteorder)

        # 初始化 gso_v_type 和 gso_o_type 属性，分别赋值为 "I" 和 "Q"，表示 uint32 和 uint64 类型
        gso_v_type = "I"  # uint32
        gso_o_type = "Q"  # uint64
        # 设置对象的编码方式为 UTF-8
        self._encoding = "utf-8"
        # 根据版本号进行条件判断
        if version == 117:
            o_size = 4
            gso_o_type = "I"  # 117 使用 uint32 类型
            self._encoding = "latin-1"  # 117 版本使用 Latin-1 编码
        elif version == 118:
            o_size = 6
        else:  # version == 119
            o_size = 5
        # 根据字节序是否与系统匹配，设定 _o_offet 属性的值
        if self._native_byteorder:
            self._o_offet = 2 ** (8 * (8 - o_size))
        else:
            self._o_offet = 2 ** (8 * o_size)
        # 将 gso_o_type 和 gso_v_type 分别赋值给对象的属性 _gso_o_type 和 _gso_v_type
        self._gso_o_type = gso_o_type
        self._gso_v_type = gso_v_type

    def _convert_key(self, key: tuple[int, int]) -> int:
        # 解构元组 key，得到 v 和 o 两个值
        v, o = key
        # 根据当前的字节序判断计算返回值
        if self._native_byteorder:
            return v + self._o_offet * o
        else:
            # 当字节序不匹配时，交换 v 和 o 的顺序再进行计算
            return o + self._o_offet * v
    def generate_table(self) -> tuple[dict[str, tuple[int, int]], DataFrame]:
        """
        生成用于 DataFrame 的 GSO 查找表

        Returns
        -------
        gso_table : dict
            使用字符串作为键，(v,o) 作为值的有序字典
        gso_df : DataFrame
            将 str 列转换为 (v,o) 值的 DataFrame

        Notes
        -----
        直接修改原始 DataFrame。

        返回的 DataFrame 将 (v,o) 值编码为 uint64。编码取决于数据版本，可以表达为：

        enc = v + o * 2 ** (o_size * 8)

        其中，v 存储在低位，o 存储在高位。o_size 如下：

          * 117: 4
          * 118: 6
          * 119: 5
        """
        gso_table = self._gso_table  # 获取现有的 GSO 表
        gso_df = self.df  # 获取当前的 DataFrame
        columns = list(gso_df.columns)  # 获取 DataFrame 的列名列表
        selected = gso_df[self.columns]  # 选择指定的列
        col_index = [(col, columns.index(col)) for col in self.columns]  # 列名及其索引的列表
        keys = np.empty(selected.shape, dtype=np.uint64)  # 创建一个空的 uint64 数组用于存放键值对应的编码
        for o, (idx, row) in enumerate(selected.iterrows()):  # 遍历选定的行
            for j, (col, v) in enumerate(col_index):  # 遍历列索引
                val = row[col]  # 获取单元格的值
                # 处理包含混合字符串和 None 值的列（GH 23633）
                val = "" if val is None else val
                key = gso_table.get(val, None)  # 获取现有键值对应的 (v,o) 元组
                if key is None:
                    # 如果不存在对应的键，创建一个新的键并添加到 GSO 表中
                    key = (v + 1, o + 1)  # Stata 喜欢人类数值，所以增加 1
                    gso_table[val] = key
                keys[o, j] = self._convert_key(key)  # 将 (v,o) 元组编码为 uint64 存入数组
        for i, col in enumerate(self.columns):
            gso_df[col] = keys[:, i]  # 将编码后的值存回 DataFrame 的相应列

        return gso_table, gso_df  # 返回更新后的 GSO 表和 DataFrame
    def generate_blob(self, gso_table: dict[str, tuple[int, int]]) -> bytes:
        """
        Generates the binary blob of GSOs that is written to the dta file.

        Parameters
        ----------
        gso_table : dict
            Ordered dictionary mapping strings to tuples of integers (v, o)

        Returns
        -------
        gso : bytes
            Binary content of dta file to be placed between strl tags

        Notes
        -----
        Output format depends on dta version.  117 uses two uint32s to
        express v and o while 118+ uses a uint32 for v and a uint64 for o.
        """
        # 创建一个字节流对象
        bio = BytesIO()
        
        # 将字符串 "GSO" 转换为 ASCII 码的字节序列
        gso = bytes("GSO", "ascii")
        
        # 使用指定的字节序列和版本号来打包成二进制数据
        gso_type = struct.pack(self._byteorder + "B", 130)
        
        # 打包一个空字节
        null = struct.pack(self._byteorder + "B", 0)
        
        # 使用字节序列和数据类型来打包版本号 v
        v_type = self._byteorder + self._gso_v_type
        
        # 使用字节序列和数据类型来打包偏移量 o
        o_type = self._byteorder + self._gso_o_type
        
        # 使用字节序列和数据类型来打包长度信息
        len_type = self._byteorder + "I"
        
        # 遍历 gso_table 中的每一个条目
        for strl, vo in gso_table.items():
            v, o = vo
            
            # 写入 "GSO"
            bio.write(gso)
            
            # 写入 vvvv
            bio.write(struct.pack(v_type, v))
            
            # 写入 oooo / oooooooo
            bio.write(struct.pack(o_type, o))
            
            # 写入类型信息
            bio.write(gso_type)
            
            # 计算 UTF-8 编码的字符串长度并写入长度信息
            utf8_string = bytes(strl, "utf-8")
            bio.write(struct.pack(len_type, len(utf8_string) + 1))
            
            # 写入 UTF-8 编码的字符串及其结束符
            bio.write(utf8_string)
            bio.write(null)

        # 返回生成的字节流内容
        return bio.getvalue()
# 定义一个用于写入 Stata 13 格式（117）二进制 dta 文件的类 StataWriter117，继承自 StataWriter 类
class StataWriter117(StataWriter):
    """
    A class for writing Stata binary dta files in Stata 13 format (117)

    Parameters
    ----------
    fname : path (string), buffer or path object
        string, pathlib.Path or
        object implementing a binary write() functions. If using a buffer
        then the buffer will not be automatically closed after the file
        is written.
        文件名：可以是路径（字符串）、缓冲区或路径对象；如果使用缓冲区，则文件写入后不会自动关闭。
    data : DataFrame
        Input to save
        要保存的输入数据(DataFrame)
    convert_dates : dict
        Dictionary mapping columns containing datetime types to stata internal
        format to use when writing the dates. Options are 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'. Column can be either an integer or a name.
        Datetime columns that do not have a conversion type specified will be
        converted to 'tc'. Raises NotImplementedError if a datetime column has
        timezone information
        将包含日期时间类型的列映射到写入日期时要使用的 Stata 内部格式的字典。选项包括 'tc', 'td', 'tm',
        'tw', 'th', 'tq', 'ty'。列可以是整数或名称。未指定转换类型的日期时间列将转换为 'tc'。
        如果日期时间列包含时区信息，则引发 NotImplementedError。
    write_index : bool
        Write the index to Stata dataset.
        是否将索引写入 Stata 数据集。
    byteorder : str
        Can be ">", "<", "little", or "big". default is `sys.byteorder`
        字节顺序：可以是 ">", "<", "little" 或 "big"。默认为 `sys.byteorder`。
    time_stamp : datetime
        A datetime to use as file creation date.  Default is the current time
        用作文件创建日期的日期时间。默认为当前时间。
    data_label : str
        A label for the data set.  Must be 80 characters or smaller.
        数据集的标签。必须是 80 个字符或更少。
    variable_labels : dict
        Dictionary containing columns as keys and variable labels as values.
        Each label must be 80 characters or smaller.
        包含列作为键和变量标签作为值的字典。每个标签必须是 80 个字符或更少。
    convert_strl : list
        List of columns names to convert to Stata StrL format.  Columns with
        more than 2045 characters are automatically written as StrL.
        Smaller columns can be converted by including the column name.  Using
        StrLs can reduce output file size when strings are longer than 8
        characters, and either frequently repeated or sparse.
        要转换为 Stata StrL 格式的列名列表。超过 2045 个字符的列会自动写入为 StrL。
        可以通过包含列名来转换较小的列。当字符串超过 8 个字符且频繁重复或稀疏时，使用 StrL 可以减小输出文件大小。
    {compression_options}
        Compression options. 详见版本变更说明 1.4.0。

    value_labels : dict of dicts
        Dictionary containing columns as keys and dictionaries of column value
        to labels as values. The combined length of all labels for a single
        variable must be 32,000 characters or smaller.
        包含列作为键和列值到标签的字典的字典。单个变量所有标签的组合长度必须小于等于 32,000 个字符。

        .. versionadded:: 1.4.0

    Returns
    -------
    writer : StataWriter117 instance
        The StataWriter117 instance has a write_file method, which will
        write the file to the given `fname`.
        返回 StataWriter117 实例。该实例有一个 write_file 方法，用于将文件写入给定的 `fname`。

    Raises
    ------
    NotImplementedError
        * If datetimes contain timezone information
        如果日期时间包含时区信息
    ValueError
        * Columns listed in convert_dates are neither datetime64[ns]
          or datetime
        * Column dtype is not representable in Stata
        * Column listed in convert_dates is not in DataFrame
        * Categorical label contains more than 32,000 characters
        如果列在 convert_dates 中列出的既不是 datetime64[ns] 也不是 datetime
        或者列 dtype 在 Stata 中不能表示
        或者列在 convert_dates 中列出但不在 DataFrame 中
        或者分类标签包含超过 32,000 个字符

    Examples
    --------
    >>> data = pd.DataFrame([[1.0, 1, "a"]], columns=["a", "b", "c"])
    >>> writer = pd.io.stata.StataWriter117("./data_file.dta", data)
    >>> writer.write_file()

    Directly write a zip file
    >>> compression = {"method": "zip", "archive_name": "data_file.dta"}
    """
    _max_string_length = 2045
    _dta_version = 117


# 定义最大字符串长度和 Stata 文件版本号
_max_string_length = 2045
_dta_version = 117



    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],
        data: DataFrame,
        convert_dates: dict[Hashable, str] | None = None,
        write_index: bool = True,
        byteorder: str | None = None,
        time_stamp: datetime | None = None,
        data_label: str | None = None,
        variable_labels: dict[Hashable, str] | None = None,
        convert_strl: Sequence[Hashable] | None = None,
        compression: CompressionOptions = "infer",
        storage_options: StorageOptions | None = None,
        *,
        value_labels: dict[Hashable, dict[float, str]] | None = None,
    ) -> None:


# 构造函数初始化方法，用于创建 StataWriter117 对象
def __init__(
    self,
    fname: FilePath | WriteBuffer[bytes],  # 文件路径或写入缓冲区
    data: DataFrame,  # 数据框架
    convert_dates: dict[Hashable, str] | None = None,  # 日期转换选项
    write_index: bool = True,  # 是否写入索引
    byteorder: str | None = None,  # 字节顺序
    time_stamp: datetime | None = None,  # 时间戳
    data_label: str | None = None,  # 数据标签
    variable_labels: dict[Hashable, str] | None = None,  # 变量标签
    convert_strl: Sequence[Hashable] | None = None,  # 长字符串转换选项
    compression: CompressionOptions = "infer",  # 压缩选项
    storage_options: StorageOptions | None = None,  # 存储选项
    *,
    value_labels: dict[Hashable, dict[float, str]] | None = None,  # 值标签
) -> None:



        # Copy to new list since convert_strl might be modified later
        self._convert_strl: list[Hashable] = []
        if convert_strl is not None:
            self._convert_strl.extend(convert_strl)


# 复制 convert_strl 列表，因为后续可能会修改它
self._convert_strl: list[Hashable] = []
if convert_strl is not None:
    self._convert_strl.extend(convert_strl)



        super().__init__(
            fname,
            data,
            convert_dates,
            write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            value_labels=value_labels,
            compression=compression,
            storage_options=storage_options,
        )
        self._map: dict[str, int] = {}
        self._strl_blob = b""


# 调用父类初始化方法，传递各种参数，然后初始化对象的两个属性
super().__init__(
    fname,
    data,
    convert_dates,
    write_index,
    byteorder=byteorder,
    time_stamp=time_stamp,
    data_label=data_label,
    variable_labels=variable_labels,
    value_labels=value_labels,
    compression=compression,
    storage_options=storage_options,
)
self._map: dict[str, int] = {}  # 初始化映射字典
self._strl_blob = b""  # 初始化长字符串块



    @staticmethod
    def _tag(val: str | bytes, tag: str) -> bytes:
        """Surround val with <tag></tag>"""
        if isinstance(val, str):
            val = bytes(val, "utf-8")
        return bytes("<" + tag + ">", "utf-8") + val + bytes("</" + tag + ">", "utf-8")


# 静态方法，用于将 val 用指定标签 tag 包围起来
@staticmethod
def _tag(val: str | bytes, tag: str) -> bytes:
    """Surround val with <tag></tag>"""
    if isinstance(val, str):
        val = bytes(val, "utf-8")
    return bytes("<" + tag + ">", "utf-8") + val + bytes("</" + tag + ">", "utf-8")



    def _update_map(self, tag: str) -> None:
        """Update map location for tag with file position"""
        assert self.handles.handle is not None
        self._map[tag] = self.handles.handle.tell()


# 更新映射位置的方法，关联标签 tag 和文件位置
def _update_map(self, tag: str) -> None:
    """Update map location for tag with file position"""
    assert self.handles.handle is not None
    self._map[tag] = self.handles.handle.tell()



    def _write_header(
        self,
        data_label: str | None = None,
        time_stamp: datetime | None = None,


# 写入文件头部信息的方法
def _write_header(
    self,
    data_label: str | None = None,  # 数据标签（可选）
    time_stamp: datetime | None = None,  # 时间戳（可选）
    ) -> None:
        """Write the file header"""
        # 获取字节顺序（大端或小端）
        byteorder = self._byteorder
        # 写入文件头部标识 "<stata_dta>"
        self._write_bytes(bytes("<stata_dta>", "utf-8"))
        # 创建一个字节流对象
        bio = BytesIO()
        # 写入数据格式标签，用于表示 Stata 文件版本
        bio.write(self._tag(bytes(str(self._dta_version), "utf-8"), "release"))
        # 写入字节顺序标签，根据实际情况选择 MSF 或 LSF
        bio.write(self._tag(byteorder == ">" and "MSF" or "LSF", "byteorder"))
        # 写入变量数目标签，根据 Stata 版本选择不同的数据类型
        nvar_type = "H" if self._dta_version <= 118 else "I"
        bio.write(self._tag(struct.pack(byteorder + nvar_type, self.nvar), "K"))
        # 写入观测数目标签，根据 Stata 版本选择不同的数据类型
        nobs_size = "I" if self._dta_version == 117 else "Q"
        bio.write(self._tag(struct.pack(byteorder + nobs_size, self.nobs), "N"))
        # 写入数据标签，最多 81 个字节的字符，以空字符结尾
        # 如果没有数据标签，则写入空字符串
        label = data_label[:80] if data_label is not None else ""
        encoded_label = label.encode(self._encoding)
        label_size = "B" if self._dta_version == 117 else "H"
        label_len = struct.pack(byteorder + label_size, len(encoded_label))
        encoded_label = label_len + encoded_label
        bio.write(self._tag(encoded_label, "label"))
        # 写入时间戳标签，包括日期和时间，格式为 "dd Mon yyyy hh:mm"
        if time_stamp is None:
            time_stamp = datetime.now()
        elif not isinstance(time_stamp, datetime):
            raise ValueError("time_stamp should be datetime type")
        # 避免受本地化月份转换影响，使用固定的英文月份缩写
        months = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        month_lookup = {i + 1: month for i, month in enumerate(months)}
        ts = (
            time_stamp.strftime("%d ")
            + month_lookup[time_stamp.month]
            + time_stamp.strftime(" %Y %H:%M")
        )
        # 为了符合 Stata 文件格式，添加 '\x11' 字节
        stata_ts = b"\x11" + bytes(ts, "utf-8")
        bio.write(self._tag(stata_ts, "timestamp"))
        # 将当前字节流的值写入文件头部标签
        self._write_bytes(self._tag(bio.getvalue(), "header"))
    def _write_map(self) -> None:
        """
        Called twice during file write. The first populates the values in
        the map with 0s.  The second call writes the final map locations when
        all blocks have been written.
        """
        # 如果地图尚未初始化，将其初始化为包含各项值为0的字典
        if not self._map:
            self._map = {
                "stata_data": 0,
                "map": self.handles.handle.tell(),
                "variable_types": 0,
                "varnames": 0,
                "sortlist": 0,
                "formats": 0,
                "value_label_names": 0,
                "variable_labels": 0,
                "characteristics": 0,
                "data": 0,
                "strls": 0,
                "value_labels": 0,
                "stata_data_close": 0,
                "end-of-file": 0,
            }
        # 定位到地图的起始位置
        self.handles.handle.seek(self._map["map"])
        bio = BytesIO()
        # 将地图中的每个值按照特定格式打包后写入字节流
        for val in self._map.values():
            bio.write(struct.pack(self._byteorder + "Q", val))
        # 将打包好的地图数据写入文件
        self._write_bytes(self._tag(bio.getvalue(), "map"))

    def _write_variable_types(self) -> None:
        # 更新地图中的"variable_types"项
        self._update_map("variable_types")
        bio = BytesIO()
        # 将每个变量类型打包后写入字节流
        for typ in self.typlist:
            bio.write(struct.pack(self._byteorder + "H", typ))
        # 将打包好的变量类型数据写入文件
        self._write_bytes(self._tag(bio.getvalue(), "variable_types"))

    def _write_varnames(self) -> None:
        # 更新地图中的"varnames"项
        self._update_map("varnames")
        bio = BytesIO()
        # 根据数据集版本选择合适的变量名长度
        vn_len = 32 if self._dta_version == 117 else 128
        # 将每个变量名编码后写入字节流
        for name in self.varlist:
            name = self._null_terminate_str(name)
            name = _pad_bytes_new(name[:32].encode(self._encoding), vn_len + 1)
            bio.write(name)
        # 将打包好的变量名数据写入文件
        self._write_bytes(self._tag(bio.getvalue(), "varnames"))

    def _write_sortlist(self) -> None:
        # 更新地图中的"sortlist"项
        self._update_map("sortlist")
        # 根据数据集版本选择排序列表的大小
        sort_size = 2 if self._dta_version < 119 else 4
        # 将排序列表数据写入文件
        self._write_bytes(self._tag(b"\x00" * sort_size * (self.nvar + 1), "sortlist"))

    def _write_formats(self) -> None:
        # 更新地图中的"formats"项
        self._update_map("formats")
        bio = BytesIO()
        # 根据数据集版本选择合适的格式长度
        fmt_len = 49 if self._dta_version == 117 else 57
        # 将每种格式编码后写入字节流
        for fmt in self.fmtlist:
            bio.write(_pad_bytes_new(fmt.encode(self._encoding), fmt_len))
        # 将打包好的格式数据写入文件
        self._write_bytes(self._tag(bio.getvalue(), "formats"))
    def _write_value_label_names(self) -> None:
        # 更新映射表中的"value_label_names"部分
        self._update_map("value_label_names")
        # 创建一个字节流对象
        bio = BytesIO()
        # 根据数据集版本确定value label的长度，用于最坏情况下的UTF-8数据编码
        vl_len = 32 if self._dta_version == 117 else 128
        # 遍历每个变量
        for i in range(self.nvar):
            # 如果当前变量有值标签，则使用变量名作为名称
            name = ""  # 默认名称为空字符串
            if self._has_value_labels[i]:
                name = self.varlist[i]
            # 对名称进行空字符终止处理
            name = self._null_terminate_str(name)
            # 编码名称并填充字节，保证长度符合vl_len + 1
            encoded_name = _pad_bytes_new(name[:32].encode(self._encoding), vl_len + 1)
            # 将编码后的名称写入字节流
            bio.write(encoded_name)
        # 将处理过的字节流打上标签并写入存储
        self._write_bytes(self._tag(bio.getvalue(), "value_label_names"))

    def _write_variable_labels(self) -> None:
        # 更新映射表中的"variable_labels"部分
        self._update_map("variable_labels")
        # 创建一个字节流对象
        bio = BytesIO()
        # 根据数据集版本确定variable label的长度，用于最坏情况下的UTF-8数据编码
        vl_len = 80 if self._dta_version == 117 else 320
        # 创建一个空白填充字节序列
        blank = _pad_bytes_new("", vl_len + 1)

        # 如果没有变量标签
        if self._variable_labels is None:
            # 对于每个变量，写入空白填充
            for _ in range(self.nvar):
                bio.write(blank)
            # 将填充后的字节流打上标签并写入存储
            self._write_bytes(self._tag(bio.getvalue(), "variable_labels"))
            return

        # 遍历数据集中的列
        for col in self.data:
            # 如果列有对应的变量标签
            if col in self._variable_labels:
                # 获取变量标签的内容
                label = self._variable_labels[col]
                # 如果标签内容超过80个字符，引发异常
                if len(label) > 80:
                    raise ValueError("Variable labels must be 80 characters or fewer")
                try:
                    # 尝试将标签内容编码成指定编码的字节流
                    encoded = label.encode(self._encoding)
                except UnicodeEncodeError as err:
                    # 如果编码失败，引发异常
                    raise ValueError(
                        "Variable labels must contain only characters that "
                        f"can be encoded in {self._encoding}"
                    ) from err

                # 填充编码后的标签内容，确保长度符合vl_len + 1
                bio.write(_pad_bytes_new(encoded, vl_len + 1))
            else:
                # 如果列没有对应的变量标签，写入空白填充
                bio.write(blank)
        # 将处理过的字节流打上标签并写入存储
        self._write_bytes(self._tag(bio.getvalue(), "variable_labels"))

    def _write_characteristics(self) -> None:
        # 更新映射表中的"characteristics"部分
        self._update_map("characteristics")
        # 直接写入空字节并打上标签，用于特性信息
        self._write_bytes(self._tag(b"", "characteristics"))

    def _write_data(self, records: np.rec.recarray) -> None:
        # 更新映射表中的"data"部分
        self._update_map("data")
        # 打开数据标签
        self._write_bytes(b"<data>")
        # 将记录数组转换为字节并写入存储
        self._write_bytes(records.tobytes())
        # 关闭数据标签
        self._write_bytes(b"</data>")

    def _write_strls(self) -> None:
        # 更新映射表中的"strls"部分
        self._update_map("strls")
        # 将strl_blob打上标签并写入存储
        self._write_bytes(self._tag(self._strl_blob, "strls"))

    def _write_expansion_fields(self) -> None:
        """在dta 117+中不执行任何操作"""

    def _write_value_labels(self) -> None:
        # 更新映射表中的"value_labels"部分
        self._update_map("value_labels")
        # 创建一个字节流对象
        bio = BytesIO()
        # 遍历所有的value label对象
        for vl in self._value_labels:
            # 生成value label数据，并打上标签
            lab = vl.generate_value_label(self._byteorder)
            lab = self._tag(lab, "lbl")
            # 将处理过的标签数据写入字节流
            bio.write(lab)
        # 将处理过的字节流打上标签并写入存储
        self._write_bytes(self._tag(bio.getvalue(), "value_labels"))
    # 更新内部映射以标记 Stata 数据关闭的操作
    def _write_file_close_tag(self) -> None:
        self._update_map("stata_data_close")
        # 写入 Stata 数据文件的关闭标签
        self._write_bytes(bytes("</stata_dta>", "utf-8"))
        # 更新内部映射以表示文件结束
        self._update_map("end-of-file")

    # 更新可能已更改以符合 Stata 命名规则的列名，以便将其转换为 StrL 类型
    def _update_strl_names(self) -> None:
        """
        Update column names for conversion to strl if they might have been
        changed to comply with Stata naming rules
        """
        # 如果列名已更改，则更新 convert_strl 列表
        for orig, new in self._converted_names.items():
            if orig in self._convert_strl:
                idx = self._convert_strl.index(orig)
                self._convert_strl[idx] = new

    # 将指定的数据框中的列转换为 StrL 类型，如果列非常大或在 convert_strl 变量中
    def _convert_strls(self, data: DataFrame) -> DataFrame:
        """
        Convert columns to StrLs if either very large or in the
        convert_strl variable
        """
        # 筛选需要转换为 StrL 的列
        convert_cols = [
            col
            for i, col in enumerate(data)
            if self.typlist[i] == 32768 or col in self._convert_strl
        ]

        # 如果有需要转换的列，则创建 StataStrLWriter 对象进行处理
        if convert_cols:
            ssw = StataStrLWriter(
                data, convert_cols, version=self._dta_version, byteorder=self._byteorder
            )
            # 生成 StrL 格式表和转换后的新数据
            tab, new_data = ssw.generate_table()
            data = new_data
            # 生成 StrL 格式的 blob 数据
            self._strl_blob = ssw.generate_blob(tab)
        return data

    # 设置 Stata 数据文件中各列的格式和类型信息
    def _set_formats_and_types(self, dtypes: Series) -> None:
        self.typlist = []
        self.fmtlist = []
        # 遍历数据类型 Series，并为每列设置格式和类型
        for col, dtype in dtypes.items():
            force_strl = col in self._convert_strl
            # 根据数据类型获取默认 Stata 格式，并更新相应的类型列表
            fmt = _dtype_to_default_stata_fmt(
                dtype,
                self.data[col],
                dta_version=self._dta_version,
                force_strl=force_strl,
            )
            self.fmtlist.append(fmt)
            # 更新类型列表，根据数据类型和是否强制为 StrL 类型来确定
            self.typlist.append(
                _dtype_to_stata_type_117(dtype, self.data[col], force_strl)
            )
class StataWriterUTF8(StataWriter117):
    """
    StataWriterUTF8类继承自StataWriter117类，用于写入Stata 15 (118)和Stata 16 (119)格式的二进制dta文件。

    DTA 118和119格式支持Unicode字符串数据（固定和strL格式）。值标签、变量标签和数据集标签也支持Unicode。如果文件包含超过32,767个变量，则自动使用格式119。

    Parameters
    ----------
    fname : path (string), buffer or path object
        文件路径（字符串）、缓冲区或路径对象，用于写入数据。如果使用缓冲区，则文件写入后不会自动关闭。
    data : DataFrame
        要保存的输入数据。
    convert_dates : dict, default None
        将包含日期时间类型的列映射到Stata内部格式，用于写入日期。选项有'tc', 'td', 'tm', 'tw', 'th', 'tq', 'ty'。列可以是整数或名称。
        没有指定转换类型的日期时间列将被转换为'tc'。如果日期时间列包含时区信息，则会引发NotImplementedError。
    write_index : bool, default True
        是否写入索引到Stata数据集。
    byteorder : str, default None
        字节顺序，可以是">", "<", "little"或"big"。默认为`sys.byteorder`。
    time_stamp : datetime, default None
        文件创建日期时间。默认为当前时间。
    data_label : str, default None
        数据集标签，必须小于或等于80个字符。
    variable_labels : dict, default None
        包含列名和变量标签的字典。每个标签必须小于或等于80个字符。
    convert_strl : list, default None
        要转换为Stata StrL格式的列名列表。超过2045个字符的列会自动写入为StrL格式。较小的列可以通过包括列名来转换。
        当字符串超过8个字符且频繁重复或稀疏时，使用StrL可以减小输出文件大小。
    version : int, default None
        要使用的dta版本。默认情况下，根据数据大小确定版本。如果data.shape[1] <= 32767，则使用118版本；否则使用119版本。
    {compression_options}
        .. versionchanged:: 1.4.0 支持Zstandard压缩。

    value_labels : dict of dicts
        包含列名和列值到标签的字典。单个变量所有标签的总长度必须小于或等于32,000个字符。

        .. versionadded:: 1.4.0

    Returns
    -------
    StataWriterUTF8
        实例具有write_file方法，用于将文件写入给定的`fname`。

    Raises
    ------
    NotImplementedError
        * 如果日期时间包含时区信息
    """
    pass
    # 设置_encoding变量为"utf-8"，表示编码格式为UTF-8
    _encoding: Literal["utf-8"] = "utf-8"
    
    # 定义StataWriterUTF8类的初始化方法，用于创建Stata数据文件的写入器
    def __init__(
        self,
        fname: FilePath | WriteBuffer[bytes],  # 接受文件路径或写入缓冲区作为参数fname
        data: DataFrame,  # 接受一个DataFrame对象作为数据参数data
        convert_dates: dict[Hashable, str] | None = None,  # 可选参数，用于指定日期转换规则的字典
        write_index: bool = True,  # 控制是否写入DataFrame索引，默认为True
        byteorder: str | None = None,  # 字节顺序参数，用于控制字节顺序的设置
        time_stamp: datetime | None = None,  # 时间戳参数，指定Stata数据文件的时间戳
        data_label: str | None = None,  # 数据标签参数，用于设置Stata数据文件的数据标签
        variable_labels: dict[Hashable, str] | None = None,  # 变量标签字典参数，指定每个变量的标签
        convert_strl: Sequence[Hashable] | None = None,  # 可选参数，指定要转换为strl格式的变量列表
        version: int | None = None,  # Stata数据文件版本号，默认根据DataFrame列数自动选择版本
        compression: CompressionOptions = "infer",  # 压缩选项参数，控制是否压缩以及压缩方法
        storage_options: StorageOptions | None = None,  # 存储选项参数，用于传递到底层存储的配置
        *,
        value_labels: dict[Hashable, dict[float, str]] | None = None,  # 可选参数，值标签字典参数
    ) -> None:
        # 如果版本号未指定，根据DataFrame列数自动选择版本118或119
        if version is None:
            version = 118 if data.shape[1] <= 32767 else 119
        # 如果指定版本不在118或119中，抛出数值错误异常
        elif version not in (118, 119):
            raise ValueError("version must be either 118 or 119.")
        # 如果版本为118且DataFrame列数超过32767，抛出数值错误异常
        elif version == 118 and data.shape[1] > 32767:
            raise ValueError(
                "You must use version 119 for data sets containing more than"
                "32,767 variables"
            )
    
        # 调用父类的初始化方法，初始化StataWriterUTF8对象
        super().__init__(
            fname,
            data,
            convert_dates=convert_dates,
            write_index=write_index,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            value_labels=value_labels,
            convert_strl=convert_strl,
            compression=compression,
            storage_options=storage_options,
        )
        # 覆盖在StataWriter117 init中设置的版本号
        self._dta_version = version
    # 定义一个方法，用于验证变量名是否符合 Stata 导出的要求
    def _validate_variable_name(self, name: str) -> str:
        """
        Validate variable names for Stata export.

        Parameters
        ----------
        name : str
            Variable name

        Returns
        -------
        str
            The validated name with invalid characters replaced with
            underscores.

        Notes
        -----
        Stata 118+ support most unicode characters. The only limitation is in
        the ascii range where the characters supported are a-z, A-Z, 0-9 and _.
        """
        # 遍历变量名中的每个字符
        for c in name:
            # 检查字符是否在指定的 ASCII 范围内，或者是特定的几个字符，如果不是，则替换为下划线
            if (
                (
                    ord(c) < 128
                    and (c < "A" or c > "Z")
                    and (c < "a" or c > "z")
                    and (c < "0" or c > "9")
                    and c != "_"
                )
                or 128 <= ord(c) < 192
                or c in {"×", "÷"}  # noqa: RUF001
            ):
                name = name.replace(c, "_")

        # 返回经过验证处理后的变量名
        return name
```
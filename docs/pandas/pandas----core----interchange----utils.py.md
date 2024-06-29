# `D:\src\scipysrc\pandas\pandas\core\interchange\utils.py`

```
# 引入未来版本的类型标注支持
from __future__ import annotations

# 引入类型相关的模块
import typing

# 引入 NumPy 库并使用别名 np
import numpy as np

# 从 pandas._libs 中引入 lib 模块
from pandas._libs import lib

# 从 pandas.core.dtypes.dtypes 中引入具体的数据类型
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
)

# 引入 pandas 库并使用别名 pd
import pandas as pd

# 如果在类型检查环境下，从 pandas._typing 中引入 DtypeObj 类型
if typing.TYPE_CHECKING:
    from pandas._typing import DtypeObj

# 定义一个字典，将 pyarrow.DataType 映射到 C 类型格式字符串
PYARROW_CTYPES = {
    "null": "n",
    "bool": "b",
    "uint8": "C",
    "uint16": "S",
    "uint32": "I",
    "uint64": "L",
    "int8": "c",
    "int16": "S",
    "int32": "i",
    "int64": "l",
    "halffloat": "e",  # 对应 float16
    "float": "f",      # 对应 float32
    "double": "g",     # 对应 float64
    "string": "u",
    "large_string": "U",
    "binary": "z",
    "time32[s]": "tts",
    "time32[ms]": "ttm",
    "time64[us]": "ttu",
    "time64[ns]": "ttn",
    "date32[day]": "tdD",
    "date64[ms]": "tdm",
    "timestamp[s]": "tss:",
    "timestamp[ms]": "tsm:",
    "timestamp[us]": "tsu:",
    "timestamp[ns]": "tsn:",
    "duration[s]": "tDs",
    "duration[ms]": "tDm",
    "duration[us]": "tDu",
    "duration[ns]": "tDn",
}


class ArrowCTypes:
    """
    枚举类，定义了 Apache Arrow 的 C 类型格式字符串。
    参考 Apache Arrow C 数据接口文档：
    https://arrow.apache.org/docs/format/CDataInterface.html#data-type-description-format-strings
    """

    # 定义不同数据类型的格式字符串常量
    NULL = "n"
    BOOL = "b"
    INT8 = "c"
    UINT8 = "C"
    INT16 = "s"
    UINT16 = "S"
    INT32 = "i"
    UINT32 = "I"
    INT64 = "l"
    UINT64 = "L"
    FLOAT16 = "e"
    FLOAT32 = "f"
    FLOAT64 = "g"
    STRING = "u"  # utf-8
    LARGE_STRING = "U"  # utf-8
    DATE32 = "tdD"
    DATE64 = "tdm"
    TIMESTAMP = "ts{resolution}:{tz}"
    TIME = "tt{resolution}"


class Endianness:
    """
    枚举类，表示数据类型的字节顺序。
    """

    LITTLE = "<"
    BIG = ">"
    NATIVE = "="
    NA = "|"


def dtype_to_arrow_c_fmt(dtype: DtypeObj) -> str:
    """
    将 pandas 的 dtype 转换为 Apache Arrow 的 C 格式字符串表示。

    Parameters
    ----------
    dtype : np.dtype
        要表示的 pandas DataFrame 的数据类型。

    Returns
    -------
    str
        给定 dtype 的 Apache Arrow C 格式字符串表示。
    """
    # 如果是 CategoricalDtype 类型，则返回 INT64
    if isinstance(dtype, CategoricalDtype):
        return ArrowCTypes.INT64
    # 如果是对象类型（np.dtype("O")），则返回 STRING
    elif dtype == np.dtype("O"):
        return ArrowCTypes.STRING
    # 如果数据类型是 ArrowDtype 类型
    elif isinstance(dtype, ArrowDtype):
        # 导入 pyarrow 库
        import pyarrow as pa

        # 获取 pyarrow 类型
        pa_type = dtype.pyarrow_dtype
        
        # 如果是十进制类型
        if pa.types.is_decimal(pa_type):
            # 返回十进制类型的格式字符串，如 "d:precision,scale"
            return f"d:{pa_type.precision},{pa_type.scale}"
        
        # 如果是带时区的时间戳类型
        elif pa.types.is_timestamp(pa_type) and pa_type.tz is not None:
            # 返回带时区的时间戳类型的格式字符串，如 "ts<unit>:tz"
            return f"ts{pa_type.unit[0]}:{pa_type.tz}"
        
        # 根据 pa_type 在 PYARROW_CTYPES 中查找对应的格式字符串
        format_str = PYARROW_CTYPES.get(str(pa_type), None)
        
        # 如果找到了对应的格式字符串则返回
        if format_str is not None:
            return format_str
    
    # 根据 dtype 的名称在 ArrowCTypes 中获取格式字符串
    format_str = getattr(ArrowCTypes, dtype.name.upper(), None)
    
    # 如果找到了对应的格式字符串则返回
    if format_str is not None:
        return format_str
    
    # 如果是 numpy 数据类型并且是日期时间类型 'M'
    if lib.is_np_dtype(dtype, "M"):
        # 获取日期时间类型的分辨率字符串的第一个字符
        # dtype.str -> '<M8[ns]' -> 'n'
        resolution = np.datetime_data(dtype)[0][0]
        
        # 返回时间戳类型的格式字符串，使用 ArrowCTypes.TIMESTAMP 格式化
        return ArrowCTypes.TIMESTAMP.format(resolution=resolution, tz="")
    
    # 如果数据类型是 DatetimeTZDtype 类型
    elif isinstance(dtype, DatetimeTZDtype):
        # 返回带时区的时间戳类型的格式字符串，使用 dtype.unit 和 dtype.tz 格式化
        return ArrowCTypes.TIMESTAMP.format(resolution=dtype.unit[0], tz=dtype.tz)
    
    # 如果数据类型是 pd.BooleanDtype 类型
    elif isinstance(dtype, pd.BooleanDtype):
        # 返回布尔类型的格式字符串
        return ArrowCTypes.BOOL
    
    # 抛出未实现的错误，提示无法将该数据类型转换为 Arrow C 格式字符串
    raise NotImplementedError(
        f"Conversion of {dtype} to Arrow C format string is not implemented."
    )
# 尝试将多块的 pyarrow 数组重新组合为单块数组，如果必要的话
# 
# - 如果输入的 Series 不是由 pyarrow 数组支持，则返回 None（无需重新组块）
# - 如果输入由多块 pyarrow 数组支持且允许复制（allow_copy=True），则返回由单块支持的 Series
# - 如果输入由多块 pyarrow 数组支持且不允许复制（allow_copy=False），则引发 RuntimeError
def maybe_rechunk(series: pd.Series, *, allow_copy: bool) -> pd.Series | None:
    if not isinstance(series.dtype, pd.ArrowDtype):
        return None
    # 获取 pyarrow 数组对象
    chunked_array = series.array._pa_array  # type: ignore[attr-defined]
    # 如果数组只有一块，则无需重新组块，返回 None
    if len(chunked_array.chunks) == 1:
        return None
    # 如果不允许复制，则引发 RuntimeError
    if not allow_copy:
        raise RuntimeError(
            "Found multi-chunk pyarrow array, but `allow_copy` is False. "
            "Please rechunk the array before calling this function, or set "
            "`allow_copy=True`."
        )
    # 合并多块数组为单块数组
    arr = chunked_array.combine_chunks()
    # 返回新的 Series，使用单块数组数据，保持原有的 dtype、name 和 index
    return pd.Series(arr, dtype=series.dtype, name=series.name, index=series.index)
```
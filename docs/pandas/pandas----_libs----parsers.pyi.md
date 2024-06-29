# `D:\src\scipysrc\pandas\pandas\_libs\parsers.pyi`

```
from typing import (
    Hashable,
    Literal,
)

import numpy as np

from pandas._typing import (
    ArrayLike,
    Dtype,
    npt,
)

# 默认的字符串型缺失值集合
STR_NA_VALUES: set[str]
# 默认的缓冲区启发式值
DEFAULT_BUFFER_HEURISTIC: int

# 对象类型数组的清理函数，返回清理后的值的数量
def sanitize_objects(
    values: npt.NDArray[np.object_],
    na_values: set,
) -> int: ...

# 文本读取器类
class TextReader:
    # 未命名列集合
    unnamed_cols: set[str]
    # 表格宽度（int64_t）
    table_width: int
    # 主导列数（int64_t）
    leading_cols: int
    # 表头列表，包含非负整数
    header: list[list[int]]

    # 初始化函数
    def __init__(
        self,
        source,
        delimiter: bytes | str = ...,  # 单字符分隔符
        header=...,
        header_start: int = ...,  # int64_t
        header_end: int = ...,  # uint64_t
        index_col=...,
        names=...,
        tokenize_chunksize: int = ...,  # int64_t
        delim_whitespace: bool = ...,
        converters=...,
        skipinitialspace: bool = ...,
        escapechar: bytes | str | None = ...,  # 单字符转义符
        doublequote: bool = ...,
        quotechar: str | bytes | None = ...,  # 最多1个字符
        quoting: int = ...,
        lineterminator: bytes | str | None = ...,  # 最多1个字符
        comment=...,
        decimal: bytes | str = ...,  # 单字符小数点符号
        thousands: bytes | str | None = ...,  # 单字符千位分隔符
        dtype: Dtype | dict[Hashable, Dtype] = ...,
        usecols=...,
        error_bad_lines: bool = ...,
        warn_bad_lines: bool = ...,
        na_filter: bool = ...,
        na_values=...,
        na_fvalues=...,
        keep_default_na: bool = ...,
        true_values=...,
        false_values=...,
        allow_leading_cols: bool = ...,
        skiprows=...,
        skipfooter: int = ...,  # int64_t
        verbose: bool = ...,
        float_precision: Literal["round_trip", "legacy", "high"] | None = ...,
        skip_blank_lines: bool = ...,
        encoding_errors: bytes | str = ...,
    ) -> None: ...

    # 设置不转换的列
    def set_noconvert(self, i: int) -> None: ...

    # 移除不转换的列
    def remove_noconvert(self, i: int) -> None: ...

    # 关闭文本读取器
    def close(self) -> None: ...

    # 读取指定行数的数据，返回以列索引为键的数组字典
    def read(self, rows: int | None = ...) -> dict[int, ArrayLike]: ...

    # 低内存读取模式，读取指定行数的数据，返回数组字典列表
    def read_low_memory(self, rows: int | None) -> list[dict[int, ArrayLike]]: ...

# _maybe_upcast 函数，根据需要向上转换数组类型
def _maybe_upcast(
    arr, use_dtype_backend: bool = ..., dtype_backend: str = ...
) -> np.ndarray: ...

# 用于测试的 na_values 字典
na_values: dict
```
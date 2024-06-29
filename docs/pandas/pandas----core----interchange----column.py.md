# `D:\src\scipysrc\pandas\pandas\core\interchange\column.py`

```
from __future__ import annotations
# 导入以支持类型注解中的 forward references

from typing import (
    TYPE_CHECKING,
    Any,
)
# 导入类型检查和其他类型相关的模块

import numpy as np
# 导入 NumPy 库

from pandas._libs.lib import infer_dtype
# 从 pandas 私有库中导入 infer_dtype 函数

from pandas._libs.tslibs import iNaT
# 从 pandas 时间序列库中导入 iNaT 常量

from pandas.errors import NoBufferPresent
# 导入 pandas 错误模块中的 NoBufferPresent 异常

from pandas.util._decorators import cache_readonly
# 从 pandas 工具模块中导入 cache_readonly 装饰器

from pandas.core.dtypes.dtypes import BaseMaskedDtype
# 从 pandas 核心数据类型模块中导入 BaseMaskedDtype 类

import pandas as pd
# 导入 pandas 库，并用 pd 别名表示

from pandas import (
    ArrowDtype,
    DatetimeTZDtype,
)
# 从 pandas 中导入 ArrowDtype 和 DatetimeTZDtype 类型

from pandas.api.types import is_string_dtype
# 从 pandas API 类型模块中导入 is_string_dtype 函数

from pandas.core.interchange.buffer import (
    PandasBuffer,
    PandasBufferPyarrow,
)
# 从 pandas 核心交换缓冲区模块中导入 PandasBuffer 和 PandasBufferPyarrow 类

from pandas.core.interchange.dataframe_protocol import (
    Column,
    ColumnBuffers,
    ColumnNullType,
    DtypeKind,
)
# 从 pandas 核心交换数据帧协议模块中导入 Column, ColumnBuffers, ColumnNullType, DtypeKind 等

from pandas.core.interchange.utils import (
    ArrowCTypes,
    Endianness,
    dtype_to_arrow_c_fmt,
)
# 从 pandas 核心交换工具模块中导入 ArrowCTypes, Endianness, dtype_to_arrow_c_fmt 等

if TYPE_CHECKING:
    from pandas.core.interchange.dataframe_protocol import Buffer
# 如果是类型检查阶段，则导入 Buffer 类型

_NP_KINDS = {
    "i": DtypeKind.INT,
    "u": DtypeKind.UINT,
    "f": DtypeKind.FLOAT,
    "b": DtypeKind.BOOL,
    "U": DtypeKind.STRING,
    "M": DtypeKind.DATETIME,
    "m": DtypeKind.DATETIME,
}
# 定义 NumPy 数据类型和 pandas DtypeKind 的映射关系字典

_NULL_DESCRIPTION = {
    DtypeKind.FLOAT: (ColumnNullType.USE_NAN, None),
    DtypeKind.DATETIME: (ColumnNullType.USE_SENTINEL, iNaT),
    DtypeKind.INT: (ColumnNullType.NON_NULLABLE, None),
    DtypeKind.UINT: (ColumnNullType.NON_NULLABLE, None),
    DtypeKind.BOOL: (ColumnNullType.NON_NULLABLE, None),
    # 分类类型的空值存储为 `-1` 的标志值
    # 在分类数据中（例如，`col.values.codes` 是 int8 的 np.ndarray）
    DtypeKind.CATEGORICAL: (ColumnNullType.USE_SENTINEL, -1),
    # 按照 Arrow 的标准，使用 1 表示有效值，0 表示缺失/空值
    DtypeKind.STRING: (ColumnNullType.USE_BYTEMASK, 0),
}
# 定义不同 DtypeKind 对应的空值描述字典

_NO_VALIDITY_BUFFER = {
    ColumnNullType.NON_NULLABLE: "This column is non-nullable",
    ColumnNullType.USE_NAN: "This column uses NaN as null",
    ColumnNullType.USE_SENTINEL: "This column uses a sentinel value",
}
# 定义不同 ColumnNullType 对应的无效性缓冲区描述字典

class PandasColumn(Column):
    """
    一个列对象，只包含交换协议中所需的方法和属性。
    一个列可以包含一个或多个块。每个块最多可以包含三个缓冲区 -
    数据缓冲区、掩码缓冲区（取决于空值表示方式）和偏移量缓冲区（如果是可变大小的二进制，例如可变长度字符串）。
    注意：此 Column 对象只能由 ``__dataframe__`` 生成，因此不需要自己的版本或 ``__column__`` 协议。
    """
    # 初始化方法，接受一个 pandas Series 对象作为参数，并可选地允许复制操作
    def __init__(self, column: pd.Series, allow_copy: bool = True) -> None:
        """
        Note: doesn't deal with extension arrays yet, just assume a regular
        Series/ndarray for now.
        """
        # 如果传入的 column 是一个 DataFrame，则抛出 TypeError
        if isinstance(column, pd.DataFrame):
            raise TypeError(
                "Expected a Series, got a DataFrame. This likely happened "
                "because you called __dataframe__ on a DataFrame which, "
                "after converting column names to string, resulted in duplicated "
                f"names: {column.columns}. Please rename these columns before "
                "using the interchange protocol."
            )
        # 如果传入的 column 不是一个 Series，则抛出 NotImplementedError
        if not isinstance(column, pd.Series):
            raise NotImplementedError(f"Columns of type {type(column)} not handled yet")

        # 将传入的 column 存储为私有属性 _col
        self._col = column
        # 存储是否允许复制操作的标志
        self._allow_copy = allow_copy

    # 返回列中元素的数量
    def size(self) -> int:
        """
        Size of the column, in elements.
        """
        return self._col.size

    # 返回第一个元素的偏移量，始终为零
    @property
    def offset(self) -> int:
        """
        Offset of first element. Always zero.
        """
        # TODO: chunks are implemented now, probably this should return something
        return 0

    # 用缓存装饰器标记，返回列的数据类型元组
    @cache_readonly
    def dtype(self) -> tuple[DtypeKind, int, str, str]:
        dtype = self._col.dtype

        # 如果列的数据类型是 pd.CategoricalDtype，则处理成 Arrow 库的数据类型
        if isinstance(dtype, pd.CategoricalDtype):
            codes = self._col.values.codes
            (
                _,
                bitwidth,
                c_arrow_dtype_f_str,
                _,
            ) = self._dtype_from_pandasdtype(codes.dtype)
            return (
                DtypeKind.CATEGORICAL,
                bitwidth,
                c_arrow_dtype_f_str,
                Endianness.NATIVE,
            )
        # 如果列的数据类型是字符串类型，根据具体情况返回字符串类型或抛出未实现异常
        elif is_string_dtype(dtype):
            if infer_dtype(self._col) in ("string", "empty"):
                return (
                    DtypeKind.STRING,
                    8,
                    dtype_to_arrow_c_fmt(dtype),
                    Endianness.NATIVE,
                )
            raise NotImplementedError("Non-string object dtypes are not supported yet")
        # 对于其他数据类型，调用 _dtype_from_pandasdtype 方法处理
        else:
            return self._dtype_from_pandasdtype(dtype)
    @property
    def describe_categorical(self):
        """
        如果数据类型是分类的，有两种选择：
        - 数据缓冲区中只有值。
        - 对分类值有单独的非分类列编码。

        如果数据类型不是分类的，则引发 TypeError。

        返回字典的内容：
            - "is_ordered" : bool，索引排序是否语义上有意义。
            - "is_dictionary" : bool，是否存在字典样式的分类值到其他对象的映射。
            - "categories" : 列，表示索引到类别值的（隐式）映射（例如一个包含 cat1、cat2 等的数组）。
                             如果不是字典样式的分类，则为 None。
        """
        if not self.dtype[0] == DtypeKind.CATEGORICAL:
            # 如果数据类型不是分类的，抛出 TypeError 异常
            raise TypeError(
                "describe_categorical 只能用于具有分类数据类型的列！"
            )

        # 返回描述分类的属性字典
        return {
            "is_ordered": self._col.cat.ordered,
            "is_dictionary": True,
            "categories": PandasColumn(pd.Series(self._col.cat.categories)),
        }
    def describe_null(self):
        # 如果列的数据类型是 BaseMaskedDtype，则使用字节掩码类型处理空值
        if isinstance(self._col.dtype, BaseMaskedDtype):
            column_null_dtype = ColumnNullType.USE_BYTEMASK
            null_value = 1
            return column_null_dtype, null_value
        # 如果列的数据类型是 ArrowDtype
        if isinstance(self._col.dtype, ArrowDtype):
            # 由于在初始化时已经重新分块（如果必要/允许），因此此时已经是单块数据
            if self._col.array._pa_array.chunks[0].buffers()[0] is None:  # type: ignore[attr-defined]
                # 如果单块数据的缓冲区为空，则列是非空的
                return ColumnNullType.NON_NULLABLE, None
            # 否则，使用位掩码类型处理空值
            return ColumnNullType.USE_BITMASK, 0
        # 获取数据类型的第一个字符
        kind = self.dtype[0]
        try:
            # 从预定义的字典中获取该数据类型对应的空值描述
            null, value = _NULL_DESCRIPTION[kind]
        except KeyError as err:
            # 抛出未实现的错误，说明该数据类型暂不支持
            raise NotImplementedError(f"Data type {kind} not yet supported") from err

        return null, value

    @cache_readonly
    def null_count(self) -> int:
        """
        Number of null elements. Should always be known.
        """
        # 返回列中空值的数量，使用 Pandas 的 isna() 方法统计空值，然后求和
        return self._col.isna().sum().item()

    @property
    def metadata(self) -> dict[str, pd.Index]:
        """
        Store specific metadata of the column.
        """
        # 返回列的特定元数据，这里存储了列的 Pandas 索引
        return {"pandas.index": self._col.index}

    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.
        """
        # 返回该列的分块数，这里默认是 1，表示不分块
        return 1

    def get_chunks(self, n_chunks: int | None = None):
        """
        Return an iterator yielding the chunks.
        See `DataFrame.get_chunks` for details on ``n_chunks``.
        """
        # 如果指定了 n_chunks 且大于 1
        if n_chunks and n_chunks > 1:
            size = len(self._col)
            step = size // n_chunks
            if size % n_chunks != 0:
                step += 1
            # 分块迭代生成器，返回 PandasColumn 对象的迭代器
            for start in range(0, step * n_chunks, step):
                yield PandasColumn(
                    self._col.iloc[start : start + step], self._allow_copy
                )
        else:
            # 否则，直接返回当前列对象的迭代器
            yield self
    def get_buffers(self) -> ColumnBuffers:
        """
        Return a dictionary containing the underlying buffers.
        The returned dictionary has the following contents:
            - "data": a two-element tuple whose first element is a buffer
                      containing the data and whose second element is the data
                      buffer's associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
                          containing mask values indicating missing data and
                          whose second element is the mask value buffer's
                          associated dtype. None if the null representation is
                          not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
                         containing the offset values for variable-size binary
                         data (e.g., variable-length strings) and whose second
                         element is the offsets buffer's associated dtype. None
                         if the data buffer does not have an associated offsets
                         buffer.
        """
        # 初始化一个空的 ColumnBuffers 字典
        buffers: ColumnBuffers = {
            "data": self._get_data_buffer(),  # 调用获取数据缓冲区的方法，将其作为 "data" 键的值存入字典
            "validity": None,  # 设置 "validity" 键的值为 None，表示初始时没有有效性缓冲区
            "offsets": None,   # 设置 "offsets" 键的值为 None，表示初始时没有偏移缓冲区
        }

        try:
            buffers["validity"] = self._get_validity_buffer()  # 尝试获取有效性缓冲区，若成功则更新 "validity" 键的值
        except NoBufferPresent:
            pass  # 如果获取有效性缓冲区时抛出 NoBufferPresent 异常，则捕获并继续执行

        try:
            buffers["offsets"] = self._get_offsets_buffer()  # 尝试获取偏移缓冲区，若成功则更新 "offsets" 键的值
        except NoBufferPresent:
            pass  # 如果获取偏移缓冲区时抛出 NoBufferPresent 异常，则捕获并继续执行

        return buffers  # 返回填充后的 buffers 字典，其中包含可能的 "data"、"validity" 和 "offsets" 缓冲区

    def _get_data_buffer(
        self,
    def _get_validity_buffer(self) -> tuple[Buffer, Any] | None:
        """
        Return the buffer containing the mask values indicating missing data and
        the buffer's associated dtype.
        Raises NoBufferPresent if null representation is not a bit or byte mask.
        """
        null, invalid = self.describe_null  # 获取 null 和 invalid 值

        buffer: Buffer  # 声明变量 buffer 类型为 Buffer

        if isinstance(self._col.dtype, ArrowDtype):
            # 如果列的数据类型是 ArrowDtype

            # 获取底层数组的第一个 chunk
            arr = self._col.array._pa_array.chunks[0]  # type: ignore[attr-defined]

            # 定义数据类型元组
            dtype = (DtypeKind.BOOL, 1, ArrowCTypes.BOOL, Endianness.NATIVE)

            # 如果数组的第一个缓冲区为空，则返回 None
            if arr.buffers()[0] is None:
                return None

            # 创建 PandasBufferPyarrow 对象
            buffer = PandasBufferPyarrow(
                arr.buffers()[0],  # 使用数组的第一个缓冲区作为参数
                length=len(arr),    # 设置长度为数组的长度
            )

            return buffer, dtype  # 返回 buffer 和 dtype

        if isinstance(self._col.dtype, BaseMaskedDtype):
            # 如果列的数据类型是 BaseMaskedDtype

            # 获取列的掩码数组
            mask = self._col.array._mask  # type: ignore[attr-defined]

            # 创建 PandasBuffer 对象
            buffer = PandasBuffer(mask)

            # 定义数据类型元组
            dtype = (DtypeKind.BOOL, 8, ArrowCTypes.BOOL, Endianness.NATIVE)

            return buffer, dtype  # 返回 buffer 和 dtype

        if self.dtype[0] == DtypeKind.STRING:
            # 如果数据类型的第一个元素是字符串类型

            # 将列转换为 NumPy 数组
            buf = self._col.to_numpy()

            # 确定有效值的编码
            valid = invalid == 0
            invalid = not valid

            # 创建一个布尔类型的 mask 数组
            mask = np.zeros(shape=(len(buf),), dtype=np.bool_)
            for i, obj in enumerate(buf):
                mask[i] = valid if isinstance(obj, str) else invalid

            # 使用 NumPy 数组创建 PandasBuffer 对象
            buffer = PandasBuffer(mask)

            # 定义数据类型元组
            dtype = (DtypeKind.BOOL, 8, ArrowCTypes.BOOL, Endianness.NATIVE)

            return buffer, dtype  # 返回 buffer 和 dtype

        try:
            msg = f"{_NO_VALIDITY_BUFFER[null]} so does not have a separate mask"
        except KeyError as err:
            # 如果 KeyError 发生，则抛出 NotImplementedError 异常
            raise NotImplementedError("See self.describe_null") from err

        # 抛出 NoBufferPresent 异常，提示缺少有效的缓冲区
        raise NoBufferPresent(msg)
    def _get_offsets_buffer(self) -> tuple[PandasBuffer, Any]:
        """
        Return the buffer containing the offset values for variable-size binary
        data (e.g., variable-length strings) and the buffer's associated dtype.
        Raises NoBufferPresent if the data buffer does not have an associated
        offsets buffer.
        """
        # 检查数据类型是否为字符串类型
        if self.dtype[0] == DtypeKind.STRING:
            # 对每个字符串，需要手动确定下一个偏移量
            # 将列转换为 NumPy 数组
            values = self._col.to_numpy()
            ptr = 0
            # 创建一个长度为 values 数组长度加一的零数组，用于存储偏移量
            offsets = np.zeros(shape=(len(values) + 1,), dtype=np.int64)
            for i, v in enumerate(values):
                # 对于缺失值（在这种情况下，`np.nan` 值），我们不增加指针
                if isinstance(v, str):
                    # 将字符串编码为 UTF-8，并计算其字节长度
                    b = v.encode(encoding="utf-8")
                    ptr += len(b)

                # 将计算得到的偏移量存储到数组中
                offsets[i + 1] = ptr

            # 使用 NumPy 数组作为后端存储，将偏移量转换为 Pandas 的“buffer”
            buffer = PandasBuffer(offsets)

            # 组装缓冲区的数据类型信息
            dtype = (
                DtypeKind.INT,
                64,
                ArrowCTypes.INT64,
                Endianness.NATIVE,
            )  # 注意：目前仅支持本机字节顺序
        else:
            # 如果列具有固定长度的数据类型，则抛出异常
            raise NoBufferPresent(
                "This column has a fixed-length dtype so "
                "it does not have an offsets buffer"
            )

        # 返回偏移量缓冲区和数据类型元组
        return buffer, dtype
```
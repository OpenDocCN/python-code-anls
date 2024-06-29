# `D:\src\scipysrc\pandas\pandas\io\formats\csvs.py`

```
"""
Module for formatting output data into CSV files.
"""

from __future__ import annotations  # 导入未来的注释语法支持

from collections.abc import (  # 导入集合抽象基类中的特定类
    Hashable,  # 可散列对象的抽象基类
    Iterable,  # 可迭代对象的抽象基类
    Iterator,  # 迭代器对象的抽象基类
    Sequence,  # 序列对象的抽象基类
)
import csv as csvlib  # 导入 CSV 模块，并使用别名 csvlib
import os  # 导入操作系统相关的功能
from typing import (  # 导入类型提示相关的模块
    TYPE_CHECKING,  # 用于类型检查的标志
    Any,  # 任意类型的对象
    cast,  # 类型转换函数
)

import numpy as np  # 导入 NumPy 库，并使用别名 np

from pandas._libs import writers as libwriters  # 导入 Pandas 内部的 writers 模块，并使用别名 libwriters
from pandas._typing import SequenceNotStr  # 导入 Pandas 的类型提示中的 SequenceNotStr 类型
from pandas.util._decorators import cache_readonly  # 导入 Pandas 的缓存只读装饰器

from pandas.core.dtypes.generic import (  # 导入 Pandas 核心数据类型中的特定类
    ABCDatetimeIndex,  # 抽象基类 DatetimeIndex
    ABCIndex,  # 抽象基类 Index
    ABCMultiIndex,  # 抽象基类 MultiIndex
    ABCPeriodIndex,  # 抽象基类 PeriodIndex
)
from pandas.core.dtypes.missing import notna  # 导入 Pandas 缺失数据处理模块中的 notna 函数

from pandas.core.indexes.api import Index  # 导入 Pandas 核心索引 API 中的 Index 类

from pandas.io.common import get_handle  # 导入 Pandas IO 模块中的 get_handle 函数

if TYPE_CHECKING:
    from pandas._typing import (  # 导入 Pandas 的类型提示中的特定类型
        CompressionOptions,  # 压缩选项类型
        FilePath,  # 文件路径类型
        FloatFormatType,  # 浮点格式类型
        IndexLabel,  # 索引标签类型
        StorageOptions,  # 存储选项类型
        WriteBuffer,  # 写缓冲区类型
        npt,  # NumPy 类型提示
    )

    from pandas.io.formats.format import DataFrameFormatter  # 导入 Pandas IO 格式化模块中的 DataFrameFormatter 类


_DEFAULT_CHUNKSIZE_CELLS = 100_000  # 设置默认的块大小为 100,000 个单元格


class CSVFormatter:
    cols: npt.NDArray[np.object_]  # 类属性 cols，类型为 NumPy 数组，存储对象为 np.object_

    def __init__(  # 初始化方法，用于实例化 CSVFormatter 类的对象
        self,
        formatter: DataFrameFormatter,  # 格式化器对象，类型为 DataFrameFormatter
        path_or_buf: FilePath | WriteBuffer[str] | WriteBuffer[bytes] = "",  # 文件路径或缓冲区对象
        sep: str = ",",  # CSV 文件的分隔符，默认为逗号
        cols: Sequence[Hashable] | None = None,  # 列名序列或 None，默认为 None
        index_label: IndexLabel | None = None,  # 索引标签或 None，默认为 None
        mode: str = "w",  # 文件打开模式，默认为写入模式
        encoding: str | None = None,  # 文件编码或 None
        errors: str = "strict",  # 错误处理策略，默认为严格模式
        compression: CompressionOptions = "infer",  # 压缩选项，默认为推断模式
        quoting: int | None = None,  # 引用风格，默认为 None
        lineterminator: str | None = "\n",  # 行终止符，默认为换行符
        chunksize: int | None = None,  # 块大小，默认为 None
        quotechar: str | None = '"',  # 引号字符，默认为双引号
        date_format: str | None = None,  # 日期格式，默认为 None
        doublequote: bool = True,  # 是否双引号，默认为 True
        escapechar: str | None = None,  # 转义字符，默认为 None
        storage_options: StorageOptions | None = None,  # 存储选项或 None
    ) -> None:
        self.fmt = formatter  # 将 formatter 参数赋值给实例属性 fmt

        self.obj = self.fmt.frame  # 从 fmt 属性中获取 frame 属性并赋值给实例属性 obj

        self.filepath_or_buffer = path_or_buf  # 将 path_or_buf 参数赋值给实例属性 filepath_or_buffer
        self.encoding = encoding  # 将 encoding 参数赋值给实例属性 encoding
        self.compression: CompressionOptions = compression  # 将 compression 参数赋值给实例属性 compression
        self.mode = mode  # 将 mode 参数赋值给实例属性 mode
        self.storage_options = storage_options  # 将 storage_options 参数赋值给实例属性 storage_options

        self.sep = sep  # 将 sep 参数赋值给实例属性 sep
        self.index_label = self._initialize_index_label(index_label)  # 调用 _initialize_index_label 方法初始化 index_label 属性
        self.errors = errors  # 将 errors 参数赋值给实例属性 errors
        self.quoting = quoting or csvlib.QUOTE_MINIMAL  # 设置 quoting 属性为给定值或默认为 csvlib.QUOTE_MINIMAL
        self.quotechar = self._initialize_quotechar(quotechar)  # 调用 _initialize_quotechar 方法初始化 quotechar 属性
        self.doublequote = doublequote  # 将 doublequote 参数赋值给实例属性 doublequote
        self.escapechar = escapechar  # 将 escapechar 参数赋值给实例属性 escapechar
        self.lineterminator = lineterminator or os.linesep  # 设置 lineterminator 属性为给定值或默认为系统换行符
        self.date_format = date_format  # 将 date_format 参数赋值给实例属性 date_format
        self.cols = self._initialize_columns(cols)  # 调用 _initialize_columns 方法初始化 cols 属性
        self.chunksize = self._initialize_chunksize(chunksize)  # 调用 _initialize_chunksize 方法初始化 chunksize 属性

    @property
    def na_rep(self) -> str:
        return self.fmt.na_rep  # 返回 fmt 属性中的 na_rep 属性值

    @property
    def float_format(self) -> FloatFormatType | None:
        return self.fmt.float_format  # 返回 fmt 属性中的 float_format 属性值

    @property
    def decimal(self) -> str:
        return self.fmt.decimal  # 返回 fmt 属性中的 decimal 属性值

    @property
    def header(self) -> bool | SequenceNotStr[str]:
        return self.fmt.header  # 返回 fmt 属性中的 header 属性值
    # 返回 self.fmt 的索引是否可用的布尔值
    def index(self) -> bool:
        return self.fmt.index

    # 初始化索引标签，根据给定的 index_label 参数进行处理
    def _initialize_index_label(self, index_label: IndexLabel | None) -> IndexLabel:
        if index_label is not False:
            if index_label is None:
                # 如果 index_label 为 None，则从对象中获取索引标签
                return self._get_index_label_from_obj()
            elif not isinstance(index_label, (list, tuple, np.ndarray, ABCIndex)):
                # 如果 index_label 不是列表、元组、数组或 ABCIndex 的实例，则将其视为单个索引标签字符串
                return [index_label]
        return index_label

    # 从对象中获取索引标签
    def _get_index_label_from_obj(self) -> Sequence[Hashable]:
        if isinstance(self.obj.index, ABCMultiIndex):
            # 如果对象的索引是多重索引，则获取多重索引的所有名称
            return self._get_index_label_multiindex()
        else:
            # 否则，获取平坦索引的名称
            return self._get_index_label_flat()

    # 获取多重索引的索引标签
    def _get_index_label_multiindex(self) -> Sequence[Hashable]:
        return [name or "" for name in self.obj.index.names]

    # 获取平坦索引的索引标签
    def _get_index_label_flat(self) -> Sequence[Hashable]:
        index_label = self.obj.index.name
        return [""] if index_label is None else [index_label]

    # 初始化引号字符的处理方式
    def _initialize_quotechar(self, quotechar: str | None) -> str | None:
        if self.quoting != csvlib.QUOTE_NONE:
            # 如果 quoting 不等于 QUOTE_NONE，则返回给定的 quotechar
            return quotechar
        return None

    # 检查是否具有多重索引的列
    @property
    def has_mi_columns(self) -> bool:
        return bool(isinstance(self.obj.columns, ABCMultiIndex))

    # 初始化列，并根据参数 cols 进行适当的处理
    def _initialize_columns(
        self, cols: Iterable[Hashable] | None
    ) -> npt.NDArray[np.object_]:
        # 验证是否具有多重索引列，并处理对应的错误信息
        if self.has_mi_columns:
            if cols is not None:
                msg = "cannot specify cols with a MultiIndex on the columns"
                raise TypeError(msg)

        if cols is not None:
            if isinstance(cols, ABCIndex):
                # 如果 cols 是 ABCIndex 的实例，则使用其 _get_values_for_csv 方法获取列的值
                cols = cols._get_values_for_csv(**self._number_format)
            else:
                cols = list(cols)
            # 根据给定的 cols 参数重新选择 DataFrame 的列
            self.obj = self.obj.loc[:, cols]

        # 更新列以包括可能的重复项的多重性，并确保 cols 只是一个标签列表
        new_cols = self.obj.columns
        return new_cols._get_values_for_csv(**self._number_format)

    # 初始化块大小参数 chunksize
    def _initialize_chunksize(self, chunksize: int | None) -> int:
        if chunksize is None:
            # 如果 chunksize 为 None，则根据列数计算默认的块大小
            return (_DEFAULT_CHUNKSIZE_CELLS // (len(self.cols) or 1)) or 1
        return int(chunksize)

    # 返回用于存储数字格式设置的字典
    @property
    def _number_format(self) -> dict[str, Any]:
        """Dictionary used for storing number formatting settings."""
        return {
            "na_rep": self.na_rep,
            "float_format": self.float_format,
            "date_format": self.date_format,
            "quoting": self.quoting,
            "decimal": self.decimal,
        }

    @cache_readonly
    # 返回数据的索引
    def data_index(self) -> Index:
        # 获取数据的索引
        data_index = self.obj.index
        # 如果数据的索引是日期时间索引或周期索引，并且日期格式不为空
        if (
            isinstance(data_index, (ABCDatetimeIndex, ABCPeriodIndex))
            and self.date_format is not None
        ):
            # 将日期时间索引格式化为指定的日期格式
            data_index = Index(
                [x.strftime(self.date_format) if notna(x) else "" for x in data_index]
            )
        # 如果数据的索引是多重索引
        elif isinstance(data_index, ABCMultiIndex):
            # 移除未使用的级别
            data_index = data_index.remove_unused_levels()
        # 返回数据的索引
        return data_index

    # 返回数据的级数
    @property
    def nlevels(self) -> int:
        # 如果存在索引
        if self.index:
            # 返回数据索引的级数，如果不存在则返回1
            return getattr(self.data_index, "nlevels", 1)
        else:
            # 如果不存在索引，则返回0
            return 0

    # 返回是否存在别名
    @property
    def _has_aliases(self) -> bool:
        # 判断是否存在别名
        return isinstance(self.header, (tuple, list, np.ndarray, ABCIndex))

    # 返回是否需要保存头部信息
    @property
    def _need_to_save_header(self) -> bool:
        # 判断是否需要保存头部信息
        return bool(self._has_aliases or self.header)

    # 返回要写入的列
    @property
    def write_cols(self) -> SequenceNotStr[Hashable]:
        # 如果存在别名
        if self._has_aliases:
            assert not isinstance(self.header, bool)
            # 检查列和别名的数量是否一致
            if len(self.header) != len(self.cols):
                raise ValueError(
                    f"Writing {len(self.cols)} cols but got {len(self.header)} aliases"
                )
            return self.header
        else:
            # 返回可哈希的列
            return cast(SequenceNotStr[Hashable], self.cols)

    # 返回编码后的标签
    @property
    def encoded_labels(self) -> list[Hashable]:
        encoded_labels: list[Hashable] = []

        # 如果存在索引和索引标签
        if self.index and self.index_label:
            assert isinstance(self.index_label, Sequence)
            # 将索引标签转换为列表
            encoded_labels = list(self.index_label)

        # 如果不存在多重索引列或存在别名
        if not self.has_mi_columns or self._has_aliases:
            # 将写入的列添加到编码标签中
            encoded_labels += list(self.write_cols)

        return encoded_labels

    # 保存数据
    def save(self) -> None:
        """
        创建写入器并保存数据。
        """
        # 应用压缩和字节/文本转换
        with get_handle(
            self.filepath_or_buffer,
            self.mode,
            encoding=self.encoding,
            errors=self.errors,
            compression=self.compression,
            storage_options=self.storage_options,
        ) as handles:
            # 注意：这里的编码是无关紧要的
            # 创建 CSV 写入器
            self.writer = csvlib.writer(
                handles.handle,
                lineterminator=self.lineterminator,
                delimiter=self.sep,
                quoting=self.quoting,
                doublequote=self.doublequote,
                escapechar=self.escapechar,
                quotechar=self.quotechar,
            )

            # 保存数据
            self._save()

    # 实际保存数据
    def _save(self) -> None:
        # 如果需要保存头部信息
        if self._need_to_save_header:
            # 保存头部信息
            self._save_header()
        # 保存数据主体
        self._save_body()
    # 如果没有多级索引列或者存在别名，则写入编码后的标签行
    if not self.has_mi_columns or self._has_aliases:
        self.writer.writerow(self.encoded_labels)
    else:
        # 否则，生成多级索引头部行并写入
        for row in self._generate_multiindex_header_rows():
            self.writer.writerow(row)

    # 生成多级索引头部行的生成器函数
    def _generate_multiindex_header_rows(self) -> Iterator[list[Hashable]]:
        # 获取对象的列
        columns = self.obj.columns
        for i in range(columns.nlevels):
            # 需要至少一个索引列来写入列名
            col_line = []
            if self.index:
                # 列名是第一列
                col_line.append(columns.names[i])

                # 如果索引标签是列表且长度大于1，则填充空白，以匹配长度
                if isinstance(self.index_label, list) and len(self.index_label) > 1:
                    col_line.extend([""] * (len(self.index_label) - 1))

            # 扩展该级别的所有列值到列名行中
            col_line.extend(columns._get_level_values(i))
            yield col_line

        # 如果编码后的标签存在且不全为空字符串，则生成并返回标签行及空白填充列
        if self.encoded_labels and set(self.encoded_labels) != {""}:
            yield self.encoded_labels + [""] * len(columns)

    # 保存数据主体部分
    def _save_body(self) -> None:
        # 数据行数
        nrows = len(self.data_index)
        # 数据分块数
        chunks = (nrows // self.chunksize) + 1
        for i in range(chunks):
            # 计算当前块的起始和结束索引
            start_i = i * self.chunksize
            end_i = min(start_i + self.chunksize, nrows)
            if start_i >= end_i:
                break
            # 保存当前块的数据
            self._save_chunk(start_i, end_i)

    # 保存单个数据块
    def _save_chunk(self, start_i: int, end_i: int) -> None:
        # 创建数据块的切片
        slicer = slice(start_i, end_i)
        # 从对象中选择切片的数据
        df = self.obj.iloc[slicer]

        # 获取数据块的 CSV 格式值
        res = df._get_values_for_csv(**self._number_format)
        # 将结果转换为列数组列表
        data = list(res._iter_column_arrays())

        # 获取数据索引的 CSV 格式值
        ix = self.data_index[slicer]._get_values_for_csv(**self._number_format)
        # 使用底层的写入函数写入 CSV 行
        libwriters.write_csv_rows(
            data,
            ix,
            self.nlevels,
            self.cols,
            self.writer,
        )
```
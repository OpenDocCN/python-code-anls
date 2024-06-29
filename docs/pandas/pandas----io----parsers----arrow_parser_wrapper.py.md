# `D:\src\scipysrc\pandas\pandas\io\parsers\arrow_parser_wrapper.py`

```
# 导入必要的模块和类，包括从__future__模块导入annotations特性
from __future__ import annotations

# 引入类型检查相关的模块
from typing import TYPE_CHECKING
import warnings

# 从pandas._config模块导入using_pyarrow_string_dtype函数
from pandas._config import using_pyarrow_string_dtype

# 从pandas._libs模块导入lib
from pandas._libs import lib

# 从pandas.compat._optional模块导入import_optional_dependency函数
from pandas.compat._optional import import_optional_dependency

# 从pandas.errors模块导入ParserError和ParserWarning异常类
from pandas.errors import (
    ParserError,
    ParserWarning,
)

# 从pandas.util._exceptions模块导入find_stack_level函数
from pandas.util._exceptions import find_stack_level

# 从pandas.core.dtypes.common模块导入pandas_dtype函数
from pandas.core.dtypes.common import pandas_dtype

# 从pandas.core.dtypes.inference模块导入is_integer函数
from pandas.core.dtypes.inference import is_integer

# 导入pandas和DataFrame类
import pandas as pd
from pandas import DataFrame

# 从pandas.io._util模块导入_arrow_dtype_mapping和arrow_string_types_mapper函数
from pandas.io._util import (
    _arrow_dtype_mapping,
    arrow_string_types_mapper,
)

# 从pandas.io.parsers.base_parser模块导入ParserBase类
from pandas.io.parsers.base_parser import ParserBase

# 如果TYPE_CHECKING为True，导入ReadBuffer类型
if TYPE_CHECKING:
    from pandas._typing import ReadBuffer


class ArrowParserWrapper(ParserBase):
    """
    Wrapper for the pyarrow engine for read_csv()
    """

    def __init__(self, src: ReadBuffer[bytes], **kwds) -> None:
        # 调用父类的初始化方法
        super().__init__(kwds)
        # 将传入的关键字参数保存到实例变量self.kwds中
        self.kwds = kwds
        # 将传入的数据源src保存到实例变量self.src中
        self.src = src

        # 调用内部方法_parse_kwds()，处理关键字参数
        self._parse_kwds()

    def _parse_kwds(self) -> None:
        """
        Validates keywords before passing to pyarrow.
        """
        # 获取关键字参数中的encoding参数，如果不存在，默认使用utf-8编码
        encoding: str | None = self.kwds.get("encoding")
        self.encoding = "utf-8" if encoding is None else encoding

        # 获取关键字参数中的na_values参数
        na_values = self.kwds["na_values"]
        # 如果na_values是字典类型，则抛出ValueError异常，因为pyarrow引擎不支持字典类型的na_values参数
        if isinstance(na_values, dict):
            raise ValueError(
                "The pyarrow engine doesn't support passing a dict for na_values"
            )
        # 将na_values参数转换为列表形式，并保存到self.na_values中
        self.na_values = list(self.kwds["na_values"])
    # 处理 Pandas DataFrame 输出的最终步骤，根据指定的参数进行数据处理
    def _finalize_pandas_output(self, frame: DataFrame) -> DataFrame:
        """
        Processes data read in based on kwargs.

        Parameters
        ----------
        frame: DataFrame
            The DataFrame to process.

        Returns
        -------
        DataFrame
            The processed DataFrame.
        """
        # 获取 DataFrame 的列数
        num_cols = len(frame.columns)
        # 默认情况下假设使用了多级索引命名
        multi_index_named = True
        # 如果 header 为 None，则需要处理列名
        if self.header is None:
            # 如果 names 也为 None，则尝试用数字范围作为列名
            if self.names is None:
                if self.header is None:
                    self.names = range(num_cols)
            # 如果 names 的长度不等于列数，则进行调整
            if len(self.names) != num_cols:
                # usecols 传递到 pyarrow，这里只处理索引列
                # 如果 names 长度不等于列数，说明可能是由于整数索引列造成的，我们需要填充到预期的长度
                self.names = list(range(num_cols - len(self.names))) + self.names
                multi_index_named = False
            # 将 DataFrame 的列名设置为处理后的 names
            frame.columns = self.names

        # 执行日期转换，如果需要的话
        frame = self._do_date_conversions(frame.columns, frame)

        # 如果指定了 index_col，则设置 DataFrame 的索引
        if self.index_col is not None:
            index_to_set = self.index_col.copy()
            for i, item in enumerate(self.index_col):
                # 如果 index_col 中是整数，替换为对应的列名
                if is_integer(item):
                    index_to_set[i] = frame.columns[item]
                # 如果 index_col 中是字符串，检查它是否在 DataFrame 的列名中
                elif item not in frame.columns:
                    raise ValueError(f"Index {item} invalid")

                # 处理 index_col 的数据类型，并从 dtypes 中删除
                if self.dtype is not None:
                    key, new_dtype = (
                        (item, self.dtype.get(item))
                        if self.dtype.get(item) is not None
                        else (frame.columns[item], self.dtype.get(frame.columns[item]))
                    )
                    if new_dtype is not None:
                        frame[key] = frame[key].astype(new_dtype)
                        del self.dtype[key]

            # 设置 DataFrame 的索引，并且在没有 header 的情况下清除索引名
            frame.set_index(index_to_set, drop=True, inplace=True)
            if self.header is None and not multi_index_named:
                frame.index.names = [None] * len(frame.index.names)

        # 处理整个 DataFrame 的数据类型转换，如果需要的话
        if self.dtype is not None:
            # 忽略 dtype 映射中不存在的列
            if isinstance(self.dtype, dict):
                self.dtype = {
                    k: pandas_dtype(v)
                    for k, v in self.dtype.items()
                    if k in frame.columns
                }
            else:
                self.dtype = pandas_dtype(self.dtype)
            try:
                frame = frame.astype(self.dtype)
            except TypeError as err:
                # 如果出现类型错误，重新引发 ValueError 以保持 API 一致性
                raise ValueError(str(err)) from err
        # 返回处理后的 DataFrame
        return frame
    # 定义一个方法 `_validate_usecols`，用于验证传入的参数 `usecols`
    def _validate_usecols(self, usecols) -> None:
        # 如果 `usecols` 是类似列表的对象并且不是全部由字符串组成
        if lib.is_list_like(usecols) and not all(isinstance(x, str) for x in usecols):
            # 抛出值错误异常，指明 pyarrow 引擎不允许 'usecols' 是整数列位置，应传入字符串列名的列表。
            raise ValueError(
                "The pyarrow engine does not allow 'usecols' to be integer "
                "column positions. Pass a list of string column names instead."
            )
        # 如果 `usecols` 是可调用对象
        elif callable(usecols):
            # 抛出值错误异常，指明 pyarrow 引擎不允许 'usecols' 是可调用对象。
            raise ValueError(
                "The pyarrow engine does not allow 'usecols' to be a callable."
            )
    def read(self) -> DataFrame:
        """
        Reads the contents of a CSV file into a DataFrame and
        processes it according to the kwargs passed in the
        constructor.

        Returns
        -------
        DataFrame
            The DataFrame created from the CSV file.
        """
        # 导入 pyarrow 库中的相关模块
        pa = import_optional_dependency("pyarrow")
        pyarrow_csv = import_optional_dependency("pyarrow.csv")
        # 获取 pyarrow 相关选项
        self._get_pyarrow_options()

        try:
            # 使用传入的转换选项创建 ConvertOptions 对象
            convert_options = pyarrow_csv.ConvertOptions(**self.convert_options)
        except TypeError as err:
            # 处理转换选项的类型错误异常
            include = self.convert_options.get("include_columns", None)
            if include is not None:
                # 验证使用列的有效性
                self._validate_usecols(include)

            # 检查空值设置是否符合要求
            nulls = self.convert_options.get("null_values", set())
            if not lib.is_list_like(nulls) or not all(
                isinstance(x, str) for x in nulls
            ):
                raise TypeError(
                    "The 'pyarrow' engine requires all na_values to be strings"
                ) from err

            raise

        try:
            # 使用 pyarrow_csv 读取 CSV 文件内容到 table 变量
            table = pyarrow_csv.read_csv(
                self.src,
                read_options=pyarrow_csv.ReadOptions(**self.read_options),
                parse_options=pyarrow_csv.ParseOptions(**self.parse_options),
                convert_options=convert_options,
            )
        except pa.ArrowInvalid as e:
            # 捕获 ArrowInvalid 异常并转换为 ParserError 抛出
            raise ParserError(e) from e

        # 获取 dtype_backend 参数
        dtype_backend = self.kwds["dtype_backend"]

        # 转换所有的 pa.null() 列为 float64（不可为空值）
        # 否则为 Int64（可为空值情况，见下文）
        if dtype_backend is lib.no_default:
            new_schema = table.schema
            new_type = pa.float64()
            for i, arrow_type in enumerate(table.schema.types):
                if pa.types.is_null(arrow_type):
                    new_schema = new_schema.set(
                        i, new_schema.field(i).with_type(new_type)
                    )

            # 将 table 转换为新的 schema
            table = table.cast(new_schema)

        with warnings.catch_warnings():
            # 忽略特定警告信息
            warnings.filterwarnings(
                "ignore",
                "make_block is deprecated",
                DeprecationWarning,
            )
            if dtype_backend == "pyarrow":
                # 将 table 转换为 pandas DataFrame，使用 pd.ArrowDtype 映射类型
                frame = table.to_pandas(types_mapper=pd.ArrowDtype)
            elif dtype_backend == "numpy_nullable":
                # 修改默认映射以将 null 映射为 Int64（以匹配其他引擎）
                dtype_mapping = _arrow_dtype_mapping()
                dtype_mapping[pa.null()] = pd.Int64Dtype()
                frame = table.to_pandas(types_mapper=dtype_mapping.get)
            elif using_pyarrow_string_dtype():
                # 使用特定的字符串类型映射器将 table 转换为 pandas DataFrame
                frame = table.to_pandas(types_mapper=arrow_string_types_mapper())
            else:
                # 将 table 转换为 pandas DataFrame
                frame = table.to_pandas()

        # 最终化 pandas 输出结果
        return self._finalize_pandas_output(frame)
```
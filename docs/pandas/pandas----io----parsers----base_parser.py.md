# `D:\src\scipysrc\pandas\pandas\io\parsers\base_parser.py`

```
from __future__ import annotations
# 导入将在 Python 3.10 中成为默认的类型注解语法的兼容性模块

from collections import defaultdict
# 导入 defaultdict 类型，用于创建默认值为集合的字典

from copy import copy
# 导入 copy 函数，用于复制对象

import csv
# 导入 CSV 文件读写相关的模块

from enum import Enum
# 导入枚举类型的支持模块

from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    final,
    overload,
)
# 导入类型提示相关的模块和装饰器

import warnings
# 导入警告处理模块

import numpy as np
# 导入 NumPy 数学运算库

from pandas._libs import (
    lib,
    parsers,
)
# 导入 Pandas 内部库

import pandas._libs.ops as libops
# 导入 Pandas 内部运算库

from pandas._libs.parsers import STR_NA_VALUES
# 导入 Pandas 内部解析器模块中的特定常量

from pandas.compat._optional import import_optional_dependency
# 导入 Pandas 兼容性模块，用于导入可选依赖

from pandas.errors import (
    ParserError,
    ParserWarning,
)
# 导入 Pandas 错误和警告类型

from pandas.util._exceptions import find_stack_level
# 导入 Pandas 工具中的栈级别查找函数

from pandas.core.dtypes.astype import astype_array
# 导入 Pandas 核心模块中的类型转换函数

from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_dict_like,
    is_extension_array_dtype,
    is_float_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_string_dtype,
    pandas_dtype,
)
# 导入 Pandas 核心模块中的数据类型判断函数

from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    ExtensionDtype,
)
# 导入 Pandas 核心数据类型

from pandas.core.dtypes.missing import isna
# 导入 Pandas 缺失值处理模块中的 isna 函数

from pandas import (
    ArrowDtype,
    DataFrame,
    DatetimeIndex,
    StringDtype,
)
# 从 Pandas 主命名空间中导入常用对象和数据类型

from pandas.core import algorithms
# 导入 Pandas 核心算法模块

from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    BooleanArray,
    Categorical,
    ExtensionArray,
    FloatingArray,
    IntegerArray,
)
# 导入 Pandas 数组相关的类型和类

from pandas.core.arrays.boolean import BooleanDtype
# 导入 Pandas 布尔数组数据类型

from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    default_index,
    ensure_index_from_sequences,
)
# 导入 Pandas 索引 API

from pandas.core.series import Series
# 导入 Pandas 序列类型

from pandas.core.tools import datetimes as tools
# 导入 Pandas 日期时间工具模块中的 tools 别名

from pandas.io.common import is_potential_multi_index
# 导入 Pandas IO 模块中的多索引判断函数

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Mapping,
        Sequence,
    )
    # 导入用于类型提示的集合抽象基类模块中的接口类型

    from pandas._typing import (
        ArrayLike,
        DtypeArg,
        DtypeObj,
        Hashable,
        HashableT,
        Scalar,
        SequenceT,
    )
    # 导入 Pandas 内部类型提示模块中的类型别名

class ParserBase:
    class BadLineHandleMethod(Enum):
        ERROR = 0
        WARN = 1
        SKIP = 2
    # 定义 ParserBase 类中的 BadLineHandleMethod 枚举类型，用于处理错误行的方式选择

    _implicit_index: bool
    _first_chunk: bool
    keep_default_na: bool
    dayfirst: bool
    cache_dates: bool
    usecols_dtype: str | None
    # 定义 ParserBase 类的一些成员变量，用于控制解析器行为的标志和选项
    def _validate_parse_dates_presence(self, columns: Sequence[Hashable]) -> set:
        """
        Check if parse_dates are in columns.

        If user has provided names for parse_dates, check if those columns
        are available.

        Parameters
        ----------
        columns : list
            List of names of the dataframe.

        Returns
        -------
        The names of the columns which will get parsed later if a list
        is given as specification.

        Raises
        ------
        ValueError
            If column to parse_date is not in dataframe.

        """
        # 如果 self.parse_dates 不是列表，返回空集合
        if not isinstance(self.parse_dates, list):
            return set()

        # 获取那些使用名称（字符串）而非索引引用的列名
        missing_cols = ", ".join(
            sorted(
                {
                    col
                    for col in self.parse_dates
                    if isinstance(col, str) and col not in columns
                }
            )
        )
        # 如果存在缺失的列，抛出 ValueError 异常
        if missing_cols:
            raise ValueError(
                f"Missing column provided to 'parse_dates': '{missing_cols}'"
            )
        
        # 将位置索引转换为实际列名
        return {
            col if (isinstance(col, str) or col in columns) else columns[col]
            for col in self.parse_dates
        }

    def close(self) -> None:
        pass

    @final
    def _should_parse_dates(self, i: int) -> bool:
        # 如果 parse_dates 是布尔值，直接返回其值
        if isinstance(self.parse_dates, bool):
            return self.parse_dates
        else:
            # 否则根据索引名或索引列进行判断
            if self.index_names is not None:
                name = self.index_names[i]
            else:
                name = None
            j = i if self.index_col is None else self.index_col[i]

            return (j in self.parse_dates) or (
                name is not None and name in self.parse_dates
            )

    @final
    def _extract_multi_indexer_columns(
        self,
        header,
        index_names: Sequence[Hashable] | None,
        passed_names: bool = False,
    ) -> tuple[
        Sequence[Hashable], Sequence[Hashable] | None, Sequence[Hashable] | None, bool
    ]:
        # 这个方法用于提取多重索引列的信息
    # 如果 header 的长度小于 2，直接返回 header[0]、index_names、None 和 passed_names
    # 用于处理 header 的列名为 MultiIndex 的情况，提取并返回列名、index_names 和 col_names
    def _maybe_make_multi_index_columns(
        self,
        columns: SequenceT,
        col_names: Sequence[Hashable] | None = None,
    ) -> SequenceT | MultiIndex:
        # 如果列名可能是 MultiIndex，则在这里创建 MultiIndex
        if is_potential_multi_index(columns):
            # 强制类型转换，假设 columns 是包含 tuple 的序列
            columns_mi = cast("Sequence[tuple[Hashable, ...]]", columns)
            # 使用 from_tuples 方法创建 MultiIndex 对象，指定列名为 col_names
            return MultiIndex.from_tuples(columns_mi, names=col_names)
        # 否则直接返回 columns
        return columns

    # 用于创建索引的函数，根据传入的数据、所有数据、列和索引名行创建索引
    @final
    def _make_index(
        self, data, alldata, columns, indexnamerow: list[Scalar] | None = None
    ):
    ) -> tuple[Index | None, Sequence[Hashable] | MultiIndex]:
        index: Index | None
        # 初始化 index 变量为 None
        if not is_index_col(self.index_col) or not self.index_col:
            # 如果 self.index_col 不是有效的索引列或者为空，则将 index 设置为 None
            index = None
        else:
            # 否则调用 _get_simple_index 方法获取简单索引
            simple_index = self._get_simple_index(alldata, columns)
            # 调用 _agg_index 方法处理简单索引并赋给 index
            index = self._agg_index(simple_index)

        # 为索引添加名称
        if indexnamerow:
            # 计算需要添加名称的偏移量
            coffset = len(indexnamerow) - len(columns)
            # 断言 index 不为 None
            assert index is not None
            # 设置索引的名称为 indexnamerow 的前 coffset 个元素
            index = index.set_names(indexnamerow[:coffset])

        # 可能创建多重索引的列
        columns = self._maybe_make_multi_index_columns(columns, self.col_names)

        # 返回 index 和 columns
        return index, columns

    @final
    def _get_simple_index(self, data, columns):
        def ix(col):
            # 如果 col 不是字符串，则直接返回
            if not isinstance(col, str):
                return col
            # 否则抛出值错误，指明无效的索引列
            raise ValueError(f"Index {col} invalid")

        to_remove = []
        index = []
        # 遍历 self.index_col 中的索引列
        for idx in self.index_col:
            # 调用 ix 方法处理索引列，获取对应的索引值 i
            i = ix(idx)
            # 将处理过的索引值加入待移除列表
            to_remove.append(i)
            # 将 data 中对应索引值的内容加入索引列表
            index.append(data[i])

        # 从数据和列中移除索引项，不在循环中使用 pop 方法
        for i in sorted(to_remove, reverse=True):
            # 从 data 中移除索引项 i
            data.pop(i)
            # 如果不是隐式索引，则从 columns 中移除对应列
            if not self._implicit_index:
                columns.pop(i)

        # 返回索引列表
        return index

    @final
    def _clean_mapping(self, mapping):
        """converts col numbers to names"""
        # 如果 mapping 不是字典类型，则直接返回 mapping
        if not isinstance(mapping, dict):
            return mapping
        clean = {}
        # 为了类型检查
        assert self.orig_names is not None

        # 遍历 mapping 中的键值对
        for col, v in mapping.items():
            # 如果 col 是整数且不在原始名称中，则将其转换为对应的原始名称
            if isinstance(col, int) and col not in self.orig_names:
                col = self.orig_names[col]
            # 将处理后的键值对加入 clean 字典中
            clean[col] = v

        # 如果 mapping 是 defaultdict 类型，则处理剩余的列
        if isinstance(mapping, defaultdict):
            # 计算剩余列的原始名称集合
            remaining_cols = set(self.orig_names) - set(clean.keys())
            # 将剩余列的映射加入 clean 字典中
            clean.update({col: mapping[col] for col in remaining_cols})

        # 返回处理后的映射结果 clean
        return clean

    @final
    # 定义一个方法 `_agg_index`，接受一个索引和一个布尔类型的参数 `try_parse_dates`，返回一个索引对象
    def _agg_index(self, index, try_parse_dates: bool = True) -> Index:
        # 初始化一个空数组 `arrays` 用来存储处理后的索引数据
        arrays = []
        # 从 `self.converters` 中清理并获取转换器信息
        converters = self._clean_mapping(self.converters)

        # 遍历索引中的每一个元素
        for i, arr in enumerate(index):
            # 如果 `try_parse_dates` 为真，并且应该解析日期（根据当前索引位置判断）
            if try_parse_dates and self._should_parse_dates(i):
                # 尝试对当前数组 `arr` 进行日期转换，传入列名（如果有的话）
                arr = self._date_conv(
                    arr,
                    col=self.index_names[i] if self.index_names is not None else None,
                )

            # 如果启用了 `na_filter` 过滤器
            if self.na_filter:
                # 获取当前列的 NA 值和 NA 填充值
                col_na_values = self.na_values
                col_na_fvalues = self.na_fvalues
            else:
                # 否则，设置当前列的 NA 值和 NA 填充值为空集合
                col_na_values = set()
                col_na_fvalues = set()

            # 如果 `na_values` 是一个字典
            if isinstance(self.na_values, dict):
                # 断言 `self.index_names` 不为 `None`
                assert self.index_names is not None
                # 获取当前列的名称
                col_name = self.index_names[i]
                if col_name is not None:
                    # 根据列名获取该列的 NA 值和 NA 填充值
                    col_na_values, col_na_fvalues = _get_na_values(
                        col_name, self.na_values, self.na_fvalues, self.keep_default_na
                    )
                else:
                    # 否则，将 NA 值和 NA 填充值设置为空集合
                    col_na_values, col_na_fvalues = set(), set()

            # 清理数据类型映射信息并获取
            clean_dtypes = self._clean_mapping(self.dtype)

            # 初始化变量 `cast_type` 和 `index_converter`
            cast_type = None
            index_converter = False
            # 如果存在索引名称
            if self.index_names is not None:
                # 如果 `clean_dtypes` 是一个字典，获取当前索引名称的数据类型转换信息
                if isinstance(clean_dtypes, dict):
                    cast_type = clean_dtypes.get(self.index_names[i], None)

                # 如果 `converters` 是一个字典，检查当前索引名称是否有对应的转换器
                if isinstance(converters, dict):
                    index_converter = converters.get(self.index_names[i]) is not None

            # 尝试对当前数组 `arr` 进行类型推断和转换
            try_num_bool = not (
                cast_type and is_string_dtype(cast_type) or index_converter
            )
            arr, _ = self._infer_types(
                arr, col_na_values | col_na_fvalues, cast_type is None, try_num_bool
            )
            # 将处理后的数组 `arr` 添加到 `arrays` 中
            arrays.append(arr)

        # 获取索引名称
        names = self.index_names
        # 根据处理后的数组 `arrays` 和索引名称 `names` 确保生成一个索引对象
        index = ensure_index_from_sequences(arrays, names)

        # 返回生成的索引对象
        return index

    # 定义一个修饰器方法 `_convert_to_ndarrays`，接受一组参数并返回处理后的对象
    @final
    def _convert_to_ndarrays(
        self,
        dct: Mapping,
        na_values,
        na_fvalues,
        converters=None,
        dtypes=None,
    ):
        # 略
        pass

    # 定义一个修饰器方法 `_set_noconvert_dtype_columns`，接受列索引列表和名称序列作为参数
    @final
    def _set_noconvert_dtype_columns(
        self, col_indices: list[int], names: Sequence[Hashable]
    ):
        # 略
        pass
    ) -> set[int]:
        """
        Set the columns that should not undergo dtype conversions.

        Currently, any column that is involved with date parsing will not
        undergo such conversions. If usecols is specified, the positions of the columns
        not to cast is relative to the usecols not to all columns.

        Parameters
        ----------
        col_indices: The indices specifying order and positions of the columns
        names: The column names which order is corresponding with the order
               of col_indices

        Returns
        -------
        A set of integers containing the positions of the columns not to convert.
        """
        # Define the possible types for the usecols variable
        usecols: list[int] | list[str] | None
        # Initialize an empty set to store column positions that should not be converted
        noconvert_columns = set()
        
        # Determine the appropriate usecols based on usecols_dtype attribute
        if self.usecols_dtype == "integer":
            # If usecols_dtype is "integer", convert set of integers to a sorted list
            usecols = sorted(self.usecols)
        elif callable(self.usecols) or self.usecols_dtype not in ("empty", None):
            # If usecols is callable or usecols_dtype is not "empty" or None, use col_indices
            usecols = col_indices
        else:
            # If usecols is empty, set it to None
            usecols = None

        # Define a helper function _set that resolves column indices or names to positions
        def _set(x) -> int:
            if usecols is not None and is_integer(x):
                # If usecols is defined and x is an integer, map x to corresponding usecols index
                x = usecols[x]

            if not is_integer(x):
                # If x is not an integer, map x to col_indices using column names
                x = col_indices[names.index(x)]

            return x

        # Process parse_dates attribute to determine columns not to convert
        if isinstance(self.parse_dates, list):
            # If parse_dates is a list, iterate over its values and add resolved positions to set
            for val in self.parse_dates:
                noconvert_columns.add(_set(val))
        elif self.parse_dates:
            # If parse_dates is not empty and not a list, process index_col accordingly
            if isinstance(self.index_col, list):
                for k in self.index_col:
                    noconvert_columns.add(_set(k))
            elif self.index_col is not None:
                noconvert_columns.add(_set(self.index_col))

        # Return the set of column positions that should not undergo dtype conversions
        return noconvert_columns

    @final
    def _infer_types(
        self, values, na_values, no_dtype_specified, try_num_bool: bool = True
    ) -> DataFrame:
        # Placeholder for inferring types, not fully specified in provided snippet

    @final
    @overload
    def _do_date_conversions(
        self,
        names: Index,
        data: DataFrame,
    ) -> DataFrame: ...
    
    @overload
    def _do_date_conversions(
        self,
        names: Sequence[Hashable],
        data: Mapping[Hashable, ArrayLike],
    ) -> Mapping[Hashable, ArrayLike]: ...

    @final
    def _do_date_conversions(
        self,
        names: Sequence[Hashable] | Index,
        data: Mapping[Hashable, ArrayLike] | DataFrame,
    ) -> Mapping[Hashable, ArrayLike] | DataFrame:
        # Perform date conversions based on parse_dates attribute and return processed data
        if isinstance(self.parse_dates, list):
            return _process_date_conversion(
                data,
                self._date_conv,
                self.parse_dates,
                self.index_col,
                self.index_names,
                names,
                dtype_backend=self.dtype_backend,
            )

        return data

    @final
    def _check_data_length(
        self,
        columns: Sequence[Hashable],
        data: Sequence[ArrayLike],
    ) -> None:
        """Checks if length of data is equal to length of column names.

        One set of trailing commas is allowed. self.index_col not False
        results in a ParserError previously when lengths do not match.

        Parameters
        ----------
        columns: list of column names
            The names of the columns expected in the data.
        data: list of array-likes containing the data column-wise.
            The actual data organized column-wise as array-likes.
        """
        if not self.index_col and len(columns) != len(data) and columns:
            empty_str = is_object_dtype(data[-1]) and data[-1] == ""
            # error: No overload variant of "__ror__" of "ndarray" matches
            # argument type "ExtensionArray"
            empty_str_or_na = empty_str | isna(data[-1])  # type: ignore[operator]
            if len(columns) == len(data) - 1 and np.all(empty_str_or_na):
                return
            warnings.warn(
                "Length of header or names does not match length of data. This leads "
                "to a loss of data with index_col=False.",
                ParserWarning,
                stacklevel=find_stack_level(),
            )

    @overload
    def _evaluate_usecols(
        self,
        usecols: Callable[[Hashable], object],
        names: Iterable[Hashable],
    ) -> set[int]: ...
    """Overload 1: Type signature for _evaluate_usecols method.

    Checks if 'usecols' is callable and returns a set of indices
    where 'usecols(name)' is True for each 'name' in 'names'.
    """
    
    @overload
    def _evaluate_usecols(
        self, usecols: SequenceT, names: Iterable[Hashable]
    ) -> SequenceT: ...
    """Overload 2: Type signature for _evaluate_usecols method.

    Returns 'usecols' as-is if it's not callable.
    """

    @final
    def _evaluate_usecols(
        self,
        usecols: Callable[[Hashable], object] | SequenceT,
        names: Iterable[Hashable],
    ) -> SequenceT | set[int]:
        """
        Check whether or not the 'usecols' parameter
        is a callable.  If so, enumerates the 'names'
        parameter and returns a set of indices for
        each entry in 'names' that evaluates to True.
        If not a callable, returns 'usecols'.
        """
        if callable(usecols):
            return {i for i, name in enumerate(names) if usecols(name)}
        return usecols
    def _validate_usecols_names(self, usecols: SequenceT, names: Sequence) -> SequenceT:
        """
        Validates that all usecols are present in a given
        list of names. If not, raise a ValueError that
        shows what usecols are missing.

        Parameters
        ----------
        usecols : iterable of usecols
            The columns to validate are present in names.
        names : iterable of names
            The column names to check against.

        Returns
        -------
        usecols : iterable of usecols
            The `usecols` parameter if the validation succeeds.

        Raises
        ------
        ValueError : Columns were missing. Error message will list them.
        """
        # Find all usecols that are not present in names
        missing = [c for c in usecols if c not in names]
        # If any usecols are missing, raise a ValueError
        if len(missing) > 0:
            raise ValueError(
                f"Usecols do not match columns, columns expected but not found: "
                f"{missing}"
            )

        return usecols

    @final
    def _validate_usecols_arg(self, usecols):
        """
        Validate the 'usecols' parameter.

        Checks whether or not the 'usecols' parameter contains all integers
        (column selection by index), strings (column by name) or is a callable.
        Raises a ValueError if that is not the case.

        Parameters
        ----------
        usecols : list-like, callable, or None
            List of columns to use when parsing or a callable that can be used
            to filter a list of table columns.

        Returns
        -------
        usecols_tuple : tuple
            A tuple of (verified_usecols, usecols_dtype).

            'verified_usecols' is either a set if an array-like is passed in or
            'usecols' if a callable or None is passed in.

            'usecols_dtype` is the inferred dtype of 'usecols' if an array-like
            is passed in or None if a callable or None is passed in.
        """
        msg = (
            "'usecols' must either be list-like of all strings, all unicode, "
            "all integers or a callable."
        )
        # Check if usecols is not None
        if usecols is not None:
            # If usecols is callable, return it directly
            if callable(usecols):
                return usecols, None

            # If usecols is not list-like, raise a ValueError
            if not is_list_like(usecols):
                # see gh-20529
                #
                # Ensure it is iterable container but not string.
                raise ValueError(msg)

            # Infer the dtype of usecols
            usecols_dtype = lib.infer_dtype(usecols, skipna=False)

            # If the inferred dtype is not valid, raise a ValueError
            if usecols_dtype not in ("empty", "integer", "string"):
                raise ValueError(msg)

            # Convert usecols to a set
            usecols = set(usecols)

            return usecols, usecols_dtype
        # If usecols is None, return it with None for usecols_dtype
        return usecols, None

    @final
    # 清理索引列名称，返回清理后的列名列表、索引列名列表和索引列数值列表
    def _clean_index_names(self, columns, index_col) -> tuple[list | None, list, list]:
        # 如果索引列不是有效的索引列格式，直接返回空值、列名列表和索引列列表
        if not is_index_col(index_col):
            return None, columns, index_col
        
        # 将列名转换为列表形式
        columns = list(columns)

        # 如果列名为空且存在多级索引列，将索引列名设置为多个 None 的列表，用于特殊情况处理
        if not columns:
            return [None] * len(index_col), columns, index_col

        # 复制一份列名列表，用于后续处理
        cp_cols = list(columns)
        # 初始化索引列名列表
        index_names: list[str | int | None] = []

        # 不改变原始索引列对象，创建一个索引列的副本
        index_col = list(index_col)

        # 遍历索引列和对应的列名，处理索引列名为字符串的情况
        for i, c in enumerate(index_col):
            if isinstance(c, str):
                # 将索引列名加入索引列名列表中
                index_names.append(c)
                # 遍历复制列名列表，找到匹配的列名并更新索引列数值为对应位置
                for j, name in enumerate(cp_cols):
                    if name == c:
                        index_col[i] = j
                        # 移除已匹配的列名
                        columns.remove(name)
                        break
            else:
                # 对于索引列名为数值的情况，直接使用数值对应的列名
                name = cp_cols[c]
                columns.remove(name)
                index_names.append(name)

        # 只清理作为占位符的索引列名
        for i, name in enumerate(index_names):
            if isinstance(name, str) and name in self.unnamed_cols:
                index_names[i] = None

        # 返回处理后的索引列名列表、列名列表和索引列数值列表
        return index_names, columns, index_col

    # 定义一个最终版本方法，获取空的元数据
    @final
    def _get_empty_meta(
        self, columns: Sequence[HashableT], dtype: DtypeArg | None = None
        ) -> tuple[Index, list[HashableT], dict[HashableT, Series]]:
        # 将输入的列转换为列表形式
        columns = list(columns)

        # 获取索引列和索引名称
        index_col = self.index_col
        index_names = self.index_names

        # 将 `dtype` 转换为某种默认字典。
        # 这样可以让我们在后续写 `dtype[col_name]` 时，不必担心 KeyError 问题。
        dtype_dict: defaultdict[Hashable, Any]
        if not is_dict_like(dtype):
            # 如果 dtype == None，则默认为 object 类型。
            default_dtype = dtype or object
            dtype_dict = defaultdict(lambda: default_dtype)
        else:
            dtype = cast(dict, dtype)
            dtype_dict = defaultdict(
                lambda: object,
                {columns[k] if is_integer(k) else k: v for k, v in dtype.items()},
            )

        # 即使我们没有数据，空 DataFrame 的“索引”
        # 例如仍然可以是空的 MultiIndex。因此，我们需要检查是否有指定的索引列，可以通过以下方式：
        #
        # 1）index_col（列索引）
        # 2）index_names（列名称）
        #
        # 两者必须都非空才能确保成功构建。否则，我们需要创建一个通用的空 Index。
        index: Index
        if (index_col is None or index_col is False) or index_names is None:
            index = default_index(0)
        else:
            # 创建空数据列表以便于构建索引
            data = [Series([], dtype=dtype_dict[name]) for name in index_names]
            index = ensure_index_from_sequences(data, names=index_names)
            index_col.sort()

            # 根据索引列从 `columns` 中移除相应的列名
            for i, n in enumerate(index_col):
                columns.pop(n - i)

        # 创建列名到空 Series 的映射字典
        col_dict = {
            col_name: Series([], dtype=dtype_dict[col_name]) for col_name in columns
        }

        # 返回结果：索引，列名列表，以及列名到空 Series 的映射字典
        return index, columns, col_dict
# 创建一个日期转换器函数，用于将数据列转换为日期时间类型
def _make_date_converter(
    dayfirst: bool = False,  # 是否优先使用日期中的天数作为第一项
    cache_dates: bool = True,  # 是否缓存解析过的日期
    date_format: dict[Hashable, str] | str | None = None,  # 日期格式的字典或字符串，用于解析日期字符串
):
    # 实际的日期转换函数，接受数据列和列名作为输入
    def converter(date_col, col: Hashable):
        # 如果数据列的数据类型已经是日期时间类型，则直接返回该列
        if date_col.dtype.kind in "Mm":
            return date_col

        # 获取特定列的日期格式
        date_fmt = (
            date_format.get(col) if isinstance(date_format, dict) else date_format
        )

        # 将数据列转换为字符串数组
        str_objs = lib.ensure_string_array(date_col)
        
        try:
            # 使用指定的日期格式尝试将字符串数组转换为日期时间索引
            result = tools.to_datetime(
                str_objs,
                format=date_fmt,
                utc=False,
                dayfirst=dayfirst,
                cache=cache_dates,
            )
        except (ValueError, TypeError):
            # 如果转换失败，则返回原始的字符串数组
            return str_objs

        # 如果转换结果是一个日期时间索引对象，则将其转换为可写的 NumPy 数组并返回
        if isinstance(result, DatetimeIndex):
            arr = result.to_numpy()
            arr.flags.writeable = True
            return arr
        
        # 否则，返回转换结果的值部分
        return result._values

    # 返回日期转换函数
    return converter


# CSV 解析器的默认选项字典
parser_defaults = {
    "delimiter": None,
    "escapechar": None,
    "quotechar": '"',
    "quoting": csv.QUOTE_MINIMAL,
    "doublequote": True,
    "skipinitialspace": False,
    "lineterminator": None,
    "header": "infer",
    "index_col": None,
    "names": None,
    "skiprows": None,
    "skipfooter": 0,
    "nrows": None,
    "na_values": None,
    "keep_default_na": True,
    "true_values": None,
    "false_values": None,
    "converters": None,
    "dtype": None,
    "cache_dates": True,  # 是否缓存解析过的日期
    "thousands": None,
    "comment": None,
    "decimal": ".",
    # 'engine': 'c',
    "parse_dates": False,  # 是否尝试解析日期
    "dayfirst": False,  # 是否优先使用日期中的天数作为第一项
    "date_format": None,  # 日期格式字符串，用于解析日期
    "usecols": None,
    # 'iterator': False,
    "chunksize": None,
    "encoding": None,
    "compression": None,
    "skip_blank_lines": True,
    "encoding_errors": "strict",
    "on_bad_lines": ParserBase.BadLineHandleMethod.ERROR,
    "dtype_backend": lib.no_default,
}


# 处理日期转换的函数，接受数据字典或 DataFrame、日期转换器函数、解析规格列表、索引列、索引名称、列名列表或索引
# 并返回转换后的数据字典或 DataFrame
def _process_date_conversion(
    data_dict: Mapping[Hashable, ArrayLike] | DataFrame,
    converter: Callable,  # 日期转换器函数
    parse_spec: list,  # 解析规格列表
    index_col,
    index_names,
    columns: Sequence[Hashable] | Index,
    dtype_backend=lib.no_default,  # 数据类型后端
) -> Mapping[Hashable, ArrayLike] | DataFrame:
    # 遍历解析规范中的每个列规范
    for colspec in parse_spec:
        # 如果列规范是整数且不在数据字典中，则将其转换为对应的列名
        if isinstance(colspec, int) and colspec not in data_dict:
            colspec = columns[colspec]
        
        # 如果列规范在索引列列表中或索引名称列表中，则跳过当前循环
        if (isinstance(index_col, list) and colspec in index_col) or (
            isinstance(index_names, list) and colspec in index_names
        ):
            continue
        # 如果后端数据类型为"pyarrow"
        elif dtype_backend == "pyarrow":
            import pyarrow as pa

            # 获取当前列数据的类型
            dtype = data_dict[colspec].dtype
            # 如果数据类型是ArrowDtype，并且是时间戳或日期类型，则跳过当前循环
            if isinstance(dtype, ArrowDtype) and (
                pa.types.is_timestamp(dtype.pyarrow_dtype)
                or pa.types.is_date(dtype.pyarrow_dtype)
            ):
                continue

        # 对于使用Pyarrow引擎返回的Series，需要将其转换为numpy数组，其他解析器则不需要
        result = converter(np.asarray(data_dict[colspec]), col=colspec)
        # 错误：不支持对索引赋值目标
        # ("Mapping[Hashable, ExtensionArray | ndarray[Any, Any]] | DataFrame")
        # 将转换后的结果赋值给数据字典中对应的列规范
        data_dict[colspec] = result  # type: ignore[index]

    # 返回处理后的数据字典
    return data_dict
# 判断并获取给定列的 NaN 值

def _get_na_values(col, na_values, na_fvalues, keep_default_na: bool):
    """
    Get the NaN values for a given column.

    Parameters
    ----------
    col : str
        The name of the column.
    na_values : array-like, dict
        The object listing the NaN values as strings.
    na_fvalues : array-like, dict
        The object listing the NaN values as floats.
    keep_default_na : bool
        If `na_values` is a dict, and the column is not mapped in the
        dictionary, whether to return the default NaN values or the empty set.

    Returns
    -------
    nan_tuple : A length-two tuple composed of

        1) na_values : the string NaN values for that column.
        2) na_fvalues : the float NaN values for that column.
    """
    # 如果 na_values 是一个字典
    if isinstance(na_values, dict):
        # 如果列名 col 在 na_values 字典中
        if col in na_values:
            # 返回该列对应的字符串类型的 NaN 值和浮点类型的 NaN 值
            return na_values[col], na_fvalues[col]
        else:
            # 如果 keep_default_na 为真，则返回默认的字符串 NaN 值和空集合
            if keep_default_na:
                return STR_NA_VALUES, set()

            # 否则返回空集合
            return set(), set()
    else:
        # 如果 na_values 不是字典，则直接返回 na_values 和 na_fvalues
        return na_values, na_fvalues


# 判断给定的列是否为索引列，并返回布尔值
def is_index_col(col) -> bool:
    return col is not None and col is not False
```
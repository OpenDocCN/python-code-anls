# `D:\src\scipysrc\pandas\pandas\_testing\__init__.py`

```
# 导入必要的模块和类型声明
from __future__ import annotations  # 导入未来的类型注解支持

from decimal import Decimal  # 导入 Decimal 类型
import operator  # 导入 operator 模块
import os  # 导入 os 模块
from sys import byteorder  # 从 sys 模块导入 byteorder 函数
from typing import (  # 导入类型提示相关的模块和类型
    TYPE_CHECKING,  # 类型检查标志
    ContextManager,  # 上下文管理器
    cast,  # 类型转换函数
)

import numpy as np  # 导入 NumPy 库并重命名为 np

from pandas._config.localization import (  # 导入 pandas 内部的 localization 模块中的相关函数
    can_set_locale,  # 检查是否可以设置区域设置
    get_locales,  # 获取可用的区域设置
    set_locale,  # 设置区域设置
)

from pandas.compat import pa_version_under10p1  # 导入 pandas 兼容模块中的 pa_version_under10p1 函数

from pandas.core.dtypes.common import is_string_dtype  # 导入 pandas 核心模块中的判断是否为字符串类型的函数

import pandas as pd  # 导入 pandas 库并重命名为 pd
from pandas import (  # 从 pandas 中导入多个子模块和类
    ArrowDtype,  # ArrowDtype 类型
    DataFrame,  # DataFrame 类
    Index,  # Index 类
    MultiIndex,  # MultiIndex 类
    RangeIndex,  # RangeIndex 类
    Series,  # Series 类
)
from pandas._testing._io import (  # 导入 pandas 测试模块中的 IO 相关函数
    round_trip_pathlib,  # 使用 pathlib 进行往返测试
    round_trip_pickle,  # pickle 格式往返测试
    write_to_compressed,  # 写入压缩文件进行测试
)
from pandas._testing._warnings import (  # 导入 pandas 测试模块中的警告相关函数
    assert_produces_warning,  # 断言是否产生警告
    maybe_produces_warning,  # 可能产生警告的情况
)
from pandas._testing.asserters import (  # 导入 pandas 测试模块中的断言函数
    assert_almost_equal,  # 断言几乎相等
    assert_attr_equal,  # 断言属性相等
    assert_categorical_equal,  # 断言分类数据相等
    assert_class_equal,  # 断言类相等
    assert_contains_all,  # 断言包含所有
    assert_copy,  # 断言复制操作
    assert_datetime_array_equal,  # 断言日期时间数组相等
    assert_dict_equal,  # 断言字典相等
    assert_equal,  # 断言相等
    assert_extension_array_equal,  # 断言扩展数组相等
    assert_frame_equal,  # 断言 DataFrame 相等
    assert_index_equal,  # 断言索引相等
    assert_indexing_slices_equivalent,  # 断言索引切片等价
    assert_interval_array_equal,  # 断言区间数组相等
    assert_is_sorted,  # 断言已排序
    assert_metadata_equivalent,  # 断言元数据等价
    assert_numpy_array_equal,  # 断言 NumPy 数组相等
    assert_period_array_equal,  # 断言周期数组相等
    assert_series_equal,  # 断言 Series 相等
    assert_sp_array_equal,  # 断言 Sparse 数组相等
    assert_timedelta_array_equal,  # 断言时间差数组相等
    raise_assert_detail,  # 抛出详细的断言错误
)
from pandas._testing.compat import (  # 导入 pandas 测试兼容模块中的函数
    get_dtype,  # 获取数据类型
    get_obj,  # 获取对象
)
from pandas._testing.contexts import (  # 导入 pandas 测试上下文管理相关函数
    decompress_file,  # 解压文件
    ensure_clean,  # 确保清洁状态
    raises_chained_assignment_error,  # 引发链式赋值错误
    set_timezone,  # 设置时区
    with_csv_dialect,  # 使用 CSV 方言
)
from pandas.core.arrays import (  # 导入 pandas 核心数组模块中的相关类
    BaseMaskedArray,  # 基础遮盖数组
    ExtensionArray,  # 扩展数组
    NumpyExtensionArray,  # NumPy 扩展数组
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray  # 导入 pandas 核心数组混合模块中的 NDArrayBackedExtensionArray 类
from pandas.core.construction import extract_array  # 导入 pandas 核心构造模块中的 extract_array 函数

if TYPE_CHECKING:
    from collections.abc import Callable  # 导入 Callable 类型

    from pandas._typing import (  # 导入 pandas 类型提示模块中的类型
        Dtype,  # 数据类型
        NpDtype,  # NumPy 数据类型
    )

    from pandas.core.arrays import ArrowExtensionArray  # 导入 pandas 核心数组中的 ArrowExtensionArray 类

# 定义常量列表
UNSIGNED_INT_NUMPY_DTYPES: list[NpDtype] = ["uint8", "uint16", "uint32", "uint64"]  # 无符号整数 NumPy 数据类型列表
UNSIGNED_INT_EA_DTYPES: list[Dtype] = ["UInt8", "UInt16", "UInt32", "UInt64"]  # 无符号整数扩展数组数据类型列表
SIGNED_INT_NUMPY_DTYPES: list[NpDtype] = [int, "int8", "int16", "int32", "int64"]  # 有符号整数 NumPy 数据类型列表
SIGNED_INT_EA_DTYPES: list[Dtype] = ["Int8", "Int16", "Int32", "Int64"]  # 有符号整数扩展数组数据类型列表
ALL_INT_NUMPY_DTYPES = UNSIGNED_INT_NUMPY_DTYPES + SIGNED_INT_NUMPY_DTYPES  # 所有整数 NumPy 数据类型列表
ALL_INT_EA_DTYPES = UNSIGNED_INT_EA_DTYPES + SIGNED_INT_EA_DTYPES  # 所有整数扩展数组数据类型列表
ALL_INT_DTYPES: list[Dtype] = [*ALL_INT_NUMPY_DTYPES, *ALL_INT_EA_DTYPES]  # 所有整数数据类型列表

FLOAT_NUMPY_DTYPES: list[NpDtype] = [float, "float32", "float64"]  # 浮点数 NumPy 数据类型列表
FLOAT_EA_DTYPES: list[Dtype] = ["Float32", "Float64"]  # 浮点数扩展数组数据类型列表
ALL_FLOAT_DTYPES: list[Dtype] = [*FLOAT_NUMPY_DTYPES, *FLOAT_EA_DTYPES]  # 所有浮点数数据类型列表

COMPLEX_DTYPES: list[Dtype] = [complex, "complex64", "complex128"]  # 复数数据类型列表
STRING_DTYPES: list[Dtype] = [str, "str", "U"]  # 字符串数据类型列表

DATETIME64_DTYPES: list[Dtype] = ["datetime64[ns]", "M8[ns]"]  # 日期时间数据类型列表
TIMEDELTA64_DTYPES: list[Dtype] = ["timedelta64[ns]", "m8[ns]"]  # 时间差数据类型列表
BOOL_DTYPES: list[Dtype] = [bool, "bool"]
BYTES_DTYPES: list[Dtype] = [bytes, "bytes"]
OBJECT_DTYPES: list[Dtype] = [object, "object"]

ALL_REAL_NUMPY_DTYPES = FLOAT_NUMPY_DTYPES + ALL_INT_NUMPY_DTYPES
ALL_REAL_EXTENSION_DTYPES = FLOAT_EA_DTYPES + ALL_INT_EA_DTYPES
ALL_REAL_DTYPES: list[Dtype] = [*ALL_REAL_NUMPY_DTYPES, *ALL_REAL_EXTENSION_DTYPES]
ALL_NUMERIC_DTYPES: list[Dtype] = [*ALL_REAL_DTYPES, *COMPLEX_DTYPES]

ALL_NUMPY_DTYPES = (
    ALL_REAL_NUMPY_DTYPES
    + COMPLEX_DTYPES
    + STRING_DTYPES
    + DATETIME64_DTYPES
    + TIMEDELTA64_DTYPES
    + BOOL_DTYPES
    + OBJECT_DTYPES
    + BYTES_DTYPES
)

NARROW_NP_DTYPES = [
    np.float16,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.uint8,
    np.uint16,
    np.uint32,
]

PYTHON_DATA_TYPES = [
    str,
    int,
    float,
    complex,
    list,
    tuple,
    range,
    dict,
    set,
    frozenset,
    bool,
    bytes,
    bytearray,
    memoryview,
]

ENDIAN = {"little": "<", "big": ">"}[byteorder]

NULL_OBJECTS = [None, np.nan, pd.NaT, float("nan"), pd.NA, Decimal("NaN")]
NP_NAT_OBJECTS = [
    cls("NaT", unit)
    for cls in [np.datetime64, np.timedelta64]
    for unit in [
        "Y",
        "M",
        "W",
        "D",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",
        "ps",
        "fs",
        "as",
    ]
]

# 如果不是 pyarrow 版本小于 1.0.1，则导入 pyarrow 库
if not pa_version_under10p1:
    import pyarrow as pa

    UNSIGNED_INT_PYARROW_DTYPES = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
    SIGNED_INT_PYARROW_DTYPES = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]
    ALL_INT_PYARROW_DTYPES = UNSIGNED_INT_PYARROW_DTYPES + SIGNED_INT_PYARROW_DTYPES
    ALL_INT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in ALL_INT_PYARROW_DTYPES
    ]

    FLOAT_PYARROW_DTYPES = [pa.float32(), pa.float64()]
    FLOAT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in FLOAT_PYARROW_DTYPES
    ]
    DECIMAL_PYARROW_DTYPES = [pa.decimal128(7, 3)]
    STRING_PYARROW_DTYPES = [pa.string()]
    BINARY_PYARROW_DTYPES = [pa.binary()]

    TIME_PYARROW_DTYPES = [
        pa.time32("s"),
        pa.time32("ms"),
        pa.time64("us"),
        pa.time64("ns"),
    ]
    DATE_PYARROW_DTYPES = [pa.date32(), pa.date64()]
    DATETIME_PYARROW_DTYPES = [
        pa.timestamp(unit=unit, tz=tz)
        for unit in ["s", "ms", "us", "ns"]
        for tz in [None, "UTC", "US/Pacific", "US/Eastern"]
    ]
    TIMEDELTA_PYARROW_DTYPES = [pa.duration(unit) for unit in ["s", "ms", "us", "ns"]]

    BOOL_PYARROW_DTYPES = [pa.bool_()]

    # TODO: Add container like pyarrow types:
    #  https://arrow.apache.org/docs/python/api/datatypes.html#factory-functions


注释：

BOOL_DTYPES: list[Dtype] = [bool, "bool"]  # 布尔类型数据类型列表，包含原生 bool 和字符串 "bool"
BYTES_DTYPES: list[Dtype] = [bytes, "bytes"]  # 字节类型数据类型列表，包含原生 bytes 和字符串 "bytes"
OBJECT_DTYPES: list[Dtype] = [object, "object"]  # 对象类型数据类型列表，包含原生 object 和字符串 "object"

ALL_REAL_NUMPY_DTYPES = FLOAT_NUMPY_DTYPES + ALL_INT_NUMPY_DTYPES  # 所有实数类型的 NumPy 数据类型列表
ALL_REAL_EXTENSION_DTYPES = FLOAT_EA_DTYPES + ALL_INT_EA_DTYPES  # 所有实数扩展类型的数据类型列表
ALL_REAL_DTYPES: list[Dtype] = [*ALL_REAL_NUMPY_DTYPES, *ALL_REAL_EXTENSION_DTYPES]  # 所有实数数据类型的总列表
ALL_NUMERIC_DTYPES: list[Dtype] = [*ALL_REAL_DTYPES, *COMPLEX_DTYPES]  # 所有数值数据类型的总列表，包括复数类型

ALL_NUMPY_DTYPES = (
    ALL_REAL_NUMPY_DTYPES
    + COMPLEX_DTYPES
    + STRING_DTYPES
    + DATETIME64_DTYPES
    + TIMEDELTA64_DTYPES
    + BOOL_DTYPES
    + OBJECT_DTYPES
    + BYTES_DTYPES
)  # 所有 NumPy 支持的数据类型的总列表

NARROW_NP_DTYPES = [
    np.float16,
    np.float32,
    np.int8,
    np.int16,
    np.int32,
    np.uint8,
    np.uint16,
    np.uint32,
]  # NumPy 中的窄数据类型列表

PYTHON_DATA_TYPES = [
    str,
    int,
    float,
    complex,
    list,
    tuple,
    range,
    dict,
    set,
    frozenset,
    bool,
    bytes,
    bytearray,
    memoryview,
]  # Python 中的内置数据类型列表

ENDIAN = {"little": "<", "big": ">"}[byteorder]  # 根据字节顺序选择相应的小端或大端符号

NULL_OBJECTS = [None, np.nan, pd.NaT, float("nan"), pd.NA, Decimal("NaN")]  # 表示空值的对象列表
NP_NAT_OBJECTS = [
    cls("NaT", unit)
    for cls in [np.datetime64, np.timedelta64]
    for unit in [
        "Y",
        "M",
        "W",
        "D",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",
        "ps",
        "fs",
        "as",
    ]
]  # 表示 NumPy 中 NaT (Not a Time) 对象的列表

# 如果不是 pyarrow 版本小于 1.0.1，则导入 pyarrow 库
if not pa_version_under10p1:
    import pyarrow as pa

    UNSIGNED_INT_PYARROW_DTYPES = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]  # 无符号整数 PyArrow 数据类型列表
    SIGNED_INT_PYARROW_DTYPES = [pa.int8(), pa.int16(), pa.int32(), pa.int64()]  # 有符号整数 PyArrow 数据类型列表
    ALL_INT_PYARROW_DTYPES = UNSIGNED_INT_PYARROW_DTYPES + SIGNED_INT_PYARROW_DTYPES  # 所有整数 PyArrow 数据类型的总列表
    ALL_INT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in ALL_INT_PYARROW_DTYPES
    ]  # 所有整数 PyArrow 数据类型的字符串表示列表

    FLOAT_PYARROW_DTYPES = [pa.float32(), pa.float64()]  # 浮点数 PyArrow 数据类型列表
    FLOAT_PYARROW_DTYPES_STR_REPR = [
        str(ArrowDtype(typ)) for typ in FLOAT_PYARROW_DTYPES
    ]  # 浮点数 PyArrow 数据类型的字符串表示列表
    DECIMAL_PYARROW_DTYPES = [pa.decimal128(7, 3)]  # 十进制数 PyArrow 数据类型列表
    STRING_PYARROW_DTYPES = [pa.string()]  # 字符串 PyArrow 数据类型列表
    BINARY_PYARROW_DTYPES = [pa.binary()]  # 二进制 PyArrow 数据类型列表

    TIME_PYARROW_DTYPES = [
        pa.time32("s"),
        pa.time32("ms"),
        pa.time64("us"),
        pa.time64("ns"),
    ]  # 时间 PyArrow 数据类型列表
    DATE_PYARROW_DTYPES = [pa.date32(), pa.date64()]  # 日期 PyArrow 数据类型列表
    DATETIME_PYARROW_DTYPES = [
        pa.timestamp(unit=unit, tz=tz)
        for unit in ["s", "ms", "us", "ns"]
        for tz in [None, "UTC", "US/Pacific", "US/Eastern"]
    ]  # 日期时间 PyArrow 数据类型列表
    TIMEDELTA_PYARROW_DT
    # 定义包含所有 PyArrow 支持的数据类型的元组，包括整数、浮点数、小数、字符串、二进制等
    ALL_PYARROW_DTYPES = (
        ALL_INT_PYARROW_DTYPES          # 所有整数类型的 PyArrow 数据类型
        + FLOAT_PYARROW_DTYPES           # 所有浮点数类型的 PyArrow 数据类型
        + DECIMAL_PYARROW_DTYPES         # 所有小数类型的 PyArrow 数据类型
        + STRING_PYARROW_DTYPES          # 所有字符串类型的 PyArrow 数据类型
        + BINARY_PYARROW_DTYPES          # 所有二进制类型的 PyArrow 数据类型
        + TIME_PYARROW_DTYPES            # 所有时间类型的 PyArrow 数据类型
        + DATE_PYARROW_DTYPES            # 所有日期类型的 PyArrow 数据类型
        + DATETIME_PYARROW_DTYPES        # 所有日期时间类型的 PyArrow 数据类型
        + TIMEDELTA_PYARROW_DTYPES       # 所有时间间隔类型的 PyArrow 数据类型
        + BOOL_PYARROW_DTYPES            # 所有布尔类型的 PyArrow 数据类型
    )
    
    # 定义所有真实数据类型的 PyArrow 数据类型的字符串表示形式的集合，包括整数和浮点数
    ALL_REAL_PYARROW_DTYPES_STR_REPR = (
        ALL_INT_PYARROW_DTYPES_STR_REPR   # 所有整数类型的 PyArrow 数据类型的字符串表示
        + FLOAT_PYARROW_DTYPES_STR_REPR   # 所有浮点数类型的 PyArrow 数据类型的字符串表示
    )
else:
    # 初始化空列表，用于存储 FLOAT_PYARROW_DTYPES_STR_REPR
    FLOAT_PYARROW_DTYPES_STR_REPR = []
    # 初始化空列表，用于存储 ALL_INT_PYARROW_DTYPES_STR_REPR
    ALL_INT_PYARROW_DTYPES_STR_REPR = []
    # 初始化空列表，用于存储 ALL_PYARROW_DTYPES
    ALL_PYARROW_DTYPES = []
    # 初始化空列表，用于存储 ALL_REAL_PYARROW_DTYPES_STR_REPR
    ALL_REAL_PYARROW_DTYPES_STR_REPR = []

# 创建 ALL_REAL_NULLABLE_DTYPES 列表，包含 FLOAT_NUMPY_DTYPES、ALL_REAL_EXTENSION_DTYPES 和 ALL_REAL_PYARROW_DTYPES_STR_REPR
ALL_REAL_NULLABLE_DTYPES = (
    FLOAT_NUMPY_DTYPES + ALL_REAL_EXTENSION_DTYPES + ALL_REAL_PYARROW_DTYPES_STR_REPR
)

# 定义 arithmetic_dunder_methods 列表，包含各种算术运算的魔术方法名称
arithmetic_dunder_methods = [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__floordiv__",
    "__rfloordiv__",
    "__truediv__",
    "__rtruediv__",
    "__pow__",
    "__rpow__",
    "__mod__",
    "__rmod__",
]

# 定义 comparison_dunder_methods 列表，包含各种比较运算的魔术方法名称
comparison_dunder_methods = ["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"]


# -----------------------------------------------------------------------------
# Comparators


def box_expected(expected, box_cls, transpose: bool = True):
    """
    Helper function to wrap the expected output of a test in a given box_class.

    Parameters
    ----------
    expected : np.ndarray, Index, Series
        The expected output of the test.
    box_cls : {Index, Series, DataFrame}
        The class to wrap the expected output in.
    transpose : bool, optional
        Whether to transpose the DataFrame if box_cls is DataFrame. Default is True.

    Returns
    -------
    subclass of box_cls
        The wrapped expected output in the specified box_cls subclass.
    """
    if box_cls is pd.array:
        # 如果 box_cls 是 pd.array，将 expected 转换为适当的类型
        if isinstance(expected, RangeIndex):
            expected = NumpyExtensionArray(np.asarray(expected._values))
        else:
            expected = pd.array(expected, copy=False)
    elif box_cls is Index:
        # 如果 box_cls 是 Index，创建 Index 类型的对象
        expected = Index(expected)
    elif box_cls is Series:
        # 如果 box_cls 是 Series，创建 Series 类型的对象
        expected = Series(expected)
    elif box_cls is DataFrame:
        # 如果 box_cls 是 DataFrame，将 expected 转换为 Series，并转换为单行或者两行 DataFrame
        expected = Series(expected).to_frame()
        if transpose:
            # 如果需要转置，确保 DataFrame 是单行的
            expected = expected.T
            expected = pd.concat([expected] * 2, ignore_index=True)
    elif box_cls is np.ndarray or box_cls is np.array:
        # 如果 box_cls 是 np.ndarray 或者 np.array，将 expected 转换为 numpy 数组
        expected = np.array(expected)
    elif box_cls is to_array:
        # 如果 box_cls 是 to_array，使用 to_array 函数处理 expected
        expected = to_array(expected)
    else:
        # 如果 box_cls 是其他未实现的类型，抛出 NotImplementedError 异常
        raise NotImplementedError(box_cls)
    return expected


def to_array(obj):
    """
    Similar to pd.array, but does not cast numpy dtypes to nullable dtypes.

    Parameters
    ----------
    obj : object
        The object to convert to array.

    Returns
    -------
    np.ndarray
        The converted numpy array.
    """
    # 获取对象的 dtype
    dtype = getattr(obj, "dtype", None)

    if dtype is None:
        # 如果 dtype 为 None，直接将 obj 转换为 numpy 数组
        return np.asarray(obj)

    # 否则，使用 extract_array 函数提取数组，转换为 numpy 数组
    return extract_array(obj, extract_numpy=True)


class SubclassedSeries(Series):
    """
    Subclass of pandas Series with additional metadata.

    Attributes
    ----------
    _metadata : list
        List of additional metadata attributes.
    """

    _metadata = ["testattr", "name"]

    @property
    # 定义私有方法 `_constructor`，返回一个 lambda 函数，用于生成 SubclassedSeries 实例
    def _constructor(self):
        # 用于测试，这些属性返回一个通用的可调用对象，而不是实际的类。
        # 在这种情况下，它是等效的，但是这样做是为了确保我们不依赖于属性返回一个类。
        # 参见 https://github.com/pandas-dev/pandas/pull/46018 和
        # https://github.com/pandas-dev/pandas/issues/32638 以及相关问题
        return lambda *args, **kwargs: SubclassedSeries(*args, **kwargs)

    # 定义 `_constructor_expanddim` 属性，返回一个 lambda 函数，用于生成 SubclassedDataFrame 实例
    @property
    def _constructor_expanddim(self):
        return lambda *args, **kwargs: SubclassedDataFrame(*args, **kwargs)
class SubclassedDataFrame(DataFrame):
    # 定义一个子类化的 DataFrame，设定其特有的元数据属性
    _metadata = ["testattr"]

    @property
    def _constructor(self):
        # 返回一个 lambda 表达式作为构造函数，创建 SubclassedDataFrame 实例
        return lambda *args, **kwargs: SubclassedDataFrame(*args, **kwargs)

    @property
    def _constructor_sliced(self):
        # 返回一个 lambda 表达式作为切片构造函数，创建 SubclassedSeries 实例
        return lambda *args, **kwargs: SubclassedSeries(*args, **kwargs)


def convert_rows_list_to_csv_str(rows_list: list[str]) -> str:
    """
    Convert list of CSV rows to single CSV-formatted string for current OS.

    This method is used for creating expected value of to_csv() method.

    Parameters
    ----------
    rows_list : List[str]
        Each element represents the row of csv.

    Returns
    -------
    str
        Expected output of to_csv() in current OS.
    """
    # 获取当前操作系统的换行符
    sep = os.linesep
    # 将行列表转换为单个 CSV 格式的字符串，并加上换行符
    return sep.join(rows_list) + sep


def external_error_raised(expected_exception: type[Exception]) -> ContextManager:
    """
    Helper function to mark pytest.raises that have an external error message.

    Parameters
    ----------
    expected_exception : Exception
        Expected error to raise.

    Returns
    -------
    Callable
        Regular `pytest.raises` function with `match` equal to `None`.
    """
    # 导入 pytest 模块并返回一个带有预期异常的 pytest.raises 上下文管理器
    import pytest
    return pytest.raises(expected_exception, match=None)


def get_cython_table_params(ndframe, func_names_and_expected):
    """
    Combine frame, functions from com._cython_table
    keys and expected result.

    Parameters
    ----------
    ndframe : DataFrame or Series
    func_names_and_expected : Sequence of two items
        The first item is a name of a NDFrame method ('sum', 'prod') etc.
        The second item is the expected return value.

    Returns
    -------
    list
        List of three items (DataFrame, function, expected result)
    """
    # 根据给定的函数名称和预期结果，返回一个由 DataFrame, 函数, 预期结果 组成的列表
    results = []
    for func_name, expected in func_names_and_expected:
        results.append((ndframe, func_name, expected))
    return results


def get_op_from_name(op_name: str) -> Callable:
    """
    The operator function for a given op name.

    Parameters
    ----------
    op_name : str
        The op name, in form of "add" or "__add__".

    Returns
    -------
    function
        A function performing the operation.
    """
    # 基于操作符名称返回相应的操作函数
    short_opname = op_name.strip("_")
    try:
        op = getattr(operator, short_opname)
    except AttributeError:
        # 如果找不到正常的操作符，假设是反向操作符
        rop = getattr(operator, short_opname[1:])
        op = lambda x, y: rop(y, x)

    return op


# -----------------------------------------------------------------------------
# Indexing test helpers


def getitem(x):
    # 返回传入参数 x 本身
    return x


def setitem(x):
    # 返回传入参数 x 本身
    return x


def loc(x):
    # 返回传入参数 x 的 loc 属性
    return x.loc


def iloc(x):
    # 返回传入参数 x 的 iloc 属性
    return x.iloc


def at(x):
    # 返回传入参数 x 的 at 属性
    return x.at


def iat(x):
    # 返回传入参数 x 的 iat 属性
    return x.iat


# -----------------------------------------------------------------------------

_UNITS = ["s", "ms", "us", "ns"]


def get_finest_unit(left: str, right: str) -> str:
    """
    Find the higher of two datetime64 units.
    """
    # 返回左右两个 datetime64 单位中更高级别的单位
    # 检查在 _UNITS 列表中 left 的索引是否大于等于 right 的索引
    if _UNITS.index(left) >= _UNITS.index(right):
        # 如果是，则返回 left
        return left
    # 否则返回 right
    return right
def shares_memory(left, right) -> bool:
    """
    Pandas-compat for np.shares_memory.
    """
    # 如果 left 和 right 都是 numpy 数组，则调用 np.shares_memory 检查它们是否共享内存
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.shares_memory(left, right)
    
    # 如果 left 是 numpy 数组，逆向调用 shares_memory 函数以处理下面的拆包逻辑
    elif isinstance(left, np.ndarray):
        return shares_memory(right, left)

    # 如果 left 是 RangeIndex，则返回 False，因为 RangeIndex 不共享内存
    if isinstance(left, RangeIndex):
        return False
    
    # 如果 left 是 MultiIndex，则递归调用 shares_memory 来检查 left._codes 是否与 right 共享内存
    if isinstance(left, MultiIndex):
        return shares_memory(left._codes, right)
    
    # 如果 left 是 Index 或者 Series，则递归调用 shares_memory 来检查 left._values 是否与 right 共享内存
    if isinstance(left, (Index, Series)):
        return shares_memory(left._values, right)
    
    # 如果 left 是 NDArrayBackedExtensionArray，则检查 left._ndarray 是否与 right 共享内存
    if isinstance(left, NDArrayBackedExtensionArray):
        return shares_memory(left._ndarray, right)
    
    # 如果 left 是 SparseArray，则检查 left.sp_values 是否与 right 共享内存
    if
    # 检查两个 Pandas Series 对象是否相等，抛出异常显示差异
    "assert_series_equal",
    
    # 检查两个 Pandas SparseArray 对象是否相等，抛出异常显示差异
    "assert_sp_array_equal",
    
    # 检查两个 Pandas TimedeltaArray 对象是否相等，抛出异常显示差异
    "assert_timedelta_array_equal",
    
    # 根据标签访问 DataFrame 中的单个元素
    "at",
    
    # Pandas 支持的布尔数据类型列表
    "BOOL_DTYPES",
    
    # 为测试预期的数据盒子化对象
    "box_expected",
    
    # Pandas 支持的字节数据类型列表
    "BYTES_DTYPES",
    
    # 检查是否可以设置区域设置
    "can_set_locale",
    
    # Pandas 支持的复数数据类型列表
    "COMPLEX_DTYPES",
    
    # 将行列表转换为 CSV 格式的字符串
    "convert_rows_list_to_csv_str",
    
    # 大端或小端字节顺序表示
    "ENDIAN",
    
    # 确保数据集合是干净的，没有无效的或缺失的数据
    "ensure_clean",
    
    # 抛出外部错误异常
    "external_error_raised",
    
    # Pandas 支持的可扩展精度浮点数据类型列表
    "FLOAT_EA_DTYPES",
    
    # Pandas 支持的 numpy 浮点数据类型列表
    "FLOAT_NUMPY_DTYPES",
    
    # 获取 Cython 表格参数
    "get_cython_table_params",
    
    # 获取 Pandas 对象的数据类型
    "get_dtype",
    
    # 获取 DataFrame 或 Series 中的元素
    "getitem",
    
    # 获取系统中可用的区域设置列表
    "get_locales",
    
    # 获取日期时间对象的最精确单位
    "get_finest_unit",
    
    # 获取 Pandas 对象的对象表示形式
    "get_obj",
    
    # 根据操作名称获取操作对象
    "get_op_from_name",
    
    # 根据位置获取 DataFrame 中的单个元素
    "iat",
    
    # 根据位置获取 DataFrame 或 Series 中的单个元素
    "iloc",
    
    # 根据标签获取 DataFrame 或 Series 中的单个元素
    "loc",
    
    # 如果执行某个操作可能会产生警告，则返回 True
    "maybe_produces_warning",
    
    # Pandas 支持的窄数据类型列表
    "NARROW_NP_DTYPES",
    
    # NumPy 中的自然对象类型列表
    "NP_NAT_OBJECTS",
    
    # Pandas 中的空对象类型列表
    "NULL_OBJECTS",
    
    # Pandas 支持的对象数据类型列表
    "OBJECT_DTYPES",
    
    # 抛出详细的断言异常
    "raise_assert_detail",
    
    # 抛出链式赋值错误异常
    "raises_chained_assignment_error",
    
    # 通过 pathlib 路径执行对象的往返（反序列化和序列化）
    "round_trip_pathlib",
    
    # 通过 pickle 执行对象的往返（反序列化和序列化）
    "round_trip_pickle",
    
    # 根据标签设置 DataFrame 或 Series 中的单个元素
    "setitem",
    
    # 设置区域设置
    "set_locale",
    
    # 设置时区
    "set_timezone",
    
    # 检查两个对象是否共享内存
    "shares_memory",
    
    # Pandas 支持的有符号整数扩展精度数据类型列表
    "SIGNED_INT_EA_DTYPES",
    
    # Pandas 支持的有符号整数 numpy 数据类型列表
    "SIGNED_INT_NUMPY_DTYPES",
    
    # Pandas 支持的字符串数据类型列表
    "STRING_DTYPES",
    
    # 自定义的 Pandas DataFrame 子类
    "SubclassedDataFrame",
    
    # 自定义的 Pandas Series 子类
    "SubclassedSeries",
    
    # Pandas 支持的时间增量数据类型列表
    "TIMEDELTA64_DTYPES",
    
    # 将 DataFrame 或 Series 转换为 NumPy 数组
    "to_array",
    
    # Pandas 支持的无符号整数扩展精度数据类型列表
    "UNSIGNED_INT_EA_DTYPES",
    
    # Pandas 支持的无符号整数 numpy 数据类型列表
    "UNSIGNED_INT_NUMPY_DTYPES",
    
    # 使用指定的 CSV 方言参数进行 CSV 操作
    "with_csv_dialect",
    
    # 将数据写入压缩文件
    "write_to_compressed",
]
```
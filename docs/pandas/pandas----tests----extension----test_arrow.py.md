# `D:\src\scipysrc\pandas\pandas\tests\extension\test_arrow.py`

```
"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.
The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).
Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.
"""

# 引入必要的模块和库
from __future__ import annotations  # 允许在类型注解中使用字符串形式的类型

from datetime import (  # 导入日期时间相关的模块和类
    date,            # 日期类
    datetime,        # 日期时间类
    time,            # 时间类
    timedelta,       # 时间间隔类
)
from decimal import Decimal  # 导入 Decimal 类用于处理精确的十进制浮点运算
from io import (  # 导入字节流和文本流处理相关的类
    BytesIO,     # 用于处理二进制数据的字节流
    StringIO,    # 用于处理字符串数据的文本流
)
import operator   # 运算符模块，提供了许多Python中的运算符函数
import pickle     # 用于序列化和反序列化 Python 对象的模块
import re         # 正则表达式模块，用于处理字符串匹配和操作
import sys        # 系统相关的功能模块

import numpy as np   # 引入 NumPy 库，用于数值计算
import pytest        # Pytest 测试框架，用于编写和运行测试

from pandas._libs import lib       # pandas 私有 C 库
from pandas._libs.tslibs import timezones  # pandas 时间区域相关工具
from pandas.compat import (  # 兼容性处理模块，用于跨不同 Python 版本和平台的兼容性处理
    PY311, PY312,           # 版本号兼容性判断
    is_ci_environment,      # 判断是否在 CI 环境中
    is_platform_windows,    # 判断是否在 Windows 平台上
    pa_version_under11p0,   # 判断 pyarrow 版本是否小于 11.0
    pa_version_under13p0,   # 判断 pyarrow 版本是否小于 13.0
    pa_version_under14p0,   # 判断 pyarrow 版本是否小于 14.0
)
import pandas.util._test_decorators as td  # pandas 测试装饰器模块

from pandas.core.dtypes.dtypes import (  # 导入 pandas 数据类型相关的类
    ArrowDtype,           # Arrow 扩展数据类型
    CategoricalDtypeType,  # 类别数据类型
)

import pandas as pd                 # 导入 pandas 库并简写为 pd
import pandas._testing as tm         # pandas 测试工具模块
from pandas.api.extensions import no_default  # pandas 扩展接口相关
from pandas.api.types import (  # pandas 数据类型判断相关
    is_bool_dtype,                 # 布尔类型判断
    is_datetime64_any_dtype,       # 日期时间类型判断
    is_float_dtype,                # 浮点类型判断
    is_integer_dtype,              # 整数类型判断
    is_numeric_dtype,              # 数值类型判断
    is_signed_integer_dtype,       # 有符号整数类型判断
    is_string_dtype,               # 字符串类型判断
    is_unsigned_integer_dtype,     # 无符号整数类型判断
)

from pandas.tests.extension import base  # pandas 扩展测试基类

pa = pytest.importorskip("pyarrow")  # 导入并检查 pyarrow 库是否可用，如果不可用则跳过测试

from pandas.core.arrays.arrow.array import ArrowExtensionArray  # 导入 Arrow 扩展数组
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType  # 导入 Arrow 时期类型


def _require_timezone_database(request):
    """
    根据环境条件为测试添加标记，以便在特定环境下跳过失败的测试。

    Args:
        request: pytest 的请求对象

    Returns:
        None
    """
    if is_platform_windows() and is_ci_environment():
        mark = pytest.mark.xfail(
            raises=pa.ArrowInvalid,
            reason=(
                "TODO: Set ARROW_TIMEZONE_DATABASE environment variable "
                "on CI to path to the tzdata for pyarrow."
            ),
        )
        request.applymarker(mark)


@pytest.fixture(params=tm.ALL_PYARROW_DTYPES, ids=str)
def dtype(request):
    """
    根据给定的参数创建 ArrowDtype 对象作为测试的数据类型。

    Args:
        request: pytest 的请求对象，包含测试参数信息

    Returns:
        ArrowDtype: 返回 ArrowDtype 对象
    """
    return ArrowDtype(pyarrow_dtype=request.param)


@pytest.fixture
def data(dtype):
    """
    根据指定的数据类型创建相应的测试数据。

    Args:
        dtype (ArrowDtype): Arrow 数据类型对象

    Returns:
        list: 包含测试数据的列表
    """
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        data = [True, False] * 4 + [None] + [True, False] * 44 + [None] + [True, False]
    elif pa.types.is_floating(pa_dtype):
        data = [1.0, 0.0] * 4 + [None] + [-2.0, -1.0] * 44 + [None] + [0.5, 99.5]
    elif pa.types.is_signed_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [-2, -1] * 44 + [None] + [1, 99]
    elif pa.types.is_unsigned_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [2, 1] * 44 + [None] + [1, 99]
    return data
    # 如果数据类型是 decimal，则创建一个包含 Decimal 对象的列表，其中包括一些空值，总共包含 92 个元素
    elif pa.types.is_decimal(pa_dtype):
        data = (
            [Decimal("1"), Decimal("0.0")] * 4  # 创建包含 Decimal("1"), Decimal("0.0") 的列表，重复 4 次
            + [None]  # 添加一个空值
            + [Decimal("-2.0"), Decimal("-1.0")] * 44  # 再次创建包含 Decimal("-2.0"), Decimal("-1.0") 的列表，重复 44 次
            + [None]  # 添加一个空值
            + [Decimal("0.5"), Decimal("33.123")]  # 最后创建包含 Decimal("0.5"), Decimal("33.123") 的列表
        )
    
    # 如果数据类型是 date，则创建一个包含 date 对象的列表，总共包含 92 个元素
    elif pa.types.is_date(pa_dtype):
        data = (
            [date(2022, 1, 1), date(1999, 12, 31)] * 4  # 创建包含 date(2022, 1, 1), date(1999, 12, 31) 的列表，重复 4 次
            + [None]  # 添加一个空值
            + [date(2022, 1, 1), date(2022, 1, 1)] * 44  # 再次创建包含 date(2022, 1, 1), date(2022, 1, 1) 的列表，重复 44 次
            + [None]  # 添加一个空值
            + [date(1999, 12, 31), date(1999, 12, 31)]  # 最后创建包含 date(1999, 12, 31), date(1999, 12, 31) 的列表
        )
    
    # 如果数据类型是 timestamp，则创建一个包含 datetime 对象的列表，总共包含 92 个元素
    elif pa.types.is_timestamp(pa_dtype):
        data = (
            [datetime(2020, 1, 1, 1, 1, 1, 1), datetime(1999, 1, 1, 1, 1, 1, 1)] * 4  # 创建包含 datetime 对象的列表，重复 4 次
            + [None]  # 添加一个空值
            + [datetime(2020, 1, 1, 1), datetime(1999, 1, 1, 1)] * 44  # 再次创建包含 datetime 对象的列表，重复 44 次
            + [None]  # 添加一个空值
            + [datetime(2020, 1, 1), datetime(1999, 1, 1)]  # 最后创建包含 datetime 对象的列表
        )
    
    # 如果数据类型是 duration，则创建一个包含 timedelta 对象的列表，总共包含 92 个元素
    elif pa.types.is_duration(pa_dtype):
        data = (
            [timedelta(1), timedelta(1, 1)] * 4  # 创建包含 timedelta 对象的列表，重复 4 次
            + [None]  # 添加一个空值
            + [timedelta(-1), timedelta(0)] * 44  # 再次创建包含 timedelta 对象的列表，重复 44 次
            + [None]  # 添加一个空值
            + [timedelta(-10), timedelta(10)]  # 最后创建包含 timedelta 对象的列表
        )
    
    # 如果数据类型是 time，则创建一个包含 time 对象的列表，总共包含 92 个元素
    elif pa.types.is_time(pa_dtype):
        data = (
            [time(12, 0), time(0, 12)] * 4  # 创建包含 time 对象的列表，重复 4 次
            + [None]  # 添加一个空值
            + [time(0, 0), time(1, 1)] * 44  # 再次创建包含 time 对象的列表，重复 44 次
            + [None]  # 添加一个空值
            + [time(0, 5), time(5, 0)]  # 最后创建包含 time 对象的列表
        )
    
    # 如果数据类型是 string，则创建一个包含字符串的列表，总共包含 92 个元素
    elif pa.types.is_string(pa_dtype):
        data = ["a", "b"] * 4  # 创建包含字符串 "a", "b" 的列表，重复 4 次
        + [None]  # 添加一个空值
        + ["1", "2"] * 44  # 再次创建包含字符串 "1", "2" 的列表，重复 44 次
        + [None]  # 添加一个空值
        + ["!", ">"]  # 最后创建包含字符串 "!", ">" 的列表
    
    # 如果数据类型是 binary，则创建一个包含字节对象的列表，总共包含 92 个元素
    elif pa.types.is_binary(pa_dtype):
        data = [b"a", b"b"] * 4  # 创建包含字节对象 b"a", b"b" 的列表，重复 4 次
        + [None]  # 添加一个空值
        + [b"1", b"2"] * 44  # 再次创建包含字节对象 b"1", b"2" 的列表，重复 44 次
        + [None]  # 添加一个空值
        + [b"!", b">"]  # 最后创建包含字节对象 b"!", b">" 的列表
    
    # 如果数据类型不属于以上类型，则抛出未实现的错误
    else:
        raise NotImplementedError
    
    # 使用 Pandas 的 pd.array 方法创建数据数组，并指定数据类型为 dtype
    return pd.array(data, dtype=dtype)
@pytest.fixture
def data_missing(data):
    """
    Length-2 array with [NA, Valid]

    Fixture that returns an array with two elements: None (representing missing data)
    and the first element of the input data array (representing valid data).
    """
    return type(data)._from_sequence([None, data[0]], dtype=data.dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """
    Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    This fixture is used for testing dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_for_grouping(dtype):
    """
    Data for factorization, grouping, and unique tests.

    Generates an array based on the data type 'dtype' with values arranged such that:
    [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA represents missing values.
    """
    pa_dtype = dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        A = False
        B = True
        C = True
    elif pa.types.is_floating(pa_dtype):
        A = -1.1
        B = 0.0
        C = 1.1
    elif pa.types.is_signed_integer(pa_dtype):
        A = -1
        B = 0
        C = 1
    elif pa.types.is_unsigned_integer(pa_dtype):
        A = 0
        B = 1
        C = 10
    elif pa.types.is_date(pa_dtype):
        A = date(1999, 12, 31)
        B = date(2010, 1, 1)
        C = date(2022, 1, 1)
    elif pa.types.is_timestamp(pa_dtype):
        A = datetime(1999, 1, 1, 1, 1, 1, 1)
        B = datetime(2020, 1, 1)
        C = datetime(2020, 1, 1, 1)
    elif pa.types.is_duration(pa_dtype):
        A = timedelta(-1)
        B = timedelta(0)
        C = timedelta(1, 4)
    elif pa.types.is_time(pa_dtype):
        A = time(0, 0)
        B = time(0, 12)
        C = time(12, 12)
    elif pa.types.is_string(pa_dtype):
        A = "a"
        B = "b"
        C = "c"
    elif pa.types.is_binary(pa_dtype):
        A = b"a"
        B = b"b"
        C = b"c"
    elif pa.types.is_decimal(pa_dtype):
        A = Decimal("-1.1")
        B = Decimal("0.0")
        C = Decimal("1.1")
    else:
        raise NotImplementedError

    return pd.array([B, B, None, None, A, A, B, C], dtype=dtype)


@pytest.fixture
def data_for_sorting(data_for_grouping):
    """
    Length-3 array with a known sort order.

    Returns an array of length 3 sorted in the order [B, C, A], where A < B < C.
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[7], data_for_grouping[4]],
        dtype=data_for_grouping.dtype,
    )


@pytest.fixture
def data_missing_for_sorting(data_for_grouping):
    """
    Length-3 array with a known sort order.

    Returns an array of length 3 sorted in the order [B, NA, A], where A < B and NA represents missing values.
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[2], data_for_grouping[4]],
        dtype=data_for_grouping.dtype,
    )


@pytest.fixture
def data_for_twos(data):
    """
    Length-100 array in which all the elements are two.

    Fixture that returns an array of length 100 where all elements are the integer value 2.
    """
    pa_dtype = data.dtype.pyarrow_dtype
    # 检查给定的数据类型是否为整数、浮点数、十进制数或持续时间
    if (
        pa.types.is_integer(pa_dtype)
        or pa.types.is_floating(pa_dtype)
        or pa.types.is_decimal(pa_dtype)
        or pa.types.is_duration(pa_dtype)
    ):
        # 如果是上述类型之一，返回一个包含100个元素，每个元素为2的数组，数据类型与输入数据相同
        return pd.array([2] * 100, dtype=data.dtype)
    
    # 如果不满足上述条件，则返回原始数据
    # TODO: 否则跳过？
    return data
# 定义一个名为 TestArrowArray 的测试类，继承自 base.ExtensionTests
class TestArrowArray(base.ExtensionTests):

    # 定义一个测试方法 test_compare_scalar，接受两个参数 data 和 comparison_op
    def test_compare_scalar(self, data, comparison_op):
        # 将 data 转换为 pandas Series 对象
        ser = pd.Series(data)
        # 调用父类方法 _compare_other，比较 ser 与 data 的相关性，使用 data 的第一个元素作为比较标准
        self._compare_other(ser, data, comparison_op, data[0])

    # 使用 pytest.mark.parametrize 标记的参数化测试方法 test_map，接受两个参数 data_missing 和 na_action
    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data_missing, na_action):
        # 如果 data_missing 的 dtype 的类型在 "mM" 中
        if data_missing.dtype.kind in "mM":
            # 调用 Series 的 map 方法，使用 lambda 函数进行映射，na_action 参数指定如何处理缺失值
            result = data_missing.map(lambda x: x, na_action=na_action)
            # 将 data_missing 转换为 numpy 数组，类型为 object
            expected = data_missing.to_numpy(dtype=object)
            # 断言 result 与 expected 的 numpy 数组是否相等
            tm.assert_numpy_array_equal(result, expected)
        else:
            # 否则，调用 Series 的 map 方法，同样使用 lambda 函数进行映射，na_action 参数指定如何处理缺失值
            result = data_missing.map(lambda x: x, na_action=na_action)
            # 如果 data_missing 的 dtype 为 "float32[pyarrow]"
            if data_missing.dtype == "float32[pyarrow]":
                # 对 map 操作进行回转，通过对象转换为 float64 类型
                expected = data_missing.to_numpy(dtype="float64", na_value=np.nan)
            else:
                # 否则，将 data_missing 转换为 numpy 数组
                expected = data_missing.to_numpy()
            # 断言 result 与 expected 的 numpy 数组是否相等
            tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法 test_astype_str，接受两个参数 data 和 request
    def test_astype_str(self, data, request):
        # 获取 data 的 pyarrow 数据类型
        pa_dtype = data.dtype.pyarrow_dtype
        # 如果 pa_dtype 是二进制类型
        if pa.types.is_binary(pa_dtype):
            # 应用 pytest.mark.xfail 标记，原因是对于 pa_dtype 的 .astype(str) 操作会解码
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"For {pa_dtype} .astype(str) decodes.",
                )
            )
        # 或者，如果 pa_dtype 是无时区的时间戳类型或者持续时间类型
        elif (
            pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is None
        ) or pa.types.is_duration(pa_dtype):
            # 应用 pytest.mark.xfail 标记，原因是 pd.Timestamp/pd.Timedelta 的表示与 numpy 的表示不同
            request.applymarker(
                pytest.mark.xfail(
                    reason="pd.Timestamp/pd.Timedelta repr different from numpy repr",
                )
            )
        # 调用父类的 test_astype_str 方法，传入 data 参数
        super().test_astype_str(data)

    # 使用 pytest.mark.parametrize 标记的参数化测试方法 test_astype_string，接受三个参数 data、nullable_string_dtype 和 request
    @pytest.mark.parametrize(
        "nullable_string_dtype",
        [
            "string[python]",
            pytest.param("string[pyarrow]", marks=td.skip_if_no("pyarrow")),
        ],
    )
    def test_astype_string(self, data, nullable_string_dtype, request):
        # 获取 data 的 pyarrow 数据类型
        pa_dtype = data.dtype.pyarrow_dtype
        # 如果 pa_dtype 是无时区的时间戳类型或者持续时间类型
        if (
            pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is None
        ) or pa.types.is_duration(pa_dtype):
            # 应用 pytest.mark.xfail 标记，原因是 pd.Timestamp/pd.Timedelta 的表示与 numpy 的表示不同
            request.applymarker(
                pytest.mark.xfail(
                    reason="pd.Timestamp/pd.Timedelta repr different from numpy repr",
                )
            )
        # 调用父类的 test_astype_string 方法，传入 data 和 nullable_string_dtype 参数
        super().test_astype_string(data, nullable_string_dtype)

    # 定义一个测试方法 test_from_dtype，接受两个参数 data 和 request
    def test_from_dtype(self, data, request):
        # 获取 data 的 pyarrow 数据类型
        pa_dtype = data.dtype.pyarrow_dtype
        # 如果 pa_dtype 是字符串类型或者是十进制类型
        if pa.types.is_string(pa_dtype) or pa.types.is_decimal(pa_dtype):
            # 如果 pa_dtype 是字符串类型，设置原因字符串
            if pa.types.is_string(pa_dtype):
                reason = "ArrowDtype(pa.string()) != StringDtype('pyarrow')"
            else:
                reason = f"pyarrow.type_for_alias cannot infer {pa_dtype}"
            # 应用 pytest.mark.xfail 标记，原因是 reason 变量中指定的原因
            request.applymarker(
                pytest.mark.xfail(
                    reason=reason,
                )
            )
        # 调用父类的 test_from_dtype 方法，传入 data 参数
        super().test_from_dtype(data)
    # 测试函数，用于从序列创建扩展数组
    def test_from_sequence_pa_array(self, data):
        # 根据指定链接讨论的建议进行注释，设置 ChunkedArray 为数据的底层存储
        # 从序列创建新的对象，类型与原始数据相同，并检查其相等性
        result = type(data)._from_sequence(data._pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)
        # 断言结果对象的底层存储是 ChunkedArray 类型
        assert isinstance(result._pa_array, pa.ChunkedArray)

        # 对数据的 ChunkedArray 合并后再次创建新对象，并检查其相等性
        result = type(data)._from_sequence(
            data._pa_array.combine_chunks(), dtype=data.dtype
        )
        tm.assert_extension_array_equal(result, data)
        # 再次断言结果对象的底层存储是 ChunkedArray 类型
        assert isinstance(result._pa_array, pa.ChunkedArray)

    # 测试函数，处理无法实现的序列创建情况
    def test_from_sequence_pa_array_notimplemented(self, request):
        # 使用 ArrowDtype 创建特定的数据类型，预期引发 NotImplementedError
        dtype = ArrowDtype(pa.month_day_nano_interval())
        with pytest.raises(NotImplementedError, match="Converting strings to"):
            ArrowExtensionArray._from_sequence_of_strings(["12-1"], dtype=dtype)

    # 测试函数，从字符串序列创建扩展数组
    def test_from_sequence_of_strings_pa_array(self, data, request):
        # 获取数据的 PyArrow 数据类型
        pa_dtype = data.dtype.pyarrow_dtype
        # 根据条件设置测试标记，对不支持的时间解析进行标记
        if pa.types.is_time64(pa_dtype) and pa_dtype.equals("time64[ns]") and not PY311:
            request.applymarker(
                pytest.mark.xfail(
                    reason="Nanosecond time parsing not supported.",
                )
            )
        elif pa_version_under11p0 and (
            pa.types.is_duration(pa_dtype) or pa.types.is_decimal(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow doesn't support parsing {pa_dtype}",
                )
            )
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None:
            _require_timezone_database(request)

        # 将数据的 ChunkedArray 转换为字符串类型的 PyArrow 数组
        pa_array = data._pa_array.cast(pa.string())
        # 从字符串序列创建新对象，并检查其相等性
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

        # 合并 ChunkedArray 后再次创建新对象，并检查其相等性
        pa_array = pa_array.combine_chunks()
        result = type(data)._from_sequence_of_strings(pa_array, dtype=data.dtype)
        tm.assert_extension_array_equal(result, data)

    # 辅助函数，检查累积操作的结果
    def check_accumulate(self, ser, op_name, skipna):
        # 调用指定操作名的累积方法，并获取结果
        result = getattr(ser, op_name)(skipna=skipna)

        # 获取数据的 PyArrow 数据类型
        pa_type = ser.dtype.pyarrow_dtype
        # 如果数据类型是时间类型，则将结果转换为整数类型以保持一致性
        if pa.types.is_temporal(pa_type):
            # 根据位宽选择对应的整数类型
            if pa_type.bit_width == 32:
                int_type = "int32[pyarrow]"
            else:
                int_type = "int64[pyarrow]"
            # 将数据和结果转换为选定的整数类型
            ser = ser.astype(int_type)
            result = result.astype(int_type)

        # 将结果强制转换为 Float64 类型
        result = result.astype("Float64")
        # 获取预期结果，并确保其与结果相等（忽略数据类型）
        expected = getattr(ser.astype("Float64"), op_name)(skipna=skipna)
        tm.assert_series_equal(result, expected, check_dtype=False)
    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        # 检查是否支持累积操作，返回布尔值

        # 获取 pandas Series 的 pyarrow 数据类型
        pa_type = ser.dtype.pyarrow_dtype  # type: ignore[union-attr]

        # 根据数据类型判断是否支持特定的累积操作
        if (
            pa.types.is_string(pa_type)
            or pa.types.is_binary(pa_type)
            or pa.types.is_decimal(pa_type)
        ):
            if op_name in ["cumsum", "cumprod", "cummax", "cummin"]:
                return False
        elif pa.types.is_boolean(pa_type):
            if op_name in ["cumprod", "cummax", "cummin"]:
                return False
        elif pa.types.is_temporal(pa_type):
            if op_name == "cumsum" and not pa.types.is_duration(pa_type):
                return False
            elif op_name == "cumprod":
                return False
        # 默认支持其余情况的累积操作
        return True

    @pytest.mark.parametrize("skipna", [True, False])
    def test_accumulate_series(self, data, all_numeric_accumulations, skipna, request):
        # 获取 pyarrow 数据类型
        pa_type = data.dtype.pyarrow_dtype
        # 获取累积操作名称
        op_name = all_numeric_accumulations
        # 创建 pandas Series
        ser = pd.Series(data)

        # 检查是否支持当前的累积操作
        if not self._supports_accumulation(ser, op_name):
            # 基类测试将检查我们是否引发异常
            return super().test_accumulate_series(
                data, all_numeric_accumulations, skipna
            )

        # 如果 pyarrow 版本低于 13.0，并且累积操作不是 "cumsum"
        if pa_version_under13p0 and all_numeric_accumulations != "cumsum":
            # xfailing 在运行时花费较长时间，因为 pytest 即使不显示异常消息也会渲染它们
            opt = request.config.option
            if opt.markexpr and "not slow" in opt.markexpr:
                # 如果在 markexpr 中包含 "not slow"，则跳过测试
                pytest.skip(
                    f"{all_numeric_accumulations} not implemented for pyarrow < 9"
                )
            # 标记为预期失败，并提供失败原因
            mark = pytest.mark.xfail(
                reason=f"{all_numeric_accumulations} not implemented for pyarrow < 9"
            )
            request.applymarker(mark)

        # 如果累积操作为 "cumsum"，并且数据类型是布尔型或者小数型
        elif all_numeric_accumulations == "cumsum" and (
            pa.types.is_boolean(pa_type) or pa.types.is_decimal(pa_type)
        ):
            # 标记为预期失败，提供失败原因和预期抛出的异常类型
            request.applymarker(
                pytest.mark.xfail(
                    reason=f"{all_numeric_accumulations} not implemented for {pa_type}",
                    raises=NotImplementedError,
                )
            )

        # 执行累积操作检查
        self.check_accumulate(ser, op_name, skipna)
    # 判断给定的序列是否支持进行指定操作
    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # 获取序列的数据类型
        dtype = ser.dtype
        # 获取序列的 PyArrow 数据类型（忽略类型检查错误）
        pa_dtype = dtype.pyarrow_dtype  # type: ignore[union-attr]
        # 如果数据类型是时间类型并且操作在指定的列表中
        if pa.types.is_temporal(pa_dtype) and op_name in [
            "sum",
            "var",
            "skew",
            "kurt",
            "prod",
        ]:
            # 如果是持续时间类型并且操作是求和，则支持
            if pa.types.is_duration(pa_dtype) and op_name in ["sum"]:
                # 求和时间增量是唯一一个明确定义的情况
                pass
            else:
                return False
        # 如果数据类型是字符串或二进制类型并且操作在指定的列表中
        elif (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ) and op_name in [
            "sum",
            "mean",
            "median",
            "prod",
            "std",
            "sem",
            "var",
            "skew",
            "kurt",
        ]:
            # 不支持字符串或二进制类型执行这些操作
            return False

        # 如果数据类型是时间类型并且不是持续时间，并且操作在指定的列表中
        if (
            pa.types.is_temporal(pa_dtype)
            and not pa.types.is_duration(pa_dtype)
            and op_name in ["any", "all"]
        ):
            # 我们在非 PyArrow datetime64 数据类型中支持此操作，
            # 但不明确我们是否应该。目前保持 PyArrow 行为不支持此操作。
            return False

        # 其他情况下默认支持该操作
        return True

    # 检查序列是否在指定操作上与备选序列计算结果近似相等
    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # 获取序列的 PyArrow 数据类型（忽略类型检查错误）
        pa_dtype = ser.dtype.pyarrow_dtype  # type: ignore[union-attr]
        # 如果数据类型是整数或浮点数
        if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
            # 将序列转换为 Float64 类型作为备选序列
            alt = ser.astype("Float64")
        else:
            # 否则直接使用原始序列作为备选序列
            alt = ser

        # 根据操作名执行序列的操作，并获取结果
        if op_name == "count":
            # 对于计数操作，直接获取结果和备选序列的结果
            result = getattr(ser, op_name)()
            expected = getattr(alt, op_name)()
        else:
            # 对于其他操作，传入 skipna 参数执行操作，并获取结果
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(alt, op_name)(skipna=skipna)
        
        # 使用测试工具库比较序列操作的结果和期望结果的近似性
        tm.assert_almost_equal(result, expected)

    # 使用 pytest 参数化装饰器定义测试用例参数 skipna 为 True 和 False
    @pytest.mark.parametrize("skipna", [True, False])
    # 测试函数，用于处理数值序列的缩减操作
    def test_reduce_series_numeric(self, data, all_numeric_reductions, skipna, request):
        # 获取数据的 dtype
        dtype = data.dtype
        # 转换为 pyarrow 的数据类型
        pa_dtype = dtype.pyarrow_dtype

        # 标记为预期失败，如果出现 TypeError 异常，给出原因的字符串
        xfail_mark = pytest.mark.xfail(
            raises=TypeError,
            reason=(
                f"{all_numeric_reductions} is not implemented in "
                f"pyarrow={pa.__version__} for {pa_dtype}"
            ),
        )
        
        # 如果所需的缩减操作是 "skew" 或 "kurt"，并且数据类型是数值类型或布尔类型
        if all_numeric_reductions in {"skew", "kurt"} and (
            dtype._is_numeric or dtype.kind == "b"
        ):
            # 应用预期失败标记
            request.applymarker(xfail_mark)

        # 如果数据类型是布尔类型，并且所需的缩减操作是 "sem", "std", "var", "median"
        elif pa.types.is_boolean(pa_dtype) and all_numeric_reductions in {
            "sem",
            "std",
            "var",
            "median",
        }:
            # 应用预期失败标记
            request.applymarker(xfail_mark)
        
        # 调用父类的测试函数，执行数值序列的缩减操作测试
        super().test_reduce_series_numeric(data, all_numeric_reductions, skipna)

    # 使用参数化标记，测试布尔序列的缩减操作
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series_boolean(
        self, data, all_boolean_reductions, skipna, na_value, request
    ):
        # 获取数据的 pyarrow 数据类型
        pa_dtype = data.dtype.pyarrow_dtype
        # 标记为预期失败，如果出现 TypeError 异常，给出原因的字符串
        xfail_mark = pytest.mark.xfail(
            raises=TypeError,
            reason=(
                f"{all_boolean_reductions} is not implemented in "
                f"pyarrow={pa.__version__} for {pa_dtype}"
            ),
        )
        
        # 如果数据类型是字符串或二进制类型
        if pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype):
            # 应用预期失败标记
            # 我们可能希望使其行为像非 pyarrow 的情况一样，
            # 但目前还没有决定。
            request.applymarker(xfail_mark)

        # 调用父类的测试函数，执行布尔序列的缩减操作测试
        return super().test_reduce_series_boolean(data, all_boolean_reductions, skipna)

    # 获取预期的缩减操作后的数据类型
    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        # 如果操作名是 "max" 或 "min"
        if op_name in ["max", "min"]:
            cmp_dtype = arr.dtype
        # 如果数据类型是 "decimal128(7, 3)[pyarrow]"，根据操作名选择数据类型
        elif arr.dtype.name == "decimal128(7, 3)[pyarrow]":
            if op_name not in ["median", "var", "std"]:
                cmp_dtype = arr.dtype
            else:
                cmp_dtype = "float64[pyarrow]"
        # 如果操作名是 "median", "var", "std", "mean", "skew"
        elif op_name in ["median", "var", "std", "mean", "skew"]:
            cmp_dtype = "float64[pyarrow]"
        # 否则，根据数据类型的种类选择数据类型
        else:
            cmp_dtype = {
                "i": "int64[pyarrow]",
                "u": "uint64[pyarrow]",
                "f": "float64[pyarrow]",
            }[arr.dtype.kind]
        
        # 返回预期的比较数据类型
        return cmp_dtype

    # 使用参数化标记，测试数据框的缩减操作
    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_frame(self, data, all_numeric_reductions, skipna, request):
        # 获取缩减操作的名称
        op_name = all_numeric_reductions
        
        # 如果操作名是 "skew"，并且数据类型是数值类型
        if op_name == "skew":
            if data.dtype._is_numeric:
                # 标记为预期失败，给出原因字符串
                mark = pytest.mark.xfail(reason="skew not implemented")
                # 应用预期失败标记
                request.applymarker(mark)
        
        # 调用父类的测试函数，执行数据框的缩减操作测试
        return super().test_reduce_frame(data, all_numeric_reductions, skipna)

    # 使用参数化标记，测试不同类型的数据
    @pytest.mark.parametrize("typ", ["int64", "uint64", "float64"])
    # 定义测试方法，用于验证不是近似中位数的情况
    def test_median_not_approximate(self, typ):
        # GH 52679：GitHub 上的问题编号
        # 创建包含整数序列的 Pandas Series，并计算其中位数，使用指定的数据类型
        result = pd.Series([1, 2], dtype=f"{typ}[pyarrow]").median()
        # 断言中位数计算结果是否为 1.5
        assert result == 1.5

    # 定义测试方法，用于在数值分组中处理字符串类型数据
    def test_in_numeric_groupby(self, data_for_grouping):
        dtype = data_for_grouping.dtype
        # 如果数据类型为字符串
        if is_string_dtype(dtype):
            # 创建包含指定数据的 DataFrame
            df = pd.DataFrame(
                {
                    "A": [1, 1, 2, 2, 3, 3, 1, 4],
                    "B": data_for_grouping,
                    "C": [1, 1, 1, 1, 1, 1, 1, 1],
                }
            )
            # 预期的结果为索引 ["C"]
            expected = pd.Index(["C"])
            # 准备用于异常信息匹配的消息字符串
            msg = re.escape(f"agg function failed [how->sum,dtype->{dtype}")
            # 使用 pytest 断言引发 TypeError 异常，并验证异常信息是否匹配
            with pytest.raises(TypeError, match=msg):
                df.groupby("A").sum()
            # 计算 DataFrame 按 "A" 列分组并求和后的数值列的列名
            result = df.groupby("A").sum(numeric_only=True).columns
            # 使用测试模块的函数验证计算结果的列名与预期是否相等
            tm.assert_index_equal(result, expected)
        else:
            # 如果数据类型不是字符串，则调用父类的相应测试方法
            super().test_in_numeric_groupby(data_for_grouping)

    # 定义测试方法，用于验证从字符串构造自身名称的情况
    def test_construct_from_string_own_name(self, dtype, request):
        # 获取 PyArrow 的数据类型
        pa_dtype = dtype.pyarrow_dtype
        # 如果数据类型是十进制数类型
        if pa.types.is_decimal(pa_dtype):
            # 应用 pytest.mark.xfail 标记，标记为预期抛出 NotImplementedError 异常
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                )
            )

        # 如果数据类型是字符串类型
        if pa.types.is_string(pa_dtype):
            # 验证使用旧方法构造字符串类型是否引发 TypeError 异常
            msg = r"string\[pyarrow\] should be constructed by StringDtype"
            with pytest.raises(TypeError, match=msg):
                dtype.construct_from_string(dtype.name)

            return

        # 如果数据类型不是字符串类型，则调用父类的相应测试方法
        super().test_construct_from_string_own_name(dtype)

    # 定义测试方法，用于验证从名称判断数据类型的情况
    def test_is_dtype_from_name(self, dtype, request):
        # 获取 PyArrow 的数据类型
        pa_dtype = dtype.pyarrow_dtype
        # 如果数据类型是字符串类型
        if pa.types.is_string(pa_dtype):
            # 验证不是从名称判断数据类型
            assert not type(dtype).is_dtype(dtype.name)
        else:
            # 如果数据类型不是字符串类型，并且是十进制数类型
            if pa.types.is_decimal(pa_dtype):
                # 应用 pytest.mark.xfail 标记，标记为预期抛出 NotImplementedError 异常
                request.applymarker(
                    pytest.mark.xfail(
                        raises=NotImplementedError,
                        reason=f"pyarrow.type_for_alias cannot infer {pa_dtype}",
                    )
                )
            # 调用父类的相应测试方法
            super().test_is_dtype_from_name(dtype)

    # 定义测试方法，用于验证从字符串构造其他类型数据时引发异常的情况
    def test_construct_from_string_another_type_raises(self, dtype):
        # 准备用于异常信息匹配的消息字符串
        msg = r"'another_type' must end with '\[pyarrow\]'"
        # 使用 pytest 断言引发 TypeError 异常，并验证异常信息是否匹配
        with pytest.raises(TypeError, match=msg):
            type(dtype).construct_from_string("another_type")
    # 定义一个测试方法，用于检查给定数据类型的通用数据类型获取功能
    def test_get_common_dtype(self, dtype, request):
        # 获取PyArrow数据类型
        pa_dtype = dtype.pyarrow_dtype
        # 如果数据类型是日期、时间、带时区的时间戳、二进制或者是十进制类型之一，则添加一个标记来标记预期测试失败
        if (
            pa.types.is_date(pa_dtype)
            or pa.types.is_time(pa_dtype)
            or (pa.types.is_timestamp(pa_dtype) and pa_dtype.tz is not None)
            or pa.types.is_binary(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    reason=(
                        f"{pa_dtype} does not have associated numpy "
                        f"dtype findable by find_common_type"
                    )
                )
            )
        # 调用父类的相同测试方法
        super().test_get_common_dtype(dtype)

    # 定义一个测试方法，用于检查给定数据类型是否不是字符串类型
    def test_is_not_string_type(self, dtype):
        # 获取PyArrow数据类型
        pa_dtype = dtype.pyarrow_dtype
        # 如果数据类型是字符串类型，则断言其为字符串数据类型
        if pa.types.is_string(pa_dtype):
            assert is_string_dtype(dtype)
        else:
            # 否则调用父类的相同测试方法
            super().test_is_not_string_type(dtype)

    # 添加一个标记，标记该测试为预期失败，指定原因是pyarrow.ChunkedArray不支持视图操作
    @pytest.mark.xfail(
        reason="GH 45419: pyarrow.ChunkedArray does not support views.", run=False
    )
    # 定义一个测试方法，测试数据的视图操作
    def test_view(self, data):
        super().test_view(data)

    # 定义一个测试方法，测试在填充操作中，如果不进行操作，是否返回副本
    def test_fillna_no_op_returns_copy(self, data):
        # 删除数据中的NaN值
        data = data[~data.isna()]

        # 获取有效值
        valid = data[0]
        # 对数据进行填充操作，并获取结果
        result = data.fillna(valid)
        # 断言结果不是原始数据的引用
        assert result is not data
        # 使用断言来比较填充后的结果和原始数据
        tm.assert_extension_array_equal(result, data)

    # 添加一个标记，标记该测试为预期失败，指定原因是pyarrow.ChunkedArray不支持视图操作
    @pytest.mark.xfail(
        reason="GH 45419: pyarrow.ChunkedArray does not support views", run=False
    )
    # 定义一个测试方法，测试数据的转置操作
    def test_transpose(self, data):
        super().test_transpose(data)

    # 添加一个标记，标记该测试为预期失败，指定原因是pyarrow.ChunkedArray不支持视图操作
    @pytest.mark.xfail(
        reason="GH 45419: pyarrow.ChunkedArray does not support views", run=False
    )
    # 定义一个测试方法，测试设置项目时是否保留视图
    def test_setitem_preserves_views(self, data):
        super().test_setitem_preserves_views(data)

    # 参数化测试，测试不同的dtype后端和引擎
    @pytest.mark.parametrize("dtype_backend", ["pyarrow", no_default])
    @pytest.mark.parametrize("engine", ["c", "python"])
    # 定义一个测试函数，用于测试不同的数据类型和引擎
    def test_EA_types(self, engine, data, dtype_backend, request):
        # 获取数据的 pyarrow 数据类型
        pa_dtype = data.dtype.pyarrow_dtype
        # 如果数据类型是 decimal，则标记测试为预期的失败，抛出 NotImplementedError
        if pa.types.is_decimal(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=NotImplementedError,
                    reason=f"Parameterized types {pa_dtype} not supported.",
                )
            )
        # 如果数据类型是 timestamp，并且单位为微秒或纳秒，则标记测试为预期的失败，抛出 ValueError
        elif pa.types.is_timestamp(pa_dtype) and pa_dtype.unit in ("us", "ns"):
            request.applymarker(
                pytest.mark.xfail(
                    raises=ValueError,
                    reason="https://github.com/pandas-dev/pandas/issues/49767",
                )
            )
        # 如果数据类型是 binary，则标记测试为预期的失败，原因是 CSV 解析器无法正确处理二进制数据
        elif pa.types.is_binary(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(reason="CSV parsers don't correctly handle binary")
            )
        # 创建一个包含特定数据类型的 DataFrame
        df = pd.DataFrame({"with_dtype": pd.Series(data, dtype=str(data.dtype))})
        # 将 DataFrame 转换为 CSV 格式，排除索引列，并处理缺失值
        csv_output = df.to_csv(index=False, na_rep=np.nan)
        # 如果数据类型是 binary，则将 CSV 输出转换为字节流
        if pa.types.is_binary(pa_dtype):
            csv_output = BytesIO(csv_output)
        # 否则，将 CSV 输出转换为字符串流
        else:
            csv_output = StringIO(csv_output)
        # 使用 pandas 读取 CSV 数据，并指定数据类型和后端引擎
        result = pd.read_csv(
            csv_output,
            dtype={"with_dtype": str(data.dtype)},
            engine=engine,
            dtype_backend=dtype_backend,
        )
        # 期望的结果是原始的 DataFrame
        expected = df
        # 断言结果与期望值相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，用于测试数据的反转操作
    def test_invert(self, data, request):
        # 获取数据的 pyarrow 数据类型
        pa_dtype = data.dtype.pyarrow_dtype
        # 如果数据类型既不是布尔型、整数型也不是字符串型，则标记测试为预期的失败，抛出 ArrowNotImplementedError
        if not (
            pa.types.is_boolean(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_string(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowNotImplementedError,
                    reason=f"pyarrow.compute.invert does support {pa_dtype}",
                )
            )
        # 如果满足 PY312 条件且数据类型是布尔型，则在测试中产生一个警告，并调用父类的 test_invert 方法
        if PY312 and pa.types.is_boolean(pa_dtype):
            with tm.assert_produces_warning(
                DeprecationWarning, match="Bitwise inversion", check_stacklevel=False
            ):
                super().test_invert(data)
        # 否则，直接调用父类的 test_invert 方法
        else:
            super().test_invert(data)

    # 使用参数化装饰器定义一个测试函数，测试数据的差分操作
    @pytest.mark.parametrize("periods", [1, -2])
    def test_diff(self, data, periods, request):
        # 获取数据的 pyarrow 数据类型
        pa_dtype = data.dtype.pyarrow_dtype
        # 如果数据类型是无符号整数，并且 periods 等于 1，则标记测试为预期的失败，抛出 ArrowInvalid
        if pa.types.is_unsigned_integer(pa_dtype) and periods == 1:
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=(
                        f"diff with {pa_dtype} and periods={periods} will overflow"
                    ),
                )
            )
        # 调用父类的 test_diff 方法，传入数据和 periods 参数
        super().test_diff(data, periods)

    # 定义一个测试函数，验证 value_counts 返回的数据类型是否为 pyarrow 的 int64
    def test_value_counts_returns_pyarrow_int64(self, data):
        # GH 51462
        # 截取数据的前10条记录
        data = data[:10]
        # 调用 value_counts 方法统计各个值出现的次数
        result = data.value_counts()
        # 断言结果的数据类型为 ArrowDtype 类型的 int64
        assert result.dtype == ArrowDtype(pa.int64())

    # 定义一个类变量，指定组合比较的预期数据类型
    _combine_le_expected_dtype = "bool[pyarrow]"

    # 定义一个类变量，指定 divmod 操作的异常为 NotImplementedError
    divmod_exc = NotImplementedError
    # 根据操作名获取相应的操作函数，处理特定的操作名
    def get_op_from_name(self, op_name):
        # 去除操作名中的下划线，生成简化的操作名
        short_opname = op_name.strip("_")
        # 如果简化后的操作名为 "rtruediv"
        if short_opname == "rtruediv":
            # 返回一个处理除法的函数，使用 numpy 的版本，避免除以零时引发异常
            def rtruediv(x, y):
                return np.divide(y, x)

            return rtruediv
        # 如果简化后的操作名为 "rfloordiv"
        elif short_opname == "rfloordiv":
            # 返回一个 lambda 函数，处理地板除法操作
            return lambda x, y: np.floor_divide(y, x)

        # 若以上条件都不满足，则调用 tm.get_op_from_name 处理操作名
        return tm.get_op_from_name(op_name)

    # 检查操作是否在时间序列数据类型中支持
    def _is_temporal_supported(self, opname, pa_dtype):
        return (
            (
                opname in ("__add__", "__radd__")
                or (
                    opname
                    in ("__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__")
                    and not pa_version_under14p0
                )
            )
            and pa.types.is_duration(pa_dtype)
            or opname in ("__sub__", "__rsub__")
            and pa.types.is_temporal(pa_dtype)
        )

    # 获取预期的异常类型
    def _get_expected_exception(
        self, op_name: str, obj, other
    ) -> type[Exception] | None:
        # 如果操作名是 "__divmod__" 或 "__rdivmod__"
        if op_name in ("__divmod__", "__rdivmod__"):
            # 返回预定义的除法求余异常类型
            return self.divmod_exc

        # 获取对象的数据类型
        dtype = tm.get_dtype(obj)
        # 获取 PyArrow 数据类型
        pa_dtype = dtype.pyarrow_dtype  # type: ignore[union-attr]

        # 检查是否支持 Arrow 时间类型的操作
        arrow_temporal_supported = self._is_temporal_supported(op_name, pa_dtype)
        # 如果操作名在 {"__mod__", "__rmod__"} 中
        if op_name in {
            "__mod__",
            "__rmod__",
        }:
            # 返回未实现错误异常
            exc = NotImplementedError
        # 如果支持 Arrow 时间类型的操作
        elif arrow_temporal_supported:
            exc = None
        # 如果操作名在 ["__add__", "__radd__"] 中，并且数据类型是字符串或二进制
        elif op_name in ["__add__", "__radd__"] and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            exc = None
        # 如果数据类型既不是浮点数、整数、也不是十进制数
        elif not (
            pa.types.is_floating(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            # TODO: 在许多情况下，例如非持续时间的时间类型，这些操作将永远不被允许。
            #  将其重新引发为 TypeError，这样与非 PyArrow 情况更一致吗？
            exc = pa.ArrowNotImplementedError
        else:
            exc = None
        # 返回最终的异常类型
        return exc
    # 获取与算术操作名和数据类型相关的 xfail 标记
    def _get_arith_xfail_marker(self, opname, pa_dtype):
        mark = None

        # 检查是否支持 Arrow 时间操作
        arrow_temporal_supported = self._is_temporal_supported(opname, pa_dtype)

        # 如果操作名是 "__rpow__" 并且数据类型是浮点型、整数型或者十进制型
        if opname == "__rpow__" and (
            pa.types.is_floating(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            # 创建 xfail 标记，用于标识问题的操作和原因
            mark = pytest.mark.xfail(
                reason=(
                    f"GH#29997: 1**pandas.NA == 1 while 1**pyarrow.NA == NULL "
                    f"for {pa_dtype}"
                )
            )
        # 如果支持 Arrow 时间操作，并且数据类型是时间类型或者是某些算术操作
        elif arrow_temporal_supported and (
            pa.types.is_time(pa_dtype)
            or (
                opname
                in ("__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__")
                and pa.types.is_duration(pa_dtype)
            )
        ):
            # 创建 xfail 标记，用于标识不支持的操作和原因
            mark = pytest.mark.xfail(
                raises=TypeError,
                reason=(
                    f"{opname} not supported between"
                    f"pd.NA and {pa_dtype} Python scalar"
                ),
            )
        # 如果操作名是 "__rfloordiv__" 并且数据类型是整数或者十进制型
        elif opname == "__rfloordiv__" and (
            pa.types.is_integer(pa_dtype) or pa.types.is_decimal(pa_dtype)
        ):
            # 创建 xfail 标记，用于标识抛出异常的操作和原因
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )
        # 如果操作名是 "__rtruediv__" 并且数据类型是十进制型
        elif opname == "__rtruediv__" and pa.types.is_decimal(pa_dtype):
            # 创建 xfail 标记，用于标识抛出异常的操作和原因
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )

        # 返回获取的标记
        return mark

    # 测试算术系列操作与标量值的组合
    def test_arith_series_with_scalar(self, data, all_arithmetic_operators, request):
        # 获取数据的 PyArrow 数据类型
        pa_dtype = data.dtype.pyarrow_dtype

        # 如果操作名是 "__rmod__" 并且数据类型是二进制型，则跳过测试
        if all_arithmetic_operators == "__rmod__" and pa.types.is_binary(pa_dtype):
            pytest.skip("Skip testing Python string formatting")
        # 如果操作名是 "__rmul__" 或 "__mul__" 并且数据类型是二进制型或字符串型
        elif all_arithmetic_operators in ("__rmul__", "__mul__") and (
            pa.types.is_binary(pa_dtype) or pa.types.is_string(pa_dtype)
        ):
            # 应用 xfail 标记，用于标识抛出异常的操作和原因
            request.applymarker(
                pytest.mark.xfail(
                    raises=TypeError, reason="Can only string multiply by an integer."
                )
            )

        # 获取算术 xfail 标记
        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        # 如果存在标记，则应用到测试请求中
        if mark is not None:
            request.applymarker(mark)

        # 调用父类方法执行算术操作与标量值的测试
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)
    # 对带标量的算术操作进行测试
    def test_arith_frame_with_scalar(self, data, all_arithmetic_operators, request):
        # 获取数据的PyArrow数据类型
        pa_dtype = data.dtype.pyarrow_dtype

        # 如果操作符为 "__rmod__" 并且数据类型是字符串或二进制，则跳过测试
        if all_arithmetic_operators == "__rmod__" and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            pytest.skip("Skip testing Python string formatting")
        # 如果操作符为 "__rmul__" 或 "__mul__" 并且数据类型是二进制或字符串，则标记为预期失败
        elif all_arithmetic_operators in ("__rmul__", "__mul__") and (
            pa.types.is_binary(pa_dtype) or pa.types.is_string(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    raises=TypeError, reason="Can only string multiply by an integer."
                )
            )

        # 根据操作符和数据类型获取标记，并应用到测试请求中
        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)

        # 调用父类的方法执行测试
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    # 对带数组的系列算术操作进行测试
    def test_arith_series_with_array(self, data, all_arithmetic_operators, request):
        # 获取数据的PyArrow数据类型
        pa_dtype = data.dtype.pyarrow_dtype

        # 如果操作符为 "__sub__" 或 "__rsub__" 并且数据类型是无符号整数，则标记为预期失败
        if all_arithmetic_operators in (
            "__sub__",
            "__rsub__",
        ) and pa.types.is_unsigned_integer(pa_dtype):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=(
                        f"Implemented pyarrow.compute.subtract_checked "
                        f"which raises on overflow for {pa_dtype}"
                    ),
                )
            )
        # 如果操作符为 "__rmul__" 或 "__mul__" 并且数据类型是二进制或字符串，则标记为预期失败
        elif all_arithmetic_operators in ("__rmul__", "__mul__") and (
            pa.types.is_binary(pa_dtype) or pa.types.is_string(pa_dtype)
        ):
            request.applymarker(
                pytest.mark.xfail(
                    raises=TypeError, reason="Can only string multiply by an integer."
                )
            )

        # 根据操作符和数据类型获取标记，并应用到测试请求中
        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.applymarker(mark)

        # 获取操作名，并创建一个与数据类型匹配的系列对象
        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        # 创建另一个系列，该系列的每个元素都与第一个系列的第一个元素相同
        other = pd.Series(pd.array([ser.iloc[0]] * len(ser), dtype=data.dtype))

        # 使用自定义方法检查操作名称及其效果
        self.check_opname(ser, op_name, other)

    # 对带扩展数组的系列加法操作进行测试
    def test_add_series_with_extension_array(self, data, request):
        # 获取数据的PyArrow数据类型
        pa_dtype = data.dtype.pyarrow_dtype

        # 如果数据类型是int8，则标记为预期失败
        if pa_dtype.equals("int8"):
            request.applymarker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=f"raises on overflow for {pa_dtype}",
                )
            )

        # 调用父类的方法执行测试
        super().test_add_series_with_extension_array(data)

    # 对不合法的比较操作进行测试
    def test_invalid_other_comp(self, data, comparison_op):
        # 使用pytest.raises检查是否会引发NotImplementedError异常，且匹配特定消息
        with pytest.raises(
            NotImplementedError, match=".* not implemented for <class 'object'>"
        ):
            comparison_op(data, object())
    @pytest.mark.parametrize("masked_dtype", ["boolean", "Int64", "Float64"])
    # 使用 pytest 的 parametrize 标记，为测试方法提供多组参数化输入，测试不同的数据类型
    def test_comp_masked_numpy(self, masked_dtype, comparison_op):
        # GH 52625
        # GH 52625 表示 GitHub 上的 issue 或 pull request 编号，这里可能是一个相关的问题或功能需求
        data = [1, 0, None]
        # 创建包含数据和指定数据类型的 Pandas Series 对象
        ser_masked = pd.Series(data, dtype=masked_dtype)
        # 使用 pyarrow 扩展的数据类型创建另一个 Pandas Series 对象
        ser_pa = pd.Series(data, dtype=f"{masked_dtype.lower()}[pyarrow]")
        # 调用传入的比较操作函数，对两个 Series 进行比较
        result = comparison_op(ser_pa, ser_masked)
        if comparison_op in [operator.lt, operator.gt, operator.ne]:
            # 如果比较操作是小于、大于或者不等于，则预期结果为指定的列表
            exp = [False, False, None]
        else:
            # 否则，预期结果为另一个指定的列表
            exp = [True, True, None]
        # 创建期望的结果 Series，指定其数据类型为 ArrowDtype 的布尔类型
        expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
        # 使用测试框架的方法来断言实际结果与期望结果是否相等
        tm.assert_series_equal(result, expected)
class TestLogicalOps:
    """Various Series and DataFrame logical ops methods."""

    def test_kleene_or(self):
        # 创建一个包含 True、False 和 None 的 Pandas Series，使用了 pyarrow 的 boolean 类型
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        # 创建另一个包含 True、False 和 None 的 Pandas Series，使用了 pyarrow 的 boolean 类型
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        # 对两个 Series 执行逻辑或运算
        result = a | b
        # 创建预期的 Pandas Series 结果，包含逻辑或运算后的期望值
        expected = pd.Series(
            [True, True, True, True, False, None, True, None, None],
            dtype="boolean[pyarrow]",
        )
        # 使用 Pandas 测试工具比较结果和预期值
        tm.assert_series_equal(result, expected)

        # 再次执行逻辑或运算，验证交换操作数是否产生相同的结果
        result = b | a
        tm.assert_series_equal(result, expected)

        # 确保原始 Series 没有被就地修改
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b, pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [True, None, None]),
            (pd.NA, [True, None, None]),
            (True, [True, True, True]),
            (np.bool_(True), [True, True, True]),
            (False, [True, False, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_or_scalar(self, other, expected):
        # 创建包含 True、False 和 None 的 Pandas Series，使用了 pyarrow 的 boolean 类型
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        # 对 Series 和标量值执行逻辑或运算
        result = a | other
        # 创建预期的 Pandas Series 结果，包含逻辑或运算后的期望值
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        # 使用 Pandas 测试工具比较结果和预期值
        tm.assert_series_equal(result, expected)

        # 再次执行逻辑或运算，验证交换操作数是否产生相同的结果
        result = other | a
        tm.assert_series_equal(result, expected)

        # 确保原始 Series 没有被就地修改
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    def test_kleene_and(self):
        # 创建一个包含 True、False 和 None 的 Pandas Series，使用了 pyarrow 的 boolean 类型
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        # 创建另一个包含 True、False 和 None 的 Pandas Series，使用了 pyarrow 的 boolean 类型
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        # 对两个 Series 执行逻辑与运算
        result = a & b
        # 创建预期的 Pandas Series 结果，包含逻辑与运算后的期望值
        expected = pd.Series(
            [True, False, None, False, False, False, None, False, None],
            dtype="boolean[pyarrow]",
        )
        # 使用 Pandas 测试工具比较结果和预期值
        tm.assert_series_equal(result, expected)

        # 再次执行逻辑与运算，验证交换操作数是否产生相同的结果
        result = b & a
        tm.assert_series_equal(result, expected)

        # 确保原始 Series 没有被就地修改
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b, pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        )

    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [None, False, None]),
            (pd.NA, [None, False, None]),
            (True, [True, False, None]),
            (False, [False, False, False]),
            (np.bool_(True), [True, False, None]),
            (np.bool_(False), [False, False, False]),
        ],
    )
    # 测试 Kleene 算子 & 和 Series 的按位与操作，验证结果与期望相等
    def test_kleene_and_scalar(self, other, expected):
        # 创建一个布尔类型的 Pandas Series，包含 True、False 和 None，使用 pyarrow 格式存储
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        # 对 Series a 和参数 other 执行按位与操作，存储结果到 result
        result = a & other
        # 创建一个期望的 Pandas Series，使用 pyarrow 格式存储
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        # 使用 Pandas 测试框架验证 result 和 expected 的内容是否相等
        tm.assert_series_equal(result, expected)

        # 再次对参数 other 和 Series a 执行按位与操作，存储结果到 result
        result = other & a
        # 使用 Pandas 测试框架验证 result 和 expected 的内容是否相等
        tm.assert_series_equal(result, expected)

        # 确保在任何地方都没有对原始 Series a 进行就地修改
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    # 测试 Kleene 算子 ^ 的操作
    def test_kleene_xor(self):
        # 创建两个布尔类型的 Pandas Series，a 包含 True、False 和 None，使用 pyarrow 格式存储，b 包含 True、False 和 None 的重复序列，使用 pyarrow 格式存储
        a = pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]")
        b = pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        # 对 Series a 和 b 执行按位异或操作，存储结果到 result
        result = a ^ b
        # 创建一个期望的 Pandas Series，使用 pyarrow 格式存储
        expected = pd.Series(
            [False, True, None, True, False, None, None, None, None],
            dtype="boolean[pyarrow]",
        )
        # 使用 Pandas 测试框架验证 result 和 expected 的内容是否相等
        tm.assert_series_equal(result, expected)

        # 再次对 Series b 和 a 执行按位异或操作，存储结果到 result
        result = b ^ a
        # 使用 Pandas 测试框架验证 result 和 expected 的内容是否相等
        tm.assert_series_equal(result, expected)

        # 确保在任何地方都没有对原始 Series a 和 b 进行就地修改
        tm.assert_series_equal(
            a,
            pd.Series([True] * 3 + [False] * 3 + [None] * 3, dtype="boolean[pyarrow]"),
        )
        tm.assert_series_equal(
            b, pd.Series([True, False, None] * 3, dtype="boolean[pyarrow]")
        )

    # 使用参数化测试，测试 Kleene 算子 ^ 与标量的操作
    @pytest.mark.parametrize(
        "other, expected",
        [
            (None, [None, None, None]),
            (pd.NA, [None, None, None]),
            (True, [False, True, None]),
            (np.bool_(True), [False, True, None]),
            (np.bool_(False), [True, False, None]),
        ],
    )
    def test_kleene_xor_scalar(self, other, expected):
        # 创建一个布尔类型的 Pandas Series，包含 True、False 和 None，使用 pyarrow 格式存储
        a = pd.Series([True, False, None], dtype="boolean[pyarrow]")
        # 对 Series a 和参数 other 执行按位异或操作，存储结果到 result
        result = a ^ other
        # 创建一个期望的 Pandas Series，使用 pyarrow 格式存储
        expected = pd.Series(expected, dtype="boolean[pyarrow]")
        # 使用 Pandas 测试框架验证 result 和 expected 的内容是否相等
        tm.assert_series_equal(result, expected)

        # 再次对参数 other 和 Series a 执行按位异或操作，存储结果到 result
        result = other ^ a
        # 使用 Pandas 测试框架验证 result 和 expected 的内容是否相等
        tm.assert_series_equal(result, expected)

        # 确保在任何地方都没有对原始 Series a 进行就地修改
        tm.assert_series_equal(
            a, pd.Series([True, False, None], dtype="boolean[pyarrow]")
        )

    # 使用参数化测试，测试逻辑运算函数 __and__、__or__ 和 __xor__ 的操作
    @pytest.mark.parametrize(
        "op, exp",
        [
            ["__and__", True],
            ["__or__", True],
            ["__xor__", False],
        ],
    )
    def test_logical_masked_numpy(self, op, exp):
        # 创建一个包含 True、False 和 None 的列表
        data = [True, False, None]
        # 创建两个布尔类型的 Pandas Series，一个使用标准 boolean 类型，另一个使用 pyarrow 格式存储
        ser_masked = pd.Series(data, dtype="boolean")
        ser_pa = pd.Series(data, dtype="boolean[pyarrow]")
        # 使用 getattr 函数动态调用 ser_pa 对象的 op 方法（即 __and__、__or__ 或 __xor__），传入 ser_masked 作为参数，存储结果到 result
        result = getattr(ser_pa, op)(ser_masked)
        # 创建一个期望的 Pandas Series，使用 ArrowDtype 指定数据类型
        expected = pd.Series([exp, False, None], dtype=ArrowDtype(pa.bool_()))
        # 使用 Pandas 测试框架验证 result 和 expected 的内容是否相等
        tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES)
# 使用 pytest.mark.parametrize 装饰器，对 pa_type 参数进行参数化测试，使用 tm.ALL_INT_PYARROW_DTYPES 中的所有整数类型
def test_bitwise(pa_type):
    # 测试函数 test_bitwise，用于测试位运算功能
    # GH 54495
    # GH 54495 GitHub 上的 issue 编号，用于跟踪相关问题
    dtype = ArrowDtype(pa_type)
    # 创建 ArrowDtype 对象，使用参数 pa_type
    left = pd.Series([1, None, 3, 4], dtype=dtype)
    # 创建左侧的 Pandas Series，包含特定类型和值
    right = pd.Series([None, 3, 5, 4], dtype=dtype)
    # 创建右侧的 Pandas Series，包含特定类型和值

    result = left | right
    # 执行位或操作，并将结果存储在 result 中
    expected = pd.Series([None, None, 3 | 5, 4 | 4], dtype=dtype)
    # 创建预期的 Pandas Series 结果，包含位或操作的预期结果
    tm.assert_series_equal(result, expected)
    # 使用 tm.assert_series_equal 进行结果断言比较

    result = left & right
    # 执行位与操作，并将结果存储在 result 中
    expected = pd.Series([None, None, 3 & 5, 4 & 4], dtype=dtype)
    # 创建预期的 Pandas Series 结果，包含位与操作的预期结果
    tm.assert_series_equal(result, expected)
    # 使用 tm.assert_series_equal 进行结果断言比较

    result = left ^ right
    # 执行位异或操作，并将结果存储在 result 中
    expected = pd.Series([None, None, 3 ^ 5, 4 ^ 4], dtype=dtype)
    # 创建预期的 Pandas Series 结果，包含位异或操作的预期结果
    tm.assert_series_equal(result, expected)
    # 使用 tm.assert_series_equal 进行结果断言比较

    result = ~left
    # 执行位非操作，并将结果存储在 result 中
    expected = ~(left.fillna(0).to_numpy())
    # 创建预期的结果，对 left 执行填充缺失值并转换为 NumPy 数组后再进行位非操作
    expected = pd.Series(expected, dtype=dtype).mask(left.isnull())
    # 将预期结果转换为 Pandas Series，并在 left 为空时应用掩码
    tm.assert_series_equal(result, expected)
    # 使用 tm.assert_series_equal 进行结果断言比较


def test_arrowdtype_construct_from_string_type_with_unsupported_parameters():
    # 测试 ArrowDtype.construct_from_string 方法对不支持的参数进行处理
    with pytest.raises(NotImplementedError, match="Passing pyarrow type"):
        ArrowDtype.construct_from_string("not_a_real_dype[s, tz=UTC][pyarrow]")

    with pytest.raises(NotImplementedError, match="Passing pyarrow type"):
        ArrowDtype.construct_from_string("decimal(7, 2)[pyarrow]")


def test_arrowdtype_construct_from_string_supports_dt64tz():
    # 测试 ArrowDtype.construct_from_string 方法支持 dt64tz 类型
    # as of GH#50689, timestamptz is supported
    # GH#50689 GitHub 上的 issue 编号，用于跟踪相关问题
    dtype = ArrowDtype.construct_from_string("timestamp[s, tz=UTC][pyarrow]")
    # 使用 ArrowDtype.construct_from_string 方法创建特定类型的 ArrowDtype
    expected = ArrowDtype(pa.timestamp("s", "UTC"))
    # 创建预期的 ArrowDtype 对象，包含特定的时间戳类型和时区
    assert dtype == expected
    # 断言创建的 dtype 与预期的 expected 相等


def test_arrowdtype_construct_from_string_type_only_one_pyarrow():
    # 测试 ArrowDtype.construct_from_string 方法对只包含一个 pyarrow 参数的情况
    # GH#51225
    # GH#51225 GitHub 上的 issue 编号，用于跟踪相关问题
    invalid = "int64[pyarrow]foobar[pyarrow]"
    # 创建一个不合法的 ArrowDtype 字符串
    msg = (
        r"Passing pyarrow type specific parameters \(\[pyarrow\]\) in the "
        r"string is not supported\."
    )
    # 设置异常消息字符串，用于比较错误消息
    with pytest.raises(NotImplementedError, match=msg):
        pd.Series(range(3), dtype=invalid)


def test_arrow_string_multiplication():
    # 测试 Pandas Series 字符串与整数的乘法操作
    # GH 56537
    # GH 56537 GitHub 上的 issue 编号，用于跟踪相关问题
    binary = pd.Series(["abc", "defg"], dtype=ArrowDtype(pa.string()))
    # 创建包含字符串的 Pandas Series，使用 ArrowDtype 类型
    repeat = pd.Series([2, -2], dtype="int64[pyarrow]")
    # 创建包含整数的 Pandas Series，指定 pyarrow 类型
    result = binary * repeat
    # 执行字符串与整数的乘法操作，并存储结果
    expected = pd.Series(["abcabc", ""], dtype=ArrowDtype(pa.string()))
    # 创建预期的 Pandas Series 结果，包含乘法操作的预期结果
    tm.assert_series_equal(result, expected)
    # 使用 tm.assert_series_equal 进行结果断言比较
    reflected_result = repeat * binary
    # 执行整数与字符串的乘法操作，并存储结果
    tm.assert_series_equal(result, reflected_result)
    # 使用 tm.assert_series_equal 进行结果断言比较


def test_arrow_string_multiplication_scalar_repeat():
    # 测试 Pandas Series 字符串与标量整数的乘法操作
    binary = pd.Series(["abc", "defg"], dtype=ArrowDtype(pa.string()))
    # 创建包含字符串的 Pandas Series，使用 ArrowDtype 类型
    result = binary * 2
    # 执行字符串与整数的乘法操作，并存储结果
    expected = pd.Series(["abcabc", "defgdefg"], dtype=ArrowDtype(pa.string()))
    # 创建预期的 Pandas Series 结果，包含乘法操作的预期结果
    tm.assert_series_equal(result, expected)
    # 使用 tm.assert_series_equal 进行结果断言比较
    reflected_result = 2 * binary
    # 执行整数与字符串的乘法操作，并存储结果
    tm.assert_series_equal(reflected_result, expected)
    # 使用 tm.assert_series_equal 进行结果断言比较


@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
# 使用 pytest.mark.parametrize 装饰器，对 interpolation 参数进行参数化测试
@pytest.mark.parametrize("quantile", [0.5, [0.5, 0.5]])
# 使用 pytest.mark.parametrize 装饰器，对 quantile 参数进行参数化测试
def test_quantile(data, interpolation, quantile, request):
    # 测试函数 test_quantile，用于测试分位数计算
    pa_dtype = data.dtype.pyarrow_dtype
    # 获取数据的 PyArrow 数据类型

    data = data.take([0, 0, 0])
    # 从数据中取出特定索引的数据
    ser = pd.Series(data)
    # 创建 Pandas Series 对象
    # 检查数据类型是否为字符串、二进制或布尔型，这些类型不期望使用分位数计算
    # 注意这与非 pyarrow 的行为相匹配
    msg = r"Function 'quantile' has no kernel matching input types \(.*\)"
    # 使用 pytest 的断言检查 ArrowNotImplementedError 异常，确保匹配特定的错误信息
    with pytest.raises(pa.ArrowNotImplementedError, match=msg):
        ser.quantile(q=quantile, interpolation=interpolation)
    # 函数提前返回
    return

    # 如果数据类型是整数、浮点数或者十进制数
    pass
    # 如果数据类型是时间类型
    pass

    # 如果数据类型不属于上述任何一种情况
    # 应用 pytest 的 xfail 标记，用于标记预期不通过的测试
    request.applymarker(
        pytest.mark.xfail(
            raises=pa.ArrowNotImplementedError,
            reason=f"quantile not supported by pyarrow for {pa_dtype}",
        )
    )

    # 从数据中取出第一个元素，构建 Pandas Series 对象
    data = data.take([0, 0, 0])
    ser = pd.Series(data)
    # 计算分位数
    result = ser.quantile(q=quantile, interpolation=interpolation)

    # 如果数据类型是时间戳，并且插值方式不是 "lower" 或 "higher"
    # 由于四舍五入误差可能导致以下检查失败
    # 例如 '2020-01-01 01:01:01.000001' 与 '2020-01-01 01:01:01.000001024'，
    # 因此我们首先检查是否与 numpy 的相似
    if pa.types.is_timestamp(pa_dtype) and interpolation not in ["lower", "higher"]:
        if pa_dtype.tz:
            pd_dtype = f"M8[{pa_dtype.unit}, {pa_dtype.tz}]"
        else:
            pd_dtype = f"M8[{pa_dtype.unit}]"
        # 将 Pandas Series 转换为指定的 Pandas 时间戳类型
        ser_np = ser.astype(pd_dtype)

        # 计算预期值的分位数
        expected = ser_np.quantile(q=quantile, interpolation=interpolation)
        if quantile == 0.5:
            # 如果分位数为 0.5，并且时间戳的单位是 "us"
            # 将预期值转换为 Python datetime 对象（忽略警告）
            if pa_dtype.unit == "us":
                expected = expected.to_pydatetime(warn=False)
            # 断言结果与预期值相等
            assert result == expected
        else:
            # 如果分位数不是 0.5，并且时间戳的单位是 "us"
            # 对预期值向下舍入到微秒级别
            if pa_dtype.unit == "us":
                expected = expected.dt.floor("us")
            # 使用 Pandas 测试工具断言 Series 对象相等
            tm.assert_series_equal(result, expected.astype(data.dtype))
        # 函数提前返回
        return

    # 如果分位数为 0.5
    assert result == data[0]
    # 如果分位数不是 0.5
    else:
        # 仅检查值是否相等
        # 创建一个包含第一个元素的 Pandas Series 对象，索引为 [0.5, 0.5]
        expected = pd.Series(data.take([0, 0]), index=[0.5, 0.5])
        # 如果数据类型是整数、浮点数或者十进制数
        # 将预期值和结果强制转换为 "float64[pyarrow]" 类型
        if (
            pa.types.is_integer(pa_dtype)
            or pa.types.is_floating(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            expected = expected.astype("float64[pyarrow]")
            result = result.astype("float64[pyarrow]")
        # 使用 Pandas 测试工具断言 Series 对象相等
        tm.assert_series_equal(result, expected)
# 使用 pytest 提供的参数化装饰器标记不同的测试用例
@pytest.mark.parametrize(
    "take_idx, exp_idx",
    # 定义测试参数：每个内部列表包含 take_idx 和 exp_idx 的值
    [[[0, 0, 2, 2, 4, 4], [4, 0]], [[0, 0, 0, 2, 4, 4], [0]]],
    # 为测试用例提供可读性标识
    ids=["multi_mode", "single_mode"],
)
def test_mode_dropna_true(data_for_grouping, take_idx, exp_idx):
    # 从 data_for_grouping 中选择指定索引的数据
    data = data_for_grouping.take(take_idx)
    # 将数据转换为 pandas 的 Series 对象
    ser = pd.Series(data)
    # 计算 Series 的众数，指定 dropna=True
    result = ser.mode(dropna=True)
    # 根据期望的索引值从 data_for_grouping 中获取数据并转换为 Series 对象
    expected = pd.Series(data_for_grouping.take(exp_idx))
    # 断言计算得到的结果与期望的结果相等
    tm.assert_series_equal(result, expected)


def test_mode_dropna_false_mode_na(data):
    # GH 50982
    # 创建一个包含更多 NaN 值的 Series 对象
    more_nans = pd.Series([None, None, data[0]], dtype=data.dtype)
    # 计算 Series 的众数，指定 dropna=False
    result = more_nans.mode(dropna=False)
    # 期望的结果是一个只包含 None 值的 Series 对象
    expected = pd.Series([None], dtype=data.dtype)
    # 断言计算得到的结果与期望的结果相等
    tm.assert_series_equal(result, expected)

    # 创建一个 Series 对象
    expected = pd.Series([data[0], None], dtype=data.dtype)
    # 计算 Series 的众数，指定 dropna=False
    result = expected.mode(dropna=False)
    # 断言计算得到的结果与期望的结果相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "arrow_dtype, expected_type",
    # 定义测试参数：每个内部列表包含 arrow_dtype 和 expected_type 的值
    [
        [pa.binary(), bytes],
        [pa.binary(16), bytes],
        [pa.large_binary(), bytes],
        [pa.large_string(), str],
        [pa.list_(pa.int64()), list],
        [pa.large_list(pa.int64()), list],
        [pa.map_(pa.string(), pa.int64()), list],
        [pa.struct([("f1", pa.int8()), ("f2", pa.string())]), dict],
        [pa.dictionary(pa.int64(), pa.int64()), CategoricalDtypeType],
    ],
)
def test_arrow_dtype_type(arrow_dtype, expected_type):
    # GH 51845
    # TODO: 当 arrow_dtype 存在于数据 fixture 中时，与 test_getitem_scalar 重复
    # 断言 ArrowDtype 对象的类型等于期望的类型
    assert ArrowDtype(arrow_dtype).type == expected_type


def test_is_bool_dtype():
    # GH 22667
    # 创建一个 ArrowExtensionArray 对象，包含 True、False 和 True 三个元素
    data = ArrowExtensionArray(pa.array([True, False, True]))
    # 断言该数据对象是布尔型的
    assert is_bool_dtype(data)
    # 断言该数据对象是布尔索引器
    assert pd.core.common.is_bool_indexer(data)
    # 创建一个 Series 对象
    s = pd.Series(range(len(data)))
    # 使用布尔数组进行索引，生成结果 Series 对象
    result = s[data]
    # 使用 np.asarray 将布尔数组转换为 ndarray 进行索引，生成期望的 Series 对象
    expected = s[np.asarray(data)]
    # 断言计算得到的结果与期望的结果相等
    tm.assert_series_equal(result, expected)


def test_is_numeric_dtype(data):
    # GH 50563
    # 获取数据的 pyarrow 类型
    pa_type = data.dtype.pyarrow_dtype
    # 判断数据是否是数值类型（浮点数、整数或者十进制数）
    if (
        pa.types.is_floating(pa_type)
        or pa.types.is_integer(pa_type)
        or pa.types.is_decimal(pa_type)
    ):
        # 断言数据是数值类型
        assert is_numeric_dtype(data)
    else:
        # 断言数据不是数值类型
        assert not is_numeric_dtype(data)


def test_is_integer_dtype(data):
    # GH 50667
    # 获取数据的 pyarrow 类型
    pa_type = data.dtype.pyarrow_dtype
    # 判断数据是否是整数类型
    if pa.types.is_integer(pa_type):
        # 断言数据是整数类型
        assert is_integer_dtype(data)
    else:
        # 断言数据不是整数类型
        assert not is_integer_dtype(data)


def test_is_signed_integer_dtype(data):
    # 获取数据的 pyarrow 类型
    pa_type = data.dtype.pyarrow_dtype
    # 判断数据是否是有符号整数类型
    if pa.types.is_signed_integer(pa_type):
        # 断言数据是有符号整数类型
        assert is_signed_integer_dtype(data)
    else:
        # 断言数据不是有符号整数类型
        assert not is_signed_integer_dtype(data)


def test_is_unsigned_integer_dtype(data):
    # 获取数据的 pyarrow 类型
    pa_type = data.dtype.pyarrow_dtype
    # 判断数据是否是无符号整数类型
    if pa.types.is_unsigned_integer(pa_type):
        # 断言数据是无符号整数类型
        assert is_unsigned_integer_dtype(data)
    else:
        # 断言数据不是无符号整数类型
        assert not is_unsigned_integer_dtype(data)


def test_is_datetime64_any_dtype(data):
    # 获取数据的 pyarrow 类型
    pa_type = data.dtype.pyarrow_dtype
    # 如果给定的数据类型是时间戳或日期类型
    if pa.types.is_timestamp(pa_type) or pa.types.is_date(pa_type):
        # 确保数据中存在任何一种 datetime64 数据类型
        assert is_datetime64_any_dtype(data)
    # 否则，即数据类型不是时间戳或日期类型
    else:
        # 确保数据中不存在任何 datetime64 数据类型
        assert not is_datetime64_any_dtype(data)
def test_is_float_dtype(data):
    # 获取数据的 PyArrow 数据类型
    pa_type = data.dtype.pyarrow_dtype
    # 检查 PyArrow 数据类型是否为浮点数类型
    if pa.types.is_floating(pa_type):
        # 断言数据确实是浮点数类型
        assert is_float_dtype(data)
    else:
        # 断言数据不是浮点数类型
        assert not is_float_dtype(data)


def test_pickle_roundtrip(data):
    # GH 42600
    # 创建预期的 Pandas Series 对象
    expected = pd.Series(data)
    # 取预期 Series 的前两个元素
    expected_sliced = expected.head(2)
    # 对完整的 Series 对象进行 pickle 序列化
    full_pickled = pickle.dumps(expected)
    # 对切片后的 Series 对象进行 pickle 序列化
    sliced_pickled = pickle.dumps(expected_sliced)

    # 断言完整序列化后的字节长度大于切片序列化后的字节长度
    assert len(full_pickled) > len(sliced_pickled)

    # 反序列化完整 pickle 后的结果，并进行比较
    result = pickle.loads(full_pickled)
    tm.assert_series_equal(result, expected)

    # 反序列化切片 pickle 后的结果，并进行比较
    result_sliced = pickle.loads(sliced_pickled)
    tm.assert_series_equal(result_sliced, expected_sliced)


def test_astype_from_non_pyarrow(data):
    # GH49795
    # 将 PyArrow Array 转为 Pandas Array
    pd_array = data._pa_array.to_pandas().array
    # 将 Pandas Array 转换为指定的数据类型
    result = pd_array.astype(data.dtype)
    # 断言 Pandas Array 的 dtype 不是 ArrowDtype 类型
    assert not isinstance(pd_array.dtype, ArrowDtype)
    # 断言转换后的结果 dtype 是 ArrowDtype 类型
    assert isinstance(result.dtype, ArrowDtype)
    # 使用 Pandas 测试工具比较转换结果
    tm.assert_extension_array_equal(result, data)


def test_astype_float_from_non_pyarrow_str():
    # GH50430
    # 创建包含字符串 "1.0" 的 Pandas Series 对象
    ser = pd.Series(["1.0"])
    # 将 Series 转换为指定的数据类型
    result = ser.astype("float64[pyarrow]")
    # 创建期望的 Pandas Series 对象，包含数值类型为 float64 的数据
    expected = pd.Series([1.0], dtype="float64[pyarrow]")
    # 使用 Pandas 测试工具比较转换结果
    tm.assert_series_equal(result, expected)


def test_astype_errors_ignore():
    # GH 55399
    # 创建期望的 DataFrame 对象，包含整数列 "col"，数据类型为 int32[pyarrow]
    expected = pd.DataFrame({"col": [17000000]}, dtype="int32[pyarrow]")
    # 将 DataFrame 转换为指定的数据类型，忽略错误
    result = expected.astype("float[pyarrow]", errors="ignore")
    # 使用 Pandas 测试工具比较转换结果
    tm.assert_frame_equal(result, expected)


def test_to_numpy_with_defaults(data):
    # GH49973
    # 将 Series 或 DataFrame 转换为 NumPy 数组
    result = data.to_numpy()

    # 获取数据的 PyArrow 数据类型
    pa_type = data._pa_array.type
    # 如果数据类型是时长或时间戳，则跳过测试
    if pa.types.is_duration(pa_type) or pa.types.is_timestamp(pa_type):
        pytest.skip("Tested in test_to_numpy_temporal")
    # 如果数据类型是日期，则转换为数组
    elif pa.types.is_date(pa_type):
        expected = np.array(list(data))
    else:
        expected = np.array(data._pa_array)

    # 如果数据中包含缺失值且不是数值类型，则转换为对象数组，并将缺失值替换为 NA
    if data._hasna and not is_numeric_dtype(data.dtype):
        expected = expected.astype(object)
        expected[pd.isna(data)] = pd.NA

    # 使用 Pandas 测试工具比较转换结果
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_int_with_na():
    # GH51227: ensure to_numpy does not convert int to float
    # 创建包含整数和缺失值的数组
    data = [1, None]
    arr = pd.array(data, dtype="int64[pyarrow]")
    # 将数组转换为 NumPy 数组
    result = arr.to_numpy()
    # 创建期望的 NumPy 数组，包含整数和 NaN 值
    expected = np.array([1, np.nan])
    # 断言转换后的结果第一个元素类型为 float
    assert isinstance(result[0], float)
    # 使用 Pandas 测试工具比较转换结果
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize("na_val, exp", [(lib.no_default, np.nan), (1, 1)])
def test_to_numpy_null_array(na_val, exp):
    # GH#52443
    # 创建包含 NA 值的 null[pyarrow] 类型数组
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    # 将数组转换为指定数据类型的 NumPy 数组，同时指定 NA 值的替换值
    result = arr.to_numpy(dtype="float64", na_value=na_val)
    # 创建期望的 NumPy 数组，包含指定替换值的数据
    expected = np.array([exp] * 2, dtype="float64")
    # 使用 Pandas 测试工具比较转换结果
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_null_array_no_dtype():
    # GH#52443
    # 创建包含 NA 值的 null[pyarrow] 类型数组
    arr = pd.array([pd.NA, pd.NA], dtype="null[pyarrow]")
    # 将数组转换为对象数组，保留 NA 值
    result = arr.to_numpy(dtype=None)
    # 创建期望的 NumPy 数组，包含对象类型的 NA 值
    expected = np.array([pd.NA] * 2, dtype="object")
    # 使用 Pandas 测试工具比较转换结果
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_without_dtype():
    # GH 54808
    # 此测试用例待补充具体内容，暂无注释
    pass
    # 创建一个 Pandas 的 ExtensionArray 对象，包含布尔类型数据和缺失值
    arr = pd.array([True, pd.NA], dtype="boolean[pyarrow]")
    # 将 Pandas ExtensionArray 转换为 NumPy 数组，将缺失值替换为 False
    result = arr.to_numpy(na_value=False)
    # 期望的 NumPy 数组，包含两个元素 True 和 False
    expected = np.array([True, False], dtype=np.bool_)
    # 使用 Pandas 测试模块验证 result 和 expected 数组相等
    tm.assert_numpy_array_equal(result, expected)
    
    # 创建一个 Pandas 的 ExtensionArray 对象，包含浮点数和缺失值
    arr = pd.array([1.0, pd.NA], dtype="float32[pyarrow]")
    # 将 Pandas ExtensionArray 转换为 NumPy 数组，将缺失值替换为 0.0
    result = arr.to_numpy(na_value=0.0)
    # 期望的 NumPy 数组，包含两个元素 1.0 和 0.0
    expected = np.array([1.0, 0.0], dtype=np.float32)
    # 使用 Pandas 测试模块验证 result 和 expected 数组相等
    tm.assert_numpy_array_equal(result, expected)
# 复制数据以进行测试
orig = data.copy()

# 将 result 数组的所有元素替换为 data 数组的第一个元素
result[:] = data[0]
# 创建一个期望结果的 Arrow 扩展数组，所有元素为 data[0]
expected = ArrowExtensionArray._from_sequence(
    [data[0]] * len(data),
    dtype=data.dtype,
)
# 断言 result 与期望结果 expected 相等
tm.assert_extension_array_equal(result, expected)

# 复制数据以进行测试
result = orig.copy()
# 将 result 数组的所有元素替换为 data 数组的逆序
result[:] = data[::-1]
# 创建一个期望结果，所有元素为 data 数组的逆序
expected = data[::-1]
# 断言 result 与期望结果 expected 相等
tm.assert_extension_array_equal(result, expected)

# 复制数据以进行测试
result = orig.copy()
# 将 result 数组的所有元素替换为 data 数组转换为列表后的值
result[:] = data.tolist()
# 创建一个期望结果，所有元素为原始的 data 数组
expected = data
# 断言 result 与期望结果 expected 相等
tm.assert_extension_array_equal(result, expected)


# 检查数据的数据类型是否有效
pa_type = data._pa_array.type
if pa.types.is_string(pa_type) or pa.types.is_binary(pa_type):
    # 如果数据类型是字符串或二进制，则填充值为 123，错误类型为 TypeError
    fill_value = 123
    err = TypeError
    msg = "Invalid value '123' for dtype"
elif (
    pa.types.is_integer(pa_type)
    or pa.types.is_floating(pa_type)
    or pa.types.is_boolean(pa_type)
):
    # 如果数据类型是整数、浮点数或布尔型，则填充值为 "foo"，错误类型为 ArrowInvalid
    fill_value = "foo"
    err = pa.ArrowInvalid
    msg = "Could not convert"
else:
    # 其他情况下，填充值为 "foo"，错误类型为 TypeError
    fill_value = "foo"
    err = TypeError
    msg = "Invalid value 'foo' for dtype"
# 使用 pytest 断言检查设置数据的切片是否会引发预期的错误类型及消息
with pytest.raises(err, match=msg):
    data[:] = fill_value


# 创建一个日期数组
date_array = pa.array(
    [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")], type=pa.date32()
)
# 将日期数组转换为 Pandas Series，指定类型映射为将 pa.date32() 映射到 ArrowDtype(pa.date64())
result = date_array.to_pandas(
    types_mapper={pa.date32(): ArrowDtype(pa.date64())}.get
)
# 创建一个期望结果的 Pandas Series，数据类型为 ArrowDtype(pa.date64())
expected = pd.Series(
    [pd.Timestamp("2019-12-31"), pd.Timestamp("2019-12-31")],
    dtype=ArrowDtype(pa.date64()),
)
# 断言 result 与期望结果 expected 相等
tm.assert_series_equal(result, expected)


# 创建一个浮点数数组
array = pa.array([1.5, 2.5], type=pa.float64())
# 使用 pytest 断言检查将数组转换为 Pandas 时，是否会引发预期的 ArrowInvalid 错误，并包含特定消息
with pytest.raises(pa.ArrowInvalid, match="Float value 1.5 was truncated"):
    array.to_pandas(types_mapper={pa.float64(): ArrowDtype(pa.int64())}.get)


# 创建一个浮点数 Series
dtype = "float64[pyarrow]"
ser = pd.Series([0.0, 1.23, 2.56, pd.NA], dtype=dtype)
# 对 Series 执行小数点后一位的四舍五入操作
result = ser.round(1)
# 创建一个期望结果的 Series，数据经过小数点后一位的四舍五入
expected = pd.Series([0.0, 1.2, 2.6, pd.NA], dtype=dtype)
# 断言 result 与期望结果 expected 相等
tm.assert_series_equal(result, expected)

# 创建一个浮点数 Series
ser = pd.Series([123.4, pd.NA, 56.78], dtype=dtype)
# 对 Series 执行向最接近的 10 的倍数取整的操作
result = ser.round(-1)
# 创建一个期望结果的 Series，数据经过向最接近的 10 的倍数取整的操作
expected = pd.Series([120.0, pd.NA, 60.0], dtype=dtype)
# 断言 result 与期望结果 expected 相等
tm.assert_series_equal(result, expected)


# 对排序数据执行搜索操作，检查是否引发预期的 ValueError 异常
b, c, a = data_for_sorting
arr = data_for_sorting.take([2, 0, 1])  # 以 [a, b, c] 的顺序重新排列
arr[-1] = pd.NA  # 将最后一个元素设置为 NA

if as_series:
    arr = pd.Series(arr)

msg = (
    "searchsorted requires array to be sorted, "
    "which is impossible with NAs present."
)
# 使用 pytest 断言检查搜索操作是否会引发预期的 ValueError 异常，并包含特定消息
with pytest.raises(ValueError, match=msg):
    arr.searchsorted(b)


# 测试排序值为字典时的函数
    # 创建一个 Pandas DataFrame 对象 df，其中包含两列：
    # - 列 "a" 包含两个字符串元素 'x' 和 'y'，数据类型为 ArrowDtype，指定为字典类型，键为 int32 类型，值为 string 类型
    # - 列 "b" 包含两个整数元素 1 和 2
    df = pd.DataFrame(
        {
            "a": pd.Series(
                ["x", "y"], dtype=ArrowDtype(pa.dictionary(pa.int32(), pa.string()))
            ),
            "b": [1, 2],
        },
    )
    
    # 复制 DataFrame df 到变量 expected，作为预期结果
    expected = df.copy()
    
    # 对 DataFrame df 按列 "a" 和 "b" 进行排序，生成排序后的新 DataFrame 到变量 result
    result = df.sort_values(by=["a", "b"])
    
    # 使用测试框架中的方法 tm.assert_frame_equal 检查 result 是否与 expected 相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("pat", ["abc", "a[a-z]{2}"])
def test_str_count(pat):
    # 创建包含两个元素的 Pandas Series 对象，一个包含字符串 "abc"，另一个为 None
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 对 Series 中的每个字符串元素，使用给定的模式进行计数操作
    result = ser.str.count(pat)
    # 创建期望的 Pandas Series 对象，期望结果中包含计数结果和对应的数据类型
    expected = pd.Series([1, None], dtype=ArrowDtype(pa.int32()))
    # 使用 Pandas 提供的测试工具，比较实际结果和期望结果是否相等
    tm.assert_series_equal(result, expected)


def test_str_count_flags_unsupported():
    # 创建包含两个元素的 Pandas Series 对象，一个包含字符串 "abc"，另一个为 None
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 使用 pytest 引发 NotImplementedError 异常，测试 str.count 方法不支持 flags 参数
    with pytest.raises(NotImplementedError, match="count not"):
        ser.str.count("abc", flags=1)


@pytest.mark.parametrize(
    "side, str_func", [["left", "rjust"], ["right", "ljust"], ["both", "center"]]
)
def test_str_pad(side, str_func):
    # 创建包含一个元素的 Pandas Series 对象，元素为字符串 "a"
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    # 对 Series 中的字符串进行填充操作，填充方式由 side 参数指定，填充字符为 "x"
    result = ser.str.pad(width=3, side=side, fillchar="x")
    # 创建期望的 Pandas Series 对象，期望结果中包含填充后的字符串
    expected = pd.Series(
        [getattr("a", str_func)(3, "x"), None], dtype=ArrowDtype(pa.string())
    )
    # 使用 Pandas 提供的测试工具，比较实际结果和期望结果是否相等
    tm.assert_series_equal(result, expected)


def test_str_pad_invalid_side():
    # 创建包含一个元素的 Pandas Series 对象，元素为字符串 "a"
    ser = pd.Series(["a", None], dtype=ArrowDtype(pa.string()))
    # 使用 pytest 引发 ValueError 异常，测试 str.pad 方法不支持的填充方向参数 "foo"
    with pytest.raises(ValueError, match="Invalid side: foo"):
        ser.str.pad(3, "foo", "x")


@pytest.mark.parametrize(
    "pat, case, na, regex, exp",
    [
        ["ab", False, None, False, [True, None]],
        ["Ab", True, None, False, [False, None]],
        ["ab", False, True, False, [True, True]],
        ["a[a-z]{1}", False, None, True, [True, None]],
        ["A[a-z]{1}", True, None, True, [False, None]],
    ],
)
def test_str_contains(pat, case, na, regex, exp):
    # 创建包含两个元素的 Pandas Series 对象，一个包含字符串 "abc"，另一个为 None
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 对 Series 中的每个字符串元素，使用给定的参数进行包含操作
    result = ser.str.contains(pat, case=case, na=na, regex=regex)
    # 创建期望的 Pandas Series 对象，期望结果中包含布尔型的包含结果
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    # 使用 Pandas 提供的测试工具，比较实际结果和期望结果是否相等
    tm.assert_series_equal(result, expected)


def test_str_contains_flags_unsupported():
    # 创建包含两个元素的 Pandas Series 对象，一个包含字符串 "abc"，另一个为 None
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 使用 pytest 引发 NotImplementedError 异常，测试 str.contains 方法不支持 flags 参数
    with pytest.raises(NotImplementedError, match="contains not"):
        ser.str.contains("a", flags=1)


@pytest.mark.parametrize(
    "side, pat, na, exp",
    [
        ["startswith", "ab", None, [True, None, False]],
        ["startswith", "b", False, [False, False, False]],
        ["endswith", "b", True, [False, True, False]],
        ["endswith", "bc", None, [True, None, False]],
        ["startswith", ("a", "e", "g"), None, [True, None, True]],
        ["endswith", ("a", "c", "g"), None, [True, None, True]],
        ["startswith", (), None, [False, None, False]],
        ["endswith", (), None, [False, None, False]],
    ],
)
def test_str_start_ends_with(side, pat, na, exp):
    # 创建包含三个元素的 Pandas Series 对象，分别包含字符串 "abc", None, "efg"
    ser = pd.Series(["abc", None, "efg"], dtype=ArrowDtype(pa.string()))
    # 对 Series 中的字符串进行以给定模式开头或结尾的操作，操作由 side 参数指定
    result = getattr(ser.str, side)(pat, na=na)
    # 创建期望的 Pandas Series 对象，期望结果中包含布尔型的开头或结尾判断结果
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    # 使用 Pandas 提供的测试工具，比较实际结果和期望结果是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("side", ("startswith", "endswith"))
def test_str_starts_ends_with_all_nulls_empty_tuple(side):
    # 创建包含两个元素的 Pandas Series 对象，两个元素均为 None
    ser = pd.Series([None, None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, side)(())
    # 使用 getattr() 函数根据字符串属性名 side 动态调用 ser.str 对象的方法，并传入空元组作为参数
    # 这里假设 ser 是一个 Pandas Series 对象，ser.str 是其 str 访问器，side 是一个字符串变量表示要调用的方法名

    # 创建一个预期的 Pandas Series 对象，包含两个 None 值，数据类型为 ArrowDtype(pa.bool_())
    expected = pd.Series([None, None], dtype=ArrowDtype(pa.bool_()))
    
    # 使用 pandas.testing 模块中的 assert_series_equal 函数，比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "arg_name, arg",
    [["pat", re.compile("b")], ["repl", str], ["case", False], ["flags", 1]],
)
# 定义参数化测试函数，用于测试 str.replace 方法中不支持的参数组合
def test_str_replace_unsupported(arg_name, arg):
    # 创建一个包含字符串和空值的 Pandas Series，指定数据类型为 ArrowDtype
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 初始化关键字参数字典
    kwargs = {"pat": "b", "repl": "x", "regex": True}
    # 设置测试中的特定参数及其值
    kwargs[arg_name] = arg
    # 使用 pytest 断言捕获 NotImplementedError 异常，检查是否包含特定错误信息
    with pytest.raises(NotImplementedError, match="replace is not supported"):
        # 调用 Series 的 str.replace 方法，并传入关键字参数
        ser.str.replace(**kwargs)


@pytest.mark.parametrize(
    "pat, repl, n, regex, exp",
    [
        ["a", "x", -1, False, ["xbxc", None]],
        ["a", "x", 1, False, ["xbac", None]],
        ["[a-b]", "x", -1, True, ["xxxc", None]],
    ],
)
# 定义参数化测试函数，测试 Series 的 str.replace 方法
def test_str_replace(pat, repl, n, regex, exp):
    # 创建一个包含字符串和空值的 Pandas Series，指定数据类型为 ArrowDtype
    ser = pd.Series(["abac", None], dtype=ArrowDtype(pa.string()))
    # 调用 Series 的 str.replace 方法，获取实际结果
    result = ser.str.replace(pat, repl, n=n, regex=regex)
    # 创建预期结果的 Pandas Series
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    # 使用 Pandas 测试工具包中的 assert_series_equal 检查实际结果与预期结果是否一致
    tm.assert_series_equal(result, expected)


def test_str_replace_negative_n():
    # GH 56404
    # 创建一个包含字符串的 Pandas Series，指定数据类型为 ArrowDtype
    ser = pd.Series(["abc", "aaaaaa"], dtype=ArrowDtype(pa.string()))
    # 调用 Series 的 str.replace 方法，测试负数 n 值
    actual = ser.str.replace("a", "", -3, True)
    # 创建预期结果的 Pandas Series
    expected = pd.Series(["bc", ""], dtype=ArrowDtype(pa.string()))
    # 使用 Pandas 测试工具包中的 assert_series_equal 检查实际结果与预期结果是否一致
    tm.assert_series_equal(expected, actual)


def test_str_repeat_unsupported():
    # 创建一个包含字符串和空值的 Pandas Series，指定数据类型为 ArrowDtype
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 使用 pytest 断言捕获 NotImplementedError 异常，检查是否包含特定错误信息
    with pytest.raises(NotImplementedError, match="repeat is not"):
        # 调用 Series 的 str.repeat 方法，测试不支持的重复操作
        ser.str.repeat([1, 2])


def test_str_repeat():
    # 创建一个包含字符串和空值的 Pandas Series，指定数据类型为 ArrowDtype
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 调用 Series 的 str.repeat 方法，测试字符串重复
    result = ser.str.repeat(2)
    # 创建预期结果的 Pandas Series
    expected = pd.Series(["abcabc", None], dtype=ArrowDtype(pa.string()))
    # 使用 Pandas 测试工具包中的 assert_series_equal 检查实际结果与预期结果是否一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, case, na, exp",
    [
        ["ab", False, None, [True, None]],
        ["Ab", True, None, [False, None]],
        ["bc", True, None, [False, None]],
        ["ab", False, True, [True, True]],
        ["a[a-z]{1}", False, None, [True, None]],
        ["A[a-z]{1}", True, None, [False, None]],
    ],
)
# 定义参数化测试函数，测试 Series 的 str.match 方法
def test_str_match(pat, case, na, exp):
    # 创建一个包含字符串和空值的 Pandas Series，指定数据类型为 ArrowDtype
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 调用 Series 的 str.match 方法，获取实际结果
    result = ser.str.match(pat, case=case, na=na)
    # 创建预期结果的 Pandas Series
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    # 使用 Pandas 测试工具包中的 assert_series_equal 检查实际结果与预期结果是否一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "pat, case, na, exp",
    [
        ["abc", False, None, [True, True, False, None]],
        ["Abc", True, None, [False, False, False, None]],
        ["bc", True, None, [False, False, False, None]],
        ["ab", False, None, [True, True, False, None]],
        ["a[a-z]{2}", False, None, [True, True, False, None]],
        ["A[a-z]{1}", True, None, [False, False, False, None]],
        # GH Issue: #56652
        ["abc$", False, None, [True, False, False, None]],
        ["abc\\$", False, None, [False, True, False, None]],
        ["Abc$", True, None, [False, False, False, None]],
        ["Abc\\$", True, None, [False, False, False, None]],
    ],
)
# 定义参数化测试函数，测试 Series 的 str.fullmatch 方法
def test_str_fullmatch(pat, case, na, exp):
    # 创建一个 Pandas Series 对象，包含字符串和 None 值，使用 ArrowDtype 指定数据类型为字符串
    ser = pd.Series(["abc", "abc$", "$abc", None], dtype=ArrowDtype(pa.string()))
    # 对 Series 中的每个字符串元素执行 match 操作，匹配给定的模式 pat，可选参数包括 case 敏感性和处理缺失值 na
    result = ser.str.match(pat, case=case, na=na)
    # 创建一个预期的 Pandas Series 对象，包含布尔值，使用 ArrowDtype 指定数据类型为布尔值
    expected = pd.Series(exp, dtype=ArrowDtype(pa.bool_()))
    # 使用测试工具 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "sub, start, end, exp, exp_type",
    [
        ["ab", 0, None, [0, None], pa.int32()],
        ["bc", 1, 3, [1, None], pa.int64()],
        ["ab", 1, 3, [-1, None], pa.int64()],
        ["ab", -3, -3, [-1, None], pa.int64()],
    ],
)
def test_str_find(sub, start, end, exp, exp_type):
    # 创建包含多个参数化测试用例的测试函数
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 调用 Series 对象的 str.find 方法，查找子字符串出现的位置
    result = ser.str.find(sub, start=start, end=end)
    # 创建预期结果的 Series 对象
    expected = pd.Series(exp, dtype=ArrowDtype(exp_type))
    # 使用测试框架的方法比较结果和预期结果是否相等
    tm.assert_series_equal(result, expected)


def test_str_find_negative_start():
    # GH 56411: 针对负起始位置的测试用例
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="b", start=-1000, end=3)
    expected = pd.Series([1, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_str_find_no_end():
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    if pa_version_under13p0:
        # 处理 Arrow 版本低于 1.13 的异常情况
        with pytest.raises(pa.lib.ArrowInvalid, match="Negative buffer resize"):
            ser.str.find("ab", start=1)
    else:
        # 调用 Series 对象的 str.find 方法，查找子字符串出现的位置
        result = ser.str.find("ab", start=1)
        expected = pd.Series([-1, None], dtype="int64[pyarrow]")
        tm.assert_series_equal(result, expected)


def test_str_find_negative_start_negative_end():
    # GH 56791: 针对负起始和负结束位置的测试用例
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=-6, end=-3)
    expected = pd.Series([3, None], dtype=ArrowDtype(pa.int64()))
    tm.assert_series_equal(result, expected)


def test_str_find_large_start():
    # GH 56791: 针对大起始位置的测试用例
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    if pa_version_under13p0:
        # 处理 Arrow 版本低于 1.13 的异常情况
        with pytest.raises(pa.lib.ArrowInvalid, match="Negative buffer resize"):
            ser.str.find(sub="d", start=16)
    else:
        # 调用 Series 对象的 str.find 方法，查找子字符串出现的位置
        result = ser.str.find(sub="d", start=16)
        expected = pd.Series([-1, None], dtype=ArrowDtype(pa.int64()))
        tm.assert_series_equal(result, expected)


@pytest.mark.skipif(
    pa_version_under13p0, reason="https://github.com/apache/arrow/issues/36311"
)
@pytest.mark.parametrize("start", [-15, -3, 0, 1, 15, None])
@pytest.mark.parametrize("end", [-15, -1, 0, 3, 15, None])
@pytest.mark.parametrize("sub", ["", "az", "abce", "a", "caa"])
def test_str_find_e2e(start, end, sub):
    # 对于端到端测试的参数化测试用例
    s = pd.Series(
        ["abcaadef", "abc", "abcdeddefgj8292", "ab", "a", ""],
        dtype=ArrowDtype(pa.string()),
    )
    object_series = s.astype(pd.StringDtype())
    # 调用 Series 对象的 str.find 方法，查找子字符串出现的位置
    result = s.str.find(sub, start, end)
    expected = object_series.str.find(sub, start, end).astype(result.dtype)
    tm.assert_series_equal(result, expected)


def test_str_find_negative_start_negative_end_no_match():
    # GH 56791: 针对负起始和负结束位置但无匹配的测试用例
    ser = pd.Series(["abcdefg", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.find(sub="d", start=-3, end=-6)
    # 创建一个期望的 Pandas Series 对象，包含两个元素：-1 和 None
    expected = pd.Series([-1, None], dtype=ArrowDtype(pa.int64()))
    # 使用 pytest 的 tm 模块断言两个 Series 对象 result 和 expected 相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "i, exp",
    [
        [1, ["b", "e", None]],  # 参数化测试：测试索引为1时的期望结果
        [-1, ["c", "e", None]],  # 参数化测试：测试索引为-1时的期望结果
        [2, ["c", None, None]],  # 参数化测试：测试索引为2时的期望结果
        [-3, ["a", None, None]],  # 参数化测试：测试索引为-3时的期望结果
        [4, [None, None, None]],  # 参数化测试：测试索引为4时的期望结果
    ],
)
def test_str_get(i, exp):
    ser = pd.Series(["abc", "de", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.get(i)  # 获取指定索引处的字符串片段
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等


@pytest.mark.xfail(
    reason="TODO: StringMethods._validate should support Arrow list types",  # 标记为预期失败，因为StringMethods._validate尚不支持Arrow列表类型
    raises=AttributeError,
)
def test_str_join():
    ser = pd.Series(ArrowExtensionArray(pa.array([list("abc"), list("123"), None])))
    result = ser.str.join("=")  # 将字符串数组中的每个字符串连接起来，用'='分隔
    expected = pd.Series(["a=b=c", "1=2=3", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等


def test_str_join_string_type():
    ser = pd.Series(ArrowExtensionArray(pa.array(["abc", "123", None])))
    result = ser.str.join("=")  # 将字符串数组中的每个字符串连接起来，用'='分隔
    expected = pd.Series(["a=b=c", "1=2=3", None], dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等


@pytest.mark.parametrize(
    "start, stop, step, exp",
    [
        [None, 2, None, ["ab", None]],  # 参数化测试：测试从开头到索引2的字符串片段
        [None, 2, 1, ["ab", None]],  # 参数化测试：测试从开头到索引2，步长为1的字符串片段
        [1, 3, 1, ["bc", None]],  # 参数化测试：测试从索引1到索引3，步长为1的字符串片段
    ],
)
def test_str_slice(start, stop, step, exp):
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.slice(start, stop, step)  # 切片操作，从序列中获取指定的子序列
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等


@pytest.mark.parametrize(
    "start, stop, repl, exp",
    [
        [1, 2, "x", ["axcd", None]],  # 参数化测试：测试用'x'替换从索引1到索引2的子串
        [None, 2, "x", ["xcd", None]],  # 参数化测试：测试用'x'替换从开头到索引2的子串
        [None, 2, None, ["cd", None]],  # 参数化测试：测试删除从开头到索引2的子串
    ],
)
def test_str_slice_replace(start, stop, repl, exp):
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    result = ser.str.slice_replace(start, stop, repl)  # 替换字符串中的片段
    expected = pd.Series(exp, dtype=ArrowDtype(pa.string()))
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等


@pytest.mark.parametrize(
    "value, method, exp",
    [
        ["a1c", "isalnum", True],  # 参数化测试：测试字符串是否为字母数字字符
        ["!|,", "isalnum", False],  # 参数化测试：测试字符串是否为字母数字字符
        ["aaa", "isalpha", True],  # 参数化测试：测试字符串是否全为字母
        ["!!!", "isalpha", False],  # 参数化测试：测试字符串是否全为字母
        ["٠", "isdecimal", True],  # 参数化测试：测试字符串是否为十进制数字字符
        ["~!", "isdecimal", False],  # 参数化测试：测试字符串是否为十进制数字字符
        ["2", "isdigit", True],  # 参数化测试：测试字符串是否全为数字
        ["~", "isdigit", False],  # 参数化测试：测试字符串是否全为数字
        ["aaa", "islower", True],  # 参数化测试：测试字符串是否全为小写字母
        ["aaA", "islower", False],  # 参数化测试：测试字符串是否全为小写字母
        ["123", "isnumeric", True],  # 参数化测试：测试字符串是否全为数字字符
        ["11I", "isnumeric", False],  # 参数化测试：测试字符串是否全为数字字符
        [" ", "isspace", True],  # 参数化测试：测试字符串是否全为空格字符
        ["", "isspace", False],  # 参数化测试：测试字符串是否全为空格字符
        ["The That", "istitle", True],  # 参数化测试：测试字符串是否每个单词首字母大写
        ["the That", "istitle", False],  # 参数化测试：测试字符串是否每个单词首字母大写
        ["AAA", "isupper", True],  # 参数化测试：测试字符串是否全为大写字母
        ["AAc", "isupper", False],  # 参数化测试：测试字符串是否全为大写字母
    ],
)
def test_str_is_functions(value, method, exp):
    ser = pd.Series([value, None], dtype=ArrowDtype(pa.string()))
    result = getattr(ser.str, method)()  # 调用字符串方法检查字符串属性
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.bool_()))
    tm.assert_series_equal(result, expected)  # 断言结果与期望相等
# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_str_transform_functions 函数参数化测试用例
@pytest.mark.parametrize(
    "method, exp",
    [
        ["capitalize", "Abc def"],   # 测试 str.capitalize 方法，期望结果是 "Abc def"
        ["title", "Abc Def"],        # 测试 str.title 方法，期望结果是 "Abc Def"
        ["swapcase", "AbC Def"],     # 测试 str.swapcase 方法，期望结果是 "AbC Def"
        ["lower", "abc def"],        # 测试 str.lower 方法，期望结果是 "abc def"
        ["upper", "ABC DEF"],        # 测试 str.upper 方法，期望结果是 "ABC DEF"
        ["casefold", "abc def"],     # 测试 str.casefold 方法，期望结果是 "abc def"
    ],
)
def test_str_transform_functions(method, exp):
    # 创建包含字符串和 None 值的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    ser = pd.Series(["aBc dEF", None], dtype=ArrowDtype(pa.string()))
    # 调用 getattr 函数，根据 method 字符串获取 ser.str 中相应的方法并执行，存储结果到 result
    result = getattr(ser.str, method)()
    # 创建预期结果的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    # 使用 pandas.testing 模块的 assert_series_equal 函数比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_str_len，测试 pandas Series 对象中的 str.len 方法
def test_str_len():
    # 创建包含字符串和 None 值的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    ser = pd.Series(["abcd", None], dtype=ArrowDtype(pa.string()))
    # 调用 ser.str.len() 方法计算字符串长度，存储结果到 result
    result = ser.str.len()
    # 创建预期结果的 pandas Series 对象，数据类型为 ArrowDtype(pa.int32())
    expected = pd.Series([4, None], dtype=ArrowDtype(pa.int32()))
    # 比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_str_strip 函数参数化测试用例
@pytest.mark.parametrize(
    "method, to_strip, val",
    [
        ["strip", None, " abc "],    # 测试 str.strip 方法，去除字符串两侧空格，期望结果是 "abc"
        ["strip", "x", "xabcx"],     # 测试 str.strip 方法，去除字符串两侧 "x"，期望结果是 "abc"
        ["lstrip", None, " abc"],    # 测试 str.lstrip 方法，去除字符串左侧空格，期望结果是 "abc"
        ["lstrip", "x", "xabc"],     # 测试 str.lstrip 方法，去除字符串左侧 "x"，期望结果是 "abc"
        ["rstrip", None, "abc "],    # 测试 str.rstrip 方法，去除字符串右侧空格，期望结果是 "abc"
        ["rstrip", "x", "abcx"],     # 测试 str.rstrip 方法，去除字符串右侧 "x"，期望结果是 "abc"
    ],
)
def test_str_strip(method, to_strip, val):
    # 创建包含字符串和 None 值的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    # 调用 getattr 函数，根据 method 字符串获取 ser.str 中相应的方法并执行，带有可选参数 to_strip
    result = getattr(ser.str, method)(to_strip=to_strip)
    # 创建预期结果的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_str_removesuffix 函数参数化测试用例
@pytest.mark.parametrize("val", ["abc123", "abc"])
def test_str_removesuffix(val):
    # 创建包含字符串和 None 值的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    # 调用 ser.str.removesuffix 方法，去除字符串末尾的 "123"，存储结果到 result
    result = ser.str.removesuffix("123")
    # 创建预期结果的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_str_removeprefix 函数参数化测试用例
@pytest.mark.parametrize("val", ["123abc", "abc"])
def test_str_removeprefix(val):
    # 创建包含字符串和 None 值的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    ser = pd.Series([val, None], dtype=ArrowDtype(pa.string()))
    # 调用 ser.str.removeprefix 方法，去除字符串开头的 "123"，存储结果到 result
    result = ser.str.removeprefix("123")
    # 创建预期结果的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    expected = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_str_encode 函数参数化测试用例
@pytest.mark.parametrize("errors", ["ignore", "strict"])
@pytest.mark.parametrize(
    "encoding, exp",
    [
        ("utf8", {"little": b"abc", "big": "abc"}),    # 测试 utf8 编码，期望结果是 {"little": b"abc", "big": "abc"}
        (
            "utf32",
            {
                "little": b"\xff\xfe\x00\x00a\x00\x00\x00b\x00\x00\x00c\x00\x00\x00",   # 测试 utf32 小端编码
                "big": b"\x00\x00\xfe\xff\x00\x00\x00a\x00\x00\x00b\x00\x00\x00c",      # 测试 utf32 大端编码
            },
        ),
    ],
    ids=["utf8", "utf32"],  # 为不同的编码类型指定标识符
)
def test_str_encode(errors, encoding, exp):
    # 创建包含字符串和 None 值的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 调用 ser.str.encode 方法，按指定的编码和错误处理方式进行编码，存储结果到 result
    result = ser.str.encode(encoding, errors)
    # 创建预期结果的 pandas Series 对象，数据类型为 ArrowDtype(pa.binary())
    expected = pd.Series([exp[sys.byteorder], None], dtype=ArrowDtype(pa.binary()))
    # 比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_str_findall 函数参数化测试用例
@pytest.mark.parametrize("flags", [0, 2])
def test_str_findall(flags):
    # 创建包含字符串和 None 值的 pandas Series 对象，数据类型为 ArrowDtype(pa.string())
    ser = pd.Series(["abc", "efg", None], dtype=ArrowDtype(pa.string()))
    # 调用 ser.str.findall 方法，在每个字符串中查找 "b"，带有可选参数 flags
    result = ser.str.findall("b", flags=flags)
    # 创建预期结果的 pandas Series 对象，数据类型为 ArrowDtype(pa.list_(pa.string()))
    expected = pd.Series([["b"], [], None], dtype=ArrowDtype(pa.list_(pa.string())))
    # 比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器指定参数化测试的方法和参数
@pytest.mark.parametrize("method", ["index", "rindex"])
@pytest.mark.parametrize(
    "start, end",
    [
        [0, None],  # 参数化测试的起始和结束索引，分别为0和None
        [1, 4],     # 参数化测试的起始和结束索引，分别为1和4
    ],
)
# 定义测试函数 test_str_r_index，测试 Series 对象的字符串方法的 index 和 rindex 方法
def test_str_r_index(method, start, end):
    # 创建一个包含字符串和 None 值的 Series 对象，类型为 ArrowDtype(pa.string())
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    # 调用 getattr 函数根据 method 获取 Series 对象的字符串方法结果
    result = getattr(ser.str, method)("c", start, end)
    # 创建一个期望的 Series 对象，包含预期的结果，类型为 ArrowDtype(pa.int64())
    expected = pd.Series([2, None], dtype=ArrowDtype(pa.int64()))
    # 使用 tm.assert_series_equal 函数断言测试结果和预期结果相等
    tm.assert_series_equal(result, expected)

    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并验证异常信息为 "substring not found"
    with pytest.raises(ValueError, match="substring not found"):
        # 再次调用 getattr 函数，此次调用期望抛出 ValueError 异常
        getattr(ser.str, method)("foo", start, end)


# 使用 pytest.mark.parametrize 装饰器指定参数化测试的方法和参数
@pytest.mark.parametrize("form", ["NFC", "NFKC"])
# 定义测试函数 test_str_normalize，测试 Series 对象的字符串方法的 normalize 方法
def test_str_normalize(form):
    # 创建一个包含字符串和 None 值的 Series 对象，类型为 ArrowDtype(pa.string())
    ser = pd.Series(["abc", None], dtype=ArrowDtype(pa.string()))
    # 调用 Series 对象的字符串方法 normalize，根据参数 form 规范化字符串
    result = ser.str.normalize(form)
    # 创建一个期望的 Series 对象，包含与原始 Series 相同的数据，类型为 ArrowDtype(pa.string())
    expected = ser.copy()
    # 使用 tm.assert_series_equal 函数断言测试结果和预期结果相等
    tm.assert_series_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器指定参数化测试的起始和结束索引
@pytest.mark.parametrize(
    "start, end",
    [
        [0, None],  # 参数化测试的起始和结束索引，分别为0和None
        [1, 4],     # 参数化测试的起始和结束索引，分别为1和4
    ],
)
# 定义测试函数 test_str_rfind，测试 Series 对象的字符串方法的 rfind 方法
def test_str_rfind(start, end):
    # 创建一个包含字符串、None 值的 Series 对象，类型为 ArrowDtype(pa.string())
    ser = pd.Series(["abcba", "foo", None], dtype=ArrowDtype(pa.string()))
    # 调用 Series 对象的字符串方法 rfind，查找子字符串 "c" 的位置
    result = ser.str.rfind("c", start, end)
    # 创建一个期望的 Series 对象，包含预期的查找结果，类型为 ArrowDtype(pa.int64())
    expected = pd.Series([2, -1, None], dtype=ArrowDtype(pa.int64()))
    # 使用 tm.assert_series_equal 函数断言测试结果和预期结果相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_str_translate，测试 Series 对象的字符串方法的 translate 方法
def test_str_translate():
    # 创建一个包含字符串、None 值的 Series 对象，类型为 ArrowDtype(pa.string())
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    # 调用 Series 对象的字符串方法 translate，根据字典进行字符替换
    result = ser.str.translate({97: "b"})
    # 创建一个期望的 Series 对象，包含替换后的结果，类型为 ArrowDtype(pa.string())
    expected = pd.Series(["bbcbb", None], dtype=ArrowDtype(pa.string()))
    # 使用 tm.assert_series_equal 函数断言测试结果和预期结果相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_str_wrap，测试 Series 对象的字符串方法的 wrap 方法
def test_str_wrap():
    # 创建一个包含字符串、None 值的 Series 对象，类型为 ArrowDtype(pa.string())
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    # 调用 Series 对象的字符串方法 wrap，根据指定的宽度进行文本换行
    result = ser.str.wrap(3)
    # 创建一个期望的 Series 对象，包含换行后的结果，类型为 ArrowDtype(pa.string())
    expected = pd.Series(["abc\nba", None], dtype=ArrowDtype(pa.string()))
    # 使用 tm.assert_series_equal 函数断言测试结果和预期结果相等
    tm.assert_series_equal(result, expected)


# 定义测试函数 test_get_dummies，测试 Series 对象的字符串方法的 get_dummies 方法
def test_get_dummies():
    # 创建一个包含字符串、None 值的 Series 对象，类型为 ArrowDtype(pa.string())
    ser = pd.Series(["a|b", None, "a|c"], dtype=ArrowDtype(pa.string()))
    # 调用 Series 对象的字符串方法 get_dummies，根据分隔符创建虚拟变量 DataFrame
    result = ser.str.get_dummies()
    # 创建一个期望的 DataFrame，包含根据分隔符创建的虚拟变量，列名为 ["a", "b", "c"]，类型为 ArrowDtype(pa.bool_())
    expected = pd.DataFrame(
        [[True, True, False], [False, False, False], [True, False, True]],
        dtype=ArrowDtype(pa.bool_()),
        columns=["a", "b", "c"],
    )
    # 使用 tm.assert_frame_equal 函数断言测试结果和预期结果相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数 test_str_partition，测试 Series 对象的字符串方法的 partition 和 rpartition 方法
def test_str_partition():
    # 创建一个包含字符串、None 值的 Series 对象，类型为 ArrowDtype(pa.string())
    ser = pd.Series(["abcba", None], dtype=ArrowDtype(pa.string()))
    # 调用 Series 对象的字符串方法 partition，根据指定字符串进行分区
    result = ser.str.partition("b")
    # 创建一个期望的 DataFrame，包含根据指定字符串分区后的结果，列名为 [0, 1, 2]，类型为 ArrowDtype(pa.string())
    expected = pd.DataFrame(
        [["a", "b", "cba"], [None, None, None]],
        dtype=ArrowDtype(pa.string()),
        columns=pd.RangeIndex(3),
    )
    # 使用 tm.assert_frame_equal 函数断言测试结果和预期结果相等，并验证列类型
    tm.assert_frame_equal(result, expected, check_column_type=True)

    # 再次调用 partition 方法，设置 expand=False 参数，期望返回 Series 对象
    result = ser.str.partition("b", expand=False)
    # 创建一个期望的 Series 对象，包含根据指定字符串分区后的结果，类型为 ArrowExtensionArray(pa.array())
    expected = pd.Series(ArrowExtensionArray(pa.array([["a", "b", "cba"], None])))
    # 使用 tm.assert_series_equal 函数断言测试结果和预期结果相等
    tm.assert_series_equal(result, expected)

    # 调用 Series 对象的字符串方法 rpartition，根据指定字符串进行反向分区
    result = ser.str.rpartition("b")
    # 创建一个期望的 DataFrame，包含根据指定字符串反向分区后的结果，列名为 [0, 1, 2]，类型为 ArrowDtype(pa.string())
    expected = pd.DataFrame(
        [["abc", "b", "a"], [None, None, None]],
        dtype=ArrowDtype(pa.string()),
        columns=pd.RangeIndex(3),
    )
    # 使用 tm.assert_frame_equal 函数断言测试结果和预期结果相等，并验证列类型
    tm.assert_frame_equal(result, expected, check_column_type=True)

    # 再次调用 rpartition 方法，设置 expand=False 参数，期望返回 Series 对象
    result = ser.str.rpartition
@pytest.mark.parametrize("method", ["rsplit", "split"])
# 使用 pytest 的参数化装饰器，为以下测试用例指定参数化的方法为 "rsplit" 和 "split"
def test_str_split_pat_none(method):
    # GH 56271
    # 创建一个包含 ArrowDtype 的 Pandas Series，其中包含字符串和 None 值
    ser = pd.Series(["a1 cbc\nb", None], dtype=ArrowDtype(pa.string()))
    # 调用 Pandas Series 的字符串方法（根据参数 method 动态调用 split 或 rsplit 方法）
    result = getattr(ser.str, method)()
    # 创建预期的 Pandas Series，使用 ArrowExtensionArray 存储
    expected = pd.Series(ArrowExtensionArray(pa.array([["a1", "cbc", "b"], None])))
    # 使用 Pandas 测试工具 tm 来断言结果是否相等
    tm.assert_series_equal(result, expected)


def test_str_split():
    # GH 52401
    # 创建一个包含 ArrowDtype 的 Pandas Series，其中包含字符串和 None 值
    ser = pd.Series(["a1cbcb", "a2cbcb", None], dtype=ArrowDtype(pa.string()))
    # 调用 Pandas Series 的字符串方法 split，按指定字符 "c" 分割
    result = ser.str.split("c")
    # 创建预期的 Pandas Series，使用 ArrowExtensionArray 存储
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1", "b", "b"], ["a2", "b", "b"], None]))
    )
    # 使用 Pandas 测试工具 tm 来断言结果是否相等
    tm.assert_series_equal(result, expected)

    # 使用 n=1 参数限制分割的次数
    result = ser.str.split("c", n=1)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1", "bcb"], ["a2", "bcb"], None]))
    )
    tm.assert_series_equal(result, expected)

    # 使用正则表达式进行分割
    result = ser.str.split("[1-2]", regex=True)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a", "cbcb"], ["a", "cbcb"], None]))
    )
    tm.assert_series_equal(result, expected)

    # 使用正则表达式进行分割，并展开为 DataFrame
    result = ser.str.split("[1-2]", regex=True, expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a", "a", None])),
            1: ArrowExtensionArray(pa.array(["cbcb", "cbcb", None])),
        }
    )
    tm.assert_frame_equal(result, expected)

    # 使用指定字符进行分割，并展开为 DataFrame
    result = ser.str.split("1", expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a", "a2cbcb", None])),
            1: ArrowExtensionArray(pa.array(["cbcb", None, None])),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_str_rsplit():
    # GH 52401
    # 创建一个包含 ArrowDtype 的 Pandas Series，其中包含字符串和 None 值
    ser = pd.Series(["a1cbcb", "a2cbcb", None], dtype=ArrowDtype(pa.string()))
    # 调用 Pandas Series 的字符串方法 rsplit，按指定字符 "c" 从右向左分割
    result = ser.str.rsplit("c")
    # 创建预期的 Pandas Series，使用 ArrowExtensionArray 存储
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1", "b", "b"], ["a2", "b", "b"], None]))
    )
    # 使用 Pandas 测试工具 tm 来断言结果是否相等
    tm.assert_series_equal(result, expected)

    # 使用 n=1 参数限制从右向左分割的次数
    result = ser.str.rsplit("c", n=1)
    expected = pd.Series(
        ArrowExtensionArray(pa.array([["a1cb", "b"], ["a2cb", "b"], None]))
    )
    tm.assert_series_equal(result, expected)

    # 使用 n=1 参数限制从右向左分割的次数，并展开为 DataFrame
    result = ser.str.rsplit("c", n=1, expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a1cb", "a2cb", None])),
            1: ArrowExtensionArray(pa.array(["b", "b", None])),
        }
    )
    tm.assert_frame_equal(result, expected)

    # 使用指定字符进行从右向左分割，并展开为 DataFrame
    result = ser.str.rsplit("1", expand=True)
    expected = pd.DataFrame(
        {
            0: ArrowExtensionArray(pa.array(["a", "a2cbcb", None])),
            1: ArrowExtensionArray(pa.array(["cbcb", None, None])),
        }
    )
    tm.assert_frame_equal(result, expected)


def test_str_extract_non_symbolic():
    # 创建一个包含 ArrowDtype 的 Pandas Series，其中包含字符串 "a1", "b2", "c3"
    ser = pd.Series(["a1", "b2", "c3"], dtype=ArrowDtype(pa.string()))
    # 使用 pytest 的 raises 断言，检查是否会引发 ValueError 异常，匹配给定的错误消息
    with pytest.raises(ValueError, match="pat=.* must contain a symbolic group name."):
        # 使用正则表达式提取字符串中的内容，此处应该包含一个命名组
        ser.str.extract(r"[ab](\d)")


@pytest.mark.parametrize("expand", [True, False])
# 使用 pytest 的参数化装饰器，为以下测试用例指定参数化的 expand 参数为 True 和 False
# 定义一个测试函数，用于测试字符串序列的提取功能
def test_str_extract(expand):
    # 创建一个包含字符串和Arrow类型的Series对象
    ser = pd.Series(["a1", "b2", "c3"], dtype=ArrowDtype(pa.string()))
    # 对字符串序列进行正则表达式提取，返回DataFrame，根据参数决定是否展开列
    result = ser.str.extract(r"(?P<letter>[ab])(?P<digit>\d)", expand=expand)
    # 创建预期的DataFrame结果，包含Arrow扩展数组
    expected = pd.DataFrame(
        {
            "letter": ArrowExtensionArray(pa.array(["a", "b", None])),
            "digit": ArrowExtensionArray(pa.array(["1", "2", None])),
        }
    )
    # 使用测试工具检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试字符串序列的正则表达式提取功能（展开列）
def test_str_extract_expand():
    # 创建一个包含字符串和Arrow类型的Series对象
    ser = pd.Series(["a1", "b2", "c3"], dtype=ArrowDtype(pa.string()))
    # 对字符串序列进行正则表达式提取，展开列，返回DataFrame
    result = ser.str.extract(r"[ab](?P<digit>\d)", expand=True)
    # 创建预期的DataFrame结果，包含Arrow扩展数组
    expected = pd.DataFrame(
        {
            "digit": ArrowExtensionArray(pa.array(["1", "2", None])),
        }
    )
    # 使用测试工具检查结果是否符合预期
    tm.assert_frame_equal(result, expected)

    # 对字符串序列进行正则表达式提取，不展开列，返回Series
    result = ser.str.extract(r"[ab](?P<digit>\d)", expand=False)
    # 创建预期的Series结果，包含Arrow扩展数组
    expected = pd.Series(ArrowExtensionArray(pa.array(["1", "2", None])), name="digit")
    # 使用测试工具检查结果是否符合预期
    tm.assert_series_equal(result, expected)


# 使用参数化测试装饰器，测试从字符串创建持续时间扩展数组的功能
@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_duration_from_strings_with_nat(unit):
    # 字符串列表
    strings = ["1000", "NaT"]
    # 创建Arrow持续时间类型
    pa_type = pa.duration(unit)
    dtype = ArrowDtype(pa_type)
    # 从字符串序列创建扩展数组
    result = ArrowExtensionArray._from_sequence_of_strings(strings, dtype=dtype)
    # 创建预期的扩展数组结果
    expected = ArrowExtensionArray(pa.array([1000, None], type=pa_type))
    # 使用测试工具检查结果是否符合预期
    tm.assert_extension_array_equal(result, expected)


# 测试处理不支持的日期时间属性时是否引发异常
def test_unsupported_dt(data):
    # 获取数据的PyArrow数据类型
    pa_dtype = data.dtype.pyarrow_dtype
    # 如果数据类型不是时间类型，期望引发属性错误异常
    if not pa.types.is_temporal(pa_dtype):
        with pytest.raises(
            AttributeError, match="Can only use .dt accessor with datetimelike values"
        ):
            pd.Series(data).dt


# 使用参数化测试装饰器，测试日期时间对象的属性
@pytest.mark.parametrize(
    "prop, expected",
    [
        ["year", 2023],
        ["day", 2],
        ["day_of_week", 0],
        ["dayofweek", 0],
        ["weekday", 0],
        ["day_of_year", 2],
        ["dayofyear", 2],
        ["hour", 3],
        ["minute", 4],
        ["is_leap_year", False],
        ["microsecond", 5],
        ["month", 1],
        ["nanosecond", 6],
        ["quarter", 1],
        ["second", 7],
        ["date", date(2023, 1, 2)],
        ["time", time(3, 4, 7, 5)],
    ],
)
def test_dt_properties(prop, expected):
    # 创建包含时间戳和空值的Series对象，数据类型为Arrow时间戳类型
    ser = pd.Series(
        [
            pd.Timestamp(
                year=2023,
                month=1,
                day=2,
                hour=3,
                minute=4,
                second=7,
                microsecond=5,
                nanosecond=6,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 获取时间戳属性的结果
    result = getattr(ser.dt, prop)
    # 如果预期结果是日期对象，使用date32类型
    exp_type = None
    if isinstance(expected, date):
        exp_type = pa.date32()
    # 如果预期结果是时间对象，使用time64类型
    elif isinstance(expected, time):
        exp_type = pa.time64("ns")
    # 创建预期的Series结果，包含Arrow扩展数组
    expected = pd.Series(ArrowExtensionArray(pa.array([expected, None], type=exp_type)))
    # 使用测试工具检查结果是否符合预期
    tm.assert_series_equal(result, expected)


# 测试日期时间的月初和月末属性
def test_dt_is_month_start_end():
    # 创建一个 Pandas Series 对象，包含了四个日期时间对象和一个空值
    ser = pd.Series(
        [
            datetime(year=2023, month=12, day=2, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        # 指定 Series 的数据类型为 ArrowDtype，使用微秒精度的时间戳类型
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    # 计算 Series 中每个日期是否为月初的布尔值结果
    result = ser.dt.is_month_start
    # 创建一个期望的 Pandas Series 对象，包含了与上述结果对应的期望布尔值
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    # 使用测试工具检查计算结果是否与期望相符
    tm.assert_series_equal(result, expected)

    # 计算 Series 中每个日期是否为月末的布尔值结果
    result = ser.dt.is_month_end
    # 创建一个期望的 Pandas Series 对象，包含了与上述结果对应的期望布尔值
    expected = pd.Series([False, False, True, None], dtype=ArrowDtype(pa.bool_()))
    # 使用测试工具再次检查计算结果是否与期望相符
    tm.assert_series_equal(result, expected)
# 测试日期时间序列的年初与年末
def test_dt_is_year_start_end():
    # 创建包含日期时间的 Pandas Series 对象，使用 ArrowDtype 指定时间戳类型
    ser = pd.Series(
        [
            datetime(year=2023, month=12, day=31, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    # 计算是否为年初的布尔序列
    result = ser.dt.is_year_start
    # 期望的年初布尔序列，使用 ArrowDtype 指定布尔类型
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    # 使用 assert_series_equal 检查计算结果与期望值是否一致
    tm.assert_series_equal(result, expected)

    # 计算是否为年末的布尔序列
    result = ser.dt.is_year_end
    # 期望的年末布尔序列，使用 ArrowDtype 指定布尔类型
    expected = pd.Series([True, False, False, None], dtype=ArrowDtype(pa.bool_()))
    # 使用 assert_series_equal 检查计算结果与期望值是否一致
    tm.assert_series_equal(result, expected)


# 测试日期时间序列的季初与季末
def test_dt_is_quarter_start_end():
    # 创建包含日期时间的 Pandas Series 对象，使用 ArrowDtype 指定时间戳类型
    ser = pd.Series(
        [
            datetime(year=2023, month=11, day=30, hour=3),
            datetime(year=2023, month=1, day=1, hour=3),
            datetime(year=2023, month=3, day=31, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    # 计算是否为季初的布尔序列
    result = ser.dt.is_quarter_start
    # 期望的季初布尔序列，使用 ArrowDtype 指定布尔类型
    expected = pd.Series([False, True, False, None], dtype=ArrowDtype(pa.bool_()))
    # 使用 assert_series_equal 检查计算结果与期望值是否一致
    tm.assert_series_equal(result, expected)

    # 计算是否为季末的布尔序列
    result = ser.dt.is_quarter_end
    # 期望的季末布尔序列，使用 ArrowDtype 指定布尔类型
    expected = pd.Series([False, False, True, None], dtype=ArrowDtype(pa.bool_()))
    # 使用 assert_series_equal 检查计算结果与期望值是否一致
    tm.assert_series_equal(result, expected)


# 使用参数化测试测试日期时间序列中的月份天数计算方法
@pytest.mark.parametrize("method", ["days_in_month", "daysinmonth"])
def test_dt_days_in_month(method):
    # 创建包含日期时间的 Pandas Series 对象，使用 ArrowDtype 指定时间戳类型
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30, hour=3),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    # 动态调用日期时间序列的月份天数计算方法
    result = getattr(ser.dt, method)
    # 期望的月份天数结果，使用 ArrowDtype 指定整数类型
    expected = pd.Series([31, 30, 28, None], dtype=ArrowDtype(pa.int64()))
    # 使用 assert_series_equal 检查计算结果与期望值是否一致
    tm.assert_series_equal(result, expected)


# 测试日期时间序列的时间规范化
def test_dt_normalize():
    # 创建包含日期时间的 Pandas Series 对象，使用 ArrowDtype 指定时间戳类型
    ser = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1, hour=3),
            datetime(year=2023, month=2, day=3, hour=23, minute=59, second=59),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    # 进行日期时间序列的时间规范化
    result = ser.dt.normalize()
    # 期望的时间规范化结果，保留日期部分
    expected = pd.Series(
        [
            datetime(year=2023, month=3, day=30),
            datetime(year=2023, month=4, day=1),
            datetime(year=2023, month=2, day=3),
            None,
        ],
        dtype=ArrowDtype(pa.timestamp("us")),
    )
    # 使用 assert_series_equal 检查计算结果与期望值是否一致
    tm.assert_series_equal(result, expected)


# 使用参数化测试测试日期时间序列的时间提取方法
@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_time_preserve_unit(unit):
    # 创建包含日期时间的 Pandas Series 对象，使用 ArrowDtype 指定时间戳类型
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    # 断言日期时间序列的单位是否与预期一致
    assert ser.dt.unit == unit

    # 提取日期时间序列的时间部分
    result = ser.dt.time
    # 期望的时间提取结果，使用 ArrowExtensionArray 来表示时间数组
    expected = pd.Series(
        ArrowExtensionArray(pa.array([time(3, 0), None], type=pa.time64(unit)))
    )
    # 使用 assert_series_equal 检查计算结果与期望值是否一致
    tm.assert_series_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器为 test_dt_tz 函数定义参数化测试用例，测试不同的时区值（None, "UTC", "US/Pacific"）
@pytest.mark.parametrize("tz", [None, "UTC", "US/Pacific"])
def test_dt_tz(tz):
    # 创建包含两个元素的 Pandas Series 对象，第一个元素是特定日期时间，第二个元素为 None
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        # 使用 ArrowDtype 指定 Pandas Series 的数据类型为 ArrowDtype，时间戳精度为纳秒，可能有时区 tz
        dtype=ArrowDtype(pa.timestamp("ns", tz=tz)),
    )
    # 调用 Pandas Series 的 dt.tz 方法，将结果赋给 result
    result = ser.dt.tz
    # 使用 assert 断言 result 应该等于 timezones.maybe_get_tz(tz)
    assert result == timezones.maybe_get_tz(tz)


# 定义 test_dt_isocalendar 函数，测试 Pandas Series 的 dt.isocalendar 方法
def test_dt_isocalendar():
    # 创建包含两个元素的 Pandas Series 对象，第一个元素是特定日期时间，第二个元素为 None
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        # 使用 ArrowDtype 指定 Pandas Series 的数据类型为 ArrowDtype，时间戳精度为纳秒
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 调用 Pandas Series 的 dt.isocalendar 方法，将结果赋给 result
    result = ser.dt.isocalendar()
    # 创建预期的 DataFrame 对象 expected，包含两行数据，列名为 ["year", "week", "day"]，数据类型为 int64[pyarrow]
    expected = pd.DataFrame(
        [[2023, 1, 1], [0, 0, 0]],
        columns=["year", "week", "day"],
        dtype="int64[pyarrow]",
    )
    # 使用 tm.assert_frame_equal 断言 result 应该等于 expected
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器为 test_dt_day_month_name 函数定义参数化测试用例，测试不同的方法和预期结果
@pytest.mark.parametrize(
    "method, exp", [["day_name", "Sunday"], ["month_name", "January"]]
)
def test_dt_day_month_name(method, exp, request):
    # GH 52388
    # 调用 _require_timezone_database 函数，确保时区数据库的存在
    _require_timezone_database(request)

    # 创建包含两个元素的 Pandas Series 对象，第一个元素是特定日期时间，第二个元素为 None
    ser = pd.Series([datetime(2023, 1, 1), None], dtype=ArrowDtype(pa.timestamp("ms")))
    # 调用 Pandas Series 的 dt.method() 方法，将结果赋给 result
    result = getattr(ser.dt, method)()
    # 创建预期的 Pandas Series 对象 expected，包含两个元素，数据类型为 ArrowDtype(pa.string())
    expected = pd.Series([exp, None], dtype=ArrowDtype(pa.string()))
    # 使用 tm.assert_series_equal 断言 result 应该等于 expected
    tm.assert_series_equal(result, expected)


# 定义 test_dt_strftime 函数，测试 Pandas Series 的 dt.strftime 方法
def test_dt_strftime(request):
    # 调用 _require_timezone_database 函数，确保时区数据库的存在
    _require_timezone_database(request)

    # 创建包含两个元素的 Pandas Series 对象，第一个元素是特定日期时间，第二个元素为 None
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        # 使用 ArrowDtype 指定 Pandas Series 的数据类型为 ArrowDtype，时间戳精度为纳秒
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 调用 Pandas Series 的 dt.strftime 方法，格式化日期时间字符串，将结果赋给 result
    result = ser.dt.strftime("%Y-%m-%dT%H:%M:%S")
    # 创建预期的 Pandas Series 对象 expected，包含两个元素，数据类型为 ArrowDtype(pa.string())
    expected = pd.Series(
        ["2023-01-02T03:00:00.000000000", None], dtype=ArrowDtype(pa.string())
    )
    # 使用 tm.assert_series_equal 断言 result 应该等于 expected
    tm.assert_series_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器为 test_dt_roundlike_tz_options_not_supported 函数定义参数化测试用例，测试不支持的 round-like 方法和选项
@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_tz_options_not_supported(method):
    # 创建包含两个元素的 Pandas Series 对象，第一个元素是特定日期时间，第二个元素为 None
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        # 使用 ArrowDtype 指定 Pandas Series 的数据类型为 ArrowDtype，时间戳精度为纳秒
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 使用 pytest.raises 断言调用不支持的 round-like 方法时会抛出 NotImplementedError 异常
    with pytest.raises(NotImplementedError, match="ambiguous is not supported."):
        getattr(ser.dt, method)("1h", ambiguous="NaT")

    with pytest.raises(NotImplementedError, match="nonexistent is not supported."):
        getattr(ser.dt, method)("1h", nonexistent="NaT")


# 使用 pytest.mark.parametrize 装饰器为 test_dt_roundlike_unsupported_freq 函数定义参数化测试用例，测试不支持的 round-like 方法和频率参数
@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_roundlike_unsupported_freq(method):
    # 创建包含两个元素的 Pandas Series 对象，第一个元素是特定日期时间，第二个元素为 None
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        # 使用 ArrowDtype 指定 Pandas Series 的数据类型为 ArrowDtype，时间戳精度为纳秒
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 使用 pytest.raises 断言调用不支持的 round-like 方法时会抛出 ValueError 异常
    with pytest.raises(ValueError, match="freq='1B' is not supported"):
        getattr(ser.dt, method)("1B")

    with pytest.raises(ValueError, match="Must specify a valid frequency: None"):
        getattr(ser.dt, method)(None)


# 使用 pytest.mark.parametrize 装饰器为 test_dt_ceil_year_floor 函数定义参数化测试用例，测试 Pandas Series 的 ceil、floor 和 round 方法
@pytest.mark.parametrize("freq", ["D", "h", "min", "s", "ms", "us", "ns"])
@pytest.mark.parametrize("method", ["ceil", "floor", "round"])
def test_dt_ceil_year_floor(freq, method):
    # 创建包含两个元素的 Pandas Series 对象，第一个元素是特定日期时间，第二个元素为 None
    ser = pd.Series(
        [datetime(year=2023, month=1, day=1), None],
    )
    # 使用 ArrowDtype 指定 Pandas Series 的数据类型为 ArrowDtype，时间戳精度为纳秒
    pa_dtype = ArrowDtype(pa.timestamp("ns"))
    # 调用 Pandas Series 的 dt.method() 方法，对日期时间进行 ceil、floor 或 round 操作，将结果赋给 expected
    expected = getattr(ser.dt, method)(f"1{freq}").astype(pa_dtype)
    result = getattr(ser.astype(pa_dtype).dt, method)(f"1{freq}")
    # 使用测试框架中的方法来比较结果和期望值的序列是否相等
    tm.assert_series_equal(result, expected)
def test_dt_to_pydatetime():
    # GH 51859
    # 创建包含 datetime 对象的数据列表
    data = [datetime(2022, 1, 1), datetime(2023, 1, 1)]
    # 使用 ArrowDtype 指定数据类型创建 Pandas Series
    ser = pd.Series(data, dtype=ArrowDtype(pa.timestamp("ns")))
    # 调用 dt.to_pydatetime() 方法转换为 Python 内置 datetime 对象
    result = ser.dt.to_pydatetime()
    # 创建期望结果的 Pandas Series，数据类型为 object
    expected = pd.Series(data, dtype=object)
    # 使用 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
    # 验证 expected 中每个元素的类型是否为 datetime
    assert all(type(expected.iloc[i]) is datetime for i in range(len(expected)))

    # 将 ser 转换为 datetime64[ns] 类型后再次调用 dt.to_pydatetime()
    expected = ser.astype("datetime64[ns]").dt.to_pydatetime()
    # 使用 tm.assert_series_equal 检查结果是否与期望相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("date_type", [32, 64])
def test_dt_to_pydatetime_date_error(date_type):
    # GH 52812
    # 创建包含日期对象的 Pandas Series，使用 ArrowDtype 指定数据类型
    ser = pd.Series(
        [date(2022, 12, 31)],
        dtype=ArrowDtype(getattr(pa, f"date{date_type}")()),
    )
    # 使用 pytest 检查调用 dt.to_pydatetime() 是否会引发 ValueError 异常
    with pytest.raises(ValueError, match="to_pydatetime cannot be called with"):
        ser.dt.to_pydatetime()


def test_dt_tz_localize_unsupported_tz_options():
    # 创建包含日期时间对象的 Pandas Series，使用 ArrowDtype 指定数据类型
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 使用 pytest 检查调用 dt.tz_localize() 是否会引发 NotImplementedError 异常
    with pytest.raises(NotImplementedError, match="ambiguous='NaT' is not supported"):
        ser.dt.tz_localize("UTC", ambiguous="NaT")

    with pytest.raises(NotImplementedError, match="nonexistent='NaT' is not supported"):
        ser.dt.tz_localize("UTC", nonexistent="NaT")


def test_dt_tz_localize_none():
    # 创建包含日期时间对象的 Pandas Series，使用 ArrowDtype 指定带时区信息的数据类型
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns", tz="US/Pacific")),
    )
    # 调用 dt.tz_localize(None) 移除时区信息
    result = ser.dt.tz_localize(None)
    # 创建期望结果的 Pandas Series，数据类型为带时区信息的 timestamp
    expected = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 使用 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_tz_localize(unit, request):
    _require_timezone_database(request)

    # 创建包含日期时间对象的 Pandas Series，使用 ArrowDtype 指定不同的时间单位
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit)),
    )
    # 调用 dt.tz_localize("US/Pacific") 将时区设为 "US/Pacific"
    result = ser.dt.tz_localize("US/Pacific")
    # 使用 ArrowExtensionArray 创建期望结果的 Pandas Series
    exp_data = pa.array(
        [datetime(year=2023, month=1, day=2, hour=3), None], type=pa.timestamp(unit)
    )
    exp_data = pa.compute.assume_timezone(exp_data, "US/Pacific")
    expected = pd.Series(ArrowExtensionArray(exp_data))
    # 使用 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "nonexistent, exp_date",
    [
        ["shift_forward", datetime(year=2023, month=3, day=12, hour=3)],
        ["shift_backward", pd.Timestamp("2023-03-12 01:59:59.999999999")],
    ],
)
def test_dt_tz_localize_nonexistent(nonexistent, exp_date, request):
    _require_timezone_database(request)

    # 创建包含日期时间对象的 Pandas Series，使用 ArrowDtype 指定数据类型
    ser = pd.Series(
        [datetime(year=2023, month=3, day=12, hour=2, minute=30), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 调用 dt.tz_localize("US/Pacific", nonexistent=nonexistent) 设置时区及不存在时的处理方式
    result = ser.dt.tz_localize("US/Pacific", nonexistent=nonexistent)
    # 创建期望结果的 Arrow 数组
    exp_data = pa.array([exp_date, None], type=pa.timestamp("ns"))
    exp_data = pa.compute.assume_timezone(exp_data, "US/Pacific")
    # 创建预期结果的 Pandas Series，使用 ArrowExtensionArray 封装 exp_data 数据
    expected = pd.Series(ArrowExtensionArray(exp_data))
    # 使用测试工具 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
# 测试时区转换方法，如果序列中有时区无关的时间戳，应该引发 TypeError 异常，错误信息为 "Cannot convert tz-naive timestamps"
def test_dt_tz_convert_not_tz_raises():
    # 创建包含一个带有时区的 datetime 对象和一个空值的 Pandas Series
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 使用 pytest 检查调用 ser.dt.tz_convert("UTC") 是否会引发 TypeError，并且错误信息匹配 "Cannot convert tz-naive timestamps"
    with pytest.raises(TypeError, match="Cannot convert tz-naive timestamps"):
        ser.dt.tz_convert("UTC")


# 测试时区转换方法，如果传入 None 作为目标时区，应该返回与原始 Series 相同的结果
def test_dt_tz_convert_none():
    # 创建包含一个带有 US/Pacific 时区的 datetime 对象和一个空值的 Pandas Series
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns", "US/Pacific")),
    )
    # 调用 ser.dt.tz_convert(None) 进行时区转换，预期结果与原始 Series 相同
    result = ser.dt.tz_convert(None)
    # 创建一个预期结果的 Series，目标是转换为没有时区信息的原始时间戳类型
    expected = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp("ns")),
    )
    # 使用 assert_series_equal 检查实际结果和预期结果是否相同
    tm.assert_series_equal(result, expected)


# 使用参数化测试来测试不同的时间戳单位（us 和 ns）的时区转换
@pytest.mark.parametrize("unit", ["us", "ns"])
def test_dt_tz_convert(unit):
    # 创建包含一个带有指定单位和 US/Pacific 时区的 datetime 对象和一个空值的 Pandas Series
    ser = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Pacific")),
    )
    # 调用 ser.dt.tz_convert("US/Eastern") 进行时区转换，预期结果是将时区从 US/Pacific 转换为 US/Eastern
    result = ser.dt.tz_convert("US/Eastern")
    # 创建一个预期结果的 Series，目标是将时区转换为指定的 US/Eastern 时区，时间戳单位保持不变
    expected = pd.Series(
        [datetime(year=2023, month=1, day=2, hour=3), None],
        dtype=ArrowDtype(pa.timestamp(unit, "US/Eastern")),
    )
    # 使用 assert_series_equal 检查实际结果和预期结果是否相同
    tm.assert_series_equal(result, expected)


# 使用参数化测试来测试不同类型（timestamp 和 duration）的单位转换
@pytest.mark.parametrize("dtype", ["timestamp[ms][pyarrow]", "duration[ms][pyarrow]"])
def test_as_unit(dtype):
    # 创建包含一个整数和一个空值的 Pandas Series，dtype 可以是 timestamp 或 duration
    ser = pd.Series([1000, None], dtype=dtype)
    # 调用 ser.dt.as_unit("ns") 进行单位转换，预期结果是将单位从 ms 转换为 ns
    result = ser.dt.as_unit("ns")
    # 创建一个预期结果的 Series，目标是将单位转换为 ns 的相应类型
    expected = ser.astype(dtype.replace("ms", "ns"))
    # 使用 assert_series_equal 检查实际结果和预期结果是否相同
    tm.assert_series_equal(result, expected)


# 使用参数化测试来测试不同时间间隔属性（days、seconds、microseconds、nanoseconds）的访问
@pytest.mark.parametrize(
    "prop, expected",
    [
        ["days", 1],
        ["seconds", 2],
        ["microseconds", 3],
        ["nanoseconds", 4],
    ],
)
def test_dt_timedelta_properties(prop, expected):
    # 创建包含一个 Timedelta 对象和一个空值的 Pandas Series，dtype 是 duration[ns]
    ser = pd.Series(
        [
            pd.Timedelta(
                days=1,
                seconds=2,
                microseconds=3,
                nanoseconds=4,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    # 获取 ser.dt 对象的指定属性（prop），预期结果是属性值的 Series
    result = getattr(ser.dt, prop)
    # 创建一个预期结果的 Series，目标是包含指定属性值的 Series
    expected = pd.Series(
        ArrowExtensionArray(pa.array([expected, None], type=pa.int32()))
    )
    # 使用 assert_series_equal 检查实际结果和预期结果是否相同
    tm.assert_series_equal(result, expected)


# 测试 Timedelta 对象的 total_seconds 方法
def test_dt_timedelta_total_seconds():
    # 创建包含一个 Timedelta 对象和一个空值的 Pandas Series，dtype 是 duration[ns]
    ser = pd.Series(
        [
            pd.Timedelta(
                days=1,
                seconds=2,
                microseconds=3,
                nanoseconds=4,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),
    )
    # 调用 ser.dt.total_seconds() 计算总秒数，预期结果是包含总秒数的 Series
    result = ser.dt.total_seconds()
    # 创建一个预期结果的 Series，目标是包含总秒数的 Series
    expected = pd.Series(
        ArrowExtensionArray(pa.array([86402.000003, None], type=pa.float64()))
    )
    # 使用 assert_series_equal 检查实际结果和预期结果是否相同
    tm.assert_series_equal(result, expected)


# 测试 to_pytimedelta 方法的使用情况
def test_dt_to_pytimedelta():
    # 创建包含两个 timedelta 对象的列表和相应 dtype 的 Pandas Series
    data = [timedelta(1, 2, 3), timedelta(1, 2, 4)]
    ser = pd.Series(data, dtype=ArrowDtype(pa.duration("ns")))

    # 输出一条关于 to_pytimedelta 方法被弃用行为的警告消息
    msg = "The behavior of ArrowTemporalProperties.to_pytimedelta is deprecated"
    # 使用 pytest 中的 assert_produces_warning 上下文管理器，检查是否会产生 FutureWarning 警告，并匹配指定的警告信息
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # 调用 ser 对象的 dt 属性的 to_pytimedelta 方法，并将结果赋给 result 变量
        result = ser.dt.to_pytimedelta()
    
    # 创建一个预期的 NumPy 数组，数据来自变量 data，元素类型为 object
    expected = np.array(data, dtype=object)
    # 使用 assert_numpy_array_equal 函数比较 result 和 expected 两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)
    
    # 断言 result 中的每个元素的类型是否为 timedelta
    assert all(type(res) is timedelta for res in result)
    
    # 设置警告信息字符串，用于下一个警告检查
    msg = "The behavior of TimedeltaProperties.to_pytimedelta is deprecated"
    # 使用 assert_produces_warning 上下文管理器，检查是否会产生 FutureWarning 警告，并匹配指定的警告信息
    with tm.assert_produces_warning(FutureWarning, match=msg):
        # 调用 ser 对象转换为 timedelta64[ns] 类型后，再调用 dt 属性的 to_pytimedelta 方法，并将结果赋给 expected 变量
        expected = ser.astype("timedelta64[ns]").dt.to_pytimedelta()
    # 使用 assert_numpy_array_equal 函数比较 result 和 expected 两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)
# 定义一个测试函数，用于测试 Pandas Series 的 Timedelta 数据类型的组件提取功能
def test_dt_components():
    # GH 52284: GitHub issue编号
    # 创建一个包含 Timedelta 对象的 Pandas Series，其中包括一个 Timedelta 对象和一个 None 值
    ser = pd.Series(
        [
            pd.Timedelta(
                days=1,
                seconds=2,
                microseconds=3,
                nanoseconds=4,
            ),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),  # 指定 Series 的数据类型为 ArrowDtype，表示纳秒级的时间间隔
    )
    # 提取 Series 中 Timedelta 对象的各个时间组件，结果是一个 DataFrame
    result = ser.dt.components
    # 期望的结果 DataFrame，包含了每个 Timedelta 对象的各个时间组件的值
    expected = pd.DataFrame(
        [[1, 0, 0, 2, 0, 3, 4], [None, None, None, None, None, None, None]],
        columns=[
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ],
        dtype="int32[pyarrow]",  # 结果 DataFrame 的数据类型，使用 PyArrow 的 int32
    )
    # 使用 Pandas 的 assert_frame_equal 函数比较结果和期望值的 DataFrame 是否相同
    tm.assert_frame_equal(result, expected)


# 测试处理大数值的 Timedelta 对象的组件提取功能
def test_dt_components_large_values():
    # 创建一个包含大数值 Timedelta 对象的 Pandas Series，其中包括一个 Timedelta 对象和一个 None 值
    ser = pd.Series(
        [
            pd.Timedelta("365 days 23:59:59.999000"),
            None,
        ],
        dtype=ArrowDtype(pa.duration("ns")),  # 指定 Series 的数据类型为 ArrowDtype，表示纳秒级的时间间隔
    )
    # 提取 Series 中 Timedelta 对象的各个时间组件，结果是一个 DataFrame
    result = ser.dt.components
    # 期望的结果 DataFrame，包含了每个 Timedelta 对象的各个时间组件的值
    expected = pd.DataFrame(
        [[365, 23, 59, 59, 999, 0, 0], [None, None, None, None, None, None, None]],
        columns=[
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        ],
        dtype="int32[pyarrow]",  # 结果 DataFrame 的数据类型，使用 PyArrow 的 int32
    )
    # 使用 Pandas 的 assert_frame_equal 函数比较结果和期望值的 DataFrame 是否相同
    tm.assert_frame_equal(result, expected)


# 使用参数化测试来测试处理空值的情况下，所有布尔运算的结果
@pytest.mark.parametrize("skipna", [True, False])
def test_boolean_reduce_series_all_null(all_boolean_reductions, skipna):
    # GH51624: GitHub issue编号
    # 创建一个包含一个 None 值的 Pandas Series，数据类型为 float64，使用 PyArrow 后端存储
    ser = pd.Series([None], dtype="float64[pyarrow]")
    # 执行 Pandas Series 的布尔运算操作（如 all、any），结果为一个布尔值
    result = getattr(ser, all_boolean_reductions)(skipna=skipna)
    # 根据 skipna 参数判断期望的结果
    if skipna:
        expected = all_boolean_reductions == "all"  # 如果 skipna 为 True，则期望结果为布尔值
    else:
        expected = pd.NA  # 如果 skipna 为 False，则期望结果为 NA
    # 使用 assert 语句比较结果和期望值
    assert result is expected


# 测试从字符串序列创建布尔类型的扩展数组
def test_from_sequence_of_strings_boolean():
    # 定义不同类型的字符串数组：true_strings、false_strings 和 nulls
    true_strings = ["true", "TRUE", "True", "1", "1.0"]
    false_strings = ["false", "FALSE", "False", "0", "0.0"]
    nulls = [None]
    strings = true_strings + false_strings + nulls
    # 对应的布尔值数组
    bools = (
        [True] * len(true_strings) + [False] * len(false_strings) + [None] * len(nulls)
    )

    # 指定数据类型为 ArrowDtype，表示布尔类型
    dtype = ArrowDtype(pa.bool_())
    # 调用 ArrowExtensionArray 类的方法，从字符串序列创建扩展数组
    result = ArrowExtensionArray._from_sequence_of_strings(strings, dtype=dtype)
    # 期望的结果，使用 Pandas 的 array 函数创建布尔类型的 Pandas 数组
    expected = pd.array(bools, dtype="boolean[pyarrow]")
    # 使用 Pandas 的 assert_extension_array_equal 函数比较结果和期望值的扩展数组是否相同
    tm.assert_extension_array_equal(result, expected)

    # 针对不能解析的字符串序列，验证是否能正确抛出异常
    strings = ["True", "foo"]
    with pytest.raises(pa.ArrowInvalid, match="Failed to parse"):
        ArrowExtensionArray._from_sequence_of_strings(strings, dtype=dtype)


# 测试连接空的基于 Arrow 后端的 Series
def test_concat_empty_arrow_backed_series(dtype):
    # GH#51734: GitHub issue编号
    # 创建一个空的 Pandas Series，指定数据类型为 dtype
    ser = pd.Series([], dtype=dtype)
    # 期望的结果为一个与输入 Series 相同的空 Series
    expected = ser.copy()
    # 执行 pd.concat 操作，连接一个空的 Series，结果应与期望值相同
    result = pd.concat([ser[np.array([], dtype=np.bool_)]])
    # 使用 Pandas 的 assert_series_equal 函数比较结果和期望值的 Series 是否相同
    tm.assert_series_equal(result, expected)


# 使用参数化测试来测试从字符串数组创建 Series 的功能
@pytest.mark.parametrize("dtype", ["string", "string[pyarrow]"])
def test_series_from_string_array(dtype):
    # 创建一个 PyArrow 数组，包含字符串内容 "the quick brown fox"
    arr = pa.array("the quick brown fox".split())
    # 使用 Pandas 创建一个 Series，数据类型为 dtype，存储方式为 Arrow 扩展数组
    ser = pd.Series(arr, dtype=dtype)
    # 期望的结果为一个与输入 PyArrow 数组对应的 Pandas Series
    expected = pd.Series(ArrowExtensionArray(arr), dtype=dtype)
    # 使用测试框架中的函数来比较序列 `ser` 和预期的序列 `expected` 是否相等
    tm.assert_series_equal(ser, expected)
# _data was renamed to _pa_data，这里定义了一个名为OldArrowExtensionArray的类，它继承自ArrowExtensionArray类
class OldArrowExtensionArray(ArrowExtensionArray):
    # 实现__getstate__方法，用于序列化对象时获取其状态
    def __getstate__(self):
        # 调用父类的__getstate__方法，获取父类的状态
        state = super().__getstate__()
        # 将状态字典中的"_pa_array"键重命名为"_data"，这是为了向后兼容
        state["_data"] = state.pop("_pa_array")
        return state


# 测试将OldArrowExtensionArray对象序列化和反序列化的过程
def test_pickle_old_arrowextensionarray():
    # 创建一个pa.array对象data，包含单个元素1
    data = pa.array([1])
    # 使用OldArrowExtensionArray类将data转换为扩展数组对象expected
    expected = OldArrowExtensionArray(data)
    # 将expected对象序列化然后反序列化，得到反序列化的结果result
    result = pickle.loads(pickle.dumps(expected))
    # 使用测试工具函数tm.assert_extension_array_equal检查result与expected是否相等
    tm.assert_extension_array_equal(result, expected)
    # 断言result的_pa_array属性与原始数据data的chunked_array形式相等
    assert result._pa_array == pa.chunked_array(data)
    # 断言result对象没有"_data"属性
    assert not hasattr(result, "_data")


# 测试在ArrowExtensionArray对象上设置布尔掩码时的行为
def test_setitem_boolean_replace_with_mask_segfault():
    # GH#52059
    # 定义一个大小为145000的布尔类型的ArrowExtensionArray对象arr，其数据初始化为全1
    N = 145_000
    arr = ArrowExtensionArray(pa.chunked_array([np.ones((N,), dtype=np.bool_)]))
    # 创建arr对象的副本expected
    expected = arr.copy()
    # 使用全0的布尔掩码替换arr对象的数据，将所有元素置为False
    arr[np.zeros((N,), dtype=np.bool_)] = False
    # 断言修改后的arr对象的_pa_array属性与原始数据相等
    assert arr._pa_array == expected._pa_array


# 参数化测试函数，测试从numpy数组转换为大数据类型的Arrow扩展数组的行为
@pytest.mark.parametrize(
    "data, arrow_dtype",
    [
        ([b"a", b"b"], pa.large_binary()),  # 测试二进制数据类型的转换
        (["a", "b"], pa.large_string()),   # 测试字符串数据类型的转换
    ],
)
def test_conversion_large_dtypes_from_numpy_array(data, arrow_dtype):
    # 创建Arrow数据类型的描述符
    dtype = ArrowDtype(arrow_dtype)
    # 使用pd.array函数将numpy数组data转换为指定Arrow数据类型的扩展数组result
    result = pd.array(np.array(data), dtype=dtype)
    # 使用pd.array函数将原始数据data转换为指定Arrow数据类型的扩展数组expected
    expected = pd.array(data, dtype=dtype)
    # 断言使用测试工具函数tm.assert_extension_array_equal，result与expected的内容是否一致
    tm.assert_extension_array_equal(result, expected)


# 测试在DataFrame中连接包含空值的Arrow数组时的行为
def test_concat_null_array():
    # 创建包含空值的DataFrame对象df，列名为"a"，数据类型为null类型
    df = pd.DataFrame({"a": [None, None]}, dtype=ArrowDtype(pa.null()))
    # 创建另一个DataFrame对象df2，列名为"a"，数据类型为int64
    df2 = pd.DataFrame({"a": [0, 1]}, dtype="int64[pyarrow]")
    # 使用pd.concat函数将df和df2连接起来，忽略索引重建，得到连接后的结果DataFrame对象result
    result = pd.concat([df, df2], ignore_index=True)
    # 创建期望的结果DataFrame对象expected，包含连接后的完整数据
    expected = pd.DataFrame({"a": [None, None, 0, 1]}, dtype="int64[pyarrow]")
    # 使用测试工具函数tm.assert_frame_equal检查result与expected是否相等
    tm.assert_frame_equal(result, expected)


# 参数化测试函数，测试数值类型数据的描述统计信息生成
@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES + tm.FLOAT_PYARROW_DTYPES)
def test_describe_numeric_data(pa_type):
    # GH 52470
    # 创建指定Arrow数据类型的Series对象data，包含整数或浮点数数据
    data = pd.Series([1, 2, 3], dtype=ArrowDtype(pa_type))
    # 生成数据的描述统计信息result
    result = data.describe()
    # 创建期望的描述统计信息expected
    expected = pd.Series(
        [3, 2, 1, 1, 1.5, 2.0, 2.5, 3],  # 统计信息的数值内容
        dtype=ArrowDtype(pa.float64()),  # 描述统计信息的数据类型
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],  # 统计信息的索引
    )
    # 使用测试工具函数tm.assert_series_equal检查result与expected的内容是否一致
    tm.assert_series_equal(result, expected)


# 参数化测试函数，测试时间间隔类型数据的描述统计信息生成
@pytest.mark.parametrize("pa_type", tm.TIMEDELTA_PYARROW_DTYPES)
def test_describe_timedelta_data(pa_type):
    # GH53001
    # 创建指定Arrow数据类型的Series对象data，包含时间间隔数据
    data = pd.Series(range(1, 10), dtype=ArrowDtype(pa_type))
    # 生成数据的描述统计信息result
    result = data.describe()
    # 创建期望的描述统计信息expected
    expected = pd.Series(
        [9] + pd.to_timedelta([5, 2, 1, 3, 5, 7, 9], unit=pa_type.unit).tolist(),
        dtype=object,
        index=["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
    )
    # 使用测试工具函数tm.assert_series_equal检查result与expected的内容是否一致
    tm.assert_series_equal(result, expected)


# 参数化测试函数，测试日期时间类型数据的描述统计信息生成
@pytest.mark.parametrize("pa_type", tm.DATETIME_PYARROW_DTYPES)
def test_describe_datetime_data(pa_type):
    # GH53001
    # 创建指定Arrow数据类型的Series对象data，包含日期时间数据
    data = pd.Series(range(1, 10), dtype=ArrowDtype(pa_type))
    # 生成数据的描述统计信息result
    result = data.describe()
    # 创建期望的描述统计信息expected
    expected = pd.Series(
        [9]
        + [
            pd.Timestamp(v, tz=pa_type.tz, unit=pa_type.unit)
            for v in [5, 1, 3, 5, 7, 9]
        ],
        dtype=object,
        index=["count", "mean", "min", "25%", "50%", "75%", "max"],
    )
    # 使用测试工具函数tm.assert_series_equal检查result与expected的内容是否一致
    tm.assert_series_equal(result, expected)
    # 使用测试模块中的函数验证结果序列与期望序列是否相等
        tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
# 定义测试函数，参数化 pa_type，用于测试日期时间和时间增量类型
def test_quantile_temporal(pa_type):
    # GH52678
    # 准备测试数据
    data = [1, 2, 3]
    # 创建 Pandas Series，指定 ArrowDtype 类型
    ser = pd.Series(data, dtype=ArrowDtype(pa_type))
    # 计算分位数
    result = ser.quantile(0.1)
    # 期望值为第一个元素
    expected = ser[0]
    # 断言结果是否符合预期
    assert result == expected


def test_date32_repr():
    # GH48238
    # 创建一个 Pandas Series，包含一个日期数组
    arrow_dt = pa.array([date.fromisoformat("2020-01-01")], type=pa.date32())
    ser = pd.Series(arrow_dt, dtype=ArrowDtype(arrow_dt.type))
    # 断言 Series 的字符串表示
    assert repr(ser) == "0    2020-01-01\ndtype: date32[day][pyarrow]"


def test_duration_overflow_from_ndarray_containing_nat():
    # GH52843
    # 创建包含时间戳和时间增量的数据
    data_ts = pd.to_datetime([1, None])
    data_td = pd.to_timedelta([1, None])
    # 创建 Pandas Series，指定 ArrowDtype 类型
    ser_ts = pd.Series(data_ts, dtype=ArrowDtype(pa.timestamp("ns")))
    ser_td = pd.Series(data_td, dtype=ArrowDtype(pa.duration("ns")))
    # 执行加法操作
    result = ser_ts + ser_td
    # 创建期望的 Pandas Series
    expected = pd.Series([2, None], dtype=ArrowDtype(pa.timestamp("ns")))
    # 断言结果是否符合预期
    tm.assert_series_equal(result, expected)


def test_infer_dtype_pyarrow_dtype(data, request):
    # 推断数据类型
    res = lib.infer_dtype(data)
    # 断言推断结果不是 "unknown-array"
    assert res != "unknown-array"

    # 如果数据中包含缺失值，并且推断结果是浮点数、日期时间或时间增量类型，则标记为预期失败
    if data._hasna and res in ["floating", "datetime64", "timedelta64"]:
        mark = pytest.mark.xfail(
            reason="in infer_dtype pd.NA is not ignored in these cases "
            "even with skipna=True in the list(data) check below"
        )
        request.applymarker(mark)

    # 断言推断结果与通过列表推断的结果相符，忽略缺失值
    assert res == lib.infer_dtype(list(data), skipna=True)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
# 定义测试函数，参数化 pa_type，用于测试日期时间和时间增量类型
def test_from_sequence_temporal(pa_type):
    # GH 53171
    # 准备测试数据
    val = 3
    unit = pa_type.unit
    # 根据数据类型选择创建 Pandas Series 的方法
    if pa.types.is_duration(pa_type):
        seq = [pd.Timedelta(val, unit=unit).as_unit(unit)]
    else:
        seq = [pd.Timestamp(val, unit=unit, tz=pa_type.tz).as_unit(unit)]

    # 调用 _from_sequence 方法创建 ArrowExtensionArray 对象
    result = ArrowExtensionArray._from_sequence(seq, dtype=pa_type)
    # 创建期望的 ArrowExtensionArray 对象
    expected = ArrowExtensionArray(pa.array([val], type=pa_type))
    # 断言两个 ExtensionArray 是否相等
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
# 定义测试函数，参数化 pa_type，用于测试日期时间和时间增量类型
def test_setitem_temporal(pa_type):
    # GH 53171
    unit = pa_type.unit
    # 根据数据类型选择创建值的方法
    if pa.types.is_duration(pa_type):
        val = pd.Timedelta(1, unit=unit).as_unit(unit)
    else:
        val = pd.Timestamp(1, unit=unit, tz=pa_type.tz).as_unit(unit)

    # 创建 ArrowExtensionArray 对象
    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))

    # 复制数组，然后设置所有元素的值为 val
    result = arr.copy()
    result[:] = val
    # 创建期望的 ArrowExtensionArray 对象
    expected = ArrowExtensionArray(pa.array([1, 1, 1], type=pa_type))
    # 断言两个 ExtensionArray 是否相等
    tm.assert_extension_array_equal(result, expected)


@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
# 定义测试函数，参数化 pa_type，用于测试日期时间和时间增量类型
def test_arithmetic_temporal(pa_type, request):
    # GH 53171
    # 创建 ArrowExtensionArray 对象
    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))
    unit = pa_type.unit
    # 执行减法操作
    result = arr - pd.Timedelta(1, unit=unit).as_unit(unit)
    # 创建预期的 Arrow 扩展数组，使用指定的 PyArrow 数组初始化
    expected = ArrowExtensionArray(pa.array([0, 1, 2], type=pa_type))
    # 使用 pytest 的断言方法检查 result 和 expected 是否相等
    tm.assert_extension_array_equal(result, expected)
# 使用 pytest 的 parametrize 装饰器，为 test_comparison_temporal 函数定义多个参数化的测试用例
@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_comparison_temporal(pa_type):
    # GH 53171: 标识 GitHub issue 53171，测试比较时间相关类型的操作

    # 从 pa_type 中获取时间单位
    unit = pa_type.unit

    # 根据 pa_type 的类型，设置不同的 val 值
    if pa.types.is_duration(pa_type):
        # 如果是时间间隔类型，使用 pd.Timedelta 创建时间间隔，并转换为指定单位
        val = pd.Timedelta(1, unit=unit).as_unit(unit)
    else:
        # 如果是时间戳类型，使用 pd.Timestamp 创建时间戳，并转换为指定单位
        val = pd.Timestamp(1, unit=unit, tz=pa_type.tz).as_unit(unit)

    # 创建 ArrowExtensionArray 对象
    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))

    # 进行比较操作
    result = arr > val

    # 期望的结果
    expected = ArrowExtensionArray(pa.array([False, True, True], type=pa.bool_()))

    # 断言比较结果与期望结果相等
    tm.assert_extension_array_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，为 test_getitem_temporal 函数定义多个参数化的测试用例
@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_getitem_temporal(pa_type):
    # GH 53326: 标识 GitHub issue 53326，测试获取时间相关类型的元素操作

    # 创建 ArrowExtensionArray 对象
    arr = ArrowExtensionArray(pa.array([1, 2, 3], type=pa_type))

    # 获取指定位置的元素
    result = arr[1]

    # 根据 pa_type 的类型设置期望的结果
    if pa.types.is_duration(pa_type):
        expected = pd.Timedelta(2, unit=pa_type.unit).as_unit(pa_type.unit)
        assert isinstance(result, pd.Timedelta)
    else:
        expected = pd.Timestamp(2, unit=pa_type.unit, tz=pa_type.tz).as_unit(pa_type.unit)
        assert isinstance(result, pd.Timestamp)

    # 断言结果与期望结果的单位相同
    assert result.unit == expected.unit
    # 断言结果与期望结果相等
    assert result == expected


# 使用 pytest 的 parametrize 装饰器，为 test_iter_temporal 函数定义多个参数化的测试用例
@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES
)
def test_iter_temporal(pa_type):
    # GH 53326: 标识 GitHub issue 53326，测试迭代时间相关类型的操作

    # 创建 ArrowExtensionArray 对象
    arr = ArrowExtensionArray(pa.array([1, None], type=pa_type))

    # 将 ArrowExtensionArray 转换为列表
    result = list(arr)

    # 根据 pa_type 的类型设置期望的结果
    if pa.types.is_duration(pa_type):
        expected = [
            pd.Timedelta(1, unit=pa_type.unit).as_unit(pa_type.unit),
            pd.NA,
        ]
        assert isinstance(result[0], pd.Timedelta)
    else:
        expected = [
            pd.Timestamp(1, unit=pa_type.unit, tz=pa_type.tz).as_unit(pa_type.unit),
            pd.NA,
        ]
        assert isinstance(result[0], pd.Timestamp)

    # 断言结果与期望结果的单位相同
    assert result[0].unit == expected[0].unit
    # 断言结果与期望结果相等
    assert result == expected


# 定义测试函数 test_groupby_series_size_returns_pa_int，测试分组后 Series 的大小返回类型是否为 pa.int
def test_groupby_series_size_returns_pa_int(data):
    # GH 54132: 标识 GitHub issue 54132，测试分组 Series 后大小返回类型为 pa.int

    # 创建 Series 对象
    ser = pd.Series(data[:3], index=["a", "a", "b"])

    # 对 Series 进行分组并获取大小
    result = ser.groupby(level=0).size()

    # 期望的结果
    expected = pd.Series([2, 1], dtype="int64[pyarrow]", index=["a", "b"])

    # 断言结果与期望结果相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，为 test_to_numpy_temporal 函数定义多个参数化的测试用例
@pytest.mark.parametrize(
    "pa_type", tm.DATETIME_PYARROW_DTYPES + tm.TIMEDELTA_PYARROW_DTYPES, ids=repr
)
@pytest.mark.parametrize("dtype", [None, object])
def test_to_numpy_temporal(pa_type, dtype):
    # GH 53326: 标识 GitHub issue 53326，测试将时间相关类型转换为 numpy 数组的操作
    # GH 55997: 标识 GitHub issue 55997，如果可能的话，使用 NaT 返回 datetime64/timedelta64 类型

    # 创建 ArrowExtensionArray 对象
    arr = ArrowExtensionArray(pa.array([1, None], type=pa_type))

    # 将 ArrowExtensionArray 转换为 numpy 数组
    result = arr.to_numpy(dtype=dtype)

    # 根据 pa_type 的类型设置期望的值
    if pa.types.is_duration(pa_type):
        value = pd.Timedelta(1, unit=pa_type.unit).as_unit(pa_type.unit)
    else:
        value = pd.Timestamp(1, unit=pa_type.unit, tz=pa_type.tz).as_unit(pa_type.unit)
    # 如果数据类型为 object 或者是带有时区信息的时间戳类型
    if dtype == object or (pa.types.is_timestamp(pa_type) and pa_type.tz is not None):
        # 如果数据类型为 object，设置缺失值为 pd.NA
        if dtype == object:
            na = pd.NA
        else:
            # 否则，设置缺失时间值为 pd.NaT
            na = pd.NaT
        # 创建一个包含 value 和 na 的 numpy 数组，数据类型为 object
        expected = np.array([value, na], dtype=object)
        # 断言第一个结果的单位与 value 的单位相同
        assert result[0].unit == value.unit
    else:
        # 如果数据类型不是 object，根据 pa_type 转换为对应的 pandas 数据类型，并设置为 "nat" 类型的缺失值
        na = pa_type.to_pandas_dtype().type("nat", pa_type.unit)
        # 将 value 转换为 numpy 数组
        value = value.to_numpy()
        # 创建一个包含 value 和 na 的 numpy 数组
        expected = np.array([value, na])
        # 断言第一个结果的日期时间数据与 pa_type 的单位相同
        assert np.datetime_data(result[0])[0] == pa_type.unit
    # 使用 pytest 的 assert_numpy_array_equal 方法断言 result 与 expected 数组相等
    tm.assert_numpy_array_equal(result, expected)
def test_groupby_count_return_arrow_dtype(data_missing):
    # 创建一个包含'A', 'B', 'C'列的DataFrame，其中'B'和'C'列包含缺失值data_missing
    df = pd.DataFrame({"A": [1, 1], "B": data_missing, "C": data_missing})
    # 对DataFrame按'A'列进行分组，统计每个分组中非缺失值的数量
    result = df.groupby("A").count()
    # 创建预期的DataFrame，包含一个分组，索引为'A'，列为'B'和'C'，数据类型为'int64[pyarrow]'
    expected = pd.DataFrame(
        [[1, 1]],
        index=pd.Index([1], name="A"),
        columns=["B", "C"],
        dtype="int64[pyarrow]",
    )
    # 使用测试框架检查结果DataFrame是否等于预期DataFrame
    tm.assert_frame_equal(result, expected)


def test_fixed_size_list():
    # GH#55000
    # 创建一个包含固定大小列表的Series，元素为[[1, 2], [3, 4]]，数据类型为ArrowDtype(pa.list_(pa.int64(), list_size=2))
    ser = pd.Series(
        [[1, 2], [3, 4]], dtype=ArrowDtype(pa.list_(pa.int64(), list_size=2))
    )
    # 获取Series的dtype的类型
    result = ser.dtype.type
    # 断言类型是否为list
    assert result == list


def test_arrowextensiondtype_dataframe_repr():
    # GH 54062
    # 创建一个包含日期范围的DataFrame，列名为'col'，数据类型为ArrowDtype(ArrowPeriodType("D"))
    df = pd.DataFrame(
        pd.period_range("2012", periods=3),
        columns=["col"],
        dtype=ArrowDtype(ArrowPeriodType("D")),
    )
    # 获取DataFrame的字符串表示形式
    result = repr(df)
    # TODO: repr的值可能不符合预期；解决pyarrow.ExtensionType值如何显示的问题
    expected = "     col\n0  15340\n1  15341\n2  15342"
    # 使用测试框架检查结果的字符串表示形式是否等于预期字符串
    assert result == expected


def test_pow_missing_operand():
    # GH 55512
    # 创建一个包含整数和None值的Series，数据类型为'int64[pyarrow]'
    k = pd.Series([2, None], dtype="int64[pyarrow]")
    # 对Series进行指数运算，指数为None，填充值为3
    result = k.pow(None, fill_value=3)
    # 创建预期的Series，数据类型为'int64[pyarrow]'，包含指数运算后的结果
    expected = pd.Series([8, None], dtype="int64[pyarrow]")
    # 使用测试框架检查结果Series是否等于预期Series
    tm.assert_series_equal(result, expected)


@pytest.mark.skipif(
    pa_version_under11p0, reason="Decimal128 to string cast implemented in pyarrow 11"
)
def test_decimal_parse_raises():
    # GH 56984
    # 创建一个包含字符串的Series，数据类型为ArrowDtype(pa.string())
    ser = pd.Series(["1.2345"], dtype=ArrowDtype(pa.string()))
    # 使用断言检查是否抛出指定的异常，如果字符串转换为Decimal128会导致数据丢失
    with pytest.raises(
        pa.lib.ArrowInvalid, match="Rescaling Decimal128 value would cause data loss"
    ):
        # 将Series转换为指定的数据类型ArrowDtype(pa.decimal128(1, 0))
        ser.astype(ArrowDtype(pa.decimal128(1, 0)))


@pytest.mark.skipif(
    pa_version_under11p0, reason="Decimal128 to string cast implemented in pyarrow 11"
)
def test_decimal_parse_succeeds():
    # GH 56984
    # 创建一个包含字符串的Series，数据类型为ArrowDtype(pa.string())
    ser = pd.Series(["1.2345"], dtype=ArrowDtype(pa.string()))
    # 创建要转换的数据类型ArrowDtype(pa.decimal128(5, 4))
    dtype = ArrowDtype(pa.decimal128(5, 4))
    # 将Series转换为指定的数据类型
    result = ser.astype(dtype)
    # 创建预期的Series，数据类型为指定的数据类型，包含转换后的结果
    expected = pd.Series([Decimal("1.2345")], dtype=dtype)
    # 使用测试框架检查结果Series是否等于预期Series
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.TIMEDELTA_PYARROW_DTYPES)
def test_duration_fillna_numpy(pa_type):
    # GH 54707
    # 创建一个包含None和整数的Series，数据类型为ArrowDtype(pa_type)
    ser1 = pd.Series([None, 2], dtype=ArrowDtype(pa_type))
    # 创建一个numpy数组，数据类型为时间间隔类型'np.timedelta64[...]'
    ser2 = pd.Series(np.array([1, 3], dtype=f"m8[{pa_type.unit}]"))
    # 对第一个Series执行fillna操作，使用第二个Series填充缺失值
    result = ser1.fillna(ser2)
    # 创建预期的Series，数据类型为ArrowDtype(pa_type)，包含填充缺失值后的结果
    expected = pd.Series([1, 2], dtype=ArrowDtype(pa_type))
    # 使用测试框架检查结果Series是否等于预期Series
    tm.assert_series_equal(result, expected)


def test_comparison_not_propagating_arrow_error():
    # GH#54944
    # 创建一个包含大整数的Series，数据类型为'uint64[pyarrow]'
    a = pd.Series([1 << 63], dtype="uint64[pyarrow]")
    # 创建一个包含None的Series，数据类型为'int64[pyarrow]'
    b = pd.Series([None], dtype="int64[pyarrow]")
    # 使用断言检查是否抛出指定的异常，如果比较操作传播了Arrow的错误
    with pytest.raises(pa.lib.ArrowInvalid, match="Integer value"):
        # 执行Series之间的小于操作
        a < b


def test_factorize_chunked_dictionary():
    # GH 54844
    # 创建一个包含分块数组的ArrowExtensionArray的Series
    pa_array = pa.chunked_array(
        [pa.array(["a"]).dictionary_encode(), pa.array(["b"]).dictionary_encode()]
    )
    ser = pd.Series(ArrowExtensionArray(pa_array))
    # 对Series执行factorize操作，获取索引和唯一值
    res_indices, res_uniques = ser.factorize()
    # 创建一个 NumPy 数组，包含整数元素 0 和 1，数据类型为 np.intp（平台相关的整数类型）
    exp_indicies = np.array([0, 1], dtype=np.intp)
    
    # 使用 ArrowExtensionArray 的 combine_chunks 方法组合成一个 Pandas 索引对象，并将其转换为 pd.Index 对象
    exp_uniques = pd.Index(ArrowExtensionArray(pa_array.combine_chunks()))
    
    # 使用 tm.assert_numpy_array_equal 函数断言 res_indices 与 exp_indicies 相等
    tm.assert_numpy_array_equal(res_indices, exp_indicies)
    
    # 使用 tm.assert_index_equal 函数断言 res_uniques 与 exp_uniques 相等
    tm.assert_index_equal(res_uniques, exp_uniques)
def test_dictionary_astype_categorical():
    # GH#56672
    # 创建两个包含字典编码的数组
    arrs = [
        pa.array(np.array(["a", "x", "c", "a"])).dictionary_encode(),
        pa.array(np.array(["a", "d", "c"])).dictionary_encode(),
    ]
    # 将数组转换为Series对象
    ser = pd.Series(ArrowExtensionArray(pa.chunked_array(arrs)))
    # 对Series进行类型转换为"category"
    result = ser.astype("category")
    # 创建预期的分类类型索引
    categories = pd.Index(["a", "x", "c", "d"], dtype=ArrowDtype(pa.string()))
    # 创建预期的Series对象，使用指定的分类类型
    expected = pd.Series(
        ["a", "x", "c", "a", "a", "d", "c"],
        dtype=pd.CategoricalDtype(categories=categories),
    )
    # 断言结果与预期是否相等
    tm.assert_series_equal(result, expected)


def test_arrow_floordiv():
    # GH 55561
    # 创建两个包含PyArrow类型的Series对象
    a = pd.Series([-7], dtype="int64[pyarrow]")
    b = pd.Series([4], dtype="int64[pyarrow]")
    # 创建预期的结果Series对象
    expected = pd.Series([-2], dtype="int64[pyarrow]")
    # 执行整数除法操作
    result = a // b
    # 断言结果与预期是否相等
    tm.assert_series_equal(result, expected)


def test_arrow_floordiv_large_values():
    # GH 56645
    # 创建包含大整数的Series对象
    a = pd.Series([1425801600000000000], dtype="int64[pyarrow]")
    # 创建预期的结果Series对象
    expected = pd.Series([1425801600000], dtype="int64[pyarrow]")
    # 执行整数除法操作
    result = a // 1_000_000
    # 断言结果与预期是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "uint64[pyarrow]"])
def test_arrow_floordiv_large_integral_result(dtype):
    # GH 56676
    # 创建包含大整数的Series对象
    a = pd.Series([18014398509481983], dtype=dtype)
    # 执行整数除法操作
    result = a // 1
    # 断言结果与原始Series对象是否相等
    tm.assert_series_equal(result, a)


@pytest.mark.parametrize("pa_type", tm.SIGNED_INT_PYARROW_DTYPES)
def test_arrow_floordiv_larger_divisor(pa_type):
    # GH 56676
    # 创建包含负整数的Series对象
    dtype = ArrowDtype(pa_type)
    a = pd.Series([-23], dtype=dtype)
    # 执行整数除法操作
    result = a // 24
    # 创建预期的结果Series对象
    expected = pd.Series([-1], dtype=dtype)
    # 断言结果与预期是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.SIGNED_INT_PYARROW_DTYPES)
def test_arrow_floordiv_integral_invalid(pa_type):
    # GH 56676
    # 获取指定类型的最小值
    min_value = np.iinfo(pa_type.to_pandas_dtype()).min
    # 创建包含最小值的Series对象
    a = pd.Series([min_value], dtype=ArrowDtype(pa_type))
    # 断言抛出指定异常，当进行非法整数除法操作时
    with pytest.raises(pa.lib.ArrowInvalid, match="overflow|not in range"):
        a // -1
    with pytest.raises(pa.lib.ArrowInvalid, match="divide by zero"):
        a // 0


@pytest.mark.parametrize("dtype", tm.FLOAT_PYARROW_DTYPES_STR_REPR)
def test_arrow_floordiv_floating_0_divisor(dtype):
    # GH 56676
    # 创建包含浮点数的Series对象
    a = pd.Series([2], dtype=dtype)
    # 执行整数除法操作
    result = a // 0
    # 创建预期的结果Series对象
    expected = pd.Series([float("inf")], dtype=dtype)
    # 断言结果与预期是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("dtype", ["float64", "datetime64[ns]", "timedelta64[ns]"])
def test_astype_int_with_null_to_numpy_dtype(dtype):
    # GH 57093
    # 创建包含整数和空值的Series对象
    ser = pd.Series([1, None], dtype="int64[pyarrow]")
    # 执行类型转换操作
    result = ser.astype(dtype)
    # 创建预期的结果Series对象
    expected = pd.Series([1, None], dtype=dtype)
    # 断言结果与预期是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("pa_type", tm.ALL_INT_PYARROW_DTYPES)
def test_arrow_integral_floordiv_large_values(pa_type):
    # GH 56676
    # 获取指定类型的最大值
    max_value = np.iinfo(pa_type.to_pandas_dtype()).max
    dtype = ArrowDtype(pa_type)
    # 创建包含最大值的Series对象
    a = pd.Series([max_value], dtype=dtype)
    # 创建一个包含单个元素的 Pandas Series，元素值为 1，使用指定的数据类型 dtype
    b = pd.Series([1], dtype=dtype)
    
    # 计算 Series a 与 Series b 之间的整数除法运算，将结果存储在变量 result 中
    result = a // b
    
    # 使用测试工具 tm 来比较 result 和 a 两个 Series 是否相等，如果不相等则会引发异常
    tm.assert_series_equal(result, a)
# 使用pytest.mark.parametrize装饰器，为dtype参数指定多个测试参数组合
@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "uint64[pyarrow]"])
# 定义测试函数test_arrow_true_division_large_divisor，用于测试箭头操作的真除法
# GH 56706 是 GitHub 上的 issue 编号
def test_arrow_true_division_large_divisor(dtype):
    # 创建一个包含单个元素 0 的 pandas Series 对象，指定数据类型为dtype
    a = pd.Series([0], dtype=dtype)
    # 创建一个包含单个大数值 18014398509481983 的 pandas Series 对象，指定数据类型为dtype
    b = pd.Series([18014398509481983], dtype=dtype)
    # 创建一个预期结果的 pandas Series 对象，包含单个元素 0，数据类型为 "float64[pyarrow]"
    expected = pd.Series([0], dtype="float64[pyarrow]")
    # 执行真除法操作，计算结果存储在 result 中
    result = a / b
    # 使用测试工具tm.assert_series_equal检查计算结果是否与预期结果相等
    tm.assert_series_equal(result, expected)


# 使用pytest.mark.parametrize装饰器，为dtype参数指定多个测试参数组合
@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "uint64[pyarrow]"])
# 定义测试函数test_arrow_floor_division_large_divisor，用于测试箭头操作的地板除法
# GH 56706 是 GitHub 上的 issue 编号
def test_arrow_floor_division_large_divisor(dtype):
    # 创建一个包含单个元素 0 的 pandas Series 对象，指定数据类型为dtype
    a = pd.Series([0], dtype=dtype)
    # 创建一个包含单个大数值 18014398509481983 的 pandas Series 对象，指定数据类型为dtype
    b = pd.Series([18014398509481983], dtype=dtype)
    # 创建一个预期结果的 pandas Series 对象，包含单个元素 0，数据类型与输入的dtype相同
    expected = pd.Series([0], dtype=dtype)
    # 执行地板除法操作，计算结果存储在 result 中
    result = a // b
    # 使用测试工具tm.assert_series_equal检查计算结果是否与预期结果相等
    tm.assert_series_equal(result, expected)


# 定义测试函数test_string_to_datetime_parsing_cast，用于测试字符串到日期时间类型转换和强制转换
# GH 56266 是 GitHub 上的 issue 编号
def test_string_to_datetime_parsing_cast():
    # 创建包含多个日期时间字符串的列表
    string_dates = ["2020-01-01 04:30:00", "2020-01-02 00:00:00", "2020-01-03 00:00:00"]
    # 创建一个 pandas Series 对象，包含上述日期时间字符串，数据类型为 "timestamp[s][pyarrow]"
    result = pd.Series(string_dates, dtype="timestamp[s][pyarrow]")
    # 创建一个预期结果的 pandas Series 对象，使用 ArrowExtensionArray 对象包装转换后的日期时间数组
    expected = pd.Series(
        ArrowExtensionArray(pa.array(pd.to_datetime(string_dates), from_pandas=True))
    )
    # 使用测试工具tm.assert_series_equal检查计算结果是否与预期结果相等
    tm.assert_series_equal(result, expected)


# 使用@pytest.mark.skipif装饰器，根据条件跳过测试
@pytest.mark.skipif(
    pa_version_under13p0, reason="pairwise_diff_checked not implemented in pyarrow"
)
# 定义测试函数test_interpolate_not_numeric，测试非数值数据的插值操作
def test_interpolate_not_numeric(data):
    # 如果数据不是数值类型，则预期引发 ValueError 异常，匹配错误消息 "Values must be numeric."
    if not data.dtype._is_numeric:
        with pytest.raises(ValueError, match="Values must be numeric."):
            # 将数据转换为 pandas Series 对象并尝试插值
            pd.Series(data).interpolate()


# 使用@pytest.mark.skipif装饰器，根据条件跳过测试
@pytest.mark.skipif(
    pa_version_under13p0, reason="pairwise_diff_checked not implemented in pyarrow"
)
# 使用@pytest.mark.parametrize装饰器，为dtype参数指定多个测试参数组合
@pytest.mark.parametrize("dtype", ["int64[pyarrow]", "float64[pyarrow]"])
# 定义测试函数test_interpolate_linear，测试线性插值操作
def test_interpolate_linear(dtype):
    # 创建一个包含空值的 pandas Series 对象，数据类型为dtype
    ser = pd.Series([None, 1, 2, None, 4, None], dtype=dtype)
    # 执行插值操作，计算结果存储在 result 中
    result = ser.interpolate()
    # 创建一个预期结果的 pandas Series 对象，包含经过插值后的数值，数据类型与输入的dtype相同
    expected = pd.Series([None, 1, 2, 3, 4, None], dtype=dtype)
    # 使用测试工具tm.assert_series_equal检查计算结果是否与预期结果相等
    tm.assert_series_equal(result, expected)


# 定义测试函数test_string_to_time_parsing_cast，测试字符串到时间类型转换和强制转换
# GH 56463 是 GitHub 上的 issue 编号
def test_string_to_time_parsing_cast():
    # 创建包含单个时间字符串的列表
    string_times = ["11:41:43.076160"]
    # 创建一个 pandas Series 对象，包含上述时间字符串，数据类型为 "time64[us][pyarrow]"
    result = pd.Series(string_times, dtype="time64[us][pyarrow]")
    # 创建一个预期结果的 pandas Series 对象，使用 ArrowExtensionArray 对象包装转换后的时间数组
    expected = pd.Series(
        ArrowExtensionArray(pa.array([time(11, 41, 43, 76160)], from_pandas=True))
    )
    # 使用测试工具tm.assert_series_equal检查计算结果是否与预期结果相等
    tm.assert_series_equal(result, expected)


# 定义测试函数test_to_numpy_float，测试将 Series 对象转换为 numpy 数组并强制转换数据类型
# GH#56267 是 GitHub 上的 issue 编号
def test_to_numpy_float():
    # 创建一个包含整数和空值的 pandas Series 对象，数据类型为 "float[pyarrow]"
    ser = pd.Series([32, 40, None], dtype="float[pyarrow]")
    # 执行数据类型转换操作，将数据类型转换为 "float64"
    result = ser.astype("float64")
    # 创建一个预期结果的 pandas Series 对象，包含经过转换后的数据，数据类型为 "float64"
    expected = pd.Series([32, 40, np.nan], dtype="float64")
    # 使用测试工具tm.assert_series_equal检查计算结果是否与预期结果相等
    tm.assert_series_equal(result, expected)


# 定义测试函数test_to_numpy_timestamp_to_int，测试将时间戳 Series 对象转换为 numpy 数组
# GH 55997 是 GitHub 上的 issue 编号
def test_to_numpy_timestamp_to_int():
    # 创建一个包含单个时间字符串的 pandas Series 对象，数据类型为 "timestamp[ns][pyarrow]"
    ser = pd.Series(["2020-01-01 04:30:00"], dtype="timestamp[ns][pyarrow]")
    # 执行数据类型转换操作，将 Series 对象转换为 numpy 数组，数据类型为 np.int64
    result = ser.to_numpy(dtype=np.int64)
    # 创建一个预期结果的 numpy 数组，包含经过转换后的时间戳
    expected = np.array([1577853000000000000])
    # 使用测试工具tm.assert_numpy_array_equal检查计算结果是否与预期结果相等
    tm.assert_numpy_array_equal(result, expected)


# 使用@pytest.mark.parametrize装饰器，为arrow_type参数指定多个测试参数组合
@pytest.mark.parametrize("arrow_type", [pa.large_string(), pa.string()])
# 定义测试函数test_cast_dictionary_different_value_dtype，测试将 DataFrame 列转换为不同的 Arrow 数据类型
def test_cast_dictionary_different_value_dtype(arrow_type):
    # 创建一个包含字符串数据的 DataFrame 对象
    df = pd.DataFrame({"a": ["x", "y"]}, dtype="string[pyarrow]")
    # 创建一个 ArrowDtype 对象，指定列 "a" 的数据类型为字典类型，值的数据类型为 arrow_type
    data_type = ArrowDtype(pa
# 定义一个测试函数，用于测试 Series 对象的 map 方法在处理缺失值时的行为
def test_map_numeric_na_action():
    # 创建一个 Series 对象，包含整数和缺失值，指定数据类型为 int64[pyarrow]
    ser = pd.Series([32, 40, None], dtype="int64[pyarrow]")
    # 使用 map 方法，对 Series 中的每个元素应用 lambda 函数，当遇到缺失值时指定忽略
    result = ser.map(lambda x: 42, na_action="ignore")
    # 创建一个期望的 Series 对象，其中缺失值被映射为 NaN，数据类型为 float64
    expected = pd.Series([42.0, 42.0, np.nan], dtype="float64")
    # 使用测试工具库中的 assert_series_equal 函数比较 result 和 expected 两个 Series 对象
    tm.assert_series_equal(result, expected)
```
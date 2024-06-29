# `D:\src\scipysrc\pandas\pandas\tests\extension\test_masked.py`

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

import numpy as np  # 导入 NumPy 库，用于数组操作
import pytest  # 导入 pytest 库，用于测试框架

from pandas.compat import (  # 导入 pandas 兼容性模块中的 IS64 和 is_platform_windows 函数
    IS64,
    is_platform_windows,
)
from pandas.compat.numpy import np_version_gt2  # 导入 pandas 兼容性模块中的 np_version_gt2 函数

from pandas.core.dtypes.common import (  # 导入 pandas 核心数据类型模块中的类型判断函数
    is_float_dtype,
    is_signed_integer_dtype,
    is_unsigned_integer_dtype,
)

import pandas as pd  # 导入 pandas 库并简写为 pd
import pandas._testing as tm  # 导入 pandas 测试模块中的 _testing，并简写为 tm
from pandas.core.arrays.boolean import BooleanDtype  # 导入 pandas 核心数组模块中的布尔类型
from pandas.core.arrays.floating import (  # 导入 pandas 核心数组模块中的浮点数类型
    Float32Dtype,
    Float64Dtype,
)
from pandas.core.arrays.integer import (  # 导入 pandas 核心数组模块中的整数类型
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
)
from pandas.tests.extension import base  # 导入 pandas 扩展测试模块中的 base

# 判断是否为 Windows 或 32 位系统，返回布尔值
is_windows_or_32bit = (is_platform_windows() and not np_version_gt2) or not IS64

# 设置 pytest 标记，忽略特定的警告信息
pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in divide:RuntimeWarning"
    ),
    pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning"),
    # 对于浮点数情况，忽略溢出警告信息
    pytest.mark.filterwarnings("ignore:overflow encountered in reduce:RuntimeWarning"),
]


# 创建数据生成函数，返回包含整数的列表和 NaN 值
def make_data():
    return list(range(1, 9)) + [pd.NA] + list(range(10, 98)) + [pd.NA] + [99, 100]


# 创建浮点数数据生成函数，返回包含浮点数和 NaN 值的列表
def make_float_data():
    return (
        list(np.arange(0.1, 0.9, 0.1))
        + [pd.NA]
        + list(np.arange(1, 9.8, 0.1))
        + [pd.NA]
        + [9.9, 10.0]
    )


# 创建布尔类型数据生成函数，返回包含 True、False 和 NaN 值的列表
def make_bool_data():
    return [True, False] * 4 + [np.nan] + [True, False] * 44 + [np.nan] + [True, False]


# 定义 pytest fixture，根据请求返回相应的数据类型对象
@pytest.fixture(
    params=[
        Int8Dtype,
        Int16Dtype,
        Int32Dtype,
        Int64Dtype,
        UInt8Dtype,
        UInt16Dtype,
        UInt32Dtype,
        UInt64Dtype,
        Float32Dtype,
        Float64Dtype,
        BooleanDtype,
    ]
)
def dtype(request):
    return request.param()


# 定义 pytest fixture，根据数据类型生成对应的数据数组
@pytest.fixture
def data(dtype):
    if dtype.kind == "f":
        data = make_float_data()
    elif dtype.kind == "b":
        data = make_bool_data()
    else:
        data = make_data()
    return pd.array(data, dtype=dtype)


# 定义 pytest fixture，根据数据类型生成包含全部为 2 的数据数组（仅对布尔类型有效）
@pytest.fixture
def data_for_twos(dtype):
    if dtype.kind == "b":
        return pd.array(np.ones(100), dtype=dtype)
    return pd.array(np.ones(100) * 2, dtype=dtype)


# 定义 pytest fixture，生成包含缺失值的数据数组
@pytest.fixture
def data_missing(dtype):
    if dtype.kind == "f":
        return pd.array([pd.NA, 0.1], dtype=dtype)
    # 如果数据类型是布尔类型（boolean），返回包含缺失值和True的Pandas数组
    elif dtype.kind == "b":
        return pd.array([np.nan, True], dtype=dtype)
    # 否则，返回包含缺失值和整数1的Pandas数组，数据类型由参数dtype指定
    return pd.array([pd.NA, 1], dtype=dtype)
@pytest.fixture
# 创建一个数据生成器，用于排序测试，根据数据类型生成不同类型的数据
def data_for_sorting(dtype):
    if dtype.kind == "f":
        return pd.array([0.1, 0.2, 0.0], dtype=dtype)
    elif dtype.kind == "b":
        return pd.array([True, True, False], dtype=dtype)
    return pd.array([1, 2, 0], dtype=dtype)


@pytest.fixture
# 创建一个数据生成器，用于排序测试，包含缺失值的情况
def data_missing_for_sorting(dtype):
    if dtype.kind == "f":
        return pd.array([0.1, pd.NA, 0.0], dtype=dtype)
    elif dtype.kind == "b":
        return pd.array([True, np.nan, False], dtype=dtype)
    return pd.array([1, pd.NA, 0], dtype=dtype)


@pytest.fixture
# 返回一个函数，用于比较是否为 pd.NA
def na_cmp():
    # 我们是 pd.NA
    return lambda x, y: x is pd.NA and y is pd.NA


@pytest.fixture
# 创建一个数据生成器，用于分组测试，根据数据类型生成不同类型的数据
def data_for_grouping(dtype):
    if dtype.kind == "f":
        b = 0.1
        a = 0.0
        c = 0.2
    elif dtype.kind == "b":
        b = True
        a = False
        c = b
    else:
        b = 1
        a = 0
        c = 2

    na = pd.NA
    return pd.array([b, b, na, na, a, a, b, c], dtype=dtype)


class TestMaskedArrays(base.ExtensionTests):
    @pytest.mark.parametrize("na_action", [None, "ignore"])
    # 测试数据映射操作，根据不同的 NA 操作进行测试
    def test_map(self, data_missing, na_action):
        result = data_missing.map(lambda x: x, na_action=na_action)
        if data_missing.dtype == Float32Dtype():
            # 映射操作将对象转换为 float64 数组
            expected = data_missing.to_numpy(dtype="float64", na_value=np.nan)
        else:
            expected = data_missing.to_numpy()
        tm.assert_numpy_array_equal(result, expected)

    # 测试数据映射操作，忽略 NA 值的情况
    def test_map_na_action_ignore(self, data_missing_for_sorting):
        zero = data_missing_for_sorting[2]
        result = data_missing_for_sorting.map(lambda x: zero, na_action="ignore")
        if data_missing_for_sorting.dtype.kind == "b":
            expected = np.array([False, pd.NA, False], dtype=object)
        else:
            expected = np.array([zero, np.nan, zero])
        tm.assert_numpy_array_equal(result, expected)

    # 获取预期的异常情况，用于测试操作名称、对象和另一个对象
    def _get_expected_exception(self, op_name, obj, other):
        try:
            dtype = tm.get_dtype(obj)
        except AttributeError:
            # 传入的参数顺序颠倒
            dtype = tm.get_dtype(other)

        if dtype.kind == "b":
            if op_name.strip("_").lstrip("r") in ["pow", "truediv", "floordiv"]:
                # 与非掩码布尔类型的行为匹配
                return NotImplementedError
            elif op_name in ["__sub__", "__rsub__"]:
                # 异常消息将包含 "numpy boolean subtract"
                return TypeError
            return None
        return None
    def _cast_pointwise_result(self, op_name: str, obj, other, pointwise_result):
        sdtype = tm.get_dtype(obj)  # 获取对象的数据类型
        expected = pointwise_result  # 将pointwise_result赋值给expected变量

        if op_name in ("eq", "ne", "le", "ge", "lt", "gt"):
            return expected.astype("boolean")  # 如果操作是比较操作，将结果转换为布尔类型

        if sdtype.kind in "iu":
            if op_name in ("__rtruediv__", "__truediv__", "__div__"):
                filled = expected.fillna(np.nan)  # 填充缺失值为NaN
                expected = filled.astype("Float64")  # 将结果转换为Float64类型
            else:
                # 结果与'最大'（int64）数据类型组合
                expected = expected.astype(sdtype)  # 将结果转换为与数据类型相同的类型
        elif sdtype.kind == "b":
            if op_name in (
                "__floordiv__",
                "__rfloordiv__",
                "__pow__",
                "__rpow__",
                "__mod__",
                "__rmod__",
            ):
                # 保持布尔类型
                expected = expected.astype("Int8")  # 将结果转换为Int8类型

            elif op_name in ("__truediv__", "__rtruediv__"):
                # 布尔值与数值的组合不会生成正确的结果
                # （numpy中的除法将布尔值视为数值）
                op = self.get_op_from_name(op_name)  # 从操作名称获取操作
                expected = self._combine(obj.astype(float), other, op)  # 使用_float64来组合
                expected = expected.astype("Float64")  # 将结果转换为Float64类型

            if op_name == "__rpow__":
                # 对于rpow，组合不会传播NaN
                result = getattr(obj, op_name)(other)  # 调用对象的rpow方法
                expected[result.isna()] = np.nan  # 将结果中为NaN的部分设置为NaN
        else:
            # 结果与'最大'（float64）数据类型组合
            expected = expected.astype(sdtype)  # 将结果转换为与数据类型相同的类型
        return expected  # 返回处理后的结果

    def test_divmod_series_array(self, data, data_for_twos, request):
        if data.dtype.kind == "b":
            mark = pytest.mark.xfail(
                reason="Inconsistency between floordiv and divmod; we raise for "
                "floordiv but not for divmod. This matches what we do for "
                "non-masked bool dtype."
            )
            request.applymarker(mark)  # 应用pytest的标记
        super().test_divmod_series_array(data, data_for_twos)  # 调用父类的方法进行测试

    def test_combine_le(self, data_repeated):
        # TODO: 在此处修补self是一个不好的模式
        orig_data1, orig_data2 = data_repeated(2)  # 使用data_repeated函数获取两份数据
        if orig_data1.dtype.kind == "b":
            self._combine_le_expected_dtype = "boolean"  # 如果数据类型为布尔型，则设置预期数据类型为布尔型
        else:
            # TODO: 我们能将此设置为布尔型吗？
            self._combine_le_expected_dtype = object  # 否则设置预期数据类型为对象型
        super().test_combine_le(data_repeated)  # 调用父类的方法进行测试

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        if op_name in ["any", "all"] and ser.dtype.kind != "b":
            pytest.skip(reason="Tested in tests/reductions/test_reductions.py")  # 如果操作是'any'或'all'，并且序列不是布尔类型，则跳过测试
        return True  # 否则返回True
    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # 重写以确保测试 pd.NA 而不是 np.nan
        # https://github.com/pandas-dev/pandas/issues/30958

        cmp_dtype = "int64"
        # 如果序列的数据类型是浮点数类型
        if ser.dtype.kind == "f":
            # 将 cmp_dtype 设置为序列的 numpy 数据类型
            cmp_dtype = ser.dtype.numpy_dtype  # type: ignore[union-attr]
        # 如果序列的数据类型是布尔类型
        elif ser.dtype.kind == "b":
            if op_name in ["min", "max"]:
                # 将 cmp_dtype 设置为布尔类型
                cmp_dtype = "bool"

        # 使用非空值并转换为 cmp_dtype 的类型创建备选序列
        alt = ser.dropna().astype(cmp_dtype)
        # 如果操作名称是 "count"
        if op_name == "count":
            # 计算原始序列和备选序列的操作结果
            result = getattr(ser, op_name)()
            expected = getattr(alt, op_name)()
        else:
            # 根据 skipna 参数计算原始序列和备选序列的操作结果
            result = getattr(ser, op_name)(skipna=skipna)
            expected = getattr(alt, op_name)(skipna=skipna)
            # 如果不跳过 NaN 值，并且序列中有 NaN 值，并且操作名称不在 ["any", "all"] 中
            if not skipna and ser.isna().any() and op_name not in ["any", "all"]:
                expected = pd.NA
        # 使用断言函数检查结果是否接近预期
        tm.assert_almost_equal(result, expected)

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        # 如果数组的数据类型是浮点数类型
        if is_float_dtype(arr.dtype):
            # cmp_dtype 设为数组的数据类型名称
            cmp_dtype = arr.dtype.name
        # 如果操作名称在 ["mean", "median", "var", "std", "skew"] 中
        elif op_name in ["mean", "median", "var", "std", "skew"]:
            # cmp_dtype 设为 "Float64"
            cmp_dtype = "Float64"
        # 如果操作名称在 ["max", "min"] 中
        elif op_name in ["max", "min"]:
            # cmp_dtype 设为数组的数据类型名称
            cmp_dtype = arr.dtype.name
        # 如果数组的数据类型在 ["Int64", "UInt64"] 中
        elif arr.dtype in ["Int64", "UInt64"]:
            # cmp_dtype 设为数组的数据类型名称
            cmp_dtype = arr.dtype.name
        # 如果数组的数据类型是有符号整数类型
        elif is_signed_integer_dtype(arr.dtype):
            # TODO: 为什么 Window Numpy 2.0 的数据类型依赖于 skipna？
            cmp_dtype = (
                "Int32"
                if (is_platform_windows() and (not np_version_gt2 or not skipna))
                or not IS64
                else "Int64"
            )
        # 如果数组的数据类型是无符号整数类型
        elif is_unsigned_integer_dtype(arr.dtype):
            # TODO: 为什么 Window Numpy 2.0 的数据类型依赖于 skipna？
            cmp_dtype = (
                "UInt32"
                if (is_platform_windows() and (not np_version_gt2 or not skipna))
                or not IS64
                else "UInt64"
            )
        # 如果数组的数据类型的种类是布尔类型
        elif arr.dtype.kind == "b":
            # 如果操作名称在 ["mean", "median", "var", "std", "skew"] 中
            if op_name in ["mean", "median", "var", "std", "skew"]:
                # cmp_dtype 设为 "Float64"
                cmp_dtype = "Float64"
            # 如果操作名称在 ["min", "max"] 中
            elif op_name in ["min", "max"]:
                # cmp_dtype 设为 "boolean"
                cmp_dtype = "boolean"
            # 如果操作名称在 ["sum", "prod"] 中
            elif op_name in ["sum", "prod"]:
                cmp_dtype = (
                    "Int32"
                    if (is_platform_windows() and (not np_version_gt2 or not skipna))
                    or not IS64
                    else "Int64"
                )
            else:
                # 抛出类型错误，不应该到达此处
                raise TypeError("not supposed to reach this")
        else:
            # 抛出类型错误，不应该到达此处
            raise TypeError("not supposed to reach this")
        # 返回计算得到的 cmp_dtype
        return cmp_dtype
    # 检查是否支持累积操作，始终返回 True
    def _supports_accumulation(self, ser: pd.Series, op_name: str) -> bool:
        return True

    # 检查累积操作，覆盖以确保使用 pd.NA 而不是 np.nan 进行测试
    # 参考：https://github.com/pandas-dev/pandas/issues/30958
    def check_accumulate(self, ser: pd.Series, op_name: str, skipna: bool):
        # 默认长度为 64
        length = 64
        # 如果操作系统是 Windows 或者是 32 位系统
        if is_windows_or_32bit:
            # 当序列的数据类型不为 8 时
            if not ser.dtype.itemsize == 8:  # type: ignore[union-attr]
                length = 32

        # 如果序列的数据类型以 "U" 开头（Unicode 字符串）
        if ser.dtype.name.startswith("U"):
            expected_dtype = f"UInt{length}"
        # 如果序列的数据类型以 "I" 开头（整数）
        elif ser.dtype.name.startswith("I"):
            expected_dtype = f"Int{length}"
        # 如果序列的数据类型以 "F" 开头（浮点数）
        elif ser.dtype.name.startswith("F"):
            # 将期望的数据类型设置为与序列相同的数据类型
            expected_dtype = ser.dtype  # type: ignore[assignment]
        # 如果序列的数据类型的种类是 "b"（布尔类型）
        elif ser.dtype.kind == "b":
            # 如果操作名是 "cummin" 或 "cummax"，期望的数据类型为布尔值
            if op_name in ("cummin", "cummax"):
                expected_dtype = "boolean"
            else:
                expected_dtype = f"Int{length}"

        # 如果期望的数据类型为 "Float32"，并且操作名是 "cumprod" 且 skipna 为 True
        if expected_dtype == "Float32" and op_name == "cumprod" and skipna:
            # TODO: xfail?
            # 跳过测试，因为 Float32 精度可能导致与操作 {op_name} 和 skipna={skipna} 的结果差异很大
            pytest.skip(
                f"Float32 precision lead to large differences with op {op_name} "
                f"and skipna={skipna}"
            )

        # 根据操作名执行不同的累积操作并验证结果是否符合预期
        if op_name == "cumsum":
            # 执行累加操作并验证结果
            result = getattr(ser, op_name)(skipna=skipna)
            expected = pd.Series(
                pd.array(
                    getattr(ser.astype("float64"), op_name)(skipna=skipna),
                    dtype=expected_dtype,
                )
            )
            tm.assert_series_equal(result, expected)
        elif op_name in ["cummax", "cummin"]:
            # 执行累计最大值或累计最小值操作并验证结果
            result = getattr(ser, op_name)(skipna=skipna)
            expected = pd.Series(
                pd.array(
                    getattr(ser.astype("float64"), op_name)(skipna=skipna),
                    dtype=ser.dtype,
                )
            )
            tm.assert_series_equal(result, expected)
        elif op_name == "cumprod":
            # 执行累积乘积操作并验证结果
            result = getattr(ser[:12], op_name)(skipna=skipna)
            expected = pd.Series(
                pd.array(
                    getattr(ser[:12].astype("float64"), op_name)(skipna=skipna),
                    dtype=expected_dtype,
                )
            )
            tm.assert_series_equal(result, expected)
        else:
            # 抛出未实现的操作异常
            raise NotImplementedError(f"{op_name} not supported")
# 定义一个名为 Test2DCompat 的类，该类继承自 base.Dim2CompatTests 类。
class Test2DCompat(base.Dim2CompatTests):
    # pass 语句表示类不包含任何额外的方法或属性，仅作为一个占位符存在。
    pass
```
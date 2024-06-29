# `D:\src\scipysrc\pandas\pandas\tests\extension\test_datetime.py`

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

# Import necessary libraries
import numpy as np
import pytest

# Import specific modules/classes from pandas library
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
from pandas.tests.extension import base


# Fixture to provide DatetimeTZDtype object with specific unit and timezone
@pytest.fixture
def dtype():
    return DatetimeTZDtype(unit="ns", tz="US/Central")


# Fixture to generate DatetimeArray data from a date range with a specific dtype
@pytest.fixture
def data(dtype):
    data = DatetimeArray._from_sequence(
        pd.date_range("2000", periods=100, tz=dtype.tz), dtype=dtype
    )
    return data


# Fixture to generate DatetimeArray data from an array with missing values
@pytest.fixture
def data_missing(dtype):
    return DatetimeArray._from_sequence(
        np.array(["NaT", "2000-01-01"], dtype="datetime64[ns]"), dtype=dtype
    )


# Fixture to generate DatetimeArray data from an array for sorting tests
@pytest.fixture
def data_for_sorting(dtype):
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    c = pd.Timestamp("2000-01-03")
    return DatetimeArray._from_sequence(
        np.array([b, c, a], dtype="datetime64[ns]"), dtype=dtype
    )


# Fixture to generate DatetimeArray data from an array with missing values for sorting tests
@pytest.fixture
def data_missing_for_sorting(dtype):
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    return DatetimeArray._from_sequence(
        np.array([b, "NaT", a], dtype="datetime64[ns]"), dtype=dtype
    )


# Fixture to generate DatetimeArray data for grouping tests
@pytest.fixture
def data_for_grouping(dtype):
    """
    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    a = pd.Timestamp("2000-01-01")
    b = pd.Timestamp("2000-01-02")
    c = pd.Timestamp("2000-01-03")
    na = "NaT"
    return DatetimeArray._from_sequence(
        np.array([b, b, na, na, a, a, b, c], dtype="datetime64[ns]"), dtype=dtype
    )


# Fixture to define custom comparison function for NaT values
@pytest.fixture
def na_cmp():
    def cmp(a, b):
        return a is pd.NaT and a is b

    return cmp


# ----------------------------------------------------------------------------
# Test class inheriting from base.ExtensionTests
class TestDatetimeArray(base.ExtensionTests):

    # Override method to handle expected exceptions for specific operations
    def _get_expected_exception(self, op_name, obj, other):
        if op_name in ["__sub__", "__rsub__"]:
            return None
        return super()._get_expected_exception(op_name, obj, other)

    # Override method to indicate which accumulation operations are supported
    def _supports_accumulation(self, ser, op_name: str) -> bool:
        return op_name in ["cummin", "cummax"]

    # Override method to indicate which reduction operations are supported
    def _supports_reduction(self, obj, op_name: str) -> bool:
        return op_name in ["min", "max", "median", "mean", "std", "any", "all"]

    # Parametrize test with different skipna settings
    @pytest.mark.parametrize("skipna", [True, False])
    # 测试函数，用于测试在数据上应用布尔缩减操作时是否会引发类型错误异常
    def test_reduce_series_boolean(self, data, all_boolean_reductions, skipna):
        # 获取当前布尔缩减方法的名称
        meth = all_boolean_reductions
        # 构建特定操作引发的类型错误异常消息
        msg = f"datetime64 type does not support operation '{meth}'"
        # 断言在执行测试时会引发指定消息的类型错误异常
        with pytest.raises(TypeError, match=msg):
            super().test_reduce_series_boolean(data, all_boolean_reductions, skipna)

    # 测试函数，用于测试 Series 构造函数的行为
    def test_series_constructor(self, data):
        # 移除数据中的任何时间频率属性
        data = data._with_freq(None)
        # 调用父类方法测试 Series 构造函数
        super().test_series_constructor(data)

    # 参数化测试函数，用于测试 Series 的 map 方法
    @pytest.mark.parametrize("na_action", [None, "ignore"])
    def test_map(self, data, na_action):
        # 调用 map 方法，应用简单的 lambda 函数到数据，设置缺失值处理方式
        result = data.map(lambda x: x, na_action=na_action)
        # 断言映射后的结果与原始数据相等
        tm.assert_extension_array_equal(result, data)

    # 测试函数，用于检查在 Series 上应用聚合操作的结果
    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # 如果操作名称为 "median", "mean", "std" 中的一种
        if op_name in ["median", "mean", "std"]:
            # 将序列转换为整数类型的备选视图
            alt = ser.astype("int64")

            # 获取序列上指定操作的实际结果和预期结果
            res_op = getattr(ser, op_name)
            exp_op = getattr(alt, op_name)
            result = res_op(skipna=skipna)
            expected = exp_op(skipna=skipna)

            # 如果操作是 "mean" 或 "median"，则处理额外的时区信息
            if op_name in ["mean", "median"]:
                # 错误：类型 "dtype[Any]" 的成员 "dtype[Any] | ExtensionDtype" 没有属性 "tz"
                tz = ser.dtype.tz  # type: ignore[union-attr]
                expected = pd.Timestamp(expected, tz=tz)
            else:
                expected = pd.Timedelta(expected)

            # 断言实际结果与预期结果几乎相等
            tm.assert_almost_equal(result, expected)

        else:
            # 对于其他操作名称，调用父类方法继续检查
            return super().check_reduce(ser, op_name, skipna)
# 定义一个测试类 Test2DCompat，继承自 base.NDArrayBacked2DTests
class Test2DCompat(base.NDArrayBacked2DTests):
    # 无额外的代码，继承自 base.NDArrayBacked2DTests 的所有测试方法
    pass
```
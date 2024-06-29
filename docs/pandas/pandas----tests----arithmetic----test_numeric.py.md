# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\test_numeric.py`

```
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for numeric dtypes
from __future__ import annotations

from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator

import numpy as np
import pytest

import pandas as pd
from pandas import (
    Index,
    RangeIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    array,
    date_range,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
)

# Fixture to switch between numexpr and python mode with different minimum elements
@pytest.fixture(autouse=True, params=[0, 1000000], ids=["numexpr", "python"])
def switch_numexpr_min_elements(request, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(expr, "_MIN_ELEMENTS", request.param)
        yield request.param

# Fixture providing various numeric-dtypes Index objects for testing
@pytest.fixture(
    params=[
        # TODO: add more dtypes here
        Index(np.arange(5, dtype="float64")),
        Index(np.arange(5, dtype="int64")),
        Index(np.arange(5, dtype="uint64")),
        RangeIndex(5),
    ],
    ids=lambda x: type(x).__name__,
)
def numeric_idx(request):
    """
    Several types of numeric-dtypes Index objects
    """
    return request.param

# Helper function to adjust expected results when dividing by -0.0
def adjust_negative_zero(zero, expected):
    """
    Helper to adjust the expected result if we are dividing by -0.0
    as opposed to 0.0
    """
    if np.signbit(np.array(zero)).any():
        # All entries in the `zero` fixture should be either
        # all-negative or no-negative.
        assert np.signbit(np.array(zero)).all()

        expected *= -1

    return expected

# Function to compare operations between series and other objects
def compare_op(series, other, op):
    left = np.abs(series) if op in (ops.rpow, operator.pow) else series
    right = np.abs(other) if op in (ops.rpow, operator.pow) else other

    cython_or_numpy = op(left, right)
    python = left.combine(right, op)
    if isinstance(other, Series) and not other.index.equals(series.index):
        python.index = python.index._with_freq(None)
    tm.assert_series_equal(cython_or_numpy, python)

# List of left-hand side objects for comparison operations
# This list contains various PandasObject types including RangeIndex, Series, and Index
_ldtypes = ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8"]
lefts: list[Index | Series] = [RangeIndex(10, 40, 10)]
lefts.extend([Series([10, 20, 30], dtype=dtype) for dtype in _ldtypes])
lefts.extend([Index([10, 20, 30], dtype=dtype) for dtype in _ldtypes if dtype != "f2"])

# ------------------------------------------------------------------
# Comparisons

class TestNumericComparisons:
    def test_operator_series_comparison_zerorank(self):
        # GH#13006
        # 比较 np.float64(0) 是否大于 Series([1, 2, 3])
        result = np.float64(0) > Series([1, 2, 3])
        # 期望结果是 0.0 大于 Series([1, 2, 3])
        expected = 0.0 > Series([1, 2, 3])
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)
        
        # 比较 Series([1, 2, 3]) 是否小于 np.float64(0)
        result = Series([1, 2, 3]) < np.float64(0)
        # 期望结果是 Series([1, 2, 3]) 小于 0.0
        expected = Series([1, 2, 3]) < 0.0
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)
        
        # 比较 np.array([0, 1, 2])[0] 是否大于 Series([0, 1, 2])
        result = np.array([0, 1, 2])[0] > Series([0, 1, 2])
        # 期望结果是 0.0 大于 Series([1, 2, 3])，这里似乎有误，应该是 Series([0, 1, 2]) 大于 Series([0, 1, 2])
        expected = 0.0 > Series([1, 2, 3])
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)

    def test_df_numeric_cmp_dt64_raises(self, box_with_array, fixed_now_ts):
        # GH#8932, GH#22163
        # 设置时间戳为固定值
        ts = fixed_now_ts
        # 创建一个包含 0 到 4 的数组对象
        obj = np.array(range(5))
        # 使用测试工具方法包装预期的对象
        obj = tm.box_expected(obj, box_with_array)

        # 断言比较操作引发无效比较异常
        assert_invalid_comparison(obj, ts, box_with_array)

    def test_compare_invalid(self):
        # GH#8058
        # ops testing
        # 创建一个包含标准正态分布随机数的 Series 对象，命名为 0
        a = Series(np.random.default_rng(2).standard_normal(5), name=0)
        # 创建另一个包含标准正态分布随机数的 Series 对象
        b = Series(np.random.default_rng(2).standard_normal(5))
        # 将 b 的名称设置为日期时间戳 "2000-01-01"
        b.name = pd.Timestamp("2000-01-01")
        # 断言两个 Series 对象之间的操作：a/b 和 1/(b/a) 相等
        tm.assert_series_equal(a / b, 1 / (b / a))

    def test_numeric_cmp_string_numexpr_path(self, box_with_array, monkeypatch):
        # GH#36377, GH#35700
        # 根据 box_with_array 设置 box 和 xbox
        box = box_with_array
        xbox = box if box is not Index else np.ndarray

        # 创建一个包含标准正态分布随机数的 Series 对象，长度为 51
        obj = Series(np.random.default_rng(2).standard_normal(51))
        # 使用测试工具方法包装预期的对象，并禁止转置
        obj = tm.box_expected(obj, box, transpose=False)
        
        # 使用 monkeypatch 对象上下文设置 _MIN_ELEMENTS 属性为 50
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 50)
            # 执行 obj == "a" 操作，得到比较结果
            result = obj == "a"

        # 创建一个长度为 51 的全零 Series 对象，并使用测试工具方法包装预期的对象
        expected = Series(np.zeros(51, dtype=bool))
        expected = tm.box_expected(expected, xbox, transpose=False)
        # 断言比较结果与期望结果相等
        tm.assert_equal(result, expected)

        # 使用 monkeypatch 对象上下文设置 _MIN_ELEMENTS 属性为 50
        with monkeypatch.context() as m:
            m.setattr(expr, "_MIN_ELEMENTS", 50)
            # 执行 obj != "a" 操作，得到比较结果
            result = obj != "a"
        # 断言比较结果与期望结果的取反相等
        tm.assert_equal(result, ~expected)

        # 预期抛出类型错误异常信息
        msg = "Invalid comparison between dtype=float64 and str"
        with pytest.raises(TypeError, match=msg):
            # 执行 obj < "a" 操作，验证是否抛出异常
            obj < "a"
# ------------------------------------------------------------------
# Numeric dtypes Arithmetic with Datetime/Timedelta Scalar

# 定义测试类 TestNumericArraylikeArithmeticWithDatetimeLike，用于测试数值类型与日期时间/时间差标量的算术运算
class TestNumericArraylikeArithmeticWithDatetimeLike:

    # 参数化装饰器，针对不同的数组类型进行参数化测试
    @pytest.mark.parametrize("box_cls", [np.array, Index, Series])
    # 参数化装饰器，针对不同的左操作数进行参数化测试，使用自定义的标识符命名
    @pytest.mark.parametrize(
        "left", lefts, ids=lambda x: type(x).__name__ + str(x.dtype)
    )
    # 测试乘法运算测试用例
    def test_mul_td64arr(self, left, box_cls):
        # GH#22390
        # 定义右操作数为包含 [1, 2, 3] 的 numpy 数组，数据类型为 'm8[s]'
        right = np.array([1, 2, 3], dtype="m8[s]")
        # 将右操作数转换为对应的 box_cls 类型（如 numpy 数组、Index 或 Series）
        right = box_cls(right)

        # 定义期望的结果为 TimedeltaIndex 对象，包含 ["10s", "40s", "90s"]，数据类型与 right 相同
        expected = TimedeltaIndex(["10s", "40s", "90s"], dtype=right.dtype)

        # 如果左操作数为 Series 或 box_cls 为 Series，则将期望结果转换为 Series 对象
        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)
        
        # 断言期望结果的数据类型与 right 的数据类型相同
        assert expected.dtype == right.dtype

        # 执行左操作数乘以右操作数的运算
        result = left * right
        # 使用测试工具库 tm 进行结果的相等性断言
        tm.assert_equal(result, expected)

        # 执行右操作数乘以左操作数的运算
        result = right * left
        # 使用测试工具库 tm 进行结果的相等性断言
        tm.assert_equal(result, expected)

    # 参数化装饰器，针对不同的数组类型进行参数化测试
    @pytest.mark.parametrize("box_cls", [np.array, Index, Series])
    # 参数化装饰器，针对不同的左操作数进行参数化测试，使用自定义的标识符命名
    @pytest.mark.parametrize(
        "left", lefts, ids=lambda x: type(x).__name__ + str(x.dtype)
    )
    # 测试除法运算测试用例
    def test_div_td64arr(self, left, box_cls):
        # GH#22390
        # 定义右操作数为包含 [10, 40, 90] 的 numpy 数组，数据类型为 'm8[s]'
        right = np.array([10, 40, 90], dtype="m8[s]")
        # 将右操作数转换为对应的 box_cls 类型（如 numpy 数组、Index 或 Series）
        right = box_cls(right)

        # 定义期望的结果为 TimedeltaIndex 对象，包含 ["1s", "2s", "3s"]，数据类型与 right 相同
        expected = TimedeltaIndex(["1s", "2s", "3s"], dtype=right.dtype)
        
        # 如果左操作数为 Series 或 box_cls 为 Series，则将期望结果转换为 Series 对象
        if isinstance(left, Series) or box_cls is Series:
            expected = Series(expected)
        
        # 断言期望结果的数据类型与 right 的数据类型相同
        assert expected.dtype == right.dtype

        # 执行右操作数除以左操作数的运算
        result = right / left
        # 使用测试工具库 tm 进行结果的相等性断言
        tm.assert_equal(result, expected)

        # 执行右操作数整除左操作数的运算
        result = right // left
        # 使用测试工具库 tm 进行结果的相等性断言
        tm.assert_equal(result, expected)

        # 针对 'true_divide' 和 'floor_divide' 操作，进行类型错误的异常断言
        # (true_) needed for min-versions build 2022-12-26
        msg = "ufunc '(true_)?divide' cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            left / right

        msg = "ufunc 'floor_divide' cannot use operands with types"
        with pytest.raises(TypeError, match=msg):
            left // right

    # TODO: also test Tick objects;
    #  see test_numeric_arr_rdiv_tdscalar for note on these failing
    # 参数化装饰器，针对不同的标量 Timedelta 进行参数化测试
    @pytest.mark.parametrize(
        "scalar_td",
        [
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            Timedelta(days=1).to_pytimedelta(),
            Timedelta(days=1).to_timedelta64().astype("timedelta64[s]"),
            Timedelta(days=1).to_timedelta64().astype("timedelta64[ms]"),
        ],
        # 使用类型名称作为自定义的标识符
        ids=lambda x: type(x).__name__,
    )
    # 定义一个测试方法，用于测试数值数组与时间增量标量相乘的情况
    def test_numeric_arr_mul_tdscalar(self, scalar_td, numeric_idx, box_with_array):
        # GH#19333，引用 GitHub 上的 issue 编号，提供上下文
        box = box_with_array
        index = numeric_idx
        # 创建一个预期的 TimedeltaIndex，其中每个元素为一天的时间增量，长度与 index 相同
        expected = TimedeltaIndex([Timedelta(days=n) for n in range(len(index))])
        # 检查 scalar_td 是否为 np.timedelta64 类型
        if isinstance(scalar_td, np.timedelta64):
            dtype = scalar_td.dtype
            # 将预期结果转换为与 scalar_td 相同的数据类型
            expected = expected.astype(dtype)
        # 如果 scalar_td 是 timedelta 类型，则将预期结果转换为微秒精度
        elif type(scalar_td) is timedelta:
            expected = expected.astype("m8[us]")

        # 调用测试辅助函数，将 index 对象与 box 进行包装
        index = tm.box_expected(index, box)
        # 调用测试辅助函数，将预期结果与 box 进行包装
        expected = tm.box_expected(expected, box)

        # 计算 index 与 scalar_td 的乘积，进行结果断言
        result = index * scalar_td
        tm.assert_equal(result, expected)

        # 计算 scalar_td 与 index 的乘积（交换顺序），进行结果断言
        commute = scalar_td * index
        tm.assert_equal(commute, expected)

    @pytest.mark.parametrize(
        "scalar_td",
        [
            Timedelta(days=1),
            Timedelta(days=1).to_timedelta64(),
            Timedelta(days=1).to_pytimedelta(),
        ],
        ids=lambda x: type(x).__name__,
    )
    @pytest.mark.parametrize("dtype", [np.int64, np.float64])
    # 定义另一个测试方法，用于测试数值数组与时间增量标量相乘的 Numexpr 路径
    def test_numeric_arr_mul_tdscalar_numexpr_path(
        self, dtype, scalar_td, box_with_array
    ):
        # GH#44772 for the float64 case，引用 GitHub 上的 issue 编号，提供上下文
        box = box_with_array

        # 创建一个 int64 类型的数组，范围为 [0, 2*10**4)
        arr_i8 = np.arange(2 * 10**4).astype(np.int64, copy=False)
        # 将 arr_i8 转换为指定的 dtype 类型的数组
        arr = arr_i8.astype(dtype, copy=False)
        # 调用测试辅助函数，将 arr 对象与 box 进行包装
        obj = tm.box_expected(arr, box, transpose=False)

        # 创建预期的 timedelta64[D] 类型的数组，转换为微秒精度的 timedelta64[ns] 类型
        expected = arr_i8.view("timedelta64[D]").astype("timedelta64[ns]")
        # 如果 scalar_td 是 timedelta 类型，则将预期结果转换为微秒精度
        if type(scalar_td) is timedelta:
            expected = expected.astype("timedelta64[us]")

        # 调用测试辅助函数，将预期结果与 box 进行包装
        expected = tm.box_expected(expected, box, transpose=False)

        # 计算 obj 与 scalar_td 的乘积，进行结果断言
        result = obj * scalar_td
        tm.assert_equal(result, expected)

        # 计算 scalar_td 与 obj 的乘积（交换顺序），进行结果断言
        result = scalar_td * obj
        tm.assert_equal(result, expected)

    # 定义一个测试方法，用于测试数值数组与时间增量标量相除的情况
    def test_numeric_arr_rdiv_tdscalar(self, three_days, numeric_idx, box_with_array):
        # 获取 box_with_array 的引用
        box = box_with_array

        # 获取 numeric_idx 的索引为 1 到 3 的切片
        index = numeric_idx[1:3]

        # 创建一个预期的 TimedeltaIndex，包含两个元素分别表示 "3 Days" 和 "36 Hours"
        expected = TimedeltaIndex(["3 Days", "36 Hours"])
        # 检查 three_days 的类型是否为 np.timedelta64
        if isinstance(three_days, np.timedelta64):
            dtype = three_days.dtype
            # 如果 three_days 的精度低于 "m8[s]"，则将 dtype 设为 "m8[s]"
            if dtype < np.dtype("m8[s]"):
                dtype = np.dtype("m8[s]")
            # 将预期结果转换为与 three_days 相同的数据类型
            expected = expected.astype(dtype)
        # 如果 three_days 是 timedelta 类型，则将预期结果转换为微秒精度
        elif type(three_days) is timedelta:
            expected = expected.astype("m8[us]")
        # 如果 three_days 是 pd.offsets 下的时间增量类型，则将预期结果转换为秒精度
        elif isinstance(
            three_days,
            (pd.offsets.Day, pd.offsets.Hour, pd.offsets.Minute, pd.offsets.Second),
        ):
            expected = expected.astype("m8[s]")

        # 调用测试辅助函数，将 index 对象与 box 进行包装
        index = tm.box_expected(index, box)
        # 调用测试辅助函数，将预期结果与 box 进行包装
        expected = tm.box_expected(expected, box)

        # 计算 three_days 与 index 的除法，进行结果断言
        result = three_days / index
        tm.assert_equal(result, expected)

        # 尝试计算 index 除以 three_days，预期抛出 TypeError 异常
        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            index / three_days
    # 使用 pytest 的 pytest.mark.parametrize 装饰器，为测试函数 test_add_sub_datetimedeltalike_invalid 参数化多组输入
    @pytest.mark.parametrize(
        "other",
        [
            # 创建 Timedelta 对象，表示31小时
            Timedelta(hours=31),
            # 将 Timedelta 对象转换为 Python 标准库中的 timedelta 对象
            Timedelta(hours=31).to_pytimedelta(),
            # 将 Timedelta 对象转换为 numpy 中的 timedelta64 类型
            Timedelta(hours=31).to_timedelta64(),
            # 将 Timedelta 对象转换为 timedelta64 类型，再转换为 m8[h]（小时）单位
            Timedelta(hours=31).to_timedelta64().astype("m8[h]"),
            # 表示不确定的时间值（Not a Time）
            np.timedelta64("NaT"),
            # 表示不确定的时间值（Not a Time），以天为单位
            np.timedelta64("NaT", "D"),
            # 表示 3 分钟的 pandas 时间偏移对象
            pd.offsets.Minute(3),
            # 表示 0 秒的 pandas 时间偏移对象
            pd.offsets.Second(0),
            # 表示 2021-01-01 日的 pandas 时间戳，带时区 Asia/Tokyo
            # 在 pandas 1.0 之前会引发 NullFrequencyError，但在 1.0 中移除了该行为
            pd.Timestamp("2021-01-01", tz="Asia/Tokyo"),
            # 表示 2021-01-01 日的 pandas 时间戳
            pd.Timestamp("2021-01-01"),
            # 将 pandas 时间戳转换为 Python 标准库中的 datetime 对象
            pd.Timestamp("2021-01-01").to_pydatetime(),
            # 将带有 UTC 时区的 pandas 时间戳转换为 Python 标准库中的 datetime 对象
            pd.Timestamp("2021-01-01", tz="UTC").to_pydatetime(),
            # 将 pandas 时间戳转换为 numpy 中的 datetime64 类型
            pd.Timestamp("2021-01-01").to_datetime64(),
            # 表示不确定的日期时间值（Not a Time），以纳秒为单位
            np.datetime64("NaT", "ns"),
            # 表示不确定的 pandas 日期时间值（Not a Time）
            pd.NaT,
        ],
        # 使用 repr 函数将每个输入对象转换为字符串，作为其在测试报告中的标识符
        ids=repr,
    )
    # 定义测试函数 test_add_sub_datetimedeltalike_invalid，验证加法和减法操作的非法情况
    def test_add_sub_datetimedeltalike_invalid(
        self, numeric_idx, other, box_with_array
    ):
        # 获取测试参数中的数组对象
        box = box_with_array

        # 获取与索引对应的预期值，包装为测试用的对象
        left = tm.box_expected(numeric_idx, box)
        
        # 定义错误消息，用于断言异常情况
        msg = "|".join(
            [
                "unsupported operand type",  # 不支持的操作数类型
                "Addition/subtraction of integers and integer-arrays",  # 整数和整数数组的加法/减法
                "Instead of adding/subtracting",  # 不能使用操作数进行加法/减法
                "cannot use operands with types dtype",  # 不能使用指定类型的操作数
                "Concatenation operation is not implemented for NumPy arrays",  # NumPy 数组不支持连接操作
                "Cannot (add|subtract) NaT (to|from) ndarray",  # 不能将 NaT 添加或减去 ndarray
                # pd.array vs np.datetime64 case
                r"operand type\(s\) all returned NotImplemented from __array_ufunc__",  # 所有操作数类型均从 __array_ufunc__ 返回了 NotImplemented
                "can only perform ops with numeric values",  # 只能对数值进行操作
                "cannot subtract DatetimeArray from ndarray",  # 不能从 ndarray 中减去 DatetimeArray
                # pd.Timedelta(1) + Index([0, 1, 2])
                "Cannot add or subtract Timedelta from integers",  # 不能将 Timedelta 添加或减去整数
            ]
        )
        
        # 断言异常情况
        assert_invalid_addsub_type(left, other, msg)
# ------------------------------------------------------------------
# Arithmetic

class TestDivisionByZero:
    def test_div_zero(self, zero, numeric_idx):
        idx = numeric_idx

        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        # Adjust for negative zeros in expected values using a helper function
        expected2 = adjust_negative_zero(zero, expected)

        # Perform division operation idx / zero
        result = idx / zero
        tm.assert_index_equal(result, expected2)

        # Compatibility check with Series, casting types before division
        ser_compat = Series(idx).astype("i8") / np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_floordiv_zero(self, zero, numeric_idx):
        idx = numeric_idx

        expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        # Adjust for negative zeros in expected values using a helper function
        expected2 = adjust_negative_zero(zero, expected)

        # Perform floor division operation idx // zero
        result = idx // zero
        tm.assert_index_equal(result, expected2)

        # Compatibility check with Series, casting types before floor division
        ser_compat = Series(idx).astype("i8") // np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(expected))

    def test_mod_zero(self, zero, numeric_idx):
        idx = numeric_idx

        expected = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        
        # Perform modulus operation idx % zero
        result = idx % zero
        tm.assert_index_equal(result, expected)
        
        # Compatibility check with Series, casting types before modulus operation
        ser_compat = Series(idx).astype("i8") % np.array(zero).astype("i8")
        tm.assert_series_equal(ser_compat, Series(result))

    def test_divmod_zero(self, zero, numeric_idx):
        idx = numeric_idx

        exleft = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
        exright = Index([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
        
        # Adjust for negative zeros in exleft values using a helper function
        exleft = adjust_negative_zero(zero, exleft)

        # Perform divmod operation idx // zero
        result = divmod(idx, zero)
        tm.assert_index_equal(result[0], exleft)
        tm.assert_index_equal(result[1], exright)

    @pytest.mark.parametrize("op", [operator.truediv, operator.floordiv])
    def test_div_negative_zero(self, zero, numeric_idx, op):
        # Check that -1 / -0.0 returns np.inf, not -np.inf
        if numeric_idx.dtype == np.uint64:
            pytest.skip(f"Div by negative 0 not relevant for {numeric_idx.dtype}")
        idx = numeric_idx - 3

        expected = Index([-np.inf, -np.inf, -np.inf, np.nan, np.inf], dtype=np.float64)
        # Adjust for negative zeros in expected values using a helper function
        expected = adjust_negative_zero(zero, expected)

        # Perform division operation idx / zero or idx // zero depending on op
        result = op(idx, zero)
        tm.assert_index_equal(result, expected)

    # ------------------------------------------------------------------

    @pytest.mark.parametrize("dtype1", [np.int64, np.float64, np.uint64])
    def test_ser_div_ser(
        self,
        switch_numexpr_min_elements,
        dtype1,
        any_real_numpy_dtype,
    ):
        # 不再对任何操作进行整数除法，但处理0的情况
        dtype2 = any_real_numpy_dtype

        # 创建第一个Series对象，包含整数数据并转换为dtype1类型
        first = Series([3, 4, 5, 8], name="first").astype(dtype1)
        
        # 创建第二个Series对象，包含实数数据并转换为dtype2类型
        second = Series([0, 0, 0, 3], name="second").astype(dtype2)

        # 设置numpy错误状态为忽略
        with np.errstate(all="ignore"):
            # 创建期望结果Series对象，进行浮点数除法运算
            expected = Series(
                first.values.astype(np.float64) / second.values,
                dtype="float64",
                name=None,
            )
        
        # 将期望结果的前三个元素设置为正无穷
        expected.iloc[0:3] = np.inf
        
        # 如果第一个Series对象的数据类型为'int64'且第二个Series对象的数据类型为'float32'
        if first.dtype == "int64" and second.dtype == "float32":
            # 当使用numexpr时，类型转换规则稍有不同，int64/float32组合将结果转换为float32而不是float64
            if expr.USE_NUMEXPR and switch_numexpr_min_elements == 0:
                expected = expected.astype("float32")

        # 计算两个Series对象的除法结果
        result = first / second
        
        # 断言两个Series对象不相等
        tm.assert_series_equal(result, expected)
        assert not result.equals(second / first)

    @pytest.mark.parametrize("dtype1", [np.int64, np.float64, np.uint64])
    def test_ser_divmod_zero(self, dtype1, any_real_numpy_dtype):
        # GH#26987
        dtype2 = any_real_numpy_dtype
        
        # 创建左边Series对象，包含数据类型为dtype1的两个元素
        left = Series([1, 1]).astype(dtype1)
        
        # 创建右边Series对象，包含数据类型为dtype2的两个元素
        right = Series([0, 2]).astype(dtype2)

        # 设置期望结果为左边Series对象对右边Series对象进行整数除法和求余运算的结果
        # pandas约定将1 // 0设为np.inf，而不是像numpy将其设为np.nan；修补下面的`expected[0]`
        expected = left // right, left % right
        expected = list(expected)
        expected[0] = expected[0].astype(np.float64)
        expected[0][0] = np.inf
        
        # 执行divmod操作
        result = divmod(left, right)

        # 断言两个Series对象相等
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        # rdivmod情况
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_ser_divmod_inf(self):
        # 创建左边Series对象，包含数据为正无穷和1.0
        left = Series([np.inf, 1.0])
        
        # 创建右边Series对象，包含数据为正无穷和2.0
        right = Series([np.inf, 2.0])

        # 设置期望结果为左边Series对象对右边Series对象进行整数除法和求余运算的结果
        expected = left // right, left % right
        
        # 执行divmod操作
        result = divmod(left, right)

        # 断言两个Series对象相等
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        # rdivmod情况
        result = divmod(left.values, right)
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    def test_rdiv_zero_compat(self):
        # GH#8674
        # 创建一个由5个0组成的numpy数组
        zero_array = np.array([0] * 5)
        
        # 使用标准正态分布生成一个长度为5的numpy数组
        data = np.random.default_rng(2).standard_normal(5)
        
        # 创建期望结果为包含5个0.0的Series对象
        expected = Series([0.0] * 5)

        # 对zero_array与data执行除法操作，并与期望结果进行比较
        result = zero_array / Series(data)
        tm.assert_series_equal(result, expected)

        # 对Series对象zero_array与data执行除法操作，并与期望结果进行比较
        result = Series(zero_array) / data
        tm.assert_series_equal(result, expected)

        # 对Series对象zero_array与Series对象data执行除法操作，并与期望结果进行比较
        result = Series(zero_array) / Series(data)
        tm.assert_series_equal(result, expected)
    def test_div_zero_inf_signs(self):
        # GH#9144, inf signing
        # 创建一个包含[-1, 0, 1]的Series对象，命名为"first"
        ser = Series([-1, 0, 1], name="first")
        # 创建一个期望的Series对象，包含[-∞, NaN, ∞]，命名为"first"
        expected = Series([-np.inf, np.nan, np.inf], name="first")

        # 对Series对象进行除以0的操作
        result = ser / 0
        # 使用测试工具比较result和expected的Series对象
        tm.assert_series_equal(result, expected)

    def test_rdiv_zero(self):
        # GH#9144
        # 创建一个包含[-1, 0, 1]的Series对象，命名为"first"
        ser = Series([-1, 0, 1], name="first")
        # 创建一个期望的Series对象，包含[0.0, NaN, 0.0]，命名为"first"
        expected = Series([0.0, np.nan, 0.0], name="first")

        # 对0进行除以Series对象的操作
        result = 0 / ser
        # 使用测试工具比较result和expected的Series对象
        tm.assert_series_equal(result, expected)

    def test_floordiv_div(self):
        # GH#9144
        # 创建一个包含[-1, 0, 1]的Series对象，命名为"first"

        # 对Series对象进行整数除法操作
        result = ser // 0
        # 创建一个期望的Series对象，包含[-∞, NaN, ∞]，命名为"first"
        expected = Series([-np.inf, np.nan, np.inf], name="first")
        # 使用测试工具比较result和expected的Series对象
        tm.assert_series_equal(result, expected)

    def test_df_div_zero_df(self):
        # integer div, but deal with the 0's (GH#9144)
        # 创建一个包含{"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]}的DataFrame对象
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        # 对DataFrame对象进行除以自身的操作
        result = df / df

        # 创建一个包含[1.0, 1.0, 1.0, 1.0]的Series对象，命名为"first"
        first = Series([1.0, 1.0, 1.0, 1.0])
        # 创建一个包含[NaN, NaN, NaN, 1]的Series对象，命名为"second"
        second = Series([np.nan, np.nan, np.nan, 1])
        # 创建一个期望的DataFrame对象，包含{"first": first, "second": second}
        expected = pd.DataFrame({"first": first, "second": second})
        # 使用测试工具比较result和expected的DataFrame对象
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_array(self):
        # integer div, but deal with the 0's (GH#9144)
        # 创建一个包含{"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]}的DataFrame对象
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        # 创建一个包含[1.0, 1.0, 1.0, 1.0]的Series对象，命名为"first"
        first = Series([1.0, 1.0, 1.0, 1.0])
        # 创建一个包含[NaN, NaN, NaN, 1]的Series对象，命名为"second"
        second = Series([np.nan, np.nan, np.nan, 1])
        # 创建一个期望的DataFrame对象，包含{"first": first, "second": second}
        expected = pd.DataFrame({"first": first, "second": second})

        # 忽略所有numpy错误状态
        with np.errstate(all="ignore"):
            # 将DataFrame对象转换为浮点数，并对其进行除以自身的操作
            arr = df.values.astype("float") / df.values
        # 创建一个结果DataFrame对象，使用arr的值，索引和列名与df相同
        result = pd.DataFrame(arr, index=df.index, columns=df.columns)
        # 使用测试工具比较result和expected的DataFrame对象
        tm.assert_frame_equal(result, expected)

    def test_df_div_zero_int(self):
        # integer div, but deal with the 0's (GH#9144)
        # 创建一个包含{"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]}的DataFrame对象
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})

        # 对DataFrame对象进行除以0的操作
        result = df / 0
        # 创建一个包含无穷大值的DataFrame对象，索引和列名与df相同
        expected = pd.DataFrame(np.inf, index=df.index, columns=df.columns)
        # 将第一列中的前三行设置为NaN
        expected.iloc[0:3, 1] = np.nan
        # 使用测试工具比较result和expected的DataFrame对象
        tm.assert_frame_equal(result, expected)

        # numpy有一个稍微不同（错误的）处理方式
        with np.errstate(all="ignore"):
            # 将DataFrame对象转换为浮点数，并对其进行除以0的操作
            arr = df.values.astype("float64") / 0
        # 创建一个结果DataFrame对象，使用arr的值，索引和列名与df相同
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        # 使用测试工具比较result2和expected的DataFrame对象
        tm.assert_frame_equal(result2, expected)

    def test_df_div_zero_series_does_not_commute(self):
        # integer div, but deal with the 0's (GH#9144)
        # 创建一个包含10行5列的随机标准正态分布值的DataFrame对象
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        # 从DataFrame对象中选择第一列作为Series对象
        ser = df[0]
        # 对Series对象和DataFrame对象进行除法操作
        res = ser / df
        res2 = df / ser
        # 断言res和res2填充NaN后不相等
        assert not res.fillna(0).equals(res2.fillna(0))

    # ------------------------------------------------------------------
    # Mod By Zero
    def test_df_mod_zero_df(self):
        # GH#3590, modulo as ints
        # 创建一个包含两列的DataFrame，每列包含几个整数值
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        
        # 创建一个Series对象，包含四个零，将其类型强制转换为float64
        first = Series([0, 0, 0, 0])
        first = first.astype("float64")
        
        # 创建另一个Series对象，包含四个NaN值
        second = Series([np.nan, np.nan, np.nan, 0])
        
        # 创建一个预期的DataFrame对象，包含与上述两个Series对象对应的列
        expected = pd.DataFrame({"first": first, "second": second})
        
        # 对DataFrame对象进行模运算，并将结果与预期结果进行比较
        result = df % df
        tm.assert_frame_equal(result, expected)
        
        # 如果不使用copy=False，DataFrame对象将被合并，导致"first"列的数据类型变为float64而不是int64
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]}, copy=False)
        
        # 创建一个包含四个零的Series对象，并指定数据类型为int64
        first = Series([0, 0, 0, 0], dtype="int64")
        
        # 创建另一个Series对象，包含四个NaN值
        second = Series([np.nan, np.nan, np.nan, 0])
        
        # 创建另一个预期的DataFrame对象，包含与上述两个Series对象对应的列
        expected = pd.DataFrame({"first": first, "second": second})
        
        # 再次对DataFrame对象进行模运算，并将结果与新的预期结果进行比较
        result = df % df
        tm.assert_frame_equal(result, expected)

    def test_df_mod_zero_array(self):
        # GH#3590, modulo as ints
        # 创建一个包含两列的DataFrame，每列包含几个整数值
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        
        # 创建一个包含四个零的Series对象，并指定数据类型为float64
        first = Series([0, 0, 0, 0], dtype="float64")
        
        # 创建另一个Series对象，包含四个NaN值
        second = Series([np.nan, np.nan, np.nan, 0])
        
        # 创建一个预期的DataFrame对象，包含与上述两个Series对象对应的列
        expected = pd.DataFrame({"first": first, "second": second})
        
        # 使用numpy对DataFrame对象进行模运算，结果存储在arr中，数据类型为float64
        with np.errstate(all="ignore"):
            arr = df.values % df.values
        
        # 创建另一个DataFrame对象，使用arr中的数据，指定相同的索引和列名，数据类型为float64
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns, dtype="float64")
        
        # 修改结果DataFrame对象的部分数据，将第一列的前三行设置为NaN
        result2.iloc[0:3, 1] = np.nan
        
        # 将结果DataFrame对象与预期结果进行比较
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_int(self):
        # GH#3590, modulo as ints
        # 创建一个包含两列的DataFrame，每列包含几个整数值
        df = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
        
        # 对DataFrame对象执行模运算，结果应该是一个包含NaN值的DataFrame对象
        result = df % 0
        
        # 创建一个预期的DataFrame对象，包含与原始DataFrame对象相同的索引和列名，数据为NaN
        expected = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        
        # 将结果DataFrame对象与预期结果进行比较
        tm.assert_frame_equal(result, expected)
        
        # 使用numpy对DataFrame对象进行模运算，结果存储在arr中，数据类型强制转换为float64
        with np.errstate(all="ignore"):
            arr = df.values.astype("float64") % 0
        
        # 创建另一个DataFrame对象，使用arr中的数据，指定相同的索引和列名
        result2 = pd.DataFrame(arr, index=df.index, columns=df.columns)
        
        # 将结果DataFrame对象与预期结果进行比较
        tm.assert_frame_equal(result2, expected)

    def test_df_mod_zero_series_does_not_commute(self):
        # GH#3590, modulo as ints
        # 创建一个包含随机标准正态分布数据的10行5列的DataFrame对象
        df = pd.DataFrame(np.random.default_rng(2).standard_normal((10, 5)))
        
        # 获取DataFrame对象的第一列作为Series对象
        ser = df[0]
        
        # 尝试将Series对象与DataFrame对象执行模运算，不满足交换律
        res = ser % df
        res2 = df % ser
        
        # 断言结果DataFrame对象的两个版本不相等（考虑NaN值填充后比较）
        assert not res.fillna(0).equals(res2.fillna(0)))
class TestMultiplicationDivision:
    # 定义测试类 TestMultiplicationDivision，用于测试乘法和除法运算相关功能

    # __mul__, __rmul__, __div__, __rdiv__, __floordiv__, __rfloordiv__
    # for non-timestamp/timedelta/period dtypes
    # 定义魔术方法，适用于非时间戳/时间差/周期数据类型

    def test_divide_decimal(self, box_with_array):
        # 测试使用 Decimal 进行除法操作，解决问题 GH#9787
        # 参数 box_with_array 是测试用例提供的盒子对象
        box = box_with_array
        ser = Series([Decimal(10)])
        expected = Series([Decimal(5)])

        # 对 ser 和 expected 进行盒子化处理
        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        # 执行除法操作
        result = ser / Decimal(2)

        # 断言结果是否与期望一致
        tm.assert_equal(result, expected)

        # 执行整除操作
        result = ser // Decimal(2)
        tm.assert_equal(result, expected)

    def test_div_equiv_binop(self):
        # 测试 Series.div 和 Series.__div__ 的等效性
        # 涉及到浮点数/整数的问题，解决 GH#7785
        first = Series([1, 0], name="first")
        second = Series([-0.01, -0.02], name="second")
        expected = Series([-0.01, -np.inf])

        # 使用 Series.div 进行除法运算
        result = second.div(first)
        tm.assert_series_equal(result, expected, check_names=False)

        # 使用 '/' 运算符进行除法运算
        result = second / first
        tm.assert_series_equal(result, expected)

    def test_div_int(self, numeric_idx):
        # 测试整数除法操作
        idx = numeric_idx

        # 执行 idx 除以 1 的操作
        result = idx / 1
        expected = idx.astype("float64")
        tm.assert_index_equal(result, expected)

        # 执行 idx 除以 2 的操作
        result = idx / 2
        expected = Index(idx.values / 2)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.mul, ops.rmul, operator.floordiv])
    def test_mul_int_identity(self, op, numeric_idx, box_with_array):
        # 测试整数乘法的身份性质
        idx = numeric_idx
        idx = tm.box_expected(idx, box_with_array)

        # 执行 op 操作（可能是乘法或整数除法）与 1 的操作
        result = op(idx, 1)
        tm.assert_equal(result, idx)

    def test_mul_int_array(self, numeric_idx):
        # 测试整数与数组的乘法操作
        idx = numeric_idx
        didx = idx * idx

        # 执行 idx 与整数数组乘法的操作
        result = idx * np.array(5, dtype="int64")
        tm.assert_index_equal(result, idx * 5)

        # 确定数组的数据类型
        arr_dtype = "uint64" if idx.dtype == np.uint64 else "int64"

        # 执行 idx 与整数数组范围乘法的操作
        result = idx * np.arange(5, dtype=arr_dtype)
        tm.assert_index_equal(result, didx)

    def test_mul_int_series(self, numeric_idx):
        # 测试整数与 Series 对象的乘法操作
        idx = numeric_idx
        didx = idx * idx

        # 确定数组的数据类型
        arr_dtype = "uint64" if idx.dtype == np.uint64 else "int64"

        # 执行 idx 与整数 Series 对象乘法的操作
        result = idx * Series(np.arange(5, dtype=arr_dtype))
        tm.assert_series_equal(result, Series(didx))

    def test_mul_float_series(self, numeric_idx):
        # 测试浮点数与 Series 对象的乘法操作
        idx = numeric_idx
        rng5 = np.arange(5, dtype="float64")

        # 执行 idx 与浮点数 Series 对象乘法的操作
        result = idx * Series(rng5 + 0.1)
        expected = Series(rng5 * (rng5 + 0.1))
        tm.assert_series_equal(result, expected)

    def test_mul_index(self, numeric_idx):
        # 测试索引对象的乘法操作
        idx = numeric_idx

        # 执行索引对象的乘法操作
        result = idx * idx
        tm.assert_index_equal(result, idx**2)

    def test_mul_datelike_raises(self, numeric_idx):
        # 测试日期类型引发的乘法操作错误
        idx = numeric_idx
        msg = "cannot perform __rmul__ with this index type"

        # 断言执行乘法操作引发 TypeError 异常，匹配给定消息
        with pytest.raises(TypeError, match=msg):
            idx * date_range("20130101", periods=5)
    # 定义一个测试方法，用于测试乘法操作中的尺寸不匹配是否会引发 ValueError 异常
    def test_mul_size_mismatch_raises(self, numeric_idx):
        # 将传入的 numeric_idx 参数赋值给 idx 变量
        idx = numeric_idx
        # 错误信息字符串，用于匹配异常信息
        msg = "operands could not be broadcast together"
        
        # 使用 pytest 模块断言检查乘法操作中的尺寸不匹配是否会抛出 ValueError 异常，并匹配指定的错误信息
        with pytest.raises(ValueError, match=msg):
            idx * idx[0:3]
        
        # 使用 pytest 模块再次断言检查乘法操作中的尺寸不匹配是否会抛出 ValueError 异常，并匹配指定的错误信息
        with pytest.raises(ValueError, match=msg):
            idx * np.array([1, 2])

    # 使用 pytest.mark.parametrize 装饰器，定义一个参数化测试方法，测试指数运算
    @pytest.mark.parametrize("op", [operator.pow, ops.rpow])
    def test_pow_float(self, op, numeric_idx, box_with_array):
        # 根据 box_with_array 参数创建一个 box 对象
        box = box_with_array
        # 将 numeric_idx 参数赋值给 idx 变量
        idx = numeric_idx
        # 使用 op 函数对 idx.values 和 2.0 进行指数运算，创建一个预期的 Index 对象
        expected = Index(op(idx.values, 2.0))

        # 对 idx 和 expected 变量使用 tm.box_expected 方法，进行进一步的处理
        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)

        # 使用 op 函数对 idx 和 2.0 进行指数运算，得到结果
        result = op(idx, 2.0)
        # 使用 tm.assert_equal 方法断言 result 和 expected 相等
        tm.assert_equal(result, expected)

    # 定义一个测试方法，测试取模运算
    def test_modulo(self, numeric_idx, box_with_array):
        # 根据 box_with_array 参数创建一个 box 对象
        box = box_with_array
        # 将 numeric_idx 参数赋值给 idx 变量
        idx = numeric_idx
        # 使用 idx.values 对 2 取模，创建一个预期的 Index 对象
        expected = Index(idx.values % 2)

        # 对 idx 变量使用 tm.box_expected 方法，进行进一步的处理
        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, box)

        # 对 idx 执行取模运算
        result = idx % 2
        # 使用 tm.assert_equal 方法断言 result 和 expected 相等
        tm.assert_equal(result, expected)

    # 定义一个测试方法，测试 divmod 函数对标量进行运算
    def test_divmod_scalar(self, numeric_idx):
        # 将 numeric_idx 参数赋值给 idx 变量
        idx = numeric_idx

        # 使用 divmod 函数对 idx 和 2 进行运算，获取结果
        result = divmod(idx, 2)
        # 使用 np.errstate 函数忽略所有错误，分别对 idx.values 和 2 执行 divmod 运算
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, 2)

        # 创建一个包含 div 和 mod 的预期 Index 对象
        expected = Index(div), Index(mod)
        # 使用 zip 函数遍历 result 和 expected 的每个元素，使用 tm.assert_index_equal 方法断言它们相等
        for r, e in zip(result, expected):
            tm.assert_index_equal(r, e)

    # 定义一个测试方法，测试 divmod 函数对 ndarray 进行运算
    def test_divmod_ndarray(self, numeric_idx):
        # 将 numeric_idx 参数赋值给 idx 变量
        idx = numeric_idx
        # 创建一个与 idx.values 形状相同且元素全为 2 的 ndarray 对象 other
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2

        # 使用 divmod 函数对 idx 和 other 进行运算，获取结果
        result = divmod(idx, other)
        # 使用 np.errstate 函数忽略所有错误，分别对 idx.values 和 other 执行 divmod 运算
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, other)

        # 创建一个包含 div 和 mod 的预期 Index 对象
        expected = Index(div), Index(mod)
        # 使用 zip 函数遍历 result 和 expected 的每个元素，使用 tm.assert_index_equal 方法断言它们相等
        for r, e in zip(result, expected):
            tm.assert_index_equal(r, e)

    # 定义一个测试方法，测试 divmod 函数对 Series 进行运算
    def test_divmod_series(self, numeric_idx):
        # 将 numeric_idx 参数赋值给 idx 变量
        idx = numeric_idx
        # 创建一个与 idx.values 形状相同且元素全为 2 的 ndarray 对象 other
        other = np.ones(idx.values.shape, dtype=idx.values.dtype) * 2

        # 使用 divmod 函数对 idx 和 Series(other) 进行运算，获取结果
        result = divmod(idx, Series(other))
        # 使用 np.errstate 函数忽略所有错误，分别对 idx.values 和 other 执行 divmod 运算
        with np.errstate(all="ignore"):
            div, mod = divmod(idx.values, other)

        # 创建一个包含 div 和 mod 的预期 Series 对象
        expected = Series(div), Series(mod)
        # 使用 zip 函数遍历 result 和 expected 的每个元素，使用 tm.assert_series_equal 方法断言它们相等
        for r, e in zip(result, expected):
            tm.assert_series_equal(r, e)

    # 使用 pytest.mark.parametrize 装饰器，定义一个参数化测试方法，测试操作符与 NumPy 标量的运算
    @pytest.mark.parametrize("other", [np.nan, 7, -23, 2.718, -3.14, np.inf])
    def test_ops_np_scalar(self, other):
        # 使用随机数生成器创建一个 (5, 3) 形状的标准正态分布 ndarray 对象 vals
        vals = np.random.default_rng(2).standard_normal((5, 3))
        # 定义一个 lambda 函数 f，将 vals 转换为 DataFrame 对象，设置索引和列名
        f = lambda x: pd.DataFrame(
            x, index=list("ABCDE"), columns=["jim", "joe", "jolie"]
        )

        # 使用 f 函数根据 vals 创建一个 DataFrame 对象 df
        df = f(vals)

        # 使用 tm.assert_frame_equal 方法断言 df 与 df / np.array(other) 相等
        tm.assert_frame_equal(df / np.array(other), f(vals / other))
        # 使用 tm.assert_frame_equal 方法断言 np.array(other) * df 与 f(vals * other) 相等
        tm.assert_frame_equal(np.array(other) * df, f(vals * other))
        # 使用 tm.assert_frame_equal 方法断言 df + np.array(other) 与 f(vals + other) 相等
        tm.assert_frame_equal(df + np.array(other), f(vals + other))
        # 使用 tm.assert_frame_equal 方法断言 np.array(other) - df 与 f(other - vals) 相等
        tm.assert_frame_equal(np.array(other) - df, f(other - vals))

    # TODO: This came from series.test.test_operators, needs cleanup
    # 定义一个测试函数，用于测试 DataFrame 的操作符
    def test_operators_frame(self):
        # 创建一个 Series 对象 ts，包含从 0 到 9 的浮点数，以及日期范围的索引
        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )
        # 将 Series 对象的名称设为 'ts'
        ts.name = "ts"

        # 创建一个 DataFrame df，包含一个名为 'A' 的列，列的数据来自 ts
        df = pd.DataFrame({"A": ts})

        # 使用测试模块 tm 来断言两个 Series 相等，不检查名称
        tm.assert_series_equal(ts + ts, ts + df["A"], check_names=False)
        tm.assert_series_equal(ts**ts, ts ** df["A"], check_names=False)
        tm.assert_series_equal(ts < ts, ts < df["A"], check_names=False)
        tm.assert_series_equal(ts / ts, ts / df["A"], check_names=False)

    # TODO: 这段代码来自 tests.series.test_analytics，需要清理和与上面的 test_modulo 合并
    def test_modulo2(self):
        with np.errstate(all="ignore"):
            # GH#3590，测试整数情况下的取模操作
            p = pd.DataFrame({"first": [3, 4, 5, 8], "second": [0, 0, 0, 3]})
            # 计算 p["first"] 对 p["second"] 的取模结果
            result = p["first"] % p["second"]
            # 创建期望结果 Series，与上述取模结果对比
            expected = Series(p["first"].values % p["second"].values, dtype="float64")
            expected.iloc[0:3] = np.nan
            # 使用测试模块 tm 断言结果 Series 与期望的 Series 相等
            tm.assert_series_equal(result, expected)

            # 测试对 0 取模的情况
            result = p["first"] % 0
            expected = Series(np.nan, index=p.index, name="first")
            tm.assert_series_equal(result, expected)

            # 将 DataFrame p 的数据类型转换为 float64
            p = p.astype("float64")
            result = p["first"] % p["second"]
            expected = Series(p["first"].values % p["second"].values)
            tm.assert_series_equal(result, expected)

            # 再次将 DataFrame p 的数据类型转换为 float64
            p = p.astype("float64")
            result = p["first"] % p["second"]
            result2 = p["second"] % p["first"]
            # 断言 result 与 result2 不相等
            assert not result.equals(result2)

    # 测试对整数 0 进行取模的情况，GH#9144
    def test_modulo_zero_int(self):
        with np.errstate(all="ignore"):
            # 创建一个包含整数 0 和 1 的 Series 对象 s
            s = Series([0, 1])

            # 对 s 中的元素对 0 取模，期望结果是 NaN
            result = s % 0
            expected = Series([np.nan, np.nan])
            tm.assert_series_equal(result, expected)

            # 对整数 0 对 s 中的元素取模，期望结果是 [NaN, 0.0]
            result = 0 % s
            expected = Series([np.nan, 0.0])
            tm.assert_series_equal(result, expected)
# 定义一个测试类 TestAdditionSubtraction，用于测试加法和减法相关操作
class TestAdditionSubtraction:

    # 使用 pytest 的参数化装饰器，定义测试参数化，用于测试 Series 对象的加法
    @pytest.mark.parametrize(
        "first, second, expected",
        [
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),  # 第一个 Series 对象，包含数据 [1, 2, 3]，索引为 ['A', 'B', 'C']，名称为 'x'
                Series([2, 2, 2], index=list("ABD"), name="x"),  # 第二个 Series 对象，包含数据 [2, 2, 2]，索引为 ['A', 'B', 'D']，名称为 'x'
                Series([3.0, 4.0, np.nan, np.nan], index=list("ABCD"), name="x"),  # 期望结果的 Series 对象，包含数据 [3.0, 4.0, NaN, NaN]，索引为 ['A', 'B', 'C', 'D']，名称为 'x'
            ),
            (
                Series([1, 2, 3], index=list("ABC"), name="x"),  # 第一个 Series 对象，包含数据 [1, 2, 3]，索引为 ['A', 'B', 'C']，名称为 'x'
                Series([2, 2, 2, 2], index=list("ABCD"), name="x"),  # 第二个 Series 对象，包含数据 [2, 2, 2, 2]，索引为 ['A', 'B', 'C', 'D']，名称为 'x'
                Series([3, 4, 5, np.nan], index=list("ABCD"), name="x"),  # 期望结果的 Series 对象，包含数据 [3, 4, 5, NaN]，索引为 ['A', 'B', 'C', 'D']，名称为 'x'
            ),
        ],
    )
    # 定义测试方法 test_add_series，用于测试 Series 对象的加法操作
    def test_add_series(self, first, second, expected):
        # GH#1134: 验证加法操作的正确性
        tm.assert_series_equal(first + second, expected)  # 断言加法操作的结果与期望结果相等
        tm.assert_series_equal(second + first, expected)  # 断言交换顺序的加法操作的结果与期望结果相等

    # 使用 pytest 的参数化装饰器，定义测试参数化，用于测试 DataFrame 对象的加法
    @pytest.mark.parametrize(
        "first, second, expected",
        [
            (
                pd.DataFrame({"x": [1, 2, 3]}, index=list("ABC")),  # 第一个 DataFrame 对象，包含列 'x'，数据为 [1, 2, 3]，索引为 ['A', 'B', 'C']
                pd.DataFrame({"x": [2, 2, 2]}, index=list("ABD")),  # 第二个 DataFrame 对象，包含列 'x'，数据为 [2, 2, 2]，索引为 ['A', 'B', 'D']
                pd.DataFrame({"x": [3.0, 4.0, np.nan, np.nan]}, index=list("ABCD")),  # 期望结果的 DataFrame 对象，包含列 'x'，数据为 [3.0, 4.0, NaN, NaN]，索引为 ['A', 'B', 'C', 'D']
            ),
            (
                pd.DataFrame({"x": [1, 2, 3]}, index=list("ABC")),  # 第一个 DataFrame 对象，包含列 'x'，数据为 [1, 2, 3]，索引为 ['A', 'B', 'C']
                pd.DataFrame({"x": [2, 2, 2, 2]}, index=list("ABCD")),  # 第二个 DataFrame 对象，包含列 'x'，数据为 [2, 2, 2, 2]，索引为 ['A', 'B', 'C', 'D']
                pd.DataFrame({"x": [3, 4, 5, np.nan]}, index=list("ABCD")),  # 期望结果的 DataFrame 对象，包含列 'x'，数据为 [3, 4, 5, NaN]，索引为 ['A', 'B', 'C', 'D']
            ),
        ],
    )
    # 定义测试方法 test_add_frames，用于测试 DataFrame 对象的加法操作
    def test_add_frames(self, first, second, expected):
        # GH#1134: 验证加法操作的正确性
        tm.assert_frame_equal(first + second, expected)  # 断言加法操作的结果与期望结果相等
        tm.assert_frame_equal(second + first, expected)  # 断言交换顺序的加法操作的结果与期望结果相等

    # TODO: This came from series.test.test_operators, needs cleanup
    # 定义测试方法 test_series_frame_radd_bug，用于测试 Series 和 DataFrame 对象的右加法操作
    def test_series_frame_radd_bug(self, fixed_now_ts):
        # GH#353: 验证修复右加法 Bug 的正确性

        # 对 Series 对象进行字符串连接操作
        vals = Series([str(i) for i in range(5)])
        result = "foo_" + vals  # 字符串 "foo_" 与每个元素值连接形成新的 Series 对象
        expected = vals.map(lambda x: "foo_" + x)  # 期望结果为每个元素值前加 "foo_"
        tm.assert_series_equal(result, expected)  # 断言字符串连接操作的结果与期望结果相等

        # 对 DataFrame 对象进行字符串连接操作
        frame = pd.DataFrame({"vals": vals})
        result = "foo_" + frame  # 字符串 "foo_" 与 DataFrame 中每个元素值连接形成新的 DataFrame 对象
        expected = pd.DataFrame({"vals": vals.map(lambda x: "foo_" + x)})  # 期望结果为 DataFrame 中每个元素值前加 "foo_"
        tm.assert_frame_equal(result, expected)  # 断言字符串连接操作的结果与期望结果相等

        # 创建一个时间序列对象 ts
        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10),
            name="ts",
        )

        # 此次真正引发错误
        fix_now = fixed_now_ts.to_pydatetime()  # 将固定时间戳转换为 Python 的 datetime 对象
        msg = "|".join(
            [
                "unsupported operand type",  # 错误消息关键字之一
                "Concatenation operation",  # 错误消息关键字之一，详见链接 https://github.com/numpy/numpy/issues/18832
            ]
        )
        with pytest.raises(TypeError, match=msg):  # 预期捕获 TypeError 类型的异常，并匹配特定的错误消息
            fix_now + ts  # 尝试对时间序列对象执行加法操作时预期引发异常

        with pytest.raises(TypeError, match=msg):  # 预期捕获 TypeError 类型的异常，并匹配特定的错误消息
            ts + fix_now  # 尝试对时间序列对象执行加法操作时预期引发异常

    # TODO: This came from series.test.test_operators, needs cleanup
    def test_datetime64_with_index(self):
        # 使用随机数生成器创建一个包含5个标准正态分布随机数的序列
        ser = Series(np.random.default_rng(2).standard_normal(5))
        # 创建预期结果：序列减去其索引的序列化形式
        expected = ser - ser.index.to_series()
        # 计算结果：序列减去其索引
        result = ser - ser.index
        # 检查结果是否与预期相等
        tm.assert_series_equal(result, expected)

        # GH#4629
        # 使用日期范围创建一个序列，其索引也是日期范围
        ser = Series(
            date_range("20130101", periods=5),
            index=date_range("20130101", periods=5),
        )
        # 创建预期结果：序列减去其索引的序列化形式
        expected = ser - ser.index.to_series()
        # 计算结果：序列减去其索引
        result = ser - ser.index
        # 检查结果是否与预期相等
        tm.assert_series_equal(result, expected)

        # 准备错误消息
        msg = "cannot subtract PeriodArray from DatetimeArray"
        # 断言抛出特定类型的异常，带有指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # GH#18850
            # 尝试将序列减去其索引转换为周期数据
            result = ser - ser.index.to_period()

        # 创建一个DataFrame，包含随机数填充的数据，日期索引
        df = pd.DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)),
            index=date_range("20130101", periods=5),
        )
        # 添加一个日期列
        df["date"] = pd.Timestamp("20130102")
        # 创建预期结果：日期列减去DataFrame的日期索引的序列化形式
        df["expected"] = df["date"] - df.index.to_series()
        # 计算结果：日期列减去DataFrame的日期索引
        df["result"] = df["date"] - df.index
        # 检查结果是否与预期相等，忽略列名检查
        tm.assert_series_equal(df["result"], df["expected"], check_names=False)

    # TODO: taken from tests.frame.test_operators, needs cleanup
    def test_frame_operators(self, float_frame):
        # 获取浮点DataFrame的副本
        frame = float_frame

        # 创建一个随机数数组
        garbage = np.random.default_rng(2).random(4)
        # 创建一个Series，使用随机数数组作为数据，DataFrame列作为索引
        colSeries = Series(garbage, index=np.array(frame.columns))

        # 对DataFrame执行加法运算
        idSum = frame + frame
        # 对DataFrame和Series执行加法运算
        seriesSum = frame + colSeries

        # 遍历加法结果的列和系列
        for col, series in idSum.items():
            for idx, val in series.items():
                # 计算原始值的两倍
                origVal = frame[col][idx] * 2
                # 如果值不是NaN，则断言结果值等于原始值的两倍
                if not np.isnan(val):
                    assert val == origVal
                # 如果结果值是NaN，则断言原始值也是NaN
                else:
                    assert np.isnan(origVal)

        # 遍历Series加法结果的列和系列
        for col, series in seriesSum.items():
            for idx, val in series.items():
                # 计算DataFrame列与Series元素的加法结果
                origVal = frame[col][idx] + colSeries[col]
                # 如果值不是NaN，则断言结果值等于计算出的原始值
                if not np.isnan(val):
                    assert val == origVal
                # 如果结果值是NaN，则断言原始值也是NaN
                else:
                    assert np.isnan(origVal)

    def test_frame_operators_col_align(self, float_frame):
        # 创建DataFrame的另一个版本，列顺序反转
        frame2 = pd.DataFrame(float_frame, columns=["D", "C", "B", "A"])
        # 执行DataFrame的加法运算
        added = frame2 + frame2
        # 创建预期结果：DataFrame的每个元素乘以2
        expected = frame2 * 2
        # 检查加法运算结果与预期结果是否相等
        tm.assert_frame_equal(added, expected)

    def test_frame_operators_none_to_nan(self):
        # 创建一个包含空值的DataFrame
        df = pd.DataFrame({"a": ["a", None, "b"]})
        # 断言DataFrame加上自身的结果与预期结果相等
        tm.assert_frame_equal(df + df, pd.DataFrame({"a": ["aa", np.nan, "bb"]}))

    @pytest.mark.parametrize("dtype", ("float", "int64"))
    def test_frame_operators_empty_like(self, dtype):
        # 测试问题＃10181
        # 创建不同情况下的空DataFrame，并断言加法操作的结果等于原始DataFrame
        frames = [
            pd.DataFrame(dtype=dtype),
            pd.DataFrame(columns=["A"], dtype=dtype),
            pd.DataFrame(index=[0], dtype=dtype),
        ]
        for df in frames:
            assert (df + df).equals(df)
            tm.assert_frame_equal(df + df, df)
    # 使用 pytest 的参数化装饰器，定义多个测试函数，每个函数接受一个操作函数作为参数
    @pytest.mark.parametrize(
        "func",
        [lambda x: x * 2, lambda x: x[::2], lambda x: 5],  # 定义三个不同的操作函数
        ids=["multiply", "slice", "constant"],  # 设置每个函数的标识符
    )
    # 定义测试函数，测试序列的算术操作
    def test_series_operators_arithmetic(self, all_arithmetic_functions, func):
        op = all_arithmetic_functions  # 获取所有算术操作函数
        series = Series(  # 创建一个包含浮点数的序列
            np.arange(10, dtype=np.float64),  # 生成从0到9的浮点数数组
            index=date_range("2020-01-01", periods=10),  # 使用日期范围作为索引
            name="ts",  # 序列的名称为 "ts"
        )
        other = func(series)  # 对序列应用操作函数，得到另一个序列或常量
        compare_op(series, other, op)  # 比较原始序列和操作后的序列

    # 使用 pytest 的参数化装饰器，定义另一个测试函数，每个函数接受一个比较操作函数作为参数
    @pytest.mark.parametrize(
        "func", [lambda x: x + 1, lambda x: 5],  # 定义两个不同的操作函数
        ids=["add", "constant"]  # 设置每个函数的标识符
    )
    # 定义测试函数，测试序列的比较操作
    def test_series_operators_compare(self, comparison_op, func):
        op = comparison_op  # 获取比较操作函数
        series = Series(  # 创建一个包含浮点数的序列
            np.arange(10, dtype=np.float64),  # 生成从0到9的浮点数数组
            index=date_range("2020-01-01", periods=10),  # 使用日期范围作为索引
            name="ts",  # 序列的名称为 "ts"
        )
        other = func(series)  # 对序列应用操作函数，得到另一个序列或常量
        compare_op(series, other, op)  # 比较原始序列和操作后的序列

    # 使用 pytest 的参数化装饰器，定义测试函数，每个函数接受一个操作函数作为参数
    @pytest.mark.parametrize(
        "func",
        [lambda x: x * 2, lambda x: x[::2], lambda x: 5],  # 定义三个不同的操作函数
        ids=["multiply", "slice", "constant"],  # 设置每个函数的标识符
    )
    # 定义测试函数，测试序列的 divmod 操作
    def test_divmod(self, func):
        series = Series(  # 创建一个包含浮点数的序列
            np.arange(10, dtype=np.float64),  # 生成从0到9的浮点数数组
            index=date_range("2020-01-01", periods=10),  # 使用日期范围作为索引
            name="ts",  # 序列的名称为 "ts"
        )
        other = func(series)  # 对序列应用操作函数，得到另一个序列或常量
        results = divmod(series, other)  # 对序列和另一个序列进行 divmod 运算

        if isinstance(other, abc.Iterable) and len(series) != len(other):
            # 如果序列长度不匹配，这是使用 `tser[::2]` 的测试，将 `other_np` 中每隔一个值填充为 NaN
            other_np = []
            for n in other:
                other_np.append(n)
                other_np.append(np.nan)
        else:
            other_np = other  # 否则直接使用 other

        other_np = np.asarray(other_np)  # 将 other_np 转换为 NumPy 数组
        with np.errstate(all="ignore"):  # 忽略所有 NumPy 的错误状态
            expecteds = divmod(series.values, np.asarray(other_np))  # 计算期望的 divmod 结果

        for result, expected in zip(results, expecteds):
            # 分别检查值、名称和索引是否一致
            tm.assert_almost_equal(np.asarray(result), expected)
            assert result.name == series.name
            tm.assert_index_equal(result.index, series.index._with_freq(None))

    # 定义测试函数，测试序列除以零的 divmod 操作
    def test_series_divmod_zero(self):
        # 检查 divmod 对零的处理，与 NumPy 的处理方式不同
        # pandas 的约定是:
        #  1/0 == np.inf
        #  -1/0 == -np.inf
        #  1/-0.0 == -np.inf
        #  -1/-0.0 == np.inf
        tser = Series(  # 创建一个包含浮点数的序列
            np.arange(1, 11, dtype=np.float64),  # 生成从1到10的浮点数数组
            index=date_range("2020-01-01", periods=10),  # 使用日期范围作为索引
            name="ts",  # 序列的名称为 "ts"
        )
        other = tser * 0  # 创建一个全为零的序列

        result = divmod(tser, other)  # 对序列和零序列进行 divmod 运算
        exp1 = Series([np.inf] * len(tser), index=tser.index, name="ts")  # 期望的第一个结果是全为 np.inf
        exp2 = Series([np.nan] * len(tser), index=tser.index, name="ts")  # 期望的第二个结果是全为 np.nan
        tm.assert_series_equal(result[0], exp1)  # 检查第一个结果是否与期望一致
        tm.assert_series_equal(result[1], exp2)  # 检查第二个结果是否与期望一致
# 定义一个测试类 TestUFuncCompat，用于测试通用函数兼容性
class TestUFuncCompat:
    
    # TODO: add more dtypes
    # 使用 pytest 的参数化装饰器，分别测试 Index、RangeIndex 和 Series 作为 holder 对象
    @pytest.mark.parametrize("holder", [Index, RangeIndex, Series])
    # 使用 pytest 的参数化装饰器，测试 np.int64、np.uint64 和 np.float64 作为 dtype
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    # 定义测试方法 test_ufunc_compat，接受 holder 和 dtype 两个参数
    def test_ufunc_compat(self, holder, dtype):
        # 如果 holder 是 Series，则 box 是 Series，否则是 Index
        box = Series if holder is Series else Index
        
        # 如果 holder 是 RangeIndex
        if holder is RangeIndex:
            # 如果 dtype 不是 np.int64，则跳过当前测试并输出相应的跳过消息
            pytest.skip(f"dtype {dtype} not relevant for RangeIndex")
            # 创建一个名为 "foo" 的 RangeIndex 对象
            idx = RangeIndex(0, 5, name="foo")
        else:
            # 否则，创建一个根据 dtype 和值创建的 holder 对象，名为 "foo"
            idx = holder(np.arange(5, dtype=dtype), name="foo")
        
        # 对 idx 应用 np.sin 函数，得到结果 result
        result = np.sin(idx)
        # 使用 np.sin 函数直接作用于 np.arange(5, dtype=dtype)，并创建与 box 对象对应的期望值 expected
        expected = box(np.sin(np.arange(5, dtype=dtype)), name="foo")
        # 使用测试模块 tm 的 assert_equal 方法，断言 result 与 expected 相等
        tm.assert_equal(result, expected)

    # TODO: add more dtypes
    # 使用 pytest 的参数化装饰器，测试 np.int64、np.uint64 和 np.float64 作为 dtype
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    # 定义测试方法 test_ufunc_coercions，接受 index_or_series 和 dtype 两个参数
    def test_ufunc_coercions(self, index_or_series, dtype):
        # 根据给定值和 dtype 创建一个 index_or_series 对象，名为 "x"
        idx = index_or_series([1, 2, 3, 4, 5], dtype=dtype, name="x")
        # box 与 index_or_series 相同
        box = index_or_series
        
        # 对 idx 应用 np.sqrt 函数，得到结果 result
        result = np.sqrt(idx)
        # 断言 result 的 dtype 是 "f8" 并且 result 是 box 的实例
        assert result.dtype == "f8" and isinstance(result, box)
        # 创建预期值 exp，使用 np.sqrt 函数直接作用于 np.array([1, 2, 3, 4, 5], dtype=np.float64)，并与 box 对象对应
        exp = Index(np.sqrt(np.array([1, 2, 3, 4, 5], dtype=np.float64)), name="x")
        # 使用测试模块 tm 的 box_expected 方法，对 exp 进行处理以适应 box 对象
        exp = tm.box_expected(exp, box)
        # 使用测试模块 tm 的 assert_equal 方法，断言 result 与 exp 相等
        tm.assert_equal(result, exp)

        # 对 idx 应用 np.divide 函数，得到结果 result
        result = np.divide(idx, 2.0)
        # 断言 result 的 dtype 是 "f8" 并且 result 是 box 的实例
        assert result.dtype == "f8" and isinstance(result, box)
        # 创建预期值 exp，根据 np.divide 的操作结果和 dtype 创建 Index 对象，名为 "x"
        exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name="x")
        # 使用测试模块 tm 的 box_expected 方法，对 exp 进行处理以适应 box 对象
        exp = tm.box_expected(exp, box)
        # 使用测试模块 tm 的 assert_equal 方法，断言 result 与 exp 相等
        tm.assert_equal(result, exp)

        # 对 idx 应用加法操作，加 2.0，得到结果 result
        result = idx + 2.0
        # 断言 result 的 dtype 是 "f8" 并且 result 是 box 的实例
        assert result.dtype == "f8" and isinstance(result, box)
        # 创建预期值 exp，根据加法操作结果和 dtype 创建 Index 对象，名为 "x"
        exp = Index([3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64, name="x")
        # 使用测试模块 tm 的 box_expected 方法，对 exp 进行处理以适应 box 对象
        exp = tm.box_expected(exp, box)
        # 使用测试模块 tm 的 assert_equal 方法，断言 result 与 exp 相等
        tm.assert_equal(result, exp)

        # 对 idx 应用减法操作，减 2.0，得到结果 result
        result = idx - 2.0
        # 断言 result 的 dtype 是 "f8" 并且 result 是 box 的实例
        assert result.dtype == "f8" and isinstance(result, box)
        # 创建预期值 exp，根据减法操作结果和 dtype 创建 Index 对象，名为 "x"
        exp = Index([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float64, name="x")
        # 使用测试模块 tm 的 box_expected 方法，对 exp 进行处理以适应 box 对象
        exp = tm.box_expected(exp, box)
        # 使用测试模块 tm 的 assert_equal 方法，断言 result 与 exp 相等
        tm.assert_equal(result, exp)

        # 对 idx 应用乘法操作，乘 1.0，得到结果 result
        result = idx * 1.0
        # 断言 result 的 dtype 是 "f8" 并且 result 是 box 的实例
        assert result.dtype == "f8" and isinstance(result, box)
        # 创建预期值 exp，根据乘法操作结果和 dtype 创建 Index 对象，名为 "x"
        exp = Index([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64, name="x")
        # 使用测试模块 tm 的 box_expected 方法，对 exp 进行处理以适应 box 对象
        exp = tm.box_expected(exp, box)
        # 使用测试模块 tm 的 assert_equal 方法，断言 result 与 exp 相等
        tm.assert_equal(result, exp)

        # 对 idx 应用除法操作，除以 2.0，得到结果 result
        result = idx / 2.0
        # 断言 result 的 dtype 是 "f8" 并且 result 是 box 的实例
        assert result.dtype == "f8" and isinstance(result, box)
        # 创建预期值 exp，根据除法操作结果和 dtype 创建 Index 对象，名为 "x"
        exp = Index([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64, name="x")
        # 使用测试模块 tm 的 box_expected 方法，对 exp 进行处理以适应 box 对象
        exp = tm.box_expected(exp, box)
        # 使用测试模块 tm 的 assert_equal 方法，断言 result 与 exp 相等
        tm.assert_equal(result, exp)
    
    # TODO: add more dtypes
    # 使用 pytest 的参数化装饰器，分别测试 Index 和 Series 作为 holder 对象
    @pytest.mark.parametrize("holder", [Index, Series])
    # 使用 pytest 的参数化装饰器，测试 np.int64、np.uint64 和 np.float64 作为 dtype
    @pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float64])
    # 定义一个测试函数，用于测试 Universal Function 的多返回值特性
    def test_ufunc_multiple_return_values(self, holder, dtype):
        # 创建一个对象实例，可能是 Series 或者 Index
        obj = holder([1, 2, 3], dtype=dtype, name="x")
        # 根据 holder 类型确定要使用的盒子类型（Series 或 Index）
        box = Series if holder is Series else Index

        # 调用 numpy 的 modf 函数，返回结果是一个元组
        result = np.modf(obj)
        # 断言返回结果是一个元组
        assert isinstance(result, tuple)
        
        # 准备预期的第一个返回值（整数部分）
        exp1 = Index([0.0, 0.0, 0.0], dtype=np.float64, name="x")
        # 准备预期的第二个返回值（小数部分）
        exp2 = Index([1.0, 2.0, 3.0], dtype=np.float64, name="x")
        
        # 使用测试工具函数验证第一个返回值与预期结果的盒子化
        tm.assert_equal(result[0], tm.box_expected(exp1, box))
        # 使用测试工具函数验证第二个返回值与预期结果的盒子化
        tm.assert_equal(result[1], tm.box_expected(exp2, box))

    # 定义一个测试函数，用于测试 numpy 的 ufunc 的 at 方法
    def test_ufunc_at(self):
        # 创建一个 Series 对象
        s = Series([0, 1, 2], index=[1, 2, 3], name="x")
        # 对 Series 执行 numpy 的 add.at 方法，在指定的索引位置增加值
        np.add.at(s, [0, 2], 10)
        # 准备预期的结果 Series
        expected = Series([10, 1, 12], index=[1, 2, 3], name="x")
        # 使用测试工具函数验证 Series 是否与预期结果相等
        tm.assert_series_equal(s, expected)
# 定义一个测试类，用于测试对象数据类型的等价性
class TestObjectDtypeEquivalence:

    # 使用参数化装饰器定义测试用例，测试带有 NaN 的加法操作
    @pytest.mark.parametrize("dtype", [None, object])
    def test_numarr_with_dtype_add_nan(self, dtype, box_with_array):
        # 获取包含数组的盒子对象
        box = box_with_array
        # 创建包含整数的 Series 对象，指定数据类型为 dtype
        ser = Series([1, 2, 3], dtype=dtype)
        # 创建预期结果的 Series 对象，其中包含 NaN 值，数据类型为 dtype
        expected = Series([np.nan, np.nan, np.nan], dtype=dtype)

        # 对 ser 应用包装函数，以适应盒子对象
        ser = tm.box_expected(ser, box)
        # 对 expected 应用包装函数，以适应盒子对象
        expected = tm.box_expected(expected, box)

        # 执行 np.nan + ser 操作
        result = np.nan + ser
        # 使用测试工具函数验证结果与预期是否相等
        tm.assert_equal(result, expected)

        # 执行 ser + np.nan 操作
        result = ser + np.nan
        # 使用测试工具函数验证结果与预期是否相等
        tm.assert_equal(result, expected)

    # 使用参数化装饰器定义测试用例，测试带有整数的加法操作
    @pytest.mark.parametrize("dtype", [None, object])
    def test_numarr_with_dtype_add_int(self, dtype, box_with_array):
        # 获取包含数组的盒子对象
        box = box_with_array
        # 创建包含整数的 Series 对象，指定数据类型为 dtype
        ser = Series([1, 2, 3], dtype=dtype)
        # 创建预期结果的 Series 对象，进行整数加法操作后的结果，数据类型为 dtype
        expected = Series([2, 3, 4], dtype=dtype)

        # 对 ser 应用包装函数，以适应盒子对象
        ser = tm.box_expected(ser, box)
        # 对 expected 应用包装函数，以适应盒子对象
        expected = tm.box_expected(expected, box)

        # 执行 1 + ser 操作
        result = 1 + ser
        # 使用测试工具函数验证结果与预期是否相等
        tm.assert_equal(result, expected)

        # 执行 ser + 1 操作
        result = ser + 1
        # 使用测试工具函数验证结果与预期是否相等
        tm.assert_equal(result, expected)

    # TODO: moved from tests.series.test_operators; needs cleanup
    # 使用参数化装饰器定义测试用例，测试对象类型下的逆操作
    @pytest.mark.parametrize(
        "op",
        [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv],
    )
    def test_operators_reverse_object(self, op):
        # 创建包含随机标准正态分布数据的 Series 对象，数据类型为 object
        arr = Series(
            np.random.default_rng(2).standard_normal(10),
            index=np.arange(10),
            dtype=object,
        )

        # 执行 op(1.0, arr) 操作
        result = op(1.0, arr)
        # 执行 op(1.0, arr.astype(float)) 操作作为预期结果
        expected = op(1.0, arr.astype(float))
        # 使用测试工具函数验证结果与预期是否相等，并将结果转换为 float 类型
        tm.assert_series_equal(result.astype(float), expected)


class TestNumericArithmeticUnsorted:
    # 该类中的测试用例已从特定类型的测试模块移动，但尚未排序、参数化和去重

    # 使用参数化装饰器定义测试用例，测试二元操作对索引的影响
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
        ],
    )
    @pytest.mark.parametrize(
        "idx1",
        [
            RangeIndex(0, 10, 1),
            RangeIndex(0, 20, 2),
            RangeIndex(-10, 10, 2),
            RangeIndex(5, -5, -1),
        ],
    )
    @pytest.mark.parametrize(
        "idx2",
        [
            RangeIndex(0, 10, 1),
            RangeIndex(0, 20, 2),
            RangeIndex(-10, 10, 2),
            RangeIndex(5, -5, -1),
        ],
    )
    def test_binops_index(self, op, idx1, idx2):
        # 重命名 idx1 为 "foo"，返回新的索引对象
        idx1 = idx1._rename("foo")
        # 重命名 idx2 为 "bar"，返回新的索引对象
        idx2 = idx2._rename("bar")
        # 执行 op(idx1, idx2) 操作
        result = op(idx1, idx2)
        # 执行 op(Index(idx1.to_numpy()), Index(idx2.to_numpy())) 操作作为预期结果
        expected = op(Index(idx1.to_numpy()), Index(idx2.to_numpy()))
        # 使用测试工具函数验证结果与预期是否相等，要求精确匹配
        tm.assert_index_equal(result, expected, exact="equiv")

    # 使用参数化装饰器定义测试用例，测试数字算术操作的未排序情况
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,
            operator.sub,
            operator.mul,
            operator.floordiv,
            operator.truediv,
        ],
    )
    # 使用 pytest 的 @parametrize 装饰器，为测试函数 test_binops_index_scalar 参数化不同的 RangeIndex 对象
    @pytest.mark.parametrize(
        "idx",
        [
            RangeIndex(0, 10, 1),    # 创建一个从 0 到 9 的步长为 1 的 RangeIndex 对象
            RangeIndex(0, 20, 2),    # 创建一个从 0 到 19 的步长为 2 的 RangeIndex 对象
            RangeIndex(-10, 10, 2),  # 创建一个从 -10 到 9 的步长为 2 的 RangeIndex 对象
            RangeIndex(5, -5, -1),   # 创建一个从 5 到 -4 的步长为 -1 的 RangeIndex 对象
        ],
    )
    # 为测试函数 test_binops_index_scalar 参数化不同的标量值
    @pytest.mark.parametrize("scalar", [-1, 1, 2])
    def test_binops_index_scalar(self, op, idx, scalar):
        # 使用操作符 op 对 idx 和 scalar 进行操作
        result = op(idx, scalar)
        # 将 idx 转换为 numpy 数组后再进行操作，并与标量 scalar 进行操作
        expected = op(Index(idx.to_numpy()), scalar)
        # 断言两个操作结果的索引是否相等，使用 exact="equiv" 表示精确相等
        tm.assert_index_equal(result, expected, exact="equiv")

    # 使用 pytest 的 @parametrize 装饰器，为测试函数 test_binops_index_pow 参数化不同的 RangeIndex 对象
    @pytest.mark.parametrize("idx1", [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    # 使用 pytest 的 @parametrize 装饰器，为测试函数 test_binops_index_pow 参数化不同的 RangeIndex 对象
    @pytest.mark.parametrize("idx2", [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    def test_binops_index_pow(self, idx1, idx2):
        # numpy 不允许对负整数进行乘方操作，因此需要单独测试
        # https://github.com/numpy/numpy/pull/8127
        # 将 idx1 和 idx2 重命名为 "foo" 和 "bar"
        idx1 = idx1._rename("foo")
        idx2 = idx2._rename("bar")
        # 使用 pow 函数对 idx1 和 idx2 进行乘方操作
        result = pow(idx1, idx2)
        # 将 idx1 和 idx2 转换为 numpy 数组后再进行乘方操作
        expected = pow(Index(idx1.to_numpy()), Index(idx2.to_numpy()))
        # 断言两个操作结果的索引是否相等，使用 exact="equiv" 表示精确相等
        tm.assert_index_equal(result, expected, exact="equiv")

    # 使用 pytest 的 @parametrize 装饰器，为测试函数 test_binops_index_scalar_pow 参数化不同的 RangeIndex 对象
    @pytest.mark.parametrize("idx", [RangeIndex(0, 10, 1), RangeIndex(0, 20, 2)])
    # 使用 pytest 的 @parametrize 装饰器，为测试函数 test_binops_index_scalar_pow 参数化不同的标量值
    @pytest.mark.parametrize("scalar", [1, 2])
    def test_binops_index_scalar_pow(self, idx, scalar):
        # numpy 不允许对负整数进行乘方操作，因此需要单独测试
        # https://github.com/numpy/numpy/pull/8127
        # 使用 pow 函数对 idx 和 scalar 进行乘方操作
        result = pow(idx, scalar)
        # 将 idx 转换为 numpy 数组后再与标量 scalar 进行乘方操作
        expected = pow(Index(idx.to_numpy()), scalar)
        # 断言两个操作结果的索引是否相等，使用 exact="equiv" 表示精确相等
        tm.assert_index_equal(result, expected, exact="equiv")

    # TODO: divmod?
    # 使用 pytest 的 @parametrize 装饰器，为测试函数 test_arithmetic_with_frame_or_series 参数化不同的操作符
    @pytest.mark.parametrize(
        "op",
        [
            operator.add,           # 加法操作
            operator.sub,           # 减法操作
            operator.mul,           # 乘法操作
            operator.floordiv,      # 整除操作
            operator.truediv,       # 真除操作
            operator.pow,           # 乘方操作
            operator.mod,           # 模运算操作
        ],
    )
    def test_arithmetic_with_frame_or_series(self, op):
        # 检查当操作对象为 Series 或 DataFrame 时返回 NotImplemented
        index = RangeIndex(5)
        other = Series(np.random.default_rng(2).standard_normal(5))

        # 使用操作符 op 对 Series(index) 和 other 进行操作，期望抛出 NotImplemented
        expected = op(Series(index), other)
        # 使用操作符 op 对 index 和 other 进行操作，期望抛出 NotImplemented
        result = op(index, other)
        # 断言两个操作结果的 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 将 other 赋值为一个形状为 (2, 5) 的随机标准正态分布的 DataFrame
        other = pd.DataFrame(np.random.default_rng(2).standard_normal((2, 5)))
        # 使用操作符 op 对 pd.DataFrame([index, index]) 和 other 进行操作，期望抛出 NotImplemented
        expected = op(pd.DataFrame([index, index]), other)
        # 使用操作符 op 对 index 和 other 进行操作，期望抛出 NotImplemented
        result = op(index, other)
        # 断言两个操作结果的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于验证对 RangeIndex 的数值操作覆盖处理，确保在可能的情况下返回 RangeIndex

    # 创建一个步长为2的 RangeIndex，范围从0到10
    idx = RangeIndex(0, 10, 2)

    # 测试乘法操作：RangeIndex 乘以2，期望得到步长为4的 RangeIndex
    result = idx * 2
    expected = RangeIndex(0, 20, 4)
    tm.assert_index_equal(result, expected, exact=True)

    # 测试加法操作：RangeIndex 加上2，期望得到从2到12步长为2的 RangeIndex
    result = idx + 2
    expected = RangeIndex(2, 12, 2)
    tm.assert_index_equal(result, expected, exact=True)

    # 测试减法操作：RangeIndex 减去2，期望得到从-2到8步长为2的 RangeIndex
    result = idx - 2
    expected = RangeIndex(-2, 8, 2)
    tm.assert_index_equal(result, expected, exact=True)

    # 测试除法操作：RangeIndex 除以2，期望得到从0到5步长为1的浮点数 Index
    result = idx / 2
    expected = RangeIndex(0, 5, 1).astype("float64")
    tm.assert_index_equal(result, expected, exact=True)

    # 测试除法操作：RangeIndex 除以4，期望得到步长为2的 RangeIndex
    result = idx / 4
    expected = RangeIndex(0, 10, 2) / 4
    tm.assert_index_equal(result, expected, exact=True)

    # 测试地板除法操作：RangeIndex 地板除以1，期望得到原始的 RangeIndex
    result = idx // 1
    expected = idx
    tm.assert_index_equal(result, expected, exact=True)

    # __mul__ 方法的测试：RangeIndex 乘以自身，期望得到元素值平方的 Index
    result = idx * idx
    expected = Index(idx.values * idx.values)
    tm.assert_index_equal(result, expected, exact=True)

    # __pow__ 方法的测试：创建步长为2的 RangeIndex，对其进行平方操作，期望得到元素值平方的 Index
    idx = RangeIndex(0, 1000, 2)
    result = idx**2
    expected = Index(idx._values) ** 2
    tm.assert_index_equal(Index(result.values), expected, exact=True)

@pytest.mark.parametrize(
    "idx, div, expected",
    [
        # 参数化测试，验证不同的 RangeIndex 对象在除法操作后的预期结果
        (RangeIndex(0, 1000, 2), 2, RangeIndex(0, 500, 1)),
        (RangeIndex(-99, -201, -3), -3, RangeIndex(33, 67, 1)),
        (
            RangeIndex(0, 1000, 1),
            2,
            Index(RangeIndex(0, 1000, 1)._values) // 2,
        ),
        (
            RangeIndex(0, 100, 1),
            2.0,
            Index(RangeIndex(0, 100, 1)._values) // 2.0,
        ),
        (RangeIndex(0), 50, RangeIndex(0)),
        (RangeIndex(2, 4, 2), 3, RangeIndex(0, 1, 1)),
        (RangeIndex(-5, -10, -6), 4, RangeIndex(-2, -1, 1)),
        (RangeIndex(-100, -200, 3), 2, RangeIndex(0)),
    ],
)
def test_numeric_compat2_floordiv(self, idx, div, expected):
    # __floordiv__ 方法的测试：对 RangeIndex 执行整除操作，验证预期结果
    tm.assert_index_equal(idx // div, expected, exact=True)

@pytest.mark.parametrize("dtype", [np.int64, np.float64])
@pytest.mark.parametrize("delta", [1, 0, -1])
def test_addsub_arithmetic(self, dtype, delta):
    # test_addsub_arithmetic 方法的测试：验证 Index 对象的加减算术操作

    # 将 delta 转换为指定的数据类型
    delta = dtype(delta)

    # 创建一个整数或浮点数类型的 Index，初始值为 [10, 11, 12]
    index = Index([10, 11, 12], dtype=dtype)

    # 测试加法操作：Index 加上 delta，期望得到对应元素值加上 delta 后的 Index
    result = index + delta
    expected = Index(index.values + delta, dtype=dtype)
    tm.assert_index_equal(result, expected)

    # 测试减法操作：Index 减去 delta，期望得到对应元素值减去 delta 后的 Index
    result = index - delta
    expected = Index(index.values - delta, dtype=dtype)
    tm.assert_index_equal(result, expected)

    # 测试加法操作：Index 与自身相加，期望得到元素值加倍后的 Index
    tm.assert_index_equal(index + index, 2 * index)

    # 测试减法操作：Index 减去自身，期望得到元素值全为零的 Index
    tm.assert_index_equal(index - index, 0 * index)

    # 验证索引减去自身不为空
    assert not (index - index).empty
    # 定义一个测试方法，用于测试幂运算中处理 NaN 和零值的情况
    def test_pow_nan_with_zero(self, box_with_array):
        # 创建包含 NaN 值的索引对象 left
        left = Index([np.nan, np.nan, np.nan])
        # 创建包含零值的索引对象 right
        right = Index([0, 0, 0])
        # 创建期望的索引对象 expected，其中包含了 1.0 作为每个元素的期望值
        expected = Index([1.0, 1.0, 1.0])

        # 使用 box_expected 函数将 left 对象与 box_with_array 进行盒装处理
        left = tm.box_expected(left, box_with_array)
        # 使用 box_expected 函数将 right 对象与 box_with_array 进行盒装处理
        right = tm.box_expected(right, box_with_array)
        # 使用 box_expected 函数将 expected 对象与 box_with_array 进行盒装处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算 left 对象的 right 次幂，并将结果保存在 result 中
        result = left**right
        # 断言 result 与 expected 相等
        tm.assert_equal(result, expected)
def test_fill_value_inf_masking():
    # GH #27464 make sure we mask 0/1 with Inf and not NaN

    # 创建一个包含两列的DataFrame，列"A"包含0、1、2，列"B"包含1.1、None、1.1
    df = pd.DataFrame({"A": [0, 1, 2], "B": [1.1, None, 1.1]})

    # 创建另一个DataFrame，包含列"A"，索引为[0, 2, 3]，值为[1.1, 1.2, 1.3]
    other = pd.DataFrame({"A": [1.1, 1.2, 1.3]}, index=[0, 2, 3])

    # 对df进行右除操作，使用other作为除数，fill_value参数设置为1
    result = df.rfloordiv(other, fill_value=1)

    # 创建期望的DataFrame，列"A"为[inf, 1.0, 0.0, 1.0]，列"B"为[0.0, NaN, 0.0, NaN]
    expected = pd.DataFrame(
        {"A": [np.inf, 1.0, 0.0, 1.0], "B": [0.0, np.nan, 0.0, np.nan]}
    )

    # 使用tm.assert_frame_equal断言结果和期望一致
    tm.assert_frame_equal(result, expected)


def test_dataframe_div_silenced():
    # GH#26793

    # 创建一个DataFrame，包含列"A"为0到9，列"B"为[NaN, 1, 2, 3, 4]重复两次，列"C"全部为NaN，列"D"为0到9
    pdf1 = pd.DataFrame(
        {
            "A": np.arange(10),
            "B": [np.nan, 1, 2, 3, 4] * 2,
            "C": [np.nan] * 10,
            "D": np.arange(10),
        },
        index=list("abcdefghij"),
        columns=list("ABCD"),
    )

    # 创建另一个DataFrame，随机生成符合正态分布的值，形状为(10, 4)，索引为['a'到'j'，'k']
    pdf2 = pd.DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        index=list("abcdefghjk"),
        columns=list("ABCX"),
    )

    # 使用tm.assert_produces_warning(None)上下文确保DataFrame的除法操作不会引发警告
    with tm.assert_produces_warning(None):
        pdf1.div(pdf2, fill_value=0)


@pytest.mark.parametrize(
    "data, expected_data",
    [([0, 1, 2], [0, 2, 4])],
)
@pytest.mark.parametrize("box_pandas_1d_array", [Index, Series, tm.to_array])
@pytest.mark.parametrize("box_1d_array", [Index, Series, tm.to_array, np.array, list])
def test_integer_array_add_list_like(
    box_pandas_1d_array, box_1d_array, data, expected_data
):
    # GH22606 Verify operators with IntegerArray and list-likes

    # 创建一个Int64类型的数组，内容为data参数
    arr = array(data, dtype="Int64")
    
    # 使用box_pandas_1d_array函数将数组arr装箱为容器
    container = box_pandas_1d_array(arr)

    # 执行容器和box_1d_array(data)之间的加法操作，得到左操作数
    left = container + box_1d_array(data)

    # 执行box_1d_array(data)和容器之间的加法操作，得到右操作数
    right = box_1d_array(data) + container

    # 根据box_1d_array和box_pandas_1d_array类型确定期望的结果类型
    if Series in [box_1d_array, box_pandas_1d_array]:
        cls = Series
    elif Index in [box_1d_array, box_pandas_1d_array]:
        cls = Index
    else:
        cls = array

    # 创建期望的结果，数据为expected_data，类型为Int64
    expected = cls(expected_data, dtype="Int64")

    # 使用tm.assert_equal断言左操作数和右操作数与期望结果一致
    tm.assert_equal(left, expected)
    tm.assert_equal(right, expected)


def test_sub_multiindex_swapped_levels():
    # GH 9952

    # 创建一个DataFrame，包含列"a"，其值为符合正态分布的随机数，索引为MultiIndex，层次为["levA", "levB"]
    df = pd.DataFrame(
        {"a": np.random.default_rng(2).standard_normal(6)},
        index=pd.MultiIndex.from_product(
            [["a", "b"], [0, 1, 2]], names=["levA", "levB"]
        ),
    )

    # 复制df创建df2
    df2 = df.copy()

    # 交换df2的MultiIndex的第0层和第1层
    df2.index = df2.index.swaplevel(0, 1)

    # 对df和df2执行减法操作
    result = df - df2

    # 创建期望的DataFrame，列"a"全部为0.0，索引与df相同
    expected = pd.DataFrame([0.0] * 6, columns=["a"], index=df.index)

    # 使用tm.assert_frame_equal断言结果和期望一致
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("power", [1, 2, 5])
@pytest.mark.parametrize("string_size", [0, 1, 2, 5])
def test_empty_str_comparison(power, string_size):
    # GH 37348

    # 创建一个包含10**power个整数的数组a，转换为dtype为int64的DataFrame，命名为right
    a = np.array(range(10**power))
    right = pd.DataFrame(a, dtype=np.int64)

    # 创建一个长度为string_size的空字符串left
    left = " " * string_size

    # 执行right与left之间的比较操作，生成布尔值的DataFrame
    result = right == left

    # 创建期望的DataFrame，形状与result相同，全部值为False
    expected = pd.DataFrame(np.zeros(right.shape, dtype=bool))

    # 使用tm.assert_frame_equal断言结果和期望一致
    tm.assert_frame_equal(result, expected)


def test_series_add_sub_with_UInt64():
    # GH 22023

    # 创建一个Series，数据为[1, 2, 3]
    series1 = Series([1, 2, 3])

    # 创建一个Series，数据为[2, 1, 3]，类型为UInt64
    series2 = Series([2, 1, 3], dtype="UInt64")

    # 执行series1和series2之间的加法操作
    result = series1 + series2

    # 创建期望的Series，数据为[3, 3, 6]，类型为Float64
    expected = Series([3, 3, 6], dtype="Float64")

    # 使用tm.assert_series_equal断言结果和期望一致
    tm.assert_series_equal(result, expected)
    # 计算两个 Series 的差集
    result = series1 - series2
    # 创建预期结果 Series，指定数据类型为浮点数
    expected = Series([-1, 1, 0], dtype="Float64")
    # 使用测试工具函数检查两个 Series 是否相等
    tm.assert_series_equal(result, expected)
```
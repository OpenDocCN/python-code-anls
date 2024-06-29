# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\test_period.py`

```
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for Period dtype
import operator

import numpy as np
import pytest

from pandas._libs.tslibs import (
    IncompatibleFrequency,
    Period,
    Timestamp,
    to_offset,
)

import pandas as pd
from pandas import (
    PeriodIndex,
    Series,
    Timedelta,
    TimedeltaIndex,
    period_range,
)
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)

_common_mismatch = [
    pd.offsets.YearBegin(2),
    pd.offsets.MonthBegin(1),
    pd.offsets.Minute(),
]


@pytest.fixture(
    params=[
        Timedelta(minutes=30).to_pytimedelta(),
        np.timedelta64(30, "s"),
        Timedelta(seconds=30),
    ]
    + _common_mismatch
)
def not_hourly(request):
    """
    Several timedelta-like and DateOffset instances that are _not_
    compatible with Hourly frequencies.
    """
    return request.param


@pytest.fixture(
    params=[
        np.timedelta64(365, "D"),
        Timedelta(days=365).to_pytimedelta(),
        Timedelta(days=365),
    ]
    + _common_mismatch
)
def mismatched_freq(request):
    """
    Several timedelta-like and DateOffset instances that are _not_
    compatible with Monthly or Annual frequencies.
    """
    return request.param


# ------------------------------------------------------------------
# Comparisons


class TestPeriodArrayLikeComparisons:
    # Comparison tests for PeriodDtype vectors fully parametrized over
    #  DataFrame/Series/PeriodIndex/PeriodArray.  Ideally all comparison
    #  tests will eventually end up here.

    @pytest.mark.parametrize("other", ["2017", Period("2017", freq="D")])
    def test_eq_scalar(self, other, box_with_array):
        # Create a PeriodIndex with daily frequency and box it as expected
        idx = PeriodIndex(["2017", "2017", "2018"], freq="D")
        idx = tm.box_expected(idx, box_with_array)
        # Get the upcasted box with 'other' and compare equality
        xbox = get_upcast_box(idx, other, True)

        # Expected results for equality comparison
        expected = np.array([True, True, False])
        expected = tm.box_expected(expected, xbox)

        # Perform the equality comparison
        result = idx == other

        # Assert that the result matches the expected values
        tm.assert_equal(result, expected)

    def test_compare_zerodim(self, box_with_array):
        # GH#26689 make sure we unbox zero-dimensional arrays

        # Create a PeriodIndex and convert it to numpy array, then box as expected
        pi = period_range("2000", periods=4)
        other = np.array(pi.to_numpy()[0])

        pi = tm.box_expected(pi, box_with_array)
        # Get the upcasted box with 'other' and perform comparison
        xbox = get_upcast_box(pi, other, True)

        # Perform the comparison operation (<=) and box expected results
        result = pi <= other
        expected = np.array([True, False, False, False])
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)
    # 使用 pytest 的 @pytest.mark.parametrize 装饰器，为单元测试参数化提供支持
    # 参数化：scalar 参数被设置为一组不同类型的值进行测试
    @pytest.mark.parametrize(
        "scalar",
        [
            "foo",                      # 字符串类型
            Timestamp("2021-01-01"),   # 时间戳类型
            Timedelta(days=4),          # 时间增量类型
            9,                          # 整数类型
            9.5,                        # 浮点数类型
            2000,                       # 整数，特别是不作为 Period("2000", "D") 的匹配项
            False,                      # 布尔值类型
            None,                       # 空值类型
        ],
    )
    # 定义测试函数 test_compare_invalid_scalar，用于测试与无法解释为 Period 的标量进行比较的情况
    def test_compare_invalid_scalar(self, box_with_array, scalar):
        # 创建一个时间周期范围对象，从 "2000" 开始，包含 4 个周期
        pi = period_range("2000", periods=4)
        # 使用测试辅助函数 tm.box_expected，将时间周期范围和数组盒子进行关联
        parr = tm.box_expected(pi, box_with_array)
        # 断言无效比较函数，检查 parr 与 scalar 的比较行为是否符合预期
        assert_invalid_comparison(parr, scalar, box_with_array)
    
    # 参数化：other 参数被设置为一组不同类型的值进行测试
    @pytest.mark.parametrize(
        "other",
        [
            pd.date_range("2000", periods=4).array,         # 日期范围的数组
            pd.timedelta_range("1D", periods=4).array,     # 时间增量范围的数组
            np.arange(4),                                   # NumPy 数组
            np.arange(4).astype(np.float64),                # NumPy 浮点数数组
            list(range(4)),                                 # Python 列表
            [2000, 2001, 2002, 2003],                        # 不作为 Period 的匹配项
            np.arange(2000, 2004),                          # NumPy 范围数组
            np.arange(2000, 2004).astype(object),           # NumPy 对象数组
            pd.Index([2000, 2001, 2002, 2003]),              # Pandas 索引对象
        ],
    )
    # 定义测试函数 test_compare_invalid_listlike，用于测试与无效的列表样式数据进行比较的情况
    def test_compare_invalid_listlike(self, box_with_array, other):
        # 创建一个时间周期范围对象，从 "2000" 开始，包含 4 个周期
        pi = period_range("2000", periods=4)
        # 使用测试辅助函数 tm.box_expected，将时间周期范围和数组盒子进行关联
        parr = tm.box_expected(pi, box_with_array)
        # 断言无效比较函数，检查 parr 与 other 的比较行为是否符合预期
        assert_invalid_comparison(parr, other, box_with_array)
    # 测试比较对象的数据类型
    def test_compare_object_dtype(self, box_with_array, other_box):
        # 创建一个时间周期范围，从"2000"开始，包含5个时间点
        pi = period_range("2000", periods=5)
        # 使用给定的数据盒子和时间周期范围创建一个 Pandas Series
        parr = tm.box_expected(pi, box_with_array)

        # 使用其他盒子和时间周期范围创建另一个对象
        other = other_box(pi)
        # 获取升级后的数据盒子
        xbox = get_upcast_box(parr, other, True)

        # 期望结果为全为True的NumPy数组
        expected = np.array([True, True, True, True, True])
        # 将期望结果与升级后的数据盒子进行比较
        expected = tm.box_expected(expected, xbox)

        # 检查是否 parr 等于 other
        result = parr == other
        tm.assert_equal(result, expected)
        # 检查是否 parr 小于等于 other
        result = parr <= other
        tm.assert_equal(result, expected)
        # 检查是否 parr 大于等于 other
        result = parr >= other
        tm.assert_equal(result, expected)

        # 检查是否 parr 不等于 other
        result = parr != other
        tm.assert_equal(result, ~expected)
        # 检查是否 parr 小于 other
        result = parr < other
        tm.assert_equal(result, ~expected)
        # 检查是否 parr 大于 other
        result = parr > other
        tm.assert_equal(result, ~expected)

        # 使用反转后的时间周期范围创建另一个对象
        other = other_box(pi[::-1])

        # 期望结果为指定布尔数组
        expected = np.array([False, False, True, False, False])
        expected = tm.box_expected(expected, xbox)
        result = parr == other
        tm.assert_equal(result, expected)

        expected = np.array([True, True, True, False, False])
        expected = tm.box_expected(expected, xbox)
        result = parr <= other
        tm.assert_equal(result, expected)

        expected = np.array([False, False, True, True, True])
        expected = tm.box_expected(expected, xbox)
        result = parr >= other
        tm.assert_equal(result, expected)

        expected = np.array([True, True, False, True, True])
        expected = tm.box_expected(expected, xbox)
        result = parr != other
        tm.assert_equal(result, expected)

        expected = np.array([True, True, False, False, False])
        expected = tm.box_expected(expected, xbox)
        result = parr < other
        tm.assert_equal(result, expected)

        expected = np.array([False, False, False, True, True])
        expected = tm.box_expected(expected, xbox)
        result = parr > other
        tm.assert_equal(result, expected)
# 定义一个测试类 TestPeriodIndexComparisons，用于测试 PeriodIndex 的比较操作
class TestPeriodIndexComparisons:
    
    # TODO: parameterize over boxes
    # TODO: 根据箱子参数化

    # 定义一个测试方法 test_pi_cmp_period，用于测试 PeriodIndex 对象的比较操作
    def test_pi_cmp_period(self):
        # 创建一个包含 20 个月份的 PeriodIndex 对象，频率为每月一次，从 "2007-01" 开始
        idx = period_range("2007-01", periods=20, freq="M")
        # 选择该 PeriodIndex 中的第 10 个 Period 对象
        per = idx[10]

        # 比较 idx 中每个 Period 对象是否小于 per，返回一个布尔数组
        result = idx < per
        # 期望结果是 idx.values 中的每个值与 idx.values[10] 的比较结果
        exp = idx.values < idx.values[10]
        # 使用测试框架验证两个数组是否相等
        tm.assert_numpy_array_equal(result, exp)

        # 将 idx.values 重塑为一个 10x2 的数组，再与 per 进行比较
        result = idx.values.reshape(10, 2) < per
        exp = exp.reshape(10, 2)
        tm.assert_numpy_array_equal(result, exp)

        # 将 idx 转换为 ndarray 后与 per 比较
        result = idx < np.array(per)
        tm.assert_numpy_array_equal(result, exp)

    # TODO: moved from test_datetime64; de-duplicate with version below
    # TODO: 从 test_datetime64 移动过来；与下面的版本去重

    # 定义一个测试方法 test_parr_cmp_period_scalar2，用于测试 PeriodIndex 对象与标量值的比较操作
    def test_parr_cmp_period_scalar2(self, box_with_array):
        # 创建一个包含 10 天的 PeriodIndex 对象，从 "2000-01-01" 开始，频率为每天一次
        pi = period_range("2000-01-01", periods=10, freq="D")

        # 选择 pi 中的第 3 个 Period 对象
        val = pi[3]
        # 创建一个期望的布尔数组，表示 pi 中每个 Period 对象是否大于 val
        expected = [x > val for x in pi]

        # 使用测试框架将 pi 包装后的期望值与 box_with_array 进行比较
        ser = tm.box_expected(pi, box_with_array)
        xbox = get_upcast_box(ser, val, True)

        # 使用测试框架验证 ser > val 的结果是否与 expected 相等
        expected = tm.box_expected(expected, xbox)
        result = ser > val
        tm.assert_equal(result, expected)

        # 重新设置 val，并再次比较 ser > val 的结果与 expected
        val = pi[5]
        result = ser > val
        expected = [x > val for x in pi]
        expected = tm.box_expected(expected, xbox)
        tm.assert_equal(result, expected)

    # 定义一个参数化测试方法 test_parr_cmp_period_scalar，用于测试 PeriodIndex 对象与标量值的比较操作
    @pytest.mark.parametrize("freq", ["M", "2M", "3M"])
    def test_parr_cmp_period_scalar(self, freq, box_with_array):
        # 创建一个包含四个 Period 对象的 PeriodIndex 对象，指定频率为 freq
        base = PeriodIndex(["2011-01", "2011-02", "2011-03", "2011-04"], freq=freq)
        # 使用测试框架将 base 包装后的期望值与 box_with_array 进行比较
        base = tm.box_expected(base, box_with_array)
        # 创建一个 Period 对象
        per = Period("2011-02", freq=freq)
        # 使用 get_upcast_box 函数获取 base 和 per 的最小公共类型
        xbox = get_upcast_box(base, per, True)

        # 创建期望的布尔数组，表示 base 与 per 之间的不同比较操作的结果
        exp = np.array([False, True, False, False])
        exp = tm.box_expected(exp, xbox)
        # 使用测试框架验证 base == per 的结果是否与 exp 相等
        tm.assert_equal(base == per, exp)
        tm.assert_equal(per == base, exp)

        exp = np.array([True, False, True, True])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base != per, exp)
        tm.assert_equal(per != base, exp)

        exp = np.array([False, False, True, True])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base > per, exp)
        tm.assert_equal(per < base, exp)

        exp = np.array([True, False, False, False])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base < per, exp)
        tm.assert_equal(per > base, exp)

        exp = np.array([False, True, True, True])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base >= per, exp)
        tm.assert_equal(per <= base, exp)

        exp = np.array([True, True, False, False])
        exp = tm.box_expected(exp, xbox)
        tm.assert_equal(base <= per, exp)
        tm.assert_equal(per >= base, exp)

    # 定义一个参数化测试方法 test_parr_cmp_period_scalar，用于测试 PeriodIndex 对象与标量值的比较操作
    @pytest.mark.parametrize("freq", ["M", "2M", "3M"])
    # 定义用于测试 PeriodIndex 比较的方法，参数包括频率和包含数组的盒子对象
    def test_parr_cmp_pi(self, freq, box_with_array):
        # GH#13200
        # 创建基准 PeriodIndex 对象，包含指定日期和频率
        base = PeriodIndex(["2011-01", "2011-02", "2011-03", "2011-04"], freq=freq)
        # 使用测试工具方法对基准对象进行包装
        base = tm.box_expected(base, box_with_array)

        # TODO: could also box idx?
        # 创建另一个 PeriodIndex 对象，包含不同的日期顺序和相同的频率
        idx = PeriodIndex(["2011-02", "2011-01", "2011-03", "2011-05"], freq=freq)

        # 获取基准和索引对象的比较结果，以升级后的对象形式返回
        xbox = get_upcast_box(base, idx, True)

        # 期望的比较结果，包含基准与索引相等的布尔数组
        exp = np.array([False, False, True, False])
        # 使用测试工具方法对期望结果进行包装
        exp = tm.box_expected(exp, xbox)
        # 断言基准与索引相等的结果与期望结果一致
        tm.assert_equal(base == idx, exp)

        # 期望的比较结果，包含基准与索引不等的布尔数组
        exp = np.array([True, True, False, True])
        # 使用测试工具方法对期望结果进行包装
        exp = tm.box_expected(exp, xbox)
        # 断言基准与索引不等的结果与期望结果一致
        tm.assert_equal(base != idx, exp)

        # 期望的比较结果，包含基准大于索引的布尔数组
        exp = np.array([False, True, False, False])
        # 使用测试工具方法对期望结果进行包装
        exp = tm.box_expected(exp, xbox)
        # 断言基准大于索引的结果与期望结果一致
        tm.assert_equal(base > idx, exp)

        # 期望的比较结果，包含基准小于索引的布尔数组
        exp = np.array([True, False, False, True])
        # 使用测试工具方法对期望结果进行包装
        exp = tm.box_expected(exp, xbox)
        # 断言基准小于索引的结果与期望结果一致
        tm.assert_equal(base < idx, exp)

        # 期望的比较结果，包含基准大于等于索引的布尔数组
        exp = np.array([False, True, True, False])
        # 使用测试工具方法对期望结果进行包装
        exp = tm.box_expected(exp, xbox)
        # 断言基准大于等于索引的结果与期望结果一致
        tm.assert_equal(base >= idx, exp)

        # 期望的比较结果，包含基准小于等于索引的布尔数组
        exp = np.array([True, False, True, True])
        # 使用测试工具方法对期望结果进行包装
        exp = tm.box_expected(exp, xbox)
        # 断言基准小于等于索引的结果与期望结果一致
        tm.assert_equal(base <= idx, exp)
    # 使用给定的频率创建一个周期索引对象
    def test_pi_cmp_nat(self, freq):
        # 创建一个周期索引对象，包含日期字符串和 NaT（不可用日期）值
        idx1 = PeriodIndex(["2011-01", "2011-02", "NaT", "2011-05"], freq=freq)
        
        # 选择第二个周期对象
        per = idx1[1]

        # 比较 idx1 中的每个元素是否大于 per，结果存储在 result 中
        result = idx1 > per
        # 期望的比较结果
        exp = np.array([False, False, False, True])
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)
        
        # 比较 per 是否小于 idx1 中的每个元素，结果存储在 result 中
        result = per < idx1
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)

        # 检查 idx1 中的每个元素是否等于 NaT（不可用日期），结果存储在 result 中
        result = idx1 == pd.NaT
        # 期望的比较结果
        exp = np.array([False, False, False, False])
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)
        
        # 检查 NaT（不可用日期）是否等于 idx1 中的每个元素，结果存储在 result 中
        result = pd.NaT == idx1
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)

        # 检查 idx1 中的每个元素是否不等于 NaT（不可用日期），结果存储在 result 中
        result = idx1 != pd.NaT
        # 期望的比较结果
        exp = np.array([True, True, True, True])
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)
        
        # 检查 NaT（不可用日期）是否不等于 idx1 中的每个元素，结果存储在 result 中
        result = pd.NaT != idx1
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)

        # 创建一个周期索引对象，包含日期字符串和 NaT（不可用日期）值，使用不同的频率
        idx2 = PeriodIndex(["2011-02", "2011-01", "2011-04", "NaT"], freq=freq)
        
        # 比较 idx1 中的每个元素是否小于 idx2 中的对应元素，结果存储在 result 中
        result = idx1 < idx2
        # 期望的比较结果
        exp = np.array([True, False, False, False])
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)

        # 比较 idx1 中的每个元素是否等于 idx2 中的对应元素，结果存储在 result 中
        result = idx1 == idx2
        # 期望的比较结果
        exp = np.array([False, False, False, False])
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)

        # 比较 idx1 中的每个元素是否不等于 idx2 中的对应元素，结果存储在 result 中
        result = idx1 != idx2
        # 期望的比较结果
        exp = np.array([True, True, True, True])
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)

        # 比较 idx1 中的每个元素是否等于自身，结果存储在 result 中
        result = idx1 == idx1
        # 期望的比较结果
        exp = np.array([True, True, False, True])
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)

        # 比较 idx1 中的每个元素是否不等于自身，结果存储在 result 中
        result = idx1 != idx1
        # 期望的比较结果
        exp = np.array([False, False, True, False])
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, exp)

    # 使用不匹配的频率参数对周期索引对象进行比较，预期引发类型错误异常
    @pytest.mark.parametrize("freq", ["M", "2M", "3M"])
    def test_pi_cmp_nat_mismatched_freq_raises(self, freq):
        # 创建一个周期索引对象，包含日期字符串和 NaT（不可用日期）值，使用指定的频率
        idx1 = PeriodIndex(["2011-01", "2011-02", "NaT", "2011-05"], freq=freq)

        # 创建一个不同频率的周期索引对象
        diff = PeriodIndex(["2011-02", "2011-01", "2011-04", "NaT"], freq="4M")
        
        # 定义预期的错误消息
        msg = rf"Invalid comparison between dtype=period\[{freq}\] and PeriodArray"
        
        # 使用 pytest 检查是否引发指定类型的异常，并验证错误消息
        with pytest.raises(TypeError, match=msg):
            idx1 > diff

        # 比较两个周期索引对象是否相等，结果存储在 result 中
        result = idx1 == diff
        # 期望的比较结果
        expected = np.array([False, False, False, False], dtype=bool)
        # 断言两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # TODO: 与 test_pi_cmp_nat 进行代码重复消除
    @pytest.mark.parametrize("dtype", [object, None])
    # 定义一个测试函数，用于比较PeriodIndex中的值与pd.NaT（Not a Time）的关系
    def test_comp_nat(self, dtype):
        # 创建左右两个PeriodIndex对象，包含了日期和pd.NaT
        left = PeriodIndex([Period("2011-01-01"), pd.NaT, Period("2011-01-03")])
        right = PeriodIndex([pd.NaT, pd.NaT, Period("2011-01-03")])

        # 如果dtype参数不为None，则将left和right转换为指定的数据类型
        if dtype is not None:
            left = left.astype(dtype)
            right = right.astype(dtype)

        # 比较left和right中的值是否相等，将结果存储在result中
        result = left == right
        # 预期的结果是一个包含布尔值的NumPy数组
        expected = np.array([False, False, True])
        # 使用assert_numpy_array_equal函数断言result与expected相等
        tm.assert_numpy_array_equal(result, expected)

        # 继续比较left和right中的值是否不相等
        result = left != right
        # 预期的结果是一个包含布尔值的NumPy数组
        expected = np.array([True, True, False])
        # 使用assert_numpy_array_equal函数断言result与expected相等
        tm.assert_numpy_array_equal(result, expected)

        # 将left与pd.NaT进行比较，预期结果是一个包含布尔值的NumPy数组
        expected = np.array([False, False, False])
        # 使用assert_numpy_array_equal函数断言left与pd.NaT的比较结果与expected相等
        tm.assert_numpy_array_equal(left == pd.NaT, expected)
        # 使用assert_numpy_array_equal函数断言pd.NaT与right的比较结果与expected相等
        tm.assert_numpy_array_equal(pd.NaT == right, expected)

        # 将left与pd.NaT进行不等比较，预期结果是一个包含布尔值的NumPy数组
        expected = np.array([True, True, True])
        # 使用assert_numpy_array_equal函数断言left与pd.NaT的不等比较结果与expected相等
        tm.assert_numpy_array_equal(left != pd.NaT, expected)
        # 使用assert_numpy_array_equal函数断言pd.NaT与left的不等比较结果与expected相等
        tm.assert_numpy_array_equal(pd.NaT != left, expected)

        # 将left与pd.NaT进行小于比较，预期结果是一个包含布尔值的NumPy数组
        expected = np.array([False, False, False])
        # 使用assert_numpy_array_equal函数断言left与pd.NaT的小于比较结果与expected相等
        tm.assert_numpy_array_equal(left < pd.NaT, expected)
        # 使用assert_numpy_array_equal函数断言pd.NaT与left的大于比较结果与expected相等
        tm.assert_numpy_array_equal(pd.NaT > left, expected)
class TestPeriodSeriesComparisons:
    # 定义测试类 TestPeriodSeriesComparisons

    def test_cmp_series_period_series_mixed_freq(self):
        # 定义测试方法 test_cmp_series_period_series_mixed_freq，用于比较混合频率的 Period 和 Series 对象
        # GH#13200

        # 创建基础 Series 对象 base，包含不同频率的 Period 对象
        base = Series(
            [
                Period("2011", freq="Y"),
                Period("2011-02", freq="M"),
                Period("2013", freq="Y"),
                Period("2011-04", freq="M"),
            ]
        )

        # 创建待比较的 Series 对象 ser，包含不同频率的 Period 对象
        ser = Series(
            [
                Period("2012", freq="Y"),
                Period("2011-01", freq="M"),
                Period("2013", freq="Y"),
                Period("2011-05", freq="M"),
            ]
        )

        # 创建期望的比较结果 Series 对象 exp，用于比较 base 和 ser 的相等性
        exp = Series([False, False, True, False])
        # 使用 tm.assert_series_equal 检验 base == ser 的结果是否符合期望
        tm.assert_series_equal(base == ser, exp)

        # 创建期望的比较结果 Series 对象 exp，用于比较 base 和 ser 的不等性
        exp = Series([True, True, False, True])
        # 使用 tm.assert_series_equal 检验 base != ser 的结果是否符合期望
        tm.assert_series_equal(base != ser, exp)

        # 创建期望的比较结果 Series 对象 exp，用于比较 base 是否大于 ser
        exp = Series([False, True, False, False])
        # 使用 tm.assert_series_equal 检验 base > ser 的结果是否符合期望
        tm.assert_series_equal(base > ser, exp)

        # 创建期望的比较结果 Series 对象 exp，用于比较 base 是否小于 ser
        exp = Series([True, False, False, True])
        # 使用 tm.assert_series_equal 检验 base < ser 的结果是否符合期望
        tm.assert_series_equal(base < ser, exp)

        # 创建期望的比较结果 Series 对象 exp，用于比较 base 是否大于等于 ser
        exp = Series([False, True, True, False])
        # 使用 tm.assert_series_equal 检验 base >= ser 的结果是否符合期望
        tm.assert_series_equal(base >= ser, exp)

        # 创建期望的比较结果 Series 对象 exp，用于比较 base 是否小于等于 ser
        exp = Series([True, False, True, True])
        # 使用 tm.assert_series_equal 检验 base <= ser 的结果是否符合期望
        tm.assert_series_equal(base <= ser, exp)


class TestPeriodIndexSeriesComparisonConsistency:
    """Test PeriodIndex and Period Series Ops consistency"""
    # 测试 PeriodIndex 和 Period Series 操作的一致性

    # TODO: needs parametrization+de-duplication
    # TODO: 需要参数化和去重

    def _check(self, values, func, expected):
        # 检查方法，用于测试 PeriodIndex 和 Period Series 操作的一致性

        # 创建 PeriodIndex 对象 idx，用给定的 values 初始化
        idx = PeriodIndex(values)
        # 调用传入的函数 func 处理 idx，并将结果存储在 result 中
        result = func(idx)

        # 检查 expected 是否为 pd.Index 或 np.ndarray 类型
        assert isinstance(expected, (pd.Index, np.ndarray))
        # 使用 tm.assert_equal 检验 result 是否与 expected 相等
        tm.assert_equal(result, expected)

        # 创建 Series 对象 s，用给定的 values 初始化
        s = Series(values)
        # 调用传入的函数 func 处理 s，并将结果存储在 result 中
        result = func(s)

        # 创建期望的 Series 对象 exp，用于与 result 进行比较
        exp = Series(expected, name=values.name)
        # 使用 tm.assert_series_equal 检验 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)

    def test_pi_comp_period(self):
        # 测试方法 test_pi_comp_period，用于比较 PeriodIndex 和 Period 之间的操作

        # 创建 PeriodIndex 对象 idx，包含指定的日期字符串，频率为月，并命名为 idx
        idx = PeriodIndex(
            ["2011-01", "2011-02", "2011-03", "2011-04"], freq="M", name="idx"
        )
        # 获取 idx 中的第三个 Period 对象，赋值给 per
        per = idx[2]

        # 定义 lambda 函数 f，用于比较 x 是否等于 per
        f = lambda x: x == per
        # 创建期望的比较结果 np.ndarray 对象 exp
        exp = np.array([False, False, True, False], dtype=np.bool_)
        # 调用 _check 方法，检验 idx 是否满足函数 f 的期望结果 exp
        self._check(idx, f, exp)
        
        # 重新定义 lambda 函数 f，用于比较 per 是否等于 x
        f = lambda x: per == x
        # 再次调用 _check 方法，检验 idx 是否满足函数 f 的期望结果 exp
        self._check(idx, f, exp)

        # 定义 lambda 函数 f，用于比较 x 是否不等于 per
        f = lambda x: x != per
        # 创建期望的比较结果 np.ndarray 对象 exp
        exp = np.array([True, True, False, True], dtype=np.bool_)
        # 调用 _check 方法，检验 idx 是否满足函数 f 的期望结果 exp
        self._check(idx, f, exp)
        
        # 重新定义 lambda 函数 f，用于比较 per 是否不等于 x
        f = lambda x: per != x
        # 再次调用 _check 方法，检验 idx 是否满足函数 f 的期望结果 exp
        self._check(idx, f, exp)

        # 定义 lambda 函数 f，用于比较 per 是否大于等于 x
        f = lambda x: per >= x
        # 创建期望的比较结果 np.ndarray 对象 exp
        exp = np.array([True, True, True, False], dtype=np.bool_)
        # 调用 _check 方法，检验 idx 是否满足函数 f 的期望结果 exp
        self._check(idx, f, exp)

        # 定义 lambda 函数 f，用于比较 x 是否大于 per
        f = lambda x: x > per
        # 创建期望的比较结果 np.ndarray 对象 exp
        exp = np.array([False, False, False, True], dtype=np.bool_)
        # 调用 _check 方法，检验 idx 是否满足函数 f 的期望结果 exp
        self._check(idx, f, exp)

        # 定义 lambda 函数 f，用于比较 per 是否大于等于 x
        f = lambda x: per >= x
        # 创建期望的比较结果 np.ndarray 对象 exp
        exp = np.array([True, True, True, False], dtype=np.bool_)
        # 调用 _check 方法，检验 idx 是否满足函数 f 的期望结果 exp
        self._check(idx, f, exp)
    # 定义一个测试方法，用于测试处理时间周期索引的函数
    def test_pi_comp_period_nat(self):
        # 创建一个时间周期索引对象，包括指定日期和 NaT（不可用时间）值，频率为每月，名称为 'idx'
        idx = PeriodIndex(
            ["2011-01", "NaT", "2011-03", "2011-04"], freq="M", name="idx"
        )
        # 选择索引中的第三个元素，即日期为 '2011-03' 的周期
        per = idx[2]

        # 定义一个匿名函数，用于检查索引中的每个元素是否等于 per
        f = lambda x: x == per
        # 期望的结果是一个布尔数组，指示每个元素是否等于 per
        exp = np.array([False, False, True, False], dtype=np.bool_)
        # 使用自定义方法 _check 进行实际检查
        self._check(idx, f, exp)

        # 定义另一个匿名函数，检查每个元素是否等于 per
        f = lambda x: per == x
        # 再次使用 _check 方法进行检查，期望结果与上述相同
        self._check(idx, f, exp)

        # 定义匿名函数，检查每个元素是否等于 NaT
        f = lambda x: x == pd.NaT
        # 期望的结果是一个布尔数组，指示每个元素是否等于 NaT
        exp = np.array([False, False, False, False], dtype=np.bool_)
        # 使用 _check 方法进行实际检查
        self._check(idx, f, exp)

        # 定义匿名函数，检查每个元素是否等于 NaT
        f = lambda x: pd.NaT == x
        # 再次使用 _check 方法进行检查，期望结果与上述相同
        self._check(idx, f, exp)

        # 定义匿名函数，检查每个元素是否不等于 per
        f = lambda x: x != per
        # 期望的结果是一个布尔数组，指示每个元素是否不等于 per
        exp = np.array([True, True, False, True], dtype=np.bool_)
        # 使用 _check 方法进行实际检查
        self._check(idx, f, exp)

        # 定义匿名函数，检查每个元素是否不等于 per
        f = lambda x: per != x
        # 再次使用 _check 方法进行检查，期望结果与上述相同
        self._check(idx, f, exp)

        # 定义匿名函数，检查每个元素是否不等于 NaT
        f = lambda x: x != pd.NaT
        # 期望的结果是一个布尔数组，指示每个元素是否不等于 NaT
        exp = np.array([True, True, True, True], dtype=np.bool_)
        # 使用 _check 方法进行实际检查
        self._check(idx, f, exp)

        # 定义匿名函数，检查每个元素是否不小于 per
        f = lambda x: per >= x
        # 期望的结果是一个布尔数组，指示每个元素是否不小于 per
        exp = np.array([True, False, True, False], dtype=np.bool_)
        # 使用 _check 方法进行实际检查
        self._check(idx, f, exp)

        # 定义匿名函数，检查每个元素是否小于 per
        f = lambda x: x < per
        # 期望的结果是一个布尔数组，指示每个元素是否小于 per
        exp = np.array([True, False, False, False], dtype=np.bool_)
        # 使用 _check 方法进行实际检查
        self._check(idx, f, exp)

        # 定义匿名函数，检查每个元素是否大于 NaT
        f = lambda x: x > pd.NaT
        # 期望的结果是一个布尔数组，指示每个元素是否大于 NaT
        exp = np.array([False, False, False, False], dtype=np.bool_)
        # 使用 _check 方法进行实际检查
        self._check(idx, f, exp)

        # 定义匿名函数，检查每个元素是否大于等于 NaT
        f = lambda x: pd.NaT >= x
        # 期望的结果是一个布尔数组，指示每个元素是否大于等于 NaT
        exp = np.array([False, False, False, False], dtype=np.bool_)
        # 使用 _check 方法进行实际检查
        self._check(idx, f, exp)
# ------------------------------------------------------------------
# Arithmetic

class TestPeriodFrameArithmetic:
    def test_ops_frame_period(self):
        # GH#13043
        # 创建一个包含两列的 DataFrame，每列包含一个 Period 对象
        df = pd.DataFrame(
            {
                "A": [Period("2015-01", freq="M"), Period("2015-02", freq="M")],
                "B": [Period("2014-01", freq="M"), Period("2014-02", freq="M")],
            }
        )
        # 检查列 'A' 的数据类型是否为 "Period[M]"
        assert df["A"].dtype == "Period[M]"
        # 检查列 'B' 的数据类型是否为 "Period[M]"
        assert df["B"].dtype == "Period[M]"

        # 创建一个新的 Period 对象 p
        p = Period("2015-03", freq="M")
        # 获取 p 的频率
        off = p.freq
        # 创建期望的 DataFrame exp，包含根据 p 和 df 的计算结果构建的 numpy 数组
        exp = pd.DataFrame(
            {
                "A": np.array([2 * off, 1 * off], dtype=object),
                "B": np.array([14 * off, 13 * off], dtype=object),
            }
        )
        # 比较 p 减去 df 后的结果与期望的 exp 是否相等
        tm.assert_frame_equal(p - df, exp)
        # 比较 df 减去 p 后的结果与期望的 -exp 是否相等
        tm.assert_frame_equal(df - p, -1 * exp)

        # 创建另一个包含两列的 DataFrame df2，每列包含一个 Period 对象
        df2 = pd.DataFrame(
            {
                "A": [Period("2015-05", freq="M"), Period("2015-06", freq="M")],
                "B": [Period("2015-05", freq="M"), Period("2015-06", freq="M")],
            }
        )
        # 检查列 'A' 的数据类型是否为 "Period[M]"
        assert df2["A"].dtype == "Period[M]"
        # 检查列 'B' 的数据类型是否为 "Period[M]"
        assert df2["B"].dtype == "Period[M]"

        # 重新定义期望的 DataFrame exp，包含根据 df 和 df2 的计算结果构建的 numpy 数组
        exp = pd.DataFrame(
            {
                "A": np.array([4 * off, 4 * off], dtype=object),
                "B": np.array([16 * off, 16 * off], dtype=object),
            }
        )
        # 比较 df2 减去 df 后的结果与期望的 exp 是否相等
        tm.assert_frame_equal(df2 - df, exp)
        # 比较 df 减去 df2 后的结果与期望的 -exp 是否相等
        tm.assert_frame_equal(df - df2, -1 * exp)


class TestPeriodIndexArithmetic:
    # ---------------------------------------------------------------
    # __add__/__sub__ with PeriodIndex
    # PeriodIndex + other is defined for integers and timedelta-like others
    # PeriodIndex - other is defined for integers, timedelta-like others,
    #   and PeriodIndex (with matching freq)

    def test_parr_add_iadd_parr_raises(self, box_with_array):
        # 创建两个包含 PeriodIndex 的对象 rng 和 other
        rng = period_range("1/1/2000", freq="D", periods=5)
        other = period_range("1/6/2000", freq="D", periods=5)
        # 使用 box_with_array 将 rng 进行封装
        rng = tm.box_expected(rng, box_with_array)
        # 先前的 PeriodIndex 加法实现执行了一个集合操作（并集）。
        # 现在已更改为引发 TypeError。参见 GH#14164 和 GH#13077 了解历史参考。
        msg = r"unsupported operand type\(s\) for \+: .* and .*"
        # 使用 pytest 检查 rng + other 是否引发 TypeError，并匹配指定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            rng + other

        # 使用 pytest 检查 rng += other 是否引发 TypeError，并匹配指定的错误信息 msg
        with pytest.raises(TypeError, match=msg):
            rng += other
    def test_pi_sub_isub_pi(self):
        # GH#20049
        # GH#14164, GH#13077 可以作为历史参考。
        # PeriodIndex 的减法操作最初执行集合差集，
        # 后来更改为在 GH#20049 实现之前引发 TypeError。
        rng = period_range("1/1/2000", freq="D", periods=5)
        other = period_range("1/6/2000", freq="D", periods=5)

        off = rng.freq
        expected = pd.Index([-5 * off] * 5)
        result = rng - other
        tm.assert_index_equal(result, expected)

        rng -= other
        tm.assert_index_equal(rng, expected)

    def test_pi_sub_pi_with_nat(self):
        rng = period_range("1/1/2000", freq="D", periods=5)
        other = rng[1:].insert(0, pd.NaT)
        assert other[1:].equals(rng[1:])

        result = rng - other
        off = rng.freq
        expected = pd.Index([pd.NaT, 0 * off, 0 * off, 0 * off, 0 * off])
        tm.assert_index_equal(result, expected)

    def test_parr_sub_pi_mismatched_freq(self, box_with_array, box_with_array2):
        rng = period_range("1/1/2000", freq="D", periods=5)
        other = period_range("1/6/2000", freq="h", periods=5)

        rng = tm.box_expected(rng, box_with_array)
        other = tm.box_expected(other, box_with_array2)
        msg = r"Input has different freq=[hD] from PeriodArray\(freq=[Dh]\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng - other

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_sub_n_gt_1_ticks(self, tick_classes, n):
        # GH 23878
        p1_d = "19910905"
        p2_d = "19920406"
        p1 = PeriodIndex([p1_d], freq=tick_classes(n))
        p2 = PeriodIndex([p2_d], freq=tick_classes(n))

        expected = PeriodIndex([p2_d], freq=p2.freq.base) - PeriodIndex(
            [p1_d], freq=p1.freq.base
        )

        tm.assert_index_equal((p2 - p1), expected)

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize(
        "offset, kwd_name",
        [
            (pd.offsets.YearEnd, "month"),
            (pd.offsets.QuarterEnd, "startingMonth"),
            (pd.offsets.MonthEnd, None),
            (pd.offsets.Week, "weekday"),
        ],
    )
    def test_sub_n_gt_1_offsets(self, offset, kwd_name, n):
        # GH 23878
        kwds = {kwd_name: 3} if kwd_name is not None else {}
        p1_d = "19910905"
        p2_d = "19920406"
        freq = offset(n, normalize=False, **kwds)
        p1 = PeriodIndex([p1_d], freq=freq)
        p2 = PeriodIndex([p2_d], freq=freq)

        result = p2 - p1
        expected = PeriodIndex([p2_d], freq=freq.base) - PeriodIndex(
            [p1_d], freq=freq.base
        )

        tm.assert_index_equal(result, expected)

    # -------------------------------------------------------------
    # Invalid Operations
    @pytest.mark.parametrize(
        "other",
        [
            # 定义参数化测试用例的输入 `other`
            # datetime scalars
            Timestamp("2016-01-01"),  # 创建一个 pandas Timestamp 对象
            Timestamp("2016-01-01").to_pydatetime(),  # 将 Timestamp 转换为 Python datetime 对象
            Timestamp("2016-01-01").to_datetime64(),  # 将 Timestamp 转换为 datetime64 数据类型
            # datetime-like arrays
            pd.date_range("2016-01-01", periods=3, freq="h"),  # 生成一个小时频率的日期范围
            pd.date_range("2016-01-01", periods=3, tz="Europe/Brussels"),  # 带时区的日期范围
            pd.date_range("2016-01-01", periods=3, freq="s")._data,  # 获取频率为秒的日期范围的内部数据
            pd.date_range("2016-01-01", periods=3, tz="Asia/Tokyo")._data,  # 获取带时区的日期范围的内部数据
            # Miscellaneous invalid types
            3.14,  # 浮点数类型
            np.array([2.0, 3.0, 4.0]),  # numpy 数组
        ],
    )
    def test_parr_add_sub_invalid(self, other, box_with_array):
        # GH#23215
        # 创建一个日期周期范围对象 `rng`，起始日期为 '1/1/2000'，频率为每天 ('D')，共有 3 个周期
        rng = period_range("1/1/2000", freq="D", periods=3)
        # 使用测试辅助函数将 `rng` 对象和 `box_with_array` 进行封装
        rng = tm.box_expected(rng, box_with_array)

        # 构建正则表达式模式 `msg`，用于匹配异常信息
        msg = "|".join(
            [
                r"(:?cannot add PeriodArray and .*)",  # 不能将 PeriodArray 和其他类型相加
                r"(:?cannot subtract .* from (:?a\s)?.*)",  # 不能从周期数组中减去其他类型
                r"(:?unsupported operand type\(s\) for \+: .* and .*)",  # 不支持的操作数类型（加法）
                r"unsupported operand type\(s\) for [+-]: .* and .*",  # 不支持的操作数类型（加减法）
            ]
        )
        # 断言 `assert_invalid_addsub_type` 函数调用，验证期望的异常消息 `msg` 被触发
        assert_invalid_addsub_type(rng, other, msg)
        # 使用 pytest 断言，验证加法操作 `rng + other` 抛出 TypeError 异常并匹配 `msg`
        with pytest.raises(TypeError, match=msg):
            rng + other
        # 使用 pytest 断言，验证加法操作 `other + rng` 抛出 TypeError 异常并匹配 `msg`
        with pytest.raises(TypeError, match=msg):
            other + rng
        # 使用 pytest 断言，验证减法操作 `rng - other` 抛出 TypeError 异常并匹配 `msg`
        with pytest.raises(TypeError, match=msg):
            rng - other
        # 使用 pytest 断言，验证减法操作 `other - rng` 抛出 TypeError 异常并匹配 `msg`
        with pytest.raises(TypeError, match=msg):
            other - rng

    # -----------------------------------------------------------------
    # __add__/__sub__ with ndarray[datetime64] and ndarray[timedelta64]

    def test_pi_add_sub_td64_array_non_tick_raises(self):
        # 创建一个日期周期范围对象 `rng`，起始日期为 '1/1/2000'，频率为季度 ('Q')，共有 3 个周期
        rng = period_range("1/1/2000", freq="Q", periods=3)
        # 创建一个时间增量索引对象 `tdi`，包含三个值为 '-1 Day' 的时间增量
        tdi = TimedeltaIndex(["-1 Day", "-1 Day", "-1 Day"])
        # 获取时间增量数组 `tdarr` 的值
        tdarr = tdi.values

        # 构建异常消息 `msg`，指示无法将 timedelta64[ns] 类型从 period[Q-DEC] 类型中加减
        msg = r"Cannot add or subtract timedelta64\[ns\] dtype from period\[Q-DEC\]"
        # 使用 pytest 断言，验证加法操作 `rng + tdarr` 抛出 TypeError 异常并匹配 `msg`
        with pytest.raises(TypeError, match=msg):
            rng + tdarr
        # 使用 pytest 断言，验证加法操作 `tdarr + rng` 抛出 TypeError 异常并匹配 `msg`
        with pytest.raises(TypeError, match=msg):
            tdarr + rng

        # 使用 pytest 断言，验证减法操作 `rng - tdarr` 抛出 TypeError 异常并匹配 `msg`
        with pytest.raises(TypeError, match=msg):
            rng - tdarr

        # 重新定义异常消息 `msg`，指示无法从 TimedeltaArray 中减去 PeriodArray
        msg = r"cannot subtract PeriodArray from TimedeltaArray"
        # 使用 pytest 断言，验证减法操作 `tdarr - rng` 抛出 TypeError 异常并匹配 `msg`
        with pytest.raises(TypeError, match=msg):
            tdarr - rng
    # 定义测试方法：测试 PeriodIndex 加减 Timedelta-like 的情况，只允许与 tick-like 频率一起使用
    def test_pi_add_sub_td64_array_tick(self):
        # 创建一个周期范围，起始于 "1/1/2000"，频率为 "90D"，共计 3 个周期
        rng = period_range("1/1/2000", freq="90D", periods=3)
        # 创建一个 TimedeltaIndex，包含三个值为 "-1 Day" 的时间增量
        tdi = TimedeltaIndex(["-1 Day", "-1 Day", "-1 Day"])
        # 提取 TimedeltaIndex 的值作为一个 numpy 数组
        tdarr = tdi.values

        # 期望结果：周期范围应为 "12/31/1999" 开始，频率为 "90D"，共计 3 个周期
        expected = period_range("12/31/1999", freq="90D", periods=3)
        # 测试 PeriodIndex 加 TimedeltaIndex 的结果是否符合预期
        result = rng + tdi
        tm.assert_index_equal(result, expected)
        # 测试 PeriodIndex 加 TimedeltaIndex 的数组形式的结果是否符合预期
        result = rng + tdarr
        tm.assert_index_equal(result, expected)
        # 测试 TimedeltaIndex 加 PeriodIndex 的结果是否符合预期
        result = tdi + rng
        tm.assert_index_equal(result, expected)
        # 测试 TimedeltaIndex 的数组形式加 PeriodIndex 的结果是否符合预期
        result = tdarr + rng
        tm.assert_index_equal(result, expected)

        # 期望结果：周期范围应为 "1/2/2000" 开始，频率为 "90D"，共计 3 个周期
        expected = period_range("1/2/2000", freq="90D", periods=3)
        # 测试 PeriodIndex 减 TimedeltaIndex 的结果是否符合预期
        result = rng - tdi
        tm.assert_index_equal(result, expected)
        # 测试 PeriodIndex 减 TimedeltaIndex 的数组形式的结果是否符合预期
        result = rng - tdarr
        tm.assert_index_equal(result, expected)

        # 测试不兼容操作：数组形式的 TimedeltaIndex 减 PeriodIndex 应抛出 TypeError 异常
        msg = r"cannot subtract .* from .*"
        with pytest.raises(TypeError, match=msg):
            tdarr - rng

        with pytest.raises(TypeError, match=msg):
            tdi - rng

    # 参数化测试方法：测试 PeriodArray 减 Timedelta64 数组的情况
    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "h"])
    @pytest.mark.parametrize("tdi_freq", [None, "h"])
    def test_parr_sub_td64array(self, box_with_array, tdi_freq, pi_freq):
        # 获取带有数组的测试盒子对象
        box = box_with_array
        # 如果测试盒子不是 pd.array 或 tm.to_array，使用 pd.Index
        xbox = box if box not in [pd.array, tm.to_array] else pd.Index

        # 创建一个 TimedeltaIndex，包含两个值为 "1 hours", "2 hours" 的时间增量
        tdi = TimedeltaIndex(["1 hours", "2 hours"], freq=tdi_freq)
        # 创建一个 Timestamp 对象，初始时间为 "2018-03-07 17:16:40"，加上 TimedeltaIndex 的值
        dti = Timestamp("2018-03-07 17:16:40") + tdi
        # 将 Timestamp 对象转换为 Period 对象，使用指定的频率 pi_freq
        pi = dti.to_period(pi_freq)

        # 根据测试盒子类型获取 TimedeltaIndex 对象的适配版本
        td64obj = tm.box_expected(tdi, box)

        if pi_freq == "h":
            # 测试 PeriodArray 减 Timedelta64 对象的结果是否符合预期
            result = pi - td64obj
            # 期望结果：将 PeriodArray 转换为 Timestamp 后减去 TimedeltaIndex 的值，再转换为 Period 对象
            expected = (pi.to_timestamp("s") - tdi).to_period(pi_freq)
            expected = tm.box_expected(expected, xbox)
            tm.assert_equal(result, expected)

            # 测试从标量值减去 Timedelta64 对象的结果是否符合预期
            result = pi[0] - td64obj
            expected = (pi[0].to_timestamp("s") - tdi).to_period(pi_freq)
            expected = tm.box_expected(expected, box)
            tm.assert_equal(result, expected)

        elif pi_freq == "D":
            # 对于 Tick 频率但不兼容的情况，应该抛出 IncompatibleFrequency 异常
            msg = (
                "Cannot add/subtract timedelta-like from PeriodArray that is "
                "not an integer multiple of the PeriodArray's freq."
            )
            with pytest.raises(IncompatibleFrequency, match=msg):
                pi - td64obj

            with pytest.raises(IncompatibleFrequency, match=msg):
                pi[0] - td64obj

        else:
            # 对于非 Tick 频率，无论分辨率如何，都不能对 PeriodArray 加减 Timedelta64 对象
            msg = "Cannot add or subtract timedelta64"
            with pytest.raises(TypeError, match=msg):
                pi - td64obj
            with pytest.raises(TypeError, match=msg):
                pi[0] - td64obj
    # operations with array/Index of DateOffset objects

    @pytest.mark.parametrize("box", [np.array, pd.Index])
    # 使用 pytest.mark.parametrize 装饰器对 test_pi_add_offset_array 函数进行参数化，box 参数可以是 np.array 或 pd.Index
    def test_pi_add_offset_array(self, performance_warning, box):
        # GH#18849
        # 创建一个 PeriodIndex 包含两个 Period 对象，分别为 "2015Q1" 和 "2016Q2"
        pi = PeriodIndex([Period("2015Q1"), Period("2016Q2")])
        # 根据参数 box 的类型，创建一个包含两个 QuarterEnd DateOffset 对象的数组或索引
        offs = box(
            [
                pd.offsets.QuarterEnd(n=1, startingMonth=12),
                pd.offsets.QuarterEnd(n=-2, startingMonth=12),
            ]
        )
        # 创建预期的 PeriodIndex，包含两个 Period 对象，分别为 "2015Q2" 和 "2015Q4"，并转换为 object 类型
        expected = PeriodIndex([Period("2015Q2"), Period("2015Q4")]).astype(object)

        # 断言在产生性能警告的情况下，pi + offs 的结果与 expected 相等
        with tm.assert_produces_warning(performance_warning):
            res = pi + offs
        tm.assert_index_equal(res, expected)

        # 断言在产生性能警告的情况下，offs + pi 的结果与 expected 相等
        with tm.assert_produces_warning(performance_warning):
            res2 = offs + pi
        tm.assert_index_equal(res2, expected)

        # 创建一个包含两个不锚定的 DateOffset 对象的数组
        unanchored = np.array([pd.offsets.Hour(n=1), pd.offsets.Minute(n=-2)])
        # 对不兼容的 DateOffset 进行加法或减法操作应当引发性能警告，并最终引发 TypeError 异常。
        msg = r"Input cannot be converted to Period\(freq=Q-DEC\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(performance_warning):
                pi + unanchored
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(performance_warning):
                unanchored + pi

    @pytest.mark.parametrize("box", [np.array, pd.Index])
    # 使用 pytest.mark.parametrize 装饰器对 test_pi_sub_offset_array 函数进行参数化，box 参数可以是 np.array 或 pd.Index
    def test_pi_sub_offset_array(self, performance_warning, box):
        # GH#18824
        # 创建一个 PeriodIndex 包含两个 Period 对象，分别为 "2015Q1" 和 "2016Q2"
        pi = PeriodIndex([Period("2015Q1"), Period("2016Q2")])
        # 根据参数 box 的类型，创建一个包含两个 QuarterEnd DateOffset 对象的数组或索引
        other = box(
            [
                pd.offsets.QuarterEnd(n=1, startingMonth=12),
                pd.offsets.QuarterEnd(n=-2, startingMonth=12),
            ]
        )

        # 创建预期的 PeriodIndex，通过 pi 和 other 计算得到每个 Period 的差值，并转换为 object 类型
        expected = PeriodIndex([pi[n] - other[n] for n in range(len(pi))])
        expected = expected.astype(object)

        # 断言在产生性能警告的情况下，pi - other 的结果与 expected 相等
        with tm.assert_produces_warning(performance_warning):
            res = pi - other
        tm.assert_index_equal(res, expected)

        # 根据参数 box 的类型，创建一个包含两个 Anchored DateOffset 对象的数组或索引
        anchored = box([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)])

        # 对锚定的 DateOffset 进行加法或减法操作应当引发性能警告，并最终引发 TypeError 异常。
        msg = r"Input has different freq=-1M from Period\(freq=Q-DEC\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(performance_warning):
                pi - anchored
        with pytest.raises(IncompatibleFrequency, match=msg):
            with tm.assert_produces_warning(performance_warning):
                anchored - pi
    def test_pi_add_iadd_int(self, one):
        # 创建一个时间段范围，从 "2000-01-01 09:00" 开始，每小时一个时间段，总共10个时间段
        rng = period_range("2000-01-01 09:00", freq="h", periods=10)
        # 将时间段范围 `rng` 中的每个时间段都加上 `one`
        result = rng + one
        # 期望结果是从 "2000-01-01 10:00" 开始，每小时一个时间段，总共10个时间段
        expected = period_range("2000-01-01 10:00", freq="h", periods=10)
        # 断言 `result` 和 `expected` 相等
        tm.assert_index_equal(result, expected)
        # 将时间段范围 `rng` 自身每个时间段都加上 `one`
        rng += one
        # 断言 `rng` 和 `expected` 相等
        tm.assert_index_equal(rng, expected)

    def test_pi_sub_isub_int(self, one):
        """
        PeriodIndex.__sub__ and __isub__ with several representations of
        the integer 1, e.g. int, np.int64, np.uint8, ...
        """
        # 创建一个时间段范围，从 "2000-01-01 09:00" 开始，每小时一个时间段，总共10个时间段
        rng = period_range("2000-01-01 09:00", freq="h", periods=10)
        # 将时间段范围 `rng` 中的每个时间段都减去 `one`
        result = rng - one
        # 期望结果是从 "2000-01-01 08:00" 开始，每小时一个时间段，总共10个时间段
        expected = period_range("2000-01-01 08:00", freq="h", periods=10)
        # 断言 `result` 和 `expected` 相等
        tm.assert_index_equal(result, expected)
        # 将时间段范围 `rng` 自身每个时间段都减去 `one`
        rng -= one
        # 断言 `rng` 和 `expected` 相等
        tm.assert_index_equal(rng, expected)

    @pytest.mark.parametrize("five", [5, np.array(5, dtype=np.int64)])
    def test_pi_sub_intlike(self, five):
        # 创建一个包含50个时间段的时间段范围，从 "2007-01" 开始
        rng = period_range("2007-01", periods=50)
        # 将时间段范围 `rng` 中的每个时间段都减去 `five`
        result = rng - five
        # 期望结果是时间段范围 `rng` 中每个时间段加上 `-five`
        exp = rng + (-five)
        # 断言 `result` 和 `exp` 相等
        tm.assert_index_equal(result, exp)

    def test_pi_add_sub_int_array_freqn_gt1(self):
        # GH#47209 测试当 freq.n > 1 时，向 PeriodIndex 添加整数数组的行为与标量相同
        # 创建一个时间段范围，从 "2016-01-01" 开始，每两天一个时间段，总共10个时间段
        pi = period_range("2016-01-01", periods=10, freq="2D")
        # 创建一个包含0到9的整数数组
        arr = np.arange(10)
        # 将时间段范围 `pi` 中的每个时间段都加上对应位置的数组元素 `arr`
        result = pi + arr
        # 期望结果是一个包含每个时间段加上对应数组元素后的新时间段的索引
        expected = pd.Index([x + y for x, y in zip(pi, arr)])
        # 断言 `result` 和 `expected` 相等
        tm.assert_index_equal(result, expected)

        # 将时间段范围 `pi` 中的每个时间段都减去对应位置的数组元素 `arr`
        result = pi - arr
        # 期望结果是一个包含每个时间段减去对应数组元素后的新时间段的索引
        expected = pd.Index([x - y for x, y in zip(pi, arr)])
        # 断言 `result` 和 `expected` 相等
        tm.assert_index_equal(result, expected)

    def test_pi_sub_isub_offset(self):
        # offset
        # DateOffset
        # 创建一个时间段范围，从 "2014" 到 "2024"，每年一个时间段
        rng = period_range("2014", "2024", freq="Y")
        # 将时间段范围 `rng` 中的每个时间段都减去 5年
        result = rng - pd.offsets.YearEnd(5)
        # 期望结果是从 "2009" 到 "2019"，每年一个时间段
        expected = period_range("2009", "2019", freq="Y")
        # 断言 `result` 和 `expected` 相等
        tm.assert_index_equal(result, expected)
        # 将时间段范围 `rng` 自身每个时间段都减去 5年
        rng -= pd.offsets.YearEnd(5)
        # 断言 `rng` 和 `expected` 相等
        tm.assert_index_equal(rng, expected)

        # 创建一个时间段范围，从 "2014-01" 到 "2016-12"，每月一个时间段
        rng = period_range("2014-01", "2016-12", freq="M")
        # 将时间段范围 `rng` 中的每个时间段都减去 5个月
        result = rng - pd.offsets.MonthEnd(5)
        # 期望结果是从 "2013-08" 到 "2016-07"，每月一个时间段
        expected = period_range("2013-08", "2016-07", freq="M")
        # 断言 `result` 和 `expected` 相等
        tm.assert_index_equal(result, expected)

        # 将时间段范围 `rng` 自身每个时间段都减去 5个月
        rng -= pd.offsets.MonthEnd(5)
        # 断言 `rng` 和 `expected` 相等
        tm.assert_index_equal(rng, expected)

    @pytest.mark.parametrize("transpose", [True, False])
    def test_pi_add_offset_n_gt1(self, box_with_array, transpose):
        # GH#23215
        # add offset to PeriodIndex with freq.n > 1

        # 创建一个时间段 `per`，从 "2016-01" 开始，每两个月一个时间段
        per = Period("2016-01", freq="2M")
        # 将时间段 `per` 放入时间段索引 `pi` 中
        pi = PeriodIndex([per])

        # 期望的时间段索引
        expected = PeriodIndex(["2016-03"], freq="2M")

        # 使用测试工具函数对 `pi` 进行包装处理，与数组盒子 `box_with_array` 和 `transpose` 参数相关
        pi = tm.box_expected(pi, box_with_array, transpose=transpose)
        # 使用测试工具函数对 `expected` 进行包装处理，与数组盒子 `box_with_array` 和 `transpose` 参数相关
        expected = tm.box_expected(expected, box_with_array, transpose=transpose)

        # 将时间段 `per.freq` 加到时间段索引 `pi` 中
        result = pi + per.freq
        # 断言 `result` 和 `expected` 相等
        tm.assert_equal(result, expected)

        # 将时间段 `per.freq` 加到时间段索引 `pi` 中
        result = per.freq + pi
        # 断言 `result` 和 `expected` 相等
        tm.assert_equal(result, expected)
    # 定义测试函数，测试 PeriodIndex 对象与偏移量相加，偏移量的频率大于1且不能整除频率时的情况
    def test_pi_add_offset_n_gt1_not_divisible(self, box_with_array):
        # GH#23215
        # 创建一个频率为"2M"的 PeriodIndex 对象，包含日期"2016-01"
        pi = PeriodIndex(["2016-01"], freq="2M")
        # 创建一个预期的 PeriodIndex 对象，包含日期"2016-04"，频率也为"2M"
        expected = PeriodIndex(["2016-04"], freq="2M")

        # 将 pi 对象与 box_with_array 进行封装处理
        pi = tm.box_expected(pi, box_with_array)
        # 将 expected 对象与 box_with_array 进行封装处理
        expected = tm.box_expected(expected, box_with_array)

        # 对 pi 对象加上"3ME"的偏移量，并断言结果与 expected 相等
        result = pi + to_offset("3ME")
        tm.assert_equal(result, expected)

        # 对"3ME"的偏移量与 pi 对象相加，并断言结果与 expected 相等
        result = to_offset("3ME") + pi
        tm.assert_equal(result, expected)

    # ---------------------------------------------------------------
    # __add__/__sub__ with integer arrays

    # 使用 numpy 数组或者 pandas Index 对象作为 int_holder 参数，测试 PeriodIndex 对象与整数数组相加的情况
    @pytest.mark.parametrize("int_holder", [np.array, pd.Index])
    # 使用 operator.add 或 ops.radd 函数作为 op 参数，测试 PeriodIndex 对象与整数数组相加的情况
    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    def test_pi_add_intarray(self, int_holder, op):
        # GH#19959
        # 创建一个包含 Period 对象"2015Q1"和"NaT"的 PeriodIndex 对象
        pi = PeriodIndex([Period("2015Q1"), Period("NaT")])
        # 创建一个整数数组 [4, -1]
        other = int_holder([4, -1])

        # 使用 op 函数将 pi 与 other 相加，断言结果与预期的 PeriodIndex 对象相等
        result = op(pi, other)
        expected = PeriodIndex([Period("2016Q1"), Period("NaT")])
        tm.assert_index_equal(result, expected)

    # 使用 numpy 数组或者 pandas Index 对象作为 int_holder 参数，测试 PeriodIndex 对象与整数数组相减的情况
    @pytest.mark.parametrize("int_holder", [np.array, pd.Index])
    def test_pi_sub_intarray(self, int_holder):
        # GH#19959
        # 创建一个包含 Period 对象"2015Q1"和"NaT"的 PeriodIndex 对象
        pi = PeriodIndex([Period("2015Q1"), Period("NaT")])
        # 创建一个整数数组 [4, -1]
        other = int_holder([4, -1])

        # 将 pi 对象与 other 相减，断言结果与预期的 PeriodIndex 对象相等
        result = pi - other
        expected = PeriodIndex([Period("2014Q1"), Period("NaT")])
        tm.assert_index_equal(result, expected)

        # 测试用例，检查在使用一元负号操作符时是否会抛出 TypeError 异常
        msg = r"bad operand type for unary -: 'PeriodArray'"
        with pytest.raises(TypeError, match=msg):
            other - pi

    # ---------------------------------------------------------------
    # Timedelta-like (timedelta, timedelta64, Timedelta, Tick)
    # TODO: Some of these are misnomers because of non-Tick DateOffsets
    def test_parr_add_timedeltalike_minute_gt1(self, three_days, box_with_array):
        # GH#23031 adding a time-delta-like offset to a PeriodArray that has
        # minute frequency with n != 1.  A more general case is tested below
        # in test_pi_add_timedeltalike_tick_gt1, but here we write out the
        # expected result more explicitly.

        # 使用传入的参数作为时间间隔对象 `three_days`
        other = three_days
        # 创建一个包含三个日期的周期范围，频率为每两天一次
        rng = period_range("2014-05-01", periods=3, freq="2D")
        # 通过 `tm.box_expected` 函数将结果转换为预期格式，传入 `box_with_array` 进行处理
        rng = tm.box_expected(rng, box_with_array)

        # 创建预期的周期索引对象，包含三个日期，频率为每两天一次
        expected = PeriodIndex(["2014-05-04", "2014-05-06", "2014-05-08"], freq="2D")
        # 通过 `tm.box_expected` 函数将结果转换为预期格式，传入 `box_with_array` 进行处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算结果为周期范围 `rng` 加上时间间隔 `other`
        result = rng + other
        # 使用 `tm.assert_equal` 函数比较 `result` 和 `expected` 是否相等
        tm.assert_equal(result, expected)

        # 计算结果为时间间隔 `other` 加上周期范围 `rng`
        result = other + rng
        # 使用 `tm.assert_equal` 函数比较 `result` 和 `expected` 是否相等
        tm.assert_equal(result, expected)

        # 执行减法操作，计算结果为周期范围 `rng` 减去时间间隔 `other`
        expected = PeriodIndex(["2014-04-28", "2014-04-30", "2014-05-02"], freq="2D")
        # 通过 `tm.box_expected` 函数将结果转换为预期格式，传入 `box_with_array` 进行处理
        expected = tm.box_expected(expected, box_with_array)
        result = rng - other
        # 使用 `tm.assert_equal` 函数比较 `result` 和 `expected` 是否相等
        tm.assert_equal(result, expected)

        # 设置错误消息的正则表达式模式，用于检查是否引发了特定的类型错误
        msg = "|".join(
            [
                r"bad operand type for unary -: 'PeriodArray'",
                r"cannot subtract PeriodArray from timedelta64\[[hD]\]",
            ]
        )
        # 使用 `pytest.raises` 检查是否引发了 `TypeError` 异常，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            other - rng

    @pytest.mark.parametrize("freqstr", ["5ns", "5us", "5ms", "5s", "5min", "5h", "5d"])
    def test_parr_add_timedeltalike_tick_gt1(self, three_days, freqstr, box_with_array):
        # GH#23031 adding a time-delta-like offset to a PeriodArray that has
        # tick-like frequency with n != 1

        # 使用传入的参数作为时间间隔对象 `three_days`
        other = three_days
        # 创建一个包含六个日期的周期范围，频率由参数 `freqstr` 指定
        rng = period_range("2014-05-01", periods=6, freq=freqstr)
        # 获取范围中的第一个日期
        first = rng[0]
        # 通过 `tm.box_expected` 函数将结果转换为预期格式，传入 `box_with_array` 进行处理
        rng = tm.box_expected(rng, box_with_array)

        # 创建预期的周期范围对象，从第一个日期开始，加上时间间隔 `other`，频率由 `freqstr` 指定
        expected = period_range(first + other, periods=6, freq=freqstr)
        # 通过 `tm.box_expected` 函数将结果转换为预期格式，传入 `box_with_array` 进行处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算结果为周期范围 `rng` 加上时间间隔 `other`
        result = rng + other
        # 使用 `tm.assert_equal` 函数比较 `result` 和 `expected` 是否相等
        tm.assert_equal(result, expected)

        # 计算结果为时间间隔 `other` 加上周期范围 `rng`
        result = other + rng
        # 使用 `tm.assert_equal` 函数比较 `result` 和 `expected` 是否相等
        tm.assert_equal(result, expected)

        # 执行减法操作，计算结果为周期范围 `rng` 减去时间间隔 `other`
        expected = period_range(first - other, periods=6, freq=freqstr)
        # 通过 `tm.box_expected` 函数将结果转换为预期格式，传入 `box_with_array` 进行处理
        expected = tm.box_expected(expected, box_with_array)
        result = rng - other
        # 使用 `tm.assert_equal` 函数比较 `result` 和 `expected` 是否相等
        tm.assert_equal(result, expected)

        # 设置错误消息的正则表达式模式，用于检查是否引发了特定的类型错误
        msg = "|".join(
            [
                r"bad operand type for unary -: 'PeriodArray'",
                r"cannot subtract PeriodArray from timedelta64\[[hD]\]",
            ]
        )
        # 使用 `pytest.raises` 检查是否引发了 `TypeError` 异常，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            other - rng

    def test_pi_add_iadd_timedeltalike_daily(self, three_days):
        # Tick

        # 使用传入的参数作为时间间隔对象 `three_days`
        other = three_days
        # 创建一个包含从 "2014-05-01" 到 "2014-05-15" 的每日周期范围
        rng = period_range("2014-05-01", "2014-05-15", freq="D")
        # 创建预期的周期范围对象，从 "2014-05-04" 到 "2014-05-18"，频率为每日
        expected = period_range("2014-05-04", "2014-05-18", freq="D")

        # 计算结果为周期范围 `rng` 加上时间间隔 `other`
        result = rng + other
        # 使用 `tm.assert_index_equal` 函数比较 `result` 和 `expected` 是否相等
        tm.assert_index_equal(result, expected)

        # 原地操作，将周期范围 `rng` 加上时间间隔 `other`
        rng += other
        # 使用 `tm.assert_index_equal` 函数比较 `rng` 和 `expected` 是否相等
        tm.assert_index_equal(rng, expected)
    # 测试用例：减法和就地减法操作对于类似时间间隔的 PeriodRange 对象
    def test_pi_sub_isub_timedeltalike_daily(self, three_days):
        # 将 three_days 参数赋给 other 变量
        other = three_days
        # 创建一个频率为每日的日期范围对象 rng，从 '2014-05-01' 到 '2014-05-15'
        rng = period_range("2014-05-01", "2014-05-15", freq="D")
        # 创建预期的日期范围对象 expected，从 '2014-04-28' 到 '2014-05-12'，频率为每日
        expected = period_range("2014-04-28", "2014-05-12", freq="D")

        # 计算 rng 减去 other 的结果，并赋给 result
        result = rng - other
        # 断言 result 等于 expected
        tm.assert_index_equal(result, expected)

        # 就地操作：rng 减去 other
        rng -= other
        # 断言 rng 等于 expected
        tm.assert_index_equal(rng, expected)

    # 测试用例：添加、就地添加、减去、就地减去操作对于频率不匹配的 PeriodRange 对象
    def test_parr_add_sub_timedeltalike_freq_mismatch_daily(
        self, not_daily, box_with_array
    ):
        # 将 not_daily 参数赋给 other 变量
        other = not_daily
        # 创建一个频率为每日的日期范围对象 rng，从 '2014-05-01' 到 '2014-05-15'
        rng = period_range("2014-05-01", "2014-05-15", freq="D")
        # 使用 tm.box_expected 方法处理 rng，将其结果重新赋给 rng
        rng = tm.box_expected(rng, box_with_array)

        # 构造匹配的错误消息，用于检测 IncompatibleFrequency 异常
        msg = "|".join(
            [
                # 非时间间隔类 DateOffset
                "Input has different freq(=.+)? from Period.*?\\(freq=D\\)",
                # 时间间隔/td64/Timedelta 但不是 PeriodArray 频率的整数倍
                "Cannot add/subtract timedelta-like from PeriodArray that is "
                "not an integer multiple of the PeriodArray's freq.",
            ]
        )

        # 使用 pytest 检查是否会引发 IncompatibleFrequency 异常，错误消息要匹配 msg
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng + other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng += other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng - other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng -= other

    # 测试用例：添加和就地添加操作对于类似时间间隔的 PeriodRange 对象（每小时频率）
    def test_pi_add_iadd_timedeltalike_hourly(self, two_hours):
        # 将 two_hours 参数赋给 other 变量
        other = two_hours
        # 创建一个频率为每小时的日期范围对象 rng，从 '2014-01-01 10:00' 到 '2014-01-05 10:00'
        rng = period_range("2014-01-01 10:00", "2014-01-05 10:00", freq="h")
        # 创建预期的日期范围对象 expected，从 '2014-01-01 12:00' 到 '2014-01-05 12:00'，频率为每小时
        expected = period_range("2014-01-01 12:00", "2014-01-05 12:00", freq="h")

        # 计算 rng 加上 other 的结果，并赋给 result
        result = rng + other
        # 断言 result 等于 expected
        tm.assert_index_equal(result, expected)

        # 就地操作：rng 加上 other
        rng += other
        # 断言 rng 等于 expected
        tm.assert_index_equal(rng, expected)

    # 测试用例：添加操作对于频率不匹配的 PeriodRange 对象（每小时频率）
    def test_parr_add_timedeltalike_mismatched_freq_hourly(
        self, not_hourly, box_with_array
    ):
        # 将 not_hourly 参数赋给 other 变量
        other = not_hourly
        # 创建一个频率为每小时的日期范围对象 rng，从 '2014-01-01 10:00' 到 '2014-01-05 10:00'
        rng = period_range("2014-01-01 10:00", "2014-01-05 10:00", freq="h")
        # 使用 tm.box_expected 方法处理 rng，将其结果重新赋给 rng
        rng = tm.box_expected(rng, box_with_array)

        # 构造匹配的错误消息，用于检测 IncompatibleFrequency 异常
        msg = "|".join(
            [
                # 非时间间隔类 DateOffset
                "Input has different freq(=.+)? from Period.*?\\(freq=h\\)",
                # 时间间隔/td64/Timedelta 但不是 PeriodArray 频率的整数倍
                "Cannot add/subtract timedelta-like from PeriodArray that is "
                "not an integer multiple of the PeriodArray's freq.",
            ]
        )

        # 使用 pytest 检查是否会引发 IncompatibleFrequency 异常，错误消息要匹配 msg
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng + other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng += other
    # 测试两个时间段相减，并验证结果
    def test_pi_sub_isub_timedeltalike_hourly(self, two_hours):
        # 将参数赋值给另一个变量
        other = two_hours
        # 创建时间段范围对象，频率为每小时
        rng = period_range("2014-01-01 10:00", "2014-01-05 10:00", freq="h")
        # 创建预期的时间段范围对象，频率为每小时
        expected = period_range("2014-01-01 08:00", "2014-01-05 08:00", freq="h")

        # 计算时间段范围对象与另一个对象的差值
        result = rng - other
        # 验证结果与预期是否相等
        tm.assert_index_equal(result, expected)

        # 将时间段范围对象与另一个对象相减，并更新时间段范围对象
        rng -= other
        # 验证更新后的时间段范围对象与预期是否相等
        tm.assert_index_equal(rng, expected)

    # 测试时间段范围对象与时间偏移对象相加，并验证结果
    def test_add_iadd_timedeltalike_annual(self):
        # 创建时间段范围对象，频率为每年
        rng = period_range("2014", "2024", freq="Y")
        # 将时间段范围对象与年末时间偏移对象相加
        result = rng + pd.offsets.YearEnd(5)
        # 创建预期的时间段范围对象，频率为每年
        expected = period_range("2019", "2029", freq="Y")
        # 验证结果与预期是否相等
        tm.assert_index_equal(result, expected)
        # 将时间段范围对象与年末时间偏移对象相加，并更新时间段范围对象
        rng += pd.offsets.YearEnd(5)
        # 验证更新后的时间段范围对象与预期是否相等
        tm.assert_index_equal(rng, expected)

    # 测试时间段范围对象与不匹配频率的对象相加和相减
    def test_pi_add_sub_timedeltalike_freq_mismatch_annual(self, mismatched_freq):
        # 将参数赋值给另一个变量
        other = mismatched_freq
        # 创建时间段范围对象，频率为每年
        rng = period_range("2014", "2024", freq="Y")
        # 定义错误消息
        msg = "Input has different freq(=.+)? from Period.*?\\(freq=Y-DEC\\)"
        # 验证相加和相减操作是否会引发不匹配频率的异常
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng + other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng += other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng - other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng -= other

    # 测试时间段范围对象与月末时间偏移对象相加，并验证结果
    def test_pi_add_iadd_timedeltalike_M(self):
        # 创建时间段范围对象，频率为每月
        rng = period_range("2014-01", "2016-12", freq="M")
        # 创建预期的时间段范围对象，频率为每月
        expected = period_range("2014-06", "2017-05", freq="M")

        # 将时间段范围对象与月末时间偏移对象相加
        result = rng + pd.offsets.MonthEnd(5)
        # 验证结果与预期是否相等
        tm.assert_index_equal(result, expected)

        # 将时间段范围对象与月末时间偏移对象相加，并更新时间段范围对象
        rng += pd.offsets.MonthEnd(5)
        # 验证更新后的时间段范围对象与预期是否相等
        tm.assert_index_equal(rng, expected)

    # 测试时间段范围对象与不匹配频率的对象相加和相减
    def test_pi_add_sub_timedeltalike_freq_mismatch_monthly(self, mismatched_freq):
        # 将参数赋值给另一个变量
        other = mismatched_freq
        # 创建时间段范围对象，频率为每月
        rng = period_range("2014-01", "2016-12", freq="M")
        # 定义错误消息
        msg = "Input has different freq(=.+)? from Period.*?\\(freq=M\\)"
        # 验证相加和相减操作是否会引发不匹配频率的异常
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng + other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng += other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng - other
        with pytest.raises(IncompatibleFrequency, match=msg):
            rng -= other

    # 参数化测试，测试是否可以转置
    @pytest.mark.parametrize("transpose", [True, False])
    def test_parr_add_sub_td64_nat(self, box_with_array, transpose):
        # GH#23320 special handling for timedelta64("NaT")
        # 创建一个包含9个时期的PeriodIndex对象，起始日期为"1994-04-01"，频率为每19天一次
        pi = period_range("1994-04-01", periods=9, freq="19D")
        # 创建一个np.timedelta64类型的"NaT"对象
        other = np.timedelta64("NaT")
        # 创建一个预期的PeriodIndex对象，包含9个"NaT"，频率为每19天一次
        expected = PeriodIndex(["NaT"] * 9, freq="19D")

        # 将pi和box_with_array结合，生成一个被测试的对象obj
        obj = tm.box_expected(pi, box_with_array, transpose=transpose)
        # 将预期的PeriodIndex对象和box_with_array结合，生成一个预期的对象
        expected = tm.box_expected(expected, box_with_array, transpose=transpose)

        # 测试obj与other相加的结果
        result = obj + other
        tm.assert_equal(result, expected)
        # 测试other与obj相加的结果
        result = other + obj
        tm.assert_equal(result, expected)
        # 测试obj与other相减的结果
        result = obj - other
        tm.assert_equal(result, expected)
        # 测试尝试将obj从other中减去会抛出TypeError异常，异常消息中包含指定的正则表达式匹配
        msg = r"cannot subtract .* from .*"
        with pytest.raises(TypeError, match=msg):
            other - obj

    @pytest.mark.parametrize(
        "other",
        [
            np.array(["NaT"] * 9, dtype="m8[ns]"),
            TimedeltaArray._from_sequence(["NaT"] * 9, dtype="m8[ns]"),
        ],
    )
    def test_parr_add_sub_tdt64_nat_array(self, box_with_array, other):
        # 创建一个包含9个时期的PeriodIndex对象，起始日期为"1994-04-01"，频率为每19天一次
        pi = period_range("1994-04-01", periods=9, freq="19D")
        # 创建一个预期的PeriodIndex对象，包含9个"NaT"，频率为每19天一次
        expected = PeriodIndex(["NaT"] * 9, freq="19D")

        # 将pi和box_with_array结合，生成一个被测试的对象obj
        obj = tm.box_expected(pi, box_with_array)
        # 将预期的PeriodIndex对象和box_with_array结合，生成一个预期的对象
        expected = tm.box_expected(expected, box_with_array)

        # 测试obj与other相加的结果
        result = obj + other
        tm.assert_equal(result, expected)
        # 测试other与obj相加的结果
        result = other + obj
        tm.assert_equal(result, expected)
        # 测试obj与other相减的结果
        result = obj - other
        tm.assert_equal(result, expected)
        # 测试尝试将obj从other中减去会抛出TypeError异常，异常消息中包含指定的正则表达式匹配
        msg = r"cannot subtract .* from .*"
        with pytest.raises(TypeError, match=msg):
            other - obj

        # 修改other数组中的第一个元素，使其不再是"NaT"
        other = other.copy()
        other[0] = np.timedelta64(0, "ns")
        # 创建一个预期的PeriodIndex对象，首个元素为pi的首个元素，其余为"NaT"，频率为每19天一次
        expected = PeriodIndex([pi[0]] + ["NaT"] * 8, freq="19D")
        expected = tm.box_expected(expected, box_with_array)

        # 再次测试obj与修改后的other相加的结果
        result = obj + other
        tm.assert_equal(result, expected)
        # 再次测试修改后的other与obj相加的结果
        result = other + obj
        tm.assert_equal(result, expected)
        # 再次测试obj与修改后的other相减的结果
        result = obj - other
        tm.assert_equal(result, expected)
        # 再次测试尝试将obj从修改后的other中减去会抛出TypeError异常，异常消息中包含指定的正则表达式匹配
        with pytest.raises(TypeError, match=msg):
            other - obj

    # ---------------------------------------------------------------
    # Unsorted

    def test_parr_add_sub_index(self):
        # 检查PeriodArray在算术操作中是否委托给Index
        # 创建一个包含3个时期的PeriodIndex对象，起始日期为"2000-12-31"
        pi = period_range("2000-12-31", periods=3)
        # 从pi中获取其PeriodArray对象
        parr = pi.array

        # 测试parr与pi相减的结果
        result = parr - pi
        # 期望的结果是pi与自身相减的结果
        expected = pi - pi
        tm.assert_index_equal(result, expected)
    # 定义一个测试方法，用于测试 PeriodArray 的加法和减法操作
    def test_parr_add_sub_object_array(self, performance_warning):
        # 创建一个包含三个日期的 PeriodIndex 对象，频率为每天
        pi = period_range("2000-12-31", periods=3, freq="D")
        # 获取 PeriodIndex 对象的 array 属性，即 PeriodArray 对象
        parr = pi.array

        # 创建一个包含 Timedelta、pd.offsets.Day 和整数的 NumPy 数组
        other = np.array([Timedelta(days=1), pd.offsets.Day(2), 3])

        # 使用 assert_produces_warning 上下文管理器，检查性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行 PeriodArray 和 other 数组的加法操作
            result = parr + other

        # 创建预期的 PeriodIndex 对象，包含三个增加后的日期，频率为每天
        expected = PeriodIndex(
            ["2001-01-01", "2001-01-03", "2001-01-05"], freq="D"
        )._data.astype(object)
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

        # 使用 assert_produces_warning 上下文管理器，检查性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行 PeriodArray 和 other 数组的减法操作
            result = parr - other

        # 创建预期的 PeriodIndex 对象，包含三个减少后的日期，频率为每天
        expected = PeriodIndex(["2000-12-30"] * 3, freq="D")._data.astype(object)
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

    # 定义一个测试方法，用于测试 PeriodArray 和 Timestamp 相加会引发异常
    def test_period_add_timestamp_raises(self, box_with_array):
        # 创建一个 Timestamp 对象
        ts = Timestamp("2017")
        # 创建一个 Period 对象，频率为每月
        per = Period("2017", freq="M")

        # 创建一个包含 per 的索引数组，数据类型为 Period[M]
        arr = pd.Index([per], dtype="Period[M]")
        # 使用 box_expected 方法处理 arr 对象，与 box_with_array 参数相关
        arr = tm.box_expected(arr, box_with_array)

        # 准备异常消息，指示无法将 PeriodArray 和 Timestamp 相加
        msg = "cannot add PeriodArray and Timestamp"
        # 使用 pytest.raises 检查是否抛出预期的 TypeError 异常，匹配指定消息
        with pytest.raises(TypeError, match=msg):
            arr + ts
        with pytest.raises(TypeError, match=msg):
            ts + arr

        # 如果 box_with_array 是 pd.DataFrame 类型
        if box_with_array is pd.DataFrame:
            # 准备异常消息，指示无法将 PeriodArray 和 Timestamp 相加
            msg = "cannot add PeriodArray and Timestamp"
        else:
            # 准备异常消息，指示无法将 PeriodArray 和 DatetimeArray 相加
            msg = "cannot add PeriodArray and DatetimeArray"
        # 使用 pytest.raises 检查是否抛出预期的 TypeError 异常，匹配指定消息
        with pytest.raises(TypeError, match=msg):
            arr + Series([ts])
        with pytest.raises(TypeError, match=msg):
            Series([ts]) + arr
        with pytest.raises(TypeError, match=msg):
            arr + pd.Index([ts])
        with pytest.raises(TypeError, match=msg):
            pd.Index([ts]) + arr

        # 如果 box_with_array 是 pd.DataFrame 类型
        if box_with_array is pd.DataFrame:
            # 准备异常消息，指示无法将 PeriodArray 和 DatetimeArray 相加
            msg = "cannot add PeriodArray and DatetimeArray"
        else:
            # 准备异常消息，指示不支持将 Period 和 DatetimeArray 相加
            msg = r"unsupported operand type\(s\) for \+: 'Period' and 'DatetimeArray"
        # 使用 pytest.raises 检查是否抛出预期的 TypeError 异常，匹配指定消息
        with pytest.raises(TypeError, match=msg):
            arr + pd.DataFrame([ts])

        # 如果 box_with_array 是 pd.DataFrame 类型
        if box_with_array is pd.DataFrame:
            # 准备异常消息，指示无法将 PeriodArray 和 DatetimeArray 相加
            msg = "cannot add PeriodArray and DatetimeArray"
        else:
            # 准备异常消息，指示不支持将 DatetimeArray 和 Period 相加
            msg = r"unsupported operand type\(s\) for \+: 'DatetimeArray' and 'Period'"
        # 使用 pytest.raises 检查是否抛出预期的 TypeError 异常，匹配指定消息
        with pytest.raises(TypeError, match=msg):
            pd.DataFrame([ts]) + arr
class TestPeriodSeriesArithmetic:
    # 测试加法操作，将时间差标量与 Series 对象相加
    def test_parr_add_timedeltalike_scalar(self, three_days, box_with_array):
        # GH#13043
        # 创建包含两个 Period 对象的 Series，频率为每日
        ser = Series(
            [Period("2015-01-01", freq="D"), Period("2015-01-02", freq="D")],
            name="xxx",
        )
        assert ser.dtype == "Period[D]"

        # 期望结果，将两个 Period 对象的日期增加三天
        expected = Series(
            [Period("2015-01-04", freq="D"), Period("2015-01-05", freq="D")],
            name="xxx",
        )

        # 将 ser 和 box_with_array 组合，得到包装后的对象
        obj = tm.box_expected(ser, box_with_array)
        if box_with_array is pd.DataFrame:
            assert (obj.dtypes == "Period[D]").all()

        # 将 obj 和 three_days 相加，并验证结果是否与 expected 相等
        result = obj + three_days
        tm.assert_equal(result, expected)

        # 将 three_days 和 obj 相加，并验证结果是否与 expected 相等
        result = three_days + obj
        tm.assert_equal(result, expected)

    # 测试 Series 对象与 Period 对象的操作
    def test_ops_series_period(self):
        # GH#13043
        # 创建包含两个 Period 对象的 Series，频率为每日
        ser = Series(
            [Period("2015-01-01", freq="D"), Period("2015-01-02", freq="D")],
            name="xxx",
        )
        assert ser.dtype == "Period[D]"

        # 创建一个单独的 Period 对象
        per = Period("2015-01-10", freq="D")
        off = per.freq
        # 因为原始 dtype 为对象，所以期望结果的 dtype 也为对象
        expected = Series([9 * off, 8 * off], name="xxx", dtype=object)
        # 验证 ser 减去 per 后的结果是否等于 expected
        tm.assert_series_equal(per - ser, expected)
        # 验证 ser 减去 per 的相反数后的结果是否等于 -1 * expected
        tm.assert_series_equal(ser - per, -1 * expected)

        # 创建另一个包含两个 Period 对象的 Series，频率为每日
        s2 = Series(
            [Period("2015-01-05", freq="D"), Period("2015-01-04", freq="D")],
            name="xxx",
        )
        assert s2.dtype == "Period[D]"

        expected = Series([4 * off, 2 * off], name="xxx", dtype=object)
        # 验证 s2 减去 ser 后的结果是否等于 expected
        tm.assert_series_equal(s2 - ser, expected)
        # 验证 ser 减去 s2 的相反数后的结果是否等于 -1 * expected
        tm.assert_series_equal(ser - s2, -1 * expected)


class TestPeriodIndexSeriesMethods:
    """Test PeriodIndex and Period Series Ops consistency"""

    # 检查函数，用于验证 PeriodIndex 和 Series 上操作的一致性
    def _check(self, values, func, expected):
        # 创建 PeriodIndex 对象
        idx = PeriodIndex(values)
        # 执行指定的操作函数 func，并验证结果是否等于 expected
        result = func(idx)
        tm.assert_equal(result, expected)

        # 创建 Series 对象
        ser = Series(values)
        # 执行指定的操作函数 func，并验证结果是否等于 expected
        result = func(ser)
        exp = Series(expected, name=values.name)
        tm.assert_series_equal(result, exp)

    # 测试 PeriodIndex 上的各种操作
    def test_pi_ops(self):
        # 创建一个 PeriodIndex 对象，包含四个月份字符串，频率为每月
        idx = PeriodIndex(
            ["2011-01", "2011-02", "2011-03", "2011-04"], freq="M", name="idx"
        )

        # 期望结果的 PeriodIndex 对象，每个日期增加两个月
        expected = PeriodIndex(
            ["2011-03", "2011-04", "2011-05", "2011-06"], freq="M", name="idx"
        )

        # 调用 _check 函数，验证 idx + 2 的结果是否等于 expected
        self._check(idx, lambda x: x + 2, expected)
        # 调用 _check 函数，验证 2 + idx 的结果是否等于 expected
        self._check(idx, lambda x: 2 + x, expected)

        # 调用 _check 函数，验证 idx + 2 减去 2 的结果是否等于 idx
        self._check(idx + 2, lambda x: x - 2, idx)

        # 计算 idx 减去一个单独的 Period 对象后的结果
        result = idx - Period("2011-01", freq="M")
        off = idx.freq
        exp = pd.Index([0 * off, 1 * off, 2 * off, 3 * off], name="idx")
        tm.assert_index_equal(result, exp)

        # 计算一个单独的 Period 对象减去 idx 后的结果
        result = Period("2011-01", freq="M") - idx
        exp = pd.Index([0 * off, -1 * off, -2 * off, -3 * off], name="idx")
        tm.assert_index_equal(result, exp)

    @pytest.mark.parametrize("ng", ["str", 1.5])
    @pytest.mark.parametrize(
        "func",
        [  # 定义参数化测试的函数列表
            lambda obj, ng: obj + ng,  # 定义函数：对象加参数
            lambda obj, ng: ng + obj,  # 定义函数：参数加对象
            lambda obj, ng: obj - ng,  # 定义函数：对象减参数
            lambda obj, ng: ng - obj,  # 定义函数：参数减对象
            lambda obj, ng: np.add(obj, ng),  # 定义函数：使用NumPy进行对象加参数
            lambda obj, ng: np.add(ng, obj),  # 定义函数：使用NumPy进行参数加对象
            lambda obj, ng: np.subtract(obj, ng),  # 定义函数：使用NumPy进行对象减参数
            lambda obj, ng: np.subtract(ng, obj),  # 定义函数：使用NumPy进行参数减对象
        ],
    )
    def test_parr_ops_errors(self, ng, func, box_with_array):
        idx = PeriodIndex(
            ["2011-01", "2011-02", "2011-03", "2011-04"], freq="M", name="idx"
        )  # 创建一个周期索引对象
        obj = tm.box_expected(idx, box_with_array)  # 调用tm模块的box_expected函数，生成一个对象
        msg = "|".join(
            [  # 创建错误消息字符串，用于检查是否捕获到期望的异常
                r"unsupported operand type\(s\)",  # 操作数类型不支持的错误
                "can only concatenate",  # 只能进行连接的错误
                r"must be str",  # 必须是字符串的错误
                "object to str implicitly",  # 隐式转换为字符串的错误
            ]
        )

        with pytest.raises(TypeError, match=msg):  # 使用pytest断言捕获TypeError异常，并匹配错误消息
            func(obj, ng)  # 调用参数化函数进行测试

    def test_pi_ops_nat(self):
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2011-04"], freq="M", name="idx"
        )  # 创建一个周期索引对象，包含一个NaT（Not a Time）值
        expected = PeriodIndex(
            ["2011-03", "2011-04", "NaT", "2011-06"], freq="M", name="idx"
        )  # 创建预期的周期索引对象，期望结果

        self._check(idx, lambda x: x + 2, expected)  # 调用_check方法，验证加法操作的正确性
        self._check(idx, lambda x: 2 + x, expected)  # 调用_check方法，验证加法操作的正确性
        self._check(idx, lambda x: np.add(x, 2), expected)  # 调用_check方法，验证NumPy加法操作的正确性

        self._check(idx + 2, lambda x: x - 2, idx)  # 调用_check方法，验证减法操作的正确性
        self._check(idx + 2, lambda x: np.subtract(x, 2), idx)  # 调用_check方法，验证NumPy减法操作的正确性

        # freq with mult
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2011-04"], freq="2M", name="idx"
        )  # 创建一个周期索引对象，频率为每2个月
        expected = PeriodIndex(
            ["2011-07", "2011-08", "NaT", "2011-10"], freq="2M", name="idx"
        )  # 创建预期的周期索引对象，期望结果

        self._check(idx, lambda x: x + 3, expected)  # 调用_check方法，验证加法操作的正确性
        self._check(idx, lambda x: 3 + x, expected)  # 调用_check方法，验证加法操作的正确性
        self._check(idx, lambda x: np.add(x, 3), expected)  # 调用_check方法，验证NumPy加法操作的正确性

        self._check(idx + 3, lambda x: x - 3, idx)  # 调用_check方法，验证减法操作的正确性
        self._check(idx + 3, lambda x: np.subtract(x, 3), idx)  # 调用_check方法，验证NumPy减法操作的正确性

    def test_pi_ops_array_int(self):
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2011-04"], freq="M", name="idx"
        )  # 创建一个周期索引对象
        f = lambda x: x + np.array([1, 2, 3, 4])  # 创建一个函数，将周期索引对象与整数数组相加
        exp = PeriodIndex(
            ["2011-02", "2011-04", "NaT", "2011-08"], freq="M", name="idx"
        )  # 创建预期的周期索引对象，期望结果
        self._check(idx, f, exp)  # 调用_check方法，验证加法操作的正确性

        f = lambda x: np.add(x, np.array([4, -1, 1, 2]))  # 创建一个函数，将周期索引对象与整数数组相加
        exp = PeriodIndex(
            ["2011-05", "2011-01", "NaT", "2011-06"], freq="M", name="idx"
        )  # 创建预期的周期索引对象，期望结果
        self._check(idx, f, exp)  # 调用_check方法，验证加法操作的正确性

        f = lambda x: x - np.array([1, 2, 3, 4])  # 创建一个函数，将周期索引对象与整数数组相减
        exp = PeriodIndex(
            ["2010-12", "2010-12", "NaT", "2010-12"], freq="M", name="idx"
        )  # 创建预期的周期索引对象，期望结果
        self._check(idx, f, exp)  # 调用_check方法，验证减法操作的正确性

        f = lambda x: np.subtract(x, np.array([3, 2, 3, -2]))  # 创建一个函数，将周期索引对象与整数数组相减
        exp = PeriodIndex(
            ["2010-10", "2010-12", "NaT", "2011-06"], freq="M", name="idx"
        )  # 创建预期的周期索引对象，期望结果
        self._check(idx, f, exp)  # 调用_check方法，验证减法操作的正确性
    # 测试函数，用于验证 PeriodIndex 对象的偏移操作
    def test_pi_ops_offset(self):
        # 创建一个包含日期字符串的 PeriodIndex 对象，频率为每天
        idx = PeriodIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01", "2011-04-01"],
            freq="D",
            name="idx",
        )
        # 定义一个 lambda 函数，用于对每个元素进行加一天的操作
        f = lambda x: x + pd.offsets.Day()
        # 期望的结果 PeriodIndex 对象，日期每个元素加一天
        exp = PeriodIndex(
            ["2011-01-02", "2011-02-02", "2011-03-02", "2011-04-02"],
            freq="D",
            name="idx",
        )
        # 调用自定义函数 _check 进行验证
        self._check(idx, f, exp)

        # 定义一个 lambda 函数，对每个元素进行加两天的操作
        f = lambda x: x + pd.offsets.Day(2)
        # 期望的结果 PeriodIndex 对象，日期每个元素加两天
        exp = PeriodIndex(
            ["2011-01-03", "2011-02-03", "2011-03-03", "2011-04-03"],
            freq="D",
            name="idx",
        )
        # 调用自定义函数 _check 进行验证
        self._check(idx, f, exp)

        # 定义一个 lambda 函数，对每个元素进行减两天的操作
        f = lambda x: x - pd.offsets.Day(2)
        # 期望的结果 PeriodIndex 对象，日期每个元素减两天
        exp = PeriodIndex(
            ["2010-12-30", "2011-01-30", "2011-02-27", "2011-03-30"],
            freq="D",
            name="idx",
        )
        # 调用自定义函数 _check 进行验证
        self._check(idx, f, exp)

    # 测试函数，用于验证 PeriodIndex 对象在偏移操作中的错误处理
    def test_pi_offset_errors(self):
        # 创建一个包含日期字符串的 PeriodIndex 对象，频率为每天
        idx = PeriodIndex(
            ["2011-01-01", "2011-02-01", "2011-03-01", "2011-04-01"],
            freq="D",
            name="idx",
        )
        # 将 PeriodIndex 对象转换为 Series 对象
        ser = Series(idx)

        # 错误消息内容
        msg = (
            "Cannot add/subtract timedelta-like from PeriodArray that is not "
            "an integer multiple of the PeriodArray's freq"
        )

        # 对 idx 和 ser 中的对象执行操作，预期会引发 IncompatibleFrequency 异常，并匹配指定的错误消息
        for obj in [idx, ser]:
            with pytest.raises(IncompatibleFrequency, match=msg):
                obj + pd.offsets.Hour(2)

            with pytest.raises(IncompatibleFrequency, match=msg):
                pd.offsets.Hour(2) + obj

            with pytest.raises(IncompatibleFrequency, match=msg):
                obj - pd.offsets.Hour(2)

    # 测试函数，用于验证 PeriodIndex 对象在子期间操作中的行为
    def test_pi_sub_period(self):
        # 创建一个包含年月字符串的 PeriodIndex 对象，频率为每月
        idx = PeriodIndex(
            ["2011-01", "2011-02", "2011-03", "2011-04"], freq="M", name="idx"
        )

        # 测试 GH#13071 中的功能
        result = idx - Period("2012-01", freq="M")
        # 计算频率的偏移量
        off = idx.freq
        # 期望的结果为一个包含每个元素与指定期间之间偏移量的 Index 对象
        exp = pd.Index([-12 * off, -11 * off, -10 * off, -9 * off], name="idx")
        # 使用 assert_index_equal 进行验证
        tm.assert_index_equal(result, exp)

        # 使用 np.subtract 函数进行相同的计算，并验证结果
        result = np.subtract(idx, Period("2012-01", freq="M"))
        tm.assert_index_equal(result, exp)

        # 交换操作数的顺序，验证反向偏移的结果
        result = Period("2012-01", freq="M") - idx
        exp = pd.Index([12 * off, 11 * off, 10 * off, 9 * off], name="idx")
        tm.assert_index_equal(result, exp)

        # 使用 np.subtract 函数进行相同的计算，并验证结果
        result = np.subtract(Period("2012-01", freq="M"), idx)
        tm.assert_index_equal(result, exp)

        # 创建一个包含 NaN 的 TimedeltaIndex 对象，名称为 "idx"
        exp = TimedeltaIndex([np.nan, np.nan, np.nan, np.nan], name="idx")
        # 计算 idx 与 NaN 期间的偏移量，并验证结果
        result = idx - Period("NaT", freq="M")
        tm.assert_index_equal(result, exp)
        # 验证结果对象的频率是否与预期相同
        assert result.freq == exp.freq

        # 计算 NaN 期间与 idx 的偏移量，并验证结果
        result = Period("NaT", freq="M") - idx
        tm.assert_index_equal(result, exp)
        # 验证结果对象的频率是否与预期相同
        assert result.freq == exp.freq
    # 定义测试方法，用于测试期间索引与 NaT 相减的行为
    def test_pi_sub_pdnat(self):
        # 设置测试目的：验证 GitHub 问题 #13071 和 #19389
        idx = PeriodIndex(
            ["2011-01", "2011-02", "NaT", "2011-04"], freq="M", name="idx"
        )
        # 期望的结果：一个名称为 'idx' 的时间增量索引，包含四个 NaT
        exp = TimedeltaIndex([pd.NaT] * 4, name="idx")
        # 断言期间索引与 NaT 的减法结果与期望结果相等
        tm.assert_index_equal(pd.NaT - idx, exp)
        # 断言 NaT 与期间索引的减法结果与期望结果相等
        tm.assert_index_equal(idx - pd.NaT, exp)

    # 定义测试方法，用于测试期间索引与期间 NaT 相减的行为
    def test_pi_sub_period_nat(self):
        # 设置测试目的：验证 GitHub 问题 #13071
        idx = PeriodIndex(
            ["2011-01", "NaT", "2011-03", "2011-04"], freq="M", name="idx"
        )

        # 计算期间索引与特定期间对象之间的减法操作
        result = idx - Period("2012-01", freq="M")
        # 获取期间索引的频率
        off = idx.freq
        # 期望的结果：一个名称为 'idx' 的时间索引，包含四个相对偏移的值
        exp = pd.Index([-12 * off, pd.NaT, -10 * off, -9 * off], name="idx")
        # 断言结果与期望结果相等
        tm.assert_index_equal(result, exp)

        # 计算特定期间对象与期间索引之间的减法操作
        result = Period("2012-01", freq="M") - idx
        # 期望的结果：一个名称为 'idx' 的时间索引，包含四个相对偏移的值
        exp = pd.Index([12 * off, pd.NaT, 10 * off, 9 * off], name="idx")
        # 断言结果与期望结果相等
        tm.assert_index_equal(result, exp)

        # 期望的结果：一个名称为 'idx' 的时间增量索引，包含四个 NaN 值
        exp = TimedeltaIndex([np.nan, np.nan, np.nan, np.nan], name="idx")
        # 断言期间索引与特定 NaT 期间对象的减法结果与期望结果相等
        tm.assert_index_equal(idx - Period("NaT", freq="M"), exp)
        # 断言特定 NaT 期间对象与期间索引的减法结果与期望结果相等
        tm.assert_index_equal(Period("NaT", freq="M") - idx, exp)
```
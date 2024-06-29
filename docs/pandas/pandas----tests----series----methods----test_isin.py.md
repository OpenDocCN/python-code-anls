# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_isin.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，用于数据分析
from pandas import (  # 从 Pandas 中导入 Series 和 date_range 函数
    Series,
    date_range,
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块
from pandas.core import algorithms  # 导入 Pandas 核心算法模块
from pandas.core.arrays import PeriodArray  # 从 Pandas 数组模块中导入 PeriodArray 类


class TestSeriesIsIn:
    def test_isin(self):
        s = Series(["A", "B", "C", "a", "B", "B", "A", "C"])

        result = s.isin(["A", "C"])  # 检查 Series s 中是否包含列表 ["A", "C"] 中的元素
        expected = Series([True, False, True, False, False, False, True, True])
        tm.assert_series_equal(result, expected)  # 断言检查 result 是否等于 expected

        # GH#16012
        # This specific issue has to have a series over 1e6 in len, but the
        # comparison array (in_list) must be large enough so that numpy doesn't
        # do a manual masking trick that will avoid this issue altogether
        s = Series(list("abcdefghijk" * 10**5))  # 创建包含大量字符的 Series 对象
        # 如果 numpy 不执行手动比较/掩码操作，这些混合类型将导致 numpy 中的异常
        in_list = [-1, "a", "b", "G", "Y", "Z", "E", "K", "E", "S", "I", "R", "R"] * 6

        assert s.isin(in_list).sum() == 200000  # 断言检查 s 中与 in_list 相同的元素数量是否为 200000

    def test_isin_with_string_scalar(self):
        # GH#4763
        s = Series(["A", "B", "C", "a", "B", "B", "A", "C"])
        msg = (
            r"only list-like objects are allowed to be passed to isin\(\), "
            r"you passed a `str`"
        )
        with pytest.raises(TypeError, match=msg):  # 使用 pytest 检查是否抛出特定的 TypeError 异常
            s.isin("a")

        s = Series(["aaa", "b", "c"])
        with pytest.raises(TypeError, match=msg):  # 使用 pytest 检查是否抛出特定的 TypeError 异常
            s.isin("aaa")

    def test_isin_datetimelike_mismatched_reso(self):
        expected = Series([True, True, False, False, False])

        ser = Series(date_range("jan-01-2013", "jan-05-2013"))

        # fails on dtype conversion in the first place
        day_values = np.asarray(ser[0:2].values).astype("datetime64[D]")  # 将 ser 的前两个值转换为 datetime64[D] 类型
        result = ser.isin(day_values)  # 检查 ser 中是否包含 day_values 中的日期
        tm.assert_series_equal(result, expected)  # 断言检查 result 是否等于 expected

        dta = ser[:2]._values.astype("M8[s]")  # 将 ser 的前两个值转换为 "M8[s]" 类型的日期
        result = ser.isin(dta)  # 检查 ser 中是否包含 dta 中的日期
        tm.assert_series_equal(result, expected)  # 断言检查 result 是否等于 expected

    def test_isin_datetimelike_mismatched_reso_list(self):
        expected = Series([True, True, False, False, False])

        ser = Series(date_range("jan-01-2013", "jan-05-2013"))

        dta = ser[:2]._values.astype("M8[s]")  # 将 ser 的前两个值转换为 "M8[s]" 类型的日期
        result = ser.isin(list(dta))  # 检查 ser 中是否包含 dta 列表中的日期
        tm.assert_series_equal(result, expected)  # 断言检查 result 是否等于 expected
    def test_isin_with_i8(self):
        # 定义测试方法，用于测试 Series 的 isin 方法

        # 预期结果
        expected = Series([True, True, False, False, False])
        expected2 = Series([False, True, False, False, False])

        # 创建一个包含日期范围的 Series，datetime64[ns] 类型
        s = Series(date_range("jan-01-2013", "jan-05-2013"))

        # 使用 Series 的前两个元素测试 isin 方法，期望结果为 expected
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

        # 使用 Series 的前两个元素的值测试 isin 方法，期望结果为 expected
        result = s.isin(s[0:2].values)
        tm.assert_series_equal(result, expected)

        # 使用包含 Series 第二个元素的列表测试 isin 方法，期望结果为 expected2
        result = s.isin([s[1]])
        tm.assert_series_equal(result, expected2)

        # 使用包含 Series 第二个元素的 np.datetime64 对象测试 isin 方法，期望结果为 expected2
        result = s.isin([np.datetime64(s[1])])
        tm.assert_series_equal(result, expected2)

        # 使用 Series 的前两个元素创建的集合测试 isin 方法，期望结果为 expected
        result = s.isin(set(s[0:2]))
        tm.assert_series_equal(result, expected)

        # 创建一个包含 timedelta64[ns] 类型的 Series
        s = Series(pd.to_timedelta(range(5), unit="D"))
        # 使用 Series 的前两个元素测试 isin 方法，期望结果为 expected
        result = s.isin(s[0:2])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        # 定义测试空对象的 isin 方法

        # 创建一个包含字符串的 Series
        s = Series(["a", "b"])
        expected = Series([False, False])

        # 使用空对象 empty 测试 isin 方法，期望结果为 expected
        result = s.isin(empty)
        tm.assert_series_equal(expected, result)

    def test_isin_read_only(self):
        # 测试只读数组在 isin 方法中的使用

        # 创建一个普通的 numpy 数组 arr 和一个包含相同元素的 Series s
        arr = np.array([1, 2, 3])
        arr.setflags(write=False)
        s = Series([1, 2, 3])

        # 使用只读数组 arr 测试 Series s 的 isin 方法，期望结果为全是 True 的 Series
        result = s.isin(arr)
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [object, None])
    def test_isin_dt64_values_vs_ints(self, dtype):
        # 测试 datetime64 值和整数在 isin 方法中的比较

        # 创建一个日期范围的 DateTimeIndex 对象 dti 和一个包含该日期的 Series ser
        dti = date_range("2013-01-01", "2013-01-05")
        ser = Series(dti)

        # 创建一个包含整数值的 numpy 数组 comps
        comps = np.asarray([1356998400000000000], dtype=dtype)

        # 使用 comps 测试 dti 的 isin 方法，期望结果为全是 False 的 numpy 数组
        res = dti.isin(comps)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        # 使用 comps 测试 ser 的 isin 方法，期望结果为全是 False 的 Series
        res = ser.isin(comps)
        tm.assert_series_equal(res, Series(expected))

        # 使用 pd.core.algorithms.isin 函数测试 ser 和 comps，期望结果为全是 False 的 numpy 数组
        res = pd.core.algorithms.isin(ser, comps)
        tm.assert_numpy_array_equal(res, expected)

    def test_isin_tzawareness_mismatch(self):
        # 测试带有时区信息的日期范围对象在 isin 方法中的使用

        # 创建一个日期范围的 DateTimeIndex 对象 dti 和一个包含该日期的 Series ser
        dti = date_range("2013-01-01", "2013-01-05")
        ser = Series(dti)

        # 创建一个带有 UTC 时区的日期范围对象 other
        other = dti.tz_localize("UTC")

        # 使用 other 测试 dti 的 isin 方法，期望结果为全是 False 的 numpy 数组
        res = dti.isin(other)
        expected = np.array([False] * len(dti), dtype=bool)
        tm.assert_numpy_array_equal(res, expected)

        # 使用 other 测试 ser 的 isin 方法，期望结果为全是 False 的 Series
        res = ser.isin(other)
        tm.assert_series_equal(res, Series(expected))

        # 使用 pd.core.algorithms.isin 函数测试 ser 和 other，期望结果为全是 False 的 numpy 数组
        res = pd.core.algorithms.isin(ser, other)
        tm.assert_numpy_array_equal(res, expected)
    # 定义测试函数，用于测试不匹配周期频率的情况
    def test_isin_period_freq_mismatch(self):
        # 创建一个日期范围对象，从"2013-01-01"到"2013-01-05"
        dti = date_range("2013-01-01", "2013-01-05")
        # 将日期范围对象转换为以月份为周期的 PeriodIndex 对象
        pi = dti.to_period("M")
        # 将 PeriodIndex 对象转换为 Series 对象
        ser = Series(pi)

        # 构建另一个具有相同 i8 值但不同 dtype 的 PeriodIndex 对象
        dtype = dti.to_period("Y").dtype
        other = PeriodArray._simple_new(pi.asi8, dtype=dtype)

        # 判断 pi 中的元素是否存在于 other 中
        res = pi.isin(other)
        # 构建预期的结果数组，全为 False
        expected = np.array([False] * len(pi), dtype=bool)
        # 断言 res 与 expected 相等
        tm.assert_numpy_array_equal(res, expected)

        # 判断 ser 中的元素是否存在于 other 中
        res = ser.isin(other)
        # 构建预期的 Series 对象，元素为 expected
        tm.assert_series_equal(res, Series(expected))

        # 使用 pandas 的算法函数判断 ser 中的元素是否存在于 other 中
        res = pd.core.algorithms.isin(ser, other)
        # 断言 res 与 expected 相等
        tm.assert_numpy_array_equal(res, expected)

    # 使用 pytest 的参数化装饰器指定多组参数进行测试
    @pytest.mark.parametrize("values", [[-9.0, 0.0], [-9, 0]])
    # 定义测试函数，测试浮点数在整数序列中的情况
    def test_isin_float_in_int_series(self, values):
        # 创建一个 Series 对象，包含传入的 values
        ser = Series(values)
        # 判断 ser 中的元素是否存在于 [-9, -0.5] 中
        result = ser.isin([-9, -0.5])
        # 构建预期的 Series 对象，元素为 [True, False]
        expected = Series([True, False])
        # 断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)

    # 使用 pytest 的参数化装饰器指定多组参数进行测试
    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
    @pytest.mark.parametrize(
        "data,values,expected",
        [
            # 不同情况下的测试数据与预期结果
            ([0, 1, 0], [1], [False, True, False]),
            ([0, 1, 0], [1, pd.NA], [False, True, False]),
            ([0, pd.NA, 0], [1, 0], [True, False, True]),
            ([0, 1, pd.NA], [1, pd.NA], [False, True, True]),
            ([0, 1, pd.NA], [1, np.nan], [False, True, False]),
            ([0, pd.NA, pd.NA], [np.nan, pd.NaT, None], [False, False, False]),
        ],
    )
    # 定义测试函数，测试不同类型的遮蔽值
    def test_isin_masked_types(self, dtype, data, values, expected):
        # 创建一个指定 dtype 的 Series 对象
        ser = Series(data, dtype=dtype)

        # 判断 ser 中的元素是否存在于 values 中
        result = ser.isin(values)
        # 构建预期的 Series 对象，元素为 expected，且 dtype 为 "boolean"
        expected = Series(expected, dtype="boolean")

        # 断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试包含混合数据类型和 NaN 值的大型 Series 对象的 isin 方法
def test_isin_large_series_mixed_dtypes_and_nan(monkeypatch):
    # GitHub 上的 issue 链接，用于描述此测试函数的背景和相关问题
    # 对值使用对象数据类型和大于 _MINIMUM_COMP_ARR_LEN 元素的组合
    min_isin_comp = 5
    # 创建一个包含 1、2 和 NaN 值交替的 Series 对象，并重复 min_isin_comp 次
    ser = Series([1, 2, np.nan] * min_isin_comp)
    # 使用 monkeypatch 上下文，设置 algorithms 模块中的 _MINIMUM_COMP_ARR_LEN 为 min_isin_comp
    with monkeypatch.context() as m:
        m.setattr(algorithms, "_MINIMUM_COMP_ARR_LEN", min_isin_comp)
        # 调用 Series 的 isin 方法，检查其中是否包含 {"foo", "bar"} 中的值
        result = ser.isin({"foo", "bar"})
    # 创建预期结果，一个包含 False 的 Series 对象，长度为 3 * min_isin_comp
    expected = Series([False] * 3 * min_isin_comp)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试包含复数的 Series 对象的 isin 方法
def test_isin_complex_numbers():
    # GitHub 上的 issue 17927 链接，用于描述此测试函数的背景和相关问题
    # 创建一个包含复数的列表
    array = [0, 1j, 1j, 1, 1 + 1j, 1 + 2j, 1 + 1j]
    # 调用 Series 的 isin 方法，检查其中是否包含 [1j, 1 + 1j, 1 + 2j] 中的值
    result = Series(array).isin([1j, 1 + 1j, 1 + 2j])
    # 创建预期结果，一个包含布尔值的 Series 对象
    expected = Series([False, True, True, False, True, True, True], dtype=bool)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，定义一个参数化测试函数，用于测试包含混合对象类型的数据的 isin 方法
@pytest.mark.parametrize(
    "data,is_in",
    [([1, [2]], [1]), (["simple str", [{"values": 3}]], ["simple str"])],
)
def test_isin_filtering_with_mixed_object_types(data, is_in):
    # GitHub 上的 issue 20883 链接，用于描述此测试函数的背景和相关问题

    # 创建一个包含 data 的 Series 对象
    ser = Series(data)
    # 调用 Series 的 isin 方法，检查其中是否包含 is_in 中的值
    result = ser.isin(is_in)
    # 创建预期结果，一个包含布尔值的 Series 对象
    expected = Series([True, False])

    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 使用 pytest 的 parametrize 装饰器，定义一个参数化测试函数，用于测试在可迭代对象上使用 isin 方法的过滤功能
@pytest.mark.parametrize("data", [[1, 2, 3], [1.0, 2.0, 3.0]])
@pytest.mark.parametrize("isin", [[1, 2], [1.0, 2.0]])
def test_isin_filtering_on_iterable(data, isin):
    # GitHub 上的 issue 50234 链接，用于描述此测试函数的背景和相关问题

    # 创建一个包含 data 的 Series 对象
    ser = Series(data)
    # 调用 Series 的 isin 方法，检查其中是否包含 isin 中的值
    result = ser.isin(i for i in isin)
    # 创建预期结果，一个包含布尔值的 Series 对象
    expected_result = Series([True, True, False])

    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected_result)
```
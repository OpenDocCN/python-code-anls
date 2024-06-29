# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\test_interval.py`

```
# 导入操作符模块，用于进行比较操作
import operator

# 导入 NumPy 库，用于处理数组和数值计算
import numpy as np

# 导入 Pytest 库，用于编写和运行测试
import pytest

# 从 pandas 库中导入常用的数据类型检查函数
from pandas.core.dtypes.common import is_list_like

# 导入 pandas 库，并将其命名为 pd，用于数据处理和分析
import pandas as pd

# 从 pandas 中导入特定模块和类，包括数据结构和时间序列处理相关的对象
from pandas import (
    Categorical,
    Index,
    Interval,
    IntervalIndex,
    Period,
    Series,
    Timedelta,
    Timestamp,
    date_range,
    period_range,
    timedelta_range,
)

# 导入 pandas 内部测试模块，用于测试和验证
import pandas._testing as tm

# 从 pandas.core.arrays 中导入特定数据结构，如布尔数组和区间数组
from pandas.core.arrays import (
    BooleanArray,
    IntervalArray,
)

# 从 pandas.tests.arithmetic.common 中导入一个特定的测试函数
from pandas.tests.arithmetic.common import get_upcast_box


@pytest.fixture(
    params=[
        (Index([0, 2, 4, 4]), Index([1, 3, 5, 8])),  # 定义参数化测试用例，每个元组包含两个索引对象
        (Index([0.0, 1.0, 2.0, np.nan]), Index([1.0, 2.0, 3.0, np.nan])),  # 包含浮点数和 NaN 的索引对象
        (
            timedelta_range("0 days", periods=3).insert(3, pd.NaT),  # 时间间隔范围对象，插入 NaN 值
            timedelta_range("1 day", periods=3).insert(3, pd.NaT),  # 另一个时间间隔范围对象，插入 NaN 值
        ),
        (
            date_range("20170101", periods=3).insert(3, pd.NaT),  # 日期范围对象，插入 NaN 值
            date_range("20170102", periods=3).insert(3, pd.NaT),  # 另一个日期范围对象，插入 NaN 值
        ),
        (
            date_range("20170101", periods=3, tz="US/Eastern").insert(3, pd.NaT),  # 带时区的日期范围对象，插入 NaN 值
            date_range("20170102", periods=3, tz="US/Eastern").insert(3, pd.NaT),  # 另一个带时区的日期范围对象，插入 NaN 值
        ),
    ],
    ids=lambda x: str(x[0].dtype),  # 参数化测试用例的标识函数，返回第一个参数的数据类型字符串表示
)
def left_right_dtypes(request):
    """
    Fixture for building an IntervalArray from various dtypes
    构建不同数据类型的 IntervalArray 的测试夹具
    """
    return request.param


@pytest.fixture
def interval_array(left_right_dtypes):
    """
    Fixture to generate an IntervalArray of various dtypes containing NA if possible
    生成包含各种数据类型和可能包含 NA 的 IntervalArray 的测试夹具
    """
    left, right = left_right_dtypes
    return IntervalArray.from_arrays(left, right)


def create_categorical_intervals(left, right, closed="right"):
    """
    Function to create categorical intervals using given left and right arrays
    使用给定的左右数组创建分类间隔的函数
    """
    return Categorical(IntervalIndex.from_arrays(left, right, closed))


def create_series_intervals(left, right, closed="right"):
    """
    Function to create Series of intervals using given left and right arrays
    使用给定的左右数组创建间隔 Series 的函数
    """
    return Series(IntervalArray.from_arrays(left, right, closed))


def create_series_categorical_intervals(left, right, closed="right"):
    """
    Function to create Series of categorical intervals using given left and right arrays
    使用给定的左右数组创建分类间隔 Series 的函数
    """
    return Series(Categorical(IntervalIndex.from_arrays(left, right, closed)))


class TestComparison:
    @pytest.fixture(params=[operator.eq, operator.ne])
    def op(self, request):
        """
        Fixture to parameterize with equality and inequality comparison operators
        参数化测试用例，包含相等和不相等比较操作符的夹具
        """
        return request.param

    @pytest.fixture(
        params=[
            IntervalArray.from_arrays,
            IntervalIndex.from_arrays,
            create_categorical_intervals,
            create_series_intervals,
            create_series_categorical_intervals,
        ],
        ids=[
            "IntervalArray",
            "IntervalIndex",
            "Categorical[Interval]",
            "Series[Interval]",
            "Series[Categorical[Interval]]",
        ],
    )
    def interval_constructor(self, request):
        """
        Fixture for all pandas native interval constructors.
        To be used as the LHS of IntervalArray comparisons.
        所有 pandas 原生间隔构造函数的夹具，用作 IntervalArray 比较的左操作数
        """
        return request.param
    # 定义一个方法，用于在数组和另一个对象之间进行逐元素比较
    def elementwise_comparison(self, op, interval_array, other):
        """
        Helper that performs elementwise comparisons between `array` and `other`
        """
        # 如果 `other` 不是类列表的对象，则将其转换为长度与 `interval_array` 相同的列表
        other = other if is_list_like(other) else [other] * len(interval_array)
        # 创建一个包含逐元素比较结果的 numpy 数组
        expected = np.array([op(x, y) for x, y in zip(interval_array, other)])
        # 如果 `other` 是 Series 对象，则返回一个带有相同索引的 Series 对象
        if isinstance(other, Series):
            return Series(expected, index=other.index)
        # 否则返回 numpy 数组
        return expected

    # 测试在标量和区间数组之间进行比较
    def test_compare_scalar_interval(self, op, interval_array):
        # 匹配第一个区间
        other = interval_array[0]
        # 执行操作并获取结果
        result = op(interval_array, other)
        # 通过辅助函数获取期望结果
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试框架验证结果与期望值是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 在单个端点上匹配，但不是在两个端点上
        other = Interval(interval_array.left[0], interval_array.right[1])
        # 执行操作并获取结果
        result = op(interval_array, other)
        # 通过辅助函数获取期望结果
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试框架验证结果与期望值是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试在混合闭合标量和区间数组之间进行比较
    def test_compare_scalar_interval_mixed_closed(self, op, closed, other_closed):
        # 创建一个区间数组
        interval_array = IntervalArray.from_arrays(range(2), range(1, 3), closed=closed)
        # 创建一个区间对象
        other = Interval(0, 1, closed=other_closed)

        # 执行操作并获取结果
        result = op(interval_array, other)
        # 通过辅助函数获取期望结果
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试框架验证结果与期望值是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试在标量和包含 NA 值的区间数组之间进行比较
    def test_compare_scalar_na(self, op, interval_array, nulls_fixture, box_with_array):
        # 获取包含 NA 值的对象
        box = box_with_array
        # 封装区间数组和包含 NA 值的对象
        obj = tm.box_expected(interval_array, box)
        # 执行操作并获取结果
        result = op(obj, nulls_fixture)

        # 如果 nulls_fixture 是 pd.NA，则根据 GH#31882 的规定返回一个全为 True 的布尔数组
        if nulls_fixture is pd.NA:
            exp = np.ones(interval_array.shape, dtype=bool)
            expected = BooleanArray(exp, exp)
        else:
            # 否则通过辅助函数获取期望结果
            expected = self.elementwise_comparison(op, interval_array, nulls_fixture)

        # 如果不是在 Index 类和 pd.NA 之间的比较，则获取升级后的期望结果
        if not (box is Index and nulls_fixture is pd.NA):
            xbox = get_upcast_box(obj, nulls_fixture, True)
            expected = tm.box_expected(expected, xbox)

        # 使用测试框架验证结果与期望值是否相等
        tm.assert_equal(result, expected)

        # 反向操作并获取结果
        rev = op(nulls_fixture, obj)
        # 使用测试框架验证反向结果与期望值是否相等
        tm.assert_equal(rev, expected)

    # 参数化测试，用于在标量和其他类型对象之间进行比较
    @pytest.mark.parametrize(
        "other",
        [
            0,
            1.0,
            True,
            "foo",
            Timestamp("2017-01-01"),
            Timestamp("2017-01-01", tz="US/Eastern"),
            Timedelta("0 days"),
            Period("2017-01-01", "D"),
        ],
    )
    def test_compare_scalar_other(self, op, interval_array, other):
        # 执行操作并获取结果
        result = op(interval_array, other)
        # 通过辅助函数获取期望结果
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试框架验证结果与期望值是否相等
        tm.assert_numpy_array_equal(result, expected)
    # 定义一个测试方法，用于比较类似于列表的区间对象的行为
    def test_compare_list_like_interval(self, op, interval_array, interval_constructor):
        # 测试区间端点相同的情况
        other = interval_constructor(interval_array.left, interval_array.right)
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试工具方法验证结果是否符合预期
        tm.assert_equal(result, expected)

        # 测试区间端点不同的情况
        other = interval_constructor(
            interval_array.left[::-1], interval_array.right[::-1]
        )
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试工具方法验证结果是否符合预期
        tm.assert_equal(result, expected)

        # 测试所有端点均为 NaN 的情况
        other = interval_constructor([np.nan] * 4, [np.nan] * 4)
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试工具方法验证结果是否符合预期
        tm.assert_equal(result, expected)

    # 定义一个测试方法，用于比较混合闭合类型的类似于列表的区间对象的行为
    def test_compare_list_like_interval_mixed_closed(
        self, op, interval_constructor, closed, other_closed
    ):
        # 创建一个包含指定范围和闭合类型的区间数组对象
        interval_array = IntervalArray.from_arrays(range(2), range(1, 3), closed=closed)
        # 创建另一个具有相同范围和不同闭合类型的区间对象
        other = interval_constructor(range(2), range(1, 3), closed=other_closed)

        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试工具方法验证结果是否符合预期
        tm.assert_equal(result, expected)

    # 使用 pytest 的参数化装饰器，为对象类型的比较方法定义多组参数化测试
    @pytest.mark.parametrize(
        "other",
        [
            (
                Interval(0, 1),
                Interval(Timedelta("1 day"), Timedelta("2 days")),
                Interval(4, 5, "both"),
                Interval(10, 20, "neither"),
            ),
            (0, 1.5, Timestamp("20170103"), np.nan),
            (
                Timestamp("20170102", tz="US/Eastern"),
                Timedelta("2 days"),
                "baz",
                pd.NaT,
            ),
        ],
    )
    # 定义一个测试方法，用于比较类似于对象的行为
    def test_compare_list_like_object(self, op, interval_array, other):
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试工具方法验证结果是否符合预期
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法，用于比较类似于 NaN 的行为
    def test_compare_list_like_nan(self, op, interval_array, nulls_fixture):
        # 创建一个包含 NaN 的对象列表
        other = [nulls_fixture] * 4
        result = op(interval_array, other)
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用测试工具方法验证结果是否符合预期
        tm.assert_equal(result, expected)
    @pytest.mark.parametrize(
        "other",
        [
            np.arange(4, dtype="int64"),  # 使用 NumPy 创建一个 int64 类型的数组
            np.arange(4, dtype="float64"),  # 使用 NumPy 创建一个 float64 类型的数组
            date_range("2017-01-01", periods=4),  # 使用 pandas 创建一个日期范围，包含 4 个日期
            date_range("2017-01-01", periods=4, tz="US/Eastern"),  # 使用 pandas 创建一个带时区的日期范围，包含 4 个日期
            timedelta_range("0 days", periods=4),  # 使用 pandas 创建一个时间增量范围，包含 4 个增量
            period_range("2017-01-01", periods=4, freq="D"),  # 使用 pandas 创建一个日期周期范围，每天频率，包含 4 个周期
            Categorical(list("abab")),  # 使用 pandas 创建一个分类数据，包含字符列表 'abab'
            Categorical(date_range("2017-01-01", periods=4)),  # 使用 pandas 创建一个分类数据，包含日期范围的分类
            pd.array(list("abcd")),  # 使用 pandas 创建一个包含字符列表 'abcd' 的数组
            pd.array(["foo", 3.14, None, object()], dtype=object),  # 使用 pandas 创建一个包含不同类型对象的数组
        ],
        ids=lambda x: str(x.dtype),  # 设置参数化测试的标识符函数为数据类型的字符串表示
    )
    def test_compare_list_like_other(self, op, interval_array, other):
        # 进行操作 op 在 interval_array 和 other 上
        result = op(interval_array, other)
        # 调用自定义方法 elementwise_comparison 获取预期结果
        expected = self.elementwise_comparison(op, interval_array, other)
        # 使用 assert_numpy_array_equal 断言 result 和 expected 数组相等
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("length", [1, 3, 5])
    @pytest.mark.parametrize("other_constructor", [IntervalArray, list])
    def test_compare_length_mismatch_errors(self, op, other_constructor, length):
        # 创建一个 IntervalArray 对象
        interval_array = IntervalArray.from_arrays(range(4), range(1, 5))
        # 使用指定构造函数创建一个 other 对象，长度由 length 参数确定
        other = other_constructor([Interval(0, 1)] * length)
        # 使用 pytest 断言，验证 op 在 interval_array 和 other 上抛出 ValueError 异常，异常信息包含 "Lengths must match to compare"
        with pytest.raises(ValueError, match="Lengths must match to compare"):
            op(interval_array, other)

    @pytest.mark.parametrize(
        "constructor, expected_type, assert_func",
        [
            (IntervalIndex, np.array, tm.assert_numpy_array_equal),  # 设置 constructor 为 IntervalIndex，expected_type 为 np.array，assert_func 为 assert_numpy_array_equal
            (Series, Series, tm.assert_series_equal),  # 设置 constructor 和 expected_type 均为 Series，assert_func 为 assert_series_equal
        ],
    )
    def test_index_series_compat(self, op, constructor, expected_type, assert_func):
        # 创建一个 breaks 列表
        breaks = range(4)
        # 使用指定构造函数创建一个 IntervalIndex 对象
        index = constructor(IntervalIndex.from_breaks(breaks))

        # 标量比较
        other = index[0]
        # 执行 op 操作，得到 result
        result = op(index, other)
        # 使用 elementwise_comparison 方法获取预期结果，然后转换为 expected_type 类型
        expected = expected_type(self.elementwise_comparison(op, index, other))
        # 使用 assert_func 断言 result 和 expected 相等
        assert_func(result, expected)

        other = breaks[0]
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

        # 列表比较
        other = IntervalArray.from_breaks(breaks)
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

        other = [index[0], breaks[0], "foo"]
        result = op(index, other)
        expected = expected_type(self.elementwise_comparison(op, index, other))
        assert_func(result, expected)

    @pytest.mark.parametrize("scalars", ["a", False, 1, 1.0, None])
    def test_comparison_operations(self, scalars):
        # GH #28981
        expected = Series([False, False])
        s = Series([Interval(0, 1), Interval(1, 2)], dtype="interval")
        # 执行比较操作，得到结果 result
        result = s == scalars
        # 使用 assert_series_equal 断言 result 和 expected 系列相等
        tm.assert_series_equal(result, expected)
```
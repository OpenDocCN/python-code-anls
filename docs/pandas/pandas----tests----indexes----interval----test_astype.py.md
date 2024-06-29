# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_astype.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy模块
import pytest  # 导入pytest测试框架

from pandas.core.dtypes.dtypes import (  # 从pandas中导入数据类型相关模块
    CategoricalDtype,
    IntervalDtype,
)

from pandas import (  # 从pandas中导入多个类和函数
    CategoricalIndex,
    Index,
    IntervalIndex,
    NaT,
    Timedelta,
    Timestamp,
    interval_range,
)
import pandas._testing as tm  # 导入pandas测试模块


class AstypeTests:
    """Tests common to IntervalIndex with any subtype"""

    def test_astype_idempotent(self, index):
        # 测试类型转换为"interval"后结果是否与原索引相等
        result = index.astype("interval")
        tm.assert_index_equal(result, index)

        # 测试类型转换为当前索引的数据类型后结果是否与原索引相等
        result = index.astype(index.dtype)
        tm.assert_index_equal(result, index)

    def test_astype_object(self, index):
        # 测试类型转换为"object"后结果是否符合预期
        result = index.astype(object)
        expected = Index(index.values, dtype="object")
        tm.assert_index_equal(result, expected)
        assert not result.equals(index)  # 确保转换后的结果与原索引不相等

    def test_astype_category(self, index):
        # 测试类型转换为"category"后结果是否符合预期
        result = index.astype("category")
        expected = CategoricalIndex(index.values)
        tm.assert_index_equal(result, expected)

        # 测试类型转换为CategoricalDtype后结果是否符合预期
        result = index.astype(CategoricalDtype())
        tm.assert_index_equal(result, expected)

        # 测试带有非默认参数的类型转换是否符合预期
        categories = index.dropna().unique().values[:-1]
        dtype = CategoricalDtype(categories=categories, ordered=True)
        result = index.astype(dtype)
        expected = CategoricalIndex(index.values, categories=categories, ordered=True)
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "dtype",
        [
            "int64",
            "uint64",
            "float64",
            "complex128",
            "period[M]",
            "timedelta64",
            "timedelta64[ns]",
            "datetime64",
            "datetime64[ns]",
            "datetime64[ns, US/Eastern]",
        ],
    )
    def test_astype_cannot_cast(self, index, dtype):
        # 测试无法将IntervalIndex转换为指定dtype时是否会抛出TypeError异常
        msg = "Cannot cast IntervalIndex to dtype"
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    def test_astype_invalid_dtype(self, index):
        # 测试当指定无效dtype时是否会抛出TypeError异常
        msg = "data type [\"']fake_dtype[\"'] not understood"
        with pytest.raises(TypeError, match=msg):
            index.astype("fake_dtype")


class TestIntSubtype(AstypeTests):
    """Tests specific to IntervalIndex with integer-like subtype"""

    indexes = [
        IntervalIndex.from_breaks(np.arange(-10, 11, dtype="int64")),
        IntervalIndex.from_breaks(np.arange(100, dtype="uint64"), closed="left"),
    ]

    @pytest.fixture(params=indexes)
    def index(self, request):
        return request.param

    @pytest.mark.parametrize(
        "subtype", ["float64", "datetime64[ns]", "timedelta64[ns]"]
    )
    def test_subtype_conversion(self, index, subtype):
        # 测试子类型转换是否符合预期
        dtype = IntervalDtype(subtype, index.closed)
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype), index.right.astype(subtype), closed=index.closed
        )
        tm.assert_index_equal(result, expected)
    # 使用 pytest 的参数化装饰器标记测试函数，提供多组参数进行测试
    @pytest.mark.parametrize(
        "subtype_start, subtype_end", [("int64", "uint64"), ("uint64", "int64")]
    )
    # 定义测试子类型整数的函数，接受起始和结束子类型作为参数
    def test_subtype_integer(self, subtype_start, subtype_end):
        # 使用给定的起始子类型创建区间索引
        index = IntervalIndex.from_breaks(np.arange(100, dtype=subtype_start))
        # 使用给定的结束子类型和闭合类型创建区间数据类型
        dtype = IntervalDtype(subtype_end, index.closed)
        # 将区间索引转换为指定的数据类型
        result = index.astype(dtype)
        # 根据给定的类型转换后，创建预期的区间索引
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype_end),
            index.right.astype(subtype_end),
            closed=index.closed,
        )
        # 使用测试工具库比较结果和预期的索引对象是否相等
        tm.assert_index_equal(result, expected)

    # 使用 pytest 的标记来表示测试预期失败，提供失败原因
    @pytest.mark.xfail(reason="GH#15832")
    # 定义测试整数子类型中的错误情况的函数
    def test_subtype_integer_errors(self):
        # 创建包含负值的区间范围，测试 int64 转换为 uint64 的情况
        index = interval_range(-10, 10)
        # 使用指定的数据类型创建区间数据类型对象
        dtype = IntervalDtype("uint64", "right")

        # 在确定异常消息之前，断言不应该收到某些消息
        # 断言不应该收到 -10 被转换为大正整数的消息
        msg = "^(?!(left side of interval must be <= right side))"
        # 使用 pytest 的上下文管理器检查是否抛出预期的值错误，并匹配指定的消息
        with pytest.raises(ValueError, match=msg):
            index.astype(dtype)
class TestFloatSubtype(AstypeTests):
    """Tests specific to IntervalIndex with float subtype"""

    # 定义包含不同浮点数区间的索引列表
    indexes = [
        interval_range(-10.0, 10.0, closed="neither"),  # 从-10.0到10.0的区间，两端不闭合
        IntervalIndex.from_arrays(
            [-1.5, np.nan, 0.0, 0.0, 1.5], [-0.5, np.nan, 1.0, 1.0, 3.0], closed="both"
        ),  # 根据给定的左右边界数组创建区间索引，左闭右闭
    ]

    @pytest.fixture(params=indexes)
    def index(self, request):
        # 为测试方法提供不同的索引作为参数
        return request.param

    @pytest.mark.parametrize("subtype", ["int64", "uint64"])
    def test_subtype_integer(self, subtype):
        index = interval_range(0.0, 10.0)
        dtype = IntervalDtype(subtype, "right")
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype), index.right.astype(subtype), closed=index.closed
        )
        # 断言转换后的索引是否与预期相等
        tm.assert_index_equal(result, expected)

        # 当存在NA值或无穷大时，预期引发值错误异常
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(ValueError, match=msg):
            index.insert(0, np.nan).astype(dtype)

    @pytest.mark.parametrize("subtype", ["int64", "uint64"])
    def test_subtype_integer_with_non_integer_borders(self, subtype):
        index = interval_range(0.0, 3.0, freq=0.25)
        dtype = IntervalDtype(subtype, "right")
        result = index.astype(dtype)
        expected = IntervalIndex.from_arrays(
            index.left.astype(subtype), index.right.astype(subtype), closed=index.closed
        )
        # 断言转换后的索引是否与预期相等
        tm.assert_index_equal(result, expected)

    def test_subtype_integer_errors(self):
        # 测试float64到uint64的转换，如果存在负数值则应该失败
        index = interval_range(-10.0, 10.0)
        dtype = IntervalDtype("uint64", "right")
        msg = re.escape(
            "Cannot convert interval[float64, right] to interval[uint64, right]; "
            "subtypes are incompatible"
        )
        # 预期引发类型错误异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    @pytest.mark.parametrize("subtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_subtype_datetimelike(self, index, subtype):
        dtype = IntervalDtype(subtype, "right")
        msg = "Cannot convert .* to .*; subtypes are incompatible"
        # 预期引发类型错误异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)


class TestDatetimelikeSubtype(AstypeTests):
    """Tests specific to IntervalIndex with datetime-like subtype"""

    # 定义包含不同日期时间类似子类型的索引列表
    indexes = [
        interval_range(Timestamp("2018-01-01"), periods=10, closed="neither"),
        interval_range(Timestamp("2018-01-01"), periods=10).insert(2, NaT),
        interval_range(Timestamp("2018-01-01", tz="US/Eastern"), periods=10),
        interval_range(Timedelta("0 days"), periods=10, closed="both"),
        interval_range(Timedelta("0 days"), periods=10).insert(2, NaT),
    ]

    @pytest.fixture(params=indexes)
    def index(self, request):
        # 为测试方法提供不同的索引作为参数
        return request.param

    @pytest.mark.parametrize("subtype", ["int64", "uint64"])
    # 测试将索引转换为指定子类型的整数类型
    def test_subtype_integer(self, index, subtype):
        # 创建一个指定子类型和闭合方式的区间数据类型
        dtype = IntervalDtype(subtype, "right")

        # 如果子类型不是"int64"，则抛出类型错误异常
        if subtype != "int64":
            msg = (
                r"Cannot convert interval\[(timedelta64|datetime64)\[ns.*\], .*\] "
                r"to interval\[uint64, .*\]"
            )
            with pytest.raises(TypeError, match=msg):
                index.astype(dtype)
            return

        # 将索引转换为指定的区间数据类型
        result = index.astype(dtype)
        # 将索引的左端点和右端点分别转换为指定的子类型
        new_left = index.left.astype(subtype)
        new_right = index.right.astype(subtype)

        # 创建一个预期的区间索引对象
        expected = IntervalIndex.from_arrays(new_left, new_right, closed=index.closed)
        # 断言转换后的结果与预期结果相等
        tm.assert_index_equal(result, expected)

    # 测试将索引转换为指定子类型的浮点数类型
    def test_subtype_float(self, index):
        # 创建一个指定子类型和闭合方式的区间数据类型
        dtype = IntervalDtype("float64", "right")
        msg = "Cannot convert .* to .*; subtypes are incompatible"
        # 如果子类型不兼容，则抛出类型错误异常
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    # 测试将索引转换为日期时间类型时的异常情况
    def test_subtype_datetimelike(self):
        # datetime -> timedelta 会引发异常
        dtype = IntervalDtype("timedelta64[ns]", "right")
        msg = "Cannot convert .* to .*; subtypes are incompatible"

        # 创建一个时间戳索引
        index = interval_range(Timestamp("2018-01-01"), periods=10)
        # 尝试将索引转换为指定的区间数据类型，预期会抛出类型错误异常
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

        # 创建一个带时区的时间戳索引
        index = interval_range(Timestamp("2018-01-01", tz="CET"), periods=10)
        # 尝试将索引转换为指定的区间数据类型，预期会抛出类型错误异常
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

        # timedelta -> datetime 会引发异常
        dtype = IntervalDtype("datetime64[ns]", "right")
        # 创建一个时间间隔索引
        index = interval_range(Timedelta("0 days"), periods=10)
        # 尝试将索引转换为指定的区间数据类型，预期会抛出类型错误异常
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)
```
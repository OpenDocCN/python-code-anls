# `D:\src\scipysrc\pandas\pandas\tests\arrays\interval\test_interval.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

import pandas as pd  # 导入Pandas库，用于数据分析
from pandas import (  # 导入Pandas中的特定模块和函数
    Index,  # 导入索引对象
    Interval,  # 导入区间对象
    IntervalIndex,  # 导入区间索引对象
    Timedelta,  # 导入时间增量对象
    Timestamp,  # 导入时间戳对象
    date_range,  # 导入日期范围生成函数
    timedelta_range,  # 导入时间增量范围生成函数
)
import pandas._testing as tm  # 导入Pandas内部测试工具
from pandas.core.arrays import IntervalArray  # 导入Pandas的区间数组对象


@pytest.fixture(  # 定义pytest的fixture，用于测试数据生成
    params=[  # 参数化测试用例
        (Index([0, 2, 4]), Index([1, 3, 5])),  # 创建整数索引对
        (Index([0.0, 1.0, 2.0]), Index([1.0, 2.0, 3.0])),  # 创建浮点数索引对
        (timedelta_range("0 days", periods=3), timedelta_range("1 day", periods=3)),  # 创建时间增量范围对
        (date_range("20170101", periods=3), date_range("20170102", periods=3)),  # 创建日期范围对
        (
            date_range("20170101", periods=3, tz="US/Eastern"),  # 创建带时区的日期范围对
            date_range("20170102", periods=3, tz="US/Eastern"),
        ),
    ],
    ids=lambda x: str(x[0].dtype),  # 为每组参数设置标识符
)
def left_right_dtypes(request):
    """
    Fixture for building an IntervalArray from various dtypes
    构建不同数据类型的IntervalArray的fixture
    """
    return request.param  # 返回参数化的测试数据


class TestAttributes:  # 定义测试类TestAttributes
    @pytest.mark.parametrize(  # 参数化测试用例
        "left, right",  # 参数名
        [  # 参数列表
            (0, 1),  # 整数对
            (Timedelta("0 days"), Timedelta("1 day")),  # 时间增量对
            (Timestamp("2018-01-01"), Timestamp("2018-01-02")),  # 时间戳对
            (
                Timestamp("2018-01-01", tz="US/Eastern"),  # 带时区的时间戳对
                Timestamp("2018-01-02", tz="US/Eastern"),
            ),
        ],
    )
    @pytest.mark.parametrize("constructor", [IntervalArray, IntervalIndex])  # 参数化构造函数
    def test_is_empty(self, constructor, left, right, closed):
        # GH27219
        tuples = [(left, left), (left, right), np.nan]  # 创建元组列表
        expected = np.array([closed != "both", False, False])  # 创建预期结果数组
        result = constructor.from_tuples(tuples, closed=closed).is_empty  # 使用构造函数创建IntervalArray并检查是否为空
        tm.assert_numpy_array_equal(result, expected)  # 断言结果与预期是否一致


class TestMethods:  # 定义测试类TestMethods
    def test_set_closed(self, closed, other_closed):
        # GH 21670
        array = IntervalArray.from_breaks(range(10), closed=closed)  # 使用区间断点创建IntervalArray对象
        result = array.set_closed(other_closed)  # 设置区间的闭合方式
        expected = IntervalArray.from_breaks(range(10), closed=other_closed)  # 预期的IntervalArray对象
        tm.assert_extension_array_equal(result, expected)  # 断言结果与预期是否一致

    @pytest.mark.parametrize(
        "other",
        [
            Interval(0, 1, closed="right"),  # 创建闭合为右侧的区间对象
            IntervalArray.from_breaks([1, 2, 3, 4], closed="right"),  # 创建闭合为右侧的区间数组对象
        ],
    )
    def test_where_raises(self, other):
        # GH#45768 The IntervalArray methods raises; the Series method coerces
        ser = pd.Series(IntervalArray.from_breaks([1, 2, 3, 4], closed="left"))  # 创建左侧闭合的区间数组，并转换为Series对象
        mask = np.array([True, False, True])  # 创建布尔掩码
        match = "'value.closed' is 'right', expected 'left'."  # 匹配错误信息
        with pytest.raises(ValueError, match=match):  # 断言引发特定错误和匹配的错误信息
            ser.array._where(mask, other)  # 使用区间数组的方法执行条件筛选

        res = ser.where(mask, other=other)  # 使用Series对象的where方法执行条件筛选
        expected = ser.astype(object).where(mask, other)  # 预期的筛选结果
        tm.assert_series_equal(res, expected)  # 断言Series对象的筛选结果与预期是否一致
    # 测试 IntervalArray 的 shift() 方法
    def test_shift(self):
        # 引用的 GitHub 问题链接和问题编号
        # 创建一个 IntervalArray，从给定的断点创建
        a = IntervalArray.from_breaks([1, 2, 3])
        # 对 IntervalArray 进行 shift 操作，生成结果
        result = a.shift()
        # 期望的结果是一个 IntervalArray，将整数转换为浮点数
        expected = IntervalArray.from_tuples([(np.nan, np.nan), (1.0, 2.0)])
        # 使用测试工具比较两个 IntervalArray 是否相等
        tm.assert_interval_array_equal(result, expected)

        # 设置错误消息
        msg = "can only insert Interval objects and NA into an IntervalArray"
        # 使用 pytest 检查是否抛出特定类型的错误，并且错误消息匹配特定的文本
        with pytest.raises(TypeError, match=msg):
            # 测试在 shift 操作中插入非法值是否会引发预期的错误
            a.shift(1, fill_value=pd.NaT)

    # 测试 IntervalArray 的 shift() 方法处理日期时间的情况
    def test_shift_datetime(self):
        # 引用的 GitHub 问题编号
        # 创建一个 IntervalArray，从日期范围创建断点
        a = IntervalArray.from_breaks(date_range("2000", periods=4))
        # 对 IntervalArray 进行正向 shift 操作，生成结果
        result = a.shift(2)
        # 期望的结果是根据索引取值后的 IntervalArray
        expected = a.take([-1, -1, 0], allow_fill=True)
        # 使用测试工具比较两个 IntervalArray 是否相等
        tm.assert_interval_array_equal(result, expected)

        # 对 IntervalArray 进行反向 shift 操作，生成结果
        result = a.shift(-1)
        # 期望的结果是根据索引取值后的 IntervalArray
        expected = a.take([1, 2, -1], allow_fill=True)
        # 使用测试工具比较两个 IntervalArray 是否相等
        tm.assert_interval_array_equal(result, expected)

        # 设置错误消息
        msg = "can only insert Interval objects and NA into an IntervalArray"
        # 使用 pytest 检查是否抛出特定类型的错误，并且错误消息匹配特定的文本
        with pytest.raises(TypeError, match=msg):
            # 测试在 shift 操作中插入非法值是否会引发预期的错误
            a.shift(1, fill_value=np.timedelta64("NaT", "ns"))
class TestSetitem:
    # 测试设置元素到IntervalArray对象中
    def test_set_na(self, left_right_dtypes):
        # 解包left_right_dtypes元组
        left, right = left_right_dtypes
        # 深拷贝left和right对象
        left = left.copy(deep=True)
        right = right.copy(deep=True)
        # 使用left和right创建IntervalArray对象
        result = IntervalArray.from_arrays(left, right)

        # 检查result的数据类型的子类型是否为日期时间或时间戳
        if result.dtype.subtype.kind not in ["m", "M"]:
            # 如果不是，则抛出TypeError异常，匹配指定的错误消息
            msg = "'value' should be an interval type, got <.*NaTType'> instead."
            with pytest.raises(TypeError, match=msg):
                result[0] = pd.NaT
        # 如果是整数或无符号整数类型
        if result.dtype.subtype.kind in ["i", "u"]:
            # 抛出TypeError异常，匹配指定的错误消息
            msg = "Cannot set float NaN to integer-backed IntervalArray"
            with pytest.raises(TypeError, match=msg):
                result[0] = np.nan
            return

        # 设置result的第一个元素为np.nan
        result[0] = np.nan

        # 创建预期的left和right索引
        expected_left = Index([left._na_value] + list(left[1:]))
        expected_right = Index([right._na_value] + list(right[1:]))
        # 创建预期的IntervalArray对象
        expected = IntervalArray.from_arrays(expected_left, expected_right)

        # 断言result和expected相等
        tm.assert_extension_array_equal(result, expected)

    # 测试设置元素时关闭属性不匹配的情况
    def test_setitem_mismatched_closed(self):
        # 创建IntervalArray对象
        arr = IntervalArray.from_breaks(range(4))
        # 拷贝arr对象
        orig = arr.copy()
        # 设置arr的闭合属性为"both"
        other = arr.set_closed("both")

        # 抛出值错误异常，匹配指定的错误消息
        msg = "'value.closed' is 'both', expected 'right'"
        with pytest.raises(ValueError, match=msg):
            arr[0] = other[0]
        with pytest.raises(ValueError, match=msg):
            arr[:1] = other[:1]
        with pytest.raises(ValueError, match=msg):
            arr[:0] = other[:0]
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1]
        with pytest.raises(ValueError, match=msg):
            arr[:] = list(other[::-1])
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1].astype(object)
        with pytest.raises(ValueError, match=msg):
            arr[:] = other[::-1].astype("category")

        # 空列表操作不会改变arr对象
        arr[:0] = []
        # 断言arr和orig相等
        tm.assert_interval_array_equal(arr, orig)


class TestReductions:
    # 测试最小和最大值计算时无效的轴参数
    def test_min_max_invalid_axis(self, left_right_dtypes):
        # 解包left_right_dtypes元组
        left, right = left_right_dtypes
        # 深拷贝left和right对象
        left = left.copy(deep=True)
        right = right.copy(deep=True)
        # 创建IntervalArray对象
        arr = IntervalArray.from_arrays(left, right)

        # 抛出值错误异常，匹配指定的错误消息
        msg = "`axis` must be fewer than the number of dimensions"
        for axis in [-2, 1]:
            with pytest.raises(ValueError, match=msg):
                arr.min(axis=axis)
            with pytest.raises(ValueError, match=msg):
                arr.max(axis=axis)

        # 抛出类型错误异常，匹配指定的错误消息
        msg = "'>=' not supported between"
        with pytest.raises(TypeError, match=msg):
            arr.min(axis="foo")
        with pytest.raises(TypeError, match=msg):
            arr.max(axis="foo")
    # 定义一个测试函数，用于测试最小值和最大值的计算
    def test_min_max(self, left_right_dtypes, index_or_series_or_array):
        # GH#44746

        # 从 left_right_dtypes 中分别获取 left 和 right，并进行深拷贝
        left, right = left_right_dtypes
        left = left.copy(deep=True)
        right = right.copy(deep=True)

        # 根据 left 和 right 创建一个 IntervalArray 对象
        arr = IntervalArray.from_arrays(left, right)

        # 如果 arr 是单调递增的，以下期望结果才有效
        assert left.is_monotonic_increasing
        assert Index(arr).is_monotonic_increasing

        # 获取 arr 的最小值和最大值
        MIN = arr[0]
        MAX = arr[-1]

        # 创建一个索引数组，并对其进行随机重排
        indexer = np.arange(len(arr))
        np.random.default_rng(2).shuffle(indexer)
        arr = arr.take(indexer)

        # 在 arr 中插入一个 NaN 值
        arr_na = arr.insert(2, np.nan)

        # 将 arr 和 arr_na 转换为索引、系列或数组（取决于传入的参数）
        arr = index_or_series_or_array(arr)
        arr_na = index_or_series_or_array(arr_na)

        # 针对 skipna 参数分别计算最小值和最大值，并进行断言
        for skipna in [True, False]:
            res = arr.min(skipna=skipna)
            assert res == MIN
            assert type(res) == type(MIN)

            res = arr.max(skipna=skipna)
            assert res == MAX
            assert type(res) == type(MAX)

        # 对 arr_na 使用 skipna=False 计算最小值和最大值，并进行断言
        res = arr_na.min(skipna=False)
        assert np.isnan(res)
        res = arr_na.max(skipna=False)
        assert np.isnan(res)

        # 对 arr_na 使用 skipna=True 计算最小值和最大值，并进行断言
        res = arr_na.min(skipna=True)
        assert res == MIN
        assert type(res) == type(MIN)
        res = arr_na.max(skipna=True)
        assert res == MAX
        assert type(res) == type(MAX)
```
# `D:\src\scipysrc\pandas\pandas\tests\indexes\interval\test_constructors.py`

```
# 导入必要的库和模块
from functools import partial  # 导入 functools 模块的 partial 函数

import numpy as np  # 导入 numpy 库并重命名为 np
import pytest  # 导入 pytest 测试框架

import pandas.util._test_decorators as td  # 导入 pandas 内部测试装饰器模块

from pandas.core.dtypes.common import is_unsigned_integer_dtype  # 从 pandas 核心数据类型中导入函数
from pandas.core.dtypes.dtypes import IntervalDtype  # 从 pandas 核心数据类型中导入 IntervalDtype 类

from pandas import (  # 从 pandas 库中导入多个类和函数
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    Index,
    Interval,
    IntervalIndex,
    date_range,
    notna,
    period_range,
    timedelta_range,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块，并重命名为 tm
from pandas.core.arrays import IntervalArray  # 从 pandas 核心数组中导入 IntervalArray 类
import pandas.core.common as com  # 导入 pandas 核心通用模块，并重命名为 com


class ConstructorTests:
    """
    Common tests for all variations of IntervalIndex construction. Input data
    to be supplied in breaks format, then converted by the subclass method
    get_kwargs_from_breaks to the expected format.
    """

    @pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器进行参数化测试
        "breaks_and_expected_subtype",
        [
            ([3, 14, 15, 92, 653], np.int64),  # 提供断点和预期子类型的参数化组合
            (np.arange(10, dtype="int64"), np.int64),
            (Index(np.arange(-10, 11, dtype=np.int64)), np.int64),
            (Index(np.arange(10, 31, dtype=np.uint64)), np.uint64),
            (Index(np.arange(20, 30, 0.5), dtype=np.float64), np.float64),
            (date_range("20180101", periods=10), "M8[ns]"),
            (
                date_range("20180101", periods=10, tz="US/Eastern"),
                "datetime64[ns, US/Eastern]",
            ),
            (timedelta_range("1 day", periods=10), "m8[ns]"),
        ],
    )
    @pytest.mark.parametrize("name", [None, "foo"])  # 参数化测试参数名
    def test_constructor(self, constructor, breaks_and_expected_subtype, closed, name):
        breaks, expected_subtype = breaks_and_expected_subtype

        result_kwargs = self.get_kwargs_from_breaks(breaks, closed)  # 调用子类方法，从断点和闭合标志获取参数

        result = constructor(closed=closed, name=name, **result_kwargs)  # 调用构造函数进行对象构造

        # 断言各项属性和结果是否符合预期
        assert result.closed == closed
        assert result.name == name
        assert result.dtype.subtype == expected_subtype
        tm.assert_index_equal(result.left, Index(breaks[:-1], dtype=expected_subtype))  # 使用测试模块的函数进行索引比较
        tm.assert_index_equal(result.right, Index(breaks[1:], dtype=expected_subtype))

    @pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器进行参数化测试
        "breaks, subtype",
        [
            (Index([0, 1, 2, 3, 4], dtype=np.int64), "float64"),  # 提供断点和子类型的参数化组合
            (Index([0, 1, 2, 3, 4], dtype=np.int64), "datetime64[ns]"),
            (Index([0, 1, 2, 3, 4], dtype=np.int64), "timedelta64[ns]"),
            (Index([0, 1, 2, 3, 4], dtype=np.float64), "int64"),
            (date_range("2017-01-01", periods=5), "int64"),
            (timedelta_range("1 day", periods=5), "int64"),
        ],
    )
    @pytest.mark.parametrize(
        "breaks",
        [
            Index([0, 1, 2, 3, 4], dtype=np.int64),  # 设置断点作为整数类型
            Index([0, 1, 2, 3, 4], dtype=np.uint64),  # 设置断点作为无符号整数类型
            Index([0, 1, 2, 3, 4], dtype=np.float64),  # 设置断点作为浮点数类型
            date_range("2017-01-01", periods=5),  # 生成包含日期范围的断点
            timedelta_range("1 day", periods=5),  # 生成包含时间间隔范围的断点
        ],
    )
    def test_constructor_pass_closed(self, constructor, breaks):
        # 不传递关闭参数给 IntervalDtype，而是传递给 IntervalArray 构造函数
        iv_dtype = IntervalDtype(breaks.dtype)

        result_kwargs = self.get_kwargs_from_breaks(breaks)

        for dtype in (iv_dtype, str(iv_dtype)):
            with tm.assert_produces_warning(None):  # 断言不产生警告
                result = constructor(dtype=dtype, closed="left", **result_kwargs)
            assert result.dtype.closed == "left"  # 断言结果的关闭属性为左闭合

    @pytest.mark.parametrize("breaks", [[np.nan] * 2, [np.nan] * 4, [np.nan] * 50])
    def test_constructor_nan(self, constructor, breaks, closed):
        # GH 18421
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)

        expected_subtype = np.float64  # 期望的子类型为浮点数
        expected_values = np.array(breaks[:-1], dtype=object)  # 期望的数值为断点数组的前n-1个值，类型为对象类型

        assert result.closed == closed  # 断言结果的关闭属性与预期相符
        assert result.dtype.subtype == expected_subtype  # 断言结果的数据类型子类型与预期相符
        tm.assert_numpy_array_equal(np.array(result), expected_values)  # 使用 numpy 断言结果与期望值数组相等

    @pytest.mark.parametrize(
        "breaks",
        [
            [],  # 空数组
            np.array([], dtype="int64"),  # 空的整数类型数组
            np.array([], dtype="uint64"),  # 空的无符号整数类型数组
            np.array([], dtype="float64"),  # 空的浮点数类型数组
            np.array([], dtype="datetime64[ns]"),  # 空的日期时间类型数组
            np.array([], dtype="timedelta64[ns]"),  # 空的时间间隔类型数组
        ],
    )
    def test_constructor_empty(self, constructor, breaks, closed):
        # GH 18421
        result_kwargs = self.get_kwargs_from_breaks(breaks)
        result = constructor(closed=closed, **result_kwargs)

        expected_values = np.array([], dtype=object)  # 期望的数值为空的对象数组
        expected_subtype = getattr(breaks, "dtype", np.int64)  # 期望的子类型为断点数组的数据类型或默认为整数类型

        assert result.empty  # 断言结果为空
        assert result.closed == closed  # 断言结果的关闭属性与预期相符
        assert result.dtype.subtype == expected_subtype  # 断言结果的数据类型子类型与预期相符
        tm.assert_numpy_array_equal(np.array(result), expected_values)  # 使用 numpy 断言结果与期望值数组相等

    @pytest.mark.parametrize(
        "breaks",
        [
            tuple("0123456789"),  # 字符串元组作为断点
            list("abcdefghij"),  # 字符串列表作为断点
            np.array(list("abcdefghij"), dtype=object),  # 对象类型的字符数组作为断点
            np.array(list("abcdefghij"), dtype="<U1"),  # Unicode 字符串数组作为断点
        ],
    )
    def test_constructor_string(self, constructor, breaks):
        # GH 19016
        # 设置错误消息，指明在 IntervalIndex 中不支持 category、object 和 string 子类型
        msg = (
            "category, object, and string subtypes are not supported "
            "for IntervalIndex"
        )
        # 使用 pytest 来验证是否会抛出 TypeError 异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 调用 constructor 方法，传入从 breaks 中获取的参数作为关键字参数
            constructor(**self.get_kwargs_from_breaks(breaks))

    @pytest.mark.parametrize("cat_constructor", [Categorical, CategoricalIndex])
    def test_constructor_categorical_valid(self, constructor, cat_constructor):
        # GH 21243/21253
        # 创建一个包含 10 个 int64 类型元素的数组 breaks
        breaks = np.arange(10, dtype="int64")
        # 从 breaks 创建一个预期的 IntervalIndex 对象
        expected = IntervalIndex.from_breaks(breaks)

        # 使用 cat_constructor 创建一个分类对象 cat_breaks
        cat_breaks = cat_constructor(breaks)
        # 从 cat_breaks 中获取参数，并调用 constructor 方法创建 result
        result_kwargs = self.get_kwargs_from_breaks(cat_breaks)
        result = constructor(**result_kwargs)
        # 使用 tm.assert_index_equal 方法验证 result 是否等于预期的 expected
        tm.assert_index_equal(result, expected)

    def test_generic_errors(self, constructor):
        # 用于在提供无效关键字参数时使用的填充输入数据
        filler = self.get_kwargs_from_breaks(range(10))

        # 检查无效的 closed 参数
        msg = "closed must be one of 'right', 'left', 'both', 'neither'"
        with pytest.raises(ValueError, match=msg):
            # 尝试使用无效的 closed 参数调用 constructor
            constructor(closed="invalid", **filler)

        # 检查不支持的 dtype
        msg = "dtype must be an IntervalDtype, got int64"
        with pytest.raises(TypeError, match=msg):
            # 尝试使用不支持的 dtype 调用 constructor
            constructor(dtype="int64", **filler)

        # 检查无效的 dtype
        msg = "data type [\"']invalid[\"'] not understood"
        with pytest.raises(TypeError, match=msg):
            # 尝试使用无效的 dtype 调用 constructor
            constructor(dtype="invalid", **filler)

        # 检查在 IntervalIndex 中嵌套周期没有意义
        periods = period_range("2000-01-01", periods=10)
        periods_kwargs = self.get_kwargs_from_breaks(periods)
        msg = "Period dtypes are not supported, use a PeriodIndex instead"
        with pytest.raises(ValueError, match=msg):
            # 尝试使用周期数据 periods_kwargs 调用 constructor
            constructor(**periods_kwargs)

        # 检查降序的值
        decreasing_kwargs = self.get_kwargs_from_breaks(range(10, -1, -1))
        msg = "left side of interval must be <= right side"
        with pytest.raises(ValueError, match=msg):
            # 尝试使用降序的值调用 constructor
            constructor(**decreasing_kwargs)
class TestFromArrays(ConstructorTests):
    """Tests specific to IntervalIndex.from_arrays"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_arrays

    def get_kwargs_from_breaks(self, breaks, closed="right"):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_arrays
        """
        return {"left": breaks[:-1], "right": breaks[1:]}

    def test_constructor_errors(self):
        # GH 19016: categorical data
        data = Categorical(list("01234abcde"), ordered=True)
        msg = (
            "category, object, and string subtypes are not supported "
            "for IntervalIndex"
        )
        # 检查是否会抛出TypeError异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_arrays(data[:-1], data[1:])

        # unequal length
        left = [0, 1, 2]
        right = [2, 3]
        msg = "left and right must have the same length"
        # 检查是否会抛出ValueError异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            IntervalIndex.from_arrays(left, right)

    @pytest.mark.parametrize(
        "left_subtype, right_subtype", [(np.int64, np.float64), (np.float64, np.int64)]
    )
    def test_mixed_float_int(self, left_subtype, right_subtype):
        """mixed int/float left/right results in float for both sides"""
        left = np.arange(9, dtype=left_subtype)
        right = np.arange(1, 10, dtype=right_subtype)
        # 调用IntervalIndex.from_arrays方法，并保存返回结果
        result = IntervalIndex.from_arrays(left, right)

        expected_left = Index(left, dtype=np.float64)
        expected_right = Index(right, dtype=np.float64)
        expected_subtype = np.float64

        # 检查左右边界的索引是否相等
        tm.assert_index_equal(result.left, expected_left)
        tm.assert_index_equal(result.right, expected_right)
        # 断言结果的dtype的subtype是否符合预期
        assert result.dtype.subtype == expected_subtype

    @pytest.mark.parametrize("interval_cls", [IntervalArray, IntervalIndex])
    def test_from_arrays_mismatched_datetimelike_resos(self, interval_cls):
        # GH#55714
        left = date_range("2016-01-01", periods=3, unit="s")
        right = date_range("2017-01-01", periods=3, unit="ms")
        # 调用from_arrays方法，创建IntervalIndex对象，并保存返回结果
        result = interval_cls.from_arrays(left, right)
        expected = interval_cls.from_arrays(left.as_unit("ms"), right)
        # 检查结果与预期是否相等
        tm.assert_equal(result, expected)

        # td64
        left2 = left - left[0]
        right2 = right - left[0]
        # 创建新的IntervalIndex对象，并保存返回结果
        result2 = interval_cls.from_arrays(left2, right2)
        expected2 = interval_cls.from_arrays(left2.as_unit("ms"), right2)
        # 检查结果与预期是否相等
        tm.assert_equal(result2, expected2)

        # dt64tz
        left3 = left.tz_localize("UTC")
        right3 = right.tz_localize("UTC")
        # 创建新的IntervalIndex对象，并保存返回结果
        result3 = interval_cls.from_arrays(left3, right3)
        expected3 = interval_cls.from_arrays(left3.as_unit("ms"), right3)
        # 检查结果与预期是否相等
        tm.assert_equal(result3, expected3)


class TestFromBreaks(ConstructorTests):
    """Tests specific to IntervalIndex.from_breaks"""

    @pytest.fixture
    # 返回 IntervalIndex 类的 from_breaks 方法的引用
    def constructor(self):
        return IntervalIndex.from_breaks

    # 根据 breaks 格式的区间转换为 IntervalIndex.from_breaks 方法所需的参数字典
    def get_kwargs_from_breaks(self, breaks, closed="right"):
        """
        将 breaks 格式的区间转换为一个字典，用于传递给 IntervalIndex.from_breaks 方法
        """
        return {"breaks": breaks}

    # 测试构造函数在特定情况下是否会引发 TypeError
    def test_constructor_errors(self):
        # 创建一个有序的分类数据
        data = Categorical(list("01234abcde"), ordered=True)
        # 定义错误消息
        msg = (
            "category, object, and string subtypes are not supported "
            "for IntervalIndex"
        )
        # 使用 pytest 检查是否会引发 TypeError，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            IntervalIndex.from_breaks(data)

    # 测试长度为一的 breaks 是否会产生空的 IntervalIndex
    def test_length_one(self):
        """breaks 长度为一时产生一个空的 IntervalIndex"""
        breaks = [0]
        # 使用 IntervalIndex.from_breaks 方法创建结果
        result = IntervalIndex.from_breaks(breaks)
        # 使用 IntervalIndex.from_breaks 方法创建预期结果
        expected = IntervalIndex.from_breaks([])
        # 使用 tm.assert_index_equal 检查结果与预期结果是否相等
        tm.assert_index_equal(result, expected)

    # 测试 left/right 是否共享数据的问题
    def test_left_right_dont_share_data(self):
        # GH#36310
        # 创建一个 numpy 数组作为 breaks
        breaks = np.arange(5)
        # 调用 IntervalIndex.from_breaks 方法，并获取其 _data 属性
        result = IntervalIndex.from_breaks(breaks)._data
        # 断言检查左右端点是否不共享数据基础
        assert result._left.base is None or result._left.base is not result._right.base
class TestFromTuples(ConstructorTests):
    """Tests specific to IntervalIndex.from_tuples"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex.from_tuples

    def get_kwargs_from_breaks(self, breaks, closed="right"):
        """
        converts intervals in breaks format to a dictionary of kwargs to
        specific to the format expected by IntervalIndex.from_tuples
        """
        # 检查输入的断点类型是否为无符号整数，如果是则跳过测试
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f"{breaks.dtype} not relevant IntervalIndex.from_tuples tests")

        # 如果断点列表为空，则返回包含断点数据的字典
        if len(breaks) == 0:
            return {"data": breaks}

        # 将断点列表转换为由断点组成的元组列表
        tuples = list(zip(breaks[:-1], breaks[1:]))
        # 根据断点的类型不同，返回适合 IntervalIndex.from_tuples 函数的参数字典
        if isinstance(breaks, (list, tuple)):
            return {"data": tuples}
        elif isinstance(getattr(breaks, "dtype", None), CategoricalDtype):
            return {"data": breaks._constructor(tuples)}
        return {"data": com.asarray_tuplesafe(tuples)}

    def test_constructor_errors(self):
        # 测试非元组输入的情况
        tuples = [(0, 1), 2, (3, 4)]
        msg = "IntervalIndex.from_tuples received an invalid item, 2"
        # 断言调用 IntervalIndex.from_tuples 时会抛出 TypeError 异常，匹配指定的错误消息
        with pytest.raises(TypeError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

        # 测试元组长度不正确的情况
        tuples = [(0, 1), (2,), (3, 4)]
        msg = "IntervalIndex.from_tuples requires tuples of length 2, got {t}"
        # 断言调用 IntervalIndex.from_tuples 时会抛出 ValueError 异常，匹配指定的错误消息
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

        tuples = [(0, 1), (2, 3, 4), (5, 6)]
        # 断言调用 IntervalIndex.from_tuples 时会抛出 ValueError 异常，匹配指定的错误消息
        with pytest.raises(ValueError, match=msg.format(t=tuples)):
            IntervalIndex.from_tuples(tuples)

    def test_na_tuples(self):
        # 测试包含 NA 的元组输入情况
        na_tuple = [(0, 1), (np.nan, np.nan), (2, 3)]
        # 调用 IntervalIndex.from_tuples 创建索引对象，包含 NA 的元组等效于包含 NA 元素
        idx_na_tuple = IntervalIndex.from_tuples(na_tuple)
        idx_na_element = IntervalIndex.from_tuples([(0, 1), np.nan, (2, 3)])
        # 断言两种方式创建的索引对象相等
        tm.assert_index_equal(idx_na_tuple, idx_na_element)


class TestClassConstructors(ConstructorTests):
    """Tests specific to the IntervalIndex/Index constructors"""

    @pytest.fixture
    def constructor(self):
        return IntervalIndex
    def get_kwargs_from_breaks(self, breaks, closed="right"):
        """
        将断点格式的间隔转换为适用于IntervalIndex/Index构造函数期望的kwargs字典
        """
        # 如果断点是无符号整数类型，跳过测试
        if is_unsigned_integer_dtype(breaks):
            pytest.skip(f"{breaks.dtype} not relevant for class constructor tests")

        # 如果断点列表为空，则返回包含"data"键的字典
        if len(breaks) == 0:
            return {"data": breaks}

        # 创建区间对象列表ivs，从断点列表中生成区间对象
        ivs = [
            Interval(left, right, closed) if notna(left) else left
            for left, right in zip(breaks[:-1], breaks[1:])
        ]

        # 如果断点是列表类型，则返回包含"data"键的字典
        if isinstance(breaks, list):
            return {"data": ivs}
        # 如果断点是分类数据类型，则返回包含"data"键的字典
        elif isinstance(getattr(breaks, "dtype", None), CategoricalDtype):
            return {"data": breaks._constructor(ivs)}
        # 否则，将ivs转换为对象类型的numpy数组，返回包含"data"键的字典
        return {"data": np.array(ivs, dtype=object)}

    def test_generic_errors(self, constructor):
        """
        由于错误在Interval级别上已经处理，所以覆盖基类实现；不进行检查
        """

    def test_constructor_string(self):
        """
        GH23013
        当从断点形成区间时，字符串的区间已被禁止
        """

    @pytest.mark.parametrize(
        "klass",
        [IntervalIndex, partial(Index, dtype="interval")],
        ids=["IntervalIndex", "Index"],
    )
    def test_constructor_errors(self, klass):
        """
        不匹配的闭合方式会导致在没有构造函数重写的情况下出错
        检查特定构造函数错误的情况
        """

        # 创建不匹配闭合方式的区间列表
        ivs = [Interval(0, 1, closed="right"), Interval(2, 3, closed="left")]
        msg = "intervals must all be closed on the same side"
        # 断言引发值错误，匹配指定消息
        with pytest.raises(ValueError, match=msg):
            klass(ivs)

        # 断言引发类型错误，匹配指定消息
        msg = (
            r"(IntervalIndex|Index)\(...\) must be called with a collection of "
            "some kind, 5 was passed"
        )
        with pytest.raises(TypeError, match=msg):
            klass(5)

        # 断言引发类型错误，匹配指定消息
        msg = "type <class 'numpy.int(32|64)'> with value 0 is not an interval"
        with pytest.raises(TypeError, match=msg):
            klass([0, 1])

    @pytest.mark.parametrize(
        "data, closed",
        [
            ([], "both"),
            ([np.nan, np.nan], "neither"),
            (
                [Interval(0, 3, closed="neither"), Interval(2, 5, closed="neither")],
                "left",
            ),
            (
                [Interval(0, 3, closed="left"), Interval(2, 5, closed="right")],
                "neither",
            ),
            # 调用from_breaks方法生成IntervalIndex，测试不同的闭合方式
            (IntervalIndex.from_breaks(range(5), closed="both"), "right"),
        ],
    )
    # 测试函数，用于验证是否正确处理了被推断为关闭的数据对象
    def test_override_inferred_closed(self, constructor, data, closed):
        # GH 19370
        # 如果数据对象是 IntervalIndex 类型，则将其转换为元组列表
        if isinstance(data, IntervalIndex):
            tuples = data.to_tuples()
        else:
            # 否则，将数据对象中的 Interval 对象转换为元组，处理缺失值
            tuples = [(iv.left, iv.right) if notna(iv) else iv for iv in data]
        # 根据处理后的元组列表创建预期的 IntervalIndex 对象，指定闭合属性
        expected = IntervalIndex.from_tuples(tuples, closed=closed)
        # 使用指定的构造函数和闭合属性创建结果对象
        result = constructor(data, closed=closed)
        # 验证结果对象与预期对象是否相等
        tm.assert_index_equal(result, expected)

    # 参数化测试函数，测试 Index 对象的数据类型是否正确
    @pytest.mark.parametrize(
        "values_constructor", [list, np.array, IntervalIndex, IntervalArray]
    )
    def test_index_object_dtype(self, values_constructor):
        # 创建包含 Interval 对象的列表
        intervals = [Interval(0, 1), Interval(1, 2), Interval(2, 3)]
        # 使用指定的构造函数创建数据对象
        values = values_constructor(intervals)
        # 使用对象类型为 object 的 Index 类型创建结果对象
        result = Index(values, dtype=object)

        # 断言结果对象的类型为 Index
        assert type(result) is Index
        # 断言结果对象的值数组与原始值数组相等
        tm.assert_numpy_array_equal(result.values, np.array(values))

    # 测试函数，验证混合闭合属性的 Index 对象创建
    def test_index_mixed_closed(self):
        # GH27172
        # 创建包含不同闭合属性的 Interval 对象列表
        intervals = [
            Interval(0, 1, closed="left"),
            Interval(1, 2, closed="right"),
            Interval(2, 3, closed="neither"),
            Interval(3, 4, closed="both"),
        ]
        # 使用 Interval 对象列表创建 Index 对象
        result = Index(intervals)
        # 使用对象类型为 object 的 Index 类型创建预期对象
        expected = Index(intervals, dtype=object)
        # 验证结果对象与预期对象是否相等
        tm.assert_index_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器为 test_interval_index_subtype 函数参数化不同的时区值
@pytest.mark.parametrize("timezone", ["UTC", "US/Pacific", "GMT"])
def test_interval_index_subtype(timezone, inclusive_endpoints_fixture):
    # GH#46999: 标识 GitHub 上的问题编号
    # 使用 date_range 函数生成指定时区下的日期范围，返回一个 DatetimeIndex 对象
    dates = date_range("2022", periods=3, tz=timezone)
    # 构建 interval 的 dtype，指定了日期时间的时区和包含边界类型
    dtype = f"interval[datetime64[ns, {timezone}], {inclusive_endpoints_fixture}]"
    # 使用 from_arrays 方法创建 IntervalIndex 对象，传入起始和结束日期数组，并指定包含边界和 dtype
    result = IntervalIndex.from_arrays(
        ["2022-01-01", "2022-01-02"],
        ["2022-01-02", "2022-01-03"],
        closed=inclusive_endpoints_fixture,
        dtype=dtype,
    )
    # 使用 from_arrays 方法创建期望的 IntervalIndex 对象，传入日期范围的起始和结束日期数组，以及包含边界
    expected = IntervalIndex.from_arrays(
        dates[:-1], dates[1:], closed=inclusive_endpoints_fixture
    )
    # 使用 assert_index_equal 方法断言两个 IntervalIndex 对象是否相等
    tm.assert_index_equal(result, expected)


# 定义测试函数 test_dtype_closed_mismatch，用于测试 dtype 和 IntervalIndex 构造函数中 closed 参数不匹配的情况
def test_dtype_closed_mismatch():
    # GH#38394: 标识 GitHub 上的问题编号，说明 closed 参数在 dtype 和 IntervalIndex 构造函数中都指定了
    # 创建一个 IntervalDtype 对象，指定了 subtype 和 closed 类型
    dtype = IntervalDtype(np.int64, "left")
    # 期望抛出 ValueError 异常，消息内容为 "closed keyword does not match dtype.closed"
    msg = "closed keyword does not match dtype.closed"
    # 使用 pytest.raises 检查是否抛出指定异常和匹配的消息
    with pytest.raises(ValueError, match=msg):
        IntervalIndex([], dtype=dtype, closed="neither")

    with pytest.raises(ValueError, match=msg):
        IntervalArray([], dtype=dtype, closed="neither")


# 使用 pytest.mark.parametrize 装饰器为 test_ea_dtype 函数参数化不同的 dtype 值
@pytest.mark.parametrize(
    "dtype",
    ["Float64", pytest.param("float64[pyarrow]", marks=td.skip_if_no("pyarrow"))],
)
def test_ea_dtype(dtype):
    # GH#56765: 标识 GitHub 上的问题编号
    # 定义 bins 数组，表示区间的起始和结束值
    bins = [(0.0, 0.4), (0.4, 0.6)]
    # 创建 IntervalDtype 对象，指定 subtype 和 closed 类型
    interval_dtype = IntervalDtype(subtype=dtype, closed="left")
    # 使用 from_tuples 方法创建 IntervalIndex 对象，传入区间的起始和结束元组数组，以及指定的 closed 和 dtype
    result = IntervalIndex.from_tuples(bins, closed="left", dtype=interval_dtype)
    # 断言 result 的 dtype 是否等于指定的 interval_dtype
    assert result.dtype == interval_dtype
    # 使用 from_tuples 方法创建期望的 IntervalIndex 对象，传入区间的起始和结束元组数组，并将其转换为指定的 dtype
    expected = IntervalIndex.from_tuples(bins, closed="left").astype(interval_dtype)
    # 使用 assert_index_equal 方法断言两个 IntervalIndex 对象是否相等
    tm.assert_index_equal(result, expected)
```
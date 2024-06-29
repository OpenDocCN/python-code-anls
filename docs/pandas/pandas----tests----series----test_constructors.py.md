# `D:\src\scipysrc\pandas\pandas\tests\series\test_constructors.py`

```
from collections import OrderedDict  # 导入 OrderedDict 类，用于创建有序字典
from collections.abc import Iterator  # 导入 Iterator 抽象基类，用于检查对象是否为迭代器
from datetime import (  # 导入 datetime 模块下的以下对象
    datetime,  # datetime 对象，用于处理日期和时间
    timedelta,  # timedelta 对象，表示时间间隔
)

from dateutil.tz import tzoffset  # 导入 tzoffset 对象，表示时区偏移量
import numpy as np  # 导入 NumPy 库，用于数值计算

from numpy import ma  # 导入 ma 模块，提供 NumPy 的掩码数组支持
import pytest  # 导入 pytest 库，用于编写和运行测试

from pandas._libs import (  # 导入 pandas 私有库中的以下对象
    iNaT,  # iNaT 对象，表示不可用的日期/时间
    lib,  # lib 对象，提供底层实现支持
)
from pandas.compat.numpy import np_version_gt2  # 导入 np_version_gt2 对象，用于检查 NumPy 版本是否大于2

from pandas.errors import IntCastingNaNError  # 导入 IntCastingNaNError 错误类，处理整数转换时的 NaN 错误

from pandas.core.dtypes.dtypes import CategoricalDtype  # 导入 CategoricalDtype 类，表示分类数据类型

import pandas as pd  # 导入 pandas 库，并命名为 pd，用于数据分析和处理
from pandas import (  # 从 pandas 中导入以下对象
    Categorical,  # Categorical 类，表示分类数据
    DataFrame,  # DataFrame 类，表示二维数据结构
    DatetimeIndex,  # DatetimeIndex 类，表示日期时间索引
    DatetimeTZDtype,  # DatetimeTZDtype 类，表示带有时区的日期时间数据类型
    Index,  # Index 类，表示一维标签数组
    Interval,  # Interval 类，表示区间
    IntervalIndex,  # IntervalIndex 类，表示区间索引
    MultiIndex,  # MultiIndex 类，表示多级索引
    NaT,  # NaT 对象，表示不可用的日期/时间
    Period,  # Period 类，表示时间段
    RangeIndex,  # RangeIndex 类，表示整数范围索引
    Series,  # Series 类，表示一维标签数组，支持任意数据类型
    Timestamp,  # Timestamp 类，表示时间戳
    date_range,  # date_range 函数，生成日期范围
    isna,  # isna 函数，检查是否为缺失值
    period_range,  # period_range 函数，生成时间段范围
    timedelta_range,  # timedelta_range 函数，生成时间间隔范围
)
import pandas._testing as tm  # 导入 pandas 测试模块，并命名为 tm

from pandas.core.arrays import (  # 导入 pandas core.arrays 模块中的以下对象
    IntegerArray,  # IntegerArray 类，表示整数数组
    IntervalArray,  # IntervalArray 类，表示区间数组
    period_array,  # period_array 函数，创建时间段数组
)
from pandas.core.internals.blocks import NumpyBlock  # 导入 NumpyBlock 类，表示内部数据块

class TestSeriesConstructors:  # 定义测试类 TestSeriesConstructors
    def test_from_ints_with_non_nano_dt64_dtype(self, index_or_series):  # 定义测试方法 test_from_ints_with_non_nano_dt64_dtype，接受 index_or_series 参数
        values = np.arange(10)  # 创建一个包含 0 到 9 的 NumPy 数组

        res = index_or_series(values, dtype="M8[s]")  # 使用 index_or_series 函数，将 values 转换为 dtype="M8[s]" 类型的结果存储在 res 中
        expected = index_or_series(values.astype("M8[s]"))  # 期望的结果是将 values 转换为 "M8[s]" 类型的 Series 或 Index 对象，存储在 expected 中
        tm.assert_equal(res, expected)  # 使用 tm.assert_equal 断言函数检查 res 和 expected 是否相等

        res = index_or_series(list(values), dtype="M8[s]")  # 使用 index_or_series 函数，将 values 列表化后转换为 dtype="M8[s]" 类型的结果存储在 res 中
        tm.assert_equal(res, expected)  # 使用 tm.assert_equal 断言函数检查 res 和 expected 是否相等

    def test_from_na_value_and_interval_of_datetime_dtype(self):  # 定义测试方法 test_from_na_value_and_interval_of_datetime_dtype
        # GH#41805
        ser = Series([None], dtype="interval[datetime64[ns]]")  # 创建一个 Series 对象，包含一个 None 值，数据类型为 "interval[datetime64[ns]]"
        assert ser.isna().all()  # 使用 assert 断言方法检查 ser 中所有值是否为 NaN
        assert ser.dtype == "interval[datetime64[ns], right]"  # 使用 assert 断言方法检查 ser 的数据类型是否为 "interval[datetime64[ns], right]"

    def test_infer_with_date_and_datetime(self):  # 定义测试方法 test_infer_with_date_and_datetime
        # GH#49341 pre-2.0 we inferred datetime-and-date to datetime64, which
        #  was inconsistent with Index behavior
        ts = Timestamp(2016, 1, 1)  # 创建一个 Timestamp 对象，表示日期为 2016 年 1 月 1 日
        vals = [ts.to_pydatetime(), ts.date()]  # 创建包含 ts 的 Python datetime 对象和日期对象的列表 vals

        ser = Series(vals)  # 创建一个 Series 对象，包含 vals 列表中的元素
        expected = Series(vals, dtype=object)  # 创建一个期望的 Series 对象，数据类型为 object
        tm.assert_series_equal(ser, expected)  # 使用 tm.assert_series_equal 断言函数检查 ser 和 expected 是否相等

        idx = Index(vals)  # 创建一个 Index 对象，包含 vals 列表中的元素
        expected = Index(vals, dtype=object)  # 创建一个期望的 Index 对象，数据类型为 object
        tm.assert_index_equal(idx, expected)  # 使用 tm.assert_index_equal 断言函数检查 idx 和 expected 是否相等

    def test_unparsable_strings_with_dt64_dtype(self):  # 定义测试方法 test_unparsable_strings_with_dt64_dtype
        # pre-2.0 these would be silently ignored and come back with object dtype
        vals = ["aa"]  # 创建包含一个无法解析的日期字符串 "aa" 的列表 vals
        msg = "^Unknown datetime string format, unable to parse: aa, at position 0$"  # 定义错误消息字符串 msg

        with pytest.raises(ValueError, match=msg):  # 使用 pytest.raises 检查 ValueError 异常是否被抛出，且错误消息与 msg 匹配
            Series(vals, dtype="datetime64[ns]")  # 创建一个 dtype="datetime64[ns]" 类型的 Series 对象，检查是否抛出 ValueError

        with pytest.raises(ValueError, match=msg):  # 使用 pytest.raises 检查 ValueError 异常是否被抛出，且错误消息与 msg 匹配
            Series(np.array(vals, dtype=object), dtype="datetime64[ns]")  # 创建一个包含 vals 的 NumPy 数组对象，检查是否抛出 ValueError
    @pytest.mark.parametrize(
        "constructor",
        [
            # NOTE: some overlap with test_constructor_empty but that test does not
            # test for None or an empty generator.
            # test_constructor_pass_none tests None but only with the index also
            # passed.
            # 使用 lambda 函数创建 Series 对象构造器，测试不同参数下的构造行为
            (lambda idx: Series(index=idx)),
            (lambda idx: Series(None, index=idx)),
            (lambda idx: Series({}, index=idx)),
            (lambda idx: Series((), index=idx)),
            (lambda idx: Series([], index=idx)),
            (lambda idx: Series((_ for _ in []), index=idx)),
            (lambda idx: Series(data=None, index=idx)),
            (lambda idx: Series(data={}, index=idx)),
            (lambda idx: Series(data=(), index=idx)),
            (lambda idx: Series(data=[], index=idx)),
            (lambda idx: Series(data=(_ for _ in []), index=idx)),
        ],
    )
    @pytest.mark.parametrize("empty_index", [None, []])
    # 测试空构造器行为，引入了 empty_index 参数
    def test_empty_constructor(self, constructor, empty_index):
        # GH 49573 (addition of empty_index parameter)
        # 期望结果是空的 Series 对象
        expected = Series(index=empty_index)
        # 使用 constructor 构造 Series 对象
        result = constructor(empty_index)

        assert result.dtype == object
        assert len(result.index) == 0
        # 断言生成的 Series 对象与期望的相等
        tm.assert_series_equal(result, expected, check_index_type=True)

    def test_invalid_dtype(self):
        # GH15520
        # 测试不支持的数据类型错误处理
        msg = "not understood"
        invalid_list = [Timestamp, "Timestamp", list]
        for dtype in invalid_list:
            with pytest.raises(TypeError, match=msg):
                # 构造 Series 对象并断言抛出 TypeError 异常
                Series([], name="time", dtype=dtype)

    def test_invalid_compound_dtype(self):
        # GH#13296
        # 测试复合数据类型错误处理
        c_dtype = np.dtype([("a", "i8"), ("b", "f4")])
        cdt_arr = np.array([(1, 0.4), (256, -13)], dtype=c_dtype)

        with pytest.raises(ValueError, match="Use DataFrame instead"):
            # 当传入复合数据类型时，期望抛出 ValueError 异常
            Series(cdt_arr, index=["A", "B"])

    def test_scalar_conversion(self):
        # Pass in scalar is disabled
        # 测试标量输入是否被禁用
        scalar = Series(0.5)
        assert not isinstance(scalar, float)

    def test_scalar_extension_dtype(self, ea_scalar_and_dtype):
        # GH 28401
        # 测试标量扩展数据类型
        ea_scalar, ea_dtype = ea_scalar_and_dtype

        ser = Series(ea_scalar, index=range(3))
        expected = Series([ea_scalar] * 3, dtype=ea_dtype)

        assert ser.dtype == ea_dtype
        # 断言生成的 Series 对象与期望的相等
        tm.assert_series_equal(ser, expected)
    # 定义测试方法，用于测试 Series 对象的构造函数
    def test_constructor(self, datetime_series, using_infer_string):
        # 创建一个空的 Series 对象
        empty_series = Series()
        # 断言 datetime_series 的索引全是日期类型
        assert datetime_series.index._is_all_dates

        # 传入一个现有的 Series 对象来创建新的 Series 对象
        derived = Series(datetime_series)
        # 断言新创建的 Series 对象的索引与原始 datetime_series 的索引相等
        assert derived.index._is_all_dates
        # 使用 pytest 工具断言两个 Series 对象的索引是相等的
        tm.assert_index_equal(derived.index, datetime_series.index)
        # 再次断言新 Series 对象的索引与原始 Series 对象的索引相同
        assert id(datetime_series.index) == id(derived.index)

        # 创建一个混合类型的 Series 对象
        mixed = Series(["hello", np.nan], index=[0, 1])
        # 如果不使用 infer_string 参数，则断言 mixed 的数据类型为 np.object_
        assert mixed.dtype == np.object_ if not using_infer_string else "string"
        # 断言 mixed 中的第二个元素是 NaN
        assert np.isnan(mixed[1])

        # 断言空 Series 对象的索引不全是日期类型
        assert not empty_series.index._is_all_dates
        # 创建一个新的空 Series 对象，再次断言其索引不全是日期类型
        assert not Series().index._is_all_dates

        # 测试引发异常，期望引发 ValueError 类型的异常，消息为指定的文本
        with pytest.raises(
            ValueError,
            match=r"Data must be 1-dimensional, got ndarray of shape \(3, 3\) instead",
        ):
            # 使用随机数生成器创建一个 ndarray，尝试用其初始化 Series 对象
            Series(np.random.default_rng(2).standard_normal((3, 3)), index=np.arange(3))

        # 修改 mixed 对象的名称为 "Series"
        mixed.name = "Series"
        # 创建一个新的 Series 对象，将 mixed 作为参数传入，然后获取其名称
        rs = Series(mixed).name
        # 预期 rs 的值与 xp 相等
        xp = "Series"
        assert rs == xp

        # 测试引发异常，期望引发 NotImplementedError 类型的异常，消息为指定的文本
        m = MultiIndex.from_arrays([[1, 2], [3, 4]])
        msg = "initializing a Series from a MultiIndex is not supported"
        with pytest.raises(NotImplementedError, match=msg):
            # 尝试用 MultiIndex 对象来初始化 Series 对象，预期会引发异常
            Series(m)

    # 测试构造函数，验证当索引数据的维度大于1时是否会引发 ValueError 异常
    def test_constructor_index_ndim_gt_1_raises(self):
        # 创建一个 DataFrame 对象
        df = DataFrame([[1, 2], [3, 4], [5, 6]], index=[3, 6, 9])
        # 测试引发异常，期望引发 ValueError 类型的异常，消息为指定的文本
        with pytest.raises(ValueError, match="Index data must be 1-dimensional"):
            # 尝试使用 df 对象的索引来初始化 Series 对象，预期会引发异常
            Series([1, 3, 2], index=df)

    # 使用 pytest.mark.parametrize 装饰器为下面的测试方法指定参数化的输入类
    @pytest.mark.parametrize("input_class", [list, dict, OrderedDict])
    # 测试空构造函数：创建空的 Series 对象
    def test_constructor_empty(self, input_class, using_infer_string):
        # 使用默认构造函数创建空的 Series 对象
        empty = Series()
        # 使用指定构造函数创建空的 Series 对象
        empty2 = Series(input_class())

        # 检查两个 Series 对象是否相等，忽略索引类型的比较
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        # 使用显式指定的数据类型创建空的 Series 对象
        empty = Series(dtype="float64")
        # 使用指定构造函数和数据类型创建空的 Series 对象
        empty2 = Series(input_class(), dtype="float64")
        # 检查两个 Series 对象是否相等，忽略索引类型的比较
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        # GH 18515: 使用数据类型为 category 创建空的 Series 对象
        empty = Series(dtype="category")
        # 使用指定构造函数和数据类型为 category 创建空的 Series 对象
        empty2 = Series(input_class(), dtype="category")
        # 检查两个 Series 对象是否相等，忽略索引类型的比较
        tm.assert_series_equal(empty, empty2, check_index_type=False)

        if input_class is not list:
            # 使用指定索引创建 Series 对象
            empty = Series(index=range(10))
            # 使用指定构造函数和索引创建 Series 对象
            empty2 = Series(input_class(), index=range(10))
            # 检查两个 Series 对象是否相等
            tm.assert_series_equal(empty, empty2)

            # 使用指定索引和数据类型创建 Series 对象
            empty = Series(np.nan, index=range(10))
            # 使用指定构造函数、索引和数据类型创建 Series 对象
            empty2 = Series(input_class(), index=range(10), dtype="float64")
            # 检查两个 Series 对象是否相等
            tm.assert_series_equal(empty, empty2)

            # GH 19853: 使用空字符串、指定索引和数据类型为 str 创建 Series 对象
            empty = Series("", dtype=str, index=range(3))
            if using_infer_string:
                # 如果使用推断类型，创建空字符串、指定索引和推断类型为 object 的 Series 对象
                empty2 = Series("", index=range(3), dtype=object)
            else:
                # 否则，创建空字符串和指定索引的 Series 对象
                empty2 = Series("", index=range(3))
            # 检查两个 Series 对象是否相等
            tm.assert_series_equal(empty, empty2)

    # 使用参数化测试，测试 NaN 作为构造函数参数的情况
    @pytest.mark.parametrize("input_arg", [np.nan, float("nan")])
    def test_constructor_nan(self, input_arg):
        # 使用指定数据类型和索引创建 Series 对象
        empty = Series(dtype="float64", index=range(10))
        # 使用指定构造函数参数和索引创建 Series 对象
        empty2 = Series(input_arg, index=range(10))

        # 检查两个 Series 对象是否相等，忽略索引类型的比较
        tm.assert_series_equal(empty, empty2, check_index_type=False)

    # 使用参数化测试，测试只指定数据类型的构造函数的情况
    @pytest.mark.parametrize(
        "dtype",
        ["f8", "i8", "M8[ns]", "m8[ns]", "category", "object", "datetime64[ns, UTC]"],
    )
    @pytest.mark.parametrize("index", [None, Index([])])
    def test_constructor_dtype_only(self, dtype, index):
        # GH-20865: 使用指定数据类型和索引创建 Series 对象
        result = Series(dtype=dtype, index=index)
        # 断言结果的数据类型与指定的数据类型相等
        assert result.dtype == dtype
        # 断言结果的长度为 0
        assert len(result) == 0

    # 测试创建没有数据但有指定索引顺序的 Series 对象的情况
    def test_constructor_no_data_index_order(self):
        # 使用指定索引创建 Series 对象
        result = Series(index=["b", "a", "c"])
        # 断言结果的索引顺序是否正确
        assert result.index.tolist() == ["b", "a", "c"]

    # 测试创建没有数据但数据类型为字符串的 Series 对象的情况
    def test_constructor_no_data_string_type(self):
        # GH 22477: 使用指定索引和数据类型为 str 创建 Series 对象
        result = Series(index=[1], dtype=str)
        # 断言结果的第一个元素是否为 NaN
        assert np.isnan(result.iloc[0])

    # 使用参数化测试，测试创建包含字符串元素且数据类型为 str 的 Series 对象的情况
    @pytest.mark.parametrize("item", ["entry", "ѐ", 13])
    def test_constructor_string_element_string_type(self, item):
        # GH 22477: 使用指定索引、字符串元素和数据类型为 str 创建 Series 对象
        result = Series(item, index=[1], dtype=str)
        # 断言结果的第一个元素是否等于原始元素转换为字符串后的值
        assert result.iloc[0] == str(item)
    # 测试函数：使用指定的字符串数据类型构造一个 Series 对象，并检查是否有缺失值
    def test_constructor_dtype_str_na_values(self, string_dtype):
        # 创建包含字符串和空值的 Series 对象
        ser = Series(["x", None], dtype=string_dtype)
        # 检查 Series 对象中的缺失值情况
        result = ser.isna()
        # 预期结果中的 Series 对象，标识缺失值的位置
        expected = Series([False, True])
        # 使用测试框架的方法比较实际结果与预期结果
        tm.assert_series_equal(result, expected)
        # 断言 Series 对象中第二个元素是 None
        assert ser.iloc[1] is None

        # 使用包含字符串和 np.nan 的 Series 对象
        ser = Series(["x", np.nan], dtype=string_dtype)
        # 断言 Series 对象中第二个元素是 NaN
        assert np.isnan(ser.iloc[1])

    # 测试函数：测试从现有 Series 对象创建另一个 Series 对象，指定不同的索引顺序
    def test_constructor_series(self):
        # 创建初始索引和相应的 Series 对象
        index1 = ["d", "b", "a", "c"]
        # 对初始索引排序
        index2 = sorted(index1)
        # 创建第一个 Series 对象
        s1 = Series([4, 7, -5, 3], index=index1)
        # 使用第一个 Series 对象创建第二个 Series 对象，指定不同的索引顺序
        s2 = Series(s1, index=index2)

        # 使用测试框架的方法比较两个 Series 对象是否相等，以排序后的索引为准
        tm.assert_series_equal(s2, s1.sort_index())

    # 测试函数：测试从可迭代对象创建 Series 对象
    def test_constructor_iterable(self):
        # 定义一个迭代器类，生成包含0到9的整数序列
        class Iter:
            def __iter__(self) -> Iterator:
                yield from range(10)

        # 预期结果的 Series 对象，包含0到9的整数
        expected = Series(list(range(10)), dtype="int64")
        # 使用迭代器对象创建 Series 对象
        result = Series(Iter(), dtype="int64")
        # 使用测试框架的方法比较实际结果与预期结果
        tm.assert_series_equal(result, expected)

    # 测试函数：测试从序列对象创建 Series 对象
    def test_constructor_sequence(self):
        # 预期结果的 Series 对象，包含0到9的整数
        expected = Series(list(range(10)), dtype="int64")
        # 使用序列对象创建 Series 对象
        result = Series(range(10), dtype="int64")
        # 使用测试框架的方法比较实际结果与预期结果
        tm.assert_series_equal(result, expected)

    # 测试函数：测试从单个字符串创建 Series 对象
    def test_constructor_single_str(self):
        # 预期结果的 Series 对象，包含单个字符串 "abc"
        expected = Series(["abc"])
        # 使用单个字符串创建 Series 对象
        result = Series("abc")
        # 使用测试框架的方法比较实际结果与预期结果
        tm.assert_series_equal(result, expected)

    # 测试函数：测试从类似列表的对象创建 Series 对象
    def test_constructor_list_like(self):
        # 确保将不同类型的列表样式对象强制转换为标准的数据类型，而不是特定于平台
        expected = Series([1, 2, 3], dtype="int64")
        # 遍历不同类型的列表样式对象
        for obj in [[1, 2, 3], (1, 2, 3), np.array([1, 2, 3], dtype="int64")]:
            # 使用每个对象创建 Series 对象，指定索引
            result = Series(obj, index=[0, 1, 2])
            # 使用测试框架的方法比较实际结果与预期结果
            tm.assert_series_equal(result, expected)

    # 测试函数：测试从布尔类型的索引创建 Series 对象
    def test_constructor_boolean_index(self):
        # 创建包含整数的 Series 对象，并指定索引
        s1 = Series([1, 2, 3], index=[4, 5, 6])

        # 使用布尔类型的索引创建 Series 对象
        index = s1 == 2
        result = Series([1, 3, 2], index=index)
        # 预期结果的 Series 对象，具有布尔类型的索引
        expected = Series([1, 3, 2], index=[False, True, False])
        # 使用测试框架的方法比较实际结果与预期结果
        tm.assert_series_equal(result, expected)

    # 测试函数：测试使用指定的索引数据类型创建 Series 对象
    @pytest.mark.parametrize("dtype", ["bool", "int32", "int64", "float64"])
    def test_constructor_index_dtype(self, dtype):
        # 使用 Index 对象创建 Series 对象，并指定数据类型
        s = Series(Index([0, 2, 4]), dtype=dtype)
        # 断言 Series 对象的数据类型与指定的数据类型相同
        assert s.dtype == dtype

    # 测试函数：测试从不同类型的输入值列表创建 Series 对象
    @pytest.mark.parametrize(
        "input_vals",
        [
            [1, 2],
            ["1", "2"],
            list(date_range("1/1/2011", periods=2, freq="h")),
            list(date_range("1/1/2011", periods=2, freq="h", tz="US/Eastern")),
            [Interval(left=0, right=5)],
        ],
    )
    # 定义测试方法，接受输入数据和字符串类型的数据类型参数
    def test_constructor_list_str(self, input_vals, string_dtype):
        # GH 16605
        # 确保当数据类型为 str、'str' 或 'U' 时，从列表中取出的数据元素转换为字符串
        result = Series(input_vals, dtype=string_dtype)
        # 期望结果是将输入数据转换为指定的字符串类型后的 Series
        expected = Series(input_vals).astype(string_dtype)
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法，接受字符串类型的数据类型参数
    def test_constructor_list_str_na(self, string_dtype):
        # 创建包含浮点数和 NaN 值的 Series 对象
        result = Series([1.0, 2.0, np.nan], dtype=string_dtype)
        # 创建包含字符串和 NaN 值的 Series 对象
        expected = Series(["1.0", "2.0", np.nan], dtype=object)
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)
        # 断言 result 的第三个元素是 NaN
        assert np.isnan(result[2])

    # 定义测试方法，生成一个生成器对象
    def test_constructor_generator(self):
        gen = (i for i in range(10))

        # 使用生成器创建 Series 对象
        result = Series(gen)
        # 创建期望的 Series 对象，包含范围内的整数
        exp = Series(range(10))
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, exp)

        # 使用非默认索引创建 Series 对象
        gen = (i for i in range(10))
        result = Series(gen, index=range(10, 20))
        # 修改期望的 Series 对象的索引
        exp.index = range(10, 20)
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, exp)

    # 定义测试方法，生成一个映射对象
    def test_constructor_map(self):
        # GH8909
        m = (x for x in range(10))

        # 使用映射对象创建 Series 对象
        result = Series(m)
        # 创建期望的 Series 对象，包含范围内的整数
        exp = Series(range(10))
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, exp)

        # 使用非默认索引创建 Series 对象
        m = (x for x in range(10))
        result = Series(m, index=range(10, 20))
        # 修改期望的 Series 对象的索引
        exp.index = range(10, 20)
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, exp)

    # 定义测试方法，创建分类数据类型的 Series 对象
    def test_constructor_categorical(self):
        # 创建分类变量对象
        cat = Categorical([0, 1, 2, 0, 1, 2], ["a", "b", "c"])
        # 使用分类变量对象创建 Series 对象
        res = Series(cat)
        # 断言 Series 对象的值与分类变量对象相等
        tm.assert_categorical_equal(res.values, cat)

        # 可以将分类变量对象转换为指定的新数据类型
        result = Series(Categorical([1, 2, 3]), dtype="int64")
        # 创建期望的 Series 对象，包含指定数据类型的整数值
        expected = Series([1, 2, 3], dtype="int64")
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 定义测试方法，从分类数据类型创建 Series 对象，并指定数据类型
    def test_construct_from_categorical_with_dtype(self):
        # GH12574
        # 使用分类变量对象创建 Series 对象，并指定数据类型为 'category'
        ser = Series(Categorical([1, 2, 3]), dtype="category")
        # 断言 Series 对象的数据类型为 CategoricalDtype 类型
        assert isinstance(ser.dtype, CategoricalDtype)

    # 定义测试方法，创建整数列表值的分类数据类型的 Series 对象
    def test_construct_intlist_values_category_dtype(self):
        # 使用整数列表创建分类数据类型的 Series 对象
        ser = Series([1, 2, 3], dtype="category")
        # 断言 Series 对象的数据类型为 CategoricalDtype 类型
        assert isinstance(ser.dtype, CategoricalDtype)
    def test_constructor_categorical_with_coercion(self):
        # 创建一个 Categorical 对象并进行基本的创建和类型转换测试
        factor = Categorical(["a", "b", "b", "a", "a", "c", "c", "c"])
        # 使用 Categorical 对象创建一个 Series，指定列名为"A"
        s = Series(factor, name="A")
        # 断言 Series 的数据类型为"category"
        assert s.dtype == "category"
        # 断言 Series 的长度与原始 factor 的长度相同
        assert len(s) == len(factor)

        # 在一个 DataFrame 中使用 Categorical 对象
        df = DataFrame({"A": factor})
        # 从 DataFrame 中取出"A"列，与之前创建的 Series s 进行比较
        result = df["A"]
        tm.assert_series_equal(result, s)
        # 从 DataFrame 中取出第一列，与之前创建的 Series s 进行比较
        result = df.iloc[:, 0]
        tm.assert_series_equal(result, s)
        # 断言 DataFrame 的长度与原始 factor 的长度相同
        assert len(df) == len(factor)

        # 使用 Series s 创建一个新的 DataFrame
        df = DataFrame({"A": s})
        # 从 DataFrame 中取出"A"列，与之前创建的 Series s 进行比较
        result = df["A"]
        tm.assert_series_equal(result, s)
        # 断言 DataFrame 的长度与原始 factor 的长度相同
        assert len(df) == len(factor)

        # 多列情况下的 DataFrame 创建
        df = DataFrame({"A": s, "B": s, "C": 1})
        # 分别取出"A"列和"B"列，与之前创建的 Series s 进行比较
        result1 = df["A"]
        result2 = df["B"]
        tm.assert_series_equal(result1, s)
        tm.assert_series_equal(result2, s, check_names=False)
        # 断言 result2 的列名为"B"
        assert result2.name == "B"
        # 断言 DataFrame 的长度与原始 factor 的长度相同
        assert len(df) == len(factor)

    def test_constructor_categorical_with_coercion2(self):
        # GH8623：测试特定情况下的 DataFrame 创建
        x = DataFrame(
            [[1, "John P. Doe"], [2, "Jane Dove"], [1, "John P. Doe"]],
            columns=["person_id", "person_name"],
        )
        # 将"person_name"列转换为 Categorical 类型，这会导致 transform 出现问题
        x["person_name"] = Categorical(x.person_name)

        expected = x.iloc[0].person_name
        result = x.person_name.iloc[0]
        # 断言转换后的结果与预期一致
        assert result == expected

        result = x.person_name[0]
        # 断言取出的结果与预期一致
        assert result == expected

        result = x.person_name.loc[0]
        # 断言 loc 方法取出的结果与预期一致
        assert result == expected

    def test_constructor_series_to_categorical(self):
        # GH#16524：测试将 Series 转换为 Categorical 类型
        series = Series(["a", "b", "c"])

        result = Series(series, dtype="category")
        expected = Series(["a", "b", "c"], dtype="category")

        tm.assert_series_equal(result, expected)

    def test_constructor_categorical_dtype(self):
        # 测试指定 CategoricalDtype 的情况
        result = Series(
            ["a", "b"], dtype=CategoricalDtype(["a", "b", "c"], ordered=True)
        )
        assert isinstance(result.dtype, CategoricalDtype)
        # 断言分类的类别与预期一致
        tm.assert_index_equal(result.cat.categories, Index(["a", "b", "c"]))
        # 断言结果为有序分类
        assert result.cat.ordered

        result = Series(["a", "b"], dtype=CategoricalDtype(["b", "a"]))
        assert isinstance(result.dtype, CategoricalDtype)
        # 断言分类的类别与预期一致
        tm.assert_index_equal(result.cat.categories, Index(["b", "a"]))
        # 断言结果为无序分类
        assert result.cat.ordered is False

        # GH 19565 - 检查在 Categorical dtype 中标量的广播
        result = Series(
            "a", index=[0, 1], dtype=CategoricalDtype(["a", "b"], ordered=True)
        )
        expected = Series(
            ["a", "a"], index=[0, 1], dtype=CategoricalDtype(["a", "b"], ordered=True)
        )
        tm.assert_series_equal(result, expected)
    def test_constructor_categorical_string(self):
        # 创建一个有序的分类数据类型 CategoricalDtype，指定分类列表和排序属性
        cdt = CategoricalDtype(categories=list("dabc"), ordered=True)
        # 创建一个预期的 Series，使用上述的分类数据类型
        expected = Series(list("abcabc"), dtype=cdt)

        # 创建一个 Categorical 对象，指定分类列表和数据类型，保持现有的数据类型
        cat = Categorical(list("abcabc"), dtype=cdt)
        # 创建一个 Series，将上述的 Categorical 对象作为数据，指定数据类型为 'category'
        result = Series(cat, dtype="category")
        # 断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

        # 将上一个结果的 Series 再次作为数据创建一个新的 Series，数据类型保持 'category'
        result = Series(result, dtype="category")
        # 断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    def test_categorical_sideeffects_free(self):
        # 创建一个 Categorical 对象，指定初始的分类数据
        cat = Categorical(["a", "b", "c", "a"])
        # 创建一个 Series，将上述的 Categorical 对象作为数据，指定复制 (copy) 为 True
        s = Series(cat, copy=True)
        # 断言新创建的 Series 的 cat 属性不等于原始的 Categorical 对象
        assert s.cat is not cat
        # 修改新创建的 Series 的分类数据，并进行断言
        s = s.cat.rename_categories([1, 2, 3])
        # 创建预期的 numpy 数组
        exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
        # 创建预期的 numpy 数组，保持原始的分类数据不变
        exp_cat = np.array(["a", "b", "c", "a"], dtype=np.object_)
        # 断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(s.__array__(), exp_s)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        # 修改 Series 中的一个值
        s[0] = 2
        # 创建预期的修改后的 numpy 数组
        exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
        # 再次断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(s.__array__(), exp_s2)
        tm.assert_numpy_array_equal(cat.__array__(), exp_cat)

        # 默认情况下，copy 参数为 False
        # 创建一个新的 Categorical 对象，指定初始分类数据
        cat = Categorical(["a", "b", "c", "a"])
        # 创建一个 Series，将上述的 Categorical 对象作为数据，copy 参数为 False
        s = Series(cat, copy=False)
        # 断言新创建的 Series 的 values 属性与原始的 Categorical 对象相同
        assert s.values is cat
        # 修改新创建的 Series 的分类数据，并进行断言
        s = s.cat.rename_categories([1, 2, 3])
        # 断言新创建的 Series 的 values 属性不等于原始的 Categorical 对象
        assert s.values is not cat
        # 再次创建预期的 numpy 数组
        exp_s = np.array([1, 2, 3, 1], dtype=np.int64)
        # 再次断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(s.__array__(), exp_s)

        # 修改 Series 中的一个值
        s[0] = 2
        # 创建预期的修改后的 numpy 数组
        exp_s2 = np.array([2, 2, 3, 1], dtype=np.int64)
        # 再次断言两个 numpy 数组是否相等
        tm.assert_numpy_array_equal(s.__array__(), exp_s2)

    def test_unordered_compare_equal(self):
        # 创建一个 Series，指定初始数据和自定义的无序分类数据类型
        left = Series(["a", "b", "c"], dtype=CategoricalDtype(["a", "b"]))
        # 创建一个 Series，使用 Categorical 对象作为数据，指定分类列表和数据类型
        right = Series(Categorical(["a", "b", np.nan], categories=["a", "b"]))
        # 断言两个 Series 是否相等
        tm.assert_series_equal(left, right)

    def test_constructor_maskedarray_hardened(self):
        # 创建一个 numpy 的 masked array，所有值均为 NaN，指定数据类型为 float
        data = ma.masked_all((3,), dtype=float).harden_mask()
        # 使用上述的 masked array 创建一个 Series
        result = Series(data)
        # 创建预期的 Series，包含 NaN 值
        expected = Series([np.nan, np.nan, np.nan])
        # 断言两个 Series 是否相等
        tm.assert_series_equal(result, expected)

    def test_series_ctor_plus_datetimeindex(self):
        # 创建一个日期范围，从 "20090415" 到 "20090519"，频率为工作日 (B)
        rng = date_range("20090415", "20090519", freq="B")
        # 创建一个字典，键为日期范围中的日期，值为 1
        data = {k: 1 for k in rng}

        # 使用上述的字典和日期范围创建一个 Series
        result = Series(data, index=rng)
        # 断言结果的索引是否与原始的日期范围相同
        assert result.index.is_(rng)

    def test_constructor_default_index(self):
        # 创建一个 Series，指定初始数据为 [0, 1, 2]
        s = Series([0, 1, 2])
        # 断言结果的索引是否与预期的整数索引 [0, 1, 2] 相同，严格比较
        tm.assert_index_equal(s.index, Index(range(3)), exact=True)
    @pytest.mark.parametrize(
        "input",
        [
            [1, 2, 3],  # 定义测试参数为列表
            (1, 2, 3),  # 定义测试参数为元组
            list(range(3)),  # 定义测试参数为列表
            Categorical(["a", "b", "a"]),  # 定义分类变量作为测试参数
            (i for i in range(3)),  # 定义生成器表达式作为测试参数
            (x for x in range(3)),  # 定义生成器表达式作为测试参数
        ],
    )
    def test_constructor_index_mismatch(self, input):
        # GH 19342
        # 测试Series构造函数在索引长度不匹配时是否会引发错误
        msg = r"Length of values \(3\) does not match length of index \(4\)"
        with pytest.raises(ValueError, match=msg):
            Series(input, index=np.arange(4))

    def test_constructor_numpy_scalar(self):
        # GH 19342
        # 使用numpy标量进行构造
        # 应该不会引发错误
        result = Series(np.array(100), index=np.arange(4), dtype="int64")
        expected = Series(100, index=np.arange(4), dtype="int64")
        tm.assert_series_equal(result, expected)

    def test_constructor_broadcast_list(self):
        # GH 19342
        # 使用单元素容器和索引进行构造
        # 应该会引发错误
        msg = r"Length of values \(1\) does not match length of index \(3\)"
        with pytest.raises(ValueError, match=msg):
            Series(["foo"], index=["a", "b", "c"])

    def test_constructor_corner(self):
        # 创建一个DataFrame，并使用其构造Series
        df = DataFrame(range(5), index=date_range("2020-01-01", periods=5))
        objs = [df, df]
        s = Series(objs, index=[0, 1])
        assert isinstance(s, Series)

    def test_constructor_sanitize(self):
        # 使用指定dtype构造Series
        s = Series(np.array([1.0, 1.0, 8.0]), dtype="i8")
        assert s.dtype == np.dtype("i8")

        # 应该会引发IntCastingNaNError错误，因为无法将非有限值转换为整数
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(np.array([1.0, 1.0, np.nan]), copy=True, dtype="i8")

    def test_constructor_copy(self):
        # GH15125
        # 测试dtype参数在copy=True时不会产生副作用
        for data in [[1.0], np.array([1.0])]:
            x = Series(data)
            y = Series(x, copy=True, dtype=float)

            # copy=True会保持Series中原始数据
            tm.assert_series_equal(x, y)

            # 对原始数据的更改不会影响副本
            x[0] = 2.0
            assert not x.equals(y)
            assert x[0] == 2.0
            assert y[0] == 1.0

    @pytest.mark.parametrize(
        "index",
        [
            date_range("20170101", periods=3, tz="US/Eastern"),
            date_range("20170101", periods=3),
            timedelta_range("1 day", periods=3),
            period_range("2012Q1", periods=3, freq="Q"),
            Index(list("abc")),
            Index([1, 2, 3]),
            RangeIndex(0, 3),
        ],
        ids=lambda x: type(x).__name__,
    )
    # 测试构造函数是否限制输入的副本数量
    def test_constructor_limit_copies(self, index):
        # GH 17449
        # 在测试中创建一个 Series 对象，用于限制输入的副本数量
        s = Series(index)

        # 进行一次复制检验；这里仅为一个基本的测试
        assert s._mgr.blocks[0].values is not index

    # 测试构造函数中使用 copy=False 时的浅拷贝行为
    def test_constructor_shallow_copy(self):
        # constructing a Series from Series with copy=False should still
        # give a "shallow" copy (share data, not attributes)
        # https://github.com/pandas-dev/pandas/issues/49523
        # 创建一个包含整数的 Series 对象
        s = Series([1, 2, 3])
        # 创建 s 的浅拷贝 s_orig
        s_orig = s.copy()
        # 使用 s 创建另一个 Series 对象 s2
        s2 = Series(s)
        # 断言 s2 的数据管理器不同于 s 的数据管理器
        assert s2._mgr is not s._mgr
        # 修改 s2 的索引不会改变 s 的内容
        s2.index = ["a", "b", "c"]
        # 使用测试模块 tm 来检查 s 和 s_orig 是否相等
        tm.assert_series_equal(s, s_orig)

    # 测试构造函数中传递 None 的情况
    def test_constructor_pass_none(self):
        # 创建一个包含 None 值的 Series，指定索引为 range(5)
        s = Series(None, index=range(5))
        # 断言生成的 Series 的数据类型为 np.float64
        assert s.dtype == np.float64

        # 创建一个包含 None 值的 Series，指定索引为 range(5)，数据类型为 object
        s = Series(None, index=range(5), dtype=object)
        # 断言生成的 Series 的数据类型为 np.object_
        assert s.dtype == np.object_

        # GH 7431
        # 推断索引的类型
        # 创建一个包含 None 元素的 numpy 数组作为 Series 的索引
        s = Series(index=np.array([None]))
        # 期望的 Series 对象，索引类型为 Index([None])
        expected = Series(index=Index([None]))
        # 使用测试模块 tm 来检查 s 和 expected 是否相等
        tm.assert_series_equal(s, expected)

    # 测试构造函数中传递 np.nan 和 NaT 的情况
    def test_constructor_pass_nan_nat(self):
        # GH 13467
        # 创建一个期望的 Series 对象，包含 np.nan 值，数据类型为 np.float64
        exp = Series([np.nan, np.nan], dtype=np.float64)
        # 断言生成的 Series 的数据类型为 np.float64
        assert exp.dtype == np.float64
        # 使用测试模块 tm 来检查生成的 Series 是否与期望的 exp 相等
        tm.assert_series_equal(Series([np.nan, np.nan]), exp)
        tm.assert_series_equal(Series(np.array([np.nan, np.nan])), exp)

        # 创建一个期望的 Series 对象，包含 NaT 值，数据类型为 datetime64[s]
        exp = Series([NaT, NaT])
        # 断言生成的 Series 的数据类型为 datetime64[s]
        assert exp.dtype == "datetime64[s]"
        # 使用测试模块 tm 来检查生成的 Series 是否与期望的 exp 相等
        tm.assert_series_equal(Series([NaT, NaT]), exp)
        tm.assert_series_equal(Series(np.array([NaT, NaT])), exp)

        # 使用测试模块 tm 来检查生成的 Series 是否与期望的 exp 相等
        tm.assert_series_equal(Series([NaT, np.nan]), exp)
        tm.assert_series_equal(Series(np.array([NaT, np.nan])), exp)

        # 使用测试模块 tm 来检查生成的 Series 是否与期望的 exp 相等
        tm.assert_series_equal(Series([np.nan, NaT]), exp)
        tm.assert_series_equal(Series(np.array([np.nan, NaT])), exp)

    # 测试构造函数中数据类型转换时的异常情况
    def test_constructor_cast(self):
        # 准备错误消息字符串
        msg = "could not convert string to float"
        # 使用 pytest 检测是否会抛出 ValueError 异常，异常消息匹配指定的 msg
        with pytest.raises(ValueError, match=msg):
            Series(["a", "b", "c"], dtype=float)

    # 测试构造函数中有符号整数溢出时是否会引发异常
    def test_constructor_signed_int_overflow_raises(self):
        # GH#41734 禁止静默溢出，强制在 2.0 版本中生效
        if np_version_gt2:
            # 准备错误消息字符串
            msg = "The elements provided in the data cannot all be casted to the dtype"
            # 准备异常类型
            err = OverflowError
        else:
            # 准备错误消息字符串
            msg = "Values are too large to be losslessly converted"
            # 准备异常类型
            err = ValueError
        # 使用 pytest 检测是否会抛出指定类型的异常，异常消息匹配指定的 msg
        with pytest.raises(err, match=msg):
            Series([1, 200, 923442], dtype="int8")

        with pytest.raises(err, match=msg):
            Series([1, 200, 923442], dtype="uint8")

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1], dtype=np.uint16),
            np.array([1], dtype=np.uint32),
            np.array([1], dtype=np.uint64),
            [np.uint16(1)],
            [np.uint32(1)],
            [np.uint64(1)],
        ],
    )
    # 定义测试函数，用于测试构造函数处理 NumPy 无符号整数的情况
    def test_constructor_numpy_uints(self, values):
        # 从参数 values 中取出第一个值
        value = values[0]
        # 使用 Series 构造函数创建一个 Series 对象
        result = Series(values)

        # 断言第一个元素的数据类型应与 value 的数据类型相同
        assert result[0].dtype == value.dtype
        # 断言第一个元素的数值应与 value 相等
        assert result[0] == value

    # 定义测试函数，用于测试构造函数处理无符号数据类型溢出的情况
    def test_constructor_unsigned_dtype_overflow(self, any_unsigned_int_numpy_dtype):
        # 如果 NumPy 的版本大于 2
        if np_version_gt2:
            # 设置错误信息，说明数据中的元素不能全部转换为指定的无符号整数数据类型
            msg = (
                f"The elements provided in the data cannot "
                f"all be casted to the dtype {any_unsigned_int_numpy_dtype}"
            )
        else:
            # 设置错误信息，说明试图将负值强制转换为无符号整数
            msg = "Trying to coerce negative values to unsigned integers"
        # 使用 pytest 的断言来检查是否抛出 OverflowError 异常，并匹配错误信息
        with pytest.raises(OverflowError, match=msg):
            Series([-1], dtype=any_unsigned_int_numpy_dtype)

    # 定义测试函数，用于测试构造函数处理浮点数据与整数数据类型的情况
    def test_constructor_floating_data_int_dtype(self, frame_or_series):
        # GH#40110
        # 生成一个包含两个标准正态分布随机数的数组
        arr = np.random.default_rng(2).standard_normal(2)

        # 在 Series 和 DataFrame 中的长期行为（Series 中的新行为在 2.0 版本中开始）
        # 是忽略这些情况下的数据类型设置；
        # 目前不清楚这是否是长期期望的行为方式

        # 在 2.0 版本中，我们会抛出异常而不是默默地保留浮点数数据类型
        msg = "Trying to coerce float values to integer"
        with pytest.raises(ValueError, match=msg):
            frame_or_series(arr, dtype="i8")

        with pytest.raises(ValueError, match=msg):
            frame_or_series(list(arr), dtype="i8")

        # 在 2.0 之前，当存在 NaN 时，我们会默默地忽略整数数据类型
        arr[0] = np.nan
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            frame_or_series(arr, dtype="i8")

        # 出现异常类型为 IntCastingNaNError
        exc = IntCastingNaNError
        if frame_or_series is Series:
            # TODO: 尝试对齐这些
            exc = ValueError
            msg = "cannot convert float NaN to integer"
        with pytest.raises(exc, match=msg):
            # 如果传递的是列表而不是 ndarray，行为相同
            frame_or_series(list(arr), dtype="i8")

        # 浮点数数组可以无损地转换为整数
        arr = np.array([1.0, 2.0], dtype="float64")
        # 期望的结果是将 arr 转换为 i8 类型后的 frame_or_series 对象
        expected = frame_or_series(arr.astype("i8"))

        # 使用 dtype="i8" 参数创建 frame_or_series 对象
        obj = frame_or_series(arr, dtype="i8")
        # 使用 tm.assert_equal 来比较 obj 和 expected 是否相等
        tm.assert_equal(obj, expected)

        # 使用列表作为输入创建 frame_or_series 对象
        obj = frame_or_series(list(arr), dtype="i8")
        # 使用 tm.assert_equal 来比较 obj 和 expected 是否相等
        tm.assert_equal(obj, expected)
    def test_constructor_coerce_float_fail(self, any_int_numpy_dtype):
        # 用例名称：test_constructor_coerce_float_fail
        # 输入参数：any_int_numpy_dtype，任意整数类型的NumPy数据类型
        # 目的：测试在尝试将浮点数强制转换为整数时是否引发值错误异常
        # GH 15832：GitHub上的问题编号，参考特定的问题或变更
        # 更新：确保我们像处理等效的ndarray一样处理此列表
        # GH#49599：GitHub上的问题编号，指明在2.0之前我们悄悄保留了浮点数dtype，在2.0中我们引发异常
        vals = [1, 2, 3.5]

        msg = "Trying to coerce float values to integer"
        # 使用pytest检查是否引发值错误异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            Series(vals, dtype=any_int_numpy_dtype)
        with pytest.raises(ValueError, match=msg):
            Series(np.array(vals), dtype=any_int_numpy_dtype)

    def test_constructor_coerce_float_valid(self, float_numpy_dtype):
        # 用例名称：test_constructor_coerce_float_valid
        # 输入参数：float_numpy_dtype，浮点数的NumPy数据类型
        # 目的：测试在指定浮点数dtype时，Series构造函数的行为是否正确
        s = Series([1, 2, 3.5], dtype=float_numpy_dtype)
        expected = Series([1, 2, 3.5]).astype(float_numpy_dtype)
        # 使用tm.assert_series_equal断言确保s和expected的内容相等
        tm.assert_series_equal(s, expected)

    def test_constructor_invalid_coerce_ints_with_float_nan(self, any_int_numpy_dtype):
        # 用例名称：test_constructor_invalid_coerce_ints_with_float_nan
        # 输入参数：any_int_numpy_dtype，任意整数类型的NumPy数据类型
        # 目的：测试当列表中包含NaN时，尝试强制转换为整数是否引发正确的异常
        # GH 22585：GitHub上的问题编号，指明需要特别关注如何处理NaN值
        # 更新：确保我们像处理等效的ndarray一样处理此列表
        vals = [1, 2, np.nan]
        # 在2.0之前会返回浮点数dtype，而在2.0中我们引发异常

        msg = "cannot convert float NaN to integer"
        # 使用pytest检查是否引发值错误异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            Series(vals, dtype=any_int_numpy_dtype)
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        # 使用pytest检查是否引发IntCastingNaNError异常，并匹配特定的错误消息
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(np.array(vals), dtype=any_int_numpy_dtype)

    def test_constructor_dtype_no_cast(self):
        # 用例名称：test_constructor_dtype_no_cast
        # 目的：测试在指定dtype时，Series构造函数是否正确处理不强制转换的情况
        # GH 1572：GitHub上的问题编号，指明特定的问题或变更
        s = Series([1, 2, 3])
        s2 = Series(s, dtype=np.int64)

        s2[1] = 5
        # 使用assert断言确保s的第一个元素未被修改
        assert s[1] == 2

    def test_constructor_datelike_coercion(self):
        # 用例名称：test_constructor_datelike_coercion
        # 目的：测试在指定对象dtype时，是否正确推断日期时间样式的内容
        # GH 9477：GitHub上的问题编号，指明特定的问题或变更
        # 更新：在指定对象dtype时，确保正确推断日期时间样式的内容
        s = Series([Timestamp("20130101"), "NOV"], dtype=object)
        # 使用assert断言确保第一个元素与预期的时间戳相等
        assert s.iloc[0] == Timestamp("20130101")
        # 使用assert断言确保第二个元素与预期的字符串相等
        assert s.iloc[1] == "NOV"
        # 使用assert断言确保Series的dtype为object
        assert s.dtype == object

    def test_constructor_datelike_coercion2(self):
        # 用例名称：test_constructor_datelike_coercion2
        # 目的：测试在切片和重新推断混合块时，是否正确保持dtype
        # GH 22585：GitHub上的问题编号，指明特定的问题或变更
        belly = "216 3T19".split()
        wing1 = "2T15 4H19".split()
        wing2 = "416 4T20".split()
        mat = pd.to_datetime("2016-01-22 2019-09-07".split())
        df = DataFrame({"wing1": wing1, "wing2": wing2, "mat": mat}, index=belly)

        result = df.loc["3T19"]
        # 使用assert断言确保结果的dtype为object
        assert result.dtype == object
        result = df.loc["216"]
        # 使用assert断言确保结果的dtype为object
        assert result.dtype == object

    def test_constructor_mixed_int_and_timestamp(self, frame_or_series):
        # 用例名称：test_constructor_mixed_int_and_timestamp
        # 输入参数：frame_or_series，DataFrame或Series对象
        # 目的：测试特定情况下的构造行为，包含Timestamp和NaN值
        # specifically Timestamp with nanos, not datetimes
        objs = [Timestamp(9), 10, NaT._value]
        # 使用frame_or_series函数创建结果对象，指定dtype为"M8[ns]"
        result = frame_or_series(objs, dtype="M8[ns]")

        expected = frame_or_series([Timestamp(9), Timestamp(10), NaT])
        # 使用tm.assert_equal断言确保result和expected相等
        tm.assert_equal(result, expected)
    # 测试函数：test_constructor_datetimes_with_nulls
    def test_constructor_datetimes_with_nulls(self):
        # 测试用例名称：gh-15869
        for arr in [
            np.array([None, None, None, None, datetime.now(), None]),  # 创建包含空值和当前日期时间的 NumPy 数组
            np.array([None, None, datetime.now(), None]),  # 创建包含空值和当前日期时间的 NumPy 数组
        ]:
            result = Series(arr)  # 使用 NumPy 数组创建 Series 对象
            assert result.dtype == "M8[us]"  # 断言 Series 对象的数据类型为微秒级的 datetime64

    # 测试函数：test_constructor_dtype_datetime64
    def test_constructor_dtype_datetime64(self):
        s = Series(iNaT, dtype="M8[ns]", index=range(5))  # 使用 iNaT 创建 datetime64 纳秒级 Series 对象
        assert isna(s).all()  # 断言 Series 对象中所有值都是缺失值

        # 理论上应该全是空值，但由于没有指定 dtype，是模棱两可的
        s = Series(iNaT, index=range(5))  # 使用 iNaT 创建默认 dtype 的 Series 对象
        assert not isna(s).all()  # 断言 Series 对象中不是所有值都是缺失值

        s = Series(np.nan, dtype="M8[ns]", index=range(5))  # 使用 NaN 创建 datetime64 纳秒级 Series 对象
        assert isna(s).all()  # 断言 Series 对象中所有值都是缺失值

        s = Series([datetime(2001, 1, 2, 0, 0), iNaT], dtype="M8[ns]")  # 创建包含 datetime 和 iNaT 的 datetime64 纳秒级 Series 对象
        assert isna(s[1])  # 断言 Series 对象中第二个元素是缺失值
        assert s.dtype == "M8[ns]"  # 断言 Series 对象的数据类型为纳秒级的 datetime64

        s = Series([datetime(2001, 1, 2, 0, 0), np.nan], dtype="M8[ns]")  # 创建包含 datetime 和 NaN 的 datetime64 纳秒级 Series 对象
        assert isna(s[1])  # 断言 Series 对象中第二个元素是缺失值
        assert s.dtype == "M8[ns]"  # 断言 Series 对象的数据类型为纳秒级的 datetime64

    # 测试函数：test_constructor_dtype_datetime64_10
    def test_constructor_dtype_datetime64_10(self):
        # GH3416 相关测试
        pydates = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3)]
        dates = [np.datetime64(x) for x in pydates]  # 创建 datetime64 数组

        ser = Series(dates)  # 使用 datetime64 数组创建 Series 对象
        assert ser.dtype == "M8[us]"  # 断言 Series 对象的数据类型为微秒级的 datetime64

        ser.iloc[0] = np.nan  # 将 Series 对象中第一个元素设置为缺失值
        assert ser.dtype == "M8[us]"  # 断言 Series 对象的数据类型为微秒级的 datetime64

        # GH3414 相关测试
        expected = Series(pydates, dtype="datetime64[ms]")  # 创建指定数据类型的 datetime64 毫秒级 Series 对象

        result = Series(Series(dates).astype(np.int64) / 1000, dtype="M8[ms]")  # 创建转换后的毫秒级 Series 对象
        tm.assert_series_equal(result, expected)  # 使用测试框架断言两个 Series 对象相等

        result = Series(dates, dtype="datetime64[ms]")  # 创建毫秒级的 datetime64 Series 对象
        tm.assert_series_equal(result, expected)  # 使用测试框架断言两个 Series 对象相等

        expected = Series(
            [NaT, datetime(2013, 1, 2), datetime(2013, 1, 3)], dtype="datetime64[ns]"
        )  # 创建纳秒级的 datetime64 Series 对象

        result = Series([np.nan] + dates[1:], dtype="datetime64[ns]")  # 创建包含缺失值和部分日期的纳秒级 Series 对象
        tm.assert_series_equal(result, expected)  # 使用测试框架断言两个 Series 对象相等

    # 测试函数：test_constructor_dtype_datetime64_11
    def test_constructor_dtype_datetime64_11(self):
        pydates = [datetime(2013, 1, 1), datetime(2013, 1, 2), datetime(2013, 1, 3)]
        dates = [np.datetime64(x) for x in pydates]  # 创建 datetime64 数组

        dts = Series(dates, dtype="datetime64[ns]")  # 使用 datetime64 数组创建纳秒级 Series 对象

        # valid astype
        dts.astype("int64")  # 将 Series 对象转换为 int64 类型

        # invalid casting
        msg = r"Converting from datetime64\[ns\] to int32 is not supported"
        with pytest.raises(TypeError, match=msg):  # 断言抛出特定类型和消息的异常
            dts.astype("int32")  # 尝试将 Series 对象转换为 int32 类型

        # ints are ok
        # we test with np.int64 to get similar results on
        # windows / 32-bit platforms
        result = Series(dts, dtype=np.int64)  # 将 Series 对象转换为 np.int64 类型
        expected = Series(dts.astype(np.int64))  # 使用 astype 转换后的 Series 对象
        tm.assert_series_equal(result, expected)  # 使用测试框架断言两个 Series 对象相等

    # 测试函数：test_constructor_dtype_datetime64_9
    def test_constructor_dtype_datetime64_9(self):
        # invalid dates can be help as object
        result = Series([datetime(2, 1, 1)])  # 创建包含特定日期的 Series 对象
        assert result[0] == datetime(2, 1, 1, 0, 0)  # 断言 Series 对象中第一个元素的值为特定日期

        result = Series([datetime(3000, 1, 1)])  # 创建包含特定日期的 Series 对象
        assert result[0] == datetime(3000, 1, 1, 0, 0)  # 断言 Series 对象中第一个元素的值为特定日期
    def test_constructor_dtype_datetime64_8(self):
        # 测试：构造函数，确保不混合类型
        result = Series([Timestamp("20130101"), 1], index=["a", "b"])
        assert result["a"] == Timestamp("20130101")
        assert result["b"] == 1

    def test_constructor_dtype_datetime64_7(self):
        # GH6529
        # 测试：强制转换 datetime64 非纳秒精度
        dates = date_range("01-Jan-2015", "01-Dec-2015", freq="ME")
        values2 = dates.view(np.ndarray).astype("datetime64[ns]")
        expected = Series(values2, index=dates)

        for unit in ["s", "D", "ms", "us", "ns"]:
            dtype = np.dtype(f"M8[{unit}]")
            values1 = dates.view(np.ndarray).astype(dtype)
            result = Series(values1, dates)
            if unit == "D":
                # 对于单位为 "D" 的情况，强制转换为最接近支持的精度，即 "s"
                dtype = np.dtype("M8[s]")
            assert result.dtype == dtype
            tm.assert_series_equal(result, expected.astype(dtype))

        # GH 13876
        # 测试：将 datetime64 转换为 object 类型
        expected = Series(values2, index=dates, dtype=object)
        for dtype in ["s", "D", "ms", "us", "ns"]:
            values1 = dates.view(np.ndarray).astype(f"M8[{dtype}]")
            result = Series(values1, index=dates, dtype=object)
            tm.assert_series_equal(result, expected)

        # 保持 datetime.date 不变
        dates2 = np.array([d.date() for d in dates.to_pydatetime()], dtype=object)
        series1 = Series(dates2, dates)
        tm.assert_numpy_array_equal(series1.values, dates2)
        assert series1.dtype == object

    def test_constructor_dtype_datetime64_6(self):
        # 从 2.0 版本开始，不再根据字符串推断 datetime64 类型，与 Index 行为匹配

        ser = Series([None, NaT, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

        ser = Series([np.nan, NaT, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

        ser = Series([NaT, None, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

        ser = Series([NaT, np.nan, "2013-08-05 15:30:00.000001"])
        assert ser.dtype == object

    def test_constructor_dtype_datetime64_5(self):
        # 时区感知（UTC 和其他时区）
        # GH 8411
        dr = date_range("20130101", periods=3)
        assert Series(dr).iloc[0].tz is None
        dr = date_range("20130101", periods=3, tz="UTC")
        assert str(Series(dr).iloc[0].tz) == "UTC"
        dr = date_range("20130101", periods=3, tz="US/Eastern")
        assert str(Series(dr).iloc[0].tz) == "US/Eastern"

    def test_constructor_dtype_datetime64_4(self):
        # 不可转换的情况
        ser = Series([1479596223000, -1479590, NaT])
        assert ser.dtype == "object"
        assert ser[2] is NaT
        assert "NaT" in str(ser)
    def test_constructor_dtype_datetime64_3(self):
        # 创建一个包含日期时间对象的序列，包括一个 NaT（Not a Time）对象
        ser = Series([datetime(2010, 1, 1), datetime(2, 1, 1), NaT])
        # 断言序列的数据类型是 "M8[us]"
        assert ser.dtype == "M8[us]"
        # 断言序列中第三个元素是 NaT
        assert ser[2] is NaT
        # 断言字符串 "NaT" 出现在序列的字符串表示中
        assert "NaT" in str(ser)

    def test_constructor_dtype_datetime64_2(self):
        # 创建一个包含日期时间对象的序列，包括一个 np.nan 对象
        ser = Series([datetime(2010, 1, 1), datetime(2, 1, 1), np.nan])
        # 断言序列的数据类型是 "M8[us]"
        assert ser.dtype == "M8[us]"
        # 断言序列中第三个元素是 NaT
        assert ser[2] is NaT
        # 断言字符串 "NaT" 出现在序列的字符串表示中
        assert "NaT" in str(ser)

    def test_constructor_with_datetime_tz(self):
        # 测试支持带时区的 datetime64 对象
        dr = date_range("20130101", periods=3, tz="US/Eastern")
        s = Series(dr)
        # 断言序列的数据类型名称是 "datetime64[ns, US/Eastern]"
        assert s.dtype.name == "datetime64[ns, US/Eastern]"
        # 断言序列的数据类型是 "datetime64[ns, US/Eastern]"
        assert s.dtype == "datetime64[ns, US/Eastern]"
        # 断言序列的数据类型是 DatetimeTZDtype 类的实例
        assert isinstance(s.dtype, DatetimeTZDtype)
        # 断言字符串 "datetime64[ns, US/Eastern]" 出现在序列的字符串表示中
        assert "datetime64[ns, US/Eastern]" in str(s)

        # 导出
        result = s.values
        # 断言导出结果是一个 NumPy 数组
        assert isinstance(result, np.ndarray)
        # 断言导出结果的数据类型是 "datetime64[ns]"
        assert result.dtype == "datetime64[ns]"

        exp = DatetimeIndex(result)
        # 将预期结果本地化为 UTC，然后转换时区为序列 s 的时区
        exp = exp.tz_localize("UTC").tz_convert(tz=s.dt.tz)
        # 使用 tm.assert_index_equal 检查日期范围 dr 和 exp 是否相等
        tm.assert_index_equal(dr, exp)

        # 索引
        result = s.iloc[0]
        # 断言索引结果与指定的时间戳相等，带有 "US/Eastern" 时区
        assert result == Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern")
        result = s[0]
        # 断言索引结果与指定的时间戳相等，带有 "US/Eastern" 时区
        assert result == Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern")

        result = s[Series([True, True, False], index=s.index)]
        # 使用 tm.assert_series_equal 检查结果与 s[0:2] 是否相等
        tm.assert_series_equal(result, s[0:2])

        result = s.iloc[0:1]
        # 使用 tm.assert_series_equal 检查结果与由 dr 的第一个元素构成的 Series 是否相等
        tm.assert_series_equal(result, Series(dr[0:1]))

        # 连接
        result = pd.concat([s.iloc[0:1], s.iloc[1:]])
        # 使用 tm.assert_series_equal 检查结果与序列 s 是否相等
        tm.assert_series_equal(result, s)

        # 短字符串表示
        # 断言字符串 "datetime64[ns, US/Eastern]" 出现在序列的字符串表示中
        assert "datetime64[ns, US/Eastern]" in str(s)

        # 使用 NaT 进行格式化
        result = s.shift()
        # 断言字符串 "datetime64[ns, US/Eastern]" 出现在结果的字符串表示中
        assert "datetime64[ns, US/Eastern]" in str(result)
        # 断言字符串 "NaT" 出现在结果的字符串表示中
        assert "NaT" in str(result)

        result = DatetimeIndex(s, freq="infer")
        # 使用 tm.assert_index_equal 检查结果与日期范围 dr 是否相等
        tm.assert_index_equal(result, dr)

    def test_constructor_with_datetime_tz5(self):
        # 长字符串表示
        ser = Series(date_range("20130101", periods=1000, tz="US/Eastern"))
        # 断言字符串 "datetime64[ns, US/Eastern]" 出现在序列的字符串表示中
        assert "datetime64[ns, US/Eastern]" in str(ser)

    def test_constructor_with_datetime_tz4(self):
        # 推断
        ser = Series(
            [
                Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
                Timestamp("2013-01-02 14:00:00-0800", tz="US/Pacific"),
            ]
        )
        # 断言序列的数据类型是 "datetime64[s, US/Pacific]"
        assert ser.dtype == "datetime64[s, US/Pacific]"
        # 使用 lib.infer_dtype 推断序列的数据类型，确保跳过 NaN 值后得到 "datetime64"
        assert lib.infer_dtype(ser, skipna=True) == "datetime64"
    # 测试用例：使用带有时区信息的时间戳构造 Series 对象
    def test_constructor_with_datetime_tz3(self):
        # 创建包含时区信息的时间戳对象列表
        ser = Series(
            [
                Timestamp("2013-01-01 13:00:00-0800", tz="US/Pacific"),
                Timestamp("2013-01-02 14:00:00-0800", tz="US/Eastern"),
            ]
        )
        # 断言 Series 对象的数据类型为 "object"
        assert ser.dtype == "object"
        # 使用 lib.infer_dtype 推断 Series 的数据类型为 "datetime"
        assert lib.infer_dtype(ser, skipna=True) == "datetime"

    # 测试用例：使用全为 NaT 的 Series 对象，指定时区为 US/Eastern
    def test_constructor_with_datetime_tz2(self):
        # 创建全为 NaT 的 Series 对象，指定索引和数据类型
        ser = Series(NaT, index=[0, 1], dtype="datetime64[ns, US/Eastern]")
        # 创建预期的 DatetimeIndex 对象，其中包含两个 NaT 值，时区为 US/Eastern
        dti = DatetimeIndex(["NaT", "NaT"], tz="US/Eastern").as_unit("ns")
        expected = Series(dti)
        # 断言两个 Series 对象相等
        tm.assert_series_equal(ser, expected)

    # 测试用例：构造 Series 对象，包含各种形式的日期时间值
    def test_constructor_no_partial_datetime_casting(self):
        # GH#40111
        # 定义包含不同日期时间值的列表
        vals = [
            "nan",
            Timestamp("1990-01-01"),
            "2015-03-14T16:15:14.123-08:00",
            "2019-03-04T21:56:32.620-07:00",
            None,
        ]
        # 创建包含上述值的 Series 对象
        ser = Series(vals)
        # 断言所有索引位置上的值与预期值相等
        assert all(ser[i] is vals[i] for i in range(len(vals)))

    # 测试用例：按指定单位将数组转换为日期时间类型的 Series 对象
    @pytest.mark.parametrize("arr_dtype", [np.int64, np.float64])
    @pytest.mark.parametrize("kind", ["M", "m"])
    @pytest.mark.parametrize("unit", ["ns", "us", "ms", "s", "h", "m", "D"])
    def test_construction_to_datetimelike_unit(self, arr_dtype, kind, unit):
        # tests all units
        # gh-19223
        # TODO: GH#19223 was about .astype, doesn't belong here
        # 构造指定类型和单位的 dtype 字符串
        dtype = f"{kind}8[{unit}]"
        # 创建包含整数或浮点数的 numpy 数组
        arr = np.array([1, 2, 3], dtype=arr_dtype)
        # 创建包含上述数组的 Series 对象
        ser = Series(arr)
        # 将 Series 对象的数据类型转换为指定的 dtype
        result = ser.astype(dtype)

        # 创建预期的 Series 对象，其数据类型转换为指定的 dtype
        expected = Series(arr.astype(dtype))

        if unit in ["ns", "us", "ms", "s"]:
            # 断言结果和预期的 Series 对象的数据类型与指定的 dtype 相等
            assert result.dtype == dtype
            assert expected.dtype == dtype
        else:
            # 否则，断言结果和预期的 Series 对象的数据类型被转换为最接近支持的单位，即秒
            assert result.dtype == f"{kind}8[s]"
            assert expected.dtype == f"{kind}8[s]"

        # 断言结果和预期的 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 测试用例：使用包含字符串和带有时区的日期时间类型构造 Series 对象
    @pytest.mark.parametrize("arg", ["2013-01-01 00:00:00", NaT, np.nan, None])
    def test_constructor_with_naive_string_and_datetimetz_dtype(self, arg):
        # GH 17415: With naive string
        # 使用带有时区信息的字符串构造日期时间类型的 Series 对象
        result = Series([arg], dtype="datetime64[ns, CET]")
        # 创建预期的 Series 对象，其包含 Timestamp 对象，并设置时区为 CET
        expected = Series([Timestamp(arg)], dtype="M8[ns]").dt.tz_localize("CET")
        # 断言结果和预期的 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 测试用例：使用大端日期时间类型构造 Series 对象
    def test_constructor_datetime64_bigendian(self):
        # GH#30976
        # 创建以毫秒为单位的 np.datetime64 对象
        ms = np.datetime64(1, "ms")
        # 创建包含上述日期时间对象的数组，数据类型为 ">M8[ms]"
        arr = np.array([np.datetime64(1, "ms")], dtype=">M8[ms]")

        # 创建包含上述数组的 Series 对象
        result = Series(arr)
        # 创建预期的 Series 对象，其包含 Timestamp 对象，并设置数据类型为 "M8[ms]"
        expected = Series([Timestamp(ms)]).astype("M8[ms]")
        # 断言预期的 Series 对象的数据类型为 "M8[ms]"
        assert expected.dtype == "M8[ms]"
        # 断言结果和预期的 Series 对象相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("interval_constructor", [IntervalIndex, IntervalArray])
    # 测试从间隔和间隔数组构建对象
    def test_construction_interval(self, interval_constructor):
        # 使用间隔构造器从间隔和间隔数组创建间隔
        intervals = interval_constructor.from_breaks(np.arange(3), closed="right")
        # 创建 Series 对象，使用 intervals 作为数据
        result = Series(intervals)
        # 断言 Series 对象的数据类型为 "interval[int64, right]"
        assert result.dtype == "interval[int64, right]"
        # 断言 Series 对象的索引与 intervals 的索引相等
        tm.assert_index_equal(Index(result.values), Index(intervals))

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    # 测试从数据构造器推断间隔对象
    def test_constructor_infer_interval(self, data_constructor):
        # GH 23563: 确保 interval 数据类型的一致性闭合结果
        data = [Interval(0, 1), Interval(0, 2), None]
        # 使用 data_constructor 构造 Series 对象
        result = Series(data_constructor(data))
        # 期望的结果是一个 IntervalArray 类型的 Series 对象
        expected = Series(IntervalArray(data))
        # 断言 Series 对象的数据类型为 "interval[float64, right]"
        assert result.dtype == "interval[float64, right]"
        # 断言 result 和 expected 的内容相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "data_constructor", [list, np.array], ids=["list", "ndarray[object]"]
    )
    # 测试混合闭合状态的间隔对象构造
    def test_constructor_interval_mixed_closed(self, data_constructor):
        # GH 23563: 混合闭合状态应返回 object 数据类型而不是 interval 数据类型
        data = [Interval(0, 1, closed="both"), Interval(0, 2, closed="neither")]
        # 使用 data_constructor 构造 Series 对象
        result = Series(data_constructor(data))
        # 断言 Series 对象的数据类型为 object
        assert result.dtype == object
        # 断言 result 列表与 data 列表相等
        assert result.tolist() == data

    # 测试构造的一致性
    def test_construction_consistency(self):
        # 确保在构造过程中不重新定位
        # GH 14928
        # 创建一个 Series 对象，其中包含美国东部时区的日期范围数据
        ser = Series(date_range("20130101", periods=3, tz="US/Eastern"))

        # 使用相同的 dtype 构造一个新的 Series 对象，并断言两者相等
        result = Series(ser, dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

        # 将 ser 转换为 UTC 时区后，使用相同的 dtype 构造新的 Series 对象，并断言两者相等
        result = Series(ser.dt.tz_convert("UTC"), dtype=ser.dtype)
        tm.assert_series_equal(result, ser)

        # 在 2.0 之前，dt64 值被视为 UTC，这与 DatetimeIndex 的处理方式不一致
        # 参见 GH#33401
        # 使用 ser.values 构造 Series 对象，再将其本地化为 ser.dtype.tz 时区
        result = Series(ser.values, dtype=ser.dtype)
        expected = Series(ser.values).dt.tz_localize(ser.dtype.tz)
        tm.assert_series_equal(result, expected)

        with tm.assert_produces_warning(None):
            # 对于废弃的用法（在 2.0 中更改），以下是建议的替代方法之一
            middle = Series(ser.values).dt.tz_localize("UTC")
            result = middle.dt.tz_convert(ser.dtype.tz)
        tm.assert_series_equal(result, ser)

        with tm.assert_produces_warning(None):
            # 对于废弃的用法（在 2.0 中更改），以下是另一个建议的替代方法
            result = Series(ser.values.view("int64"), dtype=ser.dtype)
        tm.assert_series_equal(result, ser)
    # 测试函数：使用给定的数据构造函数测试 Period 对象的构造
    def test_constructor_infer_period(self, data_constructor):
        # 创建包含 Period 对象、None 的数据列表
        data = [Period("2000", "D"), Period("2001", "D"), None]
        # 使用数据构造函数创建 Series 对象
        result = Series(data_constructor(data))
        # 使用 period_array 函数创建预期的 Series 对象
        expected = Series(period_array(data))
        # 使用测试工具函数检查两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)
        # 断言结果 Series 的数据类型为 "Period[D]"
        assert result.dtype == "Period[D]"

    # 标记为预期失败的测试函数，因为暂不支持 PeriodDtype 的 Series 对象
    @pytest.mark.xfail(reason="PeriodDtype Series not supported yet")
    def test_construct_from_ints_including_iNaT_scalar_period_dtype(self):
        # 创建一个包含整数和 pd._libs.iNaT（非数字）的 Series 对象，数据类型为 "period[D]"
        series = Series([0, 1000, 2000, pd._libs.iNaT], dtype="period[D]")
        
        # 获取第四个元素的值
        val = series[3]
        # 断言第四个元素是否为缺失值（NA）
        assert isna(val)

        # 将第三个元素设置为第四个元素的值
        series[2] = val
        # 断言第三个元素是否为缺失值（NA）
        assert isna(series[2])

    # 测试函数：测试构造包含不同频率 Period 对象的 Series 对象
    def test_constructor_period_incompatible_frequency(self):
        # 创建包含不同频率 Period 对象的数据列表
        data = [Period("2000", "D"), Period("2001", "Y")]
        # 使用数据列表创建 Series 对象
        result = Series(data)
        # 断言结果 Series 的数据类型为 object
        assert result.dtype == object
        # 断言结果 Series 转换为列表后与原始数据列表相等
        assert result.tolist() == data

    # 测试函数：测试从 PeriodIndex 转换为 Series 对象
    def test_constructor_periodindex(self):
        # GH7932：测试在将 PeriodIndex 放入 Series 中时的转换
        pi = period_range("20130101", periods=5, freq="D")
        # 创建包含 PeriodIndex 对象的 Series 对象
        s = Series(pi)
        # 断言结果 Series 的数据类型为 "Period[D]"
        assert s.dtype == "Period[D]"
        # 创建预期的 Series 对象，强制转换其数据类型为 object
        expected = Series(pi.astype(object))
        # 断言预期的 Series 对象数据类型为 object
        assert expected.dtype == object

    # 测试函数：测试从字典创建 Series 对象
    def test_constructor_dict(self):
        d = {"a": 0.0, "b": 1.0, "c": 2.0}

        # 测试情况1：使用字典创建 Series 对象，索引按照字典键的排序顺序
        result = Series(d)
        expected = Series(d, index=sorted(d.keys()))
        tm.assert_series_equal(result, expected)

        # 测试情况2：使用字典和自定义索引创建 Series 对象
        result = Series(d, index=["b", "c", "d", "a"])
        expected = Series([1, 2, np.nan, 0], index=["b", "c", "d", "a"])
        tm.assert_series_equal(result, expected)

        # 测试情况3：使用 PeriodIndex 创建 Series 对象
        pidx = period_range("2020-01-01", periods=10, freq="D")
        d = {pidx[0]: 0, pidx[1]: 1}
        result = Series(d, index=pidx)
        expected = Series(np.nan, pidx, dtype=np.float64)
        expected.iloc[0] = 0
        expected.iloc[1] = 1
        tm.assert_series_equal(result, expected)

    # 测试函数：测试从字典创建 Series 对象，显式指定值的数据类型为 object
    def test_constructor_dict_list_value_explicit_dtype(self):
        # GH 18625：测试从字典创建 Series 对象，显式指定值的数据类型为 object
        d = {"a": [[2], [3], [4]]}
        result = Series(d, index=["a"], dtype="object")
        expected = Series(d, index=["a"])
        tm.assert_series_equal(result, expected)

    # 测试函数：测试从字典创建 Series 对象，检查初始化顺序
    def test_constructor_dict_order(self):
        # GH19018：测试从字典创建 Series 对象时的初始化顺序，按插入顺序排序
        d = {"b": 1, "a": 0, "c": 2}
        result = Series(d)
        expected = Series([1, 0, 2], index=list("bac"))
        tm.assert_series_equal(result, expected)

    # 测试函数：测试从字典创建 Series 对象，检查值和数据类型扩展
    def test_constructor_dict_extension(self, ea_scalar_and_dtype):
        # 从标量和数据类型元组中获取数据
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        d = {"a": ea_scalar}
        result = Series(d, index=["a"])
        expected = Series(ea_scalar, index=["a"], dtype=ea_dtype)

        # 断言结果 Series 的数据类型与期望的数据类型相等
        assert result.dtype == ea_dtype

        # 使用测试工具函数检查两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)

    # 参数化测试函数：使用不同的值进行参数化测试
    @pytest.mark.parametrize("value", [2, np.nan, None, float("nan")])
    def test_constructor_dict_nan_key(self, value):
        # GH 18480
        # 创建包含 NaN 键的字典，键为整数 1、变量 value、NaN、4，对应值为 "a"、"b"、"c"、"d"
        d = {1: "a", value: "b", float("nan"): "c", 4: "d"}
        # 使用创建的字典创建 Series 对象，并对其进行排序
        result = Series(d).sort_values()
        # 期望结果为包含键值对 ["a", "b", "c", "d"] 的 Series 对象，对应的索引为 [1, value, NaN, 4]
        expected = Series(["a", "b", "c", "d"], index=[1, value, np.nan, 4])
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

        # MultiIndex 情况:
        # 创建包含 MultiIndex 的字典，键为元组 (1, 1)、(2, NaN)、(3, value)，对应值为 "a"、"b"、"c"
        d = {(1, 1): "a", (2, np.nan): "b", (3, value): "c"}
        # 使用创建的字典创建 Series 对象，并对其进行排序
        result = Series(d).sort_values()
        # 期望结果为包含键值对 ["a", "b", "c"] 的 Series 对象，对应的索引为 [(1, 1), (2, NaN), (3, value)]
        expected = Series(
            ["a", "b", "c"], index=Index([(1, 1), (2, np.nan), (3, value)])
        )
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    def test_constructor_dict_datetime64_index(self):
        # GH 9456
        # 定义日期字符串列表和相应的数值列表
        dates_as_str = ["1984-02-19", "1988-11-06", "1989-12-03", "1990-03-15"]
        values = [42544017.198965244, 1234565, 40512335.181958228, -1]

        # 根据构造函数创建日期数据的字典
        def create_data(constructor):
            return dict(zip((constructor(x) for x in dates_as_str), values))

        # 使用不同的构造函数创建日期数据的字典
        data_datetime64 = create_data(np.datetime64)
        data_datetime = create_data(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        data_Timestamp = create_data(Timestamp)

        # 根据创建的日期数据字典分别创建 Series 对象
        expected = Series(values, (Timestamp(x) for x in dates_as_str))
        result_datetime64 = Series(data_datetime64)
        result_datetime = Series(data_datetime)
        result_Timestamp = Series(data_Timestamp)

        # 断言各 Series 对象与期望的 Series 对象相等
        tm.assert_series_equal(result_datetime64, expected)
        tm.assert_series_equal(
            result_datetime, expected.set_axis(expected.index.as_unit("us"))
        )
        tm.assert_series_equal(result_Timestamp, expected)

    def test_constructor_dict_tuple_indexer(self):
        # GH 12948
        # 创建包含元组索引的字典，键为 (1, 1, None)，对应值为 -1.0
        data = {(1, 1, None): -1.0}
        # 使用创建的字典创建 Series 对象
        result = Series(data)
        # 期望结果为值为 -1.0 的 Series 对象，索引为 MultiIndex，层级为 [[1], [1], [NaN]]，对应编码为 [[0], [0], [-1]]
        expected = Series(
            -1.0, index=MultiIndex(levels=[[1], [1], [np.nan]], codes=[[0], [0], [-1]])
        )
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    def test_constructor_mapping(self, non_dict_mapping_subclass):
        # GH 29788
        # 使用非字典映射子类创建 Series 对象，键为 3，对应值为 "three"
        ndm = non_dict_mapping_subclass({3: "three"})
        # 使用创建的对象创建 Series 对象
        result = Series(ndm)
        # 期望结果为值为 "three" 的 Series 对象，索引为 [3]
        expected = Series(["three"], index=[3])
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    def test_constructor_list_of_tuples(self):
        # 创建元组列表
        data = [(1, 1), (2, 2), (2, 3)]
        # 使用列表创建 Series 对象
        s = Series(data)
        # 断言 Series 对象与原始列表相等
        assert list(s) == data

    def test_constructor_tuple_of_tuples(self):
        # 创建元组的元组
        data = ((1, 1), (2, 2), (2, 3))
        # 使用元组的元组创建 Series 对象
        s = Series(data)
        # 断言 Series 对象与原始元组的元组相等
        assert tuple(s) == data

    def test_constructor_dict_of_tuples(self):
        # 创建包含元组键的字典，键为 (1, 2)，(None, 5)，对应值为 3，6
        data = {(1, 2): 3, (None, 5): 6}
        # 使用创建的字典创建 Series 对象，并对其进行排序
        result = Series(data).sort_values()
        # 期望结果为包含值 [3, 6] 的 Series 对象，索引为 MultiIndex，从元组创建 [(1, 2), (None, 5)]
        expected = Series([3, 6], index=MultiIndex.from_tuples([(1, 2), (None, 5)]))
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)
    # 测试从字典构建 Series 对象的功能
    def test_fromDict(self, using_infer_string):
        # 准备包含整数值的字典数据
        data = {"a": 0, "b": 1, "c": 2, "d": 3}
        # 使用字典数据创建 Series 对象
        series = Series(data)
        # 断言 Series 的索引是否按顺序排列
        tm.assert_is_sorted(series.index)

        # 准备包含混合类型值的字典数据
        data = {"a": 0, "b": "1", "c": "2", "d": datetime.now()}
        # 使用混合类型的字典数据创建 Series 对象
        series = Series(data)
        # 断言 Series 的数据类型是否为 np.object_
        assert series.dtype == np.object_

        # 根据条件准备不同类型的字典数据
        data = {"a": 0, "b": "1", "c": "2", "d": "3"}
        # 使用条件判断创建 Series 对象，并根据条件判断其数据类型
        series = Series(data)
        assert series.dtype == np.object_ if not using_infer_string else "string"

        # 准备包含字符串表示数值的字典数据，并指定数据类型为 float
        data = {"a": "0", "b": "1"}
        # 使用指定数据类型创建 Series 对象
        series = Series(data, dtype=float)
        # 断言 Series 的数据类型是否为 np.float64
        assert series.dtype == np.float64

    # 测试从单一值构建 Series 对象的功能
    def test_fromValue(self, datetime_series, using_infer_string):
        # 使用 NaN 构建 Series 对象，指定数据类型为 np.float64
        nans = Series(np.nan, index=datetime_series.index, dtype=np.float64)
        # 断言 Series 的数据类型是否为 np.float64
        assert nans.dtype == np.float64
        # 断言 Series 的长度与 datetime_series 的长度相同

        # 使用字符串构建 Series 对象
        strings = Series("foo", index=datetime_series.index)
        # 根据条件判断 Series 的数据类型
        assert strings.dtype == np.object_ if not using_infer_string else "string"
        # 断言 Series 的长度与 datetime_series 的长度相同

        # 使用当前日期时间构建 Series 对象
        d = datetime.now()
        dates = Series(d, index=datetime_series.index)
        # 断言 Series 的数据类型是否为 "M8[us]"
        assert dates.dtype == "M8[us]"
        # 断言 Series 的长度与 datetime_series 的长度相同

        # GH12336
        # 测试从单一值构建分类类型的 Series 对象
        categorical = Series(0, index=datetime_series.index, dtype="category")
        expected = Series(0, index=datetime_series.index).astype("category")
        # 断言 Series 的数据类型是否为 "category"
        assert categorical.dtype == "category"
        # 断言 Series 的长度与 datetime_series 的长度相同
        tm.assert_series_equal(categorical, expected)
    # 测试构造函数针对 timedelta64 类型的情况
    def test_constructor_dtype_timedelta64(self):
        # 基础情况：创建包含多个 timedelta 对象的 Series
        td = Series([timedelta(days=i) for i in range(3)])
        # 断言 Series 的数据类型为 "timedelta64[ns]"
        assert td.dtype == "timedelta64[ns]"
    
        td = Series([timedelta(days=1)])
        assert td.dtype == "timedelta64[ns]"
    
        td = Series([timedelta(days=1), timedelta(days=2), np.timedelta64(1, "s")])
        assert td.dtype == "timedelta64[ns]"
    
        # 混合 NaN 值情况
        td = Series([timedelta(days=1), NaT], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"
    
        td = Series([timedelta(days=1), np.nan], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"
    
        td = Series([np.timedelta64(300000000), NaT], dtype="m8[ns]")
        assert td.dtype == "timedelta64[ns]"
    
        # 改进的类型推断
        # GH5689
        td = Series([np.timedelta64(300000000), NaT])
        assert td.dtype == "timedelta64[ns]"
    
        # 因为 iNaT 是 int 类型，不会强制转换为 timedelta
        td = Series([np.timedelta64(300000000), iNaT])
        assert td.dtype == "object"
    
        td = Series([np.timedelta64(300000000), np.nan])
        assert td.dtype == "timedelta64[ns]"
    
        td = Series([NaT, np.timedelta64(300000000)])
        assert td.dtype == "timedelta64[ns]"
    
        td = Series([np.timedelta64(1, "s")])
        assert td.dtype == "timedelta64[ns]"
    
        # 有效的类型转换操作
        td.astype("int64")
    
        # 无效的类型转换，预期会抛出 TypeError 异常
        msg = r"Converting from timedelta64\[ns\] to int32 is not supported"
        with pytest.raises(TypeError, match=msg):
            td.astype("int32")
    
        # 这是一个无效的类型转换，预期会抛出 ValueError 异常
        msg = "|".join(
            [
                "Could not convert object to NumPy timedelta",
                "Could not convert 'foo' to NumPy timedelta",
            ]
        )
        with pytest.raises(ValueError, match=msg):
            Series([timedelta(days=1), "foo"], dtype="m8[ns]")
    
        # 此处保持为 object 类型
        td = Series([timedelta(days=i) for i in range(3)] + ["foo"])
        assert td.dtype == "object"
    
        # 从 2.0 版本开始，不再基于字符串推断为 timedelta64 类型，与索引行为一致
        ser = Series([None, NaT, "1 Day"])
        assert ser.dtype == object
    
        ser = Series([np.nan, NaT, "1 Day"])
        assert ser.dtype == object
    
        ser = Series([NaT, None, "1 Day"])
        assert ser.dtype == object
    
        ser = Series([NaT, np.nan, "1 Day"])
        assert ser.dtype == object
    
    
    # GH 16406 测试构造函数处理混合时区情况
    def test_constructor_mixed_tz(self):
        # 创建包含有时区信息的 Timestamp 对象的 Series
        s = Series([Timestamp("20130101"), Timestamp("20130101", tz="US/Eastern")])
        # 创建期望的 Series 对象，数据类型为 object
        expected = Series(
            [Timestamp("20130101"), Timestamp("20130101", tz="US/Eastern")],
            dtype="object",
        )
        # 断言创建的 Series 与预期的 Series 相等
        tm.assert_series_equal(s, expected)
    def test_NaT_scalar(self):
        series = Series([0, 1000, 2000, iNaT], dtype="M8[ns]")

        val = series[3]  # 获取索引为3的元素
        assert isna(val)  # 断言该元素是否为 NaT (Not a Time)

        series[2] = val  # 将索引为2的元素设置为 NaT
        assert isna(series[2])  # 断言索引为2的元素确实为 NaT

    def test_NaT_cast(self):
        # GH10747
        result = Series([np.nan]).astype("M8[ns]")  # 将包含 NaN 的 Series 转换为 M8[ns] 类型
        expected = Series([NaT], dtype="M8[ns]")  # 期望得到包含 NaT 的 Series
        tm.assert_series_equal(result, expected)  # 断言结果与期望一致

    def test_constructor_name_hashable(self):
        for n in [777, 777.0, "name", datetime(2001, 11, 11), (1,), "\u05d0"]:
            for data in [[1, 2, 3], np.ones(3), {"a": 0, "b": 1}]:
                s = Series(data, name=n)  # 使用不同的名称 n 创建 Series
                assert s.name == n  # 断言 Series 的名称确实为 n

    def test_constructor_name_unhashable(self):
        msg = r"Series\.name must be a hashable type"
        for n in [["name_list"], np.ones(2), {1: 2}]:
            for data in [["name_list"], np.ones(2), {1: 2}]:
                with pytest.raises(TypeError, match=msg):  # 断言尝试使用不可散列的类型作为 Series 名称时会引发 TypeError 异常
                    Series(data, name=n)

    def test_auto_conversion(self):
        series = Series(list(date_range("1/1/2000", periods=10)))  # 创建包含日期范围的 Series
        assert series.dtype == "M8[ns]"  # 断言 Series 的数据类型为 M8[ns]

    def test_convert_non_ns(self):
        # convert from a numpy array of non-ns timedelta64
        arr = np.array([1, 2, 3], dtype="timedelta64[s]")  # 创建 timedelta64 类型的 numpy 数组
        ser = Series(arr)  # 使用数组创建 Series
        assert ser.dtype == arr.dtype  # 断言 Series 的数据类型与原始数组的数据类型相同

        tdi = timedelta_range("00:00:01", periods=3, freq="s").as_unit("s")  # 创建时间差范围，单位为秒
        expected = Series(tdi)  # 使用时间差范围创建 Series
        assert expected.dtype == arr.dtype  # 断言期望的 Series 的数据类型与原始数组的数据类型相同
        tm.assert_series_equal(ser, expected)  # 断言两个 Series 相等

        # convert from a numpy array of non-ns datetime64
        arr = np.array(
            ["2013-01-01", "2013-01-02", "2013-01-03"], dtype="datetime64[D]"
        )  # 创建日期数组，单位为天
        ser = Series(arr)  # 使用数组创建 Series
        expected = Series(date_range("20130101", periods=3, freq="D"), dtype="M8[s]")  # 创建期望的 Series，单位为秒
        assert expected.dtype == "M8[s]"  # 断言期望的 Series 的数据类型为 M8[s]
        tm.assert_series_equal(ser, expected)  # 断言两个 Series 相等

        arr = np.array(
            ["2013-01-01 00:00:01", "2013-01-01 00:00:02", "2013-01-01 00:00:03"],
            dtype="datetime64[s]",
        )  # 创建时间数组，单位为秒
        ser = Series(arr)  # 使用数组创建 Series
        expected = Series(
            date_range("20130101 00:00:01", periods=3, freq="s"), dtype="M8[s]"
        )  # 创建期望的 Series，单位为秒
        assert expected.dtype == "M8[s]"  # 断言期望的 Series 的数据类型为 M8[s]
        tm.assert_series_equal(ser, expected)  # 断言两个 Series 相等

    @pytest.mark.parametrize(
        "index",
        [
            date_range("1/1/2000", periods=10),  # 使用日期范围作为参数之一
            timedelta_range("1 day", periods=10),  # 使用时间差范围作为参数之一
            period_range("2000-Q1", periods=10, freq="Q"),  # 使用期间范围作为参数之一
        ],
        ids=lambda x: type(x).__name__,  # 根据参数的类型设置标识符
    )
    # 测试构造函数不能将日期时间类型转换为 float 类型
    def test_constructor_cant_cast_datetimelike(self, index):
        # 浮点数不可以作为 dtype 参数
        # 将 Index 对象去除，以转换 PeriodIndex 到 Period
        # 我们不关心错误消息中是 PeriodIndex 还是 PeriodArray
        msg = f"Cannot cast {type(index).__name__.rstrip('Index')}.*? to "

        # 使用 pytest 检查是否会引发 TypeError，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            Series(index, dtype=float)

        # 整数是可以接受的
        # 我们使用 np.int64 来确保在 Windows 或 32 位平台上得到类似的结果
        result = Series(index, dtype=np.int64)
        expected = Series(index.astype(np.int64))
        # 检查结果是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            date_range("1/1/2000", periods=10),  # 生成日期范围的 Index
            timedelta_range("1 day", periods=10),  # 生成时间间隔范围的 Index
            period_range("2000-Q1", periods=10, freq="Q"),  # 生成周期范围的 Index
        ],
        ids=lambda x: type(x).__name__,  # 使用函数作为参数生成测试用例的 ID
    )
    # 测试构造函数能够将对象转换为 object 类型
    def test_constructor_cast_object(self, index):
        s = Series(index, dtype=object)
        exp = Series(index).astype(object)
        # 检查转换后的 Series 对象是否相等
        tm.assert_series_equal(s, exp)

        s = Series(Index(index, dtype=object), dtype=object)
        # 使用 Index 对象作为参数创建 Series 对象，并检查是否相等
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

        s = Series(index.astype(object), dtype=object)
        # 将 Index 对象转换为 object 类型，并检查 Series 对象是否相等
        exp = Series(index).astype(object)
        tm.assert_series_equal(s, exp)

    @pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
    # 测试构造函数能够处理通用时间戳类型且不设置频率
    def test_constructor_generic_timestamp_no_frequency(self, dtype, request):
        # 查看 GitHub issues 15524 和 15987
        msg = "dtype has no unit. Please pass in"

        # 如果 dtype 的名称不在支持的时间戳类型列表中，标记为预期失败
        if np.dtype(dtype).name not in ["timedelta64", "datetime64"]:
            mark = pytest.mark.xfail(reason="GH#33890 Is assigned ns unit")
            request.applymarker(mark)

        # 使用 pytest 检查是否会引发 ValueError，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            Series([], dtype=dtype)

    @pytest.mark.parametrize("unit", ["ps", "as", "fs", "Y", "M", "W", "D", "h", "m"])
    @pytest.mark.parametrize("kind", ["m", "M"])
    # 测试构造函数能够处理不支持的时间戳频率
    def test_constructor_generic_timestamp_bad_frequency(self, kind, unit):
        # 查看 GitHub issues 15524 和 15987
        # 从 2.0 版本开始，对于任何不支持的单位都会引发 TypeError
        # 而不再默默地转换为纳秒；在此之前，仅对高于纳秒的频率才会引发错误
        dtype = f"{kind}8[{unit}]"

        msg = "dtype=.* is not supported. Supported resolutions are"
        # 使用 pytest 检查是否会引发 TypeError，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            Series([], dtype=dtype)

        # 在 2.0 版本之前，DataFrame 的转换会引发错误，但是 Series 的情况不会
        DataFrame([[0]], dtype=dtype)

    @pytest.mark.parametrize("dtype", [None, "uint8", "category"])
    # 测试构造函数能够处理 range 类型的 dtype
    def test_constructor_range_dtype(self, dtype):
        # GitHub issue 16804
        expected = Series([0, 1, 2, 3, 4], dtype=dtype or "int64")
        result = Series(range(5), dtype=dtype)
        # 检查结果是否相等
        tm.assert_series_equal(result, expected)
    def test_constructor_range_overflows(self):
        # 测试构造器处理范围溢出的情况
        # 创建一个范围对象，其范围超过int64的最大值
        rng = range(2**63, 2**63 + 4)
        # 使用该范围对象创建一个Series对象
        ser = Series(rng)
        # 创建预期的Series对象，以便进行后续比较
        expected = Series(list(rng))
        # 断言两个Series对象是否相等
        tm.assert_series_equal(ser, expected)
        # 断言Series对象转换为列表后是否与原始范围相同
        assert list(ser) == list(rng)
        # 断言Series对象的数据类型是否为np.uint64
        assert ser.dtype == np.uint64

        # 创建另一个范围对象，其范围超过int64的最大值
        rng2 = range(2**63 + 4, 2**63, -1)
        # 使用该范围对象创建一个Series对象
        ser2 = Series(rng2)
        # 创建预期的Series对象，以便进行后续比较
        expected2 = Series(list(rng2))
        # 断言两个Series对象是否相等
        tm.assert_series_equal(ser2, expected2)
        # 断言Series对象转换为列表后是否与原始范围相同
        assert list(ser2) == list(rng2)
        # 断言Series对象的数据类型是否为np.uint64

        assert ser2.dtype == np.uint64

        # 创建范围对象，其范围超出int64的最小值
        rng3 = range(-(2**63), -(2**63) - 4, -1)
        # 使用该范围对象创建一个Series对象
        ser3 = Series(rng3)
        # 创建预期的Series对象，以便进行后续比较
        expected3 = Series(list(rng3))
        # 断言两个Series对象是否相等
        tm.assert_series_equal(ser3, expected3)
        # 断言Series对象转换为列表后是否与原始范围相同
        assert list(ser3) == list(rng3)
        # 断言Series对象的数据类型是否为object
        assert ser3.dtype == object

        # 创建一个范围对象，其范围超过int64的最大值
        rng4 = range(2**73, 2**73 + 4)
        # 使用该范围对象创建一个Series对象
        ser4 = Series(rng4)
        # 创建预期的Series对象，以便进行后续比较
        expected4 = Series(list(rng4))
        # 断言两个Series对象是否相等
        tm.assert_series_equal(ser4, expected4)
        # 断言Series对象转换为列表后是否与原始范围相同
        assert list(ser4) == list(rng4)
        # 断言Series对象的数据类型是否为object
        assert ser4.dtype == object

    def test_constructor_tz_mixed_data(self):
        # 测试构造器处理时区混合数据的情况
        dt_list = [
            Timestamp("2016-05-01 02:03:37"),
            Timestamp("2016-04-30 19:03:37-0700", tz="US/Pacific"),
        ]
        # 使用日期时间列表创建一个Series对象
        result = Series(dt_list)
        # 创建预期的Series对象，以便进行后续比较，数据类型为object
        expected = Series(dt_list, dtype=object)
        # 断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("pydt", [True, False])
    def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture, pydt):
        # 测试构造器处理日期时间数据类型感知性和非感知性的情况
        tz = tz_aware_fixture
        ts = Timestamp("2019", tz=tz)
        if pydt:
            ts = ts.to_pydatetime()

        # 准备错误信息
        msg = (
            "Cannot convert timezone-aware data to timezone-naive dtype. "
            r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
        )
        # 断言抛出特定错误类型和错误消息
        with pytest.raises(ValueError, match=msg):
            Series([ts], dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            Series(np.array([ts], dtype=object), dtype="datetime64[ns]")

        with pytest.raises(ValueError, match=msg):
            Series({0: ts}, dtype="datetime64[ns]")

        # 准备错误信息
        msg = "Cannot unbox tzaware Timestamp to tznaive dtype"
        # 断言抛出特定错误类型和错误消息
        with pytest.raises(TypeError, match=msg):
            Series(ts, index=[0], dtype="datetime64[ns]")

    def test_constructor_datetime64(self):
        # 测试构造器处理datetime64的情况
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        dates = np.asarray(rng)

        # 使用日期范围创建一个Series对象
        series = Series(dates)
        # 断言Series对象的数据类型是否为datetime64[ns]
        assert np.issubdtype(series.dtype, np.dtype("M8[ns]"))

    def test_constructor_datetimelike_scalar_to_string_dtype(
        self, nullable_string_dtype
    ):
        # 测试构造器处理日期时间标量转换为字符串数据类型的情况
        # 创建一个包含字符"M"的Series对象，使用空间索引和指定的数据类型
        result = Series("M", index=[1, 2, 3], dtype=nullable_string_dtype)
        # 创建预期的Series对象，以便进行后续比较
        expected = Series(["M", "M", "M"], index=[1, 2, 3], dtype=nullable_string_dtype)
        # 断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)
    @pytest.mark.parametrize("box", [lambda x: x, np.datetime64])
    # 使用 pytest 的 parametrize 装饰器，为测试方法参数化，box 可以是 lambda 函数或 np.datetime64 类型

    def test_constructor_sparse_datetime64(self, box):
        # 测试稀疏数据类型 SparseDtype 中的 datetime64[ns] 数据构造
        # https://github.com/pandas-dev/pandas/issues/35762

        values = [box("2012-01-01"), box("2013-01-01")]
        # 创建包含 datetime64 数据的列表 values

        dtype = pd.SparseDtype("datetime64[ns]")
        # 指定稀疏数据类型为 datetime64[ns]

        result = Series(values, dtype=dtype)
        # 使用指定的 dtype 构造 Series 对象 result

        arr = pd.arrays.SparseArray(values, dtype=dtype)
        # 使用指定的 dtype 构造稀疏数组 SparseArray 对象 arr

        expected = Series(arr)
        # 根据 SparseArray 对象 arr 构造期望的 Series 对象 expected

        tm.assert_series_equal(result, expected)
        # 使用测试框架的 assert_series_equal 方法比较 result 和 expected

    def test_construction_from_ordered_collection(self):
        # 测试从有序集合构造 Series 对象
        # https://github.com/pandas-dev/pandas/issues/36044

        result = Series({"a": 1, "b": 2}.keys())
        # 从字典的键构造 Series 对象 result

        expected = Series(["a", "b"])
        # 期望的 Series 对象 expected 包含字符串列表 ["a", "b"]

        tm.assert_series_equal(result, expected)
        # 比较 result 和 expected

        result = Series({"a": 1, "b": 2}.values())
        # 从字典的值构造 Series 对象 result

        expected = Series([1, 2])
        # 期望的 Series 对象 expected 包含整数列表 [1, 2]

        tm.assert_series_equal(result, expected)
        # 比较 result 和 expected

    def test_construction_from_large_int_scalar_no_overflow(self):
        # 测试从大整数标量构造 Series 对象而不溢出
        # https://github.com/pandas-dev/pandas/issues/36291

        n = 1_000_000_000_000_000_000_000
        # 定义一个大整数 n

        result = Series(n, index=[0])
        # 使用大整数 n 和指定索引构造 Series 对象 result

        expected = Series(n)
        # 期望的 Series 对象 expected 包含大整数 n

        tm.assert_series_equal(result, expected)
        # 比较 result 和 expected

    def test_constructor_list_of_periods_infers_period_dtype(self):
        # 测试从期间列表推断出期间数据类型 "Period[D]"
        
        series = Series(list(period_range("2000-01-01", periods=10, freq="D")))
        # 使用 pd.period_range 生成日期范围，构造包含期间对象的 Series 对象 series
        assert series.dtype == "Period[D]"
        # 断言 series 的数据类型为 "Period[D]"

        series = Series(
            [Period("2011-01-01", freq="D"), Period("2011-02-01", freq="D")]
        )
        # 使用 Period 对象列表构造 Series 对象 series

        assert series.dtype == "Period[D]"
        # 断言 series 的数据类型为 "Period[D]"

    def test_constructor_subclass_dict(self, dict_subclass):
        # 测试从字典子类构造 Series 对象
        data = dict_subclass((x, 10.0 * x) for x in range(10))
        # 使用 dict_subclass 生成的字典数据构造字典 data

        series = Series(data)
        # 使用字典 data 构造 Series 对象 series

        expected = Series(dict(data.items()))
        # 期望的 Series 对象 expected 使用标准字典数据构造

        tm.assert_series_equal(series, expected)
        # 比较 series 和 expected

    def test_constructor_ordereddict(self):
        # 测试从 OrderedDict 构造 Series 对象
        # GH3283

        data = OrderedDict(
            (f"col{i}", np.random.default_rng(2).random()) for i in range(12)
        )
        # 使用 OrderedDict 构造有序字典 data

        series = Series(data)
        # 使用有序字典 data 构造 Series 对象 series

        expected = Series(list(data.values()), list(data.keys()))
        # 期望的 Series 对象 expected 使用 data 的键值分别作为数据和索引

        tm.assert_series_equal(series, expected)
        # 比较 series 和 expected

        # Test with subclass
        class A(OrderedDict):
            pass

        series = Series(A(data))
        # 使用 OrderedDict 子类 A 的实例构造 Series 对象 series

        tm.assert_series_equal(series, expected)
        # 比较 series 和 expected

    def test_constructor_dict_multiindex(self):
        # 测试从具有多级索引的字典构造 Series 对象

        d = {("a", "a"): 0.0, ("b", "a"): 1.0, ("b", "c"): 2.0}
        # 定义具有多级索引的字典 d

        _d = sorted(d.items())
        # 对字典 d 的键值对按键进行排序，并保存在 _d 中

        result = Series(d)
        # 使用字典 d 构造 Series 对象 result

        expected = Series(
            [x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])
        )
        # 期望的 Series 对象 expected 包含按顺序提取的值和构建的多级索引

        tm.assert_series_equal(result, expected)
        # 比较 result 和 expected

        d["z"] = 111.0
        _d.insert(0, ("z", d["z"]))
        # 向字典 d 添加一个新条目，并更新排序后的键值对列表 _d

        result = Series(d)
        # 使用更新后的字典 d 构造 Series 对象 result

        expected = Series(
            [x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)
        )
        # 更新期望的 Series 对象 expected，以便与更新后的字典 d 的结构匹配

        result = result.reindex(index=expected.index)
        # 使用 expected 的索引重新索引 result

        tm.assert_series_equal(result, expected)
        # 比较 result 和 expected
    # 测试构造函数，使用多级索引重新索引为平面结构
    def test_constructor_dict_multiindex_reindex_flat(self):
        # 构建过程涉及使用多级索引的重新索引的特殊情况
        data = {("i", "i"): 0, ("i", "j"): 1, ("j", "i"): 2, "j": np.nan}
        expected = Series(data)

        # 使用从预期Series对象生成的字典作为数据，指定与预期相同的索引来构造结果Series对象
        result = Series(expected[:-1].to_dict(), index=expected.index)
        tm.assert_series_equal(result, expected)

    # 测试构造函数，使用Timedelta索引的字典数据
    def test_constructor_dict_timedelta_index(self):
        # GH #12169 : 使用Timedelta索引重新采样类别数据
        # 使用字典作为数据和TimedeltaIndex作为索引构造Series对象
        # 结果中的Series数据将包含NaN值
        expected = Series(
            data=["A", "B", "C"], index=pd.to_timedelta([0, 10, 20], unit="s")
        )

        result = Series(
            data={
                pd.to_timedelta(0, unit="s"): "A",
                pd.to_timedelta(10, unit="s"): "B",
                pd.to_timedelta(20, unit="s"): "C",
            },
            index=pd.to_timedelta([0, 10, 20], unit="s"),
        )
        tm.assert_series_equal(result, expected)

    # 测试构造函数，推断索引带有时区信息
    def test_constructor_infer_index_tz(self):
        values = [188.5, 328.25]
        tzinfo = tzoffset(None, 7200)
        index = [
            datetime(2012, 5, 11, 11, tzinfo=tzinfo),
            datetime(2012, 5, 11, 12, tzinfo=tzinfo),
        ]
        series = Series(data=values, index=index)

        # 断言Series对象的索引具有正确的时区信息
        assert series.index.tz == tzinfo

        # 验证特定情况是否正常工作，GH#2443
        repr(series.index[0])

    # 测试构造函数，使用Pandas数据类型
    def test_constructor_with_pandas_dtype(self):
        # 经过2D->1D路径的构造
        vals = [(1,), (2,), (3,)]
        ser = Series(vals)
        dtype = ser.array.dtype  # NumpyEADtype
        ser2 = Series(vals, dtype=dtype)
        tm.assert_series_equal(ser, ser2)

    # 测试构造函数，使用int64数据类型处理缺失值
    def test_constructor_int_dtype_missing_values(self):
        # GH#43017
        result = Series(index=[0], dtype="int64")
        expected = Series(np.nan, index=[0], dtype="float64")
        tm.assert_series_equal(result, expected)

    # 测试构造函数，使用bool数据类型处理缺失值
    def test_constructor_bool_dtype_missing_values(self):
        # GH#43018
        result = Series(index=[0], dtype="bool")
        expected = Series(True, index=[0], dtype="bool")
        tm.assert_series_equal(result, expected)

    # 测试构造函数，使用int64数据类型
    def test_constructor_int64_dtype(self, any_int_dtype):
        # GH#44923
        result = Series(["0", "1", "2"], dtype=any_int_dtype)
        expected = Series([0, 1, 2], dtype=any_int_dtype)
        tm.assert_series_equal(result, expected)

    # 测试构造函数，对字符串进行可能导致信息丢失的强制转换时是否引发异常
    def test_constructor_raise_on_lossy_conversion_of_strings(self):
        # GH#44923
        if not np_version_gt2:
            raises = pytest.raises(
                ValueError, match="string values cannot be losslessly cast to int8"
            )
        else:
            raises = pytest.raises(
                OverflowError, match="The elements provided in the data"
            )
        with raises:
            Series(["128"], dtype="int8")
    # 定义测试方法，验证Series构造函数对timedelta64[ns]数据类型的替代构造方式
    def test_constructor_dtype_timedelta_alternative_construct(self):
        # GH#35465
        # 创建包含时间差值的Series，指定数据类型为timedelta64[ns]
        result = Series([1000000, 200000, 3000000], dtype="timedelta64[ns]")
        # 期望的Series，使用pd.to_timedelta将整数列表转换为timedelta64[ns]类型
        expected = Series(pd.to_timedelta([1000000, 200000, 3000000], unit="ns"))
        # 使用tm.assert_series_equal比较结果与期望Series
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(
        reason="Not clear what the correct expected behavior should be with "
        "integers now that we support non-nano. ATM (2022-10-08) we treat ints "
        "as nanoseconds, then cast to the requested dtype. xref #48312"
    )
    # 标记为预期失败的测试方法，针对timedelta64[ns]和timedelta64[s]数据类型的构造比较
    def test_constructor_dtype_timedelta_ns_s(self):
        # GH#35465
        # 创建包含时间差值的Series，指定数据类型为timedelta64[ns]
        result = Series([1000000, 200000, 3000000], dtype="timedelta64[ns]")
        # 期望的Series，指定数据类型为timedelta64[s]
        expected = Series([1000000, 200000, 3000000], dtype="timedelta64[s]")
        # 使用tm.assert_series_equal比较结果与期望Series
        tm.assert_series_equal(result, expected)

    @pytest.mark.xfail(
        reason="Not clear what the correct expected behavior should be with "
        "integers now that we support non-nano. ATM (2022-10-08) we treat ints "
        "as nanoseconds, then cast to the requested dtype. xref #48312"
    )
    # 标记为预期失败的测试方法，针对timedelta64[ns]到int64的类型转换比较
    def test_constructor_dtype_timedelta_ns_s_astype_int64(self):
        # GH#35465
        # 创建包含时间差值的Series，指定数据类型为timedelta64[ns]，然后转换为int64
        result = Series([1000000, 200000, 3000000], dtype="timedelta64[ns]").astype(
            "int64"
        )
        # 期望的Series，先将数据类型为timedelta64[s]，再转换为int64
        expected = Series([1000000, 200000, 3000000], dtype="timedelta64[s]").astype(
            "int64"
        )
        # 使用tm.assert_series_equal比较结果与期望Series
        tm.assert_series_equal(result, expected)

    @pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning"
    )
    @pytest.mark.parametrize("func", [Series, DataFrame, Index, pd.array])
    # 参数化测试方法，针对不同的数据结构构造函数进行测试
    def test_constructor_mismatched_null_nullable_dtype(
        self, func, any_numeric_ea_dtype
    ):
        # GH#44514
        # 错误消息模式匹配字符串
        msg = "|".join(
            [
                "cannot safely cast non-equivalent object",
                r"int\(\) argument must be a string, a bytes-like object "
                "or a (real )?number",
                r"Cannot cast array data from dtype\('O'\) to dtype\('float64'\) "
                "according to the rule 'safe'",
                "object cannot be converted to a FloatingDtype",
                "'values' contains non-numeric NA",
            ]
        )

        # 遍历tm.NP_NAT_OBJECTS和NaT，测试在给定任意数值类型数据的情况下，func方法是否引发TypeError异常，异常信息匹配msg
        for null in tm.NP_NAT_OBJECTS + [NaT]:
            with pytest.raises(TypeError, match=msg):
                func([null, 1.0, 3.0], dtype=any_numeric_ea_dtype)

    # 验证Series构造函数能正确处理从布尔值到Int64类型的转换
    def test_series_constructor_ea_int_from_bool(self):
        # GH#42137
        # 创建包含布尔值的Series，数据类型为Int64
        result = Series([True, False, True, pd.NA], dtype="Int64")
        # 期望的Series，将布尔值转换为对应的整数表示，数据类型为Int64
        expected = Series([1, 0, 1, pd.NA], dtype="Int64")
        # 使用tm.assert_series_equal比较结果与期望Series
        tm.assert_series_equal(result, expected)

        # 创建包含布尔值的Series，数据类型为Int64
        result = Series([True, False, True], dtype="Int64")
        # 期望的Series，将布尔值转换为对应的整数表示，数据类型为Int64
        expected = Series([1, 0, 1], dtype="Int64")
        # 使用tm.assert_series_equal比较结果与期望Series
        tm.assert_series_equal(result, expected)
    def test_series_constructor_ea_int_from_string_bool(self):
        # GH#42137
        # 测试函数：验证 Series 构造函数对字符串布尔值转换为 Int64 类型的处理
        with pytest.raises(ValueError, match="invalid literal"):
            Series(["True", "False", "True", pd.NA], dtype="Int64")

    @pytest.mark.parametrize("val", [1, 1.0])
    def test_series_constructor_overflow_uint_ea(self, val):
        # GH#38798
        # 测试函数：验证 Series 构造函数处理超出范围的 UInt64 类型数据
        max_val = np.iinfo(np.uint64).max - 1
        result = Series([max_val, val], dtype="UInt64")
        expected = Series(np.array([max_val, 1], dtype="uint64"), dtype="UInt64")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("val", [1, 1.0])
    def test_series_constructor_overflow_uint_ea_with_na(self, val):
        # GH#38798
        # 测试函数：验证 Series 构造函数处理带有 NA 值的 UInt64 类型数据
        max_val = np.iinfo(np.uint64).max - 1
        result = Series([max_val, val, pd.NA], dtype="UInt64")
        expected = Series(
            IntegerArray(
                np.array([max_val, 1, 0], dtype="uint64"),
                np.array([0, 0, 1], dtype=np.bool_),
            )
        )
        tm.assert_series_equal(result, expected)

    def test_series_constructor_overflow_uint_with_nan(self):
        # GH#38798
        # 测试函数：验证 Series 构造函数处理带有 NaN 值的 UInt64 类型数据
        max_val = np.iinfo(np.uint64).max - 1
        result = Series([max_val, np.nan], dtype="UInt64")
        expected = Series(
            IntegerArray(
                np.array([max_val, 1], dtype="uint64"),
                np.array([0, 1], dtype=np.bool_),
            )
        )
        tm.assert_series_equal(result, expected)

    def test_series_constructor_ea_all_na(self):
        # GH#38798
        # 测试函数：验证 Series 构造函数处理全为 NA 值的 UInt64 类型数据
        result = Series([np.nan, np.nan], dtype="UInt64")
        expected = Series(
            IntegerArray(
                np.array([1, 1], dtype="uint64"),
                np.array([1, 1], dtype=np.bool_),
            )
        )
        tm.assert_series_equal(result, expected)

    def test_series_from_index_dtype_equal_does_not_copy(self):
        # GH#52008
        # 测试函数：验证 Series 从索引创建并保持数据类型相等且不复制数据
        idx = Index([1, 2, 3])
        expected = idx.copy(deep=True)
        ser = Series(idx, dtype="int64")
        ser.iloc[0] = 100
        tm.assert_index_equal(idx, expected)

    def test_series_string_inference(self):
        # GH#54430
        # 测试函数：验证 Series 对字符串推断的处理
        pytest.importorskip("pyarrow")
        dtype = "string[pyarrow_numpy]"
        expected = Series(["a", "b"], dtype=dtype)
        with pd.option_context("future.infer_string", True):
            ser = Series(["a", "b"])
        tm.assert_series_equal(ser, expected)

        expected = Series(["a", 1], dtype="object")
        with pd.option_context("future.infer_string", True):
            ser = Series(["a", 1])
        tm.assert_series_equal(ser, expected)

    @pytest.mark.parametrize("na_value", [None, np.nan, pd.NA])
    # 测试函数：测试带有 NA 推断的字符串序列
    def test_series_string_with_na_inference(self, na_value):
        # 导入 pyarrow 模块，如果模块不存在则跳过该测试
        pytest.importorskip("pyarrow")
        # 定义数据类型为字符串，使用 pyarrow_numpy 存储
        dtype = "string[pyarrow_numpy]"
        # 创建预期的 Series 对象，包含一个字符串和一个 NA 值
        expected = Series(["a", na_value], dtype=dtype)
        # 使用上下文设置，启用未来的字符串推断
        with pd.option_context("future.infer_string", True):
            # 创建 Series 对象，包含一个字符串和一个可能是 NA 值的对象
            ser = Series(["a", na_value])
        # 断言实际得到的 Series 对象与预期的 Series 对象相等
        tm.assert_series_equal(ser, expected)

    # 测试函数：测试推断标量字符串的 Series
    def test_series_string_inference_scalar(self):
        # 导入 pyarrow 模块，如果模块不存在则跳过该测试
        pytest.importorskip("pyarrow")
        # 创建预期的 Series 对象，包含一个标量字符串
        expected = Series("a", index=[1], dtype="string[pyarrow_numpy]")
        # 使用上下文设置，启用未来的字符串推断
        with pd.option_context("future.infer_string", True):
            # 创建 Series 对象，包含一个标量字符串和索引为 [1] 的对象
            ser = Series("a", index=[1])
        # 断言实际得到的 Series 对象与预期的 Series 对象相等
        tm.assert_series_equal(ser, expected)

    # 测试函数：测试推断数组字符串 dtype 的 Series
    def test_series_string_inference_array_string_dtype(self):
        # 导入 pyarrow 模块，如果模块不存在则跳过该测试
        pytest.importorskip("pyarrow")
        # 创建预期的 Series 对象，包含一个数组字符串
        expected = Series(["a", "b"], dtype="string[pyarrow_numpy]")
        # 使用上下文设置，启用未来的字符串推断
        with pd.option_context("future.infer_string", True):
            # 创建 Series 对象，包含一个 NumPy 数组 ["a", "b"]
            ser = Series(np.array(["a", "b"]))
        # 断言实际得到的 Series 对象与预期的 Series 对象相等
        tm.assert_series_equal(ser, expected)

    # 测试函数：测试推断存储定义的字符串 dtype 的 Series
    def test_series_string_inference_storage_definition(self):
        # 导入 pyarrow 模块，如果模块不存在则跳过该测试
        pytest.importorskip("pyarrow")
        # 创建预期的 Series 对象，包含一个存储定义的字符串数组
        expected = Series(["a", "b"], dtype="string[pyarrow_numpy]")
        # 使用上下文设置，启用未来的字符串推断
        with pd.option_context("future.infer_string", True):
            # 创建 Series 对象，包含一个存储定义为字符串的数组 ["a", "b"]
            result = Series(["a", "b"], dtype="string")
        # 断言实际得到的 Series 对象与预期的 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 测试函数：测试推断标量字符串的 Series 构造函数
    def test_series_constructor_infer_string_scalar(self):
        # 使用上下文设置，启用未来的字符串推断
        with pd.option_context("future.infer_string", True):
            # 创建 Series 对象，包含一个标量字符串 "a" 和索引为 [1, 2]
            ser = Series("a", index=[1, 2], dtype="string[python]")
        # 创建预期的 Series 对象，包含两个标量字符串 "a"
        expected = Series(["a", "a"], index=[1, 2], dtype="string[python]")
        # 断言实际得到的 Series 对象与预期的 Series 对象相等
        tm.assert_series_equal(ser, expected)
        # 断言 Series 对象的数据类型的存储为 "python"
        assert ser.dtype.storage == "python"

    # 测试函数：测试推断 NA 值优先的字符串序列
    def test_series_string_inference_na_first(self):
        # 导入 pyarrow 模块，如果模块不存在则跳过该测试
        pytest.importorskip("pyarrow")
        # 创建预期的 Series 对象，包含一个 NA 值和一个字符串 "b"
        expected = Series([pd.NA, "b"], dtype="string[pyarrow_numpy]")
        # 使用上下文设置，启用未来的字符串推断
        with pd.option_context("future.infer_string", True):
            # 创建 Series 对象，包含一个 NA 值和一个字符串 "b"
            result = Series([pd.NA, "b"])
        # 断言实际得到的 Series 对象与预期的 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 测试函数：对 Pandas 对象进行推断
    @pytest.mark.parametrize("klass", [Series, Index])
    def test_inference_on_pandas_objects(self, klass):
        # 创建一个 Pandas 对象，包含一个 Timestamp 对象，数据类型为 object
        obj = klass([Timestamp("2019-12-31")], dtype=object)
        # 创建 Series 对象，使用上述 Pandas 对象
        result = Series(obj)
        # 断言 Series 对象的数据类型为 np.object_
        assert result.dtype == np.object_
class TestSeriesConstructorIndexCoercion:
    # 测试类：测试 Series 构造器的索引强制转换

    def test_series_constructor_datetimelike_index_coercion(self):
        # 测试方法：测试 Series 构造器对日期时间索引的类型强制转换
        idx = date_range("2020-01-01", periods=5)  # 创建一个日期范围索引
        ser = Series(
            np.random.default_rng(2).standard_normal(len(idx)), idx.astype(object)
        )  # 创建一个 Series，使用随机正态分布数据，并将索引转换为对象类型

        # 从 2.0 版本开始，不再自动将对象类型的索引静默转换为 DatetimeIndex，详见 GH#39307, GH#23598
        assert not isinstance(ser.index, DatetimeIndex)

    @pytest.mark.parametrize("container", [None, np.array, Series, Index])
    @pytest.mark.parametrize("data", [1.0, range(4)])
    def test_series_constructor_infer_multiindex(self, container, data):
        # 测试方法：测试 Series 构造器推断多级索引
        indexes = [["a", "a", "b", "b"], ["x", "y", "x", "y"]]
        if container is not None:
            indexes = [container(ind) for ind in indexes]

        multi = Series(data, index=indexes)  # 创建一个包含多级索引的 Series
        assert isinstance(multi.index, MultiIndex)

    # TODO: 在 pandas 3.0 中不要再将其转换为对象类型
    @pytest.mark.skipif(
        not np_version_gt2, reason="StringDType only available in numpy 2 and above"
    )
    @pytest.mark.parametrize(
        "data",
        [
            ["a", "b", "c"],
            ["a", "b", np.nan],
        ],
    )
    def test_np_string_array_object_cast(self, data):
        # 测试方法：测试将 numpy 字符串数组转换为对象类型的 Series
        from numpy.dtypes import StringDType

        arr = np.array(data, dtype=StringDType())
        res = Series(arr)
        assert res.dtype == np.object_
        assert (res == data).all()


class TestSeriesConstructorInternals:
    # 测试类：测试 Series 构造器的内部实现细节

    def test_constructor_no_pandas_array(self):
        # 测试方法：测试构造不含 pandas 数组的 Series
        ser = Series([1, 2, 3])
        result = Series(ser.array)
        tm.assert_series_equal(ser, result)  # 断言两个 Series 相等
        assert isinstance(result._mgr.blocks[0], NumpyBlock)
        assert result._mgr.blocks[0].is_numeric

    def test_from_array(self):
        # 测试方法：测试从 pandas 数组创建 Series
        result = Series(pd.array(["1h", "2h"], dtype="timedelta64[ns]"))
        assert result._mgr.blocks[0].is_extension is False

        result = Series(pd.array(["2015"], dtype="datetime64[ns]"))
        assert result._mgr.blocks[0].is_extension is False

    def test_from_list_dtype(self):
        # 测试方法：测试从列表创建指定数据类型的 Series
        result = Series(["1h", "2h"], dtype="timedelta64[ns]")
        assert result._mgr.blocks[0].is_extension is False

        result = Series(["2015"], dtype="datetime64[ns]")
        assert result._mgr.blocks[0].is_extension is False


def test_constructor(rand_series_with_duplicate_datetimeindex):
    # 测试函数：测试 Series 的构造函数
    dups = rand_series_with_duplicate_datetimeindex
    assert isinstance(dups, Series)
    assert isinstance(dups.index, DatetimeIndex)


@pytest.mark.parametrize(
    "input_dict,expected",
    [
        ({0: 0}, np.array([[0]], dtype=np.int64)),
        ({"a": "a"}, np.array([["a"]], dtype=object)),
        ({1: 1}, np.array([[1]], dtype=np.int64)),
    ],
)
def test_numpy_array(input_dict, expected):
    # 测试函数：测试将 Series 转换为 numpy 数组
    result = np.array([Series(input_dict)])
    tm.assert_numpy_array_equal(result, expected)


def test_index_ordered_dict_keys():
    # 测试函数：测试索引的有序字典键
    # GH 22077
    # 创建一个有序字典 `param_index`，包含两个条目，每个条目都是一个元组作为键，与一个整数作为值
    param_index = OrderedDict(
        [
            # 第一个条目：键是嵌套元组 (("a", "b"), ("c", "d"))，值为整数 1
            ((("a", "b"), ("c", "d")), 1),
            # 第二个条目：键是嵌套元组 (("a", None), ("c", "d"))，值为整数 2
            ((("a", None), ("c", "d")), 2),
        ]
    )
    
    # 创建一个 pandas Series `series`，其数据来自于整数列表 [1, 2]，索引使用 `param_index` 的键集合
    series = Series([1, 2], index=param_index.keys())
    
    # 创建一个期望的 pandas Series `expected`，其数据也是 [1, 2]，但索引为 MultiIndex，由两个元组组成
    expected = Series(
        [1, 2],
        index=MultiIndex.from_tuples(
            [
                # 第一个元组索引 (("a", "b"), ("c", "d"))
                (("a", "b"), ("c", "d")),
                # 第二个元组索引 (("a", None), ("c", "d"))
                (("a", None), ("c", "d")),
            ],
        ),
    )
    
    # 使用 pandas 测试模块 `tm` 中的 assert_series_equal 函数比较 `series` 和 `expected`，确保它们相等
    tm.assert_series_equal(series, expected)
# 使用 pytest 模块的 parametrize 装饰器，为测试函数 test_series_with_complex_nan 参数化
@pytest.mark.parametrize(
    "input_list",  # 参数化的参数名称
    [
        [1, complex("nan"), 2],      # 第一个测试参数：包含整数、复数NaN和整数
        [1 + 1j, complex("nan"), 2 + 2j],  # 第二个测试参数：包含复数和复数NaN
    ],
)
def test_series_with_complex_nan(input_list):
    # GH#53627
    # 创建一个 Series 对象，使用输入列表 input_list
    ser = Series(input_list)
    # 创建一个新的 Series 对象，复制原始 Series 对象的数据
    result = Series(ser.array)
    # 断言原始 Series 对象的数据类型为 complex128
    assert ser.dtype == "complex128"
    # 使用 pandas.testing 模块的 assert_series_equal 函数断言两个 Series 对象相等
    tm.assert_series_equal(ser, result)


# 定义一个测试函数 test_dict_keys_rangeindex
def test_dict_keys_rangeindex():
    # 创建一个 Series 对象，使用字典作为输入，键是整数，值是相应的数值
    result = Series({0: 1, 1: 2})
    # 创建一个期望的 Series 对象，数据为 [1, 2]，并指定索引类型为 RangeIndex
    expected = Series([1, 2], index=RangeIndex(2))
    # 使用 pandas.testing 模块的 assert_series_equal 函数断言两个 Series 对象相等，
    # 并检查其索引类型为 RangeIndex
    tm.assert_series_equal(result, expected, check_index_type=True)
```
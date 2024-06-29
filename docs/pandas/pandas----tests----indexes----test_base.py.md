# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_base.py`

```
    # 导入必要的库
    from collections import defaultdict  # 导入 defaultdict 类，实现默认值字典
    from datetime import datetime  # 导入 datetime 类，处理日期时间
    from functools import partial  # 导入 partial 函数，创建函数的可定制版本
    import math  # 导入 math 模块，提供数学运算函数
    import operator  # 导入 operator 模块，实现各种运算符对应的函数
    import re  # 导入 re 模块，提供正则表达式操作

    import numpy as np  # 导入 NumPy 库，支持大量数值计算
    import pytest  # 导入 pytest 库，用于编写和运行测试

    from pandas.compat import IS64  # 从 pandas.compat 导入 IS64，检查是否为 64 位系统
    from pandas.errors import InvalidIndexError  # 从 pandas.errors 导入 InvalidIndexError，处理无效索引错误
    import pandas.util._test_decorators as td  # 导入 pandas.util._test_decorators，测试装饰器

    from pandas.core.dtypes.common import (  # 从 pandas.core.dtypes.common 导入多个函数
        is_any_real_numeric_dtype,  # 检查是否为任意实数数值类型
        is_numeric_dtype,  # 检查是否为数值类型
        is_object_dtype,  # 检查是否为对象类型
    )

    import pandas as pd  # 导入 pandas 库，并命名为 pd
    from pandas import (  # 从 pandas 导入多个类和函数
        CategoricalIndex,  # 分类索引类
        DataFrame,  # 数据帧类
        DatetimeIndex,  # 日期时间索引类
        IntervalIndex,  # 区间索引类
        PeriodIndex,  # 时期索引类
        RangeIndex,  # 范围索引类
        Series,  # 系列类
        TimedeltaIndex,  # 时间增量索引类
        date_range,  # 创建日期范围函数
        period_range,  # 创建时期范围函数
        timedelta_range,  # 创建时间增量范围函数
    )
    import pandas._testing as tm  # 导入 pandas._testing，用于测试支持函数

    from pandas.core.indexes.api import (  # 从 pandas.core.indexes.api 导入多个函数和类
        Index,  # 索引类
        MultiIndex,  # 多重索引类
        _get_combined_index,  # 获取组合索引函数
        ensure_index,  # 确保索引对象函数
        ensure_index_from_sequences,  # 从序列创建索引对象函数
    )


class TestIndex:
    @pytest.fixture
    def simple_index(self) -> Index:
        return Index(list("abcde"))  # 创建一个简单的索引对象，包含字符列表 ['a', 'b', 'c', 'd', 'e']

    def test_can_hold_identifiers(self, simple_index):
        index = simple_index  # 获取简单索引对象
        key = index[0]  # 获取索引位置 0 处的元素
        assert index._can_hold_identifiers_and_holds_name(key) is True  # 断言索引对象可以容纳标识符并保持名称

    @pytest.mark.parametrize("index", ["datetime"], indirect=True)
    def test_new_axis(self, index):
        # TODO: a bunch of scattered tests check this deprecation is enforced.
        #  de-duplicate/centralize them.
        with pytest.raises(ValueError, match="Multi-dimensional indexing"):
            # GH#30588 multi-dimensional indexing deprecated
            index[None, :]  # 尝试进行多维索引，预期引发 ValueError 异常

    def test_constructor_regular(self, index):
        tm.assert_contains_all(index, index)  # 使用测试辅助函数验证索引包含所有指定元素

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_constructor_casting(self, index):
        # casting
        arr = np.array(index)  # 将索引转换为 NumPy 数组
        new_index = Index(arr)  # 使用数组创建新的索引对象
        tm.assert_contains_all(arr, new_index)  # 使用测试辅助函数验证新索引包含所有数组元素
        tm.assert_index_equal(index, new_index)  # 使用测试辅助函数验证新索引与原索引相等

    def test_constructor_copy(self, using_infer_string):
        index = Index(list("abc"), name="name")  # 创建具有名称的索引对象
        arr = np.array(index)  # 将索引转换为 NumPy 数组
        new_index = Index(arr, copy=True, name="name")  # 使用数组创建新的索引对象，进行复制并指定名称
        assert isinstance(new_index, Index)  # 断言新索引是 Index 类的实例
        assert new_index.name == "name"  # 断言新索引的名称为 "name"
        if using_infer_string:
            tm.assert_extension_array_equal(
                new_index.values, pd.array(arr, dtype="string[pyarrow_numpy]")
            )  # 使用测试辅助函数验证新索引值与数组匹配（对于推断为字符串类型）
        else:
            tm.assert_numpy_array_equal(arr, new_index.values)  # 使用测试辅助函数验证新索引值与数组匹配（一般情况）
        arr[0] = "SOMEBIGLONGSTRING"
        assert new_index[0] != "SOMEBIGLONGSTRING"  # 断言新索引位置 0 处的值未被改变

    @pytest.mark.parametrize("cast_as_obj", [True, False])
    @pytest.mark.parametrize(
        "index",
        [
            date_range(
                "2015-01-01 10:00",
                freq="D",
                periods=3,
                tz="US/Eastern",
                name="Green Eggs & Ham",
            ),  # 创建一个带有时区的日期时间索引对象（DTI）
            date_range("2015-01-01 10:00", freq="D", periods=3),  # 创建一个没有时区的日期时间索引对象（DTI）
            timedelta_range("1 days", freq="D", periods=3),  # 创建一个时间增量范围对象（TD）
            period_range("2015-01-01", freq="D", periods=3),  # 创建一个时间段范围对象（Period）
        ],
    )
    def test_constructor_from_index_dtlike(self, cast_as_obj, index):
        if cast_as_obj:
            result = Index(index.astype(object))
            assert result.dtype == np.dtype(object)
            if isinstance(index, DatetimeIndex):
                # GH#23524 检查 Index(dti, dtype=object) 不会错误地引发 ValueError，且纳秒部分未丢失
                index += pd.Timedelta(nanoseconds=50)
                result = Index(index, dtype=object)
                assert result.dtype == np.object_
                assert list(result) == list(index)
        else:
            result = Index(index)
            # 断言新创建的索引与原索引相等
            tm.assert_index_equal(result, index)

    @pytest.mark.parametrize(
        "index,has_tz",
        [
            (
                date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern"),
                True,
            ),  # 创建一个带有时区的日期时间索引对象（datetimetz）
            (timedelta_range("1 days", freq="D", periods=3), False),  # 创建一个时间增量范围对象（TD）
            (period_range("2015-01-01", freq="D", periods=3), False),  # 创建一个时间段范围对象（Period）
        ],
    )
    def test_constructor_from_series_dtlike(self, index, has_tz):
        result = Index(Series(index))
        # 断言新创建的索引与原索引相等
        tm.assert_index_equal(result, index)

        if has_tz:
            # 如果原索引有时区信息，断言新创建的索引也有相同的时区
            assert result.tz == index.tz

    def test_constructor_from_series_freq(self):
        # GH 6273
        # 从一个系列对象创建日期时间索引，指定频率为月初（MS）
        dts = ["1-1-1990", "2-1-1990", "3-1-1990", "4-1-1990", "5-1-1990"]
        expected = DatetimeIndex(dts, freq="MS")

        s = Series(pd.to_datetime(dts))
        result = DatetimeIndex(s, freq="MS")

        # 断言新创建的索引与预期的索引相等
        tm.assert_index_equal(result, expected)
    # 定义测试方法，用于测试从DataFrame或Series构造DatetimeIndex对象，并指定频率
    def test_constructor_from_frame_series_freq(self, using_infer_string):
        # GH 6273
        # 从日期字符串列表创建DatetimeIndex对象，并指定频率为月初（"MS"）
        dts = ["1-1-1990", "2-1-1990", "3-1-1990", "4-1-1990", "5-1-1990"]
        expected = DatetimeIndex(dts, freq="MS")

        # 创建一个随机数据的DataFrame对象
        df = DataFrame(np.random.default_rng(2).random((5, 3)))
        # 将日期字符串列表作为DataFrame的一个列
        df["date"] = dts
        # 从DataFrame的日期列创建DatetimeIndex对象，指定频率为月初（"MS"）
        result = DatetimeIndex(df["date"], freq="MS")
        
        # 根据是否使用推断字符串来确定dtype为object或string
        dtype = object if not using_infer_string else "string"
        # 断言DataFrame的日期列dtype符合预期
        assert df["date"].dtype == dtype
        
        # 将预期的DatetimeIndex对象命名为"date"
        expected.name = "date"
        # 断言结果DatetimeIndex对象与预期对象相等
        tm.assert_index_equal(result, expected)

        # 将日期字符串列表作为Series对象，并命名为"date"
        expected = Series(dts, name="date")
        # 断言DataFrame的日期列与预期Series对象相等
        tm.assert_series_equal(df["date"], expected)

        # GH 6274
        # 推断日期频率为月初（"MS"）
        if not using_infer_string:
            # 不适用于Arrow字符串
            freq = pd.infer_freq(df["date"])
            # 断言推断的频率为月初（"MS"）
            assert freq == "MS"

    # 测试构造函数处理整数dtype的NaN值
    def test_constructor_int_dtype_nan(self):
        # see gh-15187
        # 创建一个包含NaN值的数据列表
        data = [np.nan]
        # 创建预期的Index对象，指定dtype为np.float64
        expected = Index(data, dtype=np.float64)
        # 创建结果Index对象，指定dtype为"float"
        result = Index(data, dtype="float")
        # 断言结果Index对象与预期对象相等
        tm.assert_index_equal(result, expected)

    # 使用参数化测试，测试不同类别（Index或DatetimeIndex）、dtype和NaN值的推断
    @pytest.mark.parametrize(
        "klass,dtype,na_val",
        [
            (Index, np.float64, np.nan),
            (DatetimeIndex, "datetime64[s]", pd.NaT),
        ],
    )
    def test_index_ctor_infer_nan_nat(self, klass, dtype, na_val):
        # GH 13467
        # 创建一个包含NaN值的列表
        na_list = [na_val, na_val]
        # 根据参数化的类别创建预期的Index对象
        expected = klass(na_list)
        # 断言预期Index对象的dtype与参数化的dtype相等
        assert expected.dtype == dtype

        # 创建结果Index对象，根据列表na_list
        result = Index(na_list)
        # 断言结果Index对象与预期对象相等
        tm.assert_index_equal(result, expected)

        # 创建结果Index对象，根据numpy数组na_list
        result = Index(np.array(na_list))
        # 断言结果Index对象与预期对象相等
        tm.assert_index_equal(result, expected)

    # 使用参数化测试，测试不同的值和dtype，构造简单的Index对象
    @pytest.mark.parametrize(
        "vals,dtype",
        [
            ([1, 2, 3, 4, 5], "int"),
            ([1.1, np.nan, 2.2, 3.0], "float"),
            (["A", "B", "C", np.nan], "obj"),
        ],
    )
    def test_constructor_simple_new(self, vals, dtype):
        # 创建基于vals和dtype参数化的Index对象
        index = Index(vals, name=dtype)
        # 调用_simple_new方法，创建一个新的Index对象result
        result = index._simple_new(index.values, dtype)
        # 断言结果Index对象与预期对象相等
        tm.assert_index_equal(result, index)

    # 使用参数化测试，测试不同的属性和类别（Index或DatetimeIndex）
    @pytest.mark.parametrize("attr", ["values", "asi8"])
    @pytest.mark.parametrize("klass", [Index, DatetimeIndex])
    # 定义一个测试方法，用于测试使用带有时区信息的构造函数
    def test_constructor_dtypes_datetime(self, tz_naive_fixture, attr, klass):
        # 测试使用带有datetimetz类型的构造函数
        # .values产生numpy日期时间对象，因此这些被视为naive（非时区感知）
        # .asi8产生整数，因此这些被视为epoch时间戳
        # ^ 上述将在以后的版本中成立。现在我们`.view` i8 值为NS_DTYPE，有效地将它们视为wall times（墙上时间）。
        
        # 创建一个日期范围从"2011-01-01"开始，包含5个时间点的索引
        index = date_range("2011-01-01", periods=5)
        # 获取索引的特定属性（例如：values或asi8）
        arg = getattr(index, attr)
        # 将索引本地化到指定的naive时区
        index = index.tz_localize(tz_naive_fixture)
        # 获取索引的数据类型
        dtype = index.dtype

        # 如果属性为"asi8"
        if attr == "asi8":
            # 创建一个新的DatetimeIndex对象，并本地化到指定的naive时区
            result = DatetimeIndex(arg).tz_localize(tz_naive_fixture)
            # 断言结果索引与原始索引相等
            tm.assert_index_equal(result, index)
        # 如果klass为Index类
        elif klass is Index:
            # 使用给定的参数和时区创建一个klass对象，预期会引发TypeError异常，匹配指定的错误消息
            with pytest.raises(TypeError, match="unexpected keyword"):
                klass(arg, tz=tz_naive_fixture)
        else:
            # 使用给定的参数和时区创建一个klass对象
            result = klass(arg, tz=tz_naive_fixture)
            # 断言结果索引与原始索引相等
            tm.assert_index_equal(result, index)

        # 如果属性为"asi8"
        if attr == "asi8":
            # 如果err为True，预期会引发TypeError异常，匹配指定的错误消息
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(arg).astype(dtype)
            else:
                # 将arg转换为指定的数据类型，然后创建一个新的DatetimeIndex对象
                result = DatetimeIndex(arg).astype(dtype)
                # 断言结果索引与原始索引相等
                tm.assert_index_equal(result, index)
        else:
            # 使用给定的参数和数据类型创建一个klass对象
            result = klass(arg, dtype=dtype)
            # 断言结果索引与原始索引相等
            tm.assert_index_equal(result, index)

        # 如果属性为"asi8"
        if attr == "asi8":
            # 创建一个新的DatetimeIndex对象，并本地化到指定的naive时区
            result = DatetimeIndex(list(arg)).tz_localize(tz_naive_fixture)
            # 断言结果索引与原始索引相等
            tm.assert_index_equal(result, index)
        # 如果klass为Index类
        elif klass is Index:
            # 使用给定的参数和时区创建一个klass对象，预期会引发TypeError异常，匹配指定的错误消息
            with pytest.raises(TypeError, match="unexpected keyword"):
                klass(arg, tz=tz_naive_fixture)
        else:
            # 使用给定的参数和时区创建一个klass对象
            result = klass(list(arg), tz=tz_naive_fixture)
            # 断言结果索引与原始索引相等
            tm.assert_index_equal(result, index)

        # 如果属性为"asi8"
        if attr == "asi8":
            # 如果err为True，预期会引发TypeError异常，匹配指定的错误消息
            if err:
                with pytest.raises(TypeError, match=msg):
                    DatetimeIndex(list(arg)).astype(dtype)
            else:
                # 将列表arg转换为指定的数据类型，然后创建一个新的DatetimeIndex对象
                result = DatetimeIndex(list(arg)).astype(dtype)
                # 断言结果索引与原始索引相等
                tm.assert_index_equal(result, index)
        else:
            # 使用给定的参数和数据类型创建一个klass对象
            result = klass(list(arg), dtype=dtype)
            # 断言结果索引与原始索引相等
            tm.assert_index_equal(result, index)
    # 使用 pytest.mark.parametrize 装饰器，为 value 参数传入空列表、空迭代器和生成器对象
    @pytest.mark.parametrize("value", [[], iter([]), (_ for _ in [])])
    # 使用 pytest.mark.parametrize 装饰器，为 klass 参数传入 Index、CategoricalIndex、DatetimeIndex 和 TimedeltaIndex 类
    @pytest.mark.parametrize(
        "klass",
        [
            Index,
            CategoricalIndex,
            DatetimeIndex,
            TimedeltaIndex,
        ],
    )
    # 定义测试方法 test_constructor_empty，测试不同类的构造函数处理空值的行为
    def test_constructor_empty(self, value, klass):
        # 创建 klass 类的实例 empty，传入 value 参数
        empty = klass(value)
        # 断言 empty 是 klass 类的实例
        assert isinstance(empty, klass)
        # 断言 empty 的长度为 0
        assert not len(empty)

    # 使用 pytest.mark.parametrize 装饰器，为 empty 和 klass 参数传入不同的 PeriodIndex 和 RangeIndex 对象
    @pytest.mark.parametrize(
        "empty,klass",
        [
            (PeriodIndex([], freq="D"), PeriodIndex),
            (PeriodIndex(iter([]), freq="D"), PeriodIndex),
            (PeriodIndex((_ for _ in []), freq="D"), PeriodIndex),
            (RangeIndex(step=1), RangeIndex),
            (MultiIndex(levels=[[1, 2], ["blue", "red"]], codes=[[], []]), MultiIndex),
        ],
    )
    # 定义测试方法 test_constructor_empty_special，测试特殊情况下各种索引类的构造函数处理空值的行为
    def test_constructor_empty_special(self, empty, klass):
        # 断言 empty 是 klass 类的实例
        assert isinstance(empty, klass)
        # 断言 empty 的长度为 0
        assert not len(empty)

    # 使用 pytest.mark.parametrize 装饰器，为 index 参数传入多种索引类型字符串
    @pytest.mark.parametrize(
        "index",
        [
            "datetime",
            "float64",
            "float32",
            "int64",
            "int32",
            "period",
            "range",
            "repeats",
            "timedelta",
            "tuples",
            "uint64",
            "uint32",
        ],
        indirect=True,
    )
    # 定义测试方法 test_view_with_args，测试索引对象的视图方法处理不同参数的行为
    def test_view_with_args(self, index):
        # 调用索引对象的 view 方法，将数据类型转换为 'i8'
        index.view("i8")

    # 使用 pytest.mark.parametrize 装饰器，为 index 参数传入字符串和标记为 xfail 的字符串
    @pytest.mark.parametrize(
        "index",
        [
            "string",
            pytest.param("categorical", marks=pytest.mark.xfail(reason="gh-25464")),
            "bool-object",
            "bool-dtype",
            "empty",
        ],
        indirect=True,
    )
    # 定义测试方法 test_view_with_args_object_array_raises，测试索引对象的视图方法对不同数据类型的异常处理行为
    def test_view_with_args_object_array_raises(self, index):
        # 如果索引对象的数据类型是 bool，断言调用 view 方法转换为 'i8' 时引发 ValueError 异常
        if index.dtype == bool:
            msg = "When changing to a larger dtype"
            with pytest.raises(ValueError, match=msg):
                index.view("i8")
        # 如果索引对象的数据类型是 'string'，断言调用 view 方法转换为 'i8' 时引发 NotImplementedError 异常
        elif index.dtype == "string":
            with pytest.raises(NotImplementedError, match="i8"):
                index.view("i8")
        # 否则，断言调用 view 方法转换为 'i8' 时引发 TypeError 异常，匹配指定的错误消息
        else:
            msg = (
                "Cannot change data-type for array of references.|"
                "Cannot change data-type for object array.|"
            )
            with pytest.raises(TypeError, match=msg):
                index.view("i8")

    # 使用 pytest.mark.parametrize 装饰器，为 index 参数传入不同的索引类型字符串
    @pytest.mark.parametrize(
        "index",
        ["int64", "int32", "range"],
        indirect=True,
    )
    # 定义测试方法 test_astype，测试索引对象的 astype 方法处理类型转换的行为
    def test_astype(self, index):
        # 将索引对象转换为 'i8' 类型，并将结果存储在 casted 中
        casted = index.astype("i8")

        # 调用 casted 的 get_loc 方法，验证转换后的对象仍能正常工作
        casted.get_loc(5)

        # 设置 index 对象的名称为 "foobar"
        index.name = "foobar"
        # 再次将 index 对象转换为 'i8' 类型，并断言转换后的对象名称为 "foobar"
        casted = index.astype("i8")
        assert casted.name == "foobar"

    # 定义测试方法 test_equals_object，测试索引对象的 equals 方法比较相等对象的行为
    def test_equals_object(self):
        # 断言两个相同内容的 Index 对象相等
        assert Index(["a", "b", "c"]).equals(Index(["a", "b", "c"]))

    # 使用 pytest.mark.parametrize 装饰器，为 comp 参数传入不同的对象用于测试不相等的情况
    @pytest.mark.parametrize(
        "comp", [Index(["a", "b"]), Index(["a", "b", "d"]), ["a", "b", "c"]]
    )
    # 定义测试方法 test_not_equals_object，测试索引对象的 equals 方法比较不相等对象的行为
    def test_not_equals_object(self, comp):
        # 断言 Index(["a", "b", "c"]) 与 comp 参数对象不相等
        assert not Index(["a", "b", "c"]).equals(comp)
    # 测试索引对象的相等性
    def test_identical(self):
        # 创建两个包含相同元素的索引对象
        i1 = Index(["a", "b", "c"])
        i2 = Index(["a", "b", "c"])

        # 断言这两个索引对象是相同的
        assert i1.identical(i2)

        # 对 i1 进行重命名操作，并检查其是否等于 i2，以及它们是否相同
        i1 = i1.rename("foo")
        assert i1.equals(i2)
        assert not i1.identical(i2)

        # 对 i2 进行重命名操作，并再次检查它们是否相同
        i2 = i2.rename("foo")
        assert i1.identical(i2)

        # 创建包含元组的索引对象，指定不进行元组化的参数
        i3 = Index([("a", "a"), ("a", "b"), ("b", "a")])
        i4 = Index([("a", "a"), ("a", "b"), ("b", "a")], tupleize_cols=False)
        # 断言这两个索引对象不是相同的
        assert not i3.identical(i4)

    # 测试索引对象的 is_ 方法
    def test_is_(self):
        # 创建一个包含整数范围的索引对象
        ind = Index(range(10))
        # 使用 is_ 方法比较索引对象和它的视图，以及其他情况
        assert ind.is_(ind)
        assert ind.is_(ind.view().view().view().view())
        assert not ind.is_(Index(range(10)))
        assert not ind.is_(ind.copy())
        assert not ind.is_(ind.copy(deep=False))
        assert not ind.is_(ind[:])
        assert not ind.is_(np.array(range(10)))

        # quasi-implementation dependent
        assert ind.is_(ind.view())
        # 创建一个视图，并更改其名称，然后比较是否相同
        ind2 = ind.view()
        ind2.name = "bob"
        assert ind.is_(ind2)
        assert ind2.is_(ind)
        # 不考虑索引对象是否实际上是基础数据的视图
        assert not ind.is_(Index(ind.values))
        # 创建一个不复制的索引对象，并检查它们是否相同
        arr = np.array(range(1, 11))
        ind1 = Index(arr, copy=False)
        ind2 = Index(arr, copy=False)
        assert not ind1.is_(ind2)

    # 测试 asof 方法在数值与布尔类型索引之间抛出异常
    def test_asof_numeric_vs_bool_raises(self):
        # 创建包含整数的索引对象和包含布尔类型的索引对象
        left = Index([1, 2, 3])
        right = Index([True, False], dtype=object)

        # 断言当比较整数和布尔类型时，会抛出类型错误异常
        msg = "Cannot compare dtypes int64 and bool"
        with pytest.raises(TypeError, match=msg):
            left.asof(right[0])
        # TODO: should right.asof(left[0]) also raise?

        # 断言使用不兼容的索引对象调用 asof 方法会抛出无效索引错误异常
        with pytest.raises(InvalidIndexError, match=re.escape(str(right))):
            left.asof(right)

        with pytest.raises(InvalidIndexError, match=re.escape(str(left))):
            right.asof(left)

    # 使用布尔索引测试方法
    @pytest.mark.parametrize("index", ["string"], indirect=True)
    def test_booleanindex(self, index):
        # 创建一个布尔类型数组作为索引的选择器
        bool_index = np.ones(len(index), dtype=bool)
        bool_index[5:30:2] = False

        # 使用布尔索引获取子索引，并检查其正确性
        sub_index = index[bool_index]

        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i

        # 使用布尔列表获取子索引，并再次检查其正确性
        sub_index = index[list(bool_index)]
        for i, val in enumerate(sub_index):
            assert sub_index.get_loc(val) == i

    # 测试 fancy 索引操作
    def test_fancy(self, simple_index):
        # 获取简单索引对象，并对其进行 fancy 索引操作
        index = simple_index
        sl = index[[1, 2, 3]]
        for i in sl:
            assert i == sl[sl.get_loc(i)]

    # 参数化测试索引对象的类型与数据类型
    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"],
        indirect=True,
    )
    @pytest.mark.parametrize("dtype", [int, np.bool_])
    # 定义一个测试方法，用于测试空的高级索引操作
    def test_empty_fancy(self, index, dtype, request, using_infer_string):
        # 如果数据类型是 np.bool_，并且使用推断字符串索引，并且索引的数据类型是字符串
        if dtype is np.bool_ and using_infer_string and index.dtype == "string":
            # 应用 pytest 标记，表示预期该测试会失败，原因是 numpy 的行为存在错误
            request.applymarker(pytest.mark.xfail(reason="numpy behavior is buggy"))
        # 创建一个空的 NumPy 数组，指定数据类型为 dtype
        empty_arr = np.array([], dtype=dtype)
        # 创建一个空的索引对象，类型与传入的 index 参数相同，数据类型为 index.dtype
        empty_index = type(index)([], dtype=index.dtype)

        # 断言空索引 [[]] 和 empty_index 相同
        assert index[[]].identical(empty_index)
        # 如果数据类型是 np.bool_
        if dtype == np.bool_:
            # 使用 pytest 断言引发 ValueError 异常，异常信息匹配 "length of the boolean indexer"
            with pytest.raises(ValueError, match="length of the boolean indexer"):
                # 断言 index[empty_arr] 和 empty_index 相同
                assert index[empty_arr].identical(empty_index)
        else:
            # 否则，断言 index[empty_arr] 和 empty_index 相同
            assert index[empty_arr].identical(empty_index)

    @pytest.mark.parametrize(
        "index",
        ["string", "int64", "int32", "uint64", "uint32", "float64", "float32"],
        indirect=True,
    )
    # 使用参数化测试标记，测试空高级索引引发异常的情况
    def test_empty_fancy_raises(self, index):
        # DatetimeIndex 被排除在外，因为它重写了 getitem 方法，需要单独测试
        empty_farr = np.array([], dtype=np.float64)
        # 创建一个空的索引对象，类型与传入的 index 参数相同，数据类型为 index.dtype
        empty_index = type(index)([], dtype=index.dtype)

        # 断言空索引 [[]] 和 empty_index 相同
        assert index[[]].identical(empty_index)
        # np.ndarray 只接受 int 和 bool 数据类型的 ndarray，因此 Index 也应如此
        msg = r"arrays used as indices must be of integer"
        # 使用 pytest 断言引发 IndexError 异常，异常信息匹配 msg
        with pytest.raises(IndexError, match=msg):
            # 断言 index[empty_farr] 引发异常
            index[empty_farr]

    # 测试将日期时间索引对象作为对象进行联合操作
    def test_union_dt_as_obj(self, simple_index):
        # TODO: 替换为 fixture result
        # 获取简单的索引对象作为测试用例的 index
        index = simple_index
        # 创建一个日期范围的日期索引，从 "2019-01-01" 开始，包含 10 个日期
        date_index = date_range("2019-01-01", periods=10)
        # 对第一个类别的索引对象进行联合操作，包括 date_index
        first_cat = index.union(date_index)
        # 对第二个类别的索引对象进行联合操作，包括 index 本身
        second_cat = index.union(index)

        # 将 index 和 date_index 合并后的结果进行断言，与预期的 appended 相同
        appended = Index(np.append(index, date_index.astype("O")))
        tm.assert_index_equal(first_cat, appended)
        # 断言 second_cat 与 index 相同
        tm.assert_index_equal(second_cat, index)
        # 断言 index 包含在 first_cat 中的所有元素
        tm.assert_contains_all(index, first_cat)
        # 断言 index 包含在 second_cat 中的所有元素
        tm.assert_contains_all(index, second_cat)
        # 断言 date_index 中的所有元素都包含在 first_cat 中
        tm.assert_contains_all(date_index, first_cat)

    # 测试在索引对象上使用元组的映射操作
    def test_map_with_tuples(self):
        # GH 12766

        # 测试从索引中返回单个元组时返回一个索引对象
        index = Index(np.arange(3), dtype=np.int64)
        result = index.map(lambda x: (x,))
        expected = Index([(i,) for i in index])
        tm.assert_index_equal(result, expected)

        # 测试从映射单个索引返回元组时返回一个 MultiIndex 对象
        result = index.map(lambda x: (x, x == 1))
        expected = MultiIndex.from_tuples([(i, i == 1) for i in index])
        tm.assert_index_equal(result, expected)

    # 测试在 MultiIndex 上使用元组的映射操作
    def test_map_with_tuples_mi(self):
        # 测试从 MultiIndex 中返回单个对象时返回一个索引对象
        first_level = ["foo", "bar", "baz"]
        multi_index = MultiIndex.from_tuples(zip(first_level, [1, 2, 3]))
        reduced_index = multi_index.map(lambda x: x[0])
        tm.assert_index_equal(reduced_index, Index(first_level))
    @pytest.mark.parametrize(
        "index",
        [
            # 使用date_range函数生成从 '2020-01-01' 开始的日期范围，频率为每天，共10个日期
            date_range("2020-01-01", freq="D", periods=10),
            # 使用period_range函数生成从 '2020-01-01' 开始的周期范围，频率为每天，共10个周期
            period_range("2020-01-01", freq="D", periods=10),
            # 使用timedelta_range函数生成从 '1 day' 开始的时间增量范围，共10个增量
            timedelta_range("1 day", periods=10),
        ],
    )
    # 测试函数，验证index对象的map操作返回期望的Index对象
    def test_map_tseries_indices_return_index(self, index):
        # 期望结果是长度为10的全为1的Index对象
        expected = Index([1] * 10)
        # 对index对象应用lambda函数映射，将每个元素映射为1
        result = index.map(lambda x: 1)
        # 使用pytest的assert_index_equal断言，验证result与expected相等
        tm.assert_index_equal(expected, result)

    # 测试函数，验证DatetimeIndex对象的map操作返回期望的Index对象
    def test_map_tseries_indices_accsr_return_index(self):
        # 创建一个DatetimeIndex对象，从 '2020-01-01' 开始，频率为每小时，共24个时间点
        date_index = DatetimeIndex(
            date_range("2020-01-01", periods=24, freq="h"), name="hourly"
        )
        # 对date_index对象应用lambda函数映射，将每个时间点映射为其小时数
        result = date_index.map(lambda x: x.hour)
        # 期望结果是从0到23的整数Index对象，名称为"hourly"
        expected = Index(np.arange(24, dtype="int64"), name="hourly")
        # 使用pytest的assert_index_equal断言，验证result与expected相等，精确比较
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize(
        "mapper",
        [
            # 定义一个映射函数，将values和index映射为字典类型{i: e for e, i in zip(values, index)}
            lambda values, index: {i: e for e, i in zip(values, index)},
            # 定义一个映射函数，将values和index映射为Series对象
            lambda values, index: Series(values, index),
        ],
    )
    # 测试函数，验证Index对象的map操作对于字典映射函数mapper返回期望的Index对象
    def test_map_dictlike_simple(self, mapper):
        # GH 12756
        # 期望结果是包含字符串"foo", "bar", "baz"的Index对象
        expected = Index(["foo", "bar", "baz"])
        # 创建一个整数Index对象，从0到2
        index = Index(np.arange(3), dtype=np.int64)
        # 对index对象应用mapper函数，将其映射为结果对象result
        result = index.map(mapper(expected.values, index))
        # 使用pytest的assert_index_equal断言，验证result与expected相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        [
            # 定义一个映射函数，将values和index映射为字典类型{i: e for e, i in zip(values, index)}
            lambda values, index: {i: e for e, i in zip(values, index)},
            # 定义一个映射函数，将values和index映射为Series对象
            lambda values, index: Series(values, index),
        ],
    )
    # 测试函数，验证Index对象的map操作对于字典映射函数mapper和特定index对象返回期望的Index对象
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_map_dictlike(self, index, mapper, request):
        # GH 12756
        # 如果index对象是CategoricalIndex类型，则跳过测试
        if isinstance(index, CategoricalIndex):
            pytest.skip("Tested in test_categorical")
        # 如果index对象不是唯一值，则跳过测试
        elif not index.is_unique:
            pytest.skip("Cannot map duplicated index")

        # 生成一个整数数组rng，长度与index相同，值为从长度到1的递减整数
        rng = np.arange(len(index), 0, -1, dtype=np.int64)

        # 根据index对象的特性选择期望的Index对象
        if index.empty:
            # 如果index为空，则期望结果是一个空的Index对象
            expected = Index([])
        elif is_numeric_dtype(index.dtype):
            # 如果index的数据类型是数值型，则期望结果是具有相同dtype的整数Index对象
            expected = index._constructor(rng, dtype=index.dtype)
        elif type(index) is Index and index.dtype != object:
            # 如果index是Index类型且dtype不是object，则期望结果是具有相同dtype的Index对象
            expected = Index(rng, dtype=index.dtype)
        else:
            # 否则，期望结果是一个Index对象，包含rng中的值
            expected = Index(rng)

        # 对index对象应用mapper函数，将其映射为结果对象result
        result = index.map(mapper(expected, index))
        # 使用pytest的assert_index_equal断言，验证result与expected相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "mapper",
        # 定义一个Series和字典映射对象，用于测试Index对象的map操作
        [Series(["foo", 2.0, "baz"], index=[0, 2, -1]), {0: "foo", 2: 2.0, -1: "baz"}],
    )
    # 测试函数，验证Index对象的map操作对于包含非函数类型的缺失值mapper返回期望的Index对象
    def test_map_with_non_function_missing_values(self, mapper):
        # GH 12756
        # 期望结果是包含2.0, NaN, "foo"的Index对象
        expected = Index([2.0, np.nan, "foo"])
        # 使用mapper映射函数，将Index([2, 1, 0])映射为结果对象result
        result = Index([2, 1, 0]).map(mapper)

        # 使用pytest的assert_index_equal断言，验证result与expected相等
        tm.assert_index_equal(expected, result)
    # 测试索引对象的映射功能，忽略 NaN 值
    def test_map_na_exclusion(self):
        # 创建包含浮点数和 NaN 值的索引对象
        index = Index([1.5, np.nan, 3, np.nan, 5])

        # 对索引对象进行映射操作，将每个元素乘以 2，忽略 NaN 值
        result = index.map(lambda x: x * 2, na_action="ignore")
        # 期望的结果是索引对象中的每个元素乘以 2
        expected = index * 2
        # 断言映射后的结果与期望结果相等
        tm.assert_index_equal(result, expected)

    # 测试使用 defaultdict 进行索引映射
    def test_map_defaultdict(self):
        # 创建包含整数的索引对象
        index = Index([1, 2, 3])
        # 创建一个 defaultdict，将默认值设置为 "blank"，并设置特定键的值为 "stuff"
        default_dict = defaultdict(lambda: "blank")
        default_dict[1] = "stuff"

        # 使用 defaultdict 对索引对象进行映射操作
        result = index.map(default_dict)
        # 期望的结果是根据 defaultdict 的映射规则进行映射后的索引对象
        expected = Index(["stuff", "blank", "blank"])
        # 断言映射后的结果与期望结果相等
        tm.assert_index_equal(result, expected)

    # 测试索引对象的追加操作，保持名称不变
    @pytest.mark.parametrize("name,expected", [("foo", "foo"), ("bar", None)])
    def test_append_empty_preserve_name(self, name, expected):
        # 创建一个空的索引对象，并指定名称为 "foo"
        left = Index([], name="foo")
        # 创建一个包含整数的索引对象，名称为动态传入的 name
        right = Index([1, 2, 3], name=name)

        # 将右侧的索引对象追加到左侧的索引对象上
        result = left.append(right)
        # 断言追加后结果的名称与预期名称相等
        assert result.name == expected

    # 测试判断索引对象是否为数值型
    @pytest.mark.parametrize(
        "index, expected",
        [
            ("string", False),
            ("bool-object", False),
            ("bool-dtype", False),
            ("categorical", False),
            ("int64", True),
            ("int32", True),
            ("uint64", True),
            ("uint32", True),
            ("datetime", False),
            ("float64", True),
            ("float32", True),
        ],
        indirect=["index"],
    )
    def test_is_numeric(self, index, expected):
        # 断言判断索引对象是否为任意实数数值类型，与预期结果相符
        assert is_any_real_numeric_dtype(index) is expected

    # 测试判断索引对象是否为对象类型
    @pytest.mark.parametrize(
        "index, expected",
        [
            ("string", True),
            ("bool-object", True),
            ("bool-dtype", False),
            ("categorical", False),
            ("int64", False),
            ("int32", False),
            ("uint64", False),
            ("uint32", False),
            ("datetime", False),
            ("float64", False),
            ("float32", False),
        ],
        indirect=["index"],
    )
    def test_is_object(self, index, expected, using_infer_string):
        # 如果使用推断字符串且索引对象的数据类型为 "string"，并且预期结果为 True，则将预期结果设为 False
        if using_infer_string and index.dtype == "string" and expected:
            expected = False
        # 断言判断索引对象是否为对象类型，与预期结果相符
        assert is_object_dtype(index) is expected

    # 测试索引对象的摘要方法
    def test_summary(self, index):
        # 调用索引对象的摘要方法
        index._summary()

    # 测试索引对象的逻辑兼容性
    def test_logical_compat(self, all_boolean_reductions, simple_index):
        # 获取简单索引对象
        index = simple_index
        # 对索引对象进行逻辑运算，比较左右两侧的结果是否相等
        left = getattr(index, all_boolean_reductions)()
        assert left == getattr(index.values, all_boolean_reductions)()
        # 将索引对象转换为 Series 后再进行逻辑运算，比较左右两侧的结果是否逻辑上相等
        right = getattr(index.to_series(), all_boolean_reductions)()
        # 对于字符串等情况，使用 np.any/all 而非 .any/all，左侧和右侧可能不完全相等
        assert bool(left) == bool(right)

    # 测试判断索引对象是否为对象类型
    @pytest.mark.parametrize(
        "index", ["string", "int64", "int32", "float64", "float32"], indirect=True
    )
    # 定义一个测试方法，用于测试按字符串标签删除元素
    def test_drop_by_str_label(self, index):
        # 获取索引的长度
        n = len(index)
        # 创建要删除的索引范围
        drop = index[list(range(5, 10))]
        # 在索引中删除指定的元素
        dropped = index.drop(drop)

        # 构建预期的索引，即删除指定范围后的索引
        expected = index[list(range(5)) + list(range(10, n))]
        # 断言删除操作后的索引是否与预期相等
        tm.assert_index_equal(dropped, expected)

        # 在索引中删除特定位置的元素
        dropped = index.drop(index[0])
        # 生成预期的索引，即删除首个元素后的索引
        expected = index[1:]
        # 断言删除操作后的索引是否与预期相等
        tm.assert_index_equal(dropped, expected)

    # 参数化测试，测试当传入不存在的键时是否抛出 KeyError 异常
    @pytest.mark.parametrize(
        "index", ["string", "int64", "int32", "float64", "float32"], indirect=True
    )
    @pytest.mark.parametrize("keys", [["foo", "bar"], ["1", "bar"]])
    def test_drop_by_str_label_raises_missing_keys(self, index, keys):
        # 断言在删除不存在的键时是否抛出 KeyError 异常
        with pytest.raises(KeyError, match=""):
            index.drop(keys)

    # 参数化测试，测试当忽略错误时按字符串标签删除元素的行为
    @pytest.mark.parametrize(
        "index", ["string", "int64", "int32", "float64", "float32"], indirect=True
    )
    def test_drop_by_str_label_errors_ignore(self, index):
        # 获取索引的长度
        n = len(index)
        # 创建要删除的索引范围
        drop = index[list(range(5, 10))]
        # 将 drop 转换为列表，并添加一个不存在的键 "foo"
        mixed = drop.tolist() + ["foo"]
        # 在索引中按字符串标签删除元素，忽略错误
        dropped = index.drop(mixed, errors="ignore")

        # 构建预期的索引，即删除指定范围后的索引
        expected = index[list(range(5)) + list(range(10, n))]
        # 断言删除操作后的索引是否与预期相等
        tm.assert_index_equal(dropped, expected)

        # 在索引中按字符串标签删除元素 "foo" 和 "bar"，忽略错误
        dropped = index.drop(["foo", "bar"], errors="ignore")
        # 生成预期的索引，即不删除任何元素的索引
        expected = index[list(range(n))]
        # 断言删除操作后的索引是否与预期相等
        tm.assert_index_equal(dropped, expected)

    # 测试按数值标签使用 .loc 删除元素的行为
    def test_drop_by_numeric_label_loc(self):
        # 创建一个数值类型的索引
        index = Index([1, 2, 3])
        # 在索引中按数值标签删除元素 1
        dropped = index.drop(1)
        # 生成预期的索引，即删除元素 1 后的索引
        expected = Index([2, 3])
        # 断言删除操作后的索引是否与预期相等
        tm.assert_index_equal(dropped, expected)

    # 测试按数值标签删除元素时是否抛出 KeyError 异常
    def test_drop_by_numeric_label_raises_missing_keys(self):
        # 创建一个数值类型的索引
        index = Index([1, 2, 3])
        # 断言在删除不存在的键时是否抛出 KeyError 异常
        with pytest.raises(KeyError, match=""):
            index.drop([3, 4])

    # 参数化测试，测试当忽略错误时按数值标签删除元素的行为
    @pytest.mark.parametrize(
        "key,expected", [(4, Index([1, 2, 3])), ([3, 4, 5], Index([1, 2]))]
    )
    def test_drop_by_numeric_label_errors_ignore(self, key, expected):
        # 创建一个数值类型的索引
        index = Index([1, 2, 3])
        # 在索引中按数值标签删除元素，忽略错误
        dropped = index.drop(key, errors="ignore")

        # 断言删除操作后的索引是否与预期相等
        tm.assert_index_equal(dropped, expected)

    # 参数化测试，测试在包含元组的情况下按标签删除元素的行为
    @pytest.mark.parametrize(
        "values",
        [["a", "b", ("c", "d")], ["a", ("c", "d"), "b"], [("c", "d"), "a", "b"]],
    )
    @pytest.mark.parametrize("to_drop", [[("c", "d"), "a"], ["a", ("c", "d")]])
    def test_drop_tuple(self, values, to_drop):
        # GH 18304
        # 创建一个包含字符串和元组的索引
        index = Index(values)
        # 生成预期的索引，即删除指定元素后的索引
        expected = Index(["b"], dtype=object)

        # 在索引中按标签删除元素
        result = index.drop(to_drop)
        # 断言删除操作后的索引是否与预期相等
        tm.assert_index_equal(result, expected)

        # 在索引中按标签删除第一个元组元素
        removed = index.drop(to_drop[0])
        # 逐个删除剩余的元素并断言是否抛出 KeyError 异常
        for drop_me in to_drop[1], [to_drop[1]]:
            result = removed.drop(drop_me)
            tm.assert_index_equal(result, expected)

        # 在索引中按标签删除第二个元组元素，并验证是否抛出 KeyError 异常
        removed = index.drop(to_drop[1])
        # 生成预期的异常消息
        msg = rf"\"\[{re.escape(to_drop[1].__repr__())}\] not found in axis\""
        # 逐个删除剩余的元素并断言是否抛出指定的异常消息
        for drop_me in to_drop[1], [to_drop[1]]:
            with pytest.raises(KeyError, match=msg):
                removed.drop(drop_me)
    # 使用 pytest 的标记来忽略特定的警告信息，这里忽略了关于 PeriodDtype[B] 被弃用的 FutureWarning
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 测试函数，用于测试带有重复索引的情况
    def test_drop_with_duplicates_in_index(self, index):
        # 标题：GH38051
        # 如果索引长度为0或者是 MultiIndex 类型，则跳过测试，因为对于空的 MultiIndex 没有意义
        if len(index) == 0 or isinstance(index, MultiIndex):
            pytest.skip("Test doesn't make sense for empty MultiIndex")
        # 如果索引类型是 IntervalIndex 且不是在64位系统上，则跳过测试
        if isinstance(index, IntervalIndex) and not IS64:
            pytest.skip("Cannot test IntervalIndex with int64 dtype on 32 bit platform")
        # 将索引去重并重复每个元素两次
        index = index.unique().repeat(2)
        # 预期结果是从索引的第三个元素开始到末尾
        expected = index[2:]
        # 对索引进行删除操作，删除第一个元素
        result = index.drop(index[0])
        # 使用测试模块的断言来比较两个索引是否相等
        tm.assert_index_equal(result, expected)

    # 参数化测试函数，用于测试索引的单调性属性
    @pytest.mark.parametrize(
        "attr",
        [
            "is_monotonic_increasing",
            "is_monotonic_decreasing",
            "_is_strictly_monotonic_increasing",
            "_is_strictly_monotonic_decreasing",
        ],
    )
    def test_is_monotonic_incomparable(self, attr):
        # 创建一个包含整数、当前日期时间和整数的索引
        index = Index([5, datetime.now(), 7])
        # 使用 getattr() 获取索引对象的指定属性值，断言该属性值为 False
        assert not getattr(index, attr)

    # 参数化测试函数，用于测试索引是否在给定值列表中
    @pytest.mark.parametrize("values", [["foo", "bar", "quux"], {"foo", "bar", "quux"}])
    @pytest.mark.parametrize(
        "index,expected",
        [
            # 测试索引为列表时的情况，预期结果为布尔值列表
            (["qux", "baz", "foo", "bar"], [False, False, True, True]),
            # 空列表的情况，预期结果也是空列表
            ([], []),  # empty
        ],
    )
    def test_isin(self, values, index, expected):
        # 创建一个索引对象，将给定的索引列表作为其数据
        index = Index(index)
        # 判断索引中的元素是否在给定的值列表中，返回布尔数组
        result = index.isin(values)
        # 将预期结果转换为 NumPy 的布尔数组类型
        expected = np.array(expected, dtype=bool)
        # 使用测试模块的断言来比较两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数，用于测试带有 NaN 和普通对象的索引操作
    def test_isin_nan_common_object(
        self, nulls_fixture, nulls_fixture2, using_infer_string
    ):
        # 创建一个包含字符串 'a' 和 nulls_fixture 的索引对象
        idx = Index(["a", nulls_fixture])

        # 如果 nulls_fixture 和 nulls_fixture2 都是 float 类型且均为 NaN
        if (
            isinstance(nulls_fixture, float)
            and isinstance(nulls_fixture2, float)
            and math.isnan(nulls_fixture)
            and math.isnan(nulls_fixture2)
        ):
            # 断言使用测试模块的函数，比较索引是否在给定值列表中的情况，预期结果为 False 和 True
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, True]),
            )

        # 如果 nulls_fixture 和 nulls_fixture2 相等，则应保留 NA 类型
        elif nulls_fixture is nulls_fixture2:
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, True]),
            )

        # 如果 using_infer_string 为 True 并且索引的类型是字符串
        elif using_infer_string and idx.dtype == "string":
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, True]),
            )

        else:
            # 默认情况下，比较索引是否在给定值列表中的情况，预期结果为 False 和 False
            tm.assert_numpy_array_equal(
                idx.isin([nulls_fixture2]),
                np.array([False, False]),
            )
    # 测试函数，用于检查在 float64 类型索引中处理 NaN 或 NaT 的行为
    def test_isin_nan_common_float64(self, nulls_fixture, float_numpy_dtype):
        # 获取 float64 数据类型
        dtype = float_numpy_dtype

        # 如果 nulls_fixture 是 pd.NaT 或 pd.NA，则执行以下操作
        if nulls_fixture is pd.NaT or nulls_fixture is pd.NA:
            # 检查：
            # 1) 无法使用该值构造 float64 类型的索引
            # 2) NaN 不应该在 .isin(nulls_fixture) 中
            msg = (
                r"float\(\) argument must be a string or a (real )?number, "
                f"not {type(nulls_fixture).__name__!r}"
            )
            # 使用 pytest 检查是否会引发 TypeError，并匹配预期的错误消息
            with pytest.raises(TypeError, match=msg):
                Index([1.0, nulls_fixture], dtype=dtype)

            # 创建一个索引对象 idx，包含 1.0 和 NaN
            idx = Index([1.0, np.nan], dtype=dtype)
            # 断言 .isin([nulls_fixture]) 返回任何结果为 False
            assert not idx.isin([nulls_fixture]).any()
            return

        # 如果 nulls_fixture 不是 pd.NaT 或 pd.NA，则执行以下操作
        # 创建一个索引对象 idx，包含 1.0 和 nulls_fixture
        idx = Index([1.0, nulls_fixture], dtype=dtype)
        # 检查 idx 是否包含 np.nan，返回结果存储在 res 中
        res = idx.isin([np.nan])
        # 使用测试工具函数 tm.assert_numpy_array_equal 检查 res 是否与预期的数组相等
        tm.assert_numpy_array_equal(res, np.array([False, True]))

        # 对于 idx 中的 pd.NaT，不能与 NaN 进行比较
        res = idx.isin([pd.NaT])
        tm.assert_numpy_array_equal(res, np.array([False, False]))

    # 参数化测试函数，测试 .isin 方法中的 level 参数
    @pytest.mark.parametrize("level", [0, -1])
    @pytest.mark.parametrize(
        "index",
        [
            ["qux", "baz", "foo", "bar"],  # 字符串列表作为索引
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),  # 浮点数数组作为索引
        ],
    )
    def test_isin_level_kwarg(self, level, index):
        # 创建 Index 对象，使用传入的 index
        index = Index(index)
        # 获取 index 中倒数第二和最后一个元素，并添加一个不存在的值
        values = index.tolist()[-2:] + ["nonexisting"]

        # 创建一个预期的结果数组
        expected = np.array([False, False, True, True])
        # 使用测试工具函数 tm.assert_numpy_array_equal 检查 .isin 方法的结果是否与预期一致
        tm.assert_numpy_array_equal(expected, index.isin(values, level=level))

        # 将 index 的名称设置为 "foobar"
        index.name = "foobar"
        # 再次检查 .isin 方法的结果是否与预期一致，使用名称字符串作为 level 参数
        tm.assert_numpy_array_equal(expected, index.isin(values, level="foobar"))

    # 测试函数，检查 .isin 方法中的错误 level 参数是否会引发异常
    def test_isin_level_kwarg_bad_level_raises(self, index):
        # 遍历多个错误的 level 参数值
        for level in [10, index.nlevels, -(index.nlevels + 1)]:
            # 使用 pytest 检查是否会引发 IndexError，并匹配预期的错误消息
            with pytest.raises(IndexError, match="Too many levels"):
                index.isin([], level=level)

    # 参数化测试函数，检查 .isin 方法中的错误 label 参数是否会引发异常
    @pytest.mark.parametrize("label", [1.0, "foobar", "xyzzy", np.nan])
    def test_isin_level_kwarg_bad_label_raises(self, label, index):
        # 如果 index 是 MultiIndex 类型，则重命名其名称
        if isinstance(index, MultiIndex):
            index = index.rename(["foo", "bar"] + index.names[2:])
            msg = f"'Level {label} not found'"
        else:
            index = index.rename("foo")
            msg = rf"Requested level \({label}\) does not match index name \(foo\)"
        # 使用 pytest 检查是否会引发 KeyError，并匹配预期的错误消息
        with pytest.raises(KeyError, match=msg):
            index.isin([], level=label)

    # 参数化测试函数，检查 .isin 方法处理空值的行为
    @pytest.mark.parametrize("empty", [[], Series(dtype=object), np.array([])])
    def test_isin_empty(self, empty):
        # 创建一个 Index 对象，包含 ["a", "b"] 作为索引
        index = Index(["a", "b"])
        # 创建一个预期的结果数组
        expected = np.array([False, False])

        # 调用 .isin 方法，检查其处理空值的结果
        result = index.isin(empty)
        # 使用测试工具函数 tm.assert_numpy_array_equal 检查结果是否与预期一致
        tm.assert_numpy_array_equal(expected, result)

    # 跳过测试的装饰器，依赖于是否存在 pyarrow
    @td.skip_if_no("pyarrow")
    # 定义测试方法，用于测试 Index 对象中包含特定值的情况（字符串或 None）
    def test_isin_arrow_string_null(self):
        # GH#55821
        # 创建一个 Index 对象，包含字符串数组 ["a", "b"]，数据类型为 "string[pyarrow_numpy]"
        index = Index(["a", "b"], dtype="string[pyarrow_numpy]")
        # 调用 isin 方法，检查是否包含 None，返回结果存储在 result 中
        result = index.isin([None])
        # 期望的结果是一个布尔类型的 NumPy 数组，表示每个元素是否等于 None
        expected = np.array([False, False])
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "values",
        [
            [1, 2, 3, 4],                                     # 整数列表
            [1.0, 2.0, 3.0, 4.0],                             # 浮点数列表
            [True, True, True, True],                         # 布尔值列表
            ["foo", "bar", "baz", "qux"],                     # 字符串列表
            date_range("2018-01-01", freq="D", periods=4),    # 日期范围对象
        ],
    )
    # 定义测试方法，用于测试 Index 对象与给定值的比较
    def test_boolean_cmp(self, values):
        # 创建一个 Index 对象，包含 values 中的元素
        index = Index(values)
        # 将 Index 对象与 values 列表进行比较，结果存储在 result 中
        result = index == values
        # 期望的结果是一个布尔类型的 NumPy 数组，表示每个元素是否与自身相等
        expected = np.array([True, True, True, True], dtype=bool)
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("index", ["string"], indirect=True)
    @pytest.mark.parametrize("name,level", [(None, 0), ("a", "a")])
    # 定义测试方法，用于测试获取 Index 对象中指定层级的值
    def test_get_level_values(self, index, name, level):
        # 创建一个 Index 对象的副本 expected
        expected = index.copy()
        # 如果 name 不为空，则将 expected 对象的名称设置为 name
        if name:
            expected.name = name
        # 获取 expected 对象中指定层级 level 的值，存储在 result 中
        result = expected.get_level_values(level)
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_index_equal(result, expected)

    # 定义测试方法，用于测试切片操作后保留名称的情况
    def test_slice_keep_name(self):
        # 创建一个 Index 对象，包含字符串数组 ["a", "b"]，并设置名称为 "asdf"
        index = Index(["a", "b"], name="asdf")
        # 断言 index 对象的名称与 index[1:] 切片后的对象的名称相同
        assert index.name == index[1:].name

    @pytest.mark.parametrize("index", [
        "string", "datetime", "int64", "int32", "uint64", "uint32", "float64", "float32"
    ], indirect=True)
    # 定义测试方法，用于测试 Index 对象与自身进行连接
    def test_join_self(self, index, join_type):
        # 将 index 对象与自身进行连接，连接方式由 join_type 指定，结果存储在 result 中
        result = index.join(index, how=join_type)
        # 如果连接方式为 "outer"，则对期望的结果 expected 进行排序
        expected = index
        if join_type == "outer":
            expected = expected.sort_values()
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("method", ["strip", "rstrip", "lstrip"])
    # 定义测试方法，用于测试字符串属性方法
    def test_str_attribute(self, method):
        # 创建一个 Index 对象，包含字符串数组 [" jack", "jill ", " jesse ", "frank"]
        index = Index([" jack", "jill ", " jesse ", "frank"])
        # 创建一个期望的 Index 对象，使用 getattr 函数对 index 中的每个元素应用 method 方法
        expected = Index([getattr(str, method)(x) for x in index.values])
        # 调用 index 对象的 str 属性下的 method 方法，结果存储在 result 中
        result = getattr(index.str, method)()
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("index", [
        Index(range(5)),                                         # 整数范围的 Index 对象
        date_range("2020-01-01", periods=10),                    # 日期范围对象的 Index 对象
        MultiIndex.from_tuples([("foo", "1"), ("bar", "3")]),    # 多级索引对象
        period_range(start="2000", end="2010", freq="Y"),        # 期间范围对象的 Index 对象
    ])
    # 定义测试方法，测试对不支持的字符串属性方法调用时是否引发异常
    def test_str_attribute_raises(self, index):
        # 使用 pytest 的断言检查调用 index 对象的 str 属性的 repeat 方法是否引发 AttributeError 异常
        with pytest.raises(AttributeError, match="only use .str accessor"):
            index.str.repeat(2)

    @pytest.mark.parametrize("expand,expected", [
        (None, Index([["a", "b", "c"], ["d", "e"], ["f"]])),      # expand 为 None 时的期望结果
        (False, Index([["a", "b", "c"], ["d", "e"], ["f"]])),     # expand 为 False 时的期望结果
        (True, MultiIndex.from_tuples([                           # expand 为 True 时的期望结果，使用 MultiIndex
            ("a", "b", "c"), ("d", "e", np.nan), ("f", np.nan, np.nan)
        ])),
    ])
    # 定义一个测试方法，用于测试字符串的分割操作
    def test_str_split(self, expand, expected):
        # 创建一个索引对象，包含多个字符串作为其索引
        index = Index(["a b c", "d e", "f"])
        if expand is not None:
            # 如果传入了expand参数，则调用索引对象的str.split方法进行分割
            result = index.str.split(expand=expand)
        else:
            # 否则调用索引对象的str.split方法进行默认分割
            result = index.str.split()

        # 断言分割后的结果与期望值相等
        tm.assert_index_equal(result, expected)

    # 定义一个测试方法，测试字符串的布尔类型返回
    def test_str_bool_return(self):
        # 创建一个索引对象，包含多个字符串作为其索引
        index = Index(["a1", "a2", "b1", "b2"])
        # 调用索引对象的str.startswith方法进行字符串开头匹配，返回布尔类型的np.array
        result = index.str.startswith("a")
        expected = np.array([True, True, False, False])

        # 断言返回的结果与期望值相等，并且返回结果类型为np.ndarray
        tm.assert_numpy_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    # 定义一个测试方法，测试字符串的布尔类型索引
    def test_str_bool_series_indexing(self):
        # 创建一个索引对象，包含多个字符串作为其索引
        index = Index(["a1", "a2", "b1", "b2"])
        # 创建一个Series对象，使用上述索引对象作为索引，数值为0到3
        s = Series(range(4), index=index)

        # 使用布尔类型索引获取Series中以"a"开头的元素
        result = s[s.index.str.startswith("a")]
        expected = Series(range(2), index=["a1", "a2"])
        # 断言获取的Series与期望的Series相等
        tm.assert_series_equal(result, expected)

    # 使用pytest的参数化装饰器，定义一个测试方法，用于测试Tab补全功能
    @pytest.mark.parametrize(
        "index,expected", [(list("abcd"), True), (range(4), False)]
    )
    def test_tab_completion(self, index, expected):
        # GH 9910
        # 创建一个索引对象，根据传入的index参数初始化
        index = Index(index)
        # 检查字符串"str"是否在索引对象的属性中
        result = "str" in dir(index)
        # 断言检查结果与期望值相等
        assert result == expected

    # 定义一个测试方法，验证索引操作不改变索引类别
    def test_indexing_doesnt_change_class(self):
        # 创建一个索引对象，包含整数和字符串作为其索引
        index = Index([1, 2, 3, "a", "b", "c"])

        # 断言切片后的索引对象与期望的索引对象相同
        assert index[1:3].identical(Index([2, 3], dtype=np.object_))
        # 断言选择索引后的索引对象与期望的索引对象相同
        assert index[[0, 1]].identical(Index([1, 2], dtype=np.object_))

    # 定义一个测试方法，测试外连接和排序功能
    def test_outer_join_sort(self):
        # 创建一个随机排列的整数索引对象
        left_index = Index(np.random.default_rng(2).permutation(15))
        # 创建一个日期范围索引对象
        right_index = date_range("2020-01-01", periods=10)

        # 断言外连接操作产生RuntimeWarning警告
        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            result = left_index.join(right_index, how="outer")

        # 断言外连接后的结果与预期结果相等
        with tm.assert_produces_warning(RuntimeWarning, match="not supported between"):
            expected = left_index.astype(object).union(right_index.astype(object))

        tm.assert_index_equal(result, expected)

    # 定义一个测试方法，测试使用take方法并指定填充值的功能
    def test_take_fill_value(self):
        # GH 12631
        # 创建一个具名索引对象，包含字符"ABC"作为其索引
        index = Index(list("ABC"), name="xxx")
        # 使用take方法根据传入的索引数组选择元素
        result = index.take(np.array([1, 0, -1]))
        expected = Index(list("BAC"), name="xxx")
        # 断言take后的索引对象与期望的索引对象相等
        tm.assert_index_equal(result, expected)

        # 使用填充值选项，在索引不存在时填充为True
        result = index.take(np.array([1, 0, -1]), fill_value=True)
        expected = Index(["B", "A", np.nan], name="xxx")
        # 断言take后的索引对象与期望的索引对象相等
        tm.assert_index_equal(result, expected)

        # 禁用填充功能，断言索引对象与期望的索引对象相等
        result = index.take(np.array([1, 0, -1]), allow_fill=False, fill_value=True)
        expected = Index(["B", "A", "C"], name="xxx")
        tm.assert_index_equal(result, expected)
    # 定义测试函数，测试当 allow_fill=True 且 fill_value 不为 None 时是否会引发异常
    def test_take_fill_value_none_raises(self):
        # 创建索引对象，包含元素 ["A", "B", "C"]，并指定名称为 "xxx"
        index = Index(list("ABC"), name="xxx")
        # 定义错误消息
        msg = (
            "When allow_fill=True and fill_value is not None, "
            "all indices must be >= -1"
        )

        # 断言调用 take 方法时是否引发 ValueError 异常，并检查异常消息是否匹配预期消息
        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -2]), fill_value=True)
        # 再次断言调用 take 方法时是否引发 ValueError 异常，并检查异常消息是否匹配预期消息
        with pytest.raises(ValueError, match=msg):
            index.take(np.array([1, 0, -5]), fill_value=True)

    # 定义测试函数，测试当索引超出范围时是否会引发 IndexError 异常
    def test_take_bad_bounds_raises(self):
        # 创建索引对象，包含元素 ["A", "B", "C"]，并指定名称为 "xxx"
        index = Index(list("ABC"), name="xxx")
        # 断言调用 take 方法时是否引发 IndexError 异常，并检查异常消息是否包含 "out of bounds"
        with pytest.raises(IndexError, match="out of bounds"):
            index.take(np.array([1, -5]))

    # 使用参数化测试装饰器标记的测试函数，测试 reindex 方法是否在目标为列表或 ndarray 时保留名称
    @pytest.mark.parametrize("name", [None, "foobar"])
    @pytest.mark.parametrize(
        "labels",
        [
            [],  # 空列表
            np.array([]),  # 空 ndarray
            ["A", "B", "C"],  # 包含字符串元素的列表
            ["C", "B", "A"],  # 顺序相反的字符串列表
            np.array(["A", "B", "C"]),  # 包含字符串元素的 ndarray
            np.array(["C", "B", "A"]),  # 顺序相反的字符串 ndarray
            # 即使 dtype 发生变化，也必须保留名称
            date_range("20130101", periods=3).values,  # 时间范围对象的值数组
            date_range("20130101", periods=3).tolist(),  # 时间范围对象的列表
        ],
    )
    def test_reindex_preserves_name_if_target_is_list_or_ndarray(self, name, labels):
        # GH6552
        # 创建索引对象，包含整数元素 [0, 1, 2]
        index = Index([0, 1, 2])
        # 设置索引对象的名称为指定的 name
        index.name = name
        # 断言对索引对象调用 reindex 方法后的第一个元素的名称是否与原名称相同
        assert index.reindex(labels)[0].name == name

    # 使用参数化测试装饰器标记的测试函数，测试 reindex 方法是否在目标为空列表或数组时保留类型
    @pytest.mark.parametrize("labels", [[], np.array([]), np.array([], dtype=np.int64)])
    def test_reindex_preserves_type_if_target_is_empty_list_or_array(self, labels):
        # GH7774
        # 创建索引对象，包含字符元素 ["a", "b", "c"]
        index = Index(list("abc"))
        # 断言对索引对象调用 reindex 方法后的第一个元素的数据类型是否与原类型相同
        assert index.reindex(labels)[0].dtype.type == index.dtype.type

    # 测试函数，测试 reindex 方法是否在目标为空索引对象时不保留类型
    def test_reindex_doesnt_preserve_type_if_target_is_empty_index(self):
        # GH7774
        # 创建索引对象，包含字符元素 ["a", "b", "c"]
        index = Index(list("abc"))
        # 创建空的 DatetimeIndex 对象
        labels = DatetimeIndex([])
        # 设置目标数据类型为 np.datetime64
        dtype = np.datetime64
        # 断言对索引对象调用 reindex 方法后的第一个元素的数据类型是否为预期的 dtype
        assert index.reindex(labels)[0].dtype.type == dtype

    # 测试函数，测试 reindex 方法是否在目标为空数值索引对象时不保留类型
    def test_reindex_doesnt_preserve_type_if_target_is_empty_index_numeric(
        self, any_real_numpy_dtype
    ):
        # GH7774
        # 获取任意的 numpy 数据类型
        dtype = any_real_numpy_dtype
        # 创建索引对象，包含字符元素 ["a", "b", "c"]
        index = Index(list("abc"))
        # 创建空的索引对象，指定数据类型为 dtype
        labels = Index([], dtype=dtype)
        # 断言对索引对象调用 reindex 方法后的第一个元素的数据类型是否与原数据类型相同
        assert index.reindex(labels)[0].dtype == dtype

    # 测试函数，测试 reindex 方法在忽略指定级别时的行为
    def test_reindex_no_type_preserve_target_empty_mi(self):
        # 创建索引对象，包含字符元素 ["a", "b", "c"]
        index = Index(list("abc"))
        # 调用 reindex 方法，将目标设置为包含空索引对象的 MultiIndex
        result = index.reindex(
            MultiIndex([Index([], np.int64), Index([], np.float64)], [[], []])
        )[0]
        # 断言结果的第一个级别的数据类型是否为 np.int64
        assert result.levels[0].dtype.type == np.int64
        # 断言结果的第二个级别的数据类型是否为 np.float64
        assert result.levels[1].dtype.type == np.float64

    # 测试函数，测试 reindex 方法在忽略指定级别时的行为
    def test_reindex_ignoring_level(self):
        # GH#35132
        # 创建索引对象，包含整数元素 [1, 2, 3]，并指定名称为 "x"
        idx = Index([1, 2, 3], name="x")
        # 创建另一个索引对象，包含整数元素 [1, 2, 3, 4]，并指定名称为 "x"
        idx2 = Index([1, 2, 3, 4], name="x")
        # 创建预期结果的索引对象，包含整数元素 [1, 2, 3, 4]，并指定名称为 "x"
        expected = Index([1, 2, 3, 4], name="x")
        # 调用 reindex 方法，忽略级别 "x" 的行为
        result, _ = idx.reindex(idx2, level="x")
        # 使用 assert_index_equal 检查结果是否与预期一致
        tm.assert_index_equal(result, expected)
    def test_groupby(self):
        # 创建一个 Index 对象，包含整数范围 [0, 4]
        index = Index(range(5))
        # 对 Index 对象进行按照指定的分组进行分组操作
        result = index.groupby(np.array([1, 1, 2, 2, 2]))
        # 预期的分组结果，一个字典，键是分组的值，值是对应的 Index 对象
        expected = {1: Index([0, 1]), 2: Index([2, 3, 4])}

        # 使用测试工具检查结果与预期是否一致
        tm.assert_dict_equal(result, expected)

    @pytest.mark.parametrize(
        "mi,expected",
        [
            # 使用给定的元组列表创建 MultiIndex 对象，并与预期的布尔数组进行比较
            (MultiIndex.from_tuples([(1, 2), (4, 5)]), np.array([True, True])),
            (MultiIndex.from_tuples([(1, 2), (4, 6)]), np.array([True, False])),
        ],
    )
    def test_equals_op_multiindex(self, mi, expected):
        # GH9785
        # 测试多级索引的比较操作
        df = DataFrame(
            [3, 6],
            columns=["c"],
            # 使用给定的数组列表创建 DataFrame，设置其索引为 MultiIndex 对象
            index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]),
        )

        # 执行比较操作，得到比较结果
        result = df.index == mi
        # 使用测试工具检查比较结果与预期是否一致
        tm.assert_numpy_array_equal(result, expected)

    def test_equals_op_multiindex_identify(self):
        df = DataFrame(
            [3, 6],
            columns=["c"],
            # 使用给定的数组列表创建 DataFrame，设置其索引为 MultiIndex 对象
            index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]),
        )

        # 执行索引与自身的比较操作
        result = df.index == df.index
        # 预期的比较结果，一个布尔数组，全为 True
        expected = np.array([True, True])
        # 使用测试工具检查比较结果与预期是否一致
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            # 创建不匹配长度的 MultiIndex 对象和 Index 对象
            MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)]),
            Index(["foo", "bar", "baz"]),
        ],
    )
    def test_equals_op_mismatched_multiindex_raises(self, index):
        df = DataFrame(
            [3, 6],
            columns=["c"],
            # 使用给定的数组列表创建 DataFrame，设置其索引为 MultiIndex 对象
            index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=["a", "b"]),
        )

        # 使用 pytest 断言检查是否会引发 ValueError，并且错误信息包含 "Lengths must match"
        with pytest.raises(ValueError, match="Lengths must match"):
            df.index == index

    def test_equals_op_index_vs_mi_same_length(self, using_infer_string):
        # 创建一个 MultiIndex 对象和一个 Index 对象
        mi = MultiIndex.from_tuples([(1, 2), (4, 5), (8, 9)])
        index = Index(["foo", "bar", "baz"])

        # 执行 MultiIndex 对象与 Index 对象的比较操作
        result = mi == index
        # 预期的比较结果，一个布尔数组，全为 False
        expected = np.array([False, False, False])
        # 使用测试工具检查比较结果与预期是否一致
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize(
        "dt_conv, arg",
        [
            # 使用不同的时间转换函数和参数列表进行测试
            (pd.to_datetime, ["2000-01-01", "2000-01-02"]),
            (pd.to_timedelta, ["01:02:03", "01:02:04"]),
        ],
    )
    def test_dt_conversion_preserves_name(self, dt_conv, arg):
        # GH 10875
        # 创建一个带有指定名称的 Index 对象
        index = Index(arg, name="label")
        # 使用断言验证索引对象的名称在进行时间转换后是否保持不变
        assert index.name == dt_conv(index).name

    def test_cached_properties_not_settable(self):
        # 创建一个 Index 对象
        index = Index([1, 2, 3])
        # 使用 pytest 断言检查是否会引发 AttributeError，并且错误信息包含 "Can't set attribute"
        with pytest.raises(AttributeError, match="Can't set attribute"):
            index.is_unique = False
    def test_tab_complete_warning(self, ip):
        # 导入 IPython 并检查版本，如果版本低于 6.0.0 则跳过测试
        pytest.importorskip("IPython", minversion="6.0.0")
        # 从 IPython 中导入 provisionalcompleter 类
        from IPython.core.completer import provisionalcompleter

        # 定义测试用的代码字符串
        code = "import pandas as pd; idx = pd.Index([1, 2])"
        # 在 IPython 实例中执行代码字符串
        ip.run_cell(code)

        # 在运行时忽略 Deprecation 警告的上下文管理器
        with tm.assert_produces_warning(None, raise_on_extra_warnings=False):
            with provisionalcompleter("ignore"):
                # 获取 IPython Completer 的补全列表，针对 'idx.' 的补全
                list(ip.Completer.completions("idx.", 4))

    def test_contains_method_removed(self, index):
        # 如果 index 是 IntervalIndex 类型，则调用 contains 方法
        if isinstance(index, IntervalIndex):
            index.contains(1)
        else:
            # 否则抛出 AttributeError 异常，提示对象没有 'contains' 属性
            msg = f"'{type(index).__name__}' object has no attribute 'contains'"
            with pytest.raises(AttributeError, match=msg):
                index.contains(1)

    def test_sortlevel(self):
        # 创建一个 Index 对象，包含整数列表
        index = Index([5, 4, 3, 2, 1])
        # 检查 sortlevel 方法在 ascending 参数为字符串 "True" 时抛出异常的情况
        with pytest.raises(Exception, match="ascending must be a single bool value or"):
            index.sortlevel(ascending="True")

        # 检查 sortlevel 方法在 ascending 参数为布尔列表 [True, True] 时抛出异常的情况
        with pytest.raises(
            Exception, match="ascending must be a list of bool values of length 1"
        ):
            index.sortlevel(ascending=[True, True])

        # 检查 sortlevel 方法在 ascending 参数为字符串列表 ["True"] 时抛出异常的情况
        with pytest.raises(Exception, match="ascending must be a bool value"):
            index.sortlevel(ascending=["True"])

        # 检查 sortlevel 方法在 ascending 参数为单个布尔值 True 时的排序结果
        expected = Index([1, 2, 3, 4, 5])
        result = index.sortlevel(ascending=[True])
        tm.assert_index_equal(result[0], expected)

        # 检查 sortlevel 方法在 ascending 参数为布尔值 True 时的排序结果
        expected = Index([1, 2, 3, 4, 5])
        result = index.sortlevel(ascending=True)
        tm.assert_index_equal(result[0], expected)

        # 检查 sortlevel 方法在 ascending 参数为布尔值 False 时的排序结果
        expected = Index([5, 4, 3, 2, 1])
        result = index.sortlevel(ascending=False)
        tm.assert_index_equal(result[0], expected)

    def test_sortlevel_na_position(self):
        # GH#51612 测试 sortlevel 方法在 na_position 参数为 "first" 时的排序结果
        idx = Index([1, np.nan])
        result = idx.sortlevel(na_position="first")[0]
        expected = Index([np.nan, 1])
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "periods, expected_results",
        [
            (1, [np.nan, 10, 10, 10, 10]),
            (2, [np.nan, np.nan, 20, 20, 20]),
            (3, [np.nan, np.nan, np.nan, 30, 30]),
        ],
    )
    def test_index_diff(self, periods, expected_results):
        # GH#19708 测试 Index 对象的 diff 方法在不同 periods 下的表现
        idx = Index([10, 20, 30, 40, 50])
        result = idx.diff(periods)
        expected = Index(expected_results)

        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        "decimals, expected_results",
        [
            (0, [1.0, 2.0, 3.0]),
            (1, [1.2, 2.3, 3.5]),
            (2, [1.23, 2.35, 3.46]),
        ],
    )
    # 定义一个测试方法，用于测试索引对象的舍入功能
    def test_index_round(self, decimals, expected_results):
        # GH#19708
        # 创建一个包含浮点数的索引对象
        idx = Index([1.234, 2.345, 3.456])
        # 对索引对象进行舍入操作，保留指定小数位数
        result = idx.round(decimals)
        # 创建预期的索引对象，以便与结果进行比较
        expected = Index(expected_results)

        # 断言结果索引与预期索引相等
        tm.assert_index_equal(result, expected)
class TestMixedIntIndex:
    # Mostly the tests from common.py for which the results differ
    # in py2 and py3 because ints and strings are uncomparable in py3
    # (GH 13514)

    @pytest.fixture
    def simple_index(self) -> Index:
        # 返回一个包含混合类型元素的 Index 对象作为测试数据
        return Index([0, "a", 1, "b", 2, "c"])

    def test_argsort(self, simple_index):
        # 使用 simple_index 进行测试
        index = simple_index
        # 检查是否抛出预期的 TypeError 异常，异常信息应包含指定字符串
        with pytest.raises(TypeError, match="'>|<' not supported"):
            index.argsort()

    def test_numpy_argsort(self, simple_index):
        # 使用 simple_index 进行测试
        index = simple_index
        # 检查是否抛出预期的 TypeError 异常，异常信息应包含指定字符串
        with pytest.raises(TypeError, match="'>|<' not supported"):
            np.argsort(index)

    def test_copy_name(self, simple_index):
        # 检查在初始化时传递的 "name" 参数是否被正确使用
        # GH12309
        index = simple_index

        # 使用传递了 copy=True 和 name="mario" 的构造方法创建新对象 first
        first = type(index)(index, copy=True, name="mario")
        # 使用 type(first) 创建第二个对象 second，并且 copy=False
        second = type(first)(first, copy=False)

        # 虽然使用了 "copy=False"，但我们希望得到一个新对象。
        assert first is not second
        # 检查两个 Index 对象是否相等
        tm.assert_index_equal(first, second)

        # 检查 first 和 second 的 name 属性是否都为 "mario"
        assert first.name == "mario"
        assert second.name == "mario"

        # 创建 Series 对象 s1 和 s2，并进行相关操作
        s1 = Series(2, index=first)
        s2 = Series(3, index=second[:-1])

        s3 = s1 * s2

        # 检查 s3 的索引名是否为 "mario"
        assert s3.index.name == "mario"

    def test_copy_name2(self):
        # 检查在复制时添加 "name" 参数是否被正确使用
        # GH14302
        index = Index([1, 2], name="MyName")
        index1 = index.copy()

        # 检查 index 和 index1 是否相等
        tm.assert_index_equal(index, index1)

        # 使用 copy 方法复制 index，并指定新的 name="NewName"
        index2 = index.copy(name="NewName")
        # 检查 index 和 index2 是否相等，但不检查名称
        tm.assert_index_equal(index, index2, check_names=False)
        # 检查 index 和 index2 的 name 属性是否分别为 "MyName" 和 "NewName"
        assert index.name == "MyName"
        assert index2.name == "NewName"

    def test_unique_na(self):
        # 测试 Index 对象的 unique 方法处理 NaN 值的情况
        idx = Index([2, np.nan, 2, 1], name="my_index")
        expected = Index([2, np.nan, 1], name="my_index")
        result = idx.unique()
        # 检查 unique 方法返回的结果是否与预期相符
        tm.assert_index_equal(result, expected)

    def test_logical_compat(self, simple_index):
        # 测试 Index 对象的 all 和 any 方法是否与其 values 的 all 和 any 方法一致
        index = simple_index
        assert index.all() == index.values.all()
        assert index.any() == index.values.any()

    @pytest.mark.parametrize("how", ["any", "all"])
    @pytest.mark.parametrize("dtype", [None, object, "category"])
    @pytest.mark.parametrize(
        "vals,expected",
        [
            ([1, 2, 3], [1, 2, 3]),
            ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
            ([1.0, 2.0, np.nan, 3.0], [1.0, 2.0, 3.0]),
            (["A", "B", "C"], ["A", "B", "C"]),
            (["A", np.nan, "B", "C"], ["A", "B", "C"]),
        ],
    )
    def test_dropna(self, how, dtype, vals, expected):
        # GH 6194
        # 使用给定的 vals 和 dtype 创建 Index 对象
        index = Index(vals, dtype=dtype)
        # 对 Index 对象调用 dropna 方法，传入 how 参数
        result = index.dropna(how=how)
        # 创建预期的 Index 对象
        expected = Index(expected, dtype=dtype)
        # 检查 dropna 方法返回的结果是否与预期相符
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize("how", ["any", "all"])
    # 此处省略了部分代码
    @pytest.mark.parametrize(
        "index,expected",
        [  # 参数化测试，设置不同的输入和预期输出
            (
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]),  # 输入为DatetimeIndex对象
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]),  # 预期输出为相同的DatetimeIndex对象
            ),
            (
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03", pd.NaT]),  # 输入包含pd.NaT的DatetimeIndex对象
                DatetimeIndex(["2011-01-01", "2011-01-02", "2011-01-03"]),  # 预期输出为剔除了pd.NaT的DatetimeIndex对象
            ),
            (
                TimedeltaIndex(["1 days", "2 days", "3 days"]),  # 输入为TimedeltaIndex对象
                TimedeltaIndex(["1 days", "2 days", "3 days"]),  # 预期输出为相同的TimedeltaIndex对象
            ),
            (
                TimedeltaIndex([pd.NaT, "1 days", "2 days", "3 days", pd.NaT]),  # 输入包含pd.NaT的TimedeltaIndex对象
                TimedeltaIndex(["1 days", "2 days", "3 days"]),  # 预期输出为剔除了pd.NaT的TimedeltaIndex对象
            ),
            (
                PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"),  # 输入为PeriodIndex对象
                PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"),  # 预期输出为相同的PeriodIndex对象
            ),
            (
                PeriodIndex(["2012-02", "2012-04", "NaT", "2012-05"], freq="M"),  # 输入包含NaT的PeriodIndex对象
                PeriodIndex(["2012-02", "2012-04", "2012-05"], freq="M"),  # 预期输出为剔除了NaT的PeriodIndex对象
            ),
        ],
    )
    def test_dropna_dt_like(self, how, index, expected):
        # 执行测试：调用dropna方法对index进行处理，断言处理后的结果与预期的expected相等
        result = index.dropna(how=how)
        tm.assert_index_equal(result, expected)

    def test_dropna_invalid_how_raises(self):
        # 测试无效的how选项会引发ValueError异常
        msg = "invalid how option: xxx"
        with pytest.raises(ValueError, match=msg):
            Index([1, 2, 3]).dropna(how="xxx")

    @pytest.mark.parametrize(
        "index",
        [  # 参数化测试，设置不同的输入index
            Index([np.nan]),  # 输入为包含np.nan的Index对象
            Index([np.nan, 1]),  # 输入为包含np.nan和整数的Index对象
            Index([1, 2, np.nan]),  # 输入为包含整数和np.nan的Index对象
            Index(["a", "b", np.nan]),  # 输入为包含字符串和np.nan的Index对象
            pd.to_datetime(["NaT"]),  # 输入为包含NaT的datetime64对象
            pd.to_datetime(["NaT", "2000-01-01"]),  # 输入为包含NaT和日期的datetime64对象
            pd.to_datetime(["2000-01-01", "NaT", "2000-01-02"]),  # 输入为包含日期、NaT和日期的datetime64对象
            pd.to_timedelta(["1 day", "NaT"]),  # 输入为包含Timedelta和NaT的Timedelta对象
        ],
    )
    def test_is_monotonic_na(self, index):
        # 测试index对象的单调性和唯一性
        assert index.is_monotonic_increasing is False
        assert index.is_monotonic_decreasing is False
        assert index._is_strictly_monotonic_increasing is False
        assert index._is_strictly_monotonic_decreasing is False

    @pytest.mark.parametrize("dtype", ["f8", "m8[ns]", "M8[us]"])
    @pytest.mark.parametrize("unique_first", [True, False])
    def test_is_monotonic_unique_na(self, dtype, unique_first):
        # 测试特定dtype和unique_first条件下的index对象的单调性和唯一性
        index = Index([None, 1, 1], dtype=dtype)
        if unique_first:
            assert index.is_unique is False
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
        else:
            assert index.is_monotonic_increasing is False
            assert index.is_monotonic_decreasing is False
            assert index.is_unique is False

    def test_int_name_format(self, frame_or_series):
        # 测试在给定名称的情况下，Index对象的名称格式
        index = Index(["a", "b", "c"], name=0)
        result = frame_or_series(list(range(3)), index=index)
        assert "0" in repr(result)
    def test_str_to_bytes_raises(self):
        # GH 26447
        # 创建一个包含字符串形式的索引对象，范围是从0到9
        index = Index([str(x) for x in range(10)])
        # 定义一个用于匹配 TypeError 异常的正则表达式消息
        msg = "^'str' object cannot be interpreted as an integer$"
        # 使用 pytest 来断言会抛出 TypeError 异常，并且异常消息匹配指定的正则表达式
        with pytest.raises(TypeError, match=msg):
            # 将索引对象转换为字节表示，此处预期会抛出 TypeError 异常
            bytes(index)

    @pytest.mark.filterwarnings("ignore:elementwise comparison failed:FutureWarning")
    def test_index_with_tuple_bool(self):
        # GH34123
        # 创建一个包含元组的索引对象
        idx = Index([("a", "b"), ("b", "c"), ("c", "a")])
        # 检查索引对象中的元组是否与指定的元组 ("c", "a") 相等，返回一个布尔数组
        result = idx == ("c", "a")
        # 预期的结果是一个 Numpy 数组，表示每个元组是否与 ("c", "a") 相等
        expected = np.array([False, False, True])
        # 使用测试框架中的断言方法来比较两个 Numpy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)
class TestIndexUtils:
    @pytest.mark.parametrize(
        "data, names, expected",
        [
            ([[1, 2, 4]], None, Index([1, 2, 4])),
            ([[1, 2, 4]], ["name"], Index([1, 2, 4], name="name")),
            ([[1, 2, 3]], None, RangeIndex(1, 4)),
            ([[1, 2, 3]], ["name"], RangeIndex(1, 4, name="name")),
            (
                [["a", "a"], ["c", "d"]],
                None,
                MultiIndex([["a"], ["c", "d"]], [[0, 0], [0, 1]]),
            ),
            (
                [["a", "a"], ["c", "d"]],
                ["L1", "L2"],
                MultiIndex([["a"], ["c", "d"]], [[0, 0], [0, 1]], names=["L1", "L2"]),
            ),
        ],
    )
    def test_ensure_index_from_sequences(self, data, names, expected):
        result = ensure_index_from_sequences(data, names)
        tm.assert_index_equal(result, expected, exact=True)

    def test_ensure_index_mixed_closed_intervals(self):
        # GH27172
        intervals = [
            pd.Interval(0, 1, closed="left"),
            pd.Interval(1, 2, closed="right"),
            pd.Interval(2, 3, closed="neither"),
            pd.Interval(3, 4, closed="both"),
        ]
        result = ensure_index(intervals)
        expected = Index(intervals, dtype=object)
        tm.assert_index_equal(result, expected)

    def test_ensure_index_uint64(self):
        # with both 0 and a large-uint64, np.array will infer to float64
        #  https://github.com/numpy/numpy/issues/19146
        #  but a more accurate choice would be uint64
        values = [0, np.iinfo(np.uint64).max]

        result = ensure_index(values)
        assert list(result) == values

        expected = Index(values, dtype="uint64")
        tm.assert_index_equal(result, expected)

    def test_get_combined_index(self):
        result = _get_combined_index([])
        expected = RangeIndex(0)
        tm.assert_index_equal(result, expected)


@pytest.mark.parametrize(
    "opname",
    [
        "eq",
        "ne",
        "le",
        "lt",
        "ge",
        "gt",
        "add",
        "radd",
        "sub",
        "rsub",
        "mul",
        "rmul",
        "truediv",
        "rtruediv",
        "floordiv",
        "rfloordiv",
        "pow",
        "rpow",
        "mod",
        "divmod",
    ],
)
def test_generated_op_names(opname, index):
    opname = f"__{opname}__"  # 构建特殊方法名，例如 '__eq__'
    method = getattr(index, opname)  # 获取对象中对应的特殊方法
    assert method.__name__ == opname  # 断言方法的名称与构建的方法名相符


@pytest.mark.parametrize(
    "klass",
    [
        partial(CategoricalIndex, data=[1]),
        partial(DatetimeIndex, data=["2020-01-01"]),
        partial(PeriodIndex, data=["2020-01-01"]),
        partial(TimedeltaIndex, data=["1 day"]),
        partial(RangeIndex, data=range(1)),
        partial(IntervalIndex, data=[pd.Interval(0, 1)]),
        partial(Index, data=["a"], dtype=object),
        partial(MultiIndex, levels=[1], codes=[0]),
    ],
)
def test_index_subclass_constructor_wrong_kwargs(klass):
    # GH #19348
    pass  # 测试未提供具体的断言或操作，因此使用 pass 表示占位符，保持结构完整
    # 使用 pytest 的语法，检测是否会抛出 TypeError 异常，并且异常消息中包含指定的字符串 "unexpected keyword argument"
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        # 使用 klass 类创建一个实例，传递参数 foo="bar"，预期会抛出 TypeError 异常
        klass(foo="bar")
# 定义一个测试函数，用于检查使用已弃用的快速路径创建索引时是否会引发预期的 TypeError 异常
def test_deprecated_fastpath():
    # 定义用于匹配异常消息的正则表达式模式
    msg = "[Uu]nexpected keyword argument"

    # 使用 pytest 检查当使用对象为对象数组、int64 类型数组、RangeIndex 和 CategoricalIndex 时，
    # 是否会引发带有特定异常消息的 TypeError 异常
    with pytest.raises(TypeError, match=msg):
        Index(np.array(["a", "b"], dtype=object), name="test", fastpath=True)

    with pytest.raises(TypeError, match=msg):
        Index(np.array([1, 2, 3], dtype="int64"), name="test", fastpath=True)

    with pytest.raises(TypeError, match=msg):
        RangeIndex(0, 5, 2, name="test", fastpath=True)

    with pytest.raises(TypeError, match=msg):
        CategoricalIndex(["a", "b", "c"], name="test", fastpath=True)


# 定义一个测试函数，用于验证创建索引对象时，处理无效索引形状是否正确引发异常
def test_shape_of_invalid_index():
    # 在 2.0 之前，可以创建由多维数组支持的“无效”索引对象。这个测试确保返回的形状与底层数组一致，
    # 以兼容 matplotlib
    idx = Index([0, 1, 2, 3])
    with pytest.raises(ValueError, match="Multi-dimensional indexing"):
        # GH#30588 多维索引已弃用
        idx[:, None]


# 使用参数化测试函数，验证对于不合法的输入数据（多维数组），是否会正确引发 ValueError 异常
@pytest.mark.parametrize("dtype", [None, np.int64, np.uint64, np.float64])
def test_validate_1d_input(dtype):
    # GH#27125 检查输入数据是否为一维
    msg = "Index data must be 1-dimensional"

    # 创建一个二维数组 arr，分别使用 Index 和 DataFrame 对象尝试创建索引，预期会引发 ValueError 异常
    arr = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError, match=msg):
        Index(arr, dtype=dtype)

    df = DataFrame(arr.reshape(4, 2))
    with pytest.raises(ValueError, match=msg):
        Index(df, dtype=dtype)

    # GH#13601 禁止将多维数组赋值给索引
    ser = Series(0, range(4))
    with pytest.raises(ValueError, match=msg):
        ser.index = np.array([[2, 3]] * 4, dtype=dtype)


# 使用参数化测试函数，验证在给定不同类别和额外参数的情况下，从 memoryview 创建索引对象的行为是否正确
@pytest.mark.parametrize(
    "klass, extra_kwargs",
    [
        [Index, {}],
        *[[lambda x: Index(x, dtype=dtyp), {}] for dtyp in tm.ALL_REAL_NUMPY_DTYPES],
        [DatetimeIndex, {}],
        [TimedeltaIndex, {}],
        [PeriodIndex, {"freq": "Y"}],
    ],
)
def test_construct_from_memoryview(klass, extra_kwargs):
    # GH 13120
    # 使用 memoryview(np.arange(2000, 2005)) 创建索引对象，并比较预期结果以确保精确匹配
    result = klass(memoryview(np.arange(2000, 2005)), **extra_kwargs)
    expected = klass(list(range(2000, 2005)), **extra_kwargs)
    tm.assert_index_equal(result, expected, exact=True)


# 使用参数化测试函数，验证在给定运算符下，处理 NaN 比较相同对象的行为是否正确
@pytest.mark.parametrize("op", [operator.lt, operator.gt])
def test_nan_comparison_same_object(op):
    # GH#47105
    # 创建包含 NaN 的 Index 对象，并验证使用给定运算符进行比较的结果是否与预期一致
    idx = Index([np.nan])
    expected = np.array([False])

    result = op(idx, idx)
    tm.assert_numpy_array_equal(result, expected)

    result = op(idx, idx.copy())
    tm.assert_numpy_array_equal(result, expected)


# 使用 skip_if_no 装饰器定义一个测试函数，用于验证处理 pyarrow 类型列表的索引行为
@td.skip_if_no("pyarrow")
def test_is_monotonic_pyarrow_list_type():
    # GH 57333
    # 导入 pyarrow 库，创建具有指定 ArrowDtype 的 Index 对象，并验证其不是单调递增或递减的
    import pyarrow as pa

    idx = Index([[1], [2, 3]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    assert not idx.is_monotonic_increasing
    assert not idx.is_monotonic_decreasing
```
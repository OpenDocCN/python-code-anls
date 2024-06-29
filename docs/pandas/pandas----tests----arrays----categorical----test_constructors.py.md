# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_constructors.py`

```
    # 导入所需模块和类
    from datetime import (
        date,
        datetime,
    )

    import numpy as np  # 导入 NumPy 库
    import pytest  # 导入 Pytest 测试框架

    from pandas._config import using_pyarrow_string_dtype  # 导入 Pandas 内部配置模块

    from pandas.core.dtypes.common import (
        is_float_dtype,
        is_integer_dtype,
    )  # 导入 Pandas 核心数据类型模块中的函数

    from pandas.core.dtypes.dtypes import CategoricalDtype  # 导入 Pandas 分类数据类型

    import pandas as pd  # 导入 Pandas 库
    from pandas import (  # 从 Pandas 库中导入多个类和函数
        Categorical,
        CategoricalIndex,
        DatetimeIndex,
        Index,
        Interval,
        IntervalIndex,
        MultiIndex,
        NaT,
        RangeIndex,
        Series,
        Timestamp,
        date_range,
        period_range,
        timedelta_range,
    )
    import pandas._testing as tm  # 导入 Pandas 测试模块


class TestCategoricalConstructors:
    def test_categorical_from_cat_and_dtype_str_preserve_ordered(self):
        # GH#49309 we should preserve orderedness in `res`
        cat = Categorical([3, 1], categories=[3, 2, 1], ordered=True)
        # 创建一个有序的分类变量 `cat`

        res = Categorical(cat, dtype="category")
        # 使用现有的分类变量 `cat` 创建一个新的分类变量 `res`
        assert res.dtype.ordered  # 检查新分类变量 `res` 是否保持有序

    def test_categorical_disallows_scalar(self):
        # GH#38433
        with pytest.raises(TypeError, match="Categorical input must be list-like"):
            # 禁止使用标量作为输入，应该抛出 TypeError 异常
            Categorical("A", categories=["A", "B"])

    def test_categorical_1d_only(self):
        # ndim > 1
        msg = "> 1 ndim Categorical are not supported at this time"
        # 当输入数据的维度大于 1 时，抛出 NotImplementedError 异常
        with pytest.raises(NotImplementedError, match=msg):
            Categorical(np.array([list("abcd")]))

    def test_validate_ordered(self):
        # see gh-14058
        exp_msg = "'ordered' must either be 'True' or 'False'"
        exp_err = TypeError

        # This should be a boolean.
        ordered = np.array([0, 1, 2])
        # ordered 应该是一个布尔类型的值，但这里传入了一个数组

        with pytest.raises(exp_err, match=exp_msg):
            # 应该抛出 TypeError 异常，错误信息为 "'ordered' must either be 'True' or 'False'"
            Categorical([1, 2, 3], ordered=ordered)

        with pytest.raises(exp_err, match=exp_msg):
            # 应该抛出 TypeError 异常，错误信息为 "'ordered' must either be 'True' or 'False'"
            Categorical.from_codes(
                [0, 0, 1], categories=["a", "b", "c"], ordered=ordered
            )

    def test_constructor_empty(self):
        # GH 17248
        c = Categorical([])
        # 创建一个空的分类变量 `c`
        expected = Index([])
        # 预期结果是一个空的索引对象

        tm.assert_index_equal(c.categories, expected)
        # 使用 Pandas 测试模块中的函数检查 `c` 的分类列表与预期结果是否相等

        c = Categorical([], categories=[1, 2, 3])
        # 创建一个空的分类变量 `c`，指定其分类为 [1, 2, 3]
        expected = Index([1, 2, 3], dtype=np.int64)
        # 预期结果是一个包含 [1, 2, 3] 的整数索引对象

        tm.assert_index_equal(c.categories, expected)
        # 使用 Pandas 测试模块中的函数检查 `c` 的分类列表与预期结果是否相等

    def test_constructor_empty_boolean(self):
        # see gh-22702
        cat = Categorical([], categories=[True, False])
        # 创建一个空的分类变量 `cat`，指定其分类为 [True, False]
        categories = sorted(cat.categories.tolist())
        # 获取分类列表，并排序

        assert categories == [False, True]
        # 检查分类列表是否与预期结果 [False, True] 相同

    def test_constructor_tuples(self):
        values = np.array([(1,), (1, 2), (1,), (1, 2)], dtype=object)
        # 创建一个包含元组的 NumPy 数组 `values`

        result = Categorical(values)
        # 使用 `values` 创建一个新的分类变量 `result`

        expected = Index([(1,), (1, 2)], tupleize_cols=False)
        # 预期结果是一个索引对象，包含元组 [(1,), (1, 2)]

        tm.assert_index_equal(result.categories, expected)
        # 使用 Pandas 测试模块中的函数检查 `result` 的分类列表与预期结果是否相等

        assert result.ordered is False
        # 检查新创建的分类变量 `result` 是否是无序的
    def test_constructor_tuples_datetimes(self):
        # 测试构造函数处理元组和日期时间类型

        # 创建包含元组和日期时间的 numpy 数组
        values = np.array(
            [
                (Timestamp("2010-01-01"),),
                (Timestamp("2010-01-02"),),
                (Timestamp("2010-01-01"),),
                (Timestamp("2010-01-02"),),
                ("a", "b"),
            ],
            dtype=object,
        )[:-1]
        
        # 使用 Categorical 类构造结果
        result = Categorical(values)
        
        # 创建预期的 Index 对象，包含日期时间元组，不使用元组列化
        expected = Index(
            [(Timestamp("2010-01-01"),), (Timestamp("2010-01-02"),)],
            tupleize_cols=False,
        )
        
        # 断言结果的 categories 与预期相等
        tm.assert_index_equal(result.categories, expected)

    def test_constructor_unsortable(self):
        # 测试构造函数处理无法排序的情况

        # 创建包含整数和当前日期时间的 numpy 数组
        arr = np.array([1, 2, 3, datetime.now()], dtype="O")
        
        # 使用 Categorical 类构造 factor 对象，指定 ordered=False
        factor = Categorical(arr, ordered=False)
        
        # 断言 factor 对象的 ordered 属性为 False
        assert not factor.ordered

        # 测试当数组包含无法排序的元素时是否会引发 TypeError 异常
        msg = (
            "'values' is not ordered, please explicitly specify the "
            "categories order by passing in a categories argument."
        )
        with pytest.raises(TypeError, match=msg):
            Categorical(arr, ordered=True)

    def test_constructor_interval(self):
        # 测试构造函数处理区间数据

        # 使用 Categorical 类构造区间对象的结果
        result = Categorical(
            [Interval(1, 2), Interval(2, 3), Interval(3, 6)], ordered=True
        )
        
        # 创建预期的 IntervalIndex 对象
        ii = IntervalIndex([Interval(1, 2), Interval(2, 3), Interval(3, 6)])
        exp = Categorical(ii, ordered=True)
        
        # 断言结果与预期的区间对象相等
        tm.assert_categorical_equal(result, exp)
        tm.assert_index_equal(result.categories, ii)

    def test_constructor_with_existing_categories(self):
        # 测试构造函数使用现有的分类数据

        # GH25318: 用 pd.Series 构造可能会跳过重新编码分类的问题

        # 创建两个不同的 Categorical 对象
        c0 = Categorical(["a", "b", "c", "a"])
        c1 = Categorical(["a", "b", "c", "a"], categories=["b", "c"])

        # 使用第一个对象的数据和第二个对象的分类构造新的 Categorical 对象
        c2 = Categorical(c0, categories=c1.categories)
        
        # 断言 c1 和 c2 的内容相等
        tm.assert_categorical_equal(c1, c2)

        # 使用 Series 对象和 c1 的分类构造新的 Categorical 对象
        c3 = Categorical(Series(c0), categories=c1.categories)
        
        # 断言 c1 和 c3 的内容相等
        tm.assert_categorical_equal(c1, c3)

    def test_constructor_not_sequence(self):
        # 测试构造函数处理非序列情况

        # 断言当 categories 参数不是类列表时会引发 TypeError 异常
        msg = r"^Parameter 'categories' must be list-like, was"
        with pytest.raises(TypeError, match=msg):
            Categorical(["a", "b"], categories="a")

    def test_constructor_with_null(self):
        # 测试构造函数处理空值情况

        # 不能在分类数据中包含 NaN
        msg = "Categorical categories cannot be null"
        
        # 断言当分类数据中包含 NaN 时会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical([np.nan, "a", "b", "c"], categories=[np.nan, "a", "b", "c"])

        with pytest.raises(ValueError, match=msg):
            Categorical([None, "a", "b", "c"], categories=[None, "a", "b", "c"])

        # 断言当分类数据中包含 NaT 时会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical(
                DatetimeIndex(["nat", "20160101"]),
                categories=[NaT, Timestamp("20160101")],
            )
    # 测试使用指定索引列表和类别列表构造 CategoricalIndex 对象
    def test_constructor_with_index(self):
        # 创建一个 CategoricalIndex 对象 ci，传入索引列表和类别列表
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
        # 使用测试工具检查 ci.values 是否与 Categorical(ci) 相等
        tm.assert_categorical_equal(ci.values, Categorical(ci))

        # 再次创建一个 CategoricalIndex 对象 ci，传入索引列表和类别列表
        ci = CategoricalIndex(list("aabbca"), categories=list("cab"))
        # 使用测试工具检查 ci.values 是否与转换类型后的 Categorical(ci) 相等
        tm.assert_categorical_equal(
            ci.values, Categorical(ci.astype(object), categories=ci.categories)
        )

    # 测试使用生成器构造 Categorical 对象
    def test_constructor_with_generator(self):
        # 在 isna(single_val).any() 报错，因为 isna 返回单个值而不是生成器
        # 创建一个期望的 Categorical 对象
        exp = Categorical([0, 1, 2])
        # 使用生成器创建一个 Categorical 对象 cat
        cat = Categorical(x for x in [0, 1, 2])
        # 使用测试工具检查 cat 是否与 exp 相等
        tm.assert_categorical_equal(cat, exp)
        # 使用范围创建一个 Categorical 对象 cat
        cat = Categorical(range(3))
        # 使用测试工具检查 cat 是否与 exp 相等
        tm.assert_categorical_equal(cat, exp)

        # 使用 MultiIndex.from_product 创建一个 MultiIndex 对象

        # 检查类别是否接受生成器和序列
        cat = Categorical([0, 1, 2], categories=(x for x in [0, 1, 2]))
        tm.assert_categorical_equal(cat, exp)
        cat = Categorical([0, 1, 2], categories=range(3))
        tm.assert_categorical_equal(cat, exp)

    # 测试使用 RangeIndex 构造 Categorical 对象
    def test_constructor_with_rangeindex(self):
        # RangeIndex 在类别中保持不变
        rng = Index(range(3))

        # 使用 RangeIndex 创建一个 Categorical 对象 cat
        cat = Categorical(rng)
        # 使用测试工具检查 cat 的类别是否与 rng 完全相等
        tm.assert_index_equal(cat.categories, rng, exact=True)

        # 使用 RangeIndex 和指定顺序创建一个 Categorical 对象 cat
        cat = Categorical([1, 2, 0], categories=rng)
        # 使用测试工具检查 cat 的类别是否与 rng 完全相等
        tm.assert_index_equal(cat.categories, rng, exact=True)

    # 使用 pytest 参数化测试多种日期时间类型的构造函数
    @pytest.mark.parametrize(
        "dtl",
        [
            date_range("1995-01-01 00:00:00", periods=5, freq="s"),
            date_range("1995-01-01 00:00:00", periods=5, freq="s", tz="US/Eastern"),
            timedelta_range("1 day", periods=5, freq="s"),
        ],
    )
    def test_constructor_with_datetimelike(self, dtl):
        # 见 issue gh-12077
        # 使用日期时间类型和 NaT 构造对象

        # 创建一个 Series 对象 s
        s = Series(dtl)
        # 使用 Series s 创建一个 Categorical 对象 c
        c = Categorical(s)

        # 创建一个期望的日期时间类型对象 expected
        expected = type(dtl)(s)
        expected._data.freq = None

        # 使用测试工具检查 c 的类别是否与 expected 完全相等
        tm.assert_index_equal(c.categories, expected)
        # 使用测试工具检查 c 的编码是否与预期的 np.arange(5, dtype="int8") 完全相等
        tm.assert_numpy_array_equal(c.codes, np.arange(5, dtype="int8"))

        # 在包含 NaT 的情况下进行测试
        s2 = s.copy()
        s2.iloc[-1] = NaT
        c = Categorical(s2)

        # 创建一个剔除 NaN 值后的期望对象 expected
        expected = type(dtl)(s2.dropna())
        expected._data.freq = None

        # 使用测试工具检查 c 的类别是否与 expected 完全相等
        tm.assert_index_equal(c.categories, expected)

        # 创建一个预期的 np.array([0, 1, 2, 3, -1], dtype=np.int8)
        exp = np.array([0, 1, 2, 3, -1], dtype=np.int8)
        # 使用测试工具检查 c 的编码是否与 exp 完全相等
        tm.assert_numpy_array_equal(c.codes, exp)

        # 创建 c 的字符串表示形式并检查其中是否包含 "NaT"
        result = repr(c)
        assert "NaT" in result

    # 测试从 Index、Series、带时区的日期时间类型构造 Categorical 对象
    def test_constructor_from_index_series_datetimetz(self):
        # 创建一个带时区的日期时间索引 idx
        idx = date_range("2015-01-01 10:00", freq="D", periods=3, tz="US/Eastern")
        # 在结果类别中不保留频率信息
        idx = idx._with_freq(None)
        # 使用日期时间索引创建一个 Categorical 对象 result
        result = Categorical(idx)
        # 使用测试工具检查 result 的类别是否与 idx 完全相等
        tm.assert_index_equal(result.categories, idx)

        # 使用 Series 对象创建一个 Categorical 对象 result
        result = Categorical(Series(idx))
        # 使用测试工具检查 result 的类别是否与 idx 完全相等
        tm.assert_index_equal(result.categories, idx)
    def test_constructor_date_objects(self):
        # 创建一个今天的日期对象 v
        v = date.today()

        # 创建一个包含两个相同日期对象的分类变量 cat
        cat = Categorical([v, v])
        # 断言分类变量的 categories 属性的数据类型是 object
        assert cat.categories.dtype == object
        # 断言分类变量的第一个元素的类型是 date
        assert type(cat.categories[0]) is date

    def test_constructor_from_index_series_timedelta(self):
        # 创建一个时间增量的索引对象 idx，包含 3 个时间增量
        idx = timedelta_range("1 days", freq="D", periods=3)
        # 修改 idx 的频率信息为 None，结果中不会保留频率信息
        idx = idx._with_freq(None)
        # 使用时间增量 idx 创建一个分类变量 result
        result = Categorical(idx)
        # 断言 result 的 categories 属性与 idx 相等
        tm.assert_index_equal(result.categories, idx)

        # 使用时间增量 idx 创建一个 Series 对象，再将其转换为分类变量 result
        result = Categorical(Series(idx))
        # 断言 result 的 categories 属性与 idx 相等
        tm.assert_index_equal(result.categories, idx)

    def test_constructor_from_index_series_period(self):
        # 创建一个日期周期的索引对象 idx，从 "2015-01-01" 开始，频率为每天，包含 3 个周期
        idx = period_range("2015-01-01", freq="D", periods=3)
        # 使用日期周期 idx 创建一个分类变量 result
        result = Categorical(idx)
        # 断言 result 的 categories 属性与 idx 相等
        tm.assert_index_equal(result.categories, idx)

        # 使用日期周期 idx 创建一个 Series 对象，再将其转换为分类变量 result
        result = Categorical(Series(idx))
        # 断言 result 的 categories 属性与 idx 相等
        tm.assert_index_equal(result.categories, idx)

    @pytest.mark.parametrize(
        "values",
        [
            np.array([1.0, 1.2, 1.8, np.nan]),
            np.array([1, 2, 3], dtype="int64"),
            ["a", "b", "c", np.nan],
            [pd.Period("2014-01"), pd.Period("2014-02"), NaT],
            [Timestamp("2014-01-01"), Timestamp("2014-01-02"), NaT],
            [
                Timestamp("2014-01-01", tz="US/Eastern"),
                Timestamp("2014-01-02", tz="US/Eastern"),
                NaT,
            ],
        ],
    )
    def test_constructor_invariant(self, values):
        # GH 14190
        # 创建一个包含给定值 values 的分类变量 c
        c = Categorical(values)
        # 使用 c 创建另一个分类变量 c2
        c2 = Categorical(c)
        # 断言 c 和 c2 在分类变量意义上相等
        tm.assert_categorical_equal(c, c2)

    @pytest.mark.parametrize("ordered", [True, False])
    def test_constructor_with_dtype(self, ordered):
        # 定义分类变量的分类列表
        categories = ["b", "a", "c"]
        # 创建一个指定分类类型的 dtype 对象
        dtype = CategoricalDtype(categories, ordered=ordered)
        # 使用指定 dtype 创建一个分类变量 result
        result = Categorical(["a", "b", "a", "c"], dtype=dtype)
        # 创建一个期望的分类变量 expected，与 result 对比
        expected = Categorical(
            ["a", "b", "a", "c"], categories=categories, ordered=ordered
        )
        # 断言 result 和 expected 在分类变量意义上相等
        tm.assert_categorical_equal(result, expected)
        # 断言 result 的 ordered 属性与输入的 ordered 参数相等
        assert result.ordered is ordered

    def test_constructor_dtype_and_others_raises(self):
        # 创建一个指定 dtype 的分类类型对象
        dtype = CategoricalDtype(["a", "b"], ordered=True)
        # 定义错误消息
        msg = "Cannot specify `categories` or `ordered` together with `dtype`."

        # 断言在指定 dtype 的情况下，使用 categories 参数会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], categories=["a", "b"], dtype=dtype)

        # 断言在指定 dtype 的情况下，使用 ordered=True 参数会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], ordered=True, dtype=dtype)

        # 断言在指定 dtype 的情况下，使用 ordered=False 参数会引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical(["a", "b"], ordered=False, dtype=dtype)

    @pytest.mark.parametrize("categories", [None, ["a", "b"], ["a", "c"]])
    # 定义一个测试函数，用于测试 Categorical 类的构造函数，接受字符串作为类别和有序性参数
    def test_constructor_str_category(self, categories, ordered):
        # 使用给定的参数创建一个 Categorical 对象，dtype 设置为 'category'
        result = Categorical(
            ["a", "b"], categories=categories, ordered=ordered, dtype="category"
        )
        # 创建一个期望的 Categorical 对象，不包含 dtype 参数
        expected = Categorical(["a", "b"], categories=categories, ordered=ordered)
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_categorical_equal(result, expected)

    # 定义一个测试函数，测试当传入未知的 dtype 参数时是否会引发 ValueError 异常
    def test_constructor_str_unknown(self):
        # 使用 pytest 的上下文管理器检查是否会抛出 ValueError 异常，且异常信息包含 "Unknown dtype"
        with pytest.raises(ValueError, match="Unknown dtype"):
            # 尝试创建一个 Categorical 对象，传入一个未知的 dtype 参数 'foo'
            Categorical([1, 2], dtype="foo")

    # 标记为预期失败的测试函数，测试处理 NumPy 字符串类型时的情况
    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="Can't be NumPy strings")
    def test_constructor_np_strs(self):
        # GH#31499 Hashtable.map_locations 需要能处理 np.str_ 对象
        # 创建一个包含 np.str_ 对象的 Categorical 对象
        cat = Categorical(["1", "0", "1"], [np.str_("0"), np.str_("1")])
        # 断言 cat 中的所有类别是否都是 np.str_ 类型
        assert all(isinstance(x, np.str_) for x in cat.categories)

    # 测试从另一个 Categorical 对象创建新对象，同时指定 dtype 参数
    def test_constructor_from_categorical_with_dtype(self):
        # 创建一个指定了类别和有序性的 CategoricalDtype 对象
        dtype = CategoricalDtype(["a", "b", "c"], ordered=True)
        # 创建一个普通的 Categorical 对象作为输入值
        values = Categorical(["a", "b", "d"])
        # 使用指定的 dtype 参数创建新的 Categorical 对象
        result = Categorical(values, dtype=dtype)
        # 创建一个期望的 Categorical 对象，使用 dtype 中的类别，而非 values 中的类别
        expected = Categorical(
            ["a", "b", "d"], categories=["a", "b", "c"], ordered=True
        )
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_categorical_equal(result, expected)

    # 测试从另一个 Categorical 对象创建新对象，但指定了未知的 dtype 参数
    def test_constructor_from_categorical_with_unknown_dtype(self):
        # 创建一个未指定类别但有序性为 True 的 CategoricalDtype 对象
        dtype = CategoricalDtype(None, ordered=True)
        # 创建一个普通的 Categorical 对象作为输入值
        values = Categorical(["a", "b", "d"])
        # 使用未知的 dtype 参数创建新的 Categorical 对象
        result = Categorical(values, dtype=dtype)
        # 创建一个期望的 Categorical 对象，使用 values 中的类别，因为 dtype 是 None
        expected = Categorical(
            ["a", "b", "d"], categories=["a", "b", "d"], ordered=True
        )
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_categorical_equal(result, expected)

    # 测试从另一个 Categorical 对象创建新对象，同时指定类别和有序性参数
    def test_constructor_from_categorical_string(self):
        # 创建一个普通的 Categorical 对象作为输入值
        values = Categorical(["a", "b", "d"])
        # 使用指定的类别和有序性参数创建新的 Categorical 对象，dtype 设置为 'category'
        result = Categorical(
            values, categories=["a", "b", "c"], ordered=True, dtype="category"
        )
        # 创建一个期望的 Categorical 对象，与 result 共享相同的类别和有序性参数
        expected = Categorical(
            ["a", "b", "d"], categories=["a", "b", "c"], ordered=True
        )
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_categorical_equal(result, expected)

        # 创建一个没有指定 dtype 参数的新的 Categorical 对象，使用相同的类别和有序性参数
        result = Categorical(values, categories=["a", "b", "c"], ordered=True)
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_categorical_equal(result, expected)

    # 测试从另一个 Categorical 对象创建新对象，输入的类别参数也可以是一个 Categorical 对象
    def test_constructor_with_categorical_categories(self):
        # 创建一个期望的 Categorical 对象，包含更多的类别，用于覆盖输入值中的类别
        expected = Categorical(["a", "b"], categories=["a", "b", "c"])

        # 从另一个 Categorical 对象创建新的 Categorical 对象，类别由 Categorical 对象指定
        result = Categorical(["a", "b"], categories=Categorical(["a", "b", "c"]))
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_categorical_equal(result, expected)

        # 从另一个 CategoricalIndex 对象创建新的 Categorical 对象，类别由 CategoricalIndex 对象指定
        result = Categorical(["a", "b"], categories=CategoricalIndex(["a", "b", "c"]))
        # 使用测试框架检查 result 是否等于 expected
        tm.assert_categorical_equal(result, expected)

    # 使用 pytest 参数化标记，测试接受不同类型的 klass 参数的函数
    @pytest.mark.parametrize("klass", [lambda x: np.array(x, dtype=object), list])
    def test_construction_with_null(self, klass, nulls_fixture):
        # 根据 GitHub issue 31927 进行测试构建，使用给定的类和空值数据
        values = klass(["a", nulls_fixture, "b"])
        # 使用构建出的值创建分类数据对象
        result = Categorical(values)

        # 定义分类数据类型，指定分类的类别
        dtype = CategoricalDtype(["a", "b"])
        # 定义分类的编码
        codes = [0, -1, 1]
        # 使用给定的编码和类型创建预期的分类对象
        expected = Categorical.from_codes(codes=codes, dtype=dtype)

        # 使用测试工具函数验证结果与预期是否相等
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("validate", [True, False])
    def test_from_codes_nullable_int_categories(self, any_numeric_ea_dtype, validate):
        # GitHub issue 39649 的测试用例
        # 创建包含数字范围的分类数据数组
        cats = pd.array(range(5), dtype=any_numeric_ea_dtype)
        # 随机生成一组编码
        codes = np.random.default_rng(2).integers(5, size=3)
        # 定义分类数据类型
        dtype = CategoricalDtype(cats)
        # 使用给定的编码、类型和验证选项创建分类对象
        arr = Categorical.from_codes(codes, dtype=dtype, validate=validate)
        # 验证分类的类别数据类型与原始分类数组是否相同
        assert arr.categories.dtype == cats.dtype
        # 使用测试工具函数验证分类的类别与原始分类数组是否相等
        tm.assert_index_equal(arr.categories, Index(cats))

    def test_from_codes_empty(self):
        # 使用空的编码列表和给定的类别创建分类对象
        cat = ["a", "b", "c"]
        result = Categorical.from_codes([], categories=cat)
        expected = Categorical([], categories=cat)

        # 使用测试工具函数验证结果与预期是否相等
        tm.assert_categorical_equal(result, expected)

    @pytest.mark.parametrize("validate", [True, False])
    def test_from_codes_validate(self, validate):
        # GitHub issue 53122 的测试用例
        # 定义分类数据类型
        dtype = CategoricalDtype(["a", "b"])
        if validate:
            # 如果启用验证，测试应抛出值错误异常
            with pytest.raises(ValueError, match="codes need to be between "):
                Categorical.from_codes([4, 5], dtype=dtype, validate=validate)
        else:
            # 如果未启用验证，创建分类对象，虽然包含不正确的编码，但是用户需要负责
            Categorical.from_codes([4, 5], dtype=dtype, validate=validate)

    def test_from_codes_too_few_categories(self):
        # 测试使用过少的类别创建分类对象时是否会抛出值错误异常
        dtype = CategoricalDtype(categories=[1, 2])
        msg = "codes need to be between "
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([1, 2], categories=dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([1, 2], dtype=dtype)

    def test_from_codes_non_int_codes(self):
        # 测试使用非整数编码创建分类对象时是否会抛出值错误异常
        dtype = CategoricalDtype(categories=[1, 2])
        msg = "codes need to be array-like integers"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(["a"], categories=dtype.categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(["a"], dtype=dtype)

    def test_from_codes_non_unique_categories(self):
        # 测试使用非唯一类别创建分类对象时是否会抛出值错误异常
        with pytest.raises(ValueError, match="Categorical categories must be unique"):
            Categorical.from_codes([0, 1, 2], categories=["a", "a", "b"])

    def test_from_codes_nan_cat_included(self):
        # 测试包含空值类别时是否会抛出值错误异常
        with pytest.raises(ValueError, match="Categorical categories cannot be null"):
            Categorical.from_codes([0, 1, 2], categories=["a", "b", np.nan])
    # 定义测试方法，验证当传入过小的索引时是否引发异常
    def test_from_codes_too_negative(self):
        # 创建一个分类数据类型，包含类别 "a", "b", "c"
        dtype = CategoricalDtype(categories=["a", "b", "c"])
        # 错误信息，指示索引必须在 -1 和 len(categories)-1 之间
        msg = r"codes need to be between -1 and len\(categories\)-1"
        # 验证从给定的 codes 创建 Categorical 对象时是否引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([-2, 1, 2], categories=dtype.categories)
        # 验证从给定的 codes 创建 Categorical 对象时是否引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([-2, 1, 2], dtype=dtype)

    # 定义测试方法，验证从 codes 创建 Categorical 对象的正常情况
    def test_from_codes(self):
        # 创建一个分类数据类型，包含类别 "a", "b", "c"
        dtype = CategoricalDtype(categories=["a", "b", "c"])
        # 创建预期的 Categorical 对象，非有序
        exp = Categorical(["a", "b", "c"], ordered=False)
        # 使用给定的 codes 创建 Categorical 对象，并验证其与预期对象是否相等
        res = Categorical.from_codes([0, 1, 2], categories=dtype.categories)
        tm.assert_categorical_equal(exp, res)

        # 使用给定的 codes 创建 Categorical 对象，并验证其与预期对象是否相等
        res = Categorical.from_codes([0, 1, 2], dtype=dtype)
        tm.assert_categorical_equal(exp, res)

    # 使用参数化测试，验证从 codes 创建 Categorical 对象时带有分类类别的情况
    @pytest.mark.parametrize("klass", [Categorical, CategoricalIndex])
    def test_from_codes_with_categorical_categories(self, klass):
        # GH17884
        # 创建预期的 Categorical 对象，指定了类别为 ["a", "b", "c"]
        expected = Categorical(["a", "b"], categories=["a", "b", "c"])

        # 使用给定的 codes 和分类对象创建 Categorical 对象，并验证其与预期对象是否相等
        result = Categorical.from_codes([0, 1], categories=klass(["a", "b", "c"]))
        tm.assert_categorical_equal(result, expected)

    # 使用参数化测试，验证从 codes 创建 Categorical 对象时带有非唯一分类类别的情况
    @pytest.mark.parametrize("klass", [Categorical, CategoricalIndex])
    def test_from_codes_with_non_unique_categorical_categories(self, klass):
        # 验证当类别不唯一时是否引发 ValueError 异常
        with pytest.raises(ValueError, match="Categorical categories must be unique"):
            Categorical.from_codes([0, 1], klass(["a", "b", "a"]))

    # 验证从 codes 创建 Categorical 对象时，codes 中包含 NaN 值时的异常处理
    def test_from_codes_with_nan_code(self):
        # GH21767
        # codes 包含 NaN 值的情况
        codes = [1, 2, np.nan]
        # 创建一个分类数据类型，包含类别 "a", "b", "c"
        dtype = CategoricalDtype(categories=["a", "b", "c"])
        # 验证当 codes 包含非数组整数时是否引发 ValueError 异常
        with pytest.raises(ValueError, match="codes need to be array-like integers"):
            Categorical.from_codes(codes, categories=dtype.categories)
        # 验证当 codes 包含非数组整数时是否引发 ValueError 异常
        with pytest.raises(ValueError, match="codes need to be array-like integers"):
            Categorical.from_codes(codes, dtype=dtype)

    # 使用参数化测试，验证从 codes 创建 Categorical 对象时，codes 包含浮点数的情况
    @pytest.mark.parametrize("codes", [[1.0, 2.0, 0], [1.1, 2.0, 0]])
    def test_from_codes_with_float(self, codes):
        # GH21767
        # 创建一个分类数据类型，包含类别 "a", "b", "c"
        dtype = CategoricalDtype(categories=["a", "b", "c"])

        # 验证当 codes 包含浮点数时是否引发 ValueError 异常
        msg = "codes need to be array-like integers"
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes, dtype.categories)
        # 验证当 codes 包含浮点数时是否引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes, dtype=dtype)

    # 验证从 codes 创建 Categorical 对象时，指定了 dtype 参数时的异常处理
    def test_from_codes_with_dtype_raises(self):
        # 错误信息，指示不允许同时指定 dtype 参数
        msg = "Cannot specify"
        # 验证当同时指定 categories 和 dtype 参数时是否引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(
                [0, 1], categories=["a", "b"], dtype=CategoricalDtype(["a", "b"])
            )

        # 验证当同时指定 ordered 和 dtype 参数时是否引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(
                [0, 1], ordered=True, dtype=CategoricalDtype(["a", "b"])
            )
    # 测试从指定的代码创建分类数据，当两个参数都为 None 时应该引发 ValueError 异常
    def test_from_codes_neither(self):
        msg = "Both were None"
        # 使用 pytest 的断言检查是否引发了 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes([0, 1])

    # 测试从指定的可空整数代码和类别创建分类数据
    def test_from_codes_with_nullable_int(self):
        # 创建包含可空整数的 Pandas 数组
        codes = pd.array([0, 1], dtype="Int64")
        # 定义类别列表
        categories = ["a", "b"]

        # 调用被测试的函数，创建分类数据
        result = Categorical.from_codes(codes, categories=categories)
        # 创建预期的分类数据，将可空整数转换为普通的 NumPy 整数数组
        expected = Categorical.from_codes(codes.to_numpy(int), categories=categories)

        # 使用测试框架的函数检查结果与预期是否相等
        tm.assert_categorical_equal(result, expected)

    # 测试从包含可空整数和 NA 值的代码创建分类数据时是否引发 ValueError 异常
    def test_from_codes_with_nullable_int_na_raises(self):
        # 创建包含可空整数和 NA 值的 Pandas 数组
        codes = pd.array([0, None], dtype="Int64")
        # 定义类别列表
        categories = ["a", "b"]

        # 定义错误消息
        msg = "codes cannot contain NA values"
        # 使用 pytest 的断言检查是否引发 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes, categories=categories)

    # 使用 pytest 的参数化功能测试从推断类别创建分类数据
    @pytest.mark.parametrize("dtype", [None, "category"])
    def test_from_inferred_categories(self, dtype):
        # 定义类别列表
        cats = ["a", "b"]
        # 定义整数代码数组
        codes = np.array([0, 0, 1, 1], dtype="i8")
        # 调用被测试的函数，根据推断的类别创建分类数据
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        # 创建预期的分类数据，根据指定的类别和代码创建
        expected = Categorical.from_codes(codes, cats)
        # 使用测试框架的函数检查结果与预期是否相等
        tm.assert_categorical_equal(result, expected)

    # 使用 pytest 的参数化功能测试从推断类别创建分类数据，并测试排序功能
    @pytest.mark.parametrize("dtype", [None, "category"])
    def test_from_inferred_categories_sorts(self, dtype):
        # 定义类别列表，按预期非排序顺序排列
        cats = ["b", "a"]
        # 定义整数代码数组
        codes = np.array([0, 1, 1, 1], dtype="i8")
        # 调用被测试的函数，根据推断的类别创建分类数据
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        # 创建预期的分类数据，强制排序指定类别和代码创建
        expected = Categorical.from_codes([1, 0, 0, 0], ["a", "b"])
        # 使用测试框架的函数检查结果与预期是否相等
        tm.assert_categorical_equal(result, expected)

    # 测试从推断类别创建分类数据时指定 dtype
    def test_from_inferred_categories_dtype(self):
        # 定义类别列表
        cats = ["a", "b", "d"]
        # 定义整数代码数组
        codes = np.array([0, 1, 0, 2], dtype="i8")
        # 定义分类数据类型
        dtype = CategoricalDtype(["c", "b", "a"], ordered=True)
        # 调用被测试的函数，根据推断的类别和指定的 dtype 创建分类数据
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        # 创建预期的分类数据，指定类别、分类数据类型和排序
        expected = Categorical(
            ["a", "b", "a", "d"], categories=["c", "b", "a"], ordered=True
        )
        # 使用测试框架的函数检查结果与预期是否相等
        tm.assert_categorical_equal(result, expected)

    # 测试从推断类别创建分类数据时强制转换为指定类型
    def test_from_inferred_categories_coerces(self):
        # 定义类别列表，包含一个无效的条目
        cats = ["1", "2", "bad"]
        # 定义整数代码数组
        codes = np.array([0, 0, 1, 2], dtype="i8")
        # 定义分类数据类型
        dtype = CategoricalDtype([1, 2])
        # 调用被测试的函数，根据推断的类别和指定的 dtype 创建分类数据
        result = Categorical._from_inferred_categories(cats, codes, dtype)
        # 创建预期的分类数据，将无效的条目转换为 NaN
        expected = Categorical([1, 1, 2, np.nan])
        # 使用测试框架的函数检查结果与预期是否相等
        tm.assert_categorical_equal(result, expected)

    # 测试使用指定的 ordered 参数构造分类数据
    def test_construction_with_ordered(self, ordered):
        # 创建分类数据，测试 GH 9347, 9190
        cat = Categorical([0, 1, 2], ordered=ordered)
        # 使用断言检查分类数据的 ordered 属性是否与预期一致
        assert cat.ordered == bool(ordered)

    # 测试使用复数构造分类数据，预期会将复数部分忽略
    def test_constructor_imaginary(self):
        # 定义值列表，包含一个复数
        values = [1, 2, 3 + 1j]
        # 创建分类数据，忽略复数部分
        c1 = Categorical(values)
        # 使用测试框架的函数检查分类数据的类别是否与值列表相同
        tm.assert_index_equal(c1.categories, Index(values))
        # 使用测试框架的函数检查分类数据的数据部分是否与值列表的实部相同
        tm.assert_numpy_array_equal(np.array(c1), np.array(values))
    def test_constructor_string_and_tuples(self):
        # GH 21416
        # 创建一个 Categorical 对象，使用包含字符串和元组的数组作为参数
        c = Categorical(np.array(["c", ("a", "b"), ("b", "a"), "c"], dtype=object))
        # 期望的索引对象，包含元组和字符串
        expected_index = Index([("a", "b"), ("b", "a"), "c"])
        # 断言分类对象的 categories 属性与期望的索引对象相等
        assert c.categories.equals(expected_index)

    def test_interval(self):
        # 生成一个时间间隔范围的索引对象
        idx = pd.interval_range(0, 10, periods=10)
        # 使用时间间隔索引创建 Categorical 对象，指定 categories 为 idx
        cat = Categorical(idx, categories=idx)
        # 期望的代码数组，是从0到9的整数
        expected_codes = np.arange(10, dtype="int8")
        # 断言分类对象的 codes 属性与期望的代码数组相等
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        # 断言分类对象的 categories 属性与 idx 相等
        tm.assert_index_equal(cat.categories, idx)

        # 推断 categories
        cat = Categorical(idx)
        # 断言分类对象的 codes 属性与期望的代码数组相等
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        # 断言分类对象的 categories 属性与 idx 相等
        tm.assert_index_equal(cat.categories, idx)

        # 使用列表值
        cat = Categorical(list(idx))
        # 断言分类对象的 codes 属性与期望的代码数组相等
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        # 断言分类对象的 categories 属性与 idx 相等
        tm.assert_index_equal(cat.categories, idx)

        # 使用列表值和 categories
        cat = Categorical(list(idx), categories=list(idx))
        # 断言分类对象的 codes 属性与期望的代码数组相等
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        # 断言分类对象的 categories 属性与 idx 相等
        tm.assert_index_equal(cat.categories, idx)

        # 混合顺序
        values = idx.take([1, 2, 0])
        # 创建 Categorical 对象，使用 values 和指定 categories 为 idx
        cat = Categorical(values, categories=idx)
        # 断言分类对象的 codes 属性与期望的代码数组相等
        tm.assert_numpy_array_equal(cat.codes, np.array([1, 2, 0], dtype="int8"))
        # 断言分类对象的 categories 属性与 idx 相等
        tm.assert_index_equal(cat.categories, idx)

        # 额外情况
        values = pd.interval_range(8, 11, periods=3)
        # 创建 Categorical 对象，使用 values 和指定 categories 为 idx
        cat = Categorical(values, categories=idx)
        # 期望的代码数组，包含 [8, 9, -1]
        expected_codes = np.array([8, 9, -1], dtype="int8")
        # 断言分类对象的 codes 属性与期望的代码数组相等
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        # 断言分类对象的 categories 属性与 idx 相等
        tm.assert_index_equal(cat.categories, idx)

        # 重叠情况
        idx = IntervalIndex([Interval(0, 2), Interval(0, 1)])
        # 创建 Categorical 对象，使用 idx 和指定 categories 为 idx
        cat = Categorical(idx, categories=idx)
        # 期望的代码数组，包含 [0, 1]
        expected_codes = np.array([0, 1], dtype="int8")
        # 断言分类对象的 codes 属性与期望的代码数组相等
        tm.assert_numpy_array_equal(cat.codes, expected_codes)
        # 断言分类对象的 categories 属性与 idx 相等
        tm.assert_index_equal(cat.categories, idx)

    def test_categorical_extension_array_nullable(self, nulls_fixture):
        # GH:
        # 创建一个包含空值的 StringArray 对象
        arr = pd.arrays.StringArray._from_sequence(
            [nulls_fixture] * 2, dtype=pd.StringDtype()
        )
        # 使用 StringArray 创建 Categorical 对象
        result = Categorical(arr)
        # 断言 arr 的 dtype 与 result 的 categories 的 dtype 相等
        assert arr.dtype == result.categories.dtype
        # 期望的 Categorical 对象，包含 pd.NA 值
        expected = Categorical(Series([pd.NA, pd.NA], dtype=arr.dtype))
        # 断言 result 与期望的 Categorical 对象相等
        tm.assert_categorical_equal(result, expected)

    def test_from_sequence_copy(self):
        # 创建一个 Categorical 对象，使用重复的数组作为参数
        cat = Categorical(np.arange(5).repeat(2))
        # 使用 _from_sequence 方法创建 Categorical 对象，不进行拷贝
        result = Categorical._from_sequence(cat, dtype=cat.dtype, copy=False)

        # 更一般地说，我们希望得到一个视图
        # 断言 result 的 _codes 属性与 cat 的 _codes 属性是同一个对象
        assert result._codes is cat._codes

        # 使用 _from_sequence 方法创建 Categorical 对象，进行拷贝
        result = Categorical._from_sequence(cat, dtype=cat.dtype, copy=True)

        # 断言 result 与 cat 不共享内存
        assert not tm.shares_memory(result, cat)

    def test_constructor_datetime64_non_nano(self):
        # 创建一个包含 datetime64[D] 类型的数组作为 categories
        categories = np.arange(10).view("M8[D]")
        # 创建一个 values 数组，从 categories 中每隔一个取值复制而来
        values = categories[::2].copy()

        # 创建 Categorical 对象，指定 values 和 categories
        cat = Categorical(values, categories=categories)
        # 断言 Categorical 对象中所有值与 values 相等
        assert (cat == values).all()
    # 定义测试方法，验证构造函数在保留频率时的行为
    def test_constructor_preserves_freq(self):
        # GH33830：在分类数据中保留频率信息
        # 创建一个包含5个时间点的日期范围
        dti = date_range("2016-01-01", periods=5)

        # 期望的频率是日期时间索引对象的频率
        expected = dti.freq

        # 使用日期时间索引对象构造一个分类数据对象
        cat = Categorical(dti)
        # 获取分类数据对象中的类别，并获取其频率信息
        result = cat.categories.freq

        # 断言期望的频率与实际结果相等
        assert expected == result

    # 使用参数化测试标记，对多组输入值进行测试
    @pytest.mark.parametrize(
        "values, categories",
        [
            [range(5), None],        # 测试：值为0到4的范围，类别为None
            [range(4), range(5)],    # 测试：值为0到3的范围，类别为0到4的范围
            [[0, 1, 2, 3], range(5)],  # 测试：明确指定值，类别为0到4的范围
            [[], range(5)],          # 测试：空值列表，类别为0到4的范围
        ],
    )
    # 定义测试方法，验证范围值保留RangeIndex类别时的行为
    def test_range_values_preserves_rangeindex_categories(self, values, categories):
        # 创建分类数据对象，并获取其类别
        result = Categorical(values=values, categories=categories).categories
        # 期望的结果是一个0到4的范围索引对象
        expected = RangeIndex(range(5))
        # 使用断言验证实际结果与期望结果相等
        tm.assert_index_equal(result, expected, exact=True)
```
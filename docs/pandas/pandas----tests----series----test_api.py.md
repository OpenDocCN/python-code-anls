# `D:\src\scipysrc\pandas\pandas\tests\series\test_api.py`

```
# 导入inspect和pydoc模块，用于检查和文档化Python对象
import inspect
import pydoc

# 导入numpy库，并用np作为别名
import numpy as np
# 导入pytest库，用于编写和运行测试
import pytest

# 导入pandas库，并从中导入DataFrame、Index、Series等类及相关函数
import pandas as pd
from pandas import (
    DataFrame,
    Index,
    Series,
    date_range,
    period_range,
    timedelta_range,
)
# 导入pandas内部测试工具模块
import pandas._testing as tm

# 定义测试类TestSeriesMisc，用于测试Series对象的各种功能
class TestSeriesMisc:
    # 定义测试方法test_tab_completion，用于测试Series对象的tab补全功能
    def test_tab_completion(self):
        # GH 9910
        # 创建一个包含字符列表的Series对象s
        s = Series(list("abcd"))
        # 断言Series对象s应包含'str'属性，但不应包含'dt'和'cat'属性
        assert "str" in dir(s)
        assert "dt" not in dir(s)
        assert "cat" not in dir(s)

    # 定义测试方法test_tab_completion_dt，用于测试包含日期的Series对象的tab补全功能
    def test_tab_completion_dt(self):
        # 创建一个包含日期范围的Series对象s
        s = Series(date_range("1/1/2015", periods=5))
        # 断言Series对象s应包含'dt'属性，但不应包含'str'和'cat'属性
        assert "dt" in dir(s)
        assert "str" not in dir(s)
        assert "cat" not in dir(s)

    # 定义测试方法test_tab_completion_cat，用于测试包含分类数据的Series对象的tab补全功能
    def test_tab_completion_cat(self):
        # 创建一个包含分类数据的Series对象s，分类数据类型为字符串
        s = Series(list("abbcd"), dtype="category")
        # 断言Series对象s应包含'cat'和'str'属性，但不应包含'dt'属性
        assert "cat" in dir(s)
        assert "str" in dir(s)  # 因为它是一个字符串分类数据
        assert "dt" not in dir(s)

    # 定义测试方法test_tab_completion_cat_str，用于测试将日期转换为分类数据后的Series对象的tab补全功能
    def test_tab_completion_cat_str(self):
        # 创建一个包含日期范围的Series对象s，并将其转换为分类数据类型
        s = Series(date_range("1/1/2015", periods=5)).astype("category")
        # 断言Series对象s应包含'cat'和'dt'属性，但不应包含'str'属性
        assert "cat" in dir(s)
        assert "str" not in dir(s)
        assert "dt" in dir(s)  # 因为它是一个日期时间分类数据

    # 定义测试方法test_tab_completion_with_categorical，用于测试分类数据的tab补全显示功能
    def test_tab_completion_with_categorical(self):
        # 期望的分类数据属性列表
        ok_for_cat = [
            "categories",
            "codes",
            "ordered",
            "set_categories",
            "add_categories",
            "remove_categories",
            "rename_categories",
            "reorder_categories",
            "remove_unused_categories",
            "as_ordered",
            "as_unordered",
        ]

        # 创建一个包含字符列表的Series对象s，并将其转换为分类数据类型
        s = Series(list("aabbcde")).astype("category")
        # 获取Series对象s的所有非私有属性，并排序后与期望的分类属性列表进行比较
        results = sorted({r for r in s.cat.__dir__() if not r.startswith("_")})
        tm.assert_almost_equal(results, sorted(set(ok_for_cat)))

    # 使用pytest的参数化装饰器，定义多个不同的索引类型
    @pytest.mark.parametrize(
        "index",
        [
            Index(list("ab") * 5, dtype="category"),
            Index([str(i) for i in range(10)]),
            Index(["foo", "bar", "baz"] * 2),
            date_range("2020-01-01", periods=10),
            period_range("2020-01-01", periods=10, freq="D"),
            timedelta_range("1 day", periods=10),
            Index(np.arange(10), dtype=np.uint64),
            Index(np.arange(10), dtype=np.int64),
            Index(np.arange(10), dtype=np.float64),
            Index([True, False]),
            Index([f"a{i}" for i in range(101)]),
            pd.MultiIndex.from_tuples(zip("ABCD", "EFGH")),
            pd.MultiIndex.from_tuples(zip([0, 1, 2, 3], "EFGH")),
        ],
    )
    # 定义一个测试方法，用于测试索引的自动补全功能
    def test_index_tab_completion(self, index):
        # 创建一个 Series 对象，其中的索引类型为对象类型
        s = Series(index=index, dtype=object)
        # 获得 Series 对象的 dir() 结果，返回包含字符串类型值的列表
        dir_s = dir(s)
        # 遍历 Series 对象的索引的唯一值（第一级别），并且对每个值进行断言测试
        for i, x in enumerate(s.index.unique(level=0)):
            # 如果索引小于100，断言条件不满足，则抛出 AssertionError
            if i < 100:
                assert not isinstance(x, str) or not x.isidentifier() or x in dir_s
            # 如果索引大于等于100，断言索引不在 dir_s 中
            else:
                assert x not in dir_s

    # 使用 pytest 的参数化装饰器，定义一个测试方法，测试非可哈希类型的 Series 对象
    @pytest.mark.parametrize("ser", [Series(dtype=object), Series([1])])
    def test_not_hashable(self, ser):
        # 期望的错误消息
        msg = "unhashable type: 'Series'"
        # 使用 pytest 的 assertRaises 断言，期望捕获 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            hash(ser)

    # 测试 Series 对象的包含性
    def test_contains(self, datetime_series):
        # 使用 tm 模块中的 assert_contains_all 方法，断言 datetime_series 的索引包含在 datetime_series 中
        tm.assert_contains_all(datetime_series.index, datetime_series)

    # 测试 axis 别名的使用
    def test_axis_alias(self):
        # 创建一个包含 NaN 值的 Series 对象
        s = Series([1, 2, np.nan])
        # 使用 tm 模块中的 assert_series_equal 方法，断言两次 dropna 操作结果相等
        tm.assert_series_equal(s.dropna(axis="rows"), s.dropna(axis="index"))
        # 断言 dropna 后的结果求和为 3，axis 参数为 "rows"
        assert s.dropna().sum(axis="rows") == 3
        # 断言 _get_axis_number 方法返回 "rows" 的轴编号为 0
        assert s._get_axis_number("rows") == 0
        # 断言 _get_axis_name 方法返回 "rows" 的轴名称为 "index"
        assert s._get_axis_name("rows") == "index"

    # 测试 class_axis 方法
    def test_class_axis(self):
        # 断言调用 pydoc 模块的 getdoc 方法，返回 Series.index 的文档
        assert pydoc.getdoc(Series.index)

    # 测试与 ndarray 兼容性的方法
    def test_ndarray_compat(self):
        # 创建一个 DataFrame 对象 tsdf，包含随机正态分布的数据
        tsdf = DataFrame(
            np.random.default_rng(2).standard_normal((1000, 3)),
            columns=["A", "B", "C"],
            index=date_range("1/1/2000", periods=1000),
        )

        # 定义一个函数 f，返回每列的最大值所在行的数据
        def f(x):
            return x[x.idxmax()]

        # 对 tsdf 应用函数 f，得到结果 result
        result = tsdf.apply(f)
        # 计算 tsdf 的每列的最大值，作为期望的结果 expected
        expected = tsdf.max()
        # 使用 tm 模块中的 assert_series_equal 方法，断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)

    # 测试与 ndarray 类似函数的方法
    def test_ndarray_compat_like_func(self):
        # 使用类似 ndarray 的方法创建一个随机数 Series 对象 s
        s = Series(np.random.default_rng(2).standard_normal(10))
        # 创建一个与 s 类型相同但值为 1 的 Series 对象 result
        result = Series(np.ones_like(s))
        # 创建一个期望的结果 expected，其值全为 1
        expected = Series(1, index=range(10), dtype="float64")
        # 使用 tm 模块中的 assert_series_equal 方法，断言 result 与 expected 相等
        tm.assert_series_equal(result, expected)

    # 测试空方法的情况
    def test_empty_method(self):
        # 创建一个空的 Series 对象 s_empty
        s_empty = Series(dtype=object)
        # 断言 s_empty 是否为空
        assert s_empty.empty

    # 使用 pytest 的参数化装饰器，测试整个 Series 对象非空的情况
    @pytest.mark.parametrize("dtype", ["int64", object])
    def test_empty_method_full_series(self, dtype):
        # 创建一个包含元素的 Series 对象 full_series
        full_series = Series(index=[1], dtype=dtype)
        # 断言 full_series 是否非空
        assert not full_series.empty

    # 使用 pytest 的参数化装饰器，测试整数类型 Series 对象的 size
    @pytest.mark.parametrize("dtype", [None, "Int64"])
    def test_integer_series_size(self, dtype):
        # 创建一个整数类型的 Series 对象 s，包含 0 到 8 共 9 个元素
        # GH 25580
        s = Series(range(9), dtype=dtype)
        # 断言 s 的 size 是否为 9
        assert s.size == 9

    # 测试 attrs 属性的使用
    def test_attrs(self):
        # 创建一个名为 "abc" 的 Series 对象 s
        s = Series([0, 1], name="abc")
        # 断言 s 的 attrs 属性为空字典
        assert s.attrs == {}
        # 向 s 的 attrs 属性添加键 "version"，值为 1
        s.attrs["version"] = 1
        # 对 s 执行加法操作，返回结果 result
        result = s + 1
        # 断言 result 的 attrs 属性为 {"version": 1}
        assert result.attrs == {"version": 1}

    # 测试 inspect 模块中的 getmembers 方法
    def test_inspect_getmembers(self):
        # 要求安装 jinja2 库，否则跳过该测试
        pytest.importorskip("jinja2")
        # 创建一个对象类型的 Series 对象 ser
        ser = Series(dtype=object)
        # 使用 inspect 模块中的 getmembers 方法，返回 ser 对象的成员列表
        inspect.getmembers(ser)
    def test_unknown_attribute(self):
        # 测试函数：test_unknown_attribute
        # GH#9680：GitHub 上的 issue 编号
        # 创建时间范围对象 tdi，从0开始，包含10个时间点，频率为每秒一次
        tdi = timedelta_range(start=0, periods=10, freq="1s")
        # 创建一个 Series 对象 ser，包含10个随机正态分布的数值，索引为 tdi
        ser = Series(np.random.default_rng(2).normal(size=10), index=tdi)
        # 断言确保 "foo" 不在 ser 对象的 __dict__ 属性中
        assert "foo" not in ser.__dict__
        # 准备异常消息
        msg = "'Series' object has no attribute 'foo'"
        # 使用 pytest 来检查是否抛出 AttributeError 异常，并匹配特定消息
        with pytest.raises(AttributeError, match=msg):
            ser.foo

    @pytest.mark.parametrize("op", ["year", "day", "second", "weekday"])
    def test_datetime_series_no_datelike_attrs(self, op, datetime_series):
        # 测试函数：test_datetime_series_no_datelike_attrs
        # GH#7206：GitHub 上的 issue 编号
        # 准备异常消息，指出 "Series" 对象没有属性 op 所指定的属性名
        msg = f"'Series' object has no attribute '{op}'"
        # 使用 pytest 来检查是否抛出 AttributeError 异常，并匹配特定消息
        with pytest.raises(AttributeError, match=msg):
            # 使用 getattr 函数尝试获取 datetime_series 对象的 op 属性
            getattr(datetime_series, op)

    def test_series_datetimelike_attribute_access(self):
        # 测试函数：test_series_datetimelike_attribute_access
        # 这里只是简单的断言，确保可以正常访问 ser 对象的各个时间相关属性
        # attribute access should still work!
        ser = Series({"year": 2000, "month": 1, "day": 10})
        assert ser.year == 2000
        assert ser.month == 1
        assert ser.day == 10

    def test_series_datetimelike_attribute_access_invalid(self):
        # 测试函数：test_series_datetimelike_attribute_access_invalid
        ser = Series({"year": 2000, "month": 1, "day": 10})
        # 准备异常消息，指出 "Series" 对象没有属性 'weekday'
        msg = "'Series' object has no attribute 'weekday'"
        # 使用 pytest 来检查是否抛出 AttributeError 异常，并匹配特定消息
        with pytest.raises(AttributeError, match=msg):
            # 尝试访问 ser 对象的 weekday 属性，应该抛出异常
            ser.weekday

    @pytest.mark.parametrize(
        "kernel, has_numeric_only",
        [
            # 下面是一系列测试参数化的测试用例
            ("skew", True),
            ("var", True),
            ("all", False),
            ("prod", True),
            ("any", False),
            ("idxmin", False),
            ("quantile", False),
            ("idxmax", False),
            ("min", True),
            ("sem", True),
            ("mean", True),
            ("nunique", False),
            ("max", True),
            ("sum", True),
            ("count", False),
            ("median", True),
            ("std", True),
            ("rank", True),
            ("pct_change", False),
            ("cummax", False),
            ("shift", False),
            ("diff", False),
            ("cumsum", False),
            ("cummin", False),
            ("cumprod", False),
            ("fillna", False),
            ("ffill", False),
            ("bfill", False),
            ("sample", False),
            ("tail", False),
            ("take", False),
            ("head", False),
            ("cov", False),
            ("corr", False),
        ],
    )
    @pytest.mark.parametrize("dtype", [bool, int, float, object])
    # 参数化的测试用例，分别测试不同的操作和数据类型
    # 定义一个测试方法，用于测试在指定内核和条件下的数值处理功能
    def test_numeric_only(self, kernel, has_numeric_only, dtype):
        # 创建一个包含数值的 Series 对象
        ser = Series([0, 1, 1], dtype=dtype)
        
        # 根据内核类型设置参数 args，以便后续调用方法使用
        if kernel == "corrwith":
            args = (ser,)
        elif kernel == "corr":
            args = (ser,)
        elif kernel == "cov":
            args = (ser,)
        elif kernel == "nth":
            args = (0,)
        elif kernel == "fillna":
            args = (True,)
        elif kernel == "fillna":
            args = ("ffill",)
        elif kernel == "take":
            args = ([0],)
        elif kernel == "quantile":
            args = (0.5,)
        else:
            args = ()
        
        # 获取 Series 对象中对应内核的方法
        method = getattr(ser, kernel)
        
        # 如果没有 numeric_only 参数，则预期会引发 TypeError 异常
        if not has_numeric_only:
            msg = (
                "(got an unexpected keyword argument 'numeric_only'"
                "|too many arguments passed in)"
            )
            with pytest.raises(TypeError, match=msg):
                method(*args, numeric_only=True)
        
        # 如果 dtype 是 object 类型，则 numeric_only=True 不允许，预期会引发 TypeError 异常
        elif dtype is object:
            msg = f"Series.{kernel} does not allow numeric_only=True with non-numeric"
            with pytest.raises(TypeError, match=msg):
                method(*args, numeric_only=True)
        
        # 否则，以 numeric_only=True 调用方法，并与预期结果比较
        else:
            # 调用方法获取 numeric_only=True 的结果
            result = method(*args, numeric_only=True)
            # 调用方法获取 numeric_only=False 的预期结果
            expected = method(*args, numeric_only=False)
            
            # 如果预期结果是 Series 类型，则使用 assert_series_equal 进行比较
            if isinstance(expected, Series):
                # 用于比较两个 Series 对象是否相等
                tm.assert_series_equal(result, expected)
            else:
                # 如果预期结果是简单的数值，则直接比较结果值
                assert result == expected
```
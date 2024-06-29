# `D:\src\scipysrc\pandas\pandas\tests\series\test_formats.py`

```
    from datetime import (
        datetime,        # 导入 datetime 类，用于处理日期和时间
        timedelta,       # 导入 timedelta 类，用于处理时间间隔
    )

    import numpy as np   # 导入 NumPy 库，用于数值计算
    import pytest         # 导入 pytest 库，用于编写和运行测试用例

    from pandas._config import using_pyarrow_string_dtype   # 导入 pandas 内部模块，用于配置字符串数据类型

    import pandas as pd   # 导入 pandas 库，用于数据处理和分析
    from pandas import (  # 导入 pandas 中的多个类和函数
        Categorical,      # 导入 Categorical 类，用于处理分类数据
        DataFrame,        # 导入 DataFrame 类，用于表示二维数据表格
        Index,            # 导入 Index 类，用于表示索引对象
        Series,           # 导入 Series 类，用于表示一维数据结构
        date_range,       # 导入 date_range 函数，用于生成日期范围
        option_context,   # 导入 option_context 函数，用于设置上下文选项
        period_range,     # 导入 period_range 函数，用于生成周期范围
        timedelta_range,  # 导入 timedelta_range 函数，用于生成时间间隔范围
    )

class TestSeriesRepr:
    def test_multilevel_name_print_0(self):
        # 测试多级索引的名称打印，None 不会被打印出来，但是 0 会
        # （与 DataFrame 和扁平索引行为相匹配）
        mi = pd.MultiIndex.from_product([range(2, 3), range(3, 4)], names=[0, None])
        ser = Series(1.5, index=mi)

        res = repr(ser)
        expected = "0   \n2  3    1.5\ndtype: float64"
        assert res == expected

    def test_multilevel_name_print(self, lexsorted_two_level_string_multiindex):
        # 测试多级索引的名称打印
        index = lexsorted_two_level_string_multiindex
        ser = Series(range(len(index)), index=index, name="sth")
        expected = [
            "first  second",
            "foo    one       0",
            "       two       1",
            "       three     2",
            "bar    one       3",
            "       two       4",
            "baz    two       5",
            "       three     6",
            "qux    one       7",
            "       two       8",
            "       three     9",
            "Name: sth, dtype: int64",
        ]
        expected = "\n".join(expected)
        assert repr(ser) == expected

    def test_small_name_printing(self):
        # 测试小型 Series 的名称打印
        s = Series([0, 1, 2])

        s.name = "test"
        assert "Name: test" in repr(s)

        s.name = None
        assert "Name:" not in repr(s)

    def test_big_name_printing(self):
        # 测试大型 Series 的名称打印（不同的代码路径）
        s = Series(range(1000))

        s.name = "test"
        assert "Name: test" in repr(s)

        s.name = None
        assert "Name:" not in repr(s)

    def test_empty_name_printing(self):
        # 测试空名称的 Series 打印
        s = Series(index=date_range("20010101", "20020101"), name="test", dtype=object)
        assert "Name: test" in repr(s)

    @pytest.mark.parametrize("args", [(), (0, -1)])
    def test_float_range(self, args):
        str(
            Series(
                np.random.default_rng(2).standard_normal(1000),
                index=np.arange(1000, *args),
            )
        )

    def test_empty_object(self):
        # 测试空对象的 Series 打印
        str(Series(dtype=object))

    def test_string(self, string_series):
        str(string_series)
        str(string_series.astype(int))

        # 测试包含 NaN 值的 Series 打印
        string_series[5:7] = np.nan
        str(string_series)

    def test_object(self, object_series):
        # 测试对象类型的 Series 打印
        str(object_series)

    def test_datetime(self, datetime_series):
        str(datetime_series)
        # 测试包含 None 值的 datetime Series 打印
        ots = datetime_series.astype("O")
        ots[::2] = None
        repr(ots)
    @pytest.mark.parametrize(
        "name",
        [
            "",  # 空字符串情况
            1,  # 整数类型情况
            1.2,  # 浮点数类型情况
            "foo",  # 字符串类型情况
            "\u03b1\u03b2\u03b3",  # Unicode字符串情况
            "loooooooooooooooooooooooooooooooooooooooooooooooooooong",  # 长字符串情况
            ("foo", "bar", "baz"),  # 元组类型情况
            (1, 2),  # 元组类型情况
            ("foo", 1, 2.3),  # 混合类型元组情况
            ("\u03b1", "\u03b2", "\u03b3"),  # Unicode字符串元组情况
            ("\u03b1", "bar"),  # 混合类型元组情况
        ],
    )
    def test_various_names(self, name, string_series):
        # 测试不同类型的名称参数
        string_series.name = name
        repr(string_series)

    def test_tuple_name(self):
        # 测试元组作为名称参数的情况
        biggie = Series(
            np.random.default_rng(2).standard_normal(1000),
            index=np.arange(1000),
            name=("foo", "bar", "baz"),
        )
        repr(biggie)

    @pytest.mark.parametrize("arg", [100, 1001])
    def test_tidy_repr_name_0(self, arg):
        # 测试名称为数字 0 的情况
        ser = Series(np.random.default_rng(2).standard_normal(arg), name=0)
        rep_str = repr(ser)
        assert "Name: 0" in rep_str

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="TODO: investigate why this is failing"
    )
    def test_newline(self):
        # 测试包含换行符的情况
        ser = Series(["a\n\r\tb"], name="a\n\r\td", index=["a\n\r\tf"])
        assert "\t" not in repr(ser)
        assert "\r" not in repr(ser)
        assert "a\n" not in repr(ser)

    @pytest.mark.parametrize(
        "name, expected",
        [
            ["foo", "Series([], Name: foo, dtype: int64)"],  # 测试空 Series 的情况
            [None, "Series([], dtype: int64)"],  # 测试空 Series 的情况（无名称）
        ],
    )
    def test_empty_int64(self, name, expected):
        # 测试空 Series 的情况
        s = Series([], dtype=np.int64, name=name)
        assert repr(s) == expected

    def test_repr_bool_fails(self, capsys):
        s = Series(
            [
                DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
                for i in range(5)
            ]
        )

        # 测试包含 DataFrame 的 Series 情况
        repr(s)

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_repr_name_iterable_indexable(self):
        s = Series([1, 2, 3], name=np.int64(3))

        # 测试可迭代和可索引名称的情况
        repr(s)

        s.name = ("\u05d0",) * 2
        repr(s)

    def test_repr_max_rows(self):
        # 测试设置最大行数时的情况
        with option_context("display.max_rows", None):
            str(Series(range(1001)))  # 应不会引发异常

    def test_unicode_string_with_unicode(self):
        df = Series(["\u05d0"], name="\u05d1")
        str(df)

        ser = Series(["\u03c3"] * 10)
        repr(ser)

        ser2 = Series(["\u05d0"] * 1000)
        ser2.name = "title1"
        repr(ser2)

    def test_str_to_bytes_raises(self):
        # 测试将字符串转换为字节时引发异常的情况
        df = Series(["abc"], name="abc")
        msg = "^'str' object cannot be interpreted as an integer$"
        with pytest.raises(TypeError, match=msg):
            bytes(df)
    # 定义一个测试函数，用于测试时间序列对象的字符串表示（repr），数据类型为object
    def test_timeseries_repr_object_dtype(self):
        # 创建一个时间索引，包含从2000年1月1日开始的1000天的日期，数据类型为object
        index = Index(
            [datetime(2000, 1, 1) + timedelta(i) for i in range(1000)], dtype=object
        )
        # 使用随机生成的标准正态分布数据创建时间序列对象
        ts = Series(np.random.default_rng(2).standard_normal(len(index)), index)
        # 调用repr函数打印时间序列对象的字符串表示
        repr(ts)

        # 创建一个时间序列对象，包含20个浮点数，索引从"2020-01-01"开始，20个时间点
        ts = Series(
            np.arange(20, dtype=np.float64), index=date_range("2020-01-01", periods=20)
        )
        # 断言时间序列对象的字符串表示的最后一行以"Freq:"开头
        assert repr(ts).splitlines()[-1].startswith("Freq:")

        # 从ts中随机选择400个位置的子序列对象
        ts2 = ts.iloc[np.random.default_rng(2).integers(0, len(ts) - 1, 400)]
        # 调用repr函数打印ts2的字符串表示，忽略结果的最后一行
        repr(ts2).splitlines()[-1]

    # 定义一个测试函数，用于测试Latex字符串表示
    def test_latex_repr(self):
        # 检查是否成功导入"jinja2"库，否则跳过测试（因为使用了Styler实现）
        pytest.importorskip("jinja2")
        # 定义一个Latex字符串模板，用于生成表格
        result = r"""\begin{tabular}{ll}
    def test_index_repr_in_frame_with_nan(self):
        # 测试带有NaN值的索引在DataFrame中的表示
        # 创建一个包含NaN的索引对象
        i = Index([1, np.nan])
        # 创建一个Series对象，使用上述索引
        s = Series([1, 2], index=i)
        # 期望的输出字符串，展示了带NaN值的索引和对应的数据
        exp = """1.0    1\nNaN    2\ndtype: int64"""

        # 断言Series对象的repr方法输出符合期望的字符串
        assert repr(s) == exp

    def test_series_repr_nat(self):
        # 测试包含NaT值的Series对象的字符串表示
        series = Series([0, 1000, 2000, pd.NaT._value], dtype="M8[ns]")

        # 获取Series对象的repr字符串
        result = repr(series)
        # 期望的输出字符串，展示了不同类型的时间戳数据
        expected = (
            "0   1970-01-01 00:00:00.000000\n"
            "1   1970-01-01 00:00:00.000001\n"
            "2   1970-01-01 00:00:00.000002\n"
            "3                          NaT\n"
            "dtype: datetime64[ns]"
        )
        # 断言实际输出符合期望的字符串
        assert result == expected

    def test_float_repr(self):
        # 测试将浮点数转换为对象后的字符串表示
        # 创建一个浮点数的Series对象
        ser = Series([1.0]).astype(object)
        # 期望的输出字符串，展示了浮点数的对象表示
        expected = "0    1.0\ndtype: object"
        # 断言Series对象的repr方法输出符合期望的字符串
        assert repr(ser) == expected

    def test_different_null_objects(self):
        # 测试不同空值对象的Series对象的字符串表示
        # 创建一个包含不同类型空值的Series对象
        ser = Series([1, 2, 3, 4], [True, None, np.nan, pd.NaT])
        # 获取Series对象的repr字符串
        result = repr(ser)
        # 期望的输出字符串，展示了不同空值类型的索引和对应的数据
        expected = "True    1\nNone    2\nNaN     3\nNaT     4\ndtype: int64"
        # 断言实际输出符合期望的字符串
        assert result == expected
    def test_categorical_repr(self, using_infer_string):
        # 创建一个包含整数类别的 Series 对象
        a = Series(Categorical([1, 2, 3, 4]))
        # 期望的字符串表示，包括索引和值
        exp = (
            "0    1\n1    2\n2    3\n3    4\n"
            "dtype: category\nCategories (4, int64): [1, 2, 3, 4]"
        )

        # 断言期望的字符串表示与实际的字符串表示相同
        assert exp == a.__str__()

        # 创建一个包含字符串类别的 Series 对象
        a = Series(Categorical(["a", "b"] * 25))
        if using_infer_string:
            # 如果使用推断字符串类型，则期望的字符串表示
            exp = (
                "0     a\n1     b\n"
                "     ..\n"
                "48    a\n49    b\n"
                "Length: 50, dtype: category\nCategories (2, string): [a, b]"
            )
        else:
            # 否则，期望的字符串表示为对象类型
            exp = (
                "0     a\n1     b\n"
                "     ..\n"
                "48    a\n49    b\n"
                "Length: 50, dtype: category\nCategories (2, object): ['a', 'b']"
            )
        # 在显示设置的上下文中，断言期望的字符串表示与实际的 repr 结果相同
        with option_context("display.max_rows", 5):
            assert exp == repr(a)

        # 创建一个具有指定类别和排序的 Series 对象
        levs = list("abcdefghijklmnopqrstuvwxyz")
        a = Series(Categorical(["a", "b"], categories=levs, ordered=True))
        if using_infer_string:
            # 如果使用推断字符串类型，则期望的字符串表示
            exp = (
                "0    a\n1    b\n"
                "dtype: category\n"
                "Categories (26, string): [a < b < c < d ... w < x < y < z]"
            )
        else:
            # 否则，期望的字符串表示为对象类型
            exp = (
                "0    a\n1    b\n"
                "dtype: category\n"
                "Categories (26, object): ['a' < 'b' < 'c' < 'd' ... "
                "'w' < 'x' < 'y' < 'z']"
            )
        # 断言期望的字符串表示与实际的字符串表示相同
        assert exp == a.__str__()

    def test_categorical_series_repr(self):
        # 创建一个包含整数类别的 Series 对象
        s = Series(Categorical([1, 2, 3]))
        # 期望的字符串表示，以三个双引号开始
        exp = """0    1
    def test_categorical_series_repr_ordered(self):
        # 创建一个 Series 对象，其中包含有序的分类数据，从列表 [1, 2, 3] 中创建
        s = Series(Categorical([1, 2, 3], ordered=True))
        # 期望的字符串表示，显示索引和相应的分类值
        exp = """0    1
1    2
2    3
dtype: category
Categories (3, int64): [1 < 2 < 3]"""

        # 断言生成的字符串表示与期望的字符串表示相同
        assert repr(s) == exp

        # 创建一个 Series 对象，其中包含有序的分类数据，从 numpy 数组 np.arange(10) 中创建
        s = Series(Categorical(np.arange(10), ordered=True))
        # 期望的字符串表示，显示索引和相应的分类值，其中分类是按照数值大小顺序排列的
        exp = f"""0    0
1    1
2    2
3    3
4    4
5    5
6    6
7    7
8    8
9    9
dtype: category
Categories (10, {np.dtype(int)}): [0 < 1 < 2 < 3 ... 6 < 7 < 8 < 9]"""

        # 断言生成的字符串表示与期望的字符串表示相同
        assert repr(s) == exp

    def test_categorical_series_repr_datetime(self):
        # 创建一个日期时间索引，从 "2011-01-01 09:00" 开始，每小时增加一个时间点，总共 5 个时间点
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        # 创建一个 Series 对象，其中包含日期时间的分类数据
        s = Series(Categorical(idx))
        # 期望的字符串表示，显示索引和相应的分类值，其中分类是按照日期时间的顺序排列的
        exp = """0   2011-01-01 09:00:00
1   2011-01-01 10:00:00
2   2011-01-01 11:00:00
3   2011-01-01 12:00:00
4   2011-01-01 13:00:00
dtype: category
Categories (5, datetime64[ns]): [2011-01-01 09:00:00, 2011-01-01 10:00:00, 2011-01-01 11:00:00,
                                 2011-01-01 12:00:00, 2011-01-01 13:00:00]"""  # noqa: E501

        # 断言生成的字符串表示与期望的字符串表示相同
        assert repr(s) == exp

        # 创建一个带时区的日期时间索引，从 "2011-01-01 09:00" 开始，每小时增加一个时间点，总共 5 个时间点
        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        # 创建一个 Series 对象，其中包含日期时间的分类数据
        s = Series(Categorical(idx))
        # 期望的字符串表示，显示索引和相应的分类值，其中分类是按照日期时间和时区的顺序排列的
        exp = """0   2011-01-01 09:00:00-05:00
1   2011-01-01 10:00:00-05:00
2   2011-01-01 11:00:00-05:00
3   2011-01-01 12:00:00-05:00
4   2011-01-01 13:00:00-05:00
dtype: category
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,
                                             2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        # 断言生成的字符串表示与期望的字符串表示相同
        assert repr(s) == exp

    def test_categorical_series_repr_datetime_ordered(self):
        # 创建一个日期时间索引，从 "2011-01-01 09:00" 开始，每小时增加一个时间点，总共 5 个时间点
        idx = date_range("2011-01-01 09:00", freq="h", periods=5)
        # 创建一个 Series 对象，其中包含有序的日期时间分类数据
        s = Series(Categorical(idx, ordered=True))
        # 期望的字符串表示，显示索引和相应的分类值，其中分类是按照日期时间的顺序排列的
        exp = """0   2011-01-01 09:00:00
1   2011-01-01 10:00:00
2   2011-01-01 11:00:00
3   2011-01-01 12:00:00
4   2011-01-01 13:00:00
dtype: category
Categories (5, datetime64[ns]): [2011-01-01 09:00:00 < 2011-01-01 10:00:00 < 2011-01-01 11:00:00 <
                                 2011-01-01 12:00:00 < 2011-01-01 13:00:00]"""  # noqa: E501

        # 断言生成的字符串表示与期望的字符串表示相同
        assert repr(s) == exp

        # 创建一个带时区的日期时间索引，从 "2011-01-01 09:00" 开始，每小时增加一个时间点，总共 5 个时间点
        idx = date_range("2011-01-01 09:00", freq="h", periods=5, tz="US/Eastern")
        # 创建一个 Series 对象，其中包含有序的日期时间分类数据
        s = Series(Categorical(idx, ordered=True))
        # 期望的字符串表示，显示索引和相应的分类值，其中分类是按照日期时间和时区的顺序排列的
        exp = """0   2011-01-01 09:00:00-05:00
1   2011-01-01 10:00:00-05:00
2   2011-01-01 11:00:00-05:00
3   2011-01-01 12:00:00-05:00
4   2011-01-01 13:00:00-05:00
dtype: category
Categories (5, datetime64[ns, US/Eastern]): [2011-01-01 09:00:00-05:00, 2011-01-01 10:00:00-05:00,
                                             2011-01-01 11:00:00-05:00, 2011-01-01 12:00:00-05:00,
                                             2011-01-01 13:00:00-05:00]"""  # noqa: E501

        # 断言生成的字符串表示与期望的字符串表示相同
        assert repr(s) == exp
    def test_categorical_series_repr_period(self):
        # 创建一个时间周期范围，每小时一个时间点，共5个时间点
        idx = period_range("2011-01-01 09:00", freq="h", periods=5)
        # 创建一个包含分类数据的Series对象，使用时间周期作为分类的值
        s = Series(Categorical(idx))
        # 期望的字符串表示，包括每个时间点的字符串和分类的信息
        exp = """0    2011-01-01 09:00
1    2011-01-01 10:00
2    2011-01-01 11:00
3    2011-01-01 12:00
4    2011-01-01 13:00
dtype: category
Categories (5, period[h]): [2011-01-01 09:00, 2011-01-01 10:00, 2011-01-01 11:00, 2011-01-01 12:00,
                            2011-01-01 13:00]"""  # noqa: E501

        # 断言Series对象的字符串表示与期望的字符串表示相同
        assert repr(s) == exp

        # 创建一个时间周期范围，每月一个时间点，共5个时间点
        idx = period_range("2011-01", freq="M", periods=5)
        # 创建一个包含分类数据的Series对象，使用时间周期作为分类的值
        s = Series(Categorical(idx))
        # 期望的字符串表示，包括每个时间点的字符串和分类的信息
        exp = """0    2011-01
1    2011-02
2    2011-03
3    2011-04
4    2011-05
dtype: category
Categories (5, period[M]): [2011-01, 2011-02, 2011-03, 2011-04, 2011-05]"""

        # 断言Series对象的字符串表示与期望的字符串表示相同
        assert repr(s) == exp
    # 定义一个测试函数，用于测试分类系列的表示与时间差和排序
    def test_categorical_series_repr_timedelta_ordered(self):
        # 创建一个时间差范围，从 "1 days" 开始，包含5个时间点
        idx = timedelta_range("1 days", periods=5)
        # 创建一个分类系列，使用时间差作为分类的值，并指定为有序分类
        s = Series(Categorical(idx, ordered=True))
        # 期望的结果字符串
        exp = """0   1 days
# 断言s的字符串表示与预期结果exp相同
assert repr(s) == exp

# 创建一个时间间隔索引，从 "1 hours" 开始，共10个时间间隔
idx = timedelta_range("1 hours", periods=10)

# 使用时间间隔创建一个序列，并将其转换为有序分类数据
s = Series(Categorical(idx, ordered=True))

# 预期的字符串表示，显示了序列中每个元素的时间间隔
exp = """0   0 days 01:00:00
1   1 days 01:00:00
2   2 days 01:00:00
3   3 days 01:00:00
4   4 days 01:00:00
5   5 days 01:00:00
6   6 days 01:00:00
7   7 days 01:00:00
8   8 days 01:00:00
9   9 days 01:00:00
dtype: category
Categories (10, timedelta64[ns]): [0 days 01:00:00 < 1 days 01:00:00 < 2 days 01:00:00 <
                                   3 days 01:00:00 ... 6 days 01:00:00 < 7 days 01:00:00 <
                                   8 days 01:00:00 < 9 days 01:00:00]"""  # noqa: E501

# 断言序列s的字符串表示与预期结果exp相同
assert repr(s) == exp
```
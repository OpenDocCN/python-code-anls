# `D:\src\scipysrc\pandas\pandas\tests\frame\test_repr.py`

```
    from datetime import (
        datetime,  # 导入 datetime 模块中的 datetime 类
        timedelta,  # 导入 datetime 模块中的 timedelta 类
    )
    from io import StringIO  # 导入 StringIO 类，用于内存中的文本 I/O

    import numpy as np  # 导入 NumPy 库，用于数值计算
    import pytest  # 导入 pytest 库，用于单元测试框架

    from pandas._config import using_pyarrow_string_dtype  # 导入 pandas 内部配置

    from pandas import (  # 导入 pandas 库中多个常用类和函数
        NA,
        Categorical,
        CategoricalIndex,
        DataFrame,
        IntervalIndex,
        MultiIndex,
        NaT,
        PeriodIndex,
        Series,
        Timestamp,
        date_range,
        option_context,
        period_range,
    )
    import pandas._testing as tm  # 导入 pandas 内部测试模块

    class TestDataFrameRepr:
        def test_repr_should_return_str(self):
            # https://docs.python.org/3/reference/datamodel.html#object.__repr__
            # "...The return value must be a string object."
            # 测试 DataFrame 和 Series 的 __repr__() 方法返回类型是否为字符串对象

            data = [8, 5, 3, 5]
            index1 = ["\u03c3", "\u03c4", "\u03c5", "\u03c6"]
            cols = ["\u03c8"]
            df = DataFrame(data, columns=cols, index=index1)
            assert type(df.__repr__()) is str  # noqa: E721

            ser = df[cols[0]]
            assert type(ser.__repr__()) is str  # noqa: E721

        def test_repr_bytes_61_lines(self):
            # GH#12857
            # 测试 DataFrame 对象在不同行数下的 __repr__() 方法

            lets = list("ACDEFGHIJKLMNOP")
            words = np.random.default_rng(2).choice(lets, (1000, 50))
            df = DataFrame(words).astype("U1")
            assert (df.dtypes == object).all()

            # smoke tests; at one point this raised with 61 but not 60
            repr(df)
            repr(df.iloc[:60, :])
            repr(df.iloc[:61, :])

        def test_repr_unicode_level_names(self, frame_or_series):
            # 测试多重索引 MultiIndex 对象的 __repr__() 方法

            index = MultiIndex.from_tuples([(0, 0), (1, 1)], names=["\u0394", "i1"])

            obj = DataFrame(np.random.default_rng(2).standard_normal((2, 4)), index=index)
            obj = tm.get_obj(obj, frame_or_series)
            repr(obj)

        def test_assign_index_sequences(self):
            # GH#2200
            # 测试设置索引后的 DataFrame 对象的 __repr__() 方法

            df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}).set_index(
                ["a", "b"]
            )
            index = list(df.index)
            index[0] = ("faz", "boo")
            df.index = index
            repr(df)

            # this travels an improper code path
            index[0] = ["faz", "boo"]
            df.index = index
            repr(df)

        def test_repr_with_mi_nat(self):
            # 测试包含 NaT 的 MultiIndex 对象的 __repr__() 方法

            df = DataFrame({"X": [1, 2]}, index=[[NaT, Timestamp("20130101")], ["a", "b"]])
            result = repr(df)
            expected = "              X\nNaT        a  1\n2013-01-01 b  2"
            assert result == expected

        def test_repr_with_different_nulls(self):
            # GH45263
            # 测试包含不同空值（True, None, NaN, NaT）的 DataFrame 对象的 __repr__() 方法

            df = DataFrame([1, 2, 3, 4], [True, None, np.nan, NaT])
            result = repr(df)
            expected = """      0
True  1
None  2
NaN   3
NaT   4"""
            assert result == expected

        def test_repr_with_different_nulls_cols(self):
            # GH45263
            # 测试包含不同空值（NaN, None, NaT, True）的 DataFrame 对象的 __repr__() 方法

            d = {np.nan: [1, 2], None: [3, 4], NaT: [6, 7], True: [8, 9]}
            df = DataFrame(data=d)
            result = repr(df)
            expected = """   NaN  None  NaT  True
0    1     3    6     8
1    2     4    7     9"""
            assert result == expected
    # 定义测试函数 test_multiindex_na_repr，用于测试多重索引中的空值表示
    def test_multiindex_na_repr(self):
        # 创建 DataFrame df3，包含长列名和一个具体的数据条目
        df3 = DataFrame(
            {
                "A" * 30: {("A", "A0006000", "nuit"): "A0006000"},
                "B" * 30: {("A", "A0006000", "nuit"): np.nan},
                "C" * 30: {("A", "A0006000", "nuit"): np.nan},
                "D" * 30: {("A", "A0006000", "nuit"): np.nan},
                "E" * 30: {("A", "A0006000", "nuit"): "A"},
                "F" * 30: {("A", "A0006000", "nuit"): np.nan},
            }
        )

        # 使用 set_index 方法创建一个新的索引对象 idf
        idf = df3.set_index(["A" * 30, "C" * 30])
        # 调用 repr 函数打印 idf 的表示形式
        repr(idf)

    # 定义测试函数 test_repr_name_coincide，用于测试 DataFrame 的表示形式中索引名称是否匹配
    def test_repr_name_coincide(self):
        # 创建一个 MultiIndex 对象 index，指定名称为 ["a", "b", "c"] 的索引
        index = MultiIndex.from_tuples(
            [("a", 0, "foo"), ("b", 1, "bar")], names=["a", "b", "c"]
        )

        # 使用 index 作为索引创建 DataFrame df，其中包含一个名为 "value" 的列
        df = DataFrame({"value": [0, 1]}, index=index)

        # 将 df 的字符串表示形式按行分割后存储在 lines 变量中
        lines = repr(df).split("\n")
        # 断言 df 的字符串表示形式的第三行以 "a 0 foo" 开头
        assert lines[2].startswith("a 0 foo")

    # 定义测试函数 test_repr_to_string，用于测试 DataFrame 和其转置的字符串表示形式
    def test_repr_to_string(
        self,
        multiindex_year_month_day_dataframe_random_data,
        multiindex_dataframe_random_data,
    ):
        # 获取测试数据 multiindex_year_month_day_dataframe_random_data 和 multiindex_dataframe_random_data
        ymd = multiindex_year_month_day_dataframe_random_data
        frame = multiindex_dataframe_random_data

        # 分别打印 frame 和 ymd 的字符串表示形式
        repr(frame)
        repr(ymd)
        # 打印 frame 转置后的字符串表示形式
        repr(frame.T)
        # 打印 ymd 转置后的字符串表示形式
        repr(ymd.T)

        # 创建一个 StringIO 对象 buf
        buf = StringIO()
        # 将 frame 的字符串表示形式写入 buf
        frame.to_string(buf=buf)
        # 将 ymd 的字符串表示形式写入 buf
        ymd.to_string(buf=buf)
        # 将 frame 转置后的字符串表示形式写入 buf
        frame.T.to_string(buf=buf)
        # 将 ymd 转置后的字符串表示形式写入 buf
        ymd.T.to_string(buf=buf)

    # 定义测试函数 test_repr_empty，用于测试空 DataFrame 的字符串表示形式
    def test_repr_empty(self):
        # 打印空 DataFrame 的字符串表示形式
        repr(DataFrame())

        # 创建一个带索引的空 DataFrame frame，并打印其字符串表示形式
        frame = DataFrame(index=np.arange(1000))
        repr(frame)

    # 定义测试函数 test_repr_mixed，用于测试混合类型 DataFrame 的字符串表示形式
    def test_repr_mixed(self, float_string_frame):
        # 打印混合类型 DataFrame float_string_frame 的字符串表示形式
        repr(float_string_frame)

    # 使用 pytest.mark.slow 标记的测试函数 test_repr_mixed_big，用于测试大型混合类型 DataFrame 的字符串表示形式
    @pytest.mark.slow
    def test_repr_mixed_big(self):
        # 创建一个大型混合类型 DataFrame biggie，包含两列 "A" 和 "B"
        biggie = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(200),
                "B": [str(i) for i in range(200)],
            },
            index=range(200),
        )
        # 将前 20 行 "A" 和 "B" 列设置为 NaN
        biggie.loc[:20, "A"] = np.nan
        biggie.loc[:20, "B"] = np.nan

        # 打印 biggie 的字符串表示形式
        repr(biggie)

    # 使用 pytest.mark.xfail 和 using_pyarrow_string_dtype 标记的测试函数 test_repr，用于测试特定情况下 DataFrame 的字符串表示形式
    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="/r in")
    def test_repr(self):
        # 创建一个没有索引但有列名的空 DataFrame no_index
        no_index = DataFrame(columns=[0, 1, 3])
        # 打印 no_index 的字符串表示形式
        repr(no_index)

        # 创建一个具有特定索引和列的 DataFrame df
        df = DataFrame(["a\n\r\tb"], columns=["a\n\r\td"], index=["a\n\r\tf"])
        # 断言 df 的字符串表示形式中不包含制表符、回车或换行符
        assert "\t" not in repr(df)
        assert "\r" not in repr(df)
        assert "a\n" not in repr(df)

    # 定义测试函数 test_repr_dimensions，用于测试带有显示维度设置的 DataFrame 的字符串表示形式
    def test_repr_dimensions(self):
        # 创建一个简单的 DataFrame df
        df = DataFrame([[1, 2], [3, 4]])
        # 在 display.show_dimensions 选项设置为 True 的上下文中，断言字符串表示形式包含 "2 rows x 2 columns"
        with option_context("display.show_dimensions", True):
            assert "2 rows x 2 columns" in repr(df)

        # 在 display.show_dimensions 选项设置为 False 的上下文中，断言字符串表示形式不包含 "2 rows x 2 columns"
        with option_context("display.show_dimensions", False):
            assert "2 rows x 2 columns" not in repr(df)

        # 在 display.show_dimensions 选项设置为 "truncate" 的上下文中，断言字符串表示形式不包含 "2 rows x 2 columns"
        with option_context("display.show_dimensions", "truncate"):
            assert "2 rows x 2 columns" not in repr(df)
    def test_repr_big(self):
        # 创建一个200行4列的数据帧，数据初始化为0，列索引为0到3，行索引为0到199
        biggie = DataFrame(np.zeros((200, 4)), columns=range(4), index=range(200))
        # 返回数据帧的字符串表示形式
        repr(biggie)

    def test_repr_unsortable(self):
        # 列不可排序的数据帧示例

        # 创建一个包含不可排序列的数据帧
        unsortable = DataFrame(
            {
                "foo": [1] * 50,
                datetime.today(): [1] * 50,
                "bar": ["bar"] * 50,
                datetime.today() + timedelta(1): ["bar"] * 50,
            },
            index=np.arange(50),
        )
        # 返回数据帧的字符串表示形式
        repr(unsortable)

    def test_repr_float_frame_options(self, float_frame):
        # 返回浮点数数据帧的字符串表示形式
        repr(float_frame)

        # 使用上下文设置，将显示精度设置为3
        with option_context("display.precision", 3):
            # 返回带有设置精度的浮点数数据帧的字符串表示形式
            repr(float_frame)

        # 使用上下文设置，将最大行数设置为10，最大列数设置为2
        with option_context("display.max_rows", 10, "display.max_columns", 2):
            # 返回带有设置最大行数和最大列数的浮点数数据帧的字符串表示形式
            repr(float_frame)

        # 使用上下文设置，将最大行数设置为1000，最大列数设置为1000
        with option_context("display.max_rows", 1000, "display.max_columns", 1000):
            # 返回带有设置最大行数和最大列数的浮点数数据帧的字符串表示形式
            repr(float_frame)

    def test_repr_unicode(self):
        # Unicode 值
        uval = "\u03c3\u03c3\u03c3\u03c3"

        # 创建一个包含 Unicode 列的数据帧
        df = DataFrame({"A": [uval, uval]})

        # 返回数据帧的字符串表示形式，并验证第一行
        result = repr(df)
        ex_top = "      A"
        assert result.split("\n")[0].rstrip() == ex_top

        # 创建一个包含 Unicode 列的数据帧
        df = DataFrame({"A": [uval, uval]})
        # 返回数据帧的字符串表示形式，并验证第一行
        result = repr(df)
        assert result.split("\n")[0].rstrip() == ex_top

    def test_unicode_string_with_unicode(self):
        # 创建一个包含 Unicode 字符串的数据帧，并将其转换为字符串
        df = DataFrame({"A": ["\u05d0"]})
        str(df)

    def test_repr_unicode_columns(self):
        # 创建一个包含 Unicode 列名的数据帧，并返回其列名的字符串表示形式
        df = DataFrame({"\u05d0": [1, 2, 3], "\u05d1": [4, 5, 6], "c": [7, 8, 9]})
        repr(df.columns)  # 不应引发 UnicodeDecodeError

    def test_str_to_bytes_raises(self):
        # GH 26447
        # 创建一个包含字符串列的数据帧
        df = DataFrame({"A": ["abc"]})
        # 定义错误消息正则表达式
        msg = "^'str' object cannot be interpreted as an integer$"
        # 断言在将数据帧转换为字节时引发 TypeError，且错误消息匹配预期正则表达式
        with pytest.raises(TypeError, match=msg):
            bytes(df)

    def test_very_wide_repr(self):
        # 创建一个包含标准正态分布随机数的10行20列的数据帧，列名为长度为10的字符串数组
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 20)),
            columns=np.array(["a" * 10] * 20, dtype=object),
        )
        # 返回数据帧的字符串表示形式
        repr(df)

    def test_repr_column_name_unicode_truncation_bug(self):
        # #1906
        # 创建一个包含 Unicode 字符串列的数据帧
        df = DataFrame(
            {
                "Id": [7117434],
                "StringCol": (
                    "Is it possible to modify drop plot code"
                    "so that the output graph is displayed "
                    "in iphone simulator, Is it possible to "
                    "modify drop plot code so that the "
                    "output graph is \xe2\x80\xa8displayed "
                    "in iphone simulator.Now we are adding "
                    "the CSV file externally. I want to Call "
                    "the File through the code.."
                ),
            }
        )

        # 使用上下文设置，将最大列数设置为20
        with option_context("display.max_columns", 20):
            # 断言字符串列名在数据帧的字符串表示形式中
            assert "StringCol" in repr(df)

    def test_latex_repr(self):
        # 检查是否可以导入 jinja2 库，如果无法导入则跳过该测试
        pytest.importorskip("jinja2")
        # 期望的 LaTeX 表示形式
        expected = r"""\begin{tabular}{llll}
    @pytest.mark.parametrize("arg", [np.datetime64, np.timedelta64])
    @pytest.mark.parametrize(
        "box, expected",
        [[Series, "0    NaT\ndtype: object"], [DataFrame, "     0\n0  NaT"]],
    )
    # 使用 pytest 的参数化功能，分别测试 np.datetime64 和 np.timedelta64
    # 对于不同的数据结构（Series 和 DataFrame），验证其包含 NaT 的对象在使用 object 数据类型时的表示是否符合预期
    def test_repr_np_nat_with_object(self, arg, box, expected):
        # GH 25445
        # 创建一个包含特定日期或时间间隔的对象，并转换成字符串表示形式
        result = repr(box([arg("NaT")], dtype=object))
        # 断言转换后的字符串表示是否与预期结果一致
        assert result == expected

    def test_frame_datetime64_pre1900_repr(self):
        # 创建一个包含早于 1900 年日期的 DataFrame 对象
        df = DataFrame({"year": date_range("1/1/1700", periods=50, freq="YE-DEC")})
        # 调用 DataFrame 对象的 repr 方法，返回其字符串表示形式
        repr(df)

    def test_frame_to_string_with_periodindex(self):
        # 创建一个带有 PeriodIndex 的 DataFrame 对象
        index = PeriodIndex(["2011-1", "2011-2", "2011-3"], freq="M")
        frame = DataFrame(np.random.default_rng(2).standard_normal((3, 4)), index=index)
        # 调用 DataFrame 对象的 to_string 方法，返回其字符串表示形式
        frame.to_string()

    def test_to_string_ea_na_in_multiindex(self):
        # GH#47986
        # 创建一个带有 MultiIndex 的 DataFrame 对象，其中包含 NA 值
        df = DataFrame(
            {"a": [1, 2]},
            index=MultiIndex.from_arrays([Series([NA, 1], dtype="Int64")]),
        )
        # 调用 DataFrame 对象的 to_string 方法，返回其字符串表示形式
        result = df.to_string()
        expected = """      a
(NA, 1)  1
"""
        # 断言转换后的字符串表示是否与预期结果一致
        assert result == expected
    def test_datetime64tz_slice_non_truncate(self):
        # GH 30263
        # 创建一个包含日期时间的数据框，使用UTC时区
        df = DataFrame({"x": date_range("2019", periods=10, tz="UTC")})
        # 期望的数据框字符串表示
        expected = repr(df)
        # 对数据框进行切片操作，保留前5列
        df = df.iloc[:, :5]
        # 获取切片后的数据框字符串表示
        result = repr(df)
        # 断言切片后的字符串表示与期望值相等
        assert result == expected

    def test_to_records_no_typeerror_in_repr(self):
        # GH 48526
        # 创建一个包含字符串数据的数据框，指定列名
        df = DataFrame([["a", "b"], ["c", "d"], ["e", "f"]], columns=["left", "right"])
        # 将指定列转换为记录格式，并赋值给新列'record'
        df["record"] = df[["left", "right"]].to_records()
        # 期望的数据框字符串表示，包含记录列
        expected = """  left right     record
0    a     b  [0, a, b]
1    c     d  [1, c, d]
2    e     f  [2, e, f]"""
        # 获取数据框的字符串表示
        result = repr(df)
        # 断言结果字符串与期望值相等
        assert result == expected

    def test_to_records_with_na_record_value(self):
        # GH 48526
        # 创建一个包含NaN值的数据框，指定列名
        df = DataFrame(
            [["a", np.nan], ["c", "d"], ["e", "f"]], columns=["left", "right"]
        )
        # 将指定列转换为记录格式，并赋值给新列'record'
        df["record"] = df[["left", "right"]].to_records()
        # 期望的数据框字符串表示，包含记录列和NaN值
        expected = """  left right       record
0    a   NaN  [0, a, nan]
1    c     d    [1, c, d]
2    e     f    [2, e, f]"""
        # 获取数据框的字符串表示
        result = repr(df)
        # 断言结果字符串与期望值相等
        assert result == expected

    def test_to_records_with_na_record(self):
        # GH 48526
        # 创建一个包含NaN值的数据框，指定列名
        df = DataFrame(
            [["a", "b"], [np.nan, np.nan], ["e", "f"]], columns=[np.nan, "right"]
        )
        # 将指定列转换为记录格式，并赋值给新列'record'
        df["record"] = df[[np.nan, "right"]].to_records()
        # 期望的数据框字符串表示，包含记录列和NaN值
        expected = """   NaN right         record
0    a     b      [0, a, b]
1  NaN   NaN  [1, nan, nan]
2    e     f      [2, e, f]"""
        # 获取数据框的字符串表示
        result = repr(df)
        # 断言结果字符串与期望值相等
        assert result == expected

    def test_to_records_with_inf_record(self):
        # GH 48526
        # 期望的数据框字符串表示，包含记录列和无穷大值
        expected = """   NaN  inf         record
0  inf    b    [0, inf, b]
1  NaN  NaN  [1, nan, nan]
2    e    f      [2, e, f]"""
        # 创建一个包含无穷大值的数据框，指定列名
        df = DataFrame(
            [[np.inf, "b"], [np.nan, np.nan], ["e", "f"]],
            columns=[np.nan, np.inf],
        )
        # 将指定列转换为记录格式，并赋值给新列'record'
        df["record"] = df[[np.nan, np.inf]].to_records()
        # 获取数据框的字符串表示
        result = repr(df)
        # 断言结果字符串与期望值相等
        assert result == expected

    def test_masked_ea_with_formatter(self):
        # GH#39336
        # 创建一个包含浮点数和整数数据的数据框
        df = DataFrame(
            {
                "a": Series([0.123456789, 1.123456789], dtype="Float64"),
                "b": Series([1, 2], dtype="Int64"),
            }
        )
        # 使用指定的格式化函数对数据框进行字符串表示
        result = df.to_string(formatters=["{:.2f}".format, "{:.2f}".format])
        # 期望的数据框字符串表示，保留两位小数
        expected = """      a     b
0  0.12  1.00
1  1.12  2.00"""
        # 断言结果字符串与期望值相等
        assert result == expected

    def test_repr_ea_columns(self, any_string_dtype):
        # GH#54797
        # 导入pyarrow模块，如果不存在则跳过测试
        pytest.importorskip("pyarrow")
        # 创建一个包含长列名的数据框，指定列名
        df = DataFrame({"long_column_name": [1, 2, 3], "col2": [4, 5, 6]})
        # 将数据框的列名转换为指定类型
        df.columns = df.columns.astype(any_string_dtype)
        # 期望的数据框字符串表示，包含长列名
        expected = """   long_column_name  col2
0                 1     4
1                 2     5
2                 3     6"""
        # 断言数据框的字符串表示与期望值相等
        assert repr(df) == expected
    # 定义一个包含多个元组的列表，每个元组包含三个元素：复数列表，预期的复数字符串表示列表
    [
        # 第一个元组
        ([2, complex("nan"), 1], [" 2.0+0.0j", " NaN+0.0j", " 1.0+0.0j"]),
        # 第二个元组
        ([2, complex("nan"), -1], [" 2.0+0.0j", " NaN+0.0j", "-1.0+0.0j"]),
        # 第三个元组
        ([-2, complex("nan"), -1], ["-2.0+0.0j", " NaN+0.0j", "-1.0+0.0j"]),
        # 第四个元组
        ([-1.23j, complex("nan"), -1], ["-0.00-1.23j", "  NaN+0.00j", "-1.00+0.00j"]),
        # 第五个元组
        ([1.23j, complex("nan"), 1.23], [" 0.00+1.23j", "  NaN+0.00j", " 1.23+0.00j"]),
        # 第六个元组
        (
            [-1.23j, complex(np.nan, np.nan), 1],
            ["-0.00-1.23j", "  NaN+ NaNj", " 1.00+0.00j"],
        ),
        # 第七个元组
        (
            [-1.23j, complex(1.2, np.nan), 1],
            ["-0.00-1.23j", " 1.20+ NaNj", " 1.00+0.00j"],
        ),
        # 第八个元组
        (
            [-1.23j, complex(np.nan, -1.2), 1],
            ["-0.00-1.23j", "  NaN-1.20j", " 1.00+0.00j"],
        ),
    ],
# 使用 pytest 框架的装饰器，为 test_repr_with_complex_nans 函数添加参数化测试，测试参数为 as_frame 取 True 和 False 的情况
@pytest.mark.parametrize("as_frame", [True, False])
# 定义测试函数 test_repr_with_complex_nans，用于测试复数数据的字符串表示
def test_repr_with_complex_nans(data, output, as_frame):
    # GH#53762, GH#53841
    # 创建一个 Series 对象，使用传入的 data 参数作为数据源
    obj = Series(np.array(data))
    # 根据参数 as_frame 的值进行条件判断
    if as_frame:
        # 如果 as_frame 为 True，则将 Series 对象转换为 DataFrame 对象，并命名列名为 "val"
        obj = obj.to_frame(name="val")
        # 根据输出列表 output 中的值生成每个元素的字符串表示，并保证对齐
        reprs = [f"{i} {val}" for i, val in enumerate(output)]
        # 生成期望的字符串表示，包括列名和每行数据的字符串表示
        expected = f"{'val': >{len(reprs[0])}}\n" + "\n".join(reprs)
    else:
        # 如果 as_frame 为 False，则直接生成每个元素的字符串表示，并保持对齐
        reprs = [f"{i}   {val}" for i, val in enumerate(output)]
        # 生成期望的字符串表示，包括每行数据的字符串表示和数据类型信息
        expected = "\n".join(reprs) + "\ndtype: complex128"
    # 使用断言验证 obj 的字符串表示与期望的字符串表示是否一致，若不一致则输出自定义消息
    assert str(obj) == expected, f"\n{obj!s}\n\n{expected}"
```
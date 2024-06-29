# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_to_string.py`

```
    # 导入所需的模块和库
    from datetime import (
        datetime,        # 导入 datetime 类
        timedelta,       # 导入 timedelta 类
    )
    from io import StringIO   # 导入 StringIO 类
    import re                   # 导入 re 模块，用于正则表达式操作
    import sys                  # 导入 sys 模块，提供对 Python 解释器的访问
    from textwrap import dedent   # 导入 dedent 函数，用于去除文本块的缩进

    import numpy as np            # 导入 NumPy 库，并将其命名为 np
    import pytest                 # 导入 pytest 测试框架

    from pandas._config import using_pyarrow_string_dtype   # 导入 pandas 的配置变量

    from pandas import (
        CategoricalIndex,        # 导入 CategoricalIndex 类
        DataFrame,               # 导入 DataFrame 类
        Index,                   # 导入 Index 类
        NaT,                     # 导入 NaT 常量，表示缺失日期时间
        Series,                  # 导入 Series 类
        Timestamp,               # 导入 Timestamp 类
        concat,                  # 导入 concat 函数，用于合并对象
        date_range,              # 导入 date_range 函数，用于生成日期范围
        get_option,              # 导入 get_option 函数，用于获取选项设置
        option_context,          # 导入 option_context 函数，用于设置选项上下文
        read_csv,                # 导入 read_csv 函数，用于读取 CSV 文件
        timedelta_range,         # 导入 timedelta_range 函数，用于生成时间间隔范围
        to_datetime,             # 导入 to_datetime 函数，用于将对象转换为日期时间
    )
    import pandas._testing as tm   # 导入 pandas 测试工具模块，命名为 tm

    class TestDataFrameToStringFormatters:
        def test_keyword_deprecation(self):
            # 测试关键字过时提醒
            # 定义提醒信息字符串
            msg = (
                "Starting with pandas version 4.0 all arguments of to_string "
                "except for the argument 'buf' will be keyword-only."
            )
            s = Series(["a", "b"])   # 创建 Series 对象 s
            with tm.assert_produces_warning(FutureWarning, match=msg):
                s.to_string(None, "NaN")   # 调用 to_string 方法，并验证是否产生 FutureWarning 提醒

        def test_to_string_masked_ea_with_formatter(self):
            # 测试使用格式化器进行 to_string 方法
            # 创建 DataFrame 对象 df
            df = DataFrame(
                {
                    "a": Series([0.123456789, 1.123456789], dtype="Float64"),
                    "b": Series([1, 2], dtype="Int64"),
                }
            )
            # 调用 DataFrame 的 to_string 方法，使用指定的格式化器，并保存结果
            result = df.to_string(formatters=["{:.2f}".format, "{:.2f}".format])
            # 预期的输出文本内容
            expected = dedent(
                """\
                      a     b
                0  0.12  1.00
                1  1.12  2.00"""
            )
            # 断言结果与预期输出相符
            assert result == expected

        def test_to_string_with_formatters(self):
            # 测试使用指定格式化器的 to_string 方法
            # 创建 DataFrame 对象 df
            df = DataFrame(
                {
                    "int": [1, 2, 3],
                    "float": [1.0, 2.0, 3.0],
                    "object": [(1, 2), True, False],
                },
                columns=["int", "float", "object"],   # 设置列名
            )

            # 定义格式化器列表
            formatters = [
                ("int", lambda x: f"0x{x:x}"),
                ("float", lambda x: f"[{x: 4.1f}]"),
                ("object", lambda x: f"-{x!s}-"),
            ]
            # 调用 DataFrame 的 to_string 方法，使用格式化器字典，并保存结果
            result = df.to_string(formatters=dict(formatters))
            # 再次调用 to_string 方法，使用格式化器函数列表，并保存结果
            result2 = df.to_string(formatters=list(zip(*formatters))[1])
            # 断言两次结果与预期输出相符
            assert result == (
                "  int  float    object\n"
                "0 0x1 [ 1.0]  -(1, 2)-\n"
                "1 0x2 [ 2.0]    -True-\n"
                "2 0x3 [ 3.0]   -False-"
            )
            assert result == result2

        def test_to_string_with_datetime64_monthformatter(self):
            # 测试使用日期时间格式化器的 to_string 方法
            months = [datetime(2016, 1, 1), datetime(2016, 2, 2)]
            x = DataFrame({"months": months})

            def format_func(x):
                return x.strftime("%Y-%m")

            # 调用 DataFrame 的 to_string 方法，使用格式化器函数，并保存结果
            result = x.to_string(formatters={"months": format_func})
            # 预期的输出文本内容
            expected = dedent(
                """\
                months
                0 2016-01
                1 2016-02"""
            )
            # 断言结果与预期输出相符（去除首尾空白字符后比较）
            assert result.strip() == expected
    # 定义一个测试方法，测试 DataFrame 的 to_string 方法在使用自定义时间格式化器时的输出
    def test_to_string_with_datetime64_hourformatter(self):
        # 创建一个 DataFrame 对象 x，包含一个名为 hod 的列，列值为两个时间字符串转换而成的 datetime64 对象
        x = DataFrame(
            {"hod": to_datetime(["10:10:10.100", "12:12:12.120"], format="%H:%M:%S.%f")}
        )

        # 定义一个内部函数 format_func，用于将 datetime64 对象格式化为 "%H:%M" 形式的字符串
        def format_func(x):
            return x.strftime("%H:%M")

        # 使用 DataFrame 的 to_string 方法，将 hod 列的值使用 format_func 格式化，并将结果赋给 result
        result = x.to_string(formatters={"hod": format_func})

        # 定义预期的输出 expected，使用 dedent 处理多行字符串，保证缩进一致性
        expected = dedent(
            """\
            hod
            0 10:10
            1 12:12"""
        )

        # 断言测试结果与预期输出相符合
        assert result.strip() == expected

    # 定义一个测试方法，测试 DataFrame 的 to_string 方法在使用 Unicode 列名和字符串格式化器时的输出
    def test_to_string_with_formatters_unicode(self):
        # 创建一个 DataFrame 对象 df，包含一个列名为 c/σ 的列，列值为整数列表
        df = DataFrame({"c/\u03c3": [1, 2, 3]})

        # 使用 DataFrame 的 to_string 方法，将 c/σ 列的值使用 str 格式化，并将结果赋给 result
        result = df.to_string(formatters={"c/\u03c3": str})

        # 定义预期的输出 expected，使用 dedent 处理多行字符串，保证缩进一致性
        expected = dedent(
            """\
              c/\u03c3
            0   1
            1   2
            2   3"""
        )

        # 断言测试结果与预期输出相符合
        assert result == expected

        # 定义一个内部测试方法 test_to_string_index_formatter
        def test_to_string_index_formatter(self):
            # 创建一个 DataFrame 对象 df，包含三行数据，每行为一个整数范围
            df = DataFrame([range(5), range(5, 10), range(10, 15)])

            # 使用 DataFrame 的 to_string 方法，将索引 __index__ 使用 lambda 函数将数字索引转换为 'abc' 字符
            rs = df.to_string(formatters={"__index__": lambda x: "abc"[x]})

            # 定义预期的输出 xp，使用 dedent 处理多行字符串，保证缩进一致性
            xp = dedent(
                """\
                0   1   2   3   4
            a   0   1   2   3   4
            b   5   6   7   8   9
            c  10  11  12  13  14\
            """
            )

            # 断言测试结果与预期输出相符合
            assert rs == xp

    # 定义一个测试方法，测试 DataFrame 的 to_string 方法在使用字符串格式化器时的输出
    def test_no_extra_space(self):
        # 定义字符串变量 col1, col2, col3，并赋予不同的字符串值
        col1 = "TEST"
        col2 = "PANDAS"
        col3 = "to_string"

        # 定义预期的字符串 expected，使用 f-string 格式化字符串，并设置每列的最小宽度
        expected = f"{col1:<6s} {col2:<7s} {col3:<10s}"

        # 创建一个 DataFrame 对象 df，包含一个字典列表，每个字典代表一行数据
        df = DataFrame([{"col1": "TEST", "col2": "PANDAS", "col3": "to_string"}])

        # 定义格式化器字典 d，每个列名对应一个字符串格式化函数，将 DataFrame 的内容转换为字符串
        d = {"col1": "{:<6s}".format, "col2": "{:<7s}".format, "col3": "{:<10s}".format}

        # 使用 DataFrame 的 to_string 方法，指定不输出索引和列头，并使用 d 中定义的格式化器
        result = df.to_string(index=False, header=False, formatters=d)

        # 断言测试结果与预期输出相符合
        assert result == expected
class TestDataFrameToStringColSpace:
    # 测试函数：测试在指定列空间时是否会引发异常
    def test_to_string_with_column_specific_col_space_raises(self):
        # 创建一个包含随机数据的 DataFrame，3行3列
        df = DataFrame(
            np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
        )

        # 定义错误消息的正则表达式模式，用于匹配异常信息
        msg = (
            "Col_space length\\(\\d+\\) should match "
            "DataFrame number of columns\\(\\d+\\)"
        )
        
        # 测试使用不匹配列数的 col_space 是否引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.to_string(col_space=[30, 40])

        # 再次测试使用不匹配列数的 col_space 是否引发 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.to_string(col_space=[30, 40, 50, 60])

        # 定义错误消息，测试指定未知列名是否引发 ValueError 异常
        msg = "unknown column"
        with pytest.raises(ValueError, match=msg):
            df.to_string(col_space={"a": "foo", "b": 23, "d": 34})

    # 测试函数：测试在指定列空间时的输出格式
    def test_to_string_with_column_specific_col_space(self):
        # 创建一个包含随机数据的 DataFrame，3行3列
        df = DataFrame(
            np.random.default_rng(2).random(size=(3, 3)), columns=["a", "b", "c"]
        )

        # 测试使用指定每列空间的字典 col_space 是否得到正确格式的字符串输出
        result = df.to_string(col_space={"a": 10, "b": 11, "c": 12})
        # 验证输出字符串的第二行是否符合预期的总长度
        assert len(result.split("\n")[1]) == (3 + 1 + 10 + 11 + 12)

        # 测试使用指定每列空间的列表 col_space 是否得到正确格式的字符串输出
        result = df.to_string(col_space=[10, 11, 12])
        assert len(result.split("\n")[1]) == (3 + 1 + 10 + 11 + 12)

    # 测试函数：测试在不同列空间下的字符串输出格式
    def test_to_string_with_col_space(self):
        # 创建一个包含随机数据的 DataFrame，1行3列
        df = DataFrame(np.random.default_rng(2).random(size=(1, 3)))
        
        # 获取使用不同列空间时字符串输出的第二行的长度
        c10 = len(df.to_string(col_space=10).split("\n")[1])
        c20 = len(df.to_string(col_space=20).split("\n")[1])
        c30 = len(df.to_string(col_space=30).split("\n")[1])
        # 验证列空间为 10, 20, 30 时，第二行长度的顺序关系
        assert c10 < c20 < c30

        # 测试在 header=False 的情况下是否正确应用了 col_space
        # 验证有头部的字符串输出的第二行与无头部的字符串输出的第二行长度是否相等
        with_header = df.to_string(col_space=20)
        with_header_row1 = with_header.splitlines()[1]
        no_header = df.to_string(col_space=20, header=False)
        assert len(with_header_row1) == len(no_header)

    # 测试函数：测试在包含元组的列中使用 to_string 的输出格式
    def test_to_string_repr_tuples(self):
        # 创建一个 StringIO 对象作为缓冲区
        buf = StringIO()

        # 创建一个包含元组数据的 DataFrame
        df = DataFrame({"tups": list(zip(range(10), range(10)))})
        repr(df)
        # 测试使用指定列空间时的字符串输出格式
        df.to_string(col_space=10, buf=buf)


class TestDataFrameToStringHeader:
    # 测试函数：测试在 header=False 的情况下的字符串输出格式
    def test_to_string_header_false(self):
        # 创建一个简单的 DataFrame
        df = DataFrame([1, 2])
        df.index.name = "a"
        # 测试在不包含头部的情况下的字符串输出格式
        s = df.to_string(header=False)
        expected = "a   \n0  1\n1  2"
        assert s == expected

        # 创建另一个 DataFrame
        df = DataFrame([[1, 2], [3, 4]])
        df.index.name = "a"
        # 测试在不包含头部的情况下的字符串输出格式
        s = df.to_string(header=False)
        expected = "a      \n0  1  2\n1  3  4"
        assert s == expected

    # 测试函数：测试在多级索引头部的情况下的字符串输出格式
    def test_to_string_multindex_header(self):
        # 创建一个多级索引的 DataFrame
        df = DataFrame({"a": [0], "b": [1], "c": [2], "d": [3]}).set_index(["a", "b"])
        # 测试在包含自定义头部的情况下的字符串输出格式
        res = df.to_string(header=["r1", "r2"])
        exp = "    r1 r2\na b      \n0 1  2  3"
        assert res == exp

    # 测试函数：测试在没有头部的情况下的字符串输出格式
    def test_to_string_no_header(self):
        # 创建一个简单的 DataFrame
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # 测试在不包含头部的情况下的字符串输出格式
        df_s = df.to_string(header=False)
        expected = "0  1  4\n1  2  5\n2  3  6"

        assert df_s == expected
    # 定义一个测试方法，用于测试数据框的特定头部转换为字符串的功能
    def test_to_string_specified_header(self):
        # 创建一个包含两列数据的数据框
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # 调用数据框的 to_string 方法，将头部替换为指定的列名，生成字符串表示
        df_s = df.to_string(header=["X", "Y"])
        # 预期的字符串结果
        expected = "   X  Y\n0  1  4\n1  2  5\n2  3  6"

        # 断言生成的字符串与预期结果相符
        assert df_s == expected

        # 准备错误消息字符串，用于验证异常抛出时的匹配信息
        msg = "Writing 2 cols but got 1 aliases"
        # 使用 pytest 库检查是否抛出 ValueError 异常，并验证异常消息与预期的匹配信息
        with pytest.raises(ValueError, match=msg):
            df.to_string(header=["X"])
class TestDataFrameToStringLineWidth:
    # 测试类，用于测试 DataFrame 的 to_string 方法在不同行宽下的输出

    def test_to_string_line_width(self):
        # 测试 DataFrame 的 to_string 方法在指定行宽下的输出是否符合预期
        df = DataFrame(123, index=range(10, 15), columns=range(30))
        lines = df.to_string(line_width=80)
        # 断言最长行的长度是否为80
        assert max(len(line) for line in lines.split("\n")) == 80

    def test_to_string_line_width_no_index(self):
        # 测试在不显示索引的情况下，DataFrame 的 to_string 方法在指定行宽下的输出是否符合预期
        # GH#13998, GH#22505
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, index=False)
        expected = " x  \\\n 1   \n 2   \n 3   \n\n y  \n 4  \n 5  \n 6  "
        # 断言输出是否与预期相符
        assert df_s == expected

        df = DataFrame({"x": [11, 22, 33], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, index=False)
        expected = " x  \\\n11   \n22   \n33   \n\n y  \n 4  \n 5  \n 6  "
        # 断言输出是否与预期相符
        assert df_s == expected

        df = DataFrame({"x": [11, 22, -33], "y": [4, 5, -6]})

        df_s = df.to_string(line_width=1, index=False)
        expected = "  x  \\\n 11   \n 22   \n-33   \n\n y  \n 4  \n 5  \n-6  "
        # 断言输出是否与预期相符
        assert df_s == expected

    def test_to_string_line_width_no_header(self):
        # 测试在不显示列名的情况下，DataFrame 的 to_string 方法在指定行宽下的输出是否符合预期
        # GH#53054
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, header=False)
        expected = "0  1  \\\n1  2   \n2  3   \n\n0  4  \n1  5  \n2  6  "
        # 断言输出是否与预期相符
        assert df_s == expected

        df = DataFrame({"x": [11, 22, 33], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1, header=False)
        expected = "0  11  \\\n1  22   \n2  33   \n\n0  4  \n1  5  \n2  6  "
        # 断言输出是否与预期相符
        assert df_s == expected

        df = DataFrame({"x": [11, 22, -33], "y": [4, 5, -6]})

        df_s = df.to_string(line_width=1, header=False)
        expected = "0  11  \\\n1  22   \n2 -33   \n\n0  4  \n1  5  \n2 -6  "
        # 断言输出是否与预期相符
        assert df_s == expected

    def test_to_string_line_width_with_both_index_and_header(self):
        # 测试在显示索引和列名的情况下，DataFrame 的 to_string 方法在指定行宽下的输出是否符合预期
        # GH#53054
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1)
        expected = (
            "   x  \\\n0  1   \n1  2   \n2  3   \n\n   y  \n0  4  \n1  5  \n2  6  "
        )
        # 断言输出是否与预期相符
        assert df_s == expected

        df = DataFrame({"x": [11, 22, 33], "y": [4, 5, 6]})

        df_s = df.to_string(line_width=1)
        expected = (
            "    x  \\\n0  11   \n1  22   \n2  33   \n\n   y  \n0  4  \n1  5  \n2  6  "
        )
        # 断言输出是否与预期相符
        assert df_s == expected

        df = DataFrame({"x": [11, 22, -33], "y": [4, 5, -6]})

        df_s = df.to_string(line_width=1)
        expected = (
            "    x  \\\n0  11   \n1  22   \n2 -33   \n\n   y  \n0  4  \n1  5  \n2 -6  "
        )
        # 断言输出是否与预期相符
        assert df_s == expected
    # 定义一个测试函数，测试不带索引和标题的情况下 DataFrame 对象转换为字符串的功能
    def test_to_string_line_width_no_index_no_header(self):
        # GH#53054：引用 GitHub 上的 issue 编号，指明这段代码解决的问题
        # 创建一个 DataFrame 对象，包含两列数据 'x' 和 'y'
        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        # 将 DataFrame 转换为字符串，设置每行宽度为 1，不包含索引和标题
        df_s = df.to_string(line_width=1, index=False, header=False)
        # 预期的字符串输出
        expected = "1  \\\n2   \n3   \n\n4  \n5  \n6  "

        # 断言实际输出与预期输出相等
        assert df_s == expected

        # 修改 DataFrame 的数据
        df = DataFrame({"x": [11, 22, 33], "y": [4, 5, 6]})

        # 再次将 DataFrame 转换为字符串，设置每行宽度为 1，不包含索引和标题
        df_s = df.to_string(line_width=1, index=False, header=False)
        # 更新预期的字符串输出
        expected = "11  \\\n22   \n33   \n\n4  \n5  \n6  "

        # 断言实际输出与更新后的预期输出相等
        assert df_s == expected

        # 再次修改 DataFrame 的数据
        df = DataFrame({"x": [11, 22, -33], "y": [4, 5, -6]})

        # 再次将 DataFrame 转换为字符串，设置每行宽度为 1，不包含索引和标题
        df_s = df.to_string(line_width=1, index=False, header=False)
        # 更新预期的字符串输出
        expected = " 11  \\\n 22   \n-33   \n\n 4  \n 5  \n-6  "

        # 断言实际输出与更新后的预期输出相等
        assert df_s == expected
class TestToStringNumericFormatting:
    def test_to_string_float_format_no_fixed_width(self):
        # GH#21625
        # 创建一个包含单个浮点数列的数据框
        df = DataFrame({"x": [0.19999]})
        # 预期的字符串表示，包含单独的列 'x' 和格式化后的浮点数
        expected = "      x\n0 0.200"
        # 断言数据框转换为字符串后的结果符合预期
        assert df.to_string(float_format="%.3f") == expected

        # GH#22270
        # 创建一个包含单个整数列的数据框
        df = DataFrame({"x": [100.0]})
        # 预期的字符串表示，包含单独的列 'x' 和格式化后的整数
        expected = "    x\n0 100"
        # 断言数据框转换为字符串后的结果符合预期
        assert df.to_string(float_format="%.0f") == expected

    def test_to_string_small_float_values(self):
        # 创建一个包含小浮点数的数据框
        df = DataFrame({"a": [1.5, 1e-17, -5.5e-7]})

        # 获取数据框转换为字符串的结果
        result = df.to_string()
        # 如果条件满足，设置预期的字符串表示形式，包含列 'a' 和科学计数法格式化的浮点数
        if _three_digit_exp():
            expected = (
                "               a\n"
                "0  1.500000e+000\n"
                "1  1.000000e-017\n"
                "2 -5.500000e-007"
            )
        else:
            expected = (
                "              a\n"
                "0  1.500000e+00\n"
                "1  1.000000e-17\n"
                "2 -5.500000e-07"
            )
        # 断言数据框转换为字符串后的结果符合预期
        assert result == expected

        # 将数据框所有值乘以零，得到全零或部分零的数据框
        df = df * 0
        # 获取数据框转换为字符串的结果
        result = df.to_string()
        # 设置预期的字符串表示形式，包含列 '0' 和零值的表示
        expected = "   0\n0  0\n1  0\n2 -0"
        # TODO: 断言这些结果匹配??

    def test_to_string_complex_float_formatting(self):
        # GH #25514, 25745
        # 使用指定的显示精度上下文，创建一个包含复数的数据框
        with option_context("display.precision", 5):
            df = DataFrame(
                {
                    "x": [
                        (0.4467846931321966 + 0.0715185102060818j),
                        (0.2739442392974528 + 0.23515228785438969j),
                        (0.26974928742135185 + 0.3250604054898979j),
                        (-1j),
                    ]
                }
            )
            # 获取数据框转换为字符串的结果
            result = df.to_string()
            # 设置预期的字符串表示形式，包含列 'x' 和复数的格式化表示
            expected = (
                "                  x\n0  0.44678+0.07152j\n"
                "1  0.27394+0.23515j\n"
                "2  0.26975+0.32506j\n"
                "3 -0.00000-1.00000j"
            )
            # 断言数据框转换为字符串后的结果符合预期
            assert result == expected
    def test_to_string_format_inf(self):
        # 测试处理无穷大和字符串的情况
        df = DataFrame(
            {
                "A": [-np.inf, np.inf, -1, -2.1234, 3, 4],
                "B": [-np.inf, np.inf, "foo", "foooo", "fooooo", "bar"],
            }
        )
        # 将DataFrame对象转换为字符串表示
        result = df.to_string()

        expected = (
            "        A       B\n"
            "0    -inf    -inf\n"
            "1     inf     inf\n"
            "2 -1.0000     foo\n"
            "3 -2.1234   foooo\n"
            "4  3.0000  fooooo\n"
            "5  4.0000     bar"
        )
        # 断言结果与预期输出一致
        assert result == expected

        df = DataFrame(
            {
                "A": [-np.inf, np.inf, -1.0, -2.0, 3.0, 4.0],
                "B": [-np.inf, np.inf, "foo", "foooo", "fooooo", "bar"],
            }
        )
        # 再次将DataFrame对象转换为字符串表示
        result = df.to_string()

        expected = (
            "     A       B\n"
            "0 -inf    -inf\n"
            "1  inf     inf\n"
            "2 -1.0     foo\n"
            "3 -2.0   foooo\n"
            "4  3.0  fooooo\n"
            "5  4.0     bar"
        )
        # 再次断言结果与预期输出一致
        assert result == expected

    def test_to_string_int_formatting(self):
        # 测试整数格式化的情况
        df = DataFrame({"x": [-15, 20, 25, -35]})
        # 断言DataFrame的"x"列属于numpy的整数类型
        assert issubclass(df["x"].dtype.type, np.integer)

        # 将DataFrame对象转换为字符串表示
        output = df.to_string()
        expected = "    x\n0 -15\n1  20\n2  25\n3 -35"
        # 断言结果与预期输出一致
        assert output == expected

    def test_to_string_float_formatting(self):
        # 使用上下文管理器设置显示选项
        with option_context(
            "display.precision",
            5,
            "display.notebook_repr_html",
            False,
        ):
            df = DataFrame(
                {"x": [0, 0.25, 3456.000, 12e45, 1.64e6, 1.7e8, 1.253456, np.pi, -1e6]}
            )

            # 将DataFrame对象转换为字符串表示
            df_s = df.to_string()

            if _three_digit_exp():
                expected = (
                    "              x\n0  0.00000e+000\n1  2.50000e-001\n"
                    "2  3.45600e+003\n3  1.20000e+046\n4  1.64000e+006\n"
                    "5  1.70000e+008\n6  1.25346e+000\n7  3.14159e+000\n"
                    "8 -1.00000e+006"
                )
            else:
                expected = (
                    "             x\n0  0.00000e+00\n1  2.50000e-01\n"
                    "2  3.45600e+03\n3  1.20000e+46\n4  1.64000e+06\n"
                    "5  1.70000e+08\n6  1.25346e+00\n7  3.14159e+00\n"
                    "8 -1.00000e+06"
                )
            # 断言结果与预期输出一致
            assert df_s == expected

            df = DataFrame({"x": [3234, 0.253]})
            # 再次将DataFrame对象转换为字符串表示
            df_s = df.to_string()

            expected = "          x\n0  3234.000\n1     0.253"
            # 断言结果与预期输出一致
            assert df_s == expected

        # 断言当前显示精度为6
        assert get_option("display.precision") == 6

        df = DataFrame({"x": [1e9, 0.2512]})
        # 将DataFrame对象转换为字符串表示
        df_s = df.to_string()

        if _three_digit_exp():
            expected = "               x\n0  1.000000e+009\n1  2.512000e-001"
        else:
            expected = "              x\n0  1.000000e+09\n1  2.512000e-01"
        # 断言结果与预期输出一致
        assert df_s == expected
class TestDataFrameToString:
    # TestDataFrameToString 类用于测试 DataFrame 转换为字符串的功能

    def test_to_string_decimal(self):
        # test_to_string_decimal 方法测试将 DataFrame 转换为字符串，并使用逗号作为小数点分隔符的情况
        # GH#23614

        # 创建包含浮点数列 'A' 的 DataFrame
        df = DataFrame({"A": [6.0, 3.1, 2.2]})
        
        # 预期的字符串表示，每行一个浮点数，小数点使用逗号分隔
        expected = "     A\n0  6,0\n1  3,1\n2  2,2"
        
        # 断言转换后的字符串与预期字符串相等
        assert df.to_string(decimal=",") == expected

    def test_to_string_left_justify_cols(self):
        # test_to_string_left_justify_cols 方法测试将 DataFrame 转换为字符串，并左对齐列的情况

        # 创建包含数值列 'x' 的 DataFrame
        df = DataFrame({"x": [3234, 0.253]})
        
        # 将 DataFrame 转换为左对齐的字符串表示
        df_s = df.to_string(justify="left")
        
        # 预期的字符串表示，每行一个数值，左对齐
        expected = "   x       \n0  3234.000\n1     0.253"
        
        # 断言转换后的字符串与预期字符串相等
        assert df_s == expected

    def test_to_string_format_na(self):
        # test_to_string_format_na 方法测试将 DataFrame 转换为字符串，并格式化 NaN 值的情况

        # 创建包含列 'A' 和 'B' 的 DataFrame，其中包含 NaN 值和不同类型的数据
        df = DataFrame(
            {
                "A": [np.nan, -1, -2.1234, 3, 4],
                "B": [np.nan, "foo", "foooo", "fooooo", "bar"],
            }
        )
        
        # 将 DataFrame 转换为字符串表示
        result = df.to_string()
        
        # 预期的字符串表示，包含对 NaN 值和数值的格式化
        expected = (
            "        A       B\n"
            "0     NaN     NaN\n"
            "1 -1.0000     foo\n"
            "2 -2.1234   foooo\n"
            "3  3.0000  fooooo\n"
            "4  4.0000     bar"
        )
        
        # 断言转换后的字符串与预期字符串相等
        assert result == expected

        # 另一个测试用例，测试不同的 NaN 表示方式
        df = DataFrame(
            {
                "A": [np.nan, -1.0, -2.0, 3.0, 4.0],
                "B": [np.nan, "foo", "foooo", "fooooo", "bar"],
            }
        )
        result = df.to_string()

        expected = (
            "     A       B\n"
            "0  NaN     NaN\n"
            "1 -1.0     foo\n"
            "2 -2.0   foooo\n"
            "3  3.0  fooooo\n"
            "4  4.0     bar"
        )
        assert result == expected

    def test_to_string_with_dict_entries(self):
        # test_to_string_with_dict_entries 方法测试将包含字典的列转换为字符串的情况

        # 创建包含字典列 'A' 的 DataFrame
        df = DataFrame({"A": [{"a": 1, "b": 2}]})

        # 将 DataFrame 转换为字符串表示
        val = df.to_string()
        
        # 断言结果字符串包含字典中的键和值
        assert "'a': 1" in val
        assert "'b': 2" in val

    def test_to_string_with_categorical_columns(self):
        # test_to_string_with_categorical_columns 方法测试将分类列转换为字符串的情况
        # GH#35439

        # 创建包含数据和列名的列表，创建具有分类索引的 DataFrame
        data = [[4, 2], [3, 2], [4, 3]]
        cols = ["aaaaaaaaa", "b"]
        df = DataFrame(data, columns=cols)
        df_cat_cols = DataFrame(data, columns=CategoricalIndex(cols))

        # 断言普通和具有分类索引的 DataFrame 转换后的字符串表示相同
        assert df.to_string() == df_cat_cols.to_string()

    def test_repr_embedded_ndarray(self):
        # test_repr_embedded_ndarray 方法测试包含嵌套 ndarray 的 DataFrame 的 __repr__ 方法

        # 创建一个空的 ndarray，包含具有随机正态分布值的 "err" 列
        arr = np.empty(10, dtype=[("err", object)])
        for i in range(len(arr)):
            arr["err"][i] = np.random.default_rng(2).standard_normal(i)

        # 使用 ndarray 创建 DataFrame
        df = DataFrame(arr)
        
        # 调用 DataFrame 的 __repr__ 方法
        repr(df["err"])
        repr(df)
        
        # 将 DataFrame 转换为字符串表示
        df.to_string()
    def test_to_string_truncate(self):
        # 测试用例：验证当调用 DataFrame.to_string 时不会截断
        df = DataFrame(
            [
                {
                    "a": "foo",
                    "b": "bar",
                    "c": "let's make this a very VERY long line that is longer "
                    "than the default 50 character limit",
                    "d": 1,
                },
                {"a": "foo", "b": "bar", "c": "stuff", "d": 1},
            ]
        )
        df.set_index(["a", "b", "c"])  # 设置 DataFrame 的索引，但未保存结果
        assert df.to_string() == (
            "     a    b                                         "
            "                                                c  d\n"
            "0  foo  bar  let's make this a very VERY long line t"
            "hat is longer than the default 50 character limit  1\n"
            "1  foo  bar                                         "
            "                                            stuff  1"
        )
        with option_context("max_colwidth", 20):
            # 使用上下文管理器设置最大列宽为20，但对 to_string 方法无效
            assert df.to_string() == (
                "     a    b                                         "
                "                                                c  d\n"
                "0  foo  bar  let's make this a very VERY long line t"
                "hat is longer than the default 50 character limit  1\n"
                "1  foo  bar                                         "
                "                                            stuff  1"
            )
        assert df.to_string(max_colwidth=20) == (
            "     a    b                    c  d\n"
            "0  foo  bar  let's make this ...  1\n"
            "1  foo  bar                stuff  1"
        )

    @pytest.mark.parametrize(
        "input_array, expected",
        [
            ({"A": ["a"]}, "A\na"),
            ({"A": ["a", "b"], "B": ["c", "dd"]}, "A  B\na  c\nb dd"),
            ({"A": ["a", 1], "B": ["aa", 1]}, "A  B\na aa\n1  1"),
        ],
    )
    def test_format_remove_leading_space_dataframe(self, input_array, expected):
        # 测试用例：验证 DataFrame.to_string(index=False) 移除首部空格
        df = DataFrame(input_array).to_string(index=False)
        assert df == expected

    @pytest.mark.parametrize(
        "data,expected",
        [
            (
                {"col1": [1, 2], "col2": [3, 4]},
                "   col1  col2\n0     1     3\n1     2     4",
            ),
            (
                {"col1": ["Abc", 0.756], "col2": [np.nan, 4.5435]},
                "    col1    col2\n0    Abc     NaN\n1  0.756  4.5435",
            ),
            (
                {"col1": [np.nan, "a"], "col2": [0.009, 3.543], "col3": ["Abc", 23]},
                "  col1   col2 col3\n0  NaN  0.009  Abc\n1    a  3.543   23",
            ),
        ],
    )
    # 定义一个测试方法，测试在 max_rows 设置为 0 时的 DataFrame 转换为字符串的行为
    def test_to_string_max_rows_zero(self, data, expected):
        # 使用给定的数据创建 DataFrame，并将其转换为字符串，设置 max_rows 为 0
        result = DataFrame(data=data).to_string(max_rows=0)
        # 断言转换后的字符串与期望的结果相等
        assert result == expected

    # 使用 pytest 的参数化标记，定义多组参数进行测试
    @pytest.mark.parametrize(
        "max_cols, max_rows, expected",
        [
            (
                10,
                None,
                # 生成一个表格字符串，展示 0 到 10 列的数据
                " 0   1   2   3   4   ...  6   7   8   9   10\n"
                "  0   0   0   0   0  ...   0   0   0   0   0\n"
                "  0   0   0   0   0  ...   0   0   0   0   0\n"
                "  0   0   0   0   0  ...   0   0   0   0   0",
            ),
            (
                None,
                2,
                # 生成一个表格字符串，展示包含 2 行的数据，自动省略部分行
                " 0   1   2   3   4   5   6   7   8   9   10\n"
                "  0   0   0   0   0   0   0   0   0   0   0\n"
                " ..  ..  ..  ..  ..  ..  ..  ..  ..  ..  ..\n"
                "  0   0   0   0   0   0   0   0   0   0   0",
            ),
            (
                10,
                2,
                # 生成一个表格字符串，展示 0 到 10 列，2 行的数据，自动省略部分行和列
                " 0   1   2   3   4   ...  6   7   8   9   10\n"
                "  0   0   0   0   0  ...   0   0   0   0   0\n"
                " ..  ..  ..  ..  ..  ...  ..  ..  ..  ..  ..\n"
                "  0   0   0   0   0  ...   0   0   0   0   0",
            ),
            (
                9,
                2,
                # 生成一个表格字符串，展示 0 到 9 列，2 行的数据，自动省略部分行和列
                " 0   1   2   3   ...  7   8   9   10\n"
                "  0   0   0   0  ...   0   0   0   0\n"
                " ..  ..  ..  ..  ...  ..  ..  ..  ..\n"
                "  0   0   0   0  ...   0   0   0   0",
            ),
            (
                1,
                1,
                # 生成一个表格字符串，展示第 0 列的部分数据
                " 0  ...\n 0  ...\n..  ...",
            ),
        ],
    )
    # 定义一个测试方法，测试在没有索引时 DataFrame 转换为字符串的行为
    def test_truncation_no_index(self, max_cols, max_rows, expected):
        # 创建一个包含多行数据的 DataFrame
        df = DataFrame([[0] * 11] * 4)
        # 断言 DataFrame 转换为字符串后的结果与期望的字符串相等
        assert (
            df.to_string(index=False, max_cols=max_cols, max_rows=max_rows) == expected
        )

    # 定义一个测试方法，测试在没有索引时 DataFrame 转换为字符串的行为
    def test_to_string_no_index(self):
        # 使用指定的数据创建 DataFrame
        df = DataFrame({"x": [11, 22], "y": [33, -44], "z": ["AAA", "   "]})

        # 测试将 DataFrame 转换为字符串，没有索引
        df_s = df.to_string(index=False)
        # 断言转换后的字符串与期望的结果相等
        expected = " x   y   z\n11  33 AAA\n22 -44    "
        assert df_s == expected

        # 测试将 DataFrame 的特定列转换为字符串，没有索引
        df_s = df[["y", "x", "z"]].to_string(index=False)
        # 断言转换后的字符串与期望的结果相等
        expected = "  y  x   z\n 33 11 AAA\n-44 22    "
        assert df_s == expected

    # 定义一个测试方法，测试将包含 Unicode 列名的 DataFrame 转换为字符串的行为
    def test_to_string_unicode_columns(self, float_frame):
        # 使用包含 Unicode 列名的数据创建 DataFrame
        df = DataFrame({"\u03c3": np.arange(10.0)})

        # 使用 StringIO 缓存结果并获取字符串
        buf = StringIO()
        df.to_string(buf=buf)
        buf.getvalue()

        # 使用 StringIO 缓存结果并获取字符串
        buf = StringIO()
        df.info(buf=buf)
        buf.getvalue()

        # 测试将 DataFrame 转换为字符串
        result = float_frame.to_string()
        # 断言转换后的结果是字符串类型
        assert isinstance(result, str)

    # 使用 pytest 的参数化标记，定义 na_rep 参数的多组值进行测试
    @pytest.mark.parametrize("na_rep", ["NaN", "Ted"])
    def test_to_string_na_rep_and_float_format(self, na_rep):
        # GH#13828
        # 创建一个包含两列的DataFrame，其中一列包含字符串和浮点数，另一列包含字符串和None值
        df = DataFrame([["A", 1.2225], ["A", None]], columns=["Group", "Data"])
        # 将DataFrame转换为字符串格式，使用指定的NA替代值和浮点数格式化
        result = df.to_string(na_rep=na_rep, float_format="{:.2f}".format)
        # 期望的字符串表达，使用了dedent函数对其进行格式化处理
        expected = dedent(
            f"""\
               Group  Data
             0     A  1.22
             1     A   {na_rep}"""
        )
        # 断言结果是否符合预期
        assert result == expected

    def test_to_string_string_dtype(self):
        # GH#50099
        # 导入pyarrow模块，如果失败则跳过测试
        pytest.importorskip("pyarrow")
        # 创建一个DataFrame，包含不同的数据类型
        df = DataFrame(
            {"x": ["foo", "bar", "baz"], "y": ["a", "b", "c"], "z": [1, 2, 3]}
        )
        # 将DataFrame的列转换为指定的字符串数据类型
        df = df.astype(
            {"x": "string[pyarrow]", "y": "string[python]", "z": "int64[pyarrow]"}
        )
        # 将DataFrame的数据类型转换为字符串格式
        result = df.dtypes.to_string()
        # 期望的字符串表达，使用了dedent函数对其进行格式化处理
        expected = dedent(
            """\
            x    string[pyarrow]
            y     string[python]
            z     int64[pyarrow]"""
        )
        # 断言结果是否符合预期
        assert result == expected

    def test_to_string_utf8_columns(self):
        # 创建一个包含一个Unicode编码的列名的DataFrame
        n = "\u05d0".encode()
        df = DataFrame([1, 2], columns=[n])

        # 使用option_context设置显示最大行数为1，并生成DataFrame的字符串表示
        with option_context("display.max_rows", 1):
            repr(df)

    def test_to_string_unicode_two(self):
        # 创建一个包含一个Unicode字符的列名的DataFrame
        dm = DataFrame({"c/\u03c3": []})
        buf = StringIO()
        # 将DataFrame转换为字符串并写入缓冲区
        dm.to_string(buf)

    def test_to_string_unicode_three(self):
        # 创建一个包含一个Unicode字符的字符串的DataFrame
        dm = DataFrame(["\xc2"])
        buf = StringIO()
        # 将DataFrame转换为字符串并写入缓冲区
        dm.to_string(buf)

    def test_to_string_with_float_index(self):
        # 创建一个带有浮点数索引的DataFrame
        index = Index([1.5, 2, 3, 4, 5])
        df = DataFrame(np.arange(5), index=index)

        # 将DataFrame转换为字符串表示
        result = df.to_string()
        # 期望的字符串表达
        expected = "     0\n1.5  0\n2.0  1\n3.0  2\n4.0  3\n5.0  4"
        # 断言结果是否符合预期
        assert result == expected
    def test_to_string(self):
        # 创建一个 DataFrame 对象 biggie，包含两列："A" 和 "B"
        biggie = DataFrame(
            {
                "A": np.random.default_rng(2).standard_normal(200),
                "B": Index([f"{i}?!" for i in range(200)]),
            },
        )

        # 将 biggie 的前 20 行的 "A" 和 "B" 列设置为 NaN
        biggie.loc[:20, "A"] = np.nan
        biggie.loc[:20, "B"] = np.nan

        # 将 biggie 转换为字符串格式，并将其保存到变量 s 中
        s = biggie.to_string()

        # 创建一个 StringIO 对象 buf
        buf = StringIO()

        # 将 biggie 转换为字符串，并将结果写入 buf 中，返回值为 None
        retval = biggie.to_string(buf=buf)
        
        # 断言 retval 为 None
        assert retval is None

        # 断言 buf 的值与 s 相同
        assert buf.getvalue() == s

        # 断言 s 的类型为字符串
        assert isinstance(s, str)

        # 使用指定的列顺序 ("B", "A")，列宽为 17，浮点数格式化为 "%.5f"
        result = biggie.to_string(
            columns=["B", "A"], col_space=17, float_format="%.5f".__mod__
        )

        # 将 result 按行拆分成列表 lines
        lines = result.split("\n")

        # 提取第一行作为 header，去除首尾空格并按空格分割
        header = lines[0].strip().split()

        # 对 lines 中的每行进行处理，去除多余的空白字符并合并成一个字符串 joined
        joined = "\n".join([re.sub(r"\s+", " ", x).strip() for x in lines[1:]])

        # 从 joined 中读取 CSV 数据，使用 header 作为列名，无行头，以空格分隔
        recons = read_csv(StringIO(joined), names=header, header=None, sep=" ")

        # 断言 recons 的 "B" 列与 biggie 的 "B" 列相等
        tm.assert_series_equal(recons["B"], biggie["B"])

        # 断言 recons 的 "A" 列的非 NaN 值数量与 biggie 的 "A" 列相同
        assert recons["A"].count() == biggie["A"].count()

        # 断言 recons 的 "A" 列中所有非 NaN 值与 biggie 的 "A" 列中对应值的绝对差小于 0.1
        assert (np.abs(recons["A"].dropna() - biggie["A"].dropna()) < 0.1).all()

        # TODO: split or simplify this test?

        # 使用指定列 "A"，列宽为 17，将 biggie 转换为字符串格式
        result = biggie.to_string(columns=["A"], col_space=17)

        # 提取第一行作为 header，去除首尾空格并按空格分割
        header = result.split("\n")[0].strip().split()

        # 预期的 header 为 ["A"]
        expected = ["A"]

        # 断言 header 与预期相同
        assert header == expected

        # 使用自定义格式化函数对列 "A" 进行格式化，不修改 biggie
        biggie.to_string(columns=["B", "A"], formatters={"A": lambda x: f"{x:.1f}"})

        # 将列 "B" 和 "A" 的值转换为字符串
        biggie.to_string(columns=["B", "A"], float_format=str)

        # 使用指定列顺序 ("B", "A")，列宽为 12，将 biggie 转换为字符串
        biggie.to_string(columns=["B", "A"], col_space=12, float_format=str)

        # 创建一个空的 DataFrame frame，只包含索引，行数为 200
        frame = DataFrame(index=np.arange(200))

        # 将 frame 转换为字符串格式
        frame.to_string()
    def test_to_string_index_with_nan(self):
        # GH#2850
        # 创建一个 DataFrame 对象，包含四列，每列有两行数据
        df = DataFrame(
            {
                "id1": {0: "1a3", 1: "9h4"},
                "id2": {0: np.nan, 1: "d67"},
                "id3": {0: "78d", 1: "79d"},
                "value": {0: 123, 1: 64},
            }
        )

        # 将 DataFrame 按多重索引设置，基于列 'id1', 'id2', 'id3'
        y = df.set_index(["id1", "id2", "id3"])
        # 将 DataFrame 转换为字符串表示
        result = y.to_string()
        # 预期的字符串表示
        expected = (
            "             value\nid1 id2 id3       \n"
            "1a3 NaN 78d    123\n9h4 d67 79d     64"
        )
        # 断言结果与预期相符
        assert result == expected

        # 将 DataFrame 按单一索引设置，基于列 'id2'
        y = df.set_index("id2")
        # 将 DataFrame 转换为字符串表示
        result = y.to_string()
        # 预期的字符串表示
        expected = (
            "     id1  id3  value\nid2                 \n"
            "NaN  1a3  78d    123\nd67  9h4  79d     64"
        )
        # 断言结果与预期相符
        assert result == expected

        # 带有附加索引设置的情况（在版本0.12中失败）
        y = df.set_index(["id1", "id2"]).set_index("id3", append=True)
        # 将 DataFrame 转换为字符串表示
        result = y.to_string()
        # 预期的字符串表示
        expected = (
            "             value\nid1 id2 id3       \n"
            "1a3 NaN 78d    123\n9h4 d67 79d     64"
        )
        # 断言结果与预期相符
        assert result == expected

        # 在多重索引中全部为 NaN 的情况
        df2 = df.copy()
        df2.loc[:, "id2"] = np.nan
        y = df2.set_index("id2")
        # 将 DataFrame 转换为字符串表示
        result = y.to_string()
        # 预期的字符串表示
        expected = (
            "     id1  id3  value\nid2                 \n"
            "NaN  1a3  78d    123\nNaN  9h4  79d     64"
        )
        # 断言结果与预期相符
        assert result == expected

        # 在多重索引中部分值为 NaN 的情况
        df2 = df.copy()
        df2.loc[:, "id2"] = np.nan
        y = df2.set_index(["id2", "id3"])
        # 将 DataFrame 转换为字符串表示
        result = y.to_string()
        # 预期的字符串表示
        expected = (
            "         id1  value\nid2 id3            \n"
            "NaN 78d  1a3    123\n    79d  9h4     64"
        )
        # 断言结果与预期相符
        assert result == expected

        # 创建一个 DataFrame 对象，包含四列，每列有两行数据，且全部为 NaN
        df = DataFrame(
            {
                "id1": {0: np.nan, 1: "9h4"},
                "id2": {0: np.nan, 1: "d67"},
                "id3": {0: np.nan, 1: "79d"},
                "value": {0: 123, 1: 64},
            }
        )

        # 将 DataFrame 按多重索引设置，基于列 'id1', 'id2', 'id3'
        y = df.set_index(["id1", "id2", "id3"])
        # 将 DataFrame 转换为字符串表示
        result = y.to_string()
        # 预期的字符串表示
        expected = (
            "             value\nid1 id2 id3       \n"
            "NaN NaN NaN    123\n9h4 d67 79d     64"
        )
        # 断言结果与预期相符
        assert result == expected
    # 定义一个测试方法，用于测试 DataFrame 对象的字符串表示和转换
    def test_to_string_repr_unicode(self):
        # 创建一个字符串缓冲区对象
        buf = StringIO()

        # 创建包含Unicode字符"\u03c3"的列表，重复10次
        unicode_values = ["\u03c3"] * 10
        # 将列表转换为 NumPy 数组，数据类型为 object
        unicode_values = np.array(unicode_values, dtype=object)
        # 创建一个 DataFrame 对象，包含名为 "unicode" 的列
        df = DataFrame({"unicode": unicode_values})
        # 将 DataFrame 对象转换为字符串表示形式，写入到缓冲区 buf 中，每列宽度为10
        df.to_string(col_space=10, buf=buf)

        # 打印 DataFrame 对象的字符串表示形式，确认功能正常
        repr(df)
        
        # 临时保存 sys.stdin 的引用
        _stdin = sys.stdin
        try:
            # 将 sys.stdin 设置为 None，测试 repr(df) 的行为是否正常
            sys.stdin = None
            repr(df)
        finally:
            # 恢复 sys.stdin 原来的值
            sys.stdin = _stdin

    # 定义一个测试方法，用于测试嵌套 DataFrame 对象的字符串表示
    def test_nested_dataframe(self):
        # 创建第一个 DataFrame 对象，包含名为 "level1" 的列，列值为列表
        df1 = DataFrame({"level1": [["row1"], ["row2"]]})
        # 创建第二个 DataFrame 对象，包含名为 "level3" 的列，列值为包含 df1 的字典
        df2 = DataFrame({"level3": [{"level2": df1}]})
        # 将 df2 对象转换为字符串表示形式，返回结果字符串
        result = df2.to_string()
        # 预期的字符串表示形式，用于断言测试结果是否与预期相符
        expected = "                   level3\n0  {'level2': ['level1']}"
        # 使用断言检查测试结果与预期值是否一致
        assert result == expected
class TestSeriesToString:
    def test_to_string_without_index(self):
        # GH#11729 测试 index=False 选项
        ser = Series([1, 2, 3, 4])
        result = ser.to_string(index=False)
        expected = "\n".join(["1", "2", "3", "4"])
        assert result == expected

    def test_to_string_name(self):
        ser = Series(range(100), dtype="int64")
        ser.name = "myser"
        # 将 Series 转换为字符串，显示前两行，包括名称
        res = ser.to_string(max_rows=2, name=True)
        exp = "0      0\n      ..\n99    99\nName: myser"
        assert res == exp
        # 将 Series 转换为字符串，显示前两行，不包括名称
        res = ser.to_string(max_rows=2, name=False)
        exp = "0      0\n      ..\n99    99"
        assert res == exp

    def test_to_string_dtype(self):
        ser = Series(range(100), dtype="int64")
        # 将 Series 转换为字符串，显示前两行，包括 dtype
        res = ser.to_string(max_rows=2, dtype=True)
        exp = "0      0\n      ..\n99    99\ndtype: int64"
        assert res == exp
        # 将 Series 转换为字符串，显示前两行，不包括 dtype
        res = ser.to_string(max_rows=2, dtype=False)
        exp = "0      0\n      ..\n99    99"
        assert res == exp

    def test_to_string_length(self):
        ser = Series(range(100), dtype="int64")
        # 将 Series 转换为字符串，显示前两行，包括长度信息
        res = ser.to_string(max_rows=2, length=True)
        exp = "0      0\n      ..\n99    99\nLength: 100"
        assert res == exp

    def test_to_string_na_rep(self):
        ser = Series(index=range(100), dtype=np.float64)
        # 将 Series 转换为字符串，显示前两行，使用 'foo' 替代 NaN
        res = ser.to_string(na_rep="foo", max_rows=2)
        exp = "0    foo\n      ..\n99   foo"
        assert res == exp

    def test_to_string_float_format(self):
        ser = Series(range(10), dtype="float64")
        # 将 Series 转换为字符串，显示前两行，使用自定义浮点数格式
        res = ser.to_string(float_format=lambda x: f"{x:2.1f}", max_rows=2)
        exp = "0   0.0\n     ..\n9   9.0"
        assert res == exp

    def test_to_string_header(self):
        ser = Series(range(10), dtype="int64")
        ser.index.name = "foo"
        # 将 Series 转换为字符串，显示前两行，包括列名作为标题
        res = ser.to_string(header=True, max_rows=2)
        exp = "foo\n0    0\n    ..\n9    9"
        assert res == exp
        # 将 Series 转换为字符串，显示前两行，不包括列名作为标题
        res = ser.to_string(header=False, max_rows=2)
        exp = "0    0\n    ..\n9    9"
        assert res == exp

    def test_to_string_empty_col(self):
        # GH#13653 测试空字符串列的情况
        ser = Series(["", "Hello", "World", "", "", "Mooooo", "", ""])
        res = ser.to_string(index=False)
        exp = "      \n Hello\n World\n      \n      \nMooooo\n      \n      "
        assert re.match(exp, res)
    # 定义一个测试方法，用于测试将 Series 对象转换为字符串表示，并使用不同的参数调用 to_string 方法
    def test_to_string_timedelta64(self):
        # 创建一个包含两个 timedelta64 类型元素的 Series 对象，并调用 to_string 方法
        Series(np.array([1100, 20], dtype="timedelta64[ns]")).to_string()

        # 创建一个日期范围为三天的 Series 对象
        ser = Series(date_range("2012-1-1", periods=3, freq="D"))

        # GH#2146

        # 添加 NaT (Not a Time) 值后，计算相邻日期间隔，将结果转换为字符串表示
        y = ser - ser.shift(1)
        result = y.to_string()
        assert "1 days" in result  # 断言结果中包含 "1 days"
        assert "00:00:00" not in result  # 断言结果中不包含 "00:00:00"
        assert "NaT" in result  # 断言结果中包含 "NaT"

        # 添加微秒级别的日期时间后，计算与日期范围之间的差值，将结果转换为字符串表示
        o = Series([datetime(2012, 1, 1, microsecond=150)] * 3)
        y = ser - o
        result = y.to_string()
        assert "-1 days +23:59:59.999850" in result  # 断言结果中包含 "-1 days +23:59:59.999850"

        # 添加带小时的日期时间后，计算与日期范围之间的差值，将结果转换为字符串表示
        o = Series([datetime(2012, 1, 1, 1)] * 3)
        y = ser - o
        result = y.to_string()
        assert "-1 days +23:00:00" in result  # 断言结果中包含 "-1 days +23:00:00"
        assert "1 days 23:00:00" in result  # 断言结果中包含 "1 days 23:00:00"

        # 添加带分钟的日期时间后，计算与日期范围之间的差值，将结果转换为字符串表示
        o = Series([datetime(2012, 1, 1, 1, 1)] * 3)
        y = ser - o
        result = y.to_string()
        assert "-1 days +22:59:00" in result  # 断言结果中包含 "-1 days +22:59:00"
        assert "1 days 22:59:00" in result  # 断言结果中包含 "1 days 22:59:00"

        # 添加带微秒的日期时间后，计算与日期范围之间的差值，将结果转换为字符串表示
        o = Series([datetime(2012, 1, 1, 1, 1, microsecond=150)] * 3)
        y = ser - o
        result = y.to_string()
        assert "-1 days +22:58:59.999850" in result  # 断言结果中包含 "-1 days +22:58:59.999850"
        assert "0 days 22:58:59.999850" in result  # 断言结果中包含 "0 days 22:58:59.999850"

        # 添加负时间间隔后，计算日期范围与调整后日期范围之间的差值，将结果转换为字符串表示
        td = timedelta(minutes=5, seconds=3)
        s2 = Series(date_range("2012-1-1", periods=3, freq="D")) + td
        y = ser - s2
        result = y.to_string()
        assert "-1 days +23:54:57" in result  # 断言结果中包含 "-1 days +23:54:57"

        # 添加微秒级别的时间间隔后，计算日期范围与时间间隔之间的差值，将结果转换为字符串表示
        td = timedelta(microseconds=550)
        s2 = Series(date_range("2012-1-1", periods=3, freq="D")) + td
        y = ser - td
        result = y.to_string()
        assert "2012-01-01 23:59:59.999450" in result  # 断言结果中包含 "2012-01-01 23:59:59.999450"

        # 创建一个包含 timedelta_range 结果的 Series 对象，并将其转换为字符串表示
        td = Series(timedelta_range("1 days", periods=3))
        result = td.to_string()
        assert result == "0   1 days\n1   2 days\n2   3 days"  # 断言结果与预期字符串相等

    # 定义另一个测试方法，测试将 Series 对象转换为字符串表示的功能
    def test_to_string(self):
        # 创建一个包含浮点数的 Series 对象，并设置索引为日期范围
        ts = Series(
            np.arange(10, dtype=np.float64),
            index=date_range("2020-01-01", periods=10, freq="B"),
        )
        # 创建一个 StringIO 缓冲区对象
        buf = StringIO()

        # 调用 Series 对象的 to_string 方法，并将结果保存到变量 s 中
        s = ts.to_string()

        # 使用 buf 参数调用 to_string 方法，并检查返回值为 None，并且 buf 的值与 s 相同
        retval = ts.to_string(buf=buf)
        assert retval is None
        assert buf.getvalue().strip() == s  # 断言 buf 中的值与 s 相同

        # 使用 float_format 参数调用 to_string 方法，将结果按空格分隔后，取第二部分，并进行比较
        format = "%.4f".__mod__
        result = ts.to_string(float_format=format)
        result = [x.split()[1] for x in result.split("\n")[:-1]]
        expected = [format(x) for x in ts]
        assert result == expected  # 断言结果与预期列表相同

        # 创建一个空的 Series 对象切片，并将其转换为字符串表示，断言结果为 "Series([], Freq: B)"
        result = ts[:0].to_string()
        assert result == "Series([], Freq: B)"

        # 创建一个空的 Series 对象切片，并将其转换为字符串表示，断言结果为 "Series([], Freq: B)"
        result = ts[:0].to_string(length=0)
        assert result == "Series([], Freq: B)"

        # 复制 Series 对象并设置名称和长度，并将其转换为字符串表示，断言最后一行的内容与预期相同
        cp = ts.copy()
        cp.name = "foo"
        result = cp.to_string(length=True, name=True, dtype=True)
        last_line = result.split("\n")[-1].strip()
        assert last_line == (f"Freq: B, Name: foo, Length: {len(cp)}, dtype: float64")
    @pytest.mark.parametrize(
        "input_array, expected",
        [   # 参数化测试的参数列表
            ("a", "a"),         # 单个字符串的情况
            (["a", "b"], "a\nb"),   # 字符串列表的情况
            ([1, "a"], "1\na"),     # 包含数字和字符串的情况
            (1, "1"),           # 单个整数的情况
            ([0, -1], " 0\n-1"),    # 包含整数的列表情况
            (1.0, "1.0"),       # 单个浮点数的情况
            ([" a", " b"], " a\n b"),   # 带有前导空格的字符串列表情况
            ([".1", "1"], ".1\n 1"),   # 包含小数的字符串列表情况
            (["10", "-10"], " 10\n-10"),   # 包含正负整数的字符串列表情况
        ],
    )
    def test_format_remove_leading_space_series(self, input_array, expected):
        # GH: 24980
        # 测试移除序列中元素前导空格的功能
        ser = Series(input_array)
        result = ser.to_string(index=False)
        assert result == expected

    def test_to_string_complex_number_trims_zeros(self):
        # 测试复数在转换为字符串时是否修剪末尾的零
        ser = Series([1.000000 + 1.000000j, 1.0 + 1.0j, 1.05 + 1.0j])
        result = ser.to_string()
        expected = dedent(
            """\
            0    1.00+1.00j
            1    1.00+1.00j
            2    1.05+1.00j"""
        )
        assert result == expected

    def test_nullable_float_to_string(self, float_ea_dtype):
        # https://github.com/pandas-dev/pandas/issues/36775
        # 测试可空浮点数转换为字符串的功能
        dtype = float_ea_dtype
        ser = Series([0.0, 1.0, None], dtype=dtype)
        result = ser.to_string()
        expected = dedent(
            """\
            0     0.0
            1     1.0
            2    <NA>"""
        )
        assert result == expected

    def test_nullable_int_to_string(self, any_int_ea_dtype):
        # https://github.com/pandas-dev/pandas/issues/36775
        # 测试可空整数转换为字符串的功能
        dtype = any_int_ea_dtype
        ser = Series([0, 1, None], dtype=dtype)
        result = ser.to_string()
        expected = dedent(
            """\
            0       0
            1       1
            2    <NA>"""
        )
        assert result == expected

    def test_to_string_mixed(self):
        # 测试混合类型数据转换为字符串的功能
        ser = Series(["foo", np.nan, -1.23, 4.56])
        result = ser.to_string()
        expected = "".join(["0     foo\n", "1     NaN\n", "2   -1.23\n", "3    4.56"])
        assert result == expected

        # 但不要将 NA 识别为浮点数
        ser = Series(["foo", np.nan, "bar", "baz"])
        result = ser.to_string()
        expected = "".join(["0    foo\n", "1    NaN\n", "2    bar\n", "3    baz"])
        assert result == expected

        ser = Series(["foo", 5, "bar", "baz"])
        result = ser.to_string()
        expected = "".join(["0    foo\n", "1      5\n", "2    bar\n", "3    baz"])
        assert result == expected

    def test_to_string_float_na_spacing(self):
        # 测试浮点数和 NA 之间的空格间距
        ser = Series([0.0, 1.5678, 2.0, -3.0, 4.0])
        ser[::2] = np.nan

        result = ser.to_string()
        expected = (
            "0       NaN\n"
            "1    1.5678\n"
            "2       NaN\n"
            "3   -3.0000\n"
            "4       NaN"
        )
        assert result == expected
    # 定义测试方法，验证在 DateTimeIndex 上使用 to_string 方法的行为
    def test_to_string_with_datetimeindex(self):
        # 创建一个日期范围为 2013-01-02 开始的六个日期的索引
        index = date_range("20130102", periods=6)
        # 创建一个 Series 对象，将索引设置为上面创建的日期索引，所有值均为 1
        ser = Series(1, index=index)
        # 将 Series 对象转换为字符串表示形式
        result = ser.to_string()
        # 断言字符串 "2013-01-02" 在结果中
        assert "2013-01-02" in result

        # 创建另一个 Series 对象 s2，包含两个值为 2 的元素，索引为一个具体日期和 NaT（Not a Time）
        s2 = Series(2, index=[Timestamp("20130111"), NaT])
        # 将 s2 与之前的 ser 进行连接
        ser = concat([s2, ser])
        # 将连接后的 Series 对象转换为字符串表示形式
        result = ser.to_string()
        # 断言字符串 "NaT" 在结果中，验证 NaT 的显示
        assert "NaT" in result

        # 将 s2 的索引转换为字符串
        result = str(s2.index)
        # 断言字符串 "NaT" 在结果中，验证索引转换后 NaT 的显示
        assert "NaT" in result
```
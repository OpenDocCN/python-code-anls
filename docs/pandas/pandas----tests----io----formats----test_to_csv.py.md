# `D:\src\scipysrc\pandas\pandas\tests\io\formats\test_to_csv.py`

```
# 引入所需的模块和库
import io
import os
import sys
from zipfile import ZipFile  # 从 zipfile 库中导入 ZipFile 类

from _csv import Error  # 从 _csv 模块中导入 Error 异常类
import numpy as np  # 导入 numpy 库，并用 np 别名表示
import pytest  # 导入 pytest 库

import pandas as pd  # 导入 pandas 库，并用 pd 别名表示
from pandas import (  # 从 pandas 中导入多个子模块或类
    DataFrame,  # DataFrame 类
    Index,  # Index 类
    compat,  # compat 模块
)
import pandas._testing as tm  # 导入 pandas._testing 模块，并用 tm 别名表示

# 定义一个测试类 TestToCSV
class TestToCSV:
    # 定义一个测试方法 test_to_csv_with_single_column
    def test_to_csv_with_single_column(self):
        # 引用 GitHub issue 和 Python bug 问题的注释
        # Python 的 CSV 库在 NaN 值位于第一行时会在换行符之前增加额外的 '""'
        # 否则，只添加换行符。此行为不一致已在特定修复中补丁化
        df1 = DataFrame([None, 1])  # 创建一个包含 None 和 1 的 DataFrame
        expected1 = """\
""
1.0
"""
        with tm.ensure_clean("test.csv") as path:  # 保证测试环境干净，创建临时文件 test.csv
            df1.to_csv(path, header=None, index=None)  # 将 DataFrame 写入 CSV 文件
            with open(path, encoding="utf-8") as f:  # 打开 CSV 文件进行读取
                assert f.read() == expected1  # 断言读取的内容与期望值相等

        df2 = DataFrame([1, None])  # 创建另一个包含 1 和 None 的 DataFrame
        expected2 = """\
1.0
""
"""
        with tm.ensure_clean("test.csv") as path:  # 保证测试环境干净，创建临时文件 test.csv
            df2.to_csv(path, header=None, index=None)  # 将 DataFrame 写入 CSV 文件
            with open(path, encoding="utf-8") as f:  # 打开 CSV 文件进行读取
                assert f.read() == expected2  # 断言读取的内容与期望值相等

    # 定义一个测试方法 test_to_csv_default_encoding
    def test_to_csv_default_encoding(self):
        # 引用 GitHub issue
        df = DataFrame({"col": ["AAAAA", "ÄÄÄÄÄ", "ßßßßß", "聞聞聞聞聞"]})  # 创建一个包含特定字符的 DataFrame

        with tm.ensure_clean("test.csv") as path:  # 保证测试环境干净，创建临时文件 test.csv
            # 默认的 to_csv 编码是 utf-8
            df.to_csv(path)  # 将 DataFrame 写入 CSV 文件
            tm.assert_frame_equal(pd.read_csv(path, index_col=0), df)  # 断言读取的 CSV 文件内容与 DataFrame 相等

    # 定义一个测试方法 test_to_csv_quotechar
    def test_to_csv_quotechar(self):
        df = DataFrame({"col": [1, 2]})  # 创建一个包含列名为 "col" 的 DataFrame
        expected = """\
"","col"
"0","1"
"1","2"
"""

        with tm.ensure_clean("test.csv") as path:  # 保证测试环境干净，创建临时文件 test.csv
            df.to_csv(path, quoting=1)  # 使用 quoting 参数为 QUOTE_ALL 将 DataFrame 写入 CSV 文件
            with open(path, encoding="utf-8") as f:  # 打开 CSV 文件进行读取
                assert f.read() == expected  # 断言读取的内容与期望值相等

        expected = """\
$$,$col$
$0$,$1$
$1$,$2$
"""

        with tm.ensure_clean("test.csv") as path:  # 保证测试环境干净，创建临时文件 test.csv
            df.to_csv(path, quoting=1, quotechar="$")  # 使用 quoting 参数为 QUOTE_ALL 和自定义 quotechar 将 DataFrame 写入 CSV 文件
            with open(path, encoding="utf-8") as f:  # 打开 CSV 文件进行读取
                assert f.read() == expected  # 断言读取的内容与期望值相等

        with tm.ensure_clean("test.csv") as path:  # 保证测试环境干净，创建临时文件 test.csv
            with pytest.raises(TypeError, match="quotechar"):  # 使用 pytest 检查 quotechar 参数是否引发 TypeError 异常
                df.to_csv(path, quoting=1, quotechar=None)  # 将 DataFrame 写入 CSV 文件，但 quotechar 参数为 None

    # 定义一个测试方法 test_to_csv_doublequote
    def test_to_csv_doublequote(self):
        df = DataFrame({"col": ['a"a', '"bb"']})  # 创建一个包含特殊字符的 DataFrame
        expected = '''\
"","col"
"0","a""a"
"1","""bb"""
'''

        with tm.ensure_clean("test.csv") as path:  # 保证测试环境干净，创建临时文件 test.csv
            df.to_csv(path, quoting=1, doublequote=True)  # 使用 quoting 参数为 QUOTE_ALL 和 doublequote=True 将 DataFrame 写入 CSV 文件
            with open(path, encoding="utf-8") as f:  # 打开 CSV 文件进行读取
                assert f.read() == expected  # 断言读取的内容与期望值相等

        with tm.ensure_clean("test.csv") as path:  # 保证测试环境干净，创建临时文件 test.csv
            with pytest.raises(Error, match="escapechar"):  # 使用 pytest 检查是否引发 Error 异常，并匹配 escapechar
                df.to_csv(path, doublequote=False)  # 将 DataFrame 写入 CSV 文件，但不设置 escapechar

    # 定义一个测试方法 test_to_csv_escapechar
    def test_to_csv_escapechar(self):
        df = DataFrame({"col": ['a"a', '"bb"']})  # 创建一个包含特殊字符的 DataFrame
        expected = """\
"","col"
"0","a\\"a"
"1","\\"bb\\""
"""
        # 将 DataFrame 写入 CSV 文件，并使用 quoting 参数为 QUOTE_ALL 和 escapechar 为 "\\" 将特殊字符转义
        with tm.ensure_clean("test.csv") as path:  # 保证测试环境干净，创建临时文件 test.csv
            df.to_csv(path, quoting=1, escapechar="\\")  
            with open(path, encoding="utf-8") as f:  # 打开 CSV 文件进行读取
                assert f.read() == expected  # 断言读取的内容与期望值相等
    """

        with tm.ensure_clean("test.csv") as path:  # QUOTE_ALL
            df.to_csv(path, quoting=1, doublequote=False, escapechar="\\")
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

        df = DataFrame({"col": ["a,a", ",bb,"]})
        expected = """\
,col
0,a\\,a
1,\\,bb\\,
"""

        with tm.ensure_clean("test.csv") as path:
            df.to_csv(path, quoting=3, escapechar="\\")  # QUOTE_NONE
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

    def test_csv_to_string(self):
        df = DataFrame({"col": [1, 2]})
        expected_rows = [",col", "0,1", "1,2"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv() == expected

    def test_to_csv_decimal(self):
        # see gh-781
        df = DataFrame({"col1": [1], "col2": ["a"], "col3": [10.1]})

        expected_rows = [",col1,col2,col3", "0,1,a,10.1"]
        expected_default = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv() == expected_default

        expected_rows = [";col1;col2;col3", "0;1;a;10,1"]
        expected_european_excel = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(decimal=",", sep=";") == expected_european_excel

        expected_rows = [",col1,col2,col3", "0,1,a,10.10"]
        expected_float_format_default = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(float_format="%.2f") == expected_float_format_default

        expected_rows = [";col1;col2;col3", "0;1;a;10,10"]
        expected_float_format = tm.convert_rows_list_to_csv_str(expected_rows)
        assert (
            df.to_csv(decimal=",", sep=";", float_format="%.2f")
            == expected_float_format
        )

        # see gh-11553: testing if decimal is taken into account for '0.0'
        df = DataFrame({"a": [0, 1.1], "b": [2.2, 3.3], "c": 1})

        expected_rows = ["a,b,c", "0^0,2^2,1", "1^1,3^3,1"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.to_csv(index=False, decimal="^") == expected

        # same but for an index
        assert df.set_index("a").to_csv(decimal="^") == expected

        # same for a multi-index
        assert df.set_index(["a", "b"]).to_csv(decimal="^") == expected

    def test_to_csv_float_format(self):
        # testing if float_format is taken into account for the index
        # GH 11553
        df = DataFrame({"a": [0, 1], "b": [2.2, 3.3], "c": 1})

        expected_rows = ["a,b,c", "0,2.20,1", "1,3.30,1"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        assert df.set_index("a").to_csv(float_format="%.2f") == expected

        # same for a multi-index
        assert df.set_index(["a", "b"]).to_csv(float_format="%.2f") == expected
    def test_to_csv_na_rep(self):
        # 用例 gh-11553
        #
        # 测试索引中 NaN 值的正确表示。
        # 创建包含三列的 DataFrame
        df = DataFrame({"a": [0, np.nan], "b": [0, 1], "c": [2, 3]})
        # 预期的 CSV 行列表
        expected_rows = ["a,b,c", "0.0,0,2", "_,1,3"]
        # 将预期的行列表转换为 CSV 字符串
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        # 断言：设置索引为 "a" 的 DataFrame 导出 CSV 后是否与预期相同
        assert df.set_index("a").to_csv(na_rep="_") == expected
        # 断言：设置复合索引 ["a", "b"] 的 DataFrame 导出 CSV 后是否与预期相同
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        # 再次测试，这次索引只包含 NaN
        df = DataFrame({"a": np.nan, "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "_,0,2", "_,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        # 检查在没有 NaN 时，na_rep 参数是否不会影响结果
        df = DataFrame({"a": 0, "b": [0, 1], "c": [2, 3]})
        expected_rows = ["a,b,c", "0,0,2", "0,1,3"]
        expected = tm.convert_rows_list_to_csv_str(expected_rows)

        assert df.set_index("a").to_csv(na_rep="_") == expected
        assert df.set_index(["a", "b"]).to_csv(na_rep="_") == expected

        # 测试 Series，包含一个 pd.NA 值，导出 CSV 后 na_rep 参数是否有效
        csv = pd.Series(["a", pd.NA, "c"]).to_csv(na_rep="ZZZZZ")
        expected = tm.convert_rows_list_to_csv_str([",0", "0,a", "1,ZZZZZ", "2,c"])
        assert expected == csv

    def test_to_csv_na_rep_nullable_string(self, nullable_string_dtype):
        # GH 29975
        # 确保提供 dtype 时，完整的 na_rep 能正确显示
        expected = tm.convert_rows_list_to_csv_str([",0", "0,a", "1,ZZZZZ", "2,c"])
        # 使用 nullable_string_dtype 类型创建 Series，并导出 CSV
        csv = pd.Series(["a", pd.NA, "c"], dtype=nullable_string_dtype).to_csv(
            na_rep="ZZZZZ"
        )
        assert expected == csv
    # 测试将日期格式化为 CSV 文件
    def test_to_csv_date_format(self):
        # 创建包含秒级频率日期的 DataFrame
        df_sec = DataFrame({"A": pd.date_range("20130101", periods=5, freq="s")})
        # 创建包含日级频率日期的 DataFrame
        df_day = DataFrame({"A": pd.date_range("20130101", periods=5, freq="d")})

        # 期望的 CSV 行列表
        expected_rows = [
            ",A",
            "0,2013-01-01 00:00:00",
            "1,2013-01-01 00:00:01",
            "2,2013-01-01 00:00:02",
            "3,2013-01-01 00:00:03",
            "4,2013-01-01 00:00:04",
        ]
        # 将期望的 CSV 行列表转换为字符串
        expected_default_sec = tm.convert_rows_list_to_csv_str(expected_rows)
        # 断言秒级频率日期的 DataFrame 转换为 CSV 后与期望结果相同
        assert df_sec.to_csv() == expected_default_sec

        # 更改期望的 CSV 行列表
        expected_rows = [
            ",A",
            "0,2013-01-01 00:00:00",
            "1,2013-01-02 00:00:00",
            "2,2013-01-03 00:00:00",
            "3,2013-01-04 00:00:00",
            "4,2013-01-05 00:00:00",
        ]
        # 将期望的 CSV 行列表转换为字符串
        expected_ymdhms_day = tm.convert_rows_list_to_csv_str(expected_rows)
        # 断言日级频率日期的 DataFrame 转换为 CSV 后与期望结果相同（使用指定日期格式）
        assert df_day.to_csv(date_format="%Y-%m-%d %H:%M:%S") == expected_ymdhms_day

        # 其他类似的测试用例，省略

    # 测试将不同日期时间格式化为 CSV 文件
    def test_to_csv_different_datetime_formats(self):
        # 创建包含日期和日期时间列的 DataFrame
        df = DataFrame(
            {
                "date": pd.to_datetime("1970-01-01"),
                "datetime": pd.date_range("1970-01-01", periods=2, freq="h"),
            }
        )
        # 期望的 CSV 行列表
        expected_rows = [
            "date,datetime",
            "1970-01-01,1970-01-01 00:00:00",
            "1970-01-01,1970-01-01 01:00:00",
        ]
        # 将期望的 CSV 行列表转换为字符串
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        # 断言 DataFrame 转换为 CSV 后与期望结果相同（不包括索引）
        assert df.to_csv(index=False) == expected
    def test_to_csv_date_format_in_categorical(self):
        # GH#40754
        # 创建一个包含日期和缺失值的 Pandas Series
        ser = pd.Series(pd.to_datetime(["2021-03-27", pd.NaT], format="%Y-%m-%d"))
        # 将 Series 转换为分类类型
        ser = ser.astype("category")
        # 期望的 CSV 字符串结果
        expected = tm.convert_rows_list_to_csv_str(["0", "2021-03-27", '""'])
        # 断言 Series 转为 CSV 字符串是否与期望的相同
        assert ser.to_csv(index=False) == expected

        # 创建包含日期范围和缺失值的 Pandas Series
        ser = pd.Series(
            pd.date_range(
                start="2021-03-27", freq="D", periods=1, tz="Europe/Berlin"
            ).append(pd.DatetimeIndex([pd.NaT]))
        )
        # 将 Series 转换为分类类型
        ser = ser.astype("category")
        # 断言 Series 转为 CSV 字符串是否与期望的相同，设置日期格式为 "%Y-%m-%d"
        assert ser.to_csv(index=False, date_format="%Y-%m-%d") == expected

    def test_to_csv_float_ea_float_format(self):
        # GH#45991
        # 创建一个包含浮点数和缺失值的 DataFrame
        df = DataFrame({"a": [1.1, 2.02, pd.NA, 6.000006], "b": "c"})
        # 将 DataFrame 列 "a" 转换为 Float64 类型
        df["a"] = df["a"].astype("Float64")
        # 将 DataFrame 转为 CSV 字符串，设置浮点数格式为 "%.5f"
        result = df.to_csv(index=False, float_format="%.5f")
        # 期望的 CSV 字符串结果
        expected = tm.convert_rows_list_to_csv_str(
            ["a,b", "1.10000,c", "2.02000,c", ",c", "6.00001,c"]
        )
        # 断言转换后的结果是否与期望的相同
        assert result == expected

    def test_to_csv_float_ea_no_float_format(self):
        # GH#45991
        # 创建一个包含浮点数和缺失值的 DataFrame
        df = DataFrame({"a": [1.1, 2.02, pd.NA, 6.000006], "b": "c"})
        # 将 DataFrame 列 "a" 转换为 Float64 类型
        df["a"] = df["a"].astype("Float64")
        # 将 DataFrame 转为 CSV 字符串，未设置浮点数格式
        result = df.to_csv(index=False)
        # 期望的 CSV 字符串结果
        expected = tm.convert_rows_list_to_csv_str(
            ["a,b", "1.1,c", "2.02,c", ",c", "6.000006,c"]
        )
        # 断言转换后的结果是否与期望的相同
        assert result == expected

    def test_to_csv_multi_index(self):
        # see gh-6618
        # 创建一个包含多级索引的 DataFrame
        df = DataFrame([1], columns=pd.MultiIndex.from_arrays([[1], [2]]))

        # 期望的 CSV 字符串结果，包含行索引
        exp_rows = [",1", ",2", "0,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        # 断言转换后的结果是否与期望的相同
        assert df.to_csv() == exp

        # 期望的 CSV 字符串结果，不包含行索引
        exp_rows = ["1", "2", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        # 断言转换后的结果是否与期望的相同
        assert df.to_csv(index=False) == exp

        # 创建一个包含多级索引和行索引的 DataFrame
        df = DataFrame(
            [1],
            columns=pd.MultiIndex.from_arrays([[1], [2]]),
            index=pd.MultiIndex.from_arrays([[1], [2]]),
        )

        # 期望的 CSV 字符串结果，包含行和列索引
        exp_rows = [",,1", ",,2", "1,2,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        # 断言转换后的结果是否与期望的相同
        assert df.to_csv() == exp

        # 期望的 CSV 字符串结果，不包含行和列索引
        exp_rows = ["1", "2", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        # 断言转换后的结果是否与期望的相同
        assert df.to_csv(index=False) == exp

        # 创建一个包含具名多级索引的 DataFrame
        df = DataFrame([1], columns=pd.MultiIndex.from_arrays([["foo"], ["bar"]]))

        # 期望的 CSV 字符串结果，包含行索引
        exp_rows = [",foo", ",bar", "0,1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        # 断言转换后的结果是否与期望的相同
        assert df.to_csv() == exp

        # 期望的 CSV 字符串结果，不包含行索引
        exp_rows = ["foo", "bar", "1"]
        exp = tm.convert_rows_list_to_csv_str(exp_rows)
        # 断言转换后的结果是否与期望的相同
        assert df.to_csv(index=False) == exp
    @pytest.mark.parametrize(
        "ind,expected",
        [  # 参数化测试用例，定义了多组输入参数和对应的期望输出
            (
                pd.MultiIndex(levels=[[1.0]], codes=[[0]], names=["x"]),
                "x,data\n1.0,1\n",  # 第一组输入参数对应的期望输出字符串
            ),
            (
                pd.MultiIndex(
                    levels=[[1.0], [2.0]], codes=[[0], [0]], names=["x", "y"]
                ),
                "x,y,data\n1.0,2.0,1\n",  # 第二组输入参数对应的期望输出字符串
            ),
        ],
    )
    def test_to_csv_single_level_multi_index(self, ind, expected, frame_or_series):
        # see gh-19589
        # 函数注释: 单级多索引情况下，测试对象的to_csv方法
        obj = frame_or_series(pd.Series([1], ind, name="data"))

        # 调用对象的to_csv方法，生成CSV格式的结果字符串，使用\n作为行终止符，包含表头
        result = obj.to_csv(lineterminator="\n", header=True)
        # 断言生成的结果与期望的输出字符串相等
        assert result == expected

    def test_to_csv_string_array_ascii(self):
        # GH 10813
        # 函数注释: 字符串数组转换为DataFrame对象的ASCII格式输出测试，参见GitHub issue 10813
        str_array = [{"names": ["foo", "bar"]}, {"names": ["baz", "qux"]}]
        # 使用字符串数组创建DataFrame对象
        df = DataFrame(str_array)
        expected_ascii = """\
    def test_to_csv_string_array_utf8(self):
        # GH 10813
        # 准备包含字符串数组的字典列表
        str_array = [{"names": ["foo", "bar"]}, {"names": ["baz", "qux"]}]
        # 创建 DataFrame 对象
        df = DataFrame(str_array)
        # 预期的 UTF-8 编码结果
        expected_utf8 = """\
,names
0,"['foo', 'bar']"
1,"['baz', 'qux']"
"""
        # 使用 tm.ensure_clean 确保路径干净，并在其上下文中执行操作
        with tm.ensure_clean("unicode_test.csv") as path:
            # 将 DataFrame 写入 CSV 文件，使用 UTF-8 编码
            df.to_csv(path, encoding="utf-8")
            # 打开文件并断言读取的内容与预期的 UTF-8 结果相符
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected_utf8

    def test_to_csv_string_with_lf(self):
        # GH 20353
        # 准备包含不同行结束符情况的数据字典
        data = {"int": [1, 2, 3], "str_lf": ["abc", "d\nef", "g\nh\n\ni"]}
        # 创建 DataFrame 对象
        df = DataFrame(data)
        # 使用 tm.ensure_clean 确保路径干净，并在其上下文中执行操作
        with tm.ensure_clean("lf_test.csv") as path:
            # case 1: 默认行结束符情况下的预期结果
            os_linesep = os.linesep.encode("utf-8")
            expected_noarg = (
                b"int,str_lf"
                + os_linesep
                + b"1,abc"
                + os_linesep
                + b'2,"d\nef"'
                + os_linesep
                + b'3,"g\nh\n\ni"'
                + os_linesep
            )
            # 将 DataFrame 写入 CSV 文件，不使用显式的行结束符
            df.to_csv(path, index=False)
            # 打开文件并断言读取的内容与预期的结果相符
            with open(path, "rb") as f:
                assert f.read() == expected_noarg
        with tm.ensure_clean("lf_test.csv") as path:
            # case 2: 使用 LF 作为行结束符的预期结果
            expected_lf = b'int,str_lf\n1,abc\n2,"d\nef"\n3,"g\nh\n\ni"\n'
            # 将 DataFrame 写入 CSV 文件，使用 LF 作为行结束符
            df.to_csv(path, lineterminator="\n", index=False)
            # 打开文件并断言读取的内容与预期的 LF 结果相符
            with open(path, "rb") as f:
                assert f.read() == expected_lf
        with tm.ensure_clean("lf_test.csv") as path:
            # case 3: 使用 CRLF 作为行结束符的预期结果
            # 'lineterminator' 不应改变内部元素
            expected_crlf = b'int,str_lf\r\n1,abc\r\n2,"d\nef"\r\n3,"g\nh\n\ni"\r\n'
            # 将 DataFrame 写入 CSV 文件，使用 CRLF 作为行结束符
            df.to_csv(path, lineterminator="\r\n", index=False)
            # 打开文件并断言读取的内容与预期的 CRLF 结果相符
            with open(path, "rb") as f:
                assert f.read() == expected_crlf
    def test_to_csv_string_with_crlf(self):
        # GH 20353
        # 准备测试数据，包括整型和包含换行符的字符串列表
        data = {"int": [1, 2, 3], "str_crlf": ["abc", "d\r\nef", "g\r\nh\r\n\r\ni"]}
        # 创建数据框
        df = DataFrame(data)
        # 使用临时文件路径来确保测试环境干净
        with tm.ensure_clean("crlf_test.csv") as path:
            # case 1: 默认行终止符（=os.linesep）(PR 21406)
            os_linesep = os.linesep.encode("utf-8")
            # 生成预期的字节流结果，包括每行数据和行终止符
            expected_noarg = (
                b"int,str_crlf"
                + os_linesep
                + b"1,abc"
                + os_linesep
                + b'2,"d\r\nef"'
                + os_linesep
                + b'3,"g\r\nh\r\n\r\ni"'
                + os_linesep
            )
            # 将数据框内容写入到CSV文件中
            df.to_csv(path, index=False)
            # 打开CSV文件并检查内容是否符合预期
            with open(path, "rb") as f:
                assert f.read() == expected_noarg
        with tm.ensure_clean("crlf_test.csv") as path:
            # case 2: LF 作为行终止符
            expected_lf = b'int,str_crlf\n1,abc\n2,"d\r\nef"\n3,"g\r\nh\r\n\r\ni"\n'
            # 将数据框内容写入到CSV文件中，指定LF作为行终止符
            df.to_csv(path, lineterminator="\n", index=False)
            # 打开CSV文件并检查内容是否符合预期
            with open(path, "rb") as f:
                assert f.read() == expected_lf
        with tm.ensure_clean("crlf_test.csv") as path:
            # case 3: CRLF 作为行终止符
            # 'lineterminator' 应不影响内部元素
            expected_crlf = (
                b"int,str_crlf\r\n"
                b"1,abc\r\n"
                b'2,"d\r\nef"\r\n'
                b'3,"g\r\nh\r\n\r\ni"\r\n'
            )
            # 将数据框内容写入到CSV文件中，指定CRLF作为行终止符
            df.to_csv(path, lineterminator="\r\n", index=False)
            # 打开CSV文件并检查内容是否符合预期
            with open(path, "rb") as f:
                assert f.read() == expected_crlf

    def test_to_csv_stdout_file(self, capsys):
        # GH 21561
        # 创建包含两行的数据框，每行有两列
        df = DataFrame([["foo", "bar"], ["baz", "qux"]], columns=["name_1", "name_2"])
        # 生成预期的CSV格式字符串，使用ASCII编码
        expected_rows = [",name_1,name_2", "0,foo,bar", "1,baz,qux"]
        expected_ascii = tm.convert_rows_list_to_csv_str(expected_rows)

        # 将数据框内容输出到标准输出，使用ASCII编码
        df.to_csv(sys.stdout, encoding="ascii")
        # 捕获标准输出内容
        captured = capsys.readouterr()

        # 断言捕获的标准输出与预期的ASCII格式字符串相符
        assert captured.out == expected_ascii
        assert not sys.stdout.closed

    @pytest.mark.xfail(
        compat.is_platform_windows(),
        reason=(
            "特别是在Windows中，不应在不带 newline='' 选项的情况下将文件流传递给csv writer。"
            "(https://docs.python.org/3/library/csv.html#csv.writer)"
        ),
    )
    def test_to_csv_write_to_open_file(self):
        # GH 21696
        # 创建包含单列的数据框
        df = DataFrame({"a": ["x", "y", "z"]})
        # 预期输出为空字符串
        expected = ""
    def test_to_csv_write_to_open_file_with_newline_py3(self):
        # see gh-21696
        # see gh-20353
        df = DataFrame({"a": ["x", "y", "z"]})
        expected_rows = ["x", "y", "z"]
        expected = "manual header\n" + tm.convert_rows_list_to_csv_str(expected_rows)
        
        # 创建一个临时文件并确保其在使用后被清理
        with tm.ensure_clean("test.txt") as path:
            # 打开文件以写入，指定编码为 UTF-8，并写入手动添加的标题
            with open(path, "w", encoding="utf-8") as f:
                f.write("manual header\n")
                # 将 DataFrame 内容以 CSV 格式写入到文件中，不包括标题和索引
                df.to_csv(f, header=None, index=None)
            
            # 再次打开文件以读取并验证写入的内容是否符合预期
            with open(path, encoding="utf-8") as f:
                assert f.read() == expected

    @pytest.mark.parametrize("to_infer", [True, False])
    @pytest.mark.parametrize("read_infer", [True, False])
    def test_to_csv_compression(
        self, compression_only, read_infer, to_infer, compression_to_extension
    ):
        # see gh-15008
        compression = compression_only
        
        # 根据压缩选项确定文件名后缀
        filename = "test."
        filename += compression_to_extension[compression]

        df = DataFrame({"A": [1]})
        
        # 确定写入和读取时的压缩选项
        to_compression = "infer" if to_infer else compression
        read_compression = "infer" if read_infer else compression
        
        # 创建一个临时文件并确保其在使用后被清理
        with tm.ensure_clean(filename) as path:
            # 将 DataFrame 内容以 CSV 格式写入到文件中，指定压缩选项
            df.to_csv(path, compression=to_compression)
            # 读取并验证写入的文件内容是否与原始 DataFrame 相等
            result = pd.read_csv(path, index_col=0, compression=read_compression)
            tm.assert_frame_equal(result, df)

    def test_to_csv_compression_dict(self, compression_only):
        # GH 26023
        method = compression_only
        df = DataFrame({"ABC": [1]})
        
        # 根据压缩选项确定文件名后缀
        filename = "to_csv_compress_as_dict."
        extension = {
            "gzip": "gz",
            "zstd": "zst",
        }.get(method, method)
        filename += extension
        
        # 创建一个临时文件并确保其在使用后被清理
        with tm.ensure_clean(filename) as path:
            # 将 DataFrame 内容以 CSV 格式写入到文件中，指定压缩方法为字典形式
            df.to_csv(path, compression={"method": method})
            # 读取并验证写入的文件内容是否与原始 DataFrame 相等
            read_df = pd.read_csv(path, index_col=0)
            tm.assert_frame_equal(read_df, df)

    def test_to_csv_compression_dict_no_method_raises(self):
        # GH 26023
        df = DataFrame({"ABC": [1]})
        compression = {"some_option": True}
        msg = "must have key 'method'"
        
        # 创建一个临时文件并确保其在使用后被清理
        with tm.ensure_clean("out.zip") as path:
            # 使用无效的压缩选项，预期会引发 ValueError 异常
            with pytest.raises(ValueError, match=msg):
                df.to_csv(path, compression=compression)

    @pytest.mark.parametrize("compression", ["zip", "infer"])
    @pytest.mark.parametrize("archive_name", ["test_to_csv.csv", "test_to_csv.zip"])
    def test_to_csv_zip_arguments(self, compression, archive_name):
        # 测试函数：test_to_csv_zip_arguments
        # 参数：compression - 压缩方法，archive_name - 压缩文件名
        # GH 26023
        
        # 创建一个包含单列"ABC"的DataFrame对象
        df = DataFrame({"ABC": [1]})
        
        # 使用临时路径确保写入的文件路径干净
        with tm.ensure_clean("to_csv_archive_name.zip") as path:
            # 将DataFrame对象保存为CSV文件，并设置压缩选项为给定的compression和archive_name
            df.to_csv(
                path, compression={"method": compression, "archive_name": archive_name}
            )
            
            # 打开生成的ZIP文件
            with ZipFile(path) as zp:
                # 断言ZIP文件中包含的文件数为1
                assert len(zp.filelist) == 1
                # 获取压缩后的文件名
                archived_file = zp.filelist[0].filename
                # 断言压缩后的文件名与预期的archive_name相同
                assert archived_file == archive_name

    @pytest.mark.parametrize(
        "filename,expected_arcname",
        [
            ("archive.csv", "archive.csv"),
            ("archive.tsv", "archive.tsv"),
            ("archive.csv.zip", "archive.csv"),
            ("archive.tsv.zip", "archive.tsv"),
            ("archive.zip", "archive"),
        ],
    )
    def test_to_csv_zip_infer_name(self, tmp_path, filename, expected_arcname):
        # 测试函数：test_to_csv_zip_infer_name
        # 参数：tmp_path - 临时路径，filename - 文件名，expected_arcname - 预期的压缩文件名
        # GH 39465
        
        # 创建一个包含单列"ABC"的DataFrame对象
        df = DataFrame({"ABC": [1]})
        
        # 生成文件的完整路径
        path = tmp_path / filename
        
        # 将DataFrame对象保存为ZIP压缩文件
        df.to_csv(path, compression="zip")
        
        # 打开生成的ZIP文件
        with ZipFile(path) as zp:
            # 断言ZIP文件中包含的文件数为1
            assert len(zp.filelist) == 1
            # 获取压缩后的文件名
            archived_file = zp.filelist[0].filename
            # 断言压缩后的文件名与预期的expected_arcname相同
            assert archived_file == expected_arcname

    @pytest.mark.parametrize("df_new_type", ["Int64"])
    def test_to_csv_na_rep_long_string(self, df_new_type):
        # 测试函数：test_to_csv_na_rep_long_string
        # 参数：df_new_type - DataFrame的新类型
        # see gh-25099
        
        # 创建一个包含NaN值的DataFrame对象
        df = DataFrame({"c": [float("nan")] * 3})
        
        # 将DataFrame的数据类型转换为指定的df_new_type类型
        df = df.astype(df_new_type)
        
        # 期望的CSV文件行列表
        expected_rows = ["c", "mynull", "mynull", "mynull"]
        
        # 将预期的行列表转换为CSV格式的字符串
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        
        # 将DataFrame对象保存为CSV文件，将NaN值表示为"mynull"，使用ASCII编码
        result = df.to_csv(index=False, na_rep="mynull", encoding="ascii")
        
        # 断言预期的CSV字符串与结果字符串相同
        assert expected == result

    def test_to_csv_timedelta_precision(self):
        # 测试函数：test_to_csv_timedelta_precision
        # GH 6783
        
        # 创建一个包含时间差数据的Series对象
        s = pd.Series([1, 1]).astype("timedelta64[ns]")
        
        # 创建一个StringIO对象作为缓冲区
        buf = io.StringIO()
        
        # 将Series对象的数据保存为CSV格式，并写入缓冲区
        s.to_csv(buf)
        
        # 获取缓冲区中的数据
        result = buf.getvalue()
        
        # 期望的CSV文件行列表
        expected_rows = [
            ",0",
            "0,0 days 00:00:00.000000001",
            "1,0 days 00:00:00.000000001",
        ]
        
        # 将预期的行列表转换为CSV格式的字符串
        expected = tm.convert_rows_list_to_csv_str(expected_rows)
        
        # 断言结果字符串与预期的CSV字符串相同
        assert result == expected

    def test_na_rep_truncated(self):
        # 测试函数：test_na_rep_truncated
        # https://github.com/pandas-dev/pandas/issues/31447
        
        # 将包含整数范围的Series对象保存为CSV格式，将缺失值表示为"-"
        result = pd.Series(range(8, 12)).to_csv(na_rep="-")
        
        # 期望的CSV文件行列表
        expected = tm.convert_rows_list_to_csv_str([",0", "0,8", "1,9", "2,10", "3,11"])
        
        # 断言结果字符串与预期的CSV字符串相同
        assert result == expected

        # 将包含布尔值的Series对象保存为CSV格式，将缺失值表示为"nan"
        result = pd.Series([True, False]).to_csv(na_rep="nan")
        
        # 期望的CSV文件行列表
        expected = tm.convert_rows_list_to_csv_str([",0", "0,True", "1,False"])
        
        # 断言结果字符串与预期的CSV字符串相同
        assert result == expected

        # 将包含浮点数的Series对象保存为CSV格式，将缺失值表示为"."
        result = pd.Series([1.1, 2.2]).to_csv(na_rep=".")
        
        # 期望的CSV文件行列表
        expected = tm.convert_rows_list_to_csv_str([",0", "0,1.1", "1,2.2"])
        
        # 断言结果字符串与预期的CSV字符串相同
        assert result == expected

    @pytest.mark.parametrize("errors", ["surrogatepass", "ignore", "replace"])
    def test_to_csv_errors(self, errors):
        # GH 22610
        # 创建包含特定 Unicode 错误字符的数据列表
        data = ["\ud800foo"]
        # 使用数据列表创建 Pandas Series 对象，设置索引和数据类型为对象
        ser = pd.Series(data, index=Index(data, dtype=object), dtype=object)
        # 在确保路径干净的情况下，将 Series 对象写入 CSV 文件
        with tm.ensure_clean("test.csv") as path:
            ser.to_csv(path, errors=errors)
        # 由于错误处理的存在，无需读回数据，因为数据已不同

    @pytest.mark.parametrize("mode", ["wb", "w"])
    def test_to_csv_binary_handle(self, mode):
        """
        Binary file objects should work (if 'mode' contains a 'b') or even without
        it in most cases.

        GH 35058 and GH 19827
        """
        # 创建一个包含浮点数数据的 DataFrame 对象
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD")),
            index=Index([f"i-{i}" for i in range(30)]),
        )
        # 在确保路径干净的情况下，使用指定模式打开文件句柄，写入 DataFrame 到 CSV 文件
        with tm.ensure_clean() as path:
            with open(path, mode="w+b") as handle:
                df.to_csv(handle, mode=mode)
            # 断言写入的 CSV 文件内容与原 DataFrame 内容一致
            tm.assert_frame_equal(df, pd.read_csv(path, index_col=0))

    @pytest.mark.parametrize("mode", ["wb", "w"])
    def test_to_csv_encoding_binary_handle(self, mode):
        """
        Binary file objects should honor a specified encoding.

        GH 23854 and GH 13068 with binary handles
        """
        # 示例来自 GH 23854，创建包含特定编码的字节内容
        content = "a, b, 🐟".encode("utf-8-sig")
        # 使用字节内容创建 BytesIO 对象
        buffer = io.BytesIO(content)
        # 从 BytesIO 对象读取数据到 DataFrame
        df = pd.read_csv(buffer, encoding="utf-8-sig")

        buffer = io.BytesIO()
        # 将 DataFrame 写入到 CSV 格式的字节流中，使用指定的模式和编码
        df.to_csv(buffer, mode=mode, encoding="utf-8-sig", index=False)
        buffer.seek(0)  # 检查文件句柄未关闭
        assert buffer.getvalue().startswith(content)

        # 示例来自 GH 13068，在确保路径干净的情况下，使用指定模式打开文件句柄
        with tm.ensure_clean() as path:
            with open(path, "w+b") as handle:
                # 将空 DataFrame 对象写入到 CSV 文件中，使用指定的模式和编码
                DataFrame().to_csv(handle, mode=mode, encoding="utf-8-sig")

                handle.seek(0)
                assert handle.read().startswith(b'\xef\xbb\xbf""')
# GH 38714
# 创建一个 DataFrame，包含从 0 到 119 的浮点数，reshape 成 30 行 4 列的形式，
# 列名为 "ABCD"，行名为 "i-0" 到 "i-29"
df = DataFrame(
    1.1 * np.arange(120).reshape((30, 4)),
    columns=Index(list("ABCD")),
    index=Index([f"i-{i}" for i in range(30)]),
)

# 在一个临时路径中保证清洁的上下文管理器
with tm.ensure_clean() as path:
    # 将 DataFrame 写入 CSV 文件，使用指定的压缩方式和每次写入的行数（chunksize=1）
    df.to_csv(path, compression=compression, chunksize=1)
    # 使用 pandas 读取该 CSV 文件，并断言读取的 DataFrame 与原始的 df 相等
    tm.assert_frame_equal(
        pd.read_csv(path, compression=compression, index_col=0), df
    )


# GH 38714
# 创建一个 DataFrame，包含从 0 到 119 的浮点数，reshape 成 30 行 4 列的形式，
# 列名为 "ABCD"，行名为 "i-0" 到 "i-29"
df = DataFrame(
    1.1 * np.arange(120).reshape((30, 4)),
    columns=Index(list("ABCD")),
    index=Index([f"i-{i}" for i in range(30)]),
)

# 使用内存中的字节流作为临时缓冲
with io.BytesIO() as buffer:
    # 将 DataFrame 写入 CSV 格式的数据缓冲区中，使用指定的压缩方式和每次写入的行数（chunksize=1）
    df.to_csv(buffer, compression=compression, chunksize=1)
    # 将读取位置调整到缓冲区的开头
    buffer.seek(0)
    # 使用 pandas 读取该 CSV 数据缓冲区，并断言读取的 DataFrame 与原始的 df 相等
    tm.assert_frame_equal(
        pd.read_csv(buffer, compression=compression, index_col=0), df
    )
    # 断言缓冲区没有关闭
    assert not buffer.closed
```
# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_skiprows.py`

```
"""
Tests that skipped rows are properly handled during
parsing for all of the parsers defined in parsers.py
"""

# 导入所需的模块和库
from datetime import datetime  # 导入 datetime 类用于日期时间操作
from io import StringIO  # 导入 StringIO 用于在内存中操作文本数据

import numpy as np  # 导入 numpy 库用于数值计算
import pytest  # 导入 pytest 用于单元测试

from pandas.errors import EmptyDataError  # 导入 pandas 错误类 EmptyDataError
from pandas import (  # 导入 pandas 的 DataFrame 和 Index 类
    DataFrame,
    Index,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块 pandas._testing

# 标记测试为在 pyarrow 下预期失败
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
# 设置 pytest 标记来忽略特定的警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

@xfail_pyarrow  # 标记为预期在 pyarrow 下失败的测试用例
@pytest.mark.parametrize("skiprows", [list(range(6)), 6])
def test_skip_rows_bug(all_parsers, skiprows):
    # see gh-505
    # 选择要测试的解析器
    parser = all_parsers
    # 准备包含数据和注释行的文本数据
    text = """#foo,a,b,c
#foo,a,b,c
#foo,a,b,c
#foo,a,b,c
#foo,a,b,c
#foo,a,b,c
1/1/2000,1.,2.,3.
1/2/2000,4,5,6
1/3/2000,7,8,9
"""
    # 使用解析器读取 CSV 数据，并跳过指定的行数
    result = parser.read_csv(
        StringIO(text), skiprows=skiprows, header=None, index_col=0, parse_dates=True
    )
    # 创建预期的日期索引
    index = Index(
        [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
        dtype="M8[s]",
        name=0,
    )
    # 创建预期的 DataFrame
    expected = DataFrame(
        np.arange(1.0, 10.0).reshape((3, 3)), columns=[1, 2, 3], index=index
    )
    # 断言结果与预期是否相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 标记为预期在 pyarrow 下失败的测试用例
def test_deep_skip_rows(all_parsers):
    # see gh-4382
    # 选择要测试的解析器
    parser = all_parsers
    # 准备包含数据的文本数据
    data = "a,b,c\n" + "\n".join(
        [",".join([str(i), str(i + 1), str(i + 2)]) for i in range(10)]
    )
    # 准备被压缩后的数据
    condensed_data = "a,b,c\n" + "\n".join(
        [",".join([str(i), str(i + 1), str(i + 2)]) for i in [0, 1, 2, 3, 4, 6, 8, 9]]
    )
    # 使用解析器分别读取原始数据和被压缩后的数据
    result = parser.read_csv(StringIO(data), skiprows=[6, 8])
    condensed_result = parser.read_csv(StringIO(condensed_data))
    # 断言结果与预期是否相等
    tm.assert_frame_equal(result, condensed_result)


@xfail_pyarrow  # 标记为预期在 pyarrow 下失败的测试用例
def test_skip_rows_blank(all_parsers):
    # see gh-9832
    # 选择要测试的解析器
    parser = all_parsers
    # 准备包含空行和数据的文本数据
    text = """#foo,a,b,c
#foo,a,b,c

#foo,a,b,c
#foo,a,b,c

1/1/2000,1.,2.,3.
1/2/2000,4,5,6
1/3/2000,7,8,9
"""
    # 使用解析器读取 CSV 数据，并跳过指定的行数
    data = parser.read_csv(
        StringIO(text), skiprows=6, header=None, index_col=0, parse_dates=True
    )
    # 创建预期的日期索引
    index = Index(
        [datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)],
        dtype="M8[s]",
        name=0,
    )
    # 创建预期的 DataFrame
    expected = DataFrame(
        np.arange(1.0, 10.0).reshape((3, 3)), columns=[1, 2, 3], index=index
    )
    # 断言结果与预期是否相等
    tm.assert_frame_equal(data, expected)


@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            """id,text,num_lines
1,"line 11
line 12",2
2,"line 21
line 22",2
""",
            {},
            DataFrame({"id": [1, 2], "text": ["line 11\nline 12", "line 21\nline 22"], "num_lines": [2, 2]}),
        )
    ]
)
def test_skip_rows_multiline(all_parsers, data, kwargs, expected):
    # see gh-11029
    # 选择要测试的解析器
    parser = all_parsers
    # 使用解析器读取 CSV 数据
    result = parser.read_csv(StringIO(data), **kwargs)
    # 断言结果与预期是否相等
    tm.assert_frame_equal(result, expected)
# 声明一个测试函数，用于测试读取 CSV 文件时跳过指定行的功能
@pytest.mark.parametrize(
    "data,exp_data",
    [
        (
            """id,text,num_lines
1,"line \n'11' line 12",2
2,"line \n'21' line 22",2
3,"line \n'31' line 32",1""",
            [[2, "line \n'21' line 22", 2], [3, "line \n'31' line 32", 1]],
        ),
        (
            """id,text,num_lines
1,"line '11\n' line 12",2
2,"line '21\n' line 22",2
3,"line '31\n' line 32",1""",
            [[2, "line '21\n' line 22", 2], [3, "line '31\n' line 32", 1]],
        ),
        (
            """id,text,num_lines
1,"line '11\n' \r\tline 12",2
2,"line '21\n' \r\tline 22",2
3,"line '31\n' \r\tline 32",1""",
            [[2, "line '21\n' \r\tline 22", 2], [3, "line '31\n' \r\tline 32", 1]],
        ),
    ],
)
@xfail_pyarrow  # 标记为预期失败，因为 skiprows 参数必须是整数
def test_skip_row_with_newline_and_quote(all_parsers, data, exp_data):
    # 解析器对象
    parser = all_parsers
    # 使用给定的数据字符串创建一个 CSV 数据流，并跳过指定的行（第二行）
    result = parser.read_csv(StringIO(data), skiprows=[1])

    # 预期的 DataFrame 结果
    expected = DataFrame(exp_data, columns=["id", "text", "num_lines"])
    # 断言实际输出和预期输出是否相等
    tm.assert_frame_equal(result, expected)
    # 设置解析器为所有解析器
    parser = all_parsers
    # 定义包含多行数据的字符串
    data = "\n".join(
        [
            "SMOSMANIA ThetaProbe-ML2X ",
            "2007/01/01 01:00   0.2140 U M ",
            "2007/01/01 02:00   0.2141 M O ",
            "2007/01/01 04:00   0.2142 D M ",
        ]
    )
    # 预期的数据框架，包含多行数据
    expected = DataFrame(
        [
            ["2007/01/01", "01:00", 0.2140, "U", "M"],
            ["2007/01/01", "02:00", 0.2141, "M", "O"],
            ["2007/01/01", "04:00", 0.2142, "D", "M"],
        ],
        columns=["date", "time", "var", "flag", "oflag"],
    )

    # 如果解析器的引擎是Python，并且行终止符为'\r'
    if parser.engine == "python" and lineterminator == "\r":
        # 标记该测试为预期失败，原因是"Python 解析器尚未支持 'CR' 行终止符"
        mark = pytest.mark.xfail(reason="'CR' not respect with the Python parser yet")
        # 应用预期失败标记到测试请求中
        request.applymarker(mark)

    # 将数据中的换行符替换为指定的行终止符
    data = data.replace("\n", lineterminator)

    # 使用解析器读取 CSV 数据，设置跳过第一行、分隔符为多个空白字符，并指定列名
    result = parser.read_csv(
        StringIO(data),
        skiprows=1,
        sep=r"\s+",
        names=["date", "time", "var", "flag", "oflag"],
    )
    # 断言读取的结果与预期的数据框架相等
    tm.assert_frame_equal(result, expected)
@xfail_pyarrow  # 表示此测试预期失败，因为会出现特定的异常
def test_skiprows_infield_quote(all_parsers):
    # 测试用例名称：test_skiprows_infield_quote，用于测试在字段引号中跳过行
    # see gh-14459  # 相关GitHub问题号，指向测试用例相关的讨论或问题
    parser = all_parsers  # 获取所有解析器的实例
    data = 'a"\nb"\na\n1'  # 测试数据，包含特定的换行和引号情况
    expected = DataFrame({"a": [1]})  # 预期的DataFrame输出，包含一个列名为'a'的列和值为[1]的数据

    # 执行CSV解析，跳过前两行，并将结果存储在result中
    result = parser.read_csv(StringIO(data), skiprows=2)
    # 使用测试工具比较result和expected，确保它们相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 表示此测试预期失败，因为会出现特定的异常
@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, "1"),  # 参数组合1：空字典，预期结果为"1"
        ({"header": 0, "names": ["foo"]}, "foo"),  # 参数组合2：header为0，names为["foo"]，预期结果为"foo"
    ],
)
def test_skip_rows_callable(all_parsers, kwargs, expected):
    # 测试用例名称：test_skip_rows_callable，测试可调用的skiprows参数
    parser = all_parsers  # 获取所有解析器的实例
    data = "a\n1\n2\n3\n4\n5"  # 测试数据，包含多行数字和字母
    # 预期的DataFrame输出，包含一个列名为expected的列和对应的数字列表
    expected = DataFrame({expected: [3, 5]})

    # 执行CSV解析，使用lambda函数跳过偶数行，并将结果存储在result中
    result = parser.read_csv(StringIO(data), skiprows=lambda x: x % 2 == 0, **kwargs)
    # 使用测试工具比较result和expected，确保它们相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 表示此测试预期失败，因为会出现特定的异常
def test_skip_rows_callable_not_in(all_parsers):
    # 测试用例名称：test_skip_rows_callable_not_in，测试不在指定行跳过的可调用skiprows参数
    parser = all_parsers  # 获取所有解析器的实例
    data = "0,a\n1,b\n2,c\n3,d\n4,e"  # 测试数据，包含多行数字和字母
    expected = DataFrame([[1, "b"], [3, "d"]])  # 预期的DataFrame输出，包含指定行的数据

    # 执行CSV解析，使用lambda函数跳过不在[1, 3]行的数据，并将结果存储在result中
    result = parser.read_csv(
        StringIO(data), header=None, skiprows=lambda x: x not in [1, 3]
    )
    # 使用测试工具比较result和expected，确保它们相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 表示此测试预期失败，因为会出现特定的异常
def test_skip_rows_skip_all(all_parsers):
    # 测试用例名称：test_skip_rows_skip_all，测试跳过所有行的情况
    parser = all_parsers  # 获取所有解析器的实例
    data = "a\n1\n2\n3\n4\n5"  # 测试数据，包含多行数字和字母
    msg = "No columns to parse from file"  # 预期的异常消息

    # 使用pytest断言检查是否会抛出EmptyDataError异常，且异常消息匹配msg
    with pytest.raises(EmptyDataError, match=msg):
        parser.read_csv(StringIO(data), skiprows=lambda x: True)


@xfail_pyarrow  # 表示此测试预期失败，因为会出现特定的异常
def test_skip_rows_bad_callable(all_parsers):
    msg = "by zero"  # 预期的异常消息
    parser = all_parsers  # 获取所有解析器的实例
    data = "a\n1\n2\n3\n4\n5"  # 测试数据，包含多行数字和字母

    # 使用pytest断言检查是否会抛出ZeroDivisionError异常，且异常消息匹配msg
    with pytest.raises(ZeroDivisionError, match=msg):
        parser.read_csv(StringIO(data), skiprows=lambda x: 1 / 0)


@xfail_pyarrow  # 表示此测试预期失败，因为会出现特定的异常
def test_skip_rows_and_n_rows(all_parsers):
    # 测试用例名称：test_skip_rows_and_n_rows，测试同时指定skiprows和nrows参数的情况
    # GH#44021  # 相关GitHub问题号，指向测试用例相关的讨论或问题
    data = """a,b
1,a
2,b
3,c
4,d
5,e
6,f
7,g
8,h
"""
    parser = all_parsers  # 获取所有解析器的实例
    # 执行CSV解析，跳过行[2, 4, 6]，只读取前5行数据，并将结果存储在result中
    result = parser.read_csv(StringIO(data), nrows=5, skiprows=[2, 4, 6])
    # 预期的DataFrame输出，包含两列'a'和'b'，并且仅包含指定行的数据
    expected = DataFrame({"a": [1, 3, 5, 7, 8], "b": ["a", "c", "e", "g", "h"]})
    # 使用测试工具比较result和expected，确保它们相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 表示此测试预期失败，因为会出现特定的异常
def test_skip_rows_with_chunks(all_parsers):
    # 测试用例名称：test_skip_rows_with_chunks，测试使用chunksize和可调用skiprows参数的情况
    # GH 55677  # 相关GitHub问题号，指向测试用例相关的讨论或问题
    data = """col_a
10
20
30
40
50
60
70
80
90
100
"""
    parser = all_parsers  # 获取所有解析器的实例
    # 执行CSV解析，跳过行[1, 4, 5]，每次读取4行数据，并将结果存储在reader中
    reader = parser.read_csv(
        StringIO(data), engine=parser, skiprows=lambda x: x in [1, 4, 5], chunksize=4
    )
    # 从reader中获取第一个chunk，比较其结果与预期的DataFrame{"col_a": [20, 30, 60, 70]}
    df1 = next(reader)
    # 从reader中获取第二个chunk，比较其结果与预期的DataFrame{"col_a": [80, 90, 100]}，并指定行索引
    df2 = next(reader)

    # 使用测试工具比较df1和预期的DataFrame，确保它们相等
    tm.assert_frame_equal(df1, DataFrame({"col_a": [20, 30, 60, 70]}))
    # 使用测试工具比较df2和预期的DataFrame，确保它们相等，并指定行索引为[4, 5, 6]
    tm.assert_frame_equal(df2, DataFrame({"col_a": [80, 90, 100]}, index=[4, 5, 6]))
```
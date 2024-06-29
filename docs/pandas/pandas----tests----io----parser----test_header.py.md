# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_header.py`

```
"""
Tests that the file header is properly handled or inferred
during parsing for all of the parsers defined in parsers.py
"""

# 导入必要的库和模块
from collections import namedtuple
from io import StringIO

import numpy as np
import pytest

from pandas.errors import ParserError

# 导入 pandas 相关模块和函数
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm

# 忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 标记为有问题的测试用例，预期会抛出 TypeError 异常
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
# 标记需要跳过的测试用例，通常用于特定环境下的不支持情况
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


@xfail_pyarrow  # TypeError: an integer is required
def test_read_with_bad_header(all_parsers):
    # 使用所有定义的解析器
    parser = all_parsers
    # 定义预期的错误信息正则表达式
    msg = r"but only \d+ lines in file"

    # 使用 pytest 断言预期异常被抛出，并匹配错误信息
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(",,"), header=[10])


def test_negative_header(all_parsers):
    # 查看 GitHub 问题编号 27779
    parser = all_parsers
    # 准备测试数据
    data = """1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    # 使用 pytest 断言预期异常被抛出，并匹配错误信息
    with pytest.raises(
        ValueError,
        match="Passing negative integer to header is invalid. "
        "For no header, use header=None instead",
    ):
        parser.read_csv(StringIO(data), header=-1)


@pytest.mark.parametrize("header", [([-1, 2, 4]), ([-5, 0])])
def test_negative_multi_index_header(all_parsers, header):
    # 查看 GitHub 问题编号 27779
    parser = all_parsers
    # 准备测试数据
    data = """1,2,3,4,5
        6,7,8,9,10
        11,12,13,14,15
        """
    # 使用 pytest 断言预期异常被抛出，并匹配错误信息
    with pytest.raises(
        ValueError, match="cannot specify multi-index header with negative integers"
    ):
        parser.read_csv(StringIO(data), header=header)


@pytest.mark.parametrize("header", [True, False])
def test_bool_header_arg(all_parsers, header):
    # 查看 GitHub 问题编号 6114
    parser = all_parsers
    # 准备测试数据
    data = """\
MyColumn
a
b
a
b"""
    # 准备异常信息
    msg = "Passing a bool to header is invalid"
    # 使用 pytest 断言预期异常被抛出，并匹配错误信息
    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), header=header)


@xfail_pyarrow  # AssertionError: DataFrame are different
def test_header_with_index_col(all_parsers):
    # 使用所有定义的解析器
    parser = all_parsers
    # 准备测试数据
    data = """foo,1,2,3
bar,4,5,6
baz,7,8,9
"""
    names = ["A", "B", "C"]
    # 执行 CSV 解析，并验证结果
    result = parser.read_csv(StringIO(data), names=names)

    expected = DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],
        columns=["A", "B", "C"],
    )
    tm.assert_frame_equal(result, expected)


def test_header_not_first_line(all_parsers):
    # 使用所有定义的解析器
    parser = all_parsers
    # 准备测试数据
    data = """got,to,ignore,this,line
got,to,ignore,this,line
index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
"""
    data2 = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
"""

    # 分别读取 CSV 并验证结果
    result = parser.read_csv(StringIO(data), header=2, index_col=0)
    expected = parser.read_csv(StringIO(data2), header=0, index_col=0)
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
def test_header_multi_index(all_parsers):
    # 使用所有定义的解析器
    parser = all_parsers

    data = """\
# 导入所需的测试库 pytest
import pytest

# 定义一个包含 CSV 数据的字符串
data = """\
C0,,C_l0_g0,C_l0_g1,C_l0_g2

C1,,C_l1_g0,C_l1_g1,C_l1_g2
C2,,C_l2_g0,C_l2_g1,C_l2_g2
C3,,C_l3_g0,C_l3_g1,C_l3_g2
R0,R1,,,
R_l0_g0,R_l1_g0,R0C0,R0C1,R0C2
R_l0_g1,R_l1_g1,R1C0,R1C1,R1C2
R_l0_g2,R_l1_g2,R2C0,R2C1,R2C2
R_l0_g3,R_l1_g3,R3C0,R3C1,R3C2
R_l0_g4,R_l1_g4,R4C0,R4C1,R4C2
"""

# 定义一个元组，包含测试所需的列解析器
_TestTuple = namedtuple("_TestTuple", ["first", "second"])

# 标记测试为预期失败，因为传递给函数的参数类型不正确
@xfail_pyarrow  # TypeError: an integer is required
# 参数化测试，测试不同的参数组合
@pytest.mark.parametrize(
    "kwargs",
    [
        {"header": [0, 1]},  # 设置头信息为多级索引
        {
            "skiprows": 3,  # 跳过前三行
            "names": [
                ("a", "q"),
                ("a", "r"),
                ("a", "s"),
                ("b", "t"),
                ("c", "u"),
                ("c", "v"),
            ],  # 定义列名
        },
        {
            "skiprows": 3,  # 跳过前三行
            "names": [
                _TestTuple("a", "q"),
                _TestTuple("a", "r"),
                _TestTuple("a", "s"),
                _TestTuple("b", "t"),
                _TestTuple("c", "u"),
                _TestTuple("c", "v"),
            ],  # 定义命名元组作为列名
        },
    ],
)
# 定义测试函数，测试多级索引的常见格式
def test_header_multi_index_common_format1(all_parsers, kwargs):
    parser = all_parsers  # 获取所有的解析器

    # 以下是一个完整的测试代码块，用于测试解析器对多级索引的处理是否正确
    with pytest.raises(ValueError, match=msg):  # 使用 pytest 断言抛出 ValueError 异常，并匹配错误信息
        parser.read_csv(StringIO(data), header=[0, 1, 2, 3], **kwargs)  # 读取 CSV 数据，指定头信息和其他参数
    # 创建一个预期的 DataFrame 对象，包含两行数据，索引为 "one" 和 "two"
    # DataFrame 的列使用 MultiIndex，由元组列表 [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")] 定义
    expected = DataFrame(
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],  # 指定 DataFrame 的数据部分，两行数据
        index=["one", "two"],  # 设置 DataFrame 的行索引为 ["one", "two"]
        columns=MultiIndex.from_tuples(
            [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")]
        ),  # 设置 DataFrame 的列索引为 MultiIndex 对象，由指定的元组列表构成
    )
    
    # 定义一个包含 CSV 格式数据的字符串，该字符串将用于创建 DataFrame
    data = """,a,a,a,b,c,c
@xfail_pyarrow  # 标记此测试函数为预期失败，因为预期会出现 TypeError: an integer is required 错误
@pytest.mark.parametrize(  # 使用参数化装饰器，允许多次运行同一个测试函数，每次使用不同的参数
    "kwargs",  # 参数名称
    [  # 参数化的参数列表开始
        {"header": [0, 1]},  # 第一个参数字典，指定 header 为列表 [0, 1]
        {  # 第二个参数字典，包含跳过行数 skiprows 和列名 names 的定义
            "skiprows": 2,  # 跳过前两行
            "names": [  # 指定列名为元组列表
                ("a", "q"),
                ("a", "r"),
                ("a", "s"),
                ("b", "t"),
                ("c", "u"),
                ("c", "v"),
            ],
        },
        {  # 第三个参数字典，与第二个类似，但是列名用到了自定义类 _TestTuple
            "skiprows": 2,
            "names": [
                _TestTuple("a", "q"),
                _TestTuple("a", "r"),
                _TestTuple("a", "s"),
                _TestTuple("b", "t"),
                _TestTuple("c", "u"),
                _TestTuple("c", "v"),
            ],
        },
    ],  # 参数化的参数列表结束
)
def test_header_multi_index_common_format2(all_parsers, kwargs):
    parser = all_parsers  # 获取所有解析器的实例
    expected = DataFrame(  # 期望的数据帧
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],  # 数据部分
        index=["one", "two"],  # 行索引
        columns=MultiIndex.from_tuples(  # 多级列索引，从元组列表创建
            [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")]
        ),
    )
    data = """,a,a,a,b,c,c  # 数据字符串，包含标题行
,q,r,s,t,u,v  # 第二行数据
one,1,2,3,4,5,6  # 第三行数据
two,7,8,9,10,11,12"""  # 第四行数据

    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)  # 使用给定参数解析数据
    tm.assert_frame_equal(result, expected)  # 断言结果与期望的数据帧相等


@xfail_pyarrow  # 标记此测试函数为预期失败，因为预期会出现 TypeError: an integer is required 错误
@pytest.mark.parametrize(  # 使用参数化装饰器，允许多次运行同一个测试函数，每次使用不同的参数
    "kwargs",  # 参数名称
    [  # 参数化的参数列表开始
        {"header": [0, 1]},  # 第一个参数字典，指定 header 为列表 [0, 1]
        {  # 第二个参数字典，包含跳过行数 skiprows 和列名 names 的定义
            "skiprows": 2,  # 跳过前两行
            "names": [  # 指定列名为元组列表
                ("a", "q"),
                ("a", "r"),
                ("a", "s"),
                ("b", "t"),
                ("c", "u"),
                ("c", "v"),
            ],
        },
        {  # 第三个参数字典，与第二个类似，但是列名用到了自定义类 _TestTuple
            "skiprows": 2,
            "names": [
                _TestTuple("a", "q"),
                _TestTuple("a", "r"),
                _TestTuple("a", "s"),
                _TestTuple("b", "t"),
                _TestTuple("c", "u"),
                _TestTuple("c", "v"),
            ],
        },
    ],  # 参数化的参数列表结束
)
def test_header_multi_index_common_format3(all_parsers, kwargs):
    parser = all_parsers  # 获取所有解析器的实例
    expected = DataFrame(  # 期望的数据帧
        [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]],  # 数据部分
        index=["one", "two"],  # 行索引
        columns=MultiIndex.from_tuples(  # 多级列索引，从元组列表创建
            [("a", "q"), ("a", "r"), ("a", "s"), ("b", "t"), ("c", "u"), ("c", "v")]
        ),
    )
    expected = expected.reset_index(drop=True)  # 重置行索引为默认值
    data = """a,a,a,b,c,c  # 数据字符串，不包含标题行
q,r,s,t,u,v  # 第一行数据
1,2,3,4,5,6  # 第二行数据
7,8,9,10,11,12"""  # 第三行数据

    result = parser.read_csv(StringIO(data), index_col=None, **kwargs)  # 使用给定参数解析数据
    tm.assert_frame_equal(result, expected)  # 断言结果与期望的数据帧相等


@xfail_pyarrow  # 标记此测试函数为预期失败，因为预期会出现 TypeError: an integer is required 错误
def test_header_multi_index_common_format_malformed1(all_parsers):
    parser = all_parsers  # 获取所有解析器的实例
    # 创建一个预期的 DataFrame 对象，包含指定的数据和结构
    expected = DataFrame(
        np.array([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12]], dtype="int64"),  # 使用 numpy 数组创建 DataFrame，指定数据类型为 int64
        index=Index([1, 7]),  # 设置 DataFrame 的行索引为 [1, 7]
        columns=MultiIndex(  # 设置 DataFrame 的列索引为 MultiIndex 对象
            levels=[["a", "b", "c"], ["r", "s", "t", "u", "v"]],  # 第一层级为 ['a', 'b', 'c']，第二层级为 ['r', 's', 't', 'u', 'v']
            codes=[[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]],  # 指定列索引的编码，与 levels 对应
            names=["a", "q"],  # 设置列索引的名称，分别为 "a" 和 "q"
        ),
    )
    # 创建一个字符串变量 data，包含指定的 CSV 格式数据
    data = """a,a,a,b,c,c
@xfail_pyarrow  # 这个测试标记为在使用 pyarrow 引擎时可能会失败，预期会抛出 TypeError 异常

def test_header_multi_index_common_format_malformed2(all_parsers):
    # 使用 all_parsers 对象作为 CSV 解析器
    parser = all_parsers

    # 预期的 DataFrame，包括两行数据和多级索引列
    expected = DataFrame(
        np.array([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12]], dtype="int64"),
        index=Index([1, 7]),
        columns=MultiIndex(
            levels=[["a", "b", "c"], ["r", "s", "t", "u", "v"]],
            codes=[[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]],
            names=[None, "q"],
        ),
    )

    data = """,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    # 使用 StringIO 将 data 字符串转换为文件对象，然后使用 parser 解析 CSV 数据
    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=0)
    # 使用 tm.assert_frame_equal 检查预期结果和实际结果是否相同
    tm.assert_frame_equal(expected, result)


@xfail_pyarrow  # 这个测试标记为在使用 pyarrow 引擎时可能会失败，预期会抛出 TypeError 异常
def test_header_multi_index_common_format_malformed3(all_parsers):
    # 使用 all_parsers 对象作为 CSV 解析器
    parser = all_parsers

    # 预期的 DataFrame，包括两行数据和多级索引列
    expected = DataFrame(
        np.array([[3, 4, 5, 6], [9, 10, 11, 12]], dtype="int64"),
        index=MultiIndex(levels=[[1, 7], [2, 8]], codes=[[0, 1], [0, 1]]),
        columns=MultiIndex(
            levels=[["a", "b", "c"], ["s", "t", "u", "v"]],
            codes=[[0, 1, 2, 2], [0, 1, 2, 3]],
            names=[None, "q"],
        ),
    )
    data = """,a,a,b,c,c
q,r,s,t,u,v
1,2,3,4,5,6
7,8,9,10,11,12"""

    # 使用 StringIO 将 data 字符串转换为文件对象，然后使用 parser 解析 CSV 数据
    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=[0, 1])
    # 使用 tm.assert_frame_equal 检查预期结果和实际结果是否相同
    tm.assert_frame_equal(expected, result)


@xfail_pyarrow  # 这个测试标记为在使用 pyarrow 引擎时可能会失败，预期会抛出 TypeError 异常
def test_header_multi_index_blank_line(all_parsers):
    # GH 40442
    # 使用 all_parsers 对象作为 CSV 解析器
    parser = all_parsers

    # 准备一个包含空行的数据列表
    data = [[None, None], [1, 2], [3, 4]]
    # 使用 MultiIndex.from_tuples 生成列名为 ("a", "A"), ("b", "B") 的多级索引
    columns = MultiIndex.from_tuples([("a", "A"), ("b", "B")])
    # 创建预期的 DataFrame
    expected = DataFrame(data, columns=columns)
    # 准备包含空行的 CSV 数据
    data = "a,b\nA,B\n,\n1,2\n3,4"
    # 使用 parser 解析 CSV 数据，header=[0, 1] 表示使用多级列头
    result = parser.read_csv(StringIO(data), header=[0, 1])
    # 使用 tm.assert_frame_equal 检查预期结果和实际结果是否相同
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize(
    "data,header", [("1,2,3\n4,5,6", None), ("foo,bar,baz\n1,2,3\n4,5,6", 0)]
)
def test_header_names_backward_compat(all_parsers, data, header, request):
    # see gh-2539
    # 使用 all_parsers 对象作为 CSV 解析器
    parser = all_parsers

    # 如果使用 pyarrow 引擎并且 header 不为 None，则标记为预期失败
    if parser.engine == "pyarrow" and header is not None:
        mark = pytest.mark.xfail(reason="DataFrame.columns are different")
        request.applymarker(mark)

    # 预期的 DataFrame，根据固定列名生成
    expected = parser.read_csv(StringIO("1,2,3\n4,5,6"), names=["a", "b", "c"])

    # 使用 parser 解析传入的 CSV 数据
    result = parser.read_csv(StringIO(data), names=["a", "b", "c"], header=header)
    # 使用 tm.assert_frame_equal 检查预期结果和实际结果是否相同
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block: cannot infer
@pytest.mark.parametrize("kwargs", [{}, {"index_col": False}])
def test_read_only_header_no_rows(all_parsers, kwargs):
    # See gh-7773
    # 使用 all_parsers 对象作为 CSV 解析器
    parser = all_parsers

    # 预期结果是一个空的 DataFrame，只有列名为 ["a", "b", "c"] 的列
    expected = DataFrame(columns=["a", "b", "c"])

    # 使用 parser 解析传入的 CSV 数据，kwargs 可以包含 index_col 参数
    result = parser.read_csv(StringIO("a,b,c"), **kwargs)
    # 使用 tm.assert_frame_equal 检查预期结果和实际结果是否相同
    tm.assert_frame_equal(result, expected)
    [
        # 第一个元组，空字典作为第一个元素，列表作为第二个元素
        ({}, [0, 1, 2, 3, 4]),
        # 第二个元组，包含一个带有键为"names"的字典和一个相同顺序的字符串列表
        (
            {"names": ["foo", "bar", "baz", "quux", "panda"]},
            ["foo", "bar", "baz", "quux", "panda"],
        ),
    ],
# 定义一个测试函数，测试没有标题行的 CSV 解析
def test_no_header(all_parsers, kwargs, names):
    # 从参数中获取解析器对象
    parser = all_parsers
    # 定义包含数据的字符串
    data = """1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    # 预期的数据框架，包含三行数据，列名为传入的 names 列表
    expected = DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]], columns=names
    )
    # 使用解析器解析 CSV 数据，header 设置为 None 表示没有标题行
    result = parser.read_csv(StringIO(data), header=None, **kwargs)
    # 断言解析结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的参数化装饰器来定义多个测试用例，测试非整数标题行的情况
@pytest.mark.parametrize("header", [["a", "b"], "string_header"])
def test_non_int_header(all_parsers, header):
    # 见 gh-16338
    msg = "header must be integer or list of integers"
    # 定义包含数据的字符串
    data = """1,2\n3,4"""
    # 从参数中获取解析器对象
    parser = all_parsers

    # 使用 pytest 的断言来验证是否会抛出 ValueError 异常，并检查异常消息是否符合预期
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), header=header)


# 使用装饰器 @xfail_pyarrow 来标记此测试用例为预期失败，原因是 TypeError: an integer is required
@xfail_pyarrow
def test_singleton_header(all_parsers):
    # 见 gh-7757
    # 定义包含数据的字符串
    data = """a,b,c\n0,1,2\n1,2,3"""
    # 从参数中获取解析器对象
    parser = all_parsers

    # 预期的数据框架，包含一行数据，列名为字典中的键
    expected = DataFrame({"a": [0, 1], "b": [1, 2], "c": [2, 3]})
    # 使用解析器解析 CSV 数据，header 设置为 [0] 表示第一行作为标题行
    result = parser.read_csv(StringIO(data), header=[0])
    # 断言解析结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 使用装饰器 @xfail_pyarrow 来标记此测试用例为预期失败，原因是 TypeError: an integer is required
@xfail_pyarrow
@pytest.mark.parametrize(
    "data,expected",
    [
        (
            "A,A,A,B\none,one,one,two\n0,40,34,0.1",
            DataFrame(
                [[0, 40, 34, 0.1]],
                columns=MultiIndex.from_tuples(
                    [("A", "one"), ("A", "one.1"), ("A", "one.2"), ("B", "two")]
                ),
            ),
        ),
        (
            "A,A,A,B\none,one,one.1,two\n0,40,34,0.1",
            DataFrame(
                [[0, 40, 34, 0.1]],
                columns=MultiIndex.from_tuples(
                    [("A", "one"), ("A", "one.1"), ("A", "one.1.1"), ("B", "two")]
                ),
            ),
        ),
        (
            "A,A,A,B,B\none,one,one.1,two,two\n0,40,34,0.1,0.1",
            DataFrame(
                [[0, 40, 34, 0.1, 0.1]],
                columns=MultiIndex.from_tuples(
                    [
                        ("A", "one"),
                        ("A", "one.1"),
                        ("A", "one.1.1"),
                        ("B", "two"),
                        ("B", "two.1"),
                    ]
                ),
            ),
        ),
    ],
)
def test_mangles_multi_index(all_parsers, data, expected):
    # 见 gh-18062
    # 从参数中获取解析器对象
    parser = all_parsers

    # 使用解析器解析 CSV 数据，header 设置为 [0, 1] 表示前两行作为多级索引
    result = parser.read_csv(StringIO(data), header=[0, 1])
    # 断言解析结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


# 使用装饰器 @xfail_pyarrow 来标记此测试用例为预期失败，原因是 TypeError: an integer is required
@xfail_pyarrow
@pytest.mark.parametrize("index_col", [None, [0]])
@pytest.mark.parametrize(
    "columns", [None, (["", "Unnamed"]), (["Unnamed", ""]), (["Unnamed", "NotUnnamed"])]
)
def test_multi_index_unnamed(all_parsers, index_col, columns):
    # 见 gh-23687
    #
    # 当指定多级索引标题时，确保我们不会因为标题行中的所有列名包含字符串 "Unnamed" 而出错。
    # 正确的条件是检查行是否包含
    # 创建一个名为 parser 的变量，其值是 all_parsers
    parser = all_parsers
    # 创建一个名为 header 的列表变量，包含值 [0, 1]
    header = [0, 1]

    # 如果 index_col 为 None，则生成一个包含占位符的数据字符串
    if index_col is None:
        data = ",".join(columns or ["", ""]) + "\n0,1\n2,3\n4,5\n"
    else:
        # 否则生成一个包含占位符的数据字符串，第一行用于索引列
        data = ",".join([""] + (columns or ["", ""])) + "\n,0,1\n0,2,3\n1,4,5\n"

    # 使用 StringIO 将 data 字符串转换为文件对象，并使用 parser 进行 CSV 解析
    result = parser.read_csv(StringIO(data), header=header, index_col=index_col)
    # 创建一个空列表 exp_columns
    exp_columns = []

    # 如果 columns 为 None，则将其赋值为包含三个空字符串的列表
    if columns is None:
        columns = ["", "", ""]

    # 遍历 columns 列表的索引和值
    for i, col in enumerate(columns):
        # 如果 col 为空字符串，则将其替换为 "Unnamed: {i}" 的格式化字符串
        if not col:  # Unnamed.
            col = f"Unnamed: {i if index_col is None else i + 1}_level_0"

        # 将处理后的列名加入 exp_columns 列表
        exp_columns.append(col)

    # 使用 exp_columns 和固定的索引名称创建 MultiIndex 对象作为 columns
    columns = MultiIndex.from_tuples(zip(exp_columns, ["0", "1"]))
    # 创建一个期望的 DataFrame 对象 expected，包含特定的数据和列名
    expected = DataFrame([[2, 3], [4, 5]], columns=columns)
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
@skip_pyarrow  # 在测试中跳过，因为 CSV 解析错误：预期 2 列，实际得到 3 列
def test_names_longer_than_header_but_equal_with_data_rows(all_parsers):
    # GH#38453
    # 使用所有解析器中的第一个解析器
    parser = all_parsers
    # 定义包含不等长头部但等长数据行的 CSV 数据
    data = """a, b
1,2,3
5,6,4
"""
    # 使用解析器读取 CSV 数据，指定头部行为 0，列名为 ["A", "B", "C"]
    result = parser.read_csv(StringIO(data), header=0, names=["A", "B", "C"])
    # 预期的数据框架
    expected = DataFrame({"A": [1, 5], "B": [2, 6], "C": [3, 4]})
    # 断言读取结果与预期结果相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 在测试中标记为失败，因为 TypeError: an integer is required
def test_read_csv_multiindex_columns(all_parsers):
    # GH#6051
    # 使用所有解析器中的第一个解析器
    parser = all_parsers

    # 定义包含多重索引列的 CSV 数据 s1 和 s2
    s1 = "Male, Male, Male, Female, Female\nR, R, L, R, R\n.86, .67, .88, .78, .81"
    s2 = (
        "Male, Male, Male, Female, Female\n"
        "R, R, L, R, R\n"
        ".86, .67, .88, .78, .81\n"
        ".86, .67, .88, .78, .82"
    )

    # 创建多重索引对象 mi
    mi = MultiIndex.from_tuples(
        [
            ("Male", "R"),
            (" Male", " R"),
            (" Male", " L"),
            (" Female", " R"),
            (" Female", " R.1"),
        ]
    )

    # 预期的数据框架
    expected = DataFrame(
        [[0.86, 0.67, 0.88, 0.78, 0.81], [0.86, 0.67, 0.88, 0.78, 0.82]], columns=mi
    )

    # 使用解析器读取 CSV 数据，指定多重头部为 [0, 1]
    df1 = parser.read_csv(StringIO(s1), header=[0, 1])
    # 断言读取结果的第一部分与预期结果相等
    tm.assert_frame_equal(df1, expected.iloc[:1])
    df2 = parser.read_csv(StringIO(s2), header=[0, 1])
    # 断言读取结果的第二部分与预期结果相等
    tm.assert_frame_equal(df2, expected)


@xfail_pyarrow  # 在测试中标记为失败，因为 TypeError: an integer is required
def test_read_csv_multi_header_length_check(all_parsers):
    # GH#43102
    # 使用所有解析器中的第一个解析器
    parser = all_parsers

    # 定义包含多重头部的 CSV 数据 case
    case = """row11,row12,row13
row21,row22, row23
row31,row32
"""

    # 使用 pytest 断言检查解析器读取 CSV 数据时是否抛出 ParserError 异常，匹配给定的错误信息
    with pytest.raises(
        ParserError, match="Header rows must have an equal number of columns."
    ):
        parser.read_csv(StringIO(case), header=[0, 2])


@skip_pyarrow  # 在测试中跳过，因为 CSV 解析错误：预期 3 列，实际得到 2 列
def test_header_none_and_implicit_index(all_parsers):
    # GH#22144
    # 使用所有解析器中的第一个解析器
    parser = all_parsers
    # 定义不含显式头部行但包含隐式索引的 CSV 数据
    data = "x,1,5\ny,2\nz,3\n"
    # 使用解析器读取 CSV 数据，指定列名为 ["a", "b"]，不设定头部
    result = parser.read_csv(StringIO(data), names=["a", "b"], header=None)
    # 预期的数据框架
    expected = DataFrame(
        {"a": [1, 2, 3], "b": [5, np.nan, np.nan]}, index=["x", "y", "z"]
    )
    # 断言读取结果与预期结果相等
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # 在测试中跳过，因为正则表达式不匹配 "CSV 解析错误：预期 2 列，实际得到 "
def test_header_none_and_implicit_index_in_second_row(all_parsers):
    # GH#22144
    # 使用所有解析器中的第一个解析器
    parser = all_parsers
    # 定义不含显式头部行但包含隐式索引的 CSV 数据，其中第二行出现错误
    data = "x,1\ny,2,5\nz,3\n"
    # 使用 pytest 断言检查解析器读取 CSV 数据时是否抛出 ParserError 异常，匹配给定的错误信息
    with pytest.raises(ParserError, match="Expected 2 fields in line 2, saw 3"):
        parser.read_csv(StringIO(data), names=["a", "b"], header=None)


def test_header_none_and_on_bad_lines_skip(all_parsers):
    # GH#22144
    # 使用所有解析器中的第一个解析器
    parser = all_parsers
    # 定义不含显式头部行但包含隐式索引的 CSV 数据
    data = "x,1\ny,2,5\nz,3\n"
    # 使用解析器读取 CSV 数据，指定列名为 ["a", "b"]，不设定头部，当遇到错误行时跳过
    result = parser.read_csv(
        StringIO(data), names=["a", "b"], header=None, on_bad_lines="skip"
    )
    # 预期的数据框架
    expected = DataFrame({"a": ["x", "z"], "b": [1, 3]})
    # 断言读取结果与预期结果相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 在测试中标记为失败，因为 TypeError: an integer is requireds
def test_header_missing_rows(all_parsers):
    # GH#47400
    # 使用所有解析器中的第一个解析器
    parser = all_parsers
    # 定义缺少部分头部行的 CSV 数据
    data = """a,b
1,2
"""
    # 定义一个字符串变量，用于匹配 pytest 异常时的错误消息
    msg = r"Passed header=\[0,1,2\], len of 3, but only 2 lines in file"
    # 使用 pytest 模块的 raises 函数来检查是否抛出 ValueError 异常，并验证其错误消息是否匹配预期
    with pytest.raises(ValueError, match=msg):
        # 调用 parser 对象的 read_csv 方法，传入 StringIO 对象和自定义的 header 参数
        parser.read_csv(StringIO(data), header=[0, 1, 2])
# ValueError: the 'pyarrow' engine does not support regex separators
@xfail_pyarrow
# 定义一个测试函数，用于测试 CSV 解析器在特定情况下的行为
def test_header_multiple_whitespaces(all_parsers):
    # GH#54931
    # 从参数中获取所有的解析器
    parser = all_parsers
    # 定义包含多个空格的 CSV 数据
    data = """aa    bb(1,1)   cc(1,1)
                0  2  3.5"""
    
    # 使用指定的解析器读取 CSV 数据，分隔符为一个或多个空白字符的正则表达式
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    # 期望的数据框架结果，由字典创建
    expected = DataFrame({"aa": [0], "bb(1,1)": [2], "cc(1,1)": [3.5]})
    # 断言结果与期望值相等
    tm.assert_frame_equal(result, expected)


# ValueError: the 'pyarrow' engine does not support regex separators
@xfail_pyarrow
# 定义一个测试函数，用于测试 CSV 解析器在处理特定数据格式时的行为
def test_header_delim_whitespace(all_parsers):
    # GH#54918
    # 从参数中获取所有的解析器
    parser = all_parsers
    # 定义包含空白字符作为分隔符的 CSV 数据
    data = """a,b
1,2
3,4
    """
    # 使用指定的解析器读取 CSV 数据，分隔符为一个或多个空白字符的正则表达式
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    # 期望的数据框架结果，由字典创建
    expected = DataFrame({"a,b": ["1,2", "3,4"]})
    # 断言结果与期望值相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，测试 pyarrow 引擎在没有标题行的情况下使用列选择的行为
def test_usecols_no_header_pyarrow(pyarrow_parser_only):
    # 从参数中获取只使用 pyarrow 解析器的实例
    parser = pyarrow_parser_only
    # 定义没有标题行的 CSV 数据
    data = """
a,i,x
b,j,y
"""
    # 使用 pyarrow 解析器读取 CSV 数据，指定没有标题行，选择第一列和第二列作为数据类型为 string[pyarrow] 的列
    result = parser.read_csv(
        StringIO(data),
        header=None,
        usecols=[0, 1],
        dtype="string[pyarrow]",
        dtype_backend="pyarrow",
        engine="pyarrow",
    )
    # 期望的数据框架结果，由列表创建
    expected = DataFrame([["a", "i"], ["b", "j"]], dtype="string[pyarrow]")
    # 断言结果与期望值相等
    tm.assert_frame_equal(result, expected)
```
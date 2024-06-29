# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_common_basic.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入所需的模块和库
from datetime import datetime  # 导入datetime模块中的datetime类
from inspect import signature  # 导入inspect模块中的signature函数
from io import StringIO  # 导入io模块中的StringIO类
import os  # 导入os模块
from pathlib import Path  # 导入pathlib模块中的Path类
import sys  # 导入sys模块

import numpy as np  # 导入numpy库，并将其简称为np
import pytest  # 导入pytest库

from pandas.errors import (  # 从pandas库中导入指定的错误类
    EmptyDataError,
    ParserError,
    ParserWarning,
)

from pandas import (  # 从pandas库中导入指定的类和函数
    DataFrame,
    Index,
    compat,
)
import pandas._testing as tm  # 导入pandas._testing模块，并将其简称为tm

# 设置pytest标记，忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 定义xfail_pyarrow标记，用于标记针对pyarrow引擎的预期失败测试
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
# 定义skip_pyarrow标记，用于标记针对pyarrow引擎的跳过测试
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")


# 定义测试函数test_read_csv_local，用于测试从本地读取CSV文件
def test_read_csv_local(all_parsers, csv1):
    # 根据操作系统类型设置文件路径前缀
    prefix = "file:///" if compat.is_platform_windows() else "file://"
    # 选择使用的解析器
    parser = all_parsers

    # 构造文件的完整URL路径
    fname = prefix + str(os.path.abspath(csv1))
    # 使用解析器读取CSV文件，设置第一列为索引，解析日期为时间类型
    result = parser.read_csv(fname, index_col=0, parse_dates=True)

    # 预期的DataFrame结果
    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.346270],
            [1.120202, 1.567621, 0.00364077397681, 0.675253],
            [-0.487094, 0.571455, -1.6116394093, 0.103469],
            [0.836649, 0.246462, 0.588542635376, 1.062782],
            [-0.157161, 1.340307, 1.1957779562, -1.097007],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
                datetime(2000, 1, 10),
                datetime(2000, 1, 11),
            ],
            dtype="M8[s]",
            name="index",
        ),
    )
    # 断言读取结果与预期结果相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数test_1000_sep，用于测试CSV文件中包含千位分隔符的情况
def test_1000_sep(all_parsers):
    # 选择使用的解析器
    parser = all_parsers
    # 定义包含千位分隔符的CSV数据字符串
    data = """A|B|C
1|2,334|5
10|13|10.
"""
    # 预期的DataFrame结果
    expected = DataFrame({"A": [1, 10], "B": [2334, 13], "C": [5, 10.0]})

    # 如果解析引擎为pyarrow，则预期引发特定的ValueError异常
    if parser.engine == "pyarrow":
        msg = "The 'thousands' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep="|", thousands=",")
        return

    # 使用解析器读取CSV数据，并设置分隔符和千位分隔符
    result = parser.read_csv(StringIO(data), sep="|", thousands=",")
    # 断言读取结果与预期结果相等
    tm.assert_frame_equal(result, expected)


# 定义预期在pyarrow引擎下失败的测试函数test_unnamed_columns
@xfail_pyarrow  # 标记为预期在pyarrow引擎下失败的测试
def test_unnamed_columns(all_parsers):
    # 定义包含未命名列的CSV数据字符串
    data = """A,B,C,,
1,2,3,4,5
6,7,8,9,10
11,12,13,14,15
"""
    # 选择使用的解析器
    parser = all_parsers
    # 预期的DataFrame结果
    expected = DataFrame(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dtype=np.int64,
        columns=["A", "B", "C", "Unnamed: 3", "Unnamed: 4"],
    )
    # 使用解析器读取CSV数据
    result = parser.read_csv(StringIO(data))
    # 断言读取结果与预期结果相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数test_csv_mixed_type，用于测试包含混合数据类型的CSV文件
def test_csv_mixed_type(all_parsers):
    # 定义包含混合数据类型的CSV数据字符串
    data = """A,B,C
a,1,2
b,3,4
c,4,5
"""
    # 选择使用的解析器
    parser = all_parsers
    # 将变量 parser 设为 all_parsers 的引用，假设它是一个包含所有解析器的对象或函数
    parser = all_parsers
    # 创建预期的 DataFrame，包含三列 A, B, C 和对应的数据
    expected = DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 4], "C": [2, 4, 5]})
    # 使用 parser 解析传入的数据 data，假设 data 是一个包含 CSV 数据的字符串
    result = parser.read_csv(StringIO(data))
    # 使用 pandas 的 tm.assert_frame_equal 方法比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
def test_read_csv_low_memory_no_rows_with_index(all_parsers):
    # 此测试用例涵盖 GitHub issue #21141
    # 从参数中获取所有解析器
    parser = all_parsers

    # 如果解析器不支持低内存模式，则跳过测试
    if not parser.low_memory:
        pytest.skip("This is a low-memory specific test")

    # 模拟的CSV数据
    data = """A,B,C
1,1,1,2
2,2,3,4
3,3,4,5
"""

    # 如果使用的是 'pyarrow' 引擎，则检查 'nrows' 选项是否受支持
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
        return

    # 使用解析器读取 CSV 数据，使用低内存模式，设置索引列为第一列，读取0行数据
    result = parser.read_csv(StringIO(data), low_memory=True, index_col=0, nrows=0)
    # 预期结果是一个空的 DataFrame，列名为 ["A", "B", "C"]
    expected = DataFrame(columns=["A", "B", "C"])
    # 断言实际结果与预期结果相同
    tm.assert_frame_equal(result, expected)


def test_read_csv_dataframe(all_parsers, csv1):
    # 从参数中获取所有解析器和一个 CSV 文件路径
    parser = all_parsers
    # 使用解析器读取 CSV 文件，并设定第一列为索引，解析日期列
    result = parser.read_csv(csv1, index_col=0, parse_dates=True)
    # 预期结果是一个 DataFrame，包含特定数据和索引
    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738],
            [1.047916, -0.041232, -0.16181208307, 0.212549],
            [0.498581, 0.731168, -0.537677223318, 1.346270],
            [1.120202, 1.567621, 0.00364077397681, 0.675253],
            [-0.487094, 0.571455, -1.6116394093, 0.103469],
            [0.836649, 0.246462, 0.588542635376, 1.062782],
            [-0.157161, 1.340307, 1.1957779562, -1.097007],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
                datetime(2000, 1, 10),
                datetime(2000, 1, 11),
            ],
            dtype="M8[s]",
            name="index",
        ),
    )
    # 断言实际结果与预期结果相同
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [3, 3.0])
def test_read_nrows(all_parsers, nrows):
    # 此测试用例涵盖 GitHub issue #10476
    # 模拟的 CSV 数据
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    # 预期结果是一个 DataFrame，包含特定数据和列名
    expected = DataFrame(
        [["foo", 2, 3, 4, 5], ["bar", 7, 8, 9, 10], ["baz", 12, 13, 14, 15]],
        columns=["index", "A", "B", "C", "D"],
    )
    # 从参数中获取所有解析器
    parser = all_parsers

    # 如果使用的是 'pyarrow' 引擎，则检查 'nrows' 选项是否受支持
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), nrows=nrows)
        return

    # 使用解析器读取 CSV 数据，读取指定行数（nrows）
    result = parser.read_csv(StringIO(data), nrows=nrows)
    # 断言实际结果与预期结果相同
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("nrows", [1.2, "foo", -1])
def test_read_nrows_bad(all_parsers, nrows):
    # 模拟的 CSV 数据
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""
    # 错误消息的预期文本
    msg = r"'nrows' must be an integer >=0"
    # 从参数中获取所有解析器
    parser = all_parsers
    # 如果使用的是 'pyarrow' 引擎，则更新错误消息
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
    # 使用 pytest 框架的 `raises` 上下文管理器，检查是否会抛出 ValueError 异常，并且异常消息必须与变量 msg 的内容匹配
    with pytest.raises(ValueError, match=msg):
        # 调用 parser 对象的 read_csv 方法，解析数据流 StringIO(data)，并限制读取行数为 nrows
        parser.read_csv(StringIO(data), nrows=nrows)
# 测试函数，验证 'skipfooter' 和 'nrows' 同时使用时会引发 ValueError 异常
def test_nrows_skipfooter_errors(all_parsers):
    msg = "'skipfooter' not supported with 'nrows'"
    # 准备测试数据
    data = "a\n1\n2\n3\n4\n5\n6"
    # 获取测试解析器对象
    parser = all_parsers

    # 使用 pytest 断言检查是否抛出预期的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), skipfooter=1, nrows=5)


# 装饰器函数，跳过使用 pyarrow 引擎执行的测试
@skip_pyarrow
def test_missing_trailing_delimiters(all_parsers):
    # 准备测试数据
    data = """A,B,C,D
1,2,3,4
1,3,3,
1,4,5"""

    # 获取测试解析器对象
    parser = all_parsers

    # 执行 CSV 文件读取操作
    result = parser.read_csv(StringIO(data))

    # 期望的 DataFrame 结果
    expected = DataFrame(
        [[1, 2, 3, 4], [1, 3, 3, np.nan], [1, 4, 5, np.nan]],
        columns=["A", "B", "C", "D"],
    )

    # 使用 pandas 测试工具比较结果是否一致
    tm.assert_frame_equal(result, expected)


# 测试函数，验证 'skipinitialspace' 选项是否与 pyarrow 引擎兼容
def test_skip_initial_space(all_parsers):
    # 准备测试数据
    data = (
        '"09-Apr-2012", "01:10:18.300", 2456026.548822908, 12849, '
        "1.00361,  1.12551, 330.65659, 0355626618.16711,  73.48821, "
        "314.11625,  1917.09447,   179.71425,  80.000, 240.000, -350,  "
        "70.06056, 344.98370, 1,   1, -0.689265, -0.692787,  "
        "0.212036,    14.7674,   41.605,   -9999.0,   -9999.0,   "
        "-9999.0,   -9999.0,   -9999.0,  -9999.0, 000, 012, 128"
    )
    # 获取测试解析器对象
    parser = all_parsers

    # 如果使用的是 pyarrow 引擎，则验证是否抛出 ValueError 异常
    if parser.engine == "pyarrow":
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                names=list(range(33)),
                header=None,
                na_values=["-9999.0"],
                skipinitialspace=True,
            )
        return

    # 使用其他引擎时，执行 CSV 文件读取操作
    result = parser.read_csv(
        StringIO(data),
        names=list(range(33)),
        header=None,
        na_values=["-9999.0"],
        skipinitialspace=True,
    )

    # 期望的 DataFrame 结果
    expected = DataFrame(
        [
            [
                "09-Apr-2012",
                "01:10:18.300",
                2456026.548822908,
                12849,
                1.00361,
                1.12551,
                330.65659,
                355626618.16711,
                73.48821,
                314.11625,
                1917.09447,
                179.71425,
                80.0,
                240.0,
                -350,
                70.06056,
                344.9837,
                1,
                1,
                -0.689265,
                -0.692787,
                0.212036,
                14.7674,
                41.605,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                0,
                12,
                128,
            ]
        ]
    )

    # 使用 pandas 测试工具比较结果是否一致
    tm.assert_frame_equal(result, expected)


# 装饰器函数，跳过使用 pyarrow 引擎执行的测试
@skip_pyarrow
def test_trailing_delimiters(all_parsers):
    # 准备测试数据
    data = """A,B,C
1,2,3,
4,5,6,
7,8,9,"""
    # 获取测试解析器对象
    parser = all_parsers

    # 执行 CSV 文件读取操作
    result = parser.read_csv(StringIO(data), index_col=False)

    # 期望的 DataFrame 结果
    expected = DataFrame({"A": [1, 4, 7], "B": [2, 5, 8], "C": [3, 6, 9]})

    # 使用 pandas 测试工具比较结果是否一致
    tm.assert_frame_equal(result, expected)
# 测试处理转义字符的功能
def test_escapechar(all_parsers):
    # 数据源，包含转义字符和引号
    data = '''SEARCH_TERM,ACTUAL_URL
"bra tv board","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"
"tv p\xc3\xa5 hjul","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"
"SLAGBORD, \\"Bergslagen\\", IKEA:s 1700-tals series","http://www.ikea.com/se/sv/catalog/categories/departments/living_room/10475/?se%7cps%7cnonbranded%7cvardagsrum%7cgoogle%7ctv_bord"'''

    parser = all_parsers
    # 使用指定的解析器读取 CSV 数据，设置转义字符、引号和编码
    result = parser.read_csv(
        StringIO(data), escapechar="\\", quotechar='"', encoding="utf-8"
    )

    # 断言特定位置的数据内容
    assert result["SEARCH_TERM"][2] == 'SLAGBORD, "Bergslagen", IKEA:s 1700-tals series'

    # 断言列索引与预期相等
    tm.assert_index_equal(result.columns, Index(["SEARCH_TERM", "ACTUAL_URL"]))


# 测试忽略行首空白字符的功能
def test_ignore_leading_whitespace(all_parsers):
    # 见 GitHub 问题 gh-3374, gh-6607
    parser = all_parsers
    # 定义包含前导空白字符的数据
    data = " a b c\n 1 2 3\n 4 5 6\n 7 8 9"

    if parser.engine == "pyarrow":
        # 如果使用 pyarrow 引擎，则不支持正则分隔符，期望引发 ValueError 异常
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+")
        return
    
    # 使用正则分隔符读取数据
    result = parser.read_csv(StringIO(data), sep=r"\s+")

    # 预期的 DataFrame 结果
    expected = DataFrame({"a": [1, 4, 7], "b": [2, 5, 8], "c": [3, 6, 9]})
    tm.assert_frame_equal(result, expected)


# 跳过 pyarrow 引擎的测试，并参数化测试列数不均匀的情况
@skip_pyarrow
@pytest.mark.parametrize("usecols", [None, [0, 1], ["a", "b"]])
def test_uneven_lines_with_usecols(all_parsers, usecols):
    # 见 GitHub 问题 gh-12203
    parser = all_parsers
    # 包含不均匀列数的数据
    data = r"""a,b,c
0,1,2
3,4,5,6,7
8,9,10"""

    if usecols is None:
        # 当未提供 "usecols" 参数时，确保引发错误
        msg = r"Expected \d+ fields in line \d+, saw \d+"
        with pytest.raises(ParserError, match=msg):
            parser.read_csv(StringIO(data))
    else:
        # 预期的 DataFrame 结果
        expected = DataFrame({"a": [0, 3, 8], "b": [1, 4, 9]})

        # 使用指定的列读取数据
        result = parser.read_csv(StringIO(data), usecols=usecols)
        tm.assert_frame_equal(result, expected)


# 跳过 pyarrow 引擎的测试，并参数化测试读取空数据的情况
@skip_pyarrow
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        # 首先，检查解析器在未提供列名时是否会引发正确的错误，无论是否使用 usecols 参数
        ("", {}, None),
        ("", {"usecols": ["X"]}, None),
        (
            ",,",
            {"names": ["Dummy", "X", "Dummy_2"], "usecols": ["X"]},
            DataFrame(columns=["X"], index=[0], dtype=np.float64),
        ),
        (
            "",
            {"names": ["Dummy", "X", "Dummy_2"], "usecols": ["X"]},
            DataFrame(columns=["X"]),
        ),
    ],
)
def test_read_empty_with_usecols(all_parsers, data, kwargs, expected):
    # 见 GitHub 问题 gh-12493
    parser = all_parsers
    # 如果预期结果为 None，则说明文件中没有可解析的列
    if expected is None:
        # 准备错误消息，用于 pytest 的异常断言
        msg = "No columns to parse from file"
        # 使用 pytest 的断言，期望捕获 EmptyDataError 异常，并匹配特定的错误消息
        with pytest.raises(EmptyDataError, match=msg):
            # 调用 parser 对象的 read_csv 方法，传入数据流 StringIO(data) 和其他参数 kwargs
            parser.read_csv(StringIO(data), **kwargs)
    else:
        # 如果有预期结果，则调用 parser 对象的 read_csv 方法解析数据流 StringIO(data) 和其他参数 kwargs
        result = parser.read_csv(StringIO(data), **kwargs)
        # 使用 pandas 测试工具（tm）来比较返回的结果和预期结果是否相等
        tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize(
    "kwargs,expected_data",
    [
        # 定义测试参数和预期结果数据
        (
            {
                "header": None,  # 不使用列名作为 header
                "sep": r"\s+",  # 分隔符为正则表达式中的空白字符
                "skiprows": [0, 1, 2, 3, 5, 6],  # 跳过指定行
                "skip_blank_lines": True,  # 跳过空白行
            },
            [[1.0, 2.0, 4.0], [5.1, np.nan, 10.0]],  # 预期的数据列表
        ),
        # 测试跳过一行后面的一组行，行尾有空格
        (
            {
                "sep": r"\s+",  # 分隔符为正则表达式中的空白字符
                "skiprows": [1, 2, 3, 5, 6],  # 跳过指定行
                "skip_blank_lines": True,  # 跳过空白行
            },
            {"A": [1.0, 5.1], "B": [2.0, np.nan], "C": [4.0, 10]},  # 预期的数据字典
        ),
    ],
)
def test_trailing_spaces(all_parsers, kwargs, expected_data):
    data = "A B C  \nrandom line with trailing spaces    \nskip\n1,2,3\n1,2.,4.\nrandom line with trailing tabs\t\t\t\n   \n5.1,NaN,10.0\n"  # 带有尾随空格和制表符的测试数据字符串
    parser = all_parsers

    if parser.engine == "pyarrow":
        # 对于 pyarrow 引擎，验证是否会引发 ValueError 异常
        with pytest.raises(ValueError, match="the 'pyarrow' engine does not support"):
            parser.read_csv(StringIO(data.replace(",", "  ")), **kwargs)
        return
    
    # 生成预期的 DataFrame 对象
    expected = DataFrame(expected_data)
    # 使用给定参数解析数据，并验证结果是否与预期相符
    result = parser.read_csv(StringIO(data.replace(",", "  ")), **kwargs)
    tm.assert_frame_equal(result, expected)


def test_read_filepath_or_buffer(all_parsers):
    # 测试读取文件路径或缓冲区
    parser = all_parsers

    with pytest.raises(TypeError, match="Expected file path name or file-like"):
        # 验证是否会引发 TypeError 异常
        parser.read_csv(filepath_or_buffer=b"input")


def test_single_char_leading_whitespace(all_parsers):
    # 测试处理以空白字符开头的单字符情况
    parser = all_parsers
    data = """\
MyColumn
a
b
a
b\n"""

    if parser.engine == "pyarrow":
        # 对于 pyarrow 引擎，验证是否会引发 ValueError 异常
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(data),
                skipinitialspace=True,
            )
        return
    
    # 生成预期的 DataFrame 对象
    expected = DataFrame({"MyColumn": list("abab")})
    # 使用给定参数解析数据，并验证结果是否与预期相符
    result = parser.read_csv(StringIO(data), skipinitialspace=True, sep=r"\s+")
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "sep,skip_blank_lines,exp_data",
    [
        # 测试不同分隔符和是否跳过空白行的情况
        (",", True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]),
        (r"\s+", True, [[1.0, 2.0, 4.0], [5.0, np.nan, 10.0], [-70.0, 0.4, 1.0]]),
        (
            ",",
            False,
            [
                [1.0, 2.0, 4.0],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [5.0, np.nan, 10.0],
                [np.nan, np.nan, np.nan],
                [-70.0, 0.4, 1.0],
            ],
        ),
    ],
)
def test_empty_lines(all_parsers, sep, skip_blank_lines, exp_data, request):
    # 测试处理包含空白行的数据
    parser = all_parsers
    data = """\
A,B,C
1,2.,4.


5.,NaN,10.0

-70,.4,1
"""
    # 如果分隔符为正则表达式空白字符`\s+`，则将数据中的逗号替换为空格
    if sep == r"\s+":
        data = data.replace(",", "  ")

        # 如果解析引擎为"pyarrow"，则抛出值错误，显示消息"the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match="the 'pyarrow' engine does not support regex separators"):
            # 使用 parser 对象的 read_csv 方法读取 CSV 数据，传入的参数包括数据流对象、分隔符和是否跳过空行
            parser.read_csv(StringIO(data), sep=sep, skip_blank_lines=skip_blank_lines)
        # 函数返回，不再继续执行后续的数据读取操作
        return

    # 使用 parser 对象的 read_csv 方法读取 CSV 数据，传入的参数包括数据流对象、分隔符和是否跳过空行
    result = parser.read_csv(StringIO(data), sep=sep, skip_blank_lines=skip_blank_lines)
    # 创建预期的数据帧，列名为["A", "B", "C"]，数据为 exp_data
    expected = DataFrame(exp_data, columns=["A", "B", "C"])
    # 使用 pandas.testing 模块的 assert_frame_equal 方法比较实际结果和预期结果的数据帧是否相等
    tm.assert_frame_equal(result, expected)
@skip_pyarrow
# 定义一个测试函数，跳过 pyarrow 引擎的测试
def test_whitespace_lines(all_parsers):
    # 从参数中获取所有解析器
    parser = all_parsers
    # 定义包含空白和数据的字符串
    data = """

\t  \t\t
\t
A,B,C
\t    1,2.,4.
5.,NaN,10.0
"""
    # 预期的数据帧，包含特定格式的数据
    expected = DataFrame([[1, 2.0, 4.0], [5.0, np.nan, 10.0]], columns=["A", "B", "C"])
    # 使用解析器读取 CSV 数据
    result = parser.read_csv(StringIO(data))
    # 断言读取结果与预期数据帧相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            """   A   B   C   D
a   1   2   3   4
b   1   2   3   4
c   1   2   3   4
""",
            DataFrame(
                [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
                columns=["A", "B", "C", "D"],
                index=["a", "b", "c"],
            ),
        ),
        (
            "    a b c\n1 2 3 \n4 5  6\n 7 8 9",
            DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "b", "c"]),
        ),
    ],
)
# 参数化测试函数，针对不同的输入数据和预期结果进行测试
def test_whitespace_regex_separator(all_parsers, data, expected):
    # 查看 GitHub 问题号 6607
    parser = all_parsers
    # 如果解析器使用的引擎是 pyarrow
    if parser.engine == "pyarrow":
        # 抛出 ValueError 异常，显示不支持正则分隔符
        msg = "the 'pyarrow' engine does not support regex separators"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), sep=r"\s+")
        return

    # 使用解析器读取 CSV 数据，指定正则分隔符
    result = parser.read_csv(StringIO(data), sep=r"\s+")
    # 断言读取结果与预期数据帧相等
    tm.assert_frame_equal(result, expected)


def test_sub_character(all_parsers, csv_dir_path):
    # 查看 GitHub 问题号 16893
    # 定义 CSV 文件路径
    filename = os.path.join(csv_dir_path, "sub_char.csv")
    # 预期的数据帧，包含特定的列名
    expected = DataFrame([[1, 2, 3]], columns=["a", "\x1ab", "c"])

    parser = all_parsers
    # 使用解析器读取 CSV 文件
    result = parser.read_csv(filename)
    # 断言读取结果与预期数据帧相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("filename", ["sé-es-vé.csv", "ru-sй.csv", "中文文件名.csv"])
# 参数化测试函数，测试包含特殊字符的文件名
def test_filename_with_special_chars(all_parsers, filename):
    # 查看 GitHub 问题号 15086
    parser = all_parsers
    # 创建包含数据的数据帧
    df = DataFrame({"a": [1, 2, 3]})

    # 使用确保清理的方式，将数据帧写入 CSV 文件
    with tm.ensure_clean(filename) as path:
        df.to_csv(path, index=False)

        # 使用解析器读取 CSV 文件
        result = parser.read_csv(path)
        # 断言读取结果与原始数据帧相等
        tm.assert_frame_equal(result, df)


def test_read_table_same_signature_as_read_csv(all_parsers):
    # 查看 GitHub 问题号 34976
    parser = all_parsers

    # 获取 read_table 和 read_csv 的签名信息
    table_sign = signature(parser.read_table)
    csv_sign = signature(parser.read_csv)

    # 断言两者参数列表相同
    assert table_sign.parameters.keys() == csv_sign.parameters.keys()
    # 断言两者返回值注解相同
    assert table_sign.return_annotation == csv_sign.return_annotation

    # 遍历参数列表，对比各个参数的默认值和注解
    for key, csv_param in csv_sign.parameters.items():
        table_param = table_sign.parameters[key]
        # 对于分隔符参数，特别比较其默认值和注解
        if key == "sep":
            assert csv_param.default == ","
            assert table_param.default == "\t"
            assert table_param.annotation == csv_param.annotation
            assert table_param.kind == csv_param.kind
            continue

        # 对比其他参数是否相同
        assert table_param == csv_param


def test_read_table_equivalency_to_read_csv(all_parsers):
    # 查看 GitHub 问题号 21948
    # 从参数中获取所有解析器
    parser = all_parsers
    # 定义包含数据的字符串
    data = "a\tb\n1\t2\n3\t4"
    # 使用解析器读取 CSV 数据，指定分隔符为制表符
    expected = parser.read_csv(StringIO(data), sep="\t")
    # 使用解析器读取表格数据
    result = parser.read_table(StringIO(data))
    # 使用测试框架中的函数来比较两个数据框架是否相等，并断言它们相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器标记该函数为参数化测试，测试 read_csv 和 read_table 两个函数
@pytest.mark.parametrize("read_func", ["read_csv", "read_table"])
def test_read_csv_and_table_sys_setprofile(all_parsers, read_func):
    # 设置一个简单的测试数据
    data = "a b\n0 1"
    
    # 设置系统的 profile 为一个 lambda 函数，用于性能分析
    sys.setprofile(lambda *a, **k: None)
    
    # 调用 parser 对象的 read_func 方法（read_csv 或 read_table），解析测试数据
    result = getattr(all_parsers, read_func)(StringIO(data))
    
    # 取消系统的 profile 设置
    sys.setprofile(None)
    
    # 期望的 DataFrame 结果
    expected = DataFrame({"a b": ["0 1"]})
    
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


# 标记为 skip_pyarrow 的测试函数，不在 pyarrow 环境下执行
@skip_pyarrow
def test_first_row_bom(all_parsers):
    # 设置包含 BOM 的测试数据
    data = '''\ufeff"Head1"\t"Head2"\t"Head3"'''
    
    # 调用 parser 对象的 read_csv 方法，指定 delimiter="\t" 解析测试数据
    result = all_parsers.read_csv(StringIO(data), delimiter="\t")
    
    # 期望的 DataFrame 结果
    expected = DataFrame(columns=["Head1", "Head2", "Head3"])
    
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


# 标记为 skip_pyarrow 的测试函数，不在 pyarrow 环境下执行
@skip_pyarrow
def test_first_row_bom_unquoted(all_parsers):
    # 设置包含 BOM 的无引号测试数据
    data = """\ufeffHead1\tHead2\tHead3"""
    
    # 调用 parser 对象的 read_csv 方法，指定 delimiter="\t" 解析测试数据
    result = all_parsers.read_csv(StringIO(data), delimiter="\t")
    
    # 期望的 DataFrame 结果
    expected = DataFrame(columns=["Head1", "Head2", "Head3"])
    
    # 断言结果与期望一致
    tm.assert_frame_equal(result, expected)


# 使用 pytest.mark.parametrize 装饰器标记该函数为参数化测试，测试不同行数下的空白行处理
@pytest.mark.parametrize("nrows", range(1, 6))
def test_blank_lines_between_header_and_data_rows(all_parsers, nrows):
    # 设置参考的 DataFrame 结果，包含 NaN 值的数据
    ref = DataFrame(
        [[np.nan, np.nan], [np.nan, np.nan], [1, 2], [np.nan, np.nan], [3, 4]],
        columns=list("ab"),
    )
    
    # 设置包含空白行的 CSV 数据
    csv = "\nheader\n\na,b\n\n\n1,2\n\n3,4"
    parser = all_parsers

    # 对于 pyarrow 引擎，验证不支持 nrows 参数的情况
    if parser.engine == "pyarrow":
        msg = "The 'nrows' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(
                StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False
            )
        return

    # 调用 parser 对象的 read_csv 方法，指定 header=3, nrows=nrows, skip_blank_lines=False 解析测试数据
    df = parser.read_csv(StringIO(csv), header=3, nrows=nrows, skip_blank_lines=False)
    
    # 断言结果与期望一致
    tm.assert_frame_equal(df, ref[:nrows])


# 标记为 skip_pyarrow 的测试函数，不在 pyarrow 环境下执行
@skip_pyarrow
def test_no_header_two_extra_columns(all_parsers):
    # 设置列名及对应的测试数据
    column_names = ["one", "two", "three"]
    ref = DataFrame([["foo", "bar", "baz"]], columns=column_names)
    stream = StringIO("foo,bar,baz,bam,blah")
    parser = all_parsers
    
    # 调用 parser 对象的 read_csv_check_warnings 方法，验证警告信息和测试数据，指定 header=None, names=column_names, index_col=False 解析测试数据
    df = parser.read_csv_check_warnings(
        ParserWarning,
        "Length of header or names does not match length of data. "
        "This leads to a loss of data with index_col=False.",
        stream,
        header=None,
        names=column_names,
        index_col=False,
    )
    
    # 断言结果与期望一致
    tm.assert_frame_equal(df, ref)


def test_read_csv_names_not_accepting_sets(all_parsers):
    # 设置包含非有序集合的列名测试数据
    data = """\
    1,2,3
    4,5,6\n"""
    parser = all_parsers
    
    # 验证当 names 参数为集合时会引发 ValueError 异常
    with pytest.raises(ValueError, match="Names should be an ordered collection."):
        parser.read_csv(StringIO(data), names=set("QAZ"))


def test_read_csv_delimiter_and_sep_no_default(all_parsers):
    # 设置包含 sep 和 delimiter 同时指定的测试数据
    f = StringIO("a,b\n1,2")
    parser = all_parsers
    
    # 验证当同时指定 sep 和 delimiter 参数时会引发 ValueError 异常
    msg = "Specified a sep and a delimiter; you can only specify one."
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(f, sep=" ", delimiter=".")
# 使用 pytest 的参数化装饰器，为 test_read_csv_line_break_as_separator 函数提供两个参数化的测试用例
@pytest.mark.parametrize("kwargs", [{"delimiter": "\n"}, {"sep": "\n"}])
def test_read_csv_line_break_as_separator(kwargs, all_parsers):
    # 标识 GitHub 问题编号 GH#43528
    # 选择使用给定的解析器对象
    parser = all_parsers
    # 定义包含换行符的数据字符串
    data = """a,b,c
1,2,3
    """
    # 定义错误消息字符串，用于匹配引发的 ValueError 异常
    msg = (
        r"Specified \\n as separator or delimiter. This forces the python engine "
        r"which does not accept a line terminator. Hence it is not allowed to use "
        r"the line terminator as separator."
    )
    # 使用 pytest 的 raises 方法检查是否引发指定异常，并匹配错误消息
    with pytest.raises(ValueError, match=msg):
        parser.read_csv(StringIO(data), **kwargs)


# 标记为跳过 pyarrow 的测试用例
@skip_pyarrow
def test_dict_keys_as_names(all_parsers):
    # 标识 GitHub 问题编号 GH: 36928
    # 定义包含数据的字符串
    data = "1,2"

    # 使用字典的 keys 方法获取列名
    keys = {"a": int, "b": int}.keys()
    # 选择使用给定的解析器对象
    parser = all_parsers

    # 使用指定的列名解析 CSV 数据，并存储结果
    result = parser.read_csv(StringIO(data), names=keys)
    # 创建预期的 DataFrame 对象
    expected = DataFrame({"a": [1], "b": [2]})
    # 使用 pandas 的 assert_frame_equal 方法检查结果与预期是否一致
    tm.assert_frame_equal(result, expected)


# 标记为预期失败（xfail）pyarrow 的测试用例，注明 UnicodeDecodeError 的具体错误信息
@xfail_pyarrow  # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xed in position 0
def test_encoding_surrogatepass(all_parsers):
    # 标识 GitHub 问题编号 GH39017
    # 选择使用给定的解析器对象
    parser = all_parsers
    # 定义包含字节数据的变量
    content = b"\xed\xbd\xbf"
    # 使用 surrogatepass 错误处理方式解码字节数据
    decoded = content.decode("utf-8", errors="surrogatepass")
    # 创建预期的 DataFrame 对象，使用解码后的内容作为列名和索引名
    expected = DataFrame({decoded: [decoded]}, index=[decoded * 2])
    expected.index.name = decoded * 2

    # 使用 tm.ensure_clean 上下文管理器，写入临时文件并验证读取结果
    with tm.ensure_clean() as path:
        # 将数据写入文件中，包括指定的错误处理方式
        Path(path).write_bytes(
            content * 2 + b"," + content + b"\n" + content * 2 + b"," + content
        )
        # 使用指定的编码错误处理方式读取 CSV 文件，将结果与预期进行比较
        df = parser.read_csv(path, encoding_errors="surrogatepass", index_col=0)
        tm.assert_frame_equal(df, expected)
        # 使用 pytest 的 raises 方法检查是否引发 UnicodeDecodeError 异常
        with pytest.raises(UnicodeDecodeError, match="'utf-8' codec can't decode byte"):
            parser.read_csv(path)


def test_malformed_second_line(all_parsers):
    # 标识 GitHub 问题编号 GH14782
    # 选择使用给定的解析器对象
    parser = all_parsers
    # 定义包含换行符的数据字符串，包括空行
    data = "\na\nb\n"
    # 使用指定的参数解析 CSV 数据，并存储结果
    result = parser.read_csv(StringIO(data), skip_blank_lines=False, header=1)
    # 创建预期的 DataFrame 对象
    expected = DataFrame({"a": ["b"]})
    # 使用 pandas 的 assert_frame_equal 方法检查结果与预期是否一致
    tm.assert_frame_equal(result, expected)


# 标记为跳过 pyarrow 的测试用例
@skip_pyarrow
def test_short_single_line(all_parsers):
    # 标识 GitHub 问题编号 GH 47566
    # 选择使用给定的解析器对象
    parser = all_parsers
    # 定义列名的列表
    columns = ["a", "b", "c"]
    # 定义包含部分数据的字符串
    data = "1,2"
    # 使用指定的参数解析 CSV 数据，并存储结果
    result = parser.read_csv(StringIO(data), header=None, names=columns)
    # 创建预期的 DataFrame 对象
    expected = DataFrame({"a": [1], "b": [2], "c": [np.nan]})
    # 使用 pandas 的 assert_frame_equal 方法检查结果与预期是否一致
    tm.assert_frame_equal(result, expected)


# 标记为预期失败（xfail）pyarrow 的测试用例，注明 ValueError 异常的具体错误信息
@xfail_pyarrow  # ValueError: Length mismatch: Expected axis has 2 elements
def test_short_multi_line(all_parsers):
    # 标识 GitHub 问题编号 GH 47566
    # 选择使用给定的解析器对象
    parser = all_parsers
    # 定义列名的列表
    columns = ["a", "b", "c"]
    # 定义包含多行数据的字符串
    data = "1,2\n1,2"
    # 使用指定的参数解析 CSV 数据，并存储结果
    result = parser.read_csv(StringIO(data), header=None, names=columns)
    # 创建预期的 DataFrame 对象
    expected = DataFrame({"a": [1, 1], "b": [2, 2], "c": [np.nan, np.nan]})
    # 使用 pandas 的 assert_frame_equal 方法检查结果与预期是否一致
    tm.assert_frame_equal(result, expected)


def test_read_seek(all_parsers):
    # 标识 GitHub 问题编号 GH48646
    # 选择使用给定的解析器对象
    parser = all_parsers
    # 定义包含前缀的字符串
    prefix = "### DATA\n"
    # 定义包含数据的字符串
    content = "nkey,value\ntables,rectangular\n"
    # 使用 tm.ensure_clean() 确保操作环境的清洁性，并将返回的路径赋值给 path 变量
    with tm.ensure_clean() as path:
        # 创建路径对象，并将带有特定前缀和内容的文本写入该路径，使用 UTF-8 编码
        Path(path).write_text(prefix + content, encoding="utf-8")
        
        # 打开指定路径的文件，使用 UTF-8 编码，并读取文件的第一行内容（即跳过第一行标题行）
        with open(path, encoding="utf-8") as file:
            # 从文件对象中读取一行内容，通常用于跳过标题行等情况
            file.readline()
            # 使用 parser.read_csv() 读取文件内容并解析为数据框架（DataFrame）
            actual = parser.read_csv(file)
        
        # 使用 StringIO 将 content 字符串模拟为文件输入，然后通过 parser.read_csv() 读取并解析为数据框架
        expected = parser.read_csv(StringIO(content))
    
    # 使用 tm.assert_frame_equal() 检查 actual 和 expected 数据框架是否相等
    tm.assert_frame_equal(actual, expected)
```
# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_read_fwf.py`

```
"""
Tests the 'read_fwf' function in parsers.py. This
test suite is independent of the others because the
engine is set to 'python-fwf' internally.
"""

# 导入所需的库和模块
from io import (
    BytesIO,    # 导入 BytesIO 类，用于处理二进制数据
    StringIO,   # 导入 StringIO 类，用于处理字符串数据
)
from pathlib import Path

import numpy as np
import pytest

from pandas.errors import EmptyDataError

import pandas as pd
from pandas import (
    DataFrame,       # 导入 DataFrame 类，用于处理二维表数据
    DatetimeIndex,   # 导入 DatetimeIndex 类，用于处理日期时间索引
)
import pandas._testing as tm
from pandas.core.arrays import (
    ArrowStringArray,   # 导入 ArrowStringArray 类，用于处理 Arrow 字符串数组
    StringArray,        # 导入 StringArray 类，用于处理字符串数组
)

from pandas.io.common import urlopen
from pandas.io.parsers import (
    read_csv,    # 导入 read_csv 函数，用于读取 CSV 文件
    read_fwf,    # 导入 read_fwf 函数，用于读取固定宽度格式数据
)


def test_basic():
    data = """\
A         B            C            D
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    # 调用 read_fwf 函数读取数据
    result = read_fwf(StringIO(data))
    # 创建期望结果的 DataFrame
    expected = DataFrame(
        [
            [201158, 360.242940, 149.910199, 11950.7],
            [201159, 444.953632, 166.985655, 11788.4],
            [201160, 364.136849, 183.628767, 11806.2],
            [201161, 413.836124, 184.375703, 11916.8],
            [201162, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D"],
    )
    # 使用 assert_frame_equal 检查结果是否符合预期
    tm.assert_frame_equal(result, expected)


def test_colspecs():
    data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]
    # 使用指定的 colspecs 参数调用 read_fwf 函数读取数据
    result = read_fwf(StringIO(data), colspecs=colspecs)

    expected = DataFrame(
        [
            [2011, 58, 360.242940, 149.910199, 11950.7],
            [2011, 59, 444.953632, 166.985655, 11788.4],
            [2011, 60, 364.136849, 183.628767, 11806.2],
            [2011, 61, 413.836124, 184.375703, 11916.8],
            [2011, 62, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    tm.assert_frame_equal(result, expected)


def test_widths():
    data = """\
A    B    C            D            E
2011 58   360.242940   149.910199   11950.7
2011 59   444.953632   166.985655   11788.4
2011 60   364.136849   183.628767   11806.2
2011 61   413.836124   184.375703   11916.8
2011 62   502.953953   173.237159   12468.3
"""
    # 使用指定的 widths 参数调用 read_fwf 函数读取数据
    result = read_fwf(StringIO(data), widths=[5, 5, 13, 13, 7])

    expected = DataFrame(
        [
            [2011, 58, 360.242940, 149.910199, 11950.7],
            [2011, 59, 444.953632, 166.985655, 11788.4],
            [2011, 60, 364.136849, 183.628767, 11806.2],
            [2011, 61, 413.836124, 184.375703, 11916.8],
            [2011, 62, 502.953953, 173.237159, 12468.3],
        ],
        columns=["A", "B", "C", "D", "E"],
    )
    tm.assert_frame_equal(result, expected)


def test_non_space_filler():
    # 以下是一段字符串数据，通常用作示例文本或测试数据
    # 这段字符串数据包含了一个文本块，看起来似乎有些非空格填充字符
    # 在处理类似文本块时，可以通过指定特定的分隔符来支持其它字符
    # 参考链接：http://publib.boulder.ibm.com/infocenter/dmndhelp/v6r1mx/index.jsp?topic=/com.ibm.wbit.612.help.config.doc/topics/rfixwidth.html
    data = """\
# 定义一个多行字符串，包含固定宽度格式数据
fwf_data = """\
A   B     C            D            E
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""

# 定义固定列宽度的范围，每个元组表示每列的起始和结束位置（包括起始位置但不包括结束位置）
colspecs = [(0, 4), (4, 8), (8, 20), (21, 33), (34, 43)]

# 调用read_fwf函数读取固定宽度格式数据，使用指定的列宽度范围和分隔符
result = read_fwf(StringIO(fwf_data), colspecs=colspecs)

# 创建期望的DataFrame对象，包含与fwf_data对应的数据结构和内容
expected = DataFrame(
    [
        [2011, 58, 360.242940, 149.910199, 11950.7],
        [2011, 59, 444.953632, 166.985655, 11788.4],
        [2011, 60, 364.136849, 183.628767, 11806.2],
        [2011, 61, 413.836124, 184.375703, 11916.8],
        [2011, 62, 502.953953, 173.237159, 12468.3],
    ],
    columns=["A", "B", "C", "D", "E"],
)

# 使用pytest的tm.assert_frame_equal断言方法比较result和expected是否相等
tm.assert_frame_equal(result, expected)
    # 错误消息字符串，用于匹配 pytest 抛出的异常信息
    msg = "column specifications must be a list or tuple.+"
    
    # 使用 pytest 的上下文管理器检查 read_fwf 函数调用时是否会抛出指定类型的异常，并验证异常消息是否与 msg 匹配
    with pytest.raises(TypeError, match=msg):
        # 调用 read_fwf 函数，传入参数：
        # StringIO(data): 用 data 字符串创建的内存文件对象，模拟文件输入
        # colspecs={"a": 1}: 指定的列规范，这里传入了一个字典而不是要求的列表或元组，会触发 TypeError 异常
        # delimiter=",": 列的分隔符，用于解析数据
        read_fwf(StringIO(data), colspecs={"a": 1}, delimiter=",")
def test_fwf_colspecs_is_list_or_tuple_of_two_element_tuples():
    # 准备测试数据
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
"""

    # 预期的错误消息
    msg = "Each column specification must be.+"

    # 使用 pytest 检查是否会抛出 TypeError 异常，并匹配预期错误消息
    with pytest.raises(TypeError, match=msg):
        # 调用 read_fwf 函数，传入 StringIO 对象和不符合规范的 colspecs 参数
        read_fwf(StringIO(data), colspecs=[("a", 1)])


@pytest.mark.parametrize(
    "colspecs,exp_data",
    [
        # 第一组测试数据
        ([(0, 3), (3, None)], [[123, 456], [456, 789]]),
        # 第二组测试数据
        ([(None, 3), (3, 6)], [[123, 456], [456, 789]]),
        # 第三组测试数据
        ([(0, None), (3, None)], [[123456, 456], [456789, 789]]),
        # 第四组测试数据
        ([(None, None), (3, 6)], [[123456, 456], [456789, 789]]),
    ],
)
def test_fwf_colspecs_none(colspecs, exp_data):
    # see gh-7079
    # 准备测试数据
    data = """\
123456
456789
"""
    # 期望的结果 DataFrame
    expected = DataFrame(exp_data)

    # 调用 read_fwf 函数，传入 StringIO 对象和 colspecs 参数，禁用列头
    result = read_fwf(StringIO(data), colspecs=colspecs, header=None)
    # 使用 pytest 检查结果是否与期望相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "infer_nrows,exp_data",
    [
        # infer_nrows --> colspec == [(2, 3), (5, 6)]
        (1, [[1, 2], [3, 8]]),
        # infer_nrows > number of rows
        (10, [[1, 2], [123, 98]]),
    ],
)
def test_fwf_colspecs_infer_nrows(infer_nrows, exp_data):
    # see gh-15138
    # 准备测试数据
    data = """\
  1  2
123 98
"""
    # 期望的结果 DataFrame
    expected = DataFrame(exp_data)

    # 调用 read_fwf 函数，传入 StringIO 对象和 infer_nrows 参数，禁用列头
    result = read_fwf(StringIO(data), infer_nrows=infer_nrows, header=None)
    # 使用 pytest 检查结果是否与期望相等
    tm.assert_frame_equal(result, expected)


def test_fwf_regression():
    # see gh-3594
    #
    # Turns out "T060" is parsable as a datetime slice!
    # 准备测试数据
    tz_list = [1, 10, 20, 30, 60, 80, 100]
    widths = [16] + [8] * len(tz_list)
    names = ["SST"] + [f"T{z:03d}" for z in tz_list[1:]]

    data = """  2009164202000   9.5403  9.4105  8.6571  7.8372  6.0612  5.8843  5.5192
2009164203000   9.5435  9.2010  8.6167  7.8176  6.0804  5.8728  5.4869
2009164204000   9.5873  9.1326  8.4694  7.5889  6.0422  5.8526  5.4657
2009164205000   9.5810  9.0896  8.4009  7.4652  6.0322  5.8189  5.4379
2009164210000   9.6034  9.0897  8.3822  7.4905  6.0908  5.7904  5.4039
"""
    # 期望的结果 DataFrame
    expected = DataFrame(
        [
            [9.5403, 9.4105, 8.6571, 7.8372, 6.0612, 5.8843, 5.5192],
            [9.5435, 9.2010, 8.6167, 7.8176, 6.0804, 5.8728, 5.4869],
            [9.5873, 9.1326, 8.4694, 7.5889, 6.0422, 5.8526, 5.4657],
            [9.5810, 9.0896, 8.4009, 7.4652, 6.0322, 5.8189, 5.4379],
            [9.6034, 9.0897, 8.3822, 7.4905, 6.0908, 5.7904, 5.4039],
        ],
        index=DatetimeIndex(
            [
                "2009-06-13 20:20:00",
                "2009-06-13 20:30:00",
                "2009-06-13 20:40:00",
                "2009-06-13 20:50:00",
                "2009-06-13 21:00:00",
            ],
            dtype="M8[us]",
        ),
        columns=["SST", "T010", "T020", "T030", "T060", "T080", "T100"],
    )

    # 调用 read_fwf 函数，传入 StringIO 对象和其他必要参数，解析日期，并传入 datetime 格式
    result = read_fwf(
        StringIO(data),
        index_col=0,
        header=None,
        names=names,
        widths=widths,
        parse_dates=True,
        date_format="%Y%j%H%M%S",
    )
    # 将期望结果的索引转换为日期时间类型（精确到秒）
    expected.index = expected.index.astype("M8[s]")
    # 使用测试框架中的函数比较两个数据框架（DataFrame）result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试读取固定宽度格式（FWF）数据的处理情况，数据保存在字符串变量中
def test_fwf_for_uint8():
    data = """1421302965.213420    PRI=3 PGN=0xef00      DST=0x17 SRC=0x28    04 154 00 00 00 00 00 127
1421302964.226776    PRI=6 PGN=0xf002               SRC=0x47    243 00 00 255 247 00 00 71"""  # noqa: E501
    # 使用 read_fwf 函数读取数据并处理
    df = read_fwf(
        StringIO(data),  # 将数据从字符串转为文件对象
        colspecs=[(0, 17), (25, 26), (33, 37), (49, 51), (58, 62), (63, 1000)],  # 定义列的起始和结束位置
        names=["time", "pri", "pgn", "dst", "src", "data"],  # 列名列表
        converters={  # 数据类型转换器字典
            "pgn": lambda x: int(x, 16),  # 将 pgn 列的十六进制字符串转为整数
            "src": lambda x: int(x, 16),  # 将 src 列的十六进制字符串转为整数
            "dst": lambda x: int(x, 16),  # 将 dst 列的十六进制字符串转为整数
            "data": lambda x: len(x.split(" ")),  # 计算 data 列的空格分隔数
        },
    )

    expected = DataFrame(  # 预期的数据帧
        [
            [1421302965.213420, 3, 61184, 23, 40, 8],  # 第一行数据
            [1421302964.226776, 6, 61442, None, 71, 8],  # 第二行数据
        ],
        columns=["time", "pri", "pgn", "dst", "src", "data"],  # 列名列表
    )
    expected["dst"] = expected["dst"].astype(object)  # 将 dst 列转为对象类型
    tm.assert_frame_equal(df, expected)  # 使用测试工具比较 df 和 expected


@pytest.mark.parametrize("comment", ["#", "~", "!"])  # 使用参数化装饰器定义不同的注释字符
def test_fwf_comment(comment):
    data = """\
  1   2.   4  #hello world
  5  NaN  10.0
"""
    data = data.replace("#", comment)  # 将数据中的 # 替换为指定的注释字符

    colspecs = [(0, 3), (4, 9), (9, 25)]  # 定义列的起始和结束位置
    expected = DataFrame([[1, 2.0, 4], [5, np.nan, 10.0]])  # 预期的数据帧

    result = read_fwf(StringIO(data), colspecs=colspecs, header=None, comment=comment)  # 读取并处理数据，指定注释字符
    tm.assert_almost_equal(result, expected)  # 使用测试工具比较 result 和 expected


# 测试读取固定宽度格式（FWF）数据时跳过空行的情况
def test_fwf_skip_blank_lines():
    data = """

A         B            C            D

201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4


201162    502.953953   173.237159   12468.3

"""
    result = read_fwf(StringIO(data), skip_blank_lines=True)  # 跳过空行读取数据
    expected = DataFrame(  # 预期的数据帧
        [
            [201158, 360.242940, 149.910199, 11950.7],  # 第一行数据
            [201159, 444.953632, 166.985655, 11788.4],  # 第二行数据
            [201162, 502.953953, 173.237159, 12468.3],  # 第三行数据
        ],
        columns=["A", "B", "C", "D"],  # 列名列表
    )
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较 result 和 expected

    data = """\
A         B            C            D
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4


201162    502.953953   173.237159   12468.3
"""
    result = read_fwf(StringIO(data), skip_blank_lines=False)  # 包括空行读取数据
    expected = DataFrame(  # 预期的数据帧
        [
            [201158, 360.242940, 149.910199, 11950.7],  # 第一行数据
            [201159, 444.953632, 166.985655, 11788.4],  # 第二行数据
            [np.nan, np.nan, np.nan, np.nan],  # 空行数据
            [np.nan, np.nan, np.nan, np.nan],  # 空行数据
            [201162, 502.953953, 173.237159, 12468.3],  # 第三行数据
        ],
        columns=["A", "B", "C", "D"],  # 列名列表
    )
    tm.assert_frame_equal(result, expected)  # 使用测试工具比较 result 和 expected


@pytest.mark.parametrize("thousands", [",", "#", "~"])  # 使用参数化装饰器定义不同的千位分隔符
def test_fwf_thousands(thousands):
    data = """\
 1 2,334.0    5
10   13     10.
"""
    data = data.replace(",", thousands)  # 将数据中的逗号替换为指定的千位分隔符

    colspecs = [(0, 3), (3, 11), (12, 16)]  # 定义列的起始和结束位置
    expected = DataFrame([[1, 2334.0, 5], [10, 13, 10.0]])  # 预期的数据帧

    result = read_fwf(
        StringIO(data), header=None, colspecs=colspecs, thousands=thousands
    )  # 读取并处理数据，指定千位分隔符
    # 使用测试工具 `tm` 中的 `assert_almost_equal` 函数比较 `result` 和 `expected` 的值是否几乎相等。
    tm.assert_almost_equal(result, expected)
@pytest.mark.parametrize("header", [True, False])
def test_bool_header_arg(header):
    # 使用 pytest 的 parametrize 装饰器为测试函数提供两种不同的 header 参数值：True 和 False
    # 定义测试数据，包含一个带有列标题的简单文本表格
    data = """\
MyColumn
   a
   b
   a
   b"""
    
    msg = "Passing a bool to header is invalid"
    # 使用 pytest.raises 检查 read_fwf 函数对 header 参数传入布尔值时是否会抛出 TypeError 异常
    with pytest.raises(TypeError, match=msg):
        read_fwf(StringIO(data), header=header)


def test_full_file():
    # 定义测试数据，包含完整的带有索引和多列数据的文本表格
    test = """index                             A    B    C
2000-01-03T00:00:00  0.980268513777    3  foo
2000-01-04T00:00:00  1.04791624281    -4  bar
2000-01-05T00:00:00  0.498580885705   73  baz
2000-01-06T00:00:00  1.12020151869     1  foo
2000-01-07T00:00:00  0.487094399463    0  bar
2000-01-10T00:00:00  0.836648671666    2  baz
2000-01-11T00:00:00  0.157160753327   34  foo"""
    colspecs = ((0, 19), (21, 35), (38, 40), (42, 45))
    # 通过 read_fwf 函数读取指定列宽的文本数据，生成预期结果
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    # 使用 read_fwf 函数读取未指定列宽的同一份测试数据，生成实际结果
    result = read_fwf(StringIO(test))
    # 使用 tm.assert_frame_equal 检查实际结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)


def test_full_file_with_missing():
    # 定义测试数据，包含带有缺失值的文本表格
    test = """index                             A    B    C
2000-01-03T00:00:00  0.980268513777    3  foo
2000-01-04T00:00:00  1.04791624281    -4  bar
                     0.498580885705   73  baz
2000-01-06T00:00:00  1.12020151869     1  foo
2000-01-07T00:00:00                    0  bar
2000-01-10T00:00:00  0.836648671666    2  baz
                                      34"""
    colspecs = ((0, 19), (21, 35), (38, 40), (42, 45))
    # 通过 read_fwf 函数读取指定列宽的文本数据，生成预期结果
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    # 使用 read_fwf 函数读取未指定列宽的同一份测试数据，生成实际结果
    result = read_fwf(StringIO(test))
    # 使用 tm.assert_frame_equal 检查实际结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)


def test_full_file_with_spaces():
    # 定义测试数据，包含带有空格的列名和数据的文本表格
    test = """
Account                 Name  Balance     CreditLimit   AccountCreated
101     Keanu Reeves          9315.45     10000.00           1/17/1998
312     Gerard Butler         90.00       1000.00             8/6/2003
868     Jennifer Love Hewitt  0           17000.00           5/25/1985
761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006
317     Bill Murray           789.65      5000.00             2/5/2007
""".strip("\r\n")
    colspecs = ((0, 7), (8, 28), (30, 38), (42, 53), (56, 70))
    # 通过 read_fwf 函数读取指定列宽的文本数据，生成预期结果
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    # 使用 read_fwf 函数读取未指定列宽的同一份测试数据，生成实际结果
    result = read_fwf(StringIO(test))
    # 使用 tm.assert_frame_equal 检查实际结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)


def test_full_file_with_spaces_and_missing():
    # 定义测试数据，包含带有空格和缺失值的文本表格
    test = """
Account               Name    Balance     CreditLimit   AccountCreated
101                           10000.00                       1/17/1998
312     Gerard Butler         90.00       1000.00             8/6/2003
868                                                          5/25/1985
761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006
317     Bill Murray           789.65
""".strip("\r\n")
    colspecs = ((0, 7), (8, 28), (30, 38), (42, 53), (56, 70))
    # 通过 read_fwf 函数读取指定列宽的文本数据，生成预期结果
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    # 使用 read_fwf 函数读取未指定列宽的同一份测试数据，生成实际结果
    result = read_fwf(StringIO(test))
    # 实际结果和预期结果的比较将在外部进行
    # 使用 pandas.testing.assert_frame_equal 函数比较 result 和 expected 两个数据框是否相等
    tm.assert_frame_equal(result, expected)
# 测试混乱数据的情况
def test_messed_up_data():
    # 完全混乱的文件数据
    test = """
   Account          Name             Balance     Credit Limit   Account Created
       101                           10000.00                       1/17/1998
       312     Gerard Butler         90.00       1000.00

       761     Jada Pinkett-Smith    49654.87    100000.00          12/5/2006
  317          Bill Murray           789.65
""".strip("\r\n")
    colspecs = ((2, 10), (15, 33), (37, 45), (49, 61), (64, 79))
    expected = read_fwf(StringIO(test), colspecs=colspecs)

    result = read_fwf(StringIO(test))
    tm.assert_frame_equal(result, expected)


# 测试多种分隔符的情况
def test_multiple_delimiters():
    test = r"""
col1~~~~~col2  col3++++++++++++++++++col4
~~22.....11.0+++foo~~~~~~~~~~Keanu Reeves
  33+++122.33\\\bar.........Gerard Butler
++44~~~~12.01   baz~~Jennifer Love Hewitt
~~55       11+++foo++++Jada Pinkett-Smith
..66++++++.03~~~bar           Bill Murray
""".strip("\r\n")
    delimiter = " +~.\\"
    colspecs = ((0, 4), (7, 13), (15, 19), (21, 41))
    expected = read_fwf(StringIO(test), colspecs=colspecs, delimiter=delimiter)

    result = read_fwf(StringIO(test), delimiter=delimiter)
    tm.assert_frame_equal(result, expected)


# 测试变宽度的 Unicode 数据
def test_variable_width_unicode():
    data = """
שלום שלום
ום   שלל
של   ום
""".strip("\r\n")
    encoding = "utf8"
    kwargs = {"header": None, "encoding": encoding}

    expected = read_fwf(
        BytesIO(data.encode(encoding)), colspecs=[(0, 4), (5, 9)], **kwargs
    )
    result = read_fwf(BytesIO(data.encode(encoding)), **kwargs)
    tm.assert_frame_equal(result, expected)


# 测试指定数据类型的情况
@pytest.mark.parametrize("dtype", [{}, {"a": "float64", "b": str, "c": "int32"}])
def test_dtype(dtype):
    data = """ a    b    c
1    2    3.2
3    4    5.2
"""
    colspecs = [(0, 5), (5, 10), (10, None)]
    result = read_fwf(StringIO(data), colspecs=colspecs, dtype=dtype)

    expected = DataFrame(
        {"a": [1, 3], "b": [2, 4], "c": [3.2, 5.2]}, columns=["a", "b", "c"]
    )

    for col, dt in dtype.items():
        expected[col] = expected[col].astype(dt)

    tm.assert_frame_equal(result, expected)


# 测试推断跳过行数的情况
def test_skiprows_inference():
    # 参见 gh-11256
    data = """
Text contained in the file header

DataCol1   DataCol2
     0.0        1.0
   101.6      956.1
""".strip()
    skiprows = 2

    expected = read_csv(StringIO(data), skiprows=skiprows, sep=r"\s+")
    result = read_fwf(StringIO(data), skiprows=skiprows)
    tm.assert_frame_equal(result, expected)


# 测试按索引推断跳过行数的情况
def test_skiprows_by_index_inference():
    data = """
To be skipped
Not  To  Be  Skipped
Once more to be skipped
123  34   8      123
456  78   9      456
""".strip()
    skiprows = [0, 2]

    expected = read_csv(StringIO(data), skiprows=skiprows, sep=r"\s+")
    result = read_fwf(StringIO(data), skiprows=skiprows)
    tm.assert_frame_equal(result, expected)


# 测试推断跳过行数的情况（空测试用例）
def test_skiprows_inference_empty():
    data = """
AA   BBB  C
12   345  6
78   901  2
""".strip()
    # 定义错误消息字符串，用于匹配 pytest 抛出的异常信息
    msg = "No rows from which to infer column width"
    
    # 使用 pytest 的上下文管理器，检测 read_fwf 函数是否会抛出 EmptyDataError 异常，
    # 并且异常信息要与预设的 msg 变量相匹配
    with pytest.raises(EmptyDataError, match=msg):
        # 调用 read_fwf 函数，传入数据流 StringIO(data)，并跳过前三行
        read_fwf(StringIO(data), skiprows=3)
# 测试函数：测试读取固定宽度格式（FWF）数据时是否能保留空白字符
def test_whitespace_preservation():
    # GitHub issue 16772，这里设置 header 为 None
    header = None
    # CSV 格式的数据
    csv_data = """
 a ,bbb
 cc,dd """
    
    # FWF 格式的数据
    fwf_data = """
 a bbb
 ccdd """
    
    # 调用 read_fwf 函数读取 FWF 数据
    result = read_fwf(
        StringIO(fwf_data), widths=[3, 3], header=header, skiprows=[0], delimiter="\n\t"
    )
    
    # 期望的结果是通过 read_csv 函数读取 CSV 数据
    expected = read_csv(StringIO(csv_data), header=header)
    
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试默认分隔符
def test_default_delimiter():
    # 设置 header 为 None
    header = None
    # CSV 格式的数据
    csv_data = """
a,bbb
cc,dd"""
    
    # FWF 格式的数据，使用 \t 分隔符
    fwf_data = """
a \tbbb
cc\tdd """
    
    # 调用 read_fwf 函数读取 FWF 数据
    result = read_fwf(StringIO(fwf_data), widths=[3, 3], header=header, skiprows=[0])
    
    # 期望的结果是通过 read_csv 函数读取 CSV 数据
    expected = read_csv(StringIO(csv_data), header=header)
    
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数：测试 FWF 数据压缩
@pytest.mark.parametrize("infer", [True, False])
def test_fwf_compression(compression_only, infer, compression_to_extension):
    # 原始数据
    data = """1111111111
    2222222222
    3333333333""".strip()

    # 获取压缩方式和对应的扩展名
    compression = compression_only
    extension = compression_to_extension[compression]

    # 设置读取参数
    kwargs = {"widths": [5, 5], "names": ["one", "two"]}
    # 期望的结果
    expected = read_fwf(StringIO(data), **kwargs)

    # 将数据转换为字节流
    data = bytes(data, encoding="utf-8")

    # 使用 ensure_clean 确保文件不存在
    with tm.ensure_clean(filename="tmp." + extension) as path:
        # 将数据写入压缩文件
        tm.write_to_compressed(compression, path, data)

        # 如果 infer 不为空，则根据情况设置压缩方式
        if infer is not None:
            kwargs["compression"] = "infer" if infer else compression

        # 读取压缩文件并进行 FWF 解析
        result = read_fwf(path, **kwargs)
        
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)


# 测试函数：测试二进制模式下的数据读取
def test_binary_mode():
    """
    read_fwf supports opening files in binary mode.

    GH 18035.
    """
    # 数据
    data = """aaa aaa aaa
bba bab b a"""
    # 参考 DataFrame
    df_reference = DataFrame(
        [["bba", "bab", "b a"]], columns=["aaa", "aaa.1", "aaa.2"], index=[0]
    )
    
    # 使用 ensure_clean 确保文件不存在
    with tm.ensure_clean() as path:
        # 将数据写入文件
        Path(path).write_text(data, encoding="utf-8")
        
        # 以二进制模式打开文件并读取
        with open(path, "rb") as file:
            # 使用 read_fwf 函数读取数据
            df = read_fwf(file)
            # 回到文件起始位置
            file.seek(0)
            # 断言两个 DataFrame 是否相等
            tm.assert_frame_equal(df, df_reference)


# 测试函数：测试使用内存映射文件时的编码
@pytest.mark.parametrize("memory_map", [True, False])
def test_encoding_mmap(memory_map):
    """
    encoding should be working, even when using a memory-mapped file.

    GH 23254.
    """
    # 设置编码方式
    encoding = "iso8859_1"
    
    # 使用 ensure_clean 确保文件不存在
    with tm.ensure_clean() as path:
        # 将数据写入文件
        Path(path).write_bytes(" 1 A Ä 2\n".encode(encoding))
        
        # 使用 read_fwf 函数读取数据
        df = read_fwf(
            path,
            header=None,
            widths=[2, 2, 2, 2],
            encoding=encoding,
            memory_map=memory_map,
        )
    
    # 期望的结果
    df_reference = DataFrame([[1, "A", "Ä", 2]])
    
    # 断言两个 DataFrame 是否相等
    tm.assert_frame_equal(df, df_reference)
    [
        # 第一个元组
        (
            # 定义分割索引范围
            [(0, 6), (6, 12), (12, 18), (18, None)],
            # 创建包含字符 'a', 'b', 'c', 'd', 'e' 的列表
            list("abcde"),
            # 第三个参数为 None
            None,
            # 第四个参数为 None
            None,
        ),
        # 第二个元组
        (
            # 第一个参数为 None
            None,
            # 创建包含字符 'a', 'b', 'c', 'd', 'e' 的列表
            list("abcde"),
            # 创建包含四个 6 的列表
            [6] * 4,
            # 第四个参数为 None
            None,
        ),
        # 第三个元组
        (
            # 定义分割索引范围
            [(0, 6), (6, 12), (12, 18), (18, None)],
            # 创建包含字符 'a', 'b', 'c', 'd', 'e' 的列表
            list("abcde"),
            # 第三个参数为 None
            None,
            # 第四个参数为 True
            True,
        ),
        # 第四个元组
        (
            # 第一个参数为 None
            None,
            # 创建包含字符 'a', 'b', 'c', 'd', 'e' 的列表
            list("abcde"),
            # 创建包含四个 6 的列表
            [6] * 4,
            # 第四个参数为 False
            False,
        ),
        # 第五个元组
        (
            # 第一个参数为 None
            None,
            # 创建包含字符 'a', 'b', 'c', 'd', 'e' 的列表
            list("abcde"),
            # 创建包含四个 6 的列表
            [6] * 4,
            # 第四个参数为 True
            True,
        ),
        # 第六个元组
        (
            # 定义分割索引范围
            [(0, 6), (6, 12), (12, 18), (18, None)],
            # 创建包含字符 'a', 'b', 'c', 'd', 'e' 的列表
            list("abcde"),
            # 第三个参数为 None
            None,
            # 第四个参数为 False
            False,
        ),
    ],
def test_len_colspecs_len_names(colspecs, names, widths, index_col):
    # GH#40830
    data = """col1  col2  col3  col4
    bab   ba    2"""
    msg = "Length of colspecs must match length of names"
    # 使用 pytest 的 assertRaises 方法来验证是否会抛出 ValueError 异常，且异常信息匹配 msg
    with pytest.raises(ValueError, match=msg):
        read_fwf(
            StringIO(data),
            colspecs=colspecs,
            names=names,
            widths=widths,
            index_col=index_col,
        )


@pytest.mark.parametrize(
    "colspecs, names, widths, index_col, expected",
    [
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("abc"),
            None,
            0,
            DataFrame(
                index=["col1", "ba"],
                columns=["a", "b", "c"],
                data=[["col2", "col3", "col4"], ["b   ba", "2", np.nan]],
            ),
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("ab"),
            None,
            [0, 1],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"]],
                columns=["a", "b"],
                data=[["col3", "col4"], ["2", np.nan]],
            ),
        ),
        (
            [(0, 6), (6, 12), (12, 18), (18, None)],
            list("a"),
            None,
            [0, 1, 2],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"], ["col3", "2"]],
                columns=["a"],
                data=[["col4"], [np.nan]],
            ),
        ),
        (
            None,
            list("abc"),
            [6] * 4,
            0,
            DataFrame(
                index=["col1", "ba"],
                columns=["a", "b", "c"],
                data=[["col2", "col3", "col4"], ["b   ba", "2", np.nan]],
            ),
        ),
        (
            None,
            list("ab"),
            [6] * 4,
            [0, 1],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"]],
                columns=["a", "b"],
                data=[["col3", "col4"], ["2", np.nan]],
            ),
        ),
        (
            None,
            list("a"),
            [6] * 4,
            [0, 1, 2],
            DataFrame(
                index=[["col1", "ba"], ["col2", "b   ba"], ["col3", "2"]],
                columns=["a"],
                data=[["col4"], [np.nan]],
            ),
        ),
    ],
)
def test_len_colspecs_len_names_with_index_col(
    colspecs, names, widths, index_col, expected
):
    # GH#40830
    data = """col1  col2  col3  col4
    bab   ba    2"""
    # 调用 read_fwf 函数，并将结果与期望结果 expected 进行比较
    result = read_fwf(
        StringIO(data),
        colspecs=colspecs,
        names=names,
        widths=widths,
        index_col=index_col,
    )
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_colspecs_with_comment():
    # GH 14135
    # 调用 read_fwf 函数，使用字符串 "#\nA1K\n" 和指定的 colspecs、comment、header 参数
    result = read_fwf(
        StringIO("#\nA1K\n"), colspecs=[(1, 2), (2, 3)], comment="#", header=None
    )
    # 创建期望的 DataFrame，并使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    expected = DataFrame([[1, "K"]], columns=[0, 1])
    tm.assert_frame_equal(result, expected)
def test_skip_rows_and_n_rows():
    # GH#44021
    # 定义包含制表符分隔的数据字符串
    data = """a\tb
1\t a
2\t b
3\t c
4\t d
5\t e
6\t f
    """
    # 调用 read_fwf 函数，设定读取参数，跳过指定行和读取指定行数
    result = read_fwf(StringIO(data), nrows=4, skiprows=[2, 4])
    # 期望的 DataFrame 结果
    expected = DataFrame({"a": [1, 3, 5, 6], "b": ["a", "c", "e", "f"]})
    # 使用测试工具比较结果与期望
    tm.assert_frame_equal(result, expected)


def test_skiprows_with_iterator():
    # GH#10261, GH#56323
    # 定义包含数字的数据字符串
    data = """0
1
2
3
4
5
6
7
8
9
    """
    # 使用 read_fwf 函数，设置列范围、列名、迭代器模式和块大小，跳过指定行
    df_iter = read_fwf(
        StringIO(data),
        colspecs=[(0, 2)],
        names=["a"],
        iterator=True,
        chunksize=2,
        skiprows=[0, 1, 2, 6, 9],
    )
    # 预期的 DataFrame 列表
    expected_frames = [
        DataFrame({"a": [3, 4]}),
        DataFrame({"a": [5, 7]}, index=[2, 3]),
        DataFrame({"a": [8]}, index=[4]),
    ]
    # 遍历迭代器，使用测试工具比较结果与期望
    for i, result in enumerate(df_iter):
        tm.assert_frame_equal(result, expected_frames[i])


def test_names_and_infer_colspecs():
    # GH#45337
    # 定义包含列名的数据字符串，跳过第一行，仅使用指定列索引
    data = """X   Y   Z
      959.0    345   22.2
    """
    # 使用 read_fwf 函数，设定跳过行数和使用的列，指定列名
    result = read_fwf(StringIO(data), skiprows=1, usecols=[0, 2], names=["a", "b"])
    # 期望的 DataFrame 结果
    expected = DataFrame({"a": [959.0], "b": 22.2})
    # 使用测试工具比较结果与期望
    tm.assert_frame_equal(result, expected)


def test_widths_and_usecols():
    # GH#46580
    # 定义包含宽度不规则的数据字符串，无标题行，仅使用指定列索引
    data = """0  1    n -0.4100.1
0  2    p  0.2 90.1
0  3    n -0.3140.4"""
    # 使用 read_fwf 函数，设定读取参数，指定列宽度和使用的列索引
    result = read_fwf(
        StringIO(data),
        header=None,
        usecols=(0, 1, 3),
        widths=(3, 5, 1, 5, 5),
        index_col=False,
        names=("c0", "c1", "c3"),
    )
    # 期望的 DataFrame 结果
    expected = DataFrame(
        {
            "c0": 0,
            "c1": [1, 2, 3],
            "c3": [-0.4, 0.2, -0.3],
        }
    )
    # 使用测试工具比较结果与期望
    tm.assert_frame_equal(result, expected)


def test_dtype_backend(string_storage, dtype_backend):
    # GH#50289
    # 根据 string_storage 和 dtype_backend 设置不同的数据类型处理方式
    if string_storage == "python":
        # 如果使用 Python 字符串存储，创建 StringArray 对象
        arr = StringArray(np.array(["a", "b"], dtype=np.object_))
        arr_na = StringArray(np.array([pd.NA, "a"], dtype=np.object_))
    elif dtype_backend == "pyarrow":
        # 如果使用 PyArrow，导入必要的库，并创建 ArrowExtensionArray 对象
        pa = pytest.importorskip("pyarrow")
        from pandas.arrays import ArrowExtensionArray

        arr = ArrowExtensionArray(pa.array(["a", "b"]))
        arr_na = ArrowExtensionArray(pa.array([None, "a"]))
    else:
        # 否则，使用 ArrowStringArray 对象
        pa = pytest.importorskip("pyarrow")
        arr = ArrowStringArray(pa.array(["a", "b"]))
        arr_na = ArrowStringArray(pa.array([None, "a"]))

    # 定义包含混合数据类型的数据字符串
    data = """a  b    c      d  e     f  g    h  i
1  2.5  True  a
3  4.5  False b  True  6  7.5  a"""
    # 根据 string_storage 设置的数据类型处理方式读取数据
    with pd.option_context("mode.string_storage", string_storage):
        result = read_fwf(StringIO(data), dtype_backend=dtype_backend)
    # 创建一个 DataFrame 对象 `expected`，包含不同数据类型的列
    expected = DataFrame(
        {
            "a": pd.Series([1, 3], dtype="Int64"),     # 整数类型列
            "b": pd.Series([2.5, 4.5], dtype="Float64"),  # 浮点数类型列
            "c": pd.Series([True, False], dtype="boolean"),  # 布尔类型列
            "d": arr,   # 包含已定义数组 `arr` 的列
            "e": pd.Series([pd.NA, True], dtype="boolean"),  # 包含缺失值的布尔类型列
            "f": pd.Series([pd.NA, 6], dtype="Int64"),   # 包含缺失值的整数类型列
            "g": pd.Series([pd.NA, 7.5], dtype="Float64"),   # 包含缺失值的浮点数类型列
            "h": arr_na,   # 包含已定义数组 `arr_na` 的列
            "i": pd.Series([pd.NA, pd.NA], dtype="Int64"),   # 包含缺失值的整数类型列
        }
    )
    
    # 如果使用的后端是 `pyarrow`
    if dtype_backend == "pyarrow":
        # 导入 pytest 的同时，确保 pyarrow 库已经安装，否则跳过测试
        pa = pytest.importorskip("pyarrow")
        from pandas.arrays import ArrowExtensionArray
    
        # 为 `expected` 中的每一列创建 ArrowExtensionArray 对象
        expected = DataFrame(
            {
                col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True))
                for col in expected.columns
            }
        )
        # 将列 'i' 替换为包含两个 None 值的 ArrowExtensionArray 对象
        expected["i"] = ArrowExtensionArray(pa.array([None, None]))
    
    # 使用 `tm` 模块的 assert_frame_equal 函数比较 `result` 和 `expected` 是否相等
    tm.assert_frame_equal(result, expected)
# 测试函数，用于验证当 dtype_backend 参数为 'numpy' 时抛出异常
def test_invalid_dtype_backend():
    # 错误信息，指示 dtype_backend 参数为 'numpy' 时无效，仅支持 'numpy_nullable' 和 'pyarrow'
    msg = (
        "dtype_backend numpy is invalid, only 'numpy_nullable' and "
        "'pyarrow' are allowed."
    )
    # 使用 pytest 检查是否抛出 ValueError 异常，并匹配特定错误信息
    with pytest.raises(ValueError, match=msg):
        # 调用 read_fwf 函数，传入参数 "test" 和 dtype_backend="numpy"
        read_fwf("test", dtype_backend="numpy")


# 使用 pytest 的标记，表示这是一个需要网络连接的测试
# 使用 pytest 的标记，表示这是一个单 CPU 执行的测试
def test_url_urlopen(httpserver):
    # 模拟的数据内容，包含表格数据
    data = """\
A         B            C            D
201158    360.242940   149.910199   11950.7
201159    444.953632   166.985655   11788.4
201160    364.136849   183.628767   11806.2
201161    413.836124   184.375703   11916.8
201162    502.953953   173.237159   12468.3
"""
    # 启动 HTTP 服务器，并提供模拟的数据内容
    httpserver.serve_content(content=data)
    # 预期的列索引，使用 Pandas 的 Index 对象表示列名列表 "ABCD"
    expected = pd.Index(list("ABCD"))
    # 使用 urlopen 打开 httpserver 的 URL 获取文件内容流
    with urlopen(httpserver.url) as f:
        # 调用 read_fwf 函数，传入文件内容流 f，并获取其列名
        result = read_fwf(f).columns

    # 使用 Pandas 的 tm.assert_index_equal 方法断言 result 是否与 expected 相等
    tm.assert_index_equal(result, expected)
```
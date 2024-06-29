# `D:\src\scipysrc\pandas\pandas\tests\io\parser\common\test_index.py`

```
"""
Tests that work on both the Python and C engines but do not have a
specific classification into the other test modules.
"""

# 导入所需模块
from datetime import datetime
from io import StringIO
import os

# 导入 pytest 测试框架
import pytest

# 导入 pandas 库中的特定组件
from pandas import (
    DataFrame,
    Index,
    MultiIndex,
)
# 导入 pandas 测试工具模块
import pandas._testing as tm

# 设置 pytest 标记来忽略特定警告
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

# 标记使用 pyarrow 的用例为预期失败
xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")
# 标记跳过 pyarrow 的用例
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")

# 参数化测试用例：测试不同的输入数据和预期输出
@pytest.mark.parametrize(
    "data,kwargs,expected",
    [
        (
            """foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo2,12,13,14,15
bar2,12,13,14,15
""",
            {"index_col": 0, "names": ["index", "A", "B", "C", "D"]},
            DataFrame(
                [
                    [2, 3, 4, 5],
                    [7, 8, 9, 10],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                ],
                index=Index(["foo", "bar", "baz", "qux", "foo2", "bar2"], name="index"),
                columns=["A", "B", "C", "D"],
            ),
        ),
        (
            """foo,one,2,3,4,5
foo,two,7,8,9,10
foo,three,12,13,14,15
bar,one,12,13,14,15
bar,two,12,13,14,15
""",
            {"index_col": [0, 1], "names": ["index1", "index2", "A", "B", "C", "D"]},
            DataFrame(
                [
                    [2, 3, 4, 5],
                    [7, 8, 9, 10],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                    [12, 13, 14, 15],
                ],
                index=MultiIndex.from_tuples(
                    [
                        ("foo", "one"),
                        ("foo", "two"),
                        ("foo", "three"),
                        ("bar", "one"),
                        ("bar", "two"),
                    ],
                    names=["index1", "index2"],
                ),
                columns=["A", "B", "C", "D"],
            ),
        ),
    ],
)
# 测试函数：测试读取 CSV 文件并比较结果是否符合预期
def test_pass_names_with_index(all_parsers, data, kwargs, expected):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(result, expected)

# 参数化测试用例：测试多级索引但不提供索引名称的情况
@pytest.mark.parametrize("index_col", [[0, 1], [1, 0]])
def test_multi_index_no_level_names(all_parsers, index_col):
    # 定义数据
    data = """index1,index2,A,B,C,D
foo,one,2,3,4,5
foo,two,7,8,9,10
foo,three,12,13,14,15
bar,one,12,13,14,15
bar,two,12,13,14,15
"""
    # 去除数据头部信息
    headless_data = "\n".join(data.split("\n")[1:])

    # 定义列名
    names = ["A", "B", "C", "D"]
    parser = all_parsers

    # 解析 CSV 数据并进行比较
    result = parser.read_csv(
        StringIO(headless_data), index_col=index_col, header=None, names=names
    )
    expected = parser.read_csv(StringIO(data), index_col=index_col)

    # 在无索引名称的数据中，将预期结果的索引名称设为空列表
    expected.index.names = [None] * 2
    tm.assert_frame_equal(result, expected)

# 标记跳过使用 pyarrow 的用例
@skip_pyarrow
def test_multi_index_no_level_names_implicit(all_parsers):
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 定义包含多行数据的CSV格式字符串
    data = """A,B,C,D
foo,one,2,3,4,5
foo,two,7,8,9,10
foo,three,12,13,14,15
bar,one,12,13,14,15
bar,two,12,13,14,15
"""

    # 使用解析器读取CSV数据并返回结果DataFrame
    result = parser.read_csv(StringIO(data))
    
    # 期望的DataFrame，包含预期的数据和结构
    expected = DataFrame(
        [
            [2, 3, 4, 5],
            [7, 8, 9, 10],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
        ],
        columns=["A", "B", "C", "D"],
        index=MultiIndex.from_tuples(
            [
                ("foo", "one"),
                ("foo", "two"),
                ("foo", "three"),
                ("bar", "one"),
                ("bar", "two"),
            ]
        ),
    )
    
    # 使用测试工具比较实际结果和期望结果的DataFrame是否相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 期望此测试在使用PyArrow时失败，因为会抛出TypeError异常
@pytest.mark.parametrize(
    "data,columns,header",
    [
        ("a,b", ["a", "b"], [0]),
        (
            "a,b\nc,d",
            MultiIndex.from_tuples([("a", "c"), ("b", "d")]),
            [0, 1],
        ),
    ],
)
@pytest.mark.parametrize("round_trip", [True, False])
def test_multi_index_blank_df(all_parsers, data, columns, header, round_trip):
    # 见GitHub issue-14545
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 创建一个预期的DataFrame，仅包含指定列
    expected = DataFrame(columns=columns)
    # 如果指定往返（round_trip）标志为True，则将预期DataFrame转换为CSV格式字符串
    data = expected.to_csv(index=False) if round_trip else data

    # 使用解析器读取CSV数据并返回结果DataFrame
    result = parser.read_csv(StringIO(data), header=header)
    
    # 使用测试工具比较实际结果和期望结果的DataFrame是否相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # 期望此测试在使用PyArrow时失败，因为会抛出AssertionError异常
def test_no_unnamed_index(all_parsers):
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 定义包含多行数据的CSV格式字符串
    data = """ id c0 c1 c2
0 1 0 a b
1 2 0 c d
2 2 2 e f
"""
    # 使用解析器读取CSV数据并返回结果DataFrame，指定分隔符为空格
    result = parser.read_csv(StringIO(data), sep=" ")
    
    # 期望的DataFrame，包含预期的数据和结构
    expected = DataFrame(
        [[0, 1, 0, "a", "b"], [1, 2, 0, "c", "d"], [2, 2, 2, "e", "f"]],
        columns=["Unnamed: 0", "id", "c0", "c1", "c2"],
    )
    
    # 使用测试工具比较实际结果和期望结果的DataFrame是否相等
    tm.assert_frame_equal(result, expected)


def test_read_duplicate_index_explicit(all_parsers):
    # 定义包含多行数据的CSV格式字符串
    data = """index,A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo,12,13,14,15
bar,12,13,14,15
"""
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 使用解析器读取CSV数据并返回结果DataFrame，指定第一列为索引列
    result = parser.read_csv(StringIO(data), index_col=0)

    # 期望的DataFrame，包含预期的数据和结构
    expected = DataFrame(
        [
            [2, 3, 4, 5],
            [7, 8, 9, 10],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
            [12, 13, 14, 15],
        ],
        columns=["A", "B", "C", "D"],
        index=Index(["foo", "bar", "baz", "qux", "foo", "bar"], name="index"),
    )
    
    # 使用测试工具比较实际结果和期望结果的DataFrame是否相等
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # 期望此测试在使用PyArrow时跳过执行
def test_read_duplicate_index_implicit(all_parsers):
    # 定义包含多行数据的CSV格式字符串
    data = """A,B,C,D
foo,2,3,4,5
bar,7,8,9,10
baz,12,13,14,15
qux,12,13,14,15
foo,12,13,14,15
bar,12,13,14,15
"""
    # 从参数中获取所有解析器对象
    parser = all_parsers
    # 使用解析器读取CSV数据并返回结果DataFrame

    result = parser.read_csv(StringIO(data))
    # 创建预期的 DataFrame 对象，包含指定的数据、列名和索引
    expected = DataFrame(
        [
            [2, 3, 4, 5],                 # 第一行数据
            [7, 8, 9, 10],                # 第二行数据
            [12, 13, 14, 15],             # 第三行数据
            [12, 13, 14, 15],             # 第四行数据
            [12, 13, 14, 15],             # 第五行数据
            [12, 13, 14, 15],             # 第六行数据
        ],
        columns=["A", "B", "C", "D"],     # 指定列的名称
        index=Index(["foo", "bar", "baz", "qux", "foo", "bar"]),  # 指定索引值
    )
    # 使用测试工具对计算结果和预期结果进行比较，确认它们是否相等
    tm.assert_frame_equal(result, expected)
# 跳过 pyarrow 的测试装饰器，这意味着这些测试不会涉及 pyarrow 库
@skip_pyarrow
def test_read_csv_no_index_name(all_parsers, csv_dir_path):
    # 从参数中获取所有解析器
    parser = all_parsers
    # 组合出待读取的 CSV 文件路径
    csv2 = os.path.join(csv_dir_path, "test2.csv")
    # 使用解析器读取 CSV 文件，设置第一列为索引，解析日期
    result = parser.read_csv(csv2, index_col=0, parse_dates=True)

    # 期望的 DataFrame 结果，包含特定的数据和索引
    expected = DataFrame(
        [
            [0.980269, 3.685731, -0.364216805298, -1.159738, "foo"],
            [1.047916, -0.041232, -0.16181208307, 0.212549, "bar"],
            [0.498581, 0.731168, -0.537677223318, 1.346270, "baz"],
            [1.120202, 1.567621, 0.00364077397681, 0.675253, "qux"],
            [-0.487094, 0.571455, -1.6116394093, 0.103469, "foo2"],
        ],
        columns=["A", "B", "C", "D", "E"],
        index=Index(
            [
                datetime(2000, 1, 3),
                datetime(2000, 1, 4),
                datetime(2000, 1, 5),
                datetime(2000, 1, 6),
                datetime(2000, 1, 7),
            ],
            dtype="M8[s]",
        ),
    )
    # 断言实际结果与期望结果相等
    tm.assert_frame_equal(result, expected)


# 跳过 pyarrow 的测试装饰器，这意味着这些测试不会涉及 pyarrow 库
@skip_pyarrow
def test_empty_with_index(all_parsers):
    # 查看 GitHub 问题 #10184
    # 准备测试数据
    data = "x,y"
    # 从参数中获取所有解析器
    parser = all_parsers
    # 使用解析器读取 CSV 数据，设置第一列为索引
    result = parser.read_csv(StringIO(data), index_col=0)

    # 期望的空 DataFrame 结果，具有特定的列和空索引
    expected = DataFrame(columns=["y"], index=Index([], name="x"))
    # 断言实际结果与期望结果相等
    tm.assert_frame_equal(result, expected)


# 跳过 pyarrow 的测试装饰器，这意味着这些测试不会涉及 pyarrow 库
@skip_pyarrow
def test_empty_with_multi_index(all_parsers):
    # 查看 GitHub 问题 #10467
    # 准备测试数据
    data = "x,y,z"
    # 从参数中获取所有解析器
    parser = all_parsers
    # 使用解析器读取 CSV 数据，设置多级索引
    result = parser.read_csv(StringIO(data), index_col=["x", "y"])

    # 期望的空 DataFrame 结果，具有特定的列和多级空索引
    expected = DataFrame(
        columns=["z"], index=MultiIndex.from_arrays([[]] * 2, names=["x", "y"])
    )
    # 断言实际结果与期望结果相等
    tm.assert_frame_equal(result, expected)


# 跳过 pyarrow 的测试装饰器，这意味着这些测试不会涉及 pyarrow 库
@skip_pyarrow
def test_empty_with_reversed_multi_index(all_parsers):
    # 准备测试数据
    data = "x,y,z"
    # 从参数中获取所有解析器
    parser = all_parsers
    # 使用解析器读取 CSV 数据，设置反转顺序的多级索引
    result = parser.read_csv(StringIO(data), index_col=[1, 0])

    # 期望的空 DataFrame 结果，具有特定的列和反转顺序的多级空索引
    expected = DataFrame(
        columns=["z"], index=MultiIndex.from_arrays([[]] * 2, names=["y", "x"])
    )
    # 断言实际结果与期望结果相等
    tm.assert_frame_equal(result, expected)
```
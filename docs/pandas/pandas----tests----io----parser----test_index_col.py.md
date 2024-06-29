# `D:\src\scipysrc\pandas\pandas\tests\io\parser\test_index_col.py`

```
"""
Tests that the specified index column (a.k.a "index_col")
is properly handled or inferred during parsing for all of
the parsers defined in parsers.py
"""

# 导入所需的模块和库
from io import StringIO  # 导入StringIO类，用于创建内存中的文件对象

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试

from pandas import (  # 从pandas库中导入DataFrame、Index、MultiIndex类
    DataFrame,
    Index,
    MultiIndex,
)
import pandas._testing as tm  # 导入pandas测试模块中的测试工具函数

# 忽略特定警告信息的标记
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager to DataFrame:DeprecationWarning"
)

xfail_pyarrow = pytest.mark.usefixtures("pyarrow_xfail")  # 标记pytest装饰器，用于跳过pyarrow失败的测试
skip_pyarrow = pytest.mark.usefixtures("pyarrow_skip")  # 标记pytest装饰器，用于跳过pyarrow的测试

# 参数化测试函数，测试不同的with_header参数值
@pytest.mark.parametrize("with_header", [True, False])
def test_index_col_named(all_parsers, with_header):
    parser = all_parsers  # 获取参数化的解析器对象
    no_header = """\
KORD1,19990127, 19:00:00, 18:56:00, 0.8100, 2.8100, 7.2000, 0.0000, 280.0000
KORD2,19990127, 20:00:00, 19:56:00, 0.0100, 2.2100, 7.2000, 0.0000, 260.0000
KORD3,19990127, 21:00:00, 20:56:00, -0.5900, 2.2100, 5.7000, 0.0000, 280.0000
KORD4,19990127, 21:00:00, 21:18:00, -0.9900, 2.0100, 3.6000, 0.0000, 270.0000
KORD5,19990127, 22:00:00, 21:56:00, -0.5900, 1.7100, 5.1000, 0.0000, 290.0000
KORD6,19990127, 23:00:00, 22:56:00, -0.5900, 1.7100, 4.6000, 0.0000, 280.0000"""
    header = "ID,date,NominalTime,ActualTime,TDew,TAir,Windspeed,Precip,WindDir\n"

    if with_header:
        data = header + no_header  # 将头部和内容合并为一个字符串

        result = parser.read_csv(StringIO(data), index_col="ID")  # 用解析器读取CSV数据，设定"ID"列为索引列
        expected = parser.read_csv(StringIO(data), header=0).set_index("ID")  # 读取CSV数据并设置"ID"列为索引列的期望结果
        tm.assert_frame_equal(result, expected)  # 使用测试工具函数比较实际结果和期望结果
    else:
        data = no_header  # 仅有内容，没有头部信息
        msg = "Index ID invalid"  # 异常情况的错误信息

        with pytest.raises(ValueError, match=msg):  # 检查是否引发指定异常和错误信息的断言
            parser.read_csv(StringIO(data), index_col="ID")


def test_index_col_named2(all_parsers):
    parser = all_parsers  # 获取解析器对象
    data = """\
1,2,3,4,hello
5,6,7,8,world
9,10,11,12,foo
"""

    expected = DataFrame(  # 创建预期结果的DataFrame
        {"a": [1, 5, 9], "b": [2, 6, 10], "c": [3, 7, 11], "d": [4, 8, 12]},
        index=Index(["hello", "world", "foo"], name="message"),  # 指定索引为"message"的Index对象
    )
    names = ["a", "b", "c", "d", "message"]

    result = parser.read_csv(StringIO(data), names=names, index_col=["message"])  # 使用指定列作为索引列读取CSV数据
    tm.assert_frame_equal(result, expected)  # 使用测试工具函数比较实际结果和期望结果


def test_index_col_is_true(all_parsers):
    # see gh-9798
    data = "a,b\n1,2"  # CSV数据字符串
    parser = all_parsers  # 获取解析器对象

    msg = "The value of index_col couldn't be 'True'"  # 错误信息字符串
    with pytest.raises(ValueError, match=msg):  # 检查是否引发指定异常和错误信息的断言
        parser.read_csv(StringIO(data), index_col=True)


@skip_pyarrow  # 标记pytest装饰器，用于跳过pyarrow的测试；CSV解析错误：预期3列，实际得到4列
def test_infer_index_col(all_parsers):
    data = """A,B,C
foo,1,2,3
bar,4,5,6
baz,7,8,9
"""
    parser = all_parsers  # 获取解析器对象
    result = parser.read_csv(StringIO(data))  # 读取CSV数据并解析

    expected = DataFrame(  # 创建预期结果的DataFrame
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        index=["foo", "bar", "baz"],  # 指定索引
        columns=["A", "B", "C"],  # 指定列名
    )
    tm.assert_frame_equal(result, expected)  # 使用测试工具函数比较实际结果和期望结果
    [
        # 第一个元组，不指定索引和列名，默认使用 ["x", "y", "z"]
        (None, {"columns": ["x", "y", "z"]}),
        # 第二个元组，指定不排序列名 ["x", "y", "z"]
        (False, {"columns": ["x", "y", "z"]}),
        # 第三个元组，列名 ["y", "z"]，索引为空的名为 "x" 的 Index 对象
        (0, {"columns": ["y", "z"], "index": Index([], name="x")}),
        # 第四个元组，列名 ["x", "z"]，索引为空的名为 "y" 的 Index 对象
        (1, {"columns": ["x", "z"], "index": Index([], name="y")}),
        # 第五个元组，列名 ["y", "z"]，索引为空的名为 "x" 的 Index 对象
        ("x", {"columns": ["y", "z"], "index": Index([], name="x")}),
        # 第六个元组，列名 ["x", "z"]，索引为空的名为 "y" 的 Index 对象
        ("y", {"columns": ["x", "z"], "index": Index([], name="y")}),
        # 第七个元组，列名 ["z"]，MultiIndex 对象，索引名为 ["x", "y"]，元素为空
        (
            [0, 1],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["x", "y"]),
            },
        ),
        # 第八个元组，列名 ["z"]，MultiIndex 对象，索引名为 ["x", "y"]，元素为空
        (
            ["x", "y"],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["x", "y"]),
            },
        ),
        # 第九个元组，列名 ["z"]，MultiIndex 对象，索引名为 ["y", "x"]，元素为空
        (
            [1, 0],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["y", "x"]),
            },
        ),
        # 第十个元组，列名 ["z"]，MultiIndex 对象，索引名为 ["y", "x"]，元素为空
        (
            ["y", "x"],
            {
                "columns": ["z"],
                "index": MultiIndex.from_arrays([[]] * 2, names=["y", "x"]),
            },
        ),
    ],
# 在给定所有解析器和索引列参数的情况下，测试索引列为空数据的情况
def test_index_col_empty_data(all_parsers, index_col, kwargs):
    # 定义包含数据的字符串
    data = "x,y,z"
    # 使用所有解析器进行 CSV 解析
    parser = all_parsers
    # 调用解析器的 read_csv 方法，指定索引列
    result = parser.read_csv(StringIO(data), index_col=index_col)

    # 创建预期的 DataFrame 对象，使用传递的关键字参数
    expected = DataFrame(**kwargs)
    # 使用 pytest 的辅助函数检查结果与预期是否相等
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
# 对于索引列设置为 False 的空数据测试
def test_empty_with_index_col_false(all_parsers):
    # 见 gh-10413
    # 定义包含数据的字符串
    data = "x,y"
    # 使用所有解析器进行 CSV 解析
    parser = all_parsers
    # 调用解析器的 read_csv 方法，索引列设置为 False
    result = parser.read_csv(StringIO(data), index_col=False)

    # 创建预期的 DataFrame 对象，指定列名称
    expected = DataFrame(columns=["x", "y"])
    # 使用 pytest 的辅助函数检查结果与预期是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "index_names",
    [
        ["", ""],
        ["foo", ""],
        ["", "bar"],
        ["foo", "bar"],
        ["NotReallyUnnamed", "Unnamed: 0"],
    ],
)
# 测试多级索引命名
def test_multi_index_naming(all_parsers, index_names, request):
    # 使用所有解析器进行 CSV 解析
    parser = all_parsers

    # 如果解析器的引擎为 "pyarrow" 并且空字符串在索引名称列表中
    if parser.engine == "pyarrow" and "" in index_names:
        # 标记为预期失败，原因是一种情况会引发异常，其余情况错误
        mark = pytest.mark.xfail(reason="One case raises, others are wrong")
        request.applymarker(mark)

    # 创建包含数据的字符串，索引名称与列数据分隔符为逗号
    data = ",".join(index_names + ["col\na,c,1\na,d,2\nb,c,3\nb,d,4"])
    # 调用解析器的 read_csv 方法，指定多级索引列
    result = parser.read_csv(StringIO(data), index_col=[0, 1])

    # 创建预期的 DataFrame 对象，指定列数据和索引
    expected = DataFrame(
        {"col": [1, 2, 3, 4]}, index=MultiIndex.from_product([["a", "b"], ["c", "d"]])
    )
    # 将预期 DataFrame 的索引名称设置为索引名称列表中的值，空值替换为 None
    expected.index.names = [name if name else None for name in index_names]
    # 使用 pytest 的辅助函数检查结果与预期是否相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
# 测试非所有索引级别名称都在开头的多级索引命名
def test_multi_index_naming_not_all_at_beginning(all_parsers):
    # 使用所有解析器进行 CSV 解析
    parser = all_parsers
    # 定义包含数据的字符串，包含未命名的列
    data = ",Unnamed: 2,\na,c,1\na,d,2\nb,c,3\nb,d,4"
    # 调用解析器的 read_csv 方法，指定多级索引列
    result = parser.read_csv(StringIO(data), index_col=[0, 2])

    # 创建预期的 DataFrame 对象，指定未命名的列数据和索引
    expected = DataFrame(
        {"Unnamed: 2": ["c", "d", "c", "d"]},
        index=MultiIndex(
            levels=[["a", "b"], [1, 2, 3, 4]], codes=[[0, 0, 1, 1], [0, 1, 2, 3]]
        ),
    )
    # 使用 pytest 的辅助函数检查结果与预期是否相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # ValueError: Found non-unique column index
# 测试没有多级索引级别名称的空数据情况
def test_no_multi_index_level_names_empty(all_parsers):
    # GH 10984
    # 使用所有解析器进行 CSV 解析
    parser = all_parsers
    # 创建具有多级索引的 DataFrame 对象
    midx = MultiIndex.from_tuples([("A", 1, 2), ("A", 1, 2), ("B", 1, 2)])
    expected = DataFrame(
        np.random.default_rng(2).standard_normal((3, 3)),
        index=midx,
        columns=["x", "y", "z"],
    )
    # 使用临时文件确保数据干净，将预期的 DataFrame 对象保存到 CSV 文件中
    with tm.ensure_clean() as path:
        expected.to_csv(path)
        # 调用解析器的 read_csv 方法，指定多级索引列
        result = parser.read_csv(path, index_col=[0, 1, 2])
    # 使用 pytest 的辅助函数检查结果与预期是否相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
# 测试包含索引列和标题的数据情况
def test_header_with_index_col(all_parsers):
    # GH 33476
    # 使用所有解析器进行 CSV 解析
    parser = all_parsers
    # 定义包含数据的字符串，包含列标题和索引
    data = """
I11,A,A
I12,B,B
I2,1,3
"""
    # 创建多级索引对象，包含命名
    midx = MultiIndex.from_tuples([("A", "B"), ("A", "B.1")], names=["I11", "I12"])
    idx = Index(["I2"])
    # 创建预期的 DataFrame 对象，指定列数据和索引
    expected = DataFrame([[1, 3]], index=idx, columns=midx)

    # 调用解析器的 read_csv 方法，指定索引列和标题
    result = parser.read_csv(StringIO(data), index_col=0, header=[0, 1])
    # 使用 pytest 的辅助函数检查结果与预期是否相等
    tm.assert_frame_equal(result, expected)
    # 创建一个名为 col_idx 的索引对象，包含列名为 ["A", "A.1"]
    col_idx = Index(["A", "A.1"])
    
    # 创建一个名为 idx 的索引对象，包含索引名为 "I11"，索引值为 ["I12", "I2"]
    idx = Index(["I12", "I2"], name="I11")
    
    # 创建一个名为 expected 的 DataFrame 对象，包含数据 [["B", "B"], ["1", "3"]]，
    # 索引为 idx，列索引为 col_idx
    expected = DataFrame([["B", "B"], ["1", "3"]], index=idx, columns=col_idx)
    
    # 调用 parser 对象的 read_csv 方法，解析传入的字符串数据 data，
    # 将 "I11" 列作为索引列，第 0 行作为列标题行
    result = parser.read_csv(StringIO(data), index_col="I11", header=0)
    
    # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.slow
# 标记为一个较慢的测试用例，用于测试索引列较大的 CSV 文件读取
def test_index_col_large_csv(all_parsers, monkeypatch):
    # 引用所有的解析器
    parser = all_parsers

    ARR_LEN = 100
    # 创建一个 DataFrame，包含两列：'a' 包含 ARR_LEN + 1 个整数，'b' 包含 ARR_LEN + 1 个正态分布的随机数
    df = DataFrame(
        {
            "a": range(ARR_LEN + 1),
            "b": np.random.default_rng(2).standard_normal(ARR_LEN + 1),
        }
    )

    # 使用临时文件保证测试的干净环境
    with tm.ensure_clean() as path:
        # 将 DataFrame 写入 CSV 文件，不包含索引列
        df.to_csv(path, index=False)
        # 使用 monkeypatch 设置参数，调用解析器读取 CSV 文件，设置第一列为索引列
        with monkeypatch.context() as m:
            m.setattr("pandas.core.algorithms._MINIMUM_COMP_ARR_LEN", ARR_LEN)
            result = parser.read_csv(path, index_col=[0])

    # 断言读取结果与预期的 DataFrame，将 'a' 列设置为索引列后相等
    tm.assert_frame_equal(result, df.set_index("a"))


@xfail_pyarrow  # TypeError: an integer is required
# 标记为预期失败的测试用例，用于测试多级索引列和无数据的情况
def test_index_col_multiindex_columns_no_data(all_parsers):
    # 引用所有的解析器
    parser = all_parsers
    # 使用解析器读取没有数据的多级索引列 CSV 数据，设置第一列和第二行为表头索引
    result = parser.read_csv(
        StringIO("a0,a1,a2\nb0,b1,b2\n"), header=[0, 1], index_col=0
    )
    # 预期的 DataFrame 结果，没有数据，索引为多级索引列
    expected = DataFrame(
        [],
        index=Index([]),
        columns=MultiIndex.from_arrays(
            [["a1", "a2"], ["b1", "b2"]], names=["a0", "b0"]
        ),
    )
    # 断言读取结果与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
# 标记为预期失败的测试用例，用于测试表头为单行索引列的情况
def test_index_col_header_no_data(all_parsers):
    # 引用所有的解析器
    parser = all_parsers
    # 使用解析器读取表头为单行索引列的 CSV 数据，设置第一列为索引列
    result = parser.read_csv(StringIO("a0,a1,a2\n"), header=[0], index_col=0)
    # 预期的 DataFrame 结果，没有数据，只有列名，索引为单行索引列
    expected = DataFrame(
        [],
        columns=["a1", "a2"],
        index=Index([], name="a0"),
    )
    # 断言读取结果与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
# 标记为预期失败的测试用例，用于测试多级索引列和无数据的情况
def test_multiindex_columns_no_data(all_parsers):
    # 引用所有的解析器
    parser = all_parsers
    # 使用解析器读取没有数据的多级索引列 CSV 数据，设置第一行为表头索引列
    result = parser.read_csv(StringIO("a0,a1,a2\nb0,b1,b2\n"), header=[0, 1])
    # 预期的 DataFrame 结果，没有数据，列名为多级索引列
    expected = DataFrame(
        [], columns=MultiIndex.from_arrays([["a0", "a1", "a2"], ["b0", "b1", "b2"]])
    )
    # 断言读取结果与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


@xfail_pyarrow  # TypeError: an integer is required
# 标记为预期失败的测试用例，用于测试多级索引列和有数据的情况
def test_multiindex_columns_index_col_with_data(all_parsers):
    # 引用所有的解析器
    parser = all_parsers
    # 使用解析器读取多级索引列和有数据的 CSV 数据，设置第一列为索引列，第一行和第二行为表头索引
    result = parser.read_csv(
        StringIO("a0,a1,a2\nb0,b1,b2\ndata,data,data"), header=[0, 1], index_col=0
    )
    # 预期的 DataFrame 结果，包含有数据，列名为多级索引列
    expected = DataFrame(
        [["data", "data"]],
        columns=MultiIndex.from_arrays(
            [["a1", "a2"], ["b1", "b2"]], names=["a0", "b0"]
        ),
        index=Index(["data"]),
    )
    # 断言读取结果与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


@skip_pyarrow  # CSV parse error: Empty CSV file or block
# 标记为跳过的测试用例，因为 CSV 解析错误：空的 CSV 文件或块
def test_infer_types_boolean_sum(all_parsers):
    # 引用所有的解析器
    parser = all_parsers
    # 使用解析器读取 CSV 数据，设置列名，将 'a' 列设为索引列，指定 'a' 列数据类型为 UInt8
    result = parser.read_csv(
        StringIO("0,1"),
        names=["a", "b"],
        index_col=["a"],
        dtype={"a": "UInt8"},
    )
    # 预期的 DataFrame 结果，包含数据和索引列，'a' 列为 UInt8 数据类型
    expected = DataFrame(
        data={
            "a": [
                0,
            ],
            "b": [1],
        }
    ).set_index("a")
    # 不检查索引类型，因为 C 解析器将返回 dtype 为 'object' 的索引列，而 Python 解析器将返回 dtype 为 'UInt8' 的索引列
    # 使用 `tm.assert_frame_equal` 函数比较 `result` 和 `expected` 两个数据框是否相等，
    # 并且在比较时不检查索引的数据类型是否一致。
    tm.assert_frame_equal(result, expected, check_index_type=False)
# 使用 pytest 的参数化功能，定义测试函数 test_specify_dtype_for_index_col，参数包括数据类型 dtype 和值 val
@pytest.mark.parametrize("dtype, val", [(object, "01"), ("int64", 1)])
def test_specify_dtype_for_index_col(all_parsers, dtype, val, request):
    # GH#9435
    # 定义测试数据
    data = "a,b\n01,2"
    # 获取测试中的数据解析器
    parser = all_parsers
    # 如果数据类型为 object 并且解析器引擎为 "pyarrow"，标记测试为预期失败，因为无法禁用 pyarrow 引擎的类型推断
    if dtype == object and parser.engine == "pyarrow":
        request.applymarker(
            pytest.mark.xfail(reason="Cannot disable type-inference for pyarrow engine")
        )
    # 使用解析器读取 CSV 数据，指定 'a' 列的数据类型为 dtype
    result = parser.read_csv(StringIO(data), index_col="a", dtype={"a": dtype})
    # 预期的 DataFrame 结果，设置 'a' 列的索引为指定值 val
    expected = DataFrame({"b": [2]}, index=Index([val], name="a"))
    # 断言结果与预期是否相等
    tm.assert_frame_equal(result, expected)


# 使用 @xfail_pyarrow 标记为预期失败的测试函数
@xfail_pyarrow  # TypeError: an integer is required
def test_multiindex_columns_not_leading_index_col(all_parsers):
    # GH#38549
    # 获取测试中的数据解析器
    parser = all_parsers
    # 定义包含多行数据的 CSV 字符串
    data = """a,b,c,d
e,f,g,h
x,y,1,2
"""
    # 使用解析器读取 CSV 数据，指定多级标题和将第二列作为索引列
    result = parser.read_csv(
        StringIO(data),
        header=[0, 1],
        index_col=1,
    )
    # 设置多级索引的列名称
    cols = MultiIndex.from_tuples(
        [("a", "e"), ("c", "g"), ("d", "h")], names=["b", "f"]
    )
    # 预期的 DataFrame 结果，包含指定的数据和多级索引
    expected = DataFrame([["x", 1, 2]], columns=cols, index=["y"])
    # 断言结果与预期是否相等
    tm.assert_frame_equal(result, expected)
```